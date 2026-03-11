from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clients.openf1_client import OpenF1Client, RateLimiter
from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.utils import SECTOR_COLUMNS, build_preprocessor


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_profiles.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _init_mlflow(tracking_uri: str | None, experiment_name: str) -> bool:
    if not tracking_uri:
        return False
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return True


def _mode(series: pd.Series) -> str | float:
    if series.empty:
        return np.nan
    modes = series.mode()
    if not modes.empty:
        return modes.iloc[0]
    return series.iloc[0]


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _circuit_speed_class(df: pd.DataFrame) -> pd.DataFrame:
    speed_col = _first_available(df, ["avg_speed", "st_speed", "i2_speed", "i1_speed"])
    if not speed_col:
        return pd.DataFrame(columns=["meeting_key", "circuit_speed_class"])

    meeting_speed = (
        df.groupby("meeting_key")[speed_col]
        .mean()
        .reset_index(name="meeting_avg_speed")
    )
    if meeting_speed.empty:
        return pd.DataFrame(columns=["meeting_key", "circuit_speed_class"])

    q1 = meeting_speed["meeting_avg_speed"].quantile(1 / 3)
    q2 = meeting_speed["meeting_avg_speed"].quantile(2 / 3)

    def _label(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= q1:
            return "low"
        if value <= q2:
            return "medium"
        return "high"

    meeting_speed["circuit_speed_class"] = meeting_speed["meeting_avg_speed"].apply(_label)
    return meeting_speed[["meeting_key", "circuit_speed_class"]]


def _lap_quality_labels(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df.copy()
    working = working[working["lap_duration"].notna()]
    group_sizes = working.groupby(group_cols)["lap_duration"].transform("count")
    working = working[group_sizes >= 4]

    q25 = working.groupby(group_cols)["lap_duration"].transform(
        lambda s: s.quantile(0.25)
    )
    q75 = working.groupby(group_cols)["lap_duration"].transform(
        lambda s: s.quantile(0.75)
    )

    working["lap_quality"] = np.where(
        working["lap_duration"] <= q25,
        1,
        np.where(working["lap_duration"] >= q75, 0, np.nan),
    )
    working = working[working["lap_quality"].notna()]
    working["lap_quality"] = working["lap_quality"].astype(int)
    return working


def _compute_anomaly_flags(
    df: pd.DataFrame, contamination: float, random_state: int
) -> pd.Series:
    n_jobs = int(os.getenv("ANOMALY_N_JOBS", "1"))
    features = df.copy()
    preprocessor, _, _ = build_preprocessor(features)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(features)
    preds = pipeline.predict(features)
    return pd.Series(preds == -1, index=df.index)


def _compute_degradation(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df.copy()
    working = working[working["lap_duration"].notna()]
    working = working.sort_values(group_cols + ["lap_number"])
    working["stint_first_lap_duration"] = working.groupby(group_cols)[
        "lap_duration"
    ].transform("first")
    working["degradation_delta"] = (
        working["lap_duration"] - working["stint_first_lap_duration"]
    )
    return working


def _stint_delta_pace(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df[df["lap_duration"].notna()].copy()
    stint_df = (
        working.groupby(group_cols)["lap_duration"].mean().reset_index()
    )
    stint_df = stint_df.sort_values(group_cols)
    stint_df["prev_mean_lap"] = stint_df.groupby(
        ["meeting_key", "session_key", "driver_number"]
    )["lap_duration"].shift(1)
    stint_df["delta_pace"] = stint_df["lap_duration"] - stint_df["prev_mean_lap"]
    return stint_df[stint_df["delta_pace"].notna()]


def _relative_position(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number"]
    working = df[df["lap_duration"].notna()].copy()
    driver_df = working.groupby(group_cols)["lap_duration"].mean().reset_index()
    driver_df["rank"] = driver_df.groupby("meeting_key")["lap_duration"].rank(
        method="average"
    )
    driver_df["drivers_in_meeting"] = driver_df.groupby("meeting_key")[
        "driver_number"
    ].transform("count")
    driver_df["rank_percentile"] = (
        driver_df["rank"] - 1
    ) / driver_df["drivers_in_meeting"].replace(1, np.nan)
    driver_df["rank_percentile"] = driver_df["rank_percentile"].fillna(0.0)
    return driver_df


def _driver_style_clusters(df: pd.DataFrame, clusters: int, random_state: int) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    exclude_cols = {
        "lap_duration",
        "lap_number",
        "meeting_key",
        "session_key",
        "driver_number",
        "season",
        "stint_number",
        "stint_lap_start",
        "stint_lap_end",
        "tyre_age_at_start",
        "tyre_age_at_lap",
    }
    exclude_cols.update([col for col in SECTOR_COLUMNS if col in df.columns])
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    if not feature_cols:
        return pd.DataFrame(columns=["driver_number", "driver_style_cluster"])

    style_df = (
        df.groupby(["driver_number"])[feature_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    feature_matrix = style_df[feature_cols].fillna(style_df[feature_cols].median())

    rows = len(style_df)
    if rows < 2 or clusters <= 1:
        style_df["driver_style_cluster"] = 0
        return style_df[["driver_number", "driver_style_cluster"]]

    effective_clusters = min(int(clusters), rows)
    if effective_clusters < 2:
        style_df["driver_style_cluster"] = 0
        return style_df[["driver_number", "driver_style_cluster"]]

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(
                    n_clusters=effective_clusters,
                    n_init=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    labels = pipeline.fit_predict(feature_matrix)
    style_df["driver_style_cluster"] = labels
    return style_df[["driver_number", "driver_style_cluster"]]


def _circuit_clusters(df: pd.DataFrame, clusters: int, random_state: int) -> pd.DataFrame:
    group_cols = ["meeting_key"]
    working = df[df["lap_duration"].notna()].copy()
    agg_map: dict[str, str] = {
        "lap_duration": "mean",
        "avg_speed": "mean",
        "max_speed": "mean",
        "min_speed": "mean",
        "speed_std": "mean",
        "avg_rpm": "mean",
        "max_rpm": "mean",
        "min_rpm": "mean",
        "rpm_std": "mean",
        "avg_throttle": "mean",
        "max_throttle": "mean",
        "min_throttle": "mean",
        "throttle_std": "mean",
        "full_throttle_pct": "mean",
        "brake_pct": "mean",
        "brake_events": "mean",
        "hard_brake_events": "mean",
        "drs_pct": "mean",
        "gear_changes": "mean",
        "distance_traveled": "mean",
        "trajectory_length": "mean",
        "trajectory_variation": "mean",
    }
    available = {k: v for k, v in agg_map.items() if k in working.columns}
    meeting_df = working.groupby(group_cols).agg(available).reset_index()

    numeric_cols = meeting_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "meeting_key"]
    if not numeric_cols:
        return pd.DataFrame(columns=["meeting_key", "circuit_cluster"])

    feature_matrix = meeting_df[numeric_cols].fillna(meeting_df[numeric_cols].median())

    rows = len(meeting_df)
    if rows < 2 or clusters <= 1:
        meeting_df["circuit_cluster"] = 0
        return meeting_df[["meeting_key", "circuit_cluster"]]

    effective_clusters = min(int(clusters), rows)
    if effective_clusters < 2:
        meeting_df["circuit_cluster"] = 0
        return meeting_df[["meeting_key", "circuit_cluster"]]

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(
                    n_clusters=effective_clusters,
                    n_init=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    labels = pipeline.fit_predict(feature_matrix)
    meeting_df["circuit_cluster"] = labels
    return meeting_df[["meeting_key", "circuit_cluster"]]


def _session_kind(session_name: str) -> str | None:
    name = (session_name or "").lower()
    if name == "race":
        return "race"
    if "sprint" in name:
        return "sprint"
    return None


def _fetch_results_points(
    settings: Settings,
    seasons: list[int],
    meeting_key: int | str | None = None,
    session_name_filter: str | None = None,
) -> pd.DataFrame:
    race_points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    sprint_points = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

    rate_limiter = RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
    client = OpenF1Client(settings, rate_limiter=rate_limiter)
    rows: list[dict[str, object]] = []

    for season in seasons:
        sessions = client.get("sessions", params={"year": season})
        for session in sessions:
            session_name = session.get("session_name", "")
            kind = _session_kind(session_name)
            if not kind:
                continue
            if meeting_key is not None and str(session.get("meeting_key")) != str(meeting_key):
                continue
            if session_name_filter and session_name_filter.lower() != "all":
                if str(session_name).lower() != session_name_filter.lower():
                    continue
            session_key = session.get("session_key")
            meeting_key = session.get("meeting_key")
            results = client.get("session_result", params={"session_key": session_key})
            if not results:
                continue

            max_laps = max((r.get("number_of_laps") or 0) for r in results) or None
            for result in results:
                position = result.get("position")
                dnf = bool(result.get("dnf"))
                dns = bool(result.get("dns"))
                dsq = bool(result.get("dsq"))
                number_of_laps = result.get("number_of_laps") or 0

                points = 0
                if isinstance(position, (int, float)) and not (dnf or dns or dsq):
                    if kind == "race":
                        points = race_points.get(int(position), 0)
                    else:
                        points = sprint_points.get(int(position), 0)

                lap_completion_rate = None
                finish_flag = None
                if max_laps:
                    lap_completion_rate = number_of_laps / max_laps
                    finish_flag = number_of_laps == max_laps

                rows.append(
                    {
                        "season": season,
                        "meeting_key": meeting_key,
                        "session_key": session_key,
                        "session_name": session_name,
                        "session_kind": kind,
                        "driver_number": result.get("driver_number"),
                        "position": position,
                        "points": points,
                        "dnf": dnf,
                        "dns": dns,
                        "dsq": dsq,
                        "number_of_laps": number_of_laps,
                        "lap_completion_rate": lap_completion_rate,
                        "finish_flag": finish_flag,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera perfil consolidado por piloto."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--driver-clusters", type=int, default=4)
    parser.add_argument("--circuit-clusters", type=int, default=3)
    parser.add_argument("--anomaly-contamination", type=float, default=0.02)
    parser.add_argument(
        "--finish-threshold",
        type=float,
        default=1.0,
        help="Frac. de voltas completadas para considerar corrida terminada.",
    )
    parser.add_argument("--season", type=int, default=None, help="Filtra por temporada.")
    parser.add_argument("--meeting-key", default=None, help="Filtra por meeting_key.")
    parser.add_argument(
        "--session-name",
        default=None,
        help="Filtra por session_name (ex: Race, Sprint, ou all).",
    )
    parser.add_argument(
        "--drivers-include",
        default=None,
        help="Lista de pilotos a incluir (separados por virgula).",
    )
    parser.add_argument(
        "--drivers-exclude",
        default=None,
        help="Lista de pilotos a excluir (separados por virgula).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    df = load_consolidated()
    df = df[df["lap_duration"].notna()].copy()

    if args.season is not None and "season" in df.columns:
        df = df[df["season"] == args.season]
    if args.meeting_key is not None:
        df = df[df["meeting_key"].astype(str) == str(args.meeting_key)]
    session_name_filter = args.session_name or settings.session_name
    if session_name_filter and session_name_filter.lower() != "all":
        df = df[df["session_name"].astype(str).str.lower() == session_name_filter.lower()]

    include = {name.lower() for name in _split_list(args.drivers_include)}
    exclude = {name.lower() for name in _split_list(args.drivers_exclude)}
    if include or exclude:
        names = df["driver_name"].astype(str).str.lower()
        if include:
            df = df[names.isin(include)]
        if exclude:
            df = df[~names.isin(exclude)]

    base = (
        df.groupby(["driver_number", "driver_name"], as_index=False)
        .agg(
            laps_total=("lap_duration", "count"),
            meetings_total=("meeting_key", "nunique"),
            lap_mean=("lap_duration", "mean"),
            lap_std=("lap_duration", "std"),
        )
    )
    team_mode = (
        df.groupby(["driver_number", "driver_name"])["team_name"]
        .apply(_mode)
        .reset_index()
        .rename(columns={"team_name": "team_name"})
    )
    base = base.merge(team_mode, on=["driver_number", "driver_name"], how="left")
    if "meeting_date_start" in df.columns:
        meeting_dates = (
            df.groupby(["driver_number", "driver_name"])["meeting_date_start"]
            .apply(_mode)
            .reset_index()
            .rename(columns={"meeting_date_start": "meeting_date_start"})
        )
        base = base.merge(meeting_dates, on=["driver_number", "driver_name"], how="left")
        parsed = pd.to_datetime(df["meeting_date_start"], errors="coerce", utc=True)
        df = df.assign(meeting_day=parsed.dt.day, meeting_month=parsed.dt.month)
        meeting_days = (
            df.groupby(["driver_number", "driver_name"])["meeting_day"]
            .apply(_mode)
            .reset_index()
        )
        meeting_months = (
            df.groupby(["driver_number", "driver_name"])["meeting_month"]
            .apply(_mode)
            .reset_index()
        )
        base = base.merge(meeting_days, on=["driver_number", "driver_name"], how="left")
        base = base.merge(meeting_months, on=["driver_number", "driver_name"], how="left")

    # Finish rate and lap completion (accounts for lapped finishers)
    driver_laps = (
        df.groupby(["meeting_key", "session_key", "driver_number", "driver_name"])["lap_number"]
        .max()
        .reset_index(name="driver_max_lap")
    )
    meeting_laps = (
        df.groupby(["meeting_key", "session_key"])["lap_number"]
        .max()
        .reset_index(name="meeting_total_laps")
    )
    completion = driver_laps.merge(meeting_laps, on=["meeting_key", "session_key"], how="left")
    completion["lap_completion_rate"] = (
        completion["driver_max_lap"] / completion["meeting_total_laps"]
    )
    completion["finish_flag"] = completion["lap_completion_rate"] >= args.finish_threshold
    completion_stats = (
        completion.groupby(["driver_number", "driver_name"])
        .agg(
            finish_rate=("finish_flag", "mean"),
            lap_completion_mean=("lap_completion_rate", "mean"),
        )
        .reset_index()
    )
    completion_stats = completion_stats.rename(
        columns={
            "finish_rate": "finish_rate_laps",
            "lap_completion_mean": "lap_completion_mean_laps",
        }
    )
    completion_stats["dnf_rate_laps"] = 1.0 - completion_stats["finish_rate_laps"]
    base = base.merge(completion_stats, on=["driver_number", "driver_name"], how="left")
    base["finish_rate"] = base["finish_rate_laps"]
    base["lap_completion_mean"] = base["lap_completion_mean_laps"]
    base["dnf_rate"] = 1.0 - base["finish_rate"]

    # Results / points from OpenF1 (race + sprint)
    seasons = sorted({int(s) for s in df["season"].dropna().unique()})
    if seasons:
        results_df = _fetch_results_points(
            settings,
            seasons,
            meeting_key=args.meeting_key,
            session_name_filter=session_name_filter,
        )
    else:
        results_df = pd.DataFrame()

    if not results_df.empty:
        results_df["points_race"] = np.where(
            results_df["session_kind"] == "race", results_df["points"], 0
        )
        results_df["points_sprint"] = np.where(
            results_df["session_kind"] == "sprint", results_df["points"], 0
        )
        results_df["session_count"] = 1
        results_stats = (
            results_df.groupby(["driver_number"])
            .agg(
                points_total=("points", "sum"),
                points_race=("points_race", "sum"),
                points_sprint=("points_sprint", "sum"),
                races_count=("session_kind", lambda s: (s == "race").sum()),
                sprints_count=("session_kind", lambda s: (s == "sprint").sum()),
                results_count=("session_count", "sum"),
                finish_rate_results=("finish_flag", "mean"),
                lap_completion_mean_results=("lap_completion_rate", "mean"),
            )
            .reset_index()
        )

        base = base.merge(results_stats, on="driver_number", how="left")
        # Prefer results-based reliability (includes sprints and penalizes lapped finishers)
        base["finish_rate"] = base["finish_rate_results"].fillna(base["finish_rate"])
        base["lap_completion_mean"] = base["lap_completion_mean_results"].fillna(
            base["lap_completion_mean"]
        )
        base["dnf_rate"] = 1.0 - base["finish_rate"]

    # Relative pace vs meeting mean (track-normalized reference)
    driver_meeting = (
        df.groupby(["meeting_key", "driver_number", "driver_name"])["lap_duration"]
        .mean()
        .reset_index(name="driver_meeting_lap_mean")
    )
    meeting_stats = (
        driver_meeting.groupby("meeting_key")["driver_meeting_lap_mean"]
        .agg(meeting_lap_mean="mean", meeting_lap_std="std")
        .reset_index()
    )
    meeting_stats["meeting_lap_std"] = meeting_stats["meeting_lap_std"].fillna(0.0)
    rel = driver_meeting.merge(meeting_stats, on="meeting_key", how="left")
    rel["lap_mean_delta_to_meeting_mean"] = (
        rel["driver_meeting_lap_mean"] - rel["meeting_lap_mean"]
    )
    rel["lap_mean_z_to_meeting_mean"] = rel["lap_mean_delta_to_meeting_mean"] / rel[
        "meeting_lap_std"
    ].replace(0, np.nan)
    rel["lap_mean_z_to_meeting_mean"] = rel["lap_mean_z_to_meeting_mean"].fillna(0.0)

    rel_stats = (
        rel.groupby(["driver_number", "driver_name"])
        .agg(
            meeting_lap_mean_avg=("meeting_lap_mean", "mean"),
            lap_mean_delta_to_meeting_mean=("lap_mean_delta_to_meeting_mean", "mean"),
            lap_mean_z_to_meeting_mean=("lap_mean_z_to_meeting_mean", "mean"),
        )
        .reset_index()
    )
    base = base.merge(rel_stats, on=["driver_number", "driver_name"], how="left")

    if "is_pit_out_lap" in df.columns:
        pit_rate = (
            df.groupby(["driver_number", "driver_name"])["is_pit_out_lap"]
            .mean()
            .reset_index()
            .rename(columns={"is_pit_out_lap": "pit_out_rate"})
        )
        base = base.merge(pit_rate, on=["driver_number", "driver_name"], how="left")

    # Lap quality
    quality_df = _lap_quality_labels(df)
    quality_rates = (
        quality_df.groupby(["driver_number", "driver_name"])["lap_quality"]
        .agg(lap_quality_good_rate="mean")
        .reset_index()
    )
    quality_counts = (
        quality_df.groupby(["driver_number", "driver_name"])["lap_quality"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    quality_counts = quality_counts.rename(
        columns={0: "lap_quality_bad_rate", 1: "lap_quality_good_rate"}
    )
    base = base.merge(quality_counts, on=["driver_number", "driver_name"], how="left")

    # Anomaly rate
    anomaly_flags = _compute_anomaly_flags(
        df, args.anomaly_contamination, args.random_state
    )
    anomaly_rate = (
        df.assign(is_anomaly=anomaly_flags)
        .groupby(["driver_number", "driver_name"])["is_anomaly"]
        .mean()
        .reset_index()
        .rename(columns={"is_anomaly": "anomaly_rate"})
    )
    base = base.merge(anomaly_rate, on=["driver_number", "driver_name"], how="left")

    # Tyre degradation
    degradation_df = _compute_degradation(df)
    degradation_stats = (
        degradation_df.groupby(["driver_number", "driver_name"])["degradation_delta"]
        .agg(degradation_mean="mean", degradation_p95=lambda s: s.quantile(0.95))
        .reset_index()
    )
    base = base.merge(degradation_stats, on=["driver_number", "driver_name"], how="left")

    # Tyre degradation slope vs tyre age
    slopes: list[dict[str, float]] = []
    for (driver_number, driver_name), group in degradation_df.groupby(
        ["driver_number", "driver_name"]
    ):
        sub = group.dropna(subset=["tyre_age_at_lap", "lap_duration"])
        if len(sub) >= 2:
            slope = float(np.polyfit(sub["tyre_age_at_lap"], sub["lap_duration"], 1)[0])
        else:
            slope = np.nan
        slopes.append(
            {
                "driver_number": driver_number,
                "driver_name": driver_name,
                "degradation_slope": slope,
            }
        )
    base = base.merge(pd.DataFrame(slopes), on=["driver_number", "driver_name"], how="left")

    # Stint delta pace
    stint_delta_df = _stint_delta_pace(df)
    stint_delta_stats = (
        stint_delta_df.groupby(["driver_number"])["delta_pace"]
        .agg(
            delta_pace_mean="mean",
            delta_pace_median="median",
            delta_pace_std="std",
            delta_pace_count="count",
        )
        .reset_index()
    )
    base = base.merge(stint_delta_stats, on="driver_number", how="left")

    # Relative position
    relative_df = _relative_position(df)
    relative_stats = (
        relative_df.groupby(["driver_number"])["rank_percentile"]
        .agg(rank_percentile_mean="mean", rank_percentile_median="median")
        .reset_index()
    )
    base = base.merge(relative_stats, on="driver_number", how="left")

    # Driver style clusters
    style_clusters = _driver_style_clusters(
        df, args.driver_clusters, args.random_state
    )
    base = base.merge(style_clusters, on="driver_number", how="left")

    # Circuit clusters and driver distribution
    circuit_clusters = _circuit_clusters(
        df, args.circuit_clusters, args.random_state
    )
    if not circuit_clusters.empty:
        driver_meetings = (
            df[["driver_number", "meeting_key"]].drop_duplicates().merge(
                circuit_clusters, on="meeting_key", how="left"
            )
        )
        cluster_counts = (
            driver_meetings.groupby(["driver_number", "circuit_cluster"])
            .size()
            .reset_index(name="cluster_count")
        )
        cluster_totals = (
            cluster_counts.groupby("driver_number")["cluster_count"]
            .sum()
            .reset_index(name="cluster_total")
        )
        cluster_counts = cluster_counts.merge(cluster_totals, on="driver_number")
        cluster_counts["cluster_pct"] = (
            cluster_counts["cluster_count"] / cluster_counts["cluster_total"]
        )

        dominant = (
            cluster_counts.sort_values(["driver_number", "cluster_pct"], ascending=[True, False])
            .groupby("driver_number")
            .head(1)
            .rename(
                columns={
                    "circuit_cluster": "dominant_circuit_cluster",
                    "cluster_pct": "dominant_circuit_cluster_pct",
                }
            )
        )
        base = base.merge(
            dominant[["driver_number", "dominant_circuit_cluster", "dominant_circuit_cluster_pct"]],
            on="driver_number",
            how="left",
        )

    # Circuit speed class (low/medium/high)
    speed_classes = _circuit_speed_class(df)
    if not speed_classes.empty:
        driver_meetings = (
            df[["driver_number", "meeting_key"]].drop_duplicates().merge(
                speed_classes, on="meeting_key", how="left"
            )
        )
        class_counts = (
            driver_meetings.groupby(["driver_number", "circuit_speed_class"])
            .size()
            .reset_index(name="class_count")
        )
        class_totals = (
            class_counts.groupby("driver_number")["class_count"]
            .sum()
            .reset_index(name="class_total")
        )
        class_counts = class_counts.merge(class_totals, on="driver_number")
        class_counts["class_pct"] = class_counts["class_count"] / class_counts["class_total"]

        dominant_speed = (
            class_counts.sort_values(["driver_number", "class_pct"], ascending=[True, False])
            .groupby("driver_number")
            .head(1)
            .rename(
                columns={
                    "circuit_speed_class": "dominant_circuit_speed_class",
                    "class_pct": "dominant_circuit_speed_class_pct",
                }
            )
        )
        base = base.merge(
            dominant_speed[
                ["driver_number", "dominant_circuit_speed_class", "dominant_circuit_speed_class_pct"]
            ],
            on="driver_number",
            how="left",
        )

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "driver_profiles"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "driver_profiles.csv"
    base.to_csv(csv_path, index=False)

    summary = {
        "drivers": int(base.shape[0]),
        "generated_at": run_timestamp,
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    schema = {
        "columns": [
            {"name": col, "dtype": str(base[col].dtype)} for col in base.columns
        ],
        "rows": int(base.shape[0]),
    }
    schema_path = artifacts_dir / "driver_profiles_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="driver_profiles_report") as run:
            mlflow.set_tags({"task": "driver_profiles_report"})
            mlflow.log_params(
                {
                    "driver_clusters": args.driver_clusters,
                    "circuit_clusters": args.circuit_clusters,
                    "anomaly_contamination": args.anomaly_contamination,
                    "random_state": args.random_state,
                }
            )
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(artifacts_dir / "summary.json"))
            mlflow.log_artifact(str(schema_path))
            logging.getLogger(__name__).info(
                "Relatorio registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Relatorio gerado: %s", csv_path)


if __name__ == "__main__":
    main()
