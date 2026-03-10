from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class LapContext:
    season: int
    meeting_key: int | str
    meeting_name: str
    session_key: int | str
    session_name: str
    driver_number: int | str
    driver_name: str
    team_name: str


def build_lap_dataset(
    laps_df: pd.DataFrame,
    car_df: pd.DataFrame,
    loc_df: pd.DataFrame,
    context: LapContext,
    stints_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if laps_df.empty:
        return pd.DataFrame()

    laps_df = laps_df.copy()
    laps_df = _ensure_lap_timestamps(laps_df)
    car_df = _prepare_timeseries(car_df)
    loc_df = _prepare_timeseries(loc_df)
    stints_df = _prepare_stints(stints_df)

    rows: list[dict[str, Any]] = []
    for _, lap in laps_df.iterrows():
        start = lap.get("lap_start")
        end = lap.get("lap_end")
        if pd.isna(start) or pd.isna(end):
            continue
        car_slice = _slice_by_time(car_df, start, end)
        loc_slice = _slice_by_time(loc_df, start, end)

        row = _lap_base_row(lap, context)
        row.update(_stint_info(stints_df, lap.get("lap_number")))
        row.update(_car_metrics(car_slice))
        row.update(_location_metrics(loc_slice))
        row.update(_quality_flags(car_slice, loc_slice))
        rows.append(row)

    return pd.DataFrame(rows)


def _ensure_lap_timestamps(laps_df: pd.DataFrame) -> pd.DataFrame:
    laps_df = laps_df.copy()
    if "date_start" in laps_df.columns:
        laps_df["lap_start"] = laps_df["date_start"]
    elif "date" in laps_df.columns:
        laps_df["lap_start"] = laps_df["date"]
    else:
        laps_df["lap_start"] = pd.NaT

    duration_col = _first_existing(laps_df, ["lap_duration", "duration"])
    if duration_col:
        laps_df["lap_duration_seconds"] = _parse_duration_seconds(laps_df[duration_col])
        laps_df["lap_end"] = laps_df["lap_start"] + pd.to_timedelta(laps_df["lap_duration_seconds"], unit="s")
    elif "date_end" in laps_df.columns:
        laps_df["lap_end"] = laps_df["date_end"]
    else:
        laps_df = laps_df.sort_values("lap_start")
        laps_df["lap_end"] = laps_df["lap_start"].shift(-1)

    return laps_df


def _prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date")
        df = df.set_index("date", drop=False)
    return df


def _slice_by_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    return df.loc[(df["date"] >= start) & (df["date"] <= end)]


def _lap_base_row(lap: pd.Series, context: LapContext) -> dict[str, Any]:
    return {
        "season": context.season,
        "meeting_key": context.meeting_key,
        "meeting_name": context.meeting_name,
        "session_key": context.session_key,
        "session_name": context.session_name,
        "driver_number": context.driver_number,
        "driver_name": context.driver_name,
        "team_name": context.team_name,
        "lap_number": lap.get("lap_number"),
        "lap_duration": lap.get("lap_duration") or lap.get("duration"),
        "duration_sector_1": lap.get("duration_sector_1"),
        "duration_sector_2": lap.get("duration_sector_2"),
        "duration_sector_3": lap.get("duration_sector_3"),
        "i1_speed": lap.get("i1_speed"),
        "i2_speed": lap.get("i2_speed"),
        "st_speed": lap.get("st_speed"),
        "is_pit_out_lap": lap.get("is_pit_out_lap"),
    }


def _car_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    metrics: dict[str, Any] = {}
    metrics.update(_basic_stats(df, "speed", prefix="speed"))
    metrics.update(_basic_stats(df, "rpm", prefix="rpm"))
    metrics.update(_basic_stats(df, "throttle", prefix="throttle"))

    if "throttle" in df.columns:
        throttle = df["throttle"].dropna()
        if not throttle.empty:
            scale = 1.0 if throttle.max() <= 1.0 else 100.0
            metrics["full_throttle_pct"] = float((throttle >= (0.98 * scale)).mean())

    if "brake" in df.columns:
        brake = df["brake"].fillna(0)
        scale = 1.0 if brake.max() <= 1.0 else 100.0
        metrics["brake_pct"] = float((brake > 0).mean())
        metrics["brake_events"] = int(_count_events(brake > 0))
        metrics["hard_brake_events"] = int(_count_events(brake >= (0.7 * scale)))

    if "drs" in df.columns:
        drs = df["drs"].fillna(0)
        metrics["drs_pct"] = float((drs > 0).mean())

    gear_col = _first_existing(df, ["n_gear", "gear"])
    if gear_col:
        metrics["gear_changes"] = int((df[gear_col].diff() != 0).sum())

    return metrics


def _location_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    metrics: dict[str, Any] = {}
    x_col = _first_existing(df, ["x", "x_position", "x_position_m"])
    y_col = _first_existing(df, ["y", "y_position", "y_position_m"])
    z_col = _first_existing(df, ["z", "z_position", "z_position_m"])

    if x_col and y_col:
        x = df[x_col].astype(float)
        y = df[y_col].astype(float)
        dx = x.diff().fillna(0)
        dy = y.diff().fillna(0)
        if z_col and z_col in df.columns:
            z = df[z_col].astype(float)
            dz = z.diff().fillna(0)
            distances = np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            distances = np.sqrt(dx**2 + dy**2)
        metrics["distance_traveled"] = float(distances.sum())
        metrics["trajectory_length"] = metrics["distance_traveled"]

        headings = np.arctan2(dy, dx)
        metrics["trajectory_variation"] = float(np.nanstd(headings))

    return metrics


def _quality_flags(car_df: pd.DataFrame, loc_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "telemetry_points": int(len(car_df)) if not car_df.empty else 0,
        "trajectory_points": int(len(loc_df)) if not loc_df.empty else 0,
        "has_telemetry": bool(len(car_df) > 0),
        "has_trajectory": bool(len(loc_df) > 0),
    }


def _prepare_stints(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for col in ["lap_start", "lap_end", "stint_number", "tyre_age_at_start"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lap_start" in df.columns:
        df = df.sort_values(["lap_start", "lap_end"], na_position="last")
    return df


def _stint_info(stints_df: pd.DataFrame, lap_number: Any) -> dict[str, Any]:
    if stints_df.empty or lap_number is None or pd.isna(lap_number):
        return {}
    try:
        lap_num = int(lap_number)
    except (TypeError, ValueError):
        return {}

    if "lap_start" not in stints_df.columns or "lap_end" not in stints_df.columns:
        return {}

    mask = (stints_df["lap_start"] <= lap_num) & (stints_df["lap_end"] >= lap_num)
    if not mask.any():
        return {}

    stint = stints_df.loc[mask].iloc[0]
    lap_start = stint.get("lap_start")
    tyre_age_start = stint.get("tyre_age_at_start")

    tyre_age_at_lap = None
    if pd.notna(tyre_age_start) and pd.notna(lap_start):
        try:
            tyre_age_at_lap = float(tyre_age_start) + (lap_num - float(lap_start))
        except (TypeError, ValueError):
            tyre_age_at_lap = None

    return {
        "stint_number": stint.get("stint_number"),
        "compound": stint.get("compound"),
        "stint_lap_start": stint.get("lap_start"),
        "stint_lap_end": stint.get("lap_end"),
        "tyre_age_at_start": stint.get("tyre_age_at_start"),
        "tyre_age_at_lap": tyre_age_at_lap,
    }


def _basic_stats(df: pd.DataFrame, column: str, prefix: str) -> dict[str, Any]:
    if column not in df.columns:
        return {}
    series = df[column].dropna()
    if series.empty:
        return {}
    return {
        f"avg_{prefix}": float(series.mean()),
        f"max_{prefix}": float(series.max()),
        f"min_{prefix}": float(series.min()),
        f"{prefix}_std": float(series.std(ddof=0)),
    }


def _count_events(mask: pd.Series) -> int:
    transitions = mask.astype(int).diff().fillna(0)
    return int((transitions > 0).sum())


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _parse_duration_seconds(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "f"}:
        return series
    parsed = pd.to_timedelta(series, errors="coerce")
    return parsed.dt.total_seconds()
