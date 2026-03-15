from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

DEFAULT_SEASONS = [2023, 2024, 2025, 2026]


def _format_mmss(seconds: float | int | None) -> Optional[str]:
    if seconds is None or pd.isna(seconds):
        return None
    total_ms = int(round(float(seconds) * 1000))
    if total_ms < 0:
        total_ms = 0
    minutes = total_ms // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{secs:02d}:{ms:03d}"


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _value_counts(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    counts = cleaned.value_counts().head(limit)
    return [{"value": key, "count": int(value)} for key, value in counts.items()]


def _value_counts_with_share(series: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    counts = cleaned.value_counts().head(limit)
    total = int(counts.sum())
    rows = []
    for key, value in counts.items():
        count = int(value)
        share = float(count / total) if total else 0.0
        rows.append({"value": key, "count": count, "share": share})
    return rows


def _lap_duration_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_timedelta(series, errors="coerce")
    parsed_seconds = parsed.dt.total_seconds()
    return numeric.where(numeric.notna(), parsed_seconds)


def _row_context(row: pd.Series, lap_seconds: float | None = None) -> dict[str, Any]:
    payload = {
        "meeting_name": _jsonable_value(row.get("meeting_name")),
        "meeting_key": _jsonable_value(row.get("meeting_key")),
        "meeting_date_start": _jsonable_value(row.get("meeting_date_start")),
        "session_name": _jsonable_value(row.get("session_name")),
        "session_key": _jsonable_value(row.get("session_key")),
        "driver_name": _jsonable_value(row.get("driver_name")),
        "driver_number": _jsonable_value(row.get("driver_number")),
        "team_name": _jsonable_value(row.get("team_name")),
        "lap_number": _jsonable_value(row.get("lap_number")),
        "lap_duration": _jsonable_value(row.get("lap_duration")),
    }
    if lap_seconds is not None:
        payload["lap_duration_seconds"] = _jsonable_value(float(lap_seconds))
        payload["lap_duration_min"] = _format_mmss(lap_seconds)
    return payload


def _best_row_for_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    mode: Literal["min", "max"],
) -> Optional[dict[str, Any]]:
    if metric not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    idx = values[valid].idxmin() if mode == "min" else values[valid].idxmax()
    row = df.loc[idx]
    payload = _row_context(row)
    payload["metric"] = metric
    payload["metric_value"] = _jsonable_value(float(values.loc[idx]))
    return payload


def _date_range(series: pd.Series) -> Optional[dict[str, str]]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return None
    return {
        "min": parsed.min().isoformat(),
        "max": parsed.max().isoformat(),
    }


def _build_season_summary(df: pd.DataFrame, season: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "season": season,
    }
    if "season" in df.columns:
        df = df[df["season"] == season]
    if df.empty:
        summary.update({"rows": 0, "message": "Sem dados para a temporada."})
        return summary

    summary["rows"] = int(len(df))
    summary["columns_available"] = sorted([str(c) for c in df.columns])

    if "driver_number" in df.columns:
        summary["drivers_total"] = int(df["driver_number"].nunique())
    elif "driver_name" in df.columns:
        summary["drivers_total"] = int(df["driver_name"].nunique())
    if "team_name" in df.columns:
        summary["teams_total"] = int(df["team_name"].nunique())
    if "meeting_key" in df.columns:
        summary["meetings_total"] = int(df["meeting_key"].nunique())
    if "session_name" in df.columns:
        summary["sessions"] = sorted(df["session_name"].dropna().astype(str).str.strip().unique())
    if "meeting_date_start" in df.columns:
        summary["meeting_date_start"] = _date_range(df["meeting_date_start"])

    lap_seconds: pd.Series | None = None
    if "lap_duration" in df.columns:
        lap_seconds = _lap_duration_seconds(df["lap_duration"])
        if lap_seconds.notna().any():
            summary["lap_duration_quantiles"] = {
                "p01": float(lap_seconds.quantile(0.01)),
                "p05": float(lap_seconds.quantile(0.05)),
                "p10": float(lap_seconds.quantile(0.10)),
                "p25": float(lap_seconds.quantile(0.25)),
                "p50": float(lap_seconds.quantile(0.50)),
                "p75": float(lap_seconds.quantile(0.75)),
                "p90": float(lap_seconds.quantile(0.90)),
                "p95": float(lap_seconds.quantile(0.95)),
                "p99": float(lap_seconds.quantile(0.99)),
            }

    if lap_seconds is not None:
        valid = lap_seconds.notna()
        if valid.any():
            fastest_idx = lap_seconds[valid].idxmin()
            fastest_row = df.loc[fastest_idx]
            summary["fastest_lap"] = _row_context(
                fastest_row, lap_seconds=float(lap_seconds.loc[fastest_idx])
            )
            slowest_idx = lap_seconds[valid].idxmax()
            slowest_row = df.loc[slowest_idx]
            summary["slowest_lap"] = _row_context(
                slowest_row, lap_seconds=float(lap_seconds.loc[slowest_idx])
            )

            ranked = (
                df.assign(_lap_seconds=lap_seconds)
                .loc[valid]
                .sort_values("_lap_seconds", ascending=True)
                .head(10)
            )
            summary["fastest_laps_top"] = [
                _row_context(row, lap_seconds=row.get("_lap_seconds"))
                for _, row in ranked.iterrows()
            ]
            slowest_ranked = (
                df.assign(_lap_seconds=lap_seconds)
                .loc[valid]
                .sort_values("_lap_seconds", ascending=False)
                .head(10)
            )
            summary["slowest_laps_top"] = [
                _row_context(row, lap_seconds=row.get("_lap_seconds"))
                for _, row in slowest_ranked.iterrows()
            ]

            if "meeting_key" in df.columns:
                group_cols = ["meeting_key"]
                if "meeting_name" in df.columns:
                    group_cols.append("meeting_name")
                per_meeting = (
                    df.assign(_lap_seconds=lap_seconds)
                    .loc[valid]
                    .groupby(group_cols, dropna=False)["_lap_seconds"]
                    .idxmin()
                )
                meeting_rows = df.loc[per_meeting].assign(
                    _lap_seconds=lap_seconds.loc[per_meeting]
                )
                summary["fastest_lap_by_meeting"] = [
                    _row_context(row, lap_seconds=row.get("_lap_seconds"))
                    for _, row in meeting_rows.iterrows()
                ]

            if "driver_name" in df.columns:
                group_cols = ["driver_name"]
                if "driver_number" in df.columns:
                    group_cols.append("driver_number")
                per_driver = (
                    df.assign(_lap_seconds=lap_seconds)
                    .loc[valid]
                    .groupby(group_cols, dropna=False)["_lap_seconds"]
                    .idxmin()
                )
                driver_rows = df.loc[per_driver].assign(
                    _lap_seconds=lap_seconds.loc[per_driver]
                )
                summary["fastest_lap_by_driver"] = [
                    _row_context(row, lap_seconds=row.get("_lap_seconds"))
                    for _, row in driver_rows.iterrows()
                ]

    records: dict[str, Any] = {}
    for metric in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        best = _best_row_for_metric(df, metric, mode="min")
        if best:
            records[f"best_{metric}"] = best
    for metric in ["st_speed", "i1_speed", "i2_speed", "avg_speed", "max_speed"]:
        best = _best_row_for_metric(df, metric, mode="max")
        if best:
            records[f"max_{metric}"] = best
    for metric in ["drs_pct", "full_throttle_pct", "brake_events", "hard_brake_events"]:
        best = _best_row_for_metric(df, metric, mode="max")
        if best:
            records[f"max_{metric}"] = best
    if records:
        summary["records"] = records

    if "meeting_key" in df.columns:
        counts = df.groupby("meeting_key", dropna=False).size()
        meeting_map = None
        if "meeting_name" in df.columns:
            meeting_map = (
                df[["meeting_key", "meeting_name"]]
                .drop_duplicates()
                .set_index("meeting_key")["meeting_name"]
            )
        meetings_top = []
        for key, count in counts.sort_values(ascending=False).head(10).items():
            meetings_top.append(
                {
                    "meeting_key": _jsonable_value(key),
                    "meeting_name": _jsonable_value(
                        meeting_map.get(key) if meeting_map is not None else None
                    ),
                    "laps_total": int(count),
                }
            )
        summary["laps_by_meeting_top"] = meetings_top

    if "driver_name" in df.columns:
        summary["laps_by_driver_top"] = _value_counts_with_share(df["driver_name"], limit=10)
    if "team_name" in df.columns:
        summary["laps_by_team_top"] = _value_counts_with_share(df["team_name"], limit=10)
    if "compound" in df.columns:
        summary["compound_usage"] = _value_counts_with_share(df["compound"], limit=8)

    if "is_pit_out_lap" in df.columns:
        pit = pd.to_numeric(df["is_pit_out_lap"], errors="coerce")
        if pit.notna().any():
            summary["pit_out_lap_rate"] = float(pit.mean())

    if "has_telemetry" in df.columns:
        tele = pd.to_numeric(df["has_telemetry"], errors="coerce")
        if tele.notna().any():
            summary["telemetry_coverage"] = {
                "rate": float(tele.mean()),
                "count": int(tele.sum()),
            }
    if "has_trajectory" in df.columns:
        traj = pd.to_numeric(df["has_trajectory"], errors="coerce")
        if traj.notna().any():
            summary["trajectory_coverage"] = {
                "rate": float(traj.mean()),
                "count": int(traj.sum()),
            }

    top_values: dict[str, Any] = {}
    for col in [
        "meeting_name",
        "meeting_key",
        "session_name",
        "driver_name",
        "driver_number",
        "team_name",
        "circuit_speed_class",
        "compound",
    ]:
        if col in df.columns:
            values = _value_counts(df[col])
            if values:
                top_values[col] = values
    if top_values:
        summary["top_values"] = top_values

        value_mappings = {
            "circuit_speed_class": {
                "low": "baixa velocidade",
                "medium": "media velocidade",
                "high": "alta velocidade",
            },
            "compound": {
                "SOFT": "macio",
                "MEDIUM": "medio",
                "HARD": "duro",
                "INTERMEDIATE": "intermediario",
                "WET": "chuva",
            },
        }

        def _map_value(col: str, value: str) -> Optional[str]:
            mapping = value_mappings.get(col, {})
            if col == "compound":
                key = str(value).upper().strip()
            else:
                key = str(value).lower().strip()
            return mapping.get(key)

        top_values_pt: dict[str, Any] = {}
        for col, values in top_values.items():
            if col not in value_mappings:
                continue
            mapped = []
            for item in values:
                mapped_item = dict(item)
                mapped_item["value_pt"] = _map_value(col, item.get("value", ""))
                mapped.append(mapped_item)
            top_values_pt[col] = mapped
        if top_values_pt:
            summary["top_values_pt"] = top_values_pt

        for col in ["circuit_speed_class", "compound"]:
            if col in top_values and top_values[col]:
                raw_value = top_values[col][0].get("value")
                label = _map_value(col, raw_value) if raw_value is not None else None
                summary[f"{col}_most_common"] = raw_value
                summary[f"{col}_most_common_pt"] = label

    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualizar resumo das temporadas.")
    parser.add_argument("--gold", default=None, help="Caminho para gold/consolidated.parquet")
    parser.add_argument("--output", default=None, help="Arquivo JSON de saida")
    parser.add_argument("--seasons", nargs="*", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path("./f1_dataset/data")
    gold_path = Path(args.gold) if args.gold else data_dir / "gold" / "consolidated.parquet"
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / "reports" / "season_summaries.json"
    )

    if not gold_path.exists():
        raise SystemExit(f"Arquivo gold nao encontrado: {gold_path}")

    seasons = args.seasons or DEFAULT_SEASONS

    df = pd.read_parquet(gold_path)
    summaries = {str(season): _build_season_summary(df, season) for season in seasons}

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(gold_path),
        "seasons": summaries,
    }
    _write_json_atomic(output_path, payload)
    print(f"Resumo atualizado: {output_path}")


if __name__ == "__main__":
    main()
