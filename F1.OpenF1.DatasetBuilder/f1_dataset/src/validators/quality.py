from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "season",
    "meeting_key",
    "meeting_name",
    "session_key",
    "session_name",
    "driver_number",
    "driver_name",
    "team_name",
    "lap_number",
    "lap_duration",
]


def validate_lap_dataset(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "rows": 0,
            "null_pct": 1.0,
            "valid_laps": 0,
            "discarded_laps": 0,
        }

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing_cols}")

    null_pct = float(df.isna().mean().mean())
    valid_laps = int(df["lap_number"].notna().sum())
    discarded = int(len(df) - valid_laps)

    return {
        "rows": float(len(df)),
        "null_pct": null_pct,
        "valid_laps": float(valid_laps),
        "discarded_laps": float(discarded),
    }
