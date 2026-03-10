from __future__ import annotations

from typing import Iterable
import pandas as pd


DATE_HINTS = ("date", "time")


def flatten_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep="_")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns
    ]
    return df


def parse_datetime_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    target_cols = list(columns) if columns else [
        col for col in df.columns if any(hint in col for hint in DATE_HINTS)
    ]
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def normalize_frame(records: list[dict]) -> pd.DataFrame:
    df = flatten_records(records)
    if df.empty:
        return df
    df = standardize_columns(df)
    df = parse_datetime_columns(df)
    return df
