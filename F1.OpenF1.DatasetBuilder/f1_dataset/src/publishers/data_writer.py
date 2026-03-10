from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _partition_path(base_dir: str, unit: dict[str, Any]) -> Path:
    path = Path(base_dir)
    path = path / f"season={unit['season']}"
    path = path / f"meeting_key={unit['meeting_key']}"
    path = path / f"session_key={unit['session_key']}"
    path = path / f"driver_number={unit['driver_number']}"
    return path


def write_bronze(records: list[dict[str, Any]], base_dir: str, unit: dict[str, Any], name: str) -> Path:
    path = _partition_path(base_dir, unit)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{name}.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    return file_path


def write_silver(df: pd.DataFrame, base_dir: str, unit: dict[str, Any], name: str) -> Path:
    path = _partition_path(base_dir, unit)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{name}.parquet"
    if not df.empty:
        df.to_parquet(file_path, index=False)
    return file_path


def write_gold(df: pd.DataFrame, base_dir: str, unit: dict[str, Any], formats: list[str]) -> list[Path]:
    path = _partition_path(base_dir, unit)
    path.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    if df.empty:
        return artifacts

    if "parquet" in formats:
        file_path = path / "dataset.parquet"
        df.to_parquet(file_path, index=False)
        artifacts.append(file_path)
    if "csv" in formats:
        file_path = path / "dataset.csv"
        df.to_csv(file_path, index=False)
        artifacts.append(file_path)
    return artifacts
