from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from modeling.dataset import consolidated_path


def _timestamp_from_mtime(path: Path) -> str | None:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_mlflow_tags(
    model_name: str,
    description: str,
    run_timestamp: str | None = None,
) -> dict[str, str]:
    dataset_path = Path(os.getenv("DATASET_PATH") or consolidated_path())
    dataset_name = os.getenv("DATASET_NAME", "gold.consolidated")
    dataset_version = os.getenv("DATASET_VERSION") or _timestamp_from_mtime(dataset_path)
    model_version = os.getenv("MODEL_VERSION") or run_timestamp
    model_description = os.getenv("MODEL_DESCRIPTION") or description

    if not dataset_version:
        dataset_version = "unknown"
    if not model_version:
        model_version = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    tags = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "dataset_version": dataset_version,
        "model_name": model_name,
        "model_version": model_version,
        "model_description": model_description,
        "mlflow.note.content": model_description,
    }
    return {key: str(value) for key, value in tags.items() if value is not None}
