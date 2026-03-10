from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def consolidated_path() -> Path:
    data_dir = os.getenv("DATA_DIR", "./f1_dataset/data")
    return Path(data_dir) / "gold" / "consolidated.parquet"


def load_consolidated() -> pd.DataFrame:
    path = consolidated_path()
    if not path.exists():
        raise FileNotFoundError(f"Consolidated dataset not found: {path}")
    return pd.read_parquet(path)
