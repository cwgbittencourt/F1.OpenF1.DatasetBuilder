from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "lap_duration"
SECTOR_COLUMNS = [
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "segments_sector_1",
    "segments_sector_2",
    "segments_sector_3",
]


def get_feature_frame(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    exclude_sectors: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    df = df.copy()
    df = df[df[target].notna()]

    drop_cols: list[str] = [target]
    if exclude_sectors:
        drop_cols.extend([col for col in SECTOR_COLUMNS if col in df.columns])

    features = df.drop(columns=drop_cols, errors="ignore")
    bool_cols = features.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        features[bool_cols] = features[bool_cols].astype("int8")

    return features, df[target], drop_cols


def split_indices(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.Index, pd.Index]:
    if group_col in df.columns and df[group_col].nunique() > 1:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        return df.index[train_idx], df.index[test_idx]

    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state
    )
    return train_idx, test_idx


def build_preprocessor(
    features: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    def is_categorical(series: pd.Series) -> bool:
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )

    categorical_cols = [col for col in features.columns if is_categorical(features[col])]
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols
