from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.utils import SECTOR_COLUMNS, build_preprocessor, split_indices


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "relative_position.log"
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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    epsilon = 1e-9
    denom = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def _log_model_safe(pipeline: Pipeline, artifacts_dir: Path) -> None:
    import inspect

    sig = inspect.signature(mlflow.sklearn.log_model)
    if "serialization_format" in sig.parameters:
        kwargs: dict[str, object] = {"serialization_format": "skops"}
        if "skops_trusted_types" in sig.parameters:
            kwargs["skops_trusted_types"] = ["numpy.dtype"]
        mlflow.sklearn.log_model(pipeline, "model", **kwargs)
        return

    try:
        import skops.io as sio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("skops nao disponivel para persistencia segura.") from exc

    model_path = artifacts_dir / "model.skops"
    sio.dump(pipeline, model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.set_tag("model_format", "skops_artifact")


def _build_driver_meeting_dataset(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number"]
    working = df[df["lap_duration"].notna()].copy()

    agg_map: dict[str, str] = {
        "lap_duration": "mean",
        "lap_number": "count",
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
        "telemetry_points": "mean",
        "trajectory_points": "mean",
    }

    available = {k: v for k, v in agg_map.items() if k in working.columns}
    driver_df = working.groupby(group_cols).agg(available).reset_index()

    for col in ["driver_name", "team_name", "meeting_name", "session_name", "season"]:
        if col in working.columns:
            driver_df[col] = working.groupby(group_cols)[col].first().values

    for col in SECTOR_COLUMNS:
        if col in driver_df.columns:
            driver_df = driver_df.drop(columns=col)

    driver_df = driver_df.rename(columns={"lap_number": "lap_count"})

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


def _spearman_by_meeting(df: pd.DataFrame) -> float:
    scores: list[float] = []
    for meeting_key, group in df.groupby("meeting_key"):
        if len(group) < 3:
            continue
        actual = group["rank_percentile"].to_numpy()
        pred = group["pred_rank"].to_numpy()
        actual_rank = pd.Series(actual).rank(method="average").to_numpy()
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        corr = float(np.corrcoef(actual_rank, pred_rank)[0, 1])
        scores.append(corr)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediz posicao relativa por meeting."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument(
        "--group-col",
        default="meeting_key",
        help="Coluna usada para split por grupo (default: meeting_key).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    df = load_consolidated()
    driver_df = _build_driver_meeting_dataset(df)

    target = driver_df["rank_percentile"]
    features = driver_df.drop(
        columns=[
            "rank_percentile",
            "rank",
            "drivers_in_meeting",
            "lap_duration",
        ],
        errors="ignore",
    )

    train_idx, test_idx = split_indices(
        driver_df, args.group_col, args.test_size, args.random_state
    )
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    metrics = _regression_metrics(y_test.to_numpy(), preds)

    eval_df = driver_df.loc[x_test.index].copy()
    eval_df["pred_rank"] = preds
    metrics["rank_spearman_mean"] = _spearman_by_meeting(eval_df)

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "relative_position"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "group_col": args.group_col,
        "test_size": args.test_size,
    }
    feature_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "features.json").write_text(
        json.dumps(feature_summary, indent=2), encoding="utf-8"
    )
    eval_df.to_csv(artifacts_dir / "ranking_predictions.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="relative_position") as run:
            mlflow.set_tags({"task": "relative_position", "target": "rank_percentile"})
            params: dict[str, Any] = {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "group_col": args.group_col,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "feature_count": len(x_train.columns),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "split_summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "ranking_predictions.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Treino concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
