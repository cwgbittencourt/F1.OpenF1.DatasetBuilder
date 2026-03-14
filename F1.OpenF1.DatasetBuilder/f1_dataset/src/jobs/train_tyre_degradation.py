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
from modeling.mlflow_metadata import build_mlflow_tags
from modeling.mlflow_registry import register_model_if_possible
from modeling.system_metrics import SystemMetrics
from modeling.utils import build_preprocessor, get_feature_frame, split_indices

MODEL_NAME = "tyre_degradation"
MODEL_DESCRIPTION = "Prediz degradacao de pneus por stint."


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "tyre_degradation.log"
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


def _build_degradation_target(df: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    system_metrics = SystemMetrics.start()
    parser = argparse.ArgumentParser(
        description="Treina modelo para degradacao de pneus por stint."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument(
        "--include-sectors",
        action="store_true",
        help="Inclui features de setores (por padrao sao excluidas).",
    )
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

    exclude_sectors = not args.include_sectors

    df = load_consolidated()
    df = _build_degradation_target(df)
    df = df[df["tyre_age_at_lap"].notna()]

    features, target, dropped = get_feature_frame(
        df, target="degradation_delta", exclude_sectors=exclude_sectors
    )
    df_model = df.loc[features.index]

    train_idx, test_idx = split_indices(
        df_model, args.group_col, args.test_size, args.random_state
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

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "tyre_degradation"
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
        "dropped_columns": dropped,
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

    predictions_sample = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": preds,
        }
    ).head(500)
    predictions_sample.to_csv(artifacts_dir / "predictions_sample.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="tyre_degradation", log_system_metrics=True) as run:
            tags = {
                "task": "tyre_degradation",
                "exclude_sectors": str(exclude_sectors).lower(),
                "target": "degradation_delta",
            }
            tags.update(build_mlflow_tags(MODEL_NAME, MODEL_DESCRIPTION, run_timestamp))
            mlflow.set_tags(tags)
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
            model_version = os.getenv("MODEL_VERSION")
            if model_version:
                params["model_version"] = model_version
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(system_metrics.collect())
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "split_summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "predictions_sample.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            register_model_if_possible(
                run_id=run.info.run_id,
                model_name=MODEL_NAME,
                model_version=model_version or run_timestamp,
                model_description=MODEL_DESCRIPTION,
            )
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Treino concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
