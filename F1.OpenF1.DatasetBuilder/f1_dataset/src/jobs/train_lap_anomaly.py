from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.utils import build_preprocessor


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "lap_anomaly.log"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deteccao de anomalias por volta."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    df = load_consolidated()
    df = df[df["lap_duration"].notna()].copy()

    exclude_cols = []
    features = df.drop(columns=exclude_cols, errors="ignore")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(features)
    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(features)

    scores = pipeline.decision_function(features)
    preds = pipeline.predict(features)
    is_anomaly = preds == -1

    metrics = {
        "rows": float(len(features)),
        "anomaly_count": float(is_anomaly.sum()),
        "anomaly_rate": float(is_anomaly.mean()),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
    }

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir) / "modeling" / "lap_anomaly" / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    anomaly_df = df.copy()
    anomaly_df["anomaly_score"] = scores
    anomaly_df["is_anomaly"] = is_anomaly
    top_anomalies = anomaly_df.sort_values("anomaly_score").head(200)
    top_anomalies.to_csv(artifacts_dir / "top_anomalies.csv", index=False)

    feature_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "features.json").write_text(
        json.dumps(feature_summary, indent=2), encoding="utf-8"
    )

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="lap_anomaly") as run:
            mlflow.set_tags({"task": "lap_anomaly"})
            mlflow.log_params(
                {
                    "contamination": args.contamination,
                    "n_estimators": args.n_estimators,
                    "random_state": args.random_state,
                    "feature_count": len(features.columns),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "top_anomalies.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Deteccao concluida. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
