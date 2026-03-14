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
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.mlflow_metadata import build_mlflow_tags
from modeling.system_metrics import SystemMetrics
from modeling.utils import SECTOR_COLUMNS

MODEL_NAME = "driver_style_clustering"
MODEL_DESCRIPTION = "Clustering de estilo de pilotagem."


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_style_clustering.log"
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


def main() -> None:
    system_metrics = SystemMetrics.start()
    parser = argparse.ArgumentParser(
        description="Clustering de estilo de pilotagem."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    df = load_consolidated()

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    exclude_cols = {
        "lap_duration",
        "lap_number",
        "meeting_key",
        "session_key",
        "driver_number",
        "season",
        "stint_number",
        "stint_lap_start",
        "stint_lap_end",
        "tyre_age_at_start",
        "tyre_age_at_lap",
    }
    exclude_cols.update([col for col in SECTOR_COLUMNS if col in df.columns])

    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    if not feature_cols:
        raise RuntimeError("Nenhuma feature numerica disponivel para clustering.")

    group_cols = ["driver_name", "team_name", "season"]
    style_df = (
        df.groupby(group_cols)[feature_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    feature_matrix = style_df[feature_cols].fillna(style_df[feature_cols].median())

    rows = len(style_df)
    effective_clusters = min(int(args.clusters), rows) if rows else 0

    if rows < 2 or effective_clusters < 2:
        style_df["cluster"] = 0
        labels = style_df["cluster"].to_numpy()
    else:
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "cluster",
                    KMeans(
                        n_clusters=effective_clusters,
                        n_init=10,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )

        labels = pipeline.fit_predict(feature_matrix)
        style_df["cluster"] = labels

    clusters_logged = float(effective_clusters) if effective_clusters else 0.0
    metrics = {"clusters": clusters_logged, "rows": float(len(style_df))}
    if len(style_df) >= 2 and len(set(labels)) > 1:
        metrics["silhouette"] = float(silhouette_score(feature_matrix, labels))
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(feature_matrix, labels)
        )
    else:
        metrics["silhouette"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "driver_style_clustering"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    style_df.to_csv(artifacts_dir / "driver_clusters.csv", index=False)
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "features.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="driver_style_clustering", log_system_metrics=True) as run:
            tags = {"task": "driver_style_clustering"}
            tags.update(build_mlflow_tags(MODEL_NAME, MODEL_DESCRIPTION, run_timestamp))
            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    "clusters": args.clusters,
                    "random_state": args.random_state,
                    "feature_count": len(feature_cols),
                }
            )
            model_version = os.getenv("MODEL_VERSION")
            if model_version:
                mlflow.log_param("model_version", model_version)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(system_metrics.collect())
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "driver_clusters.csv"))
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Clustering concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
