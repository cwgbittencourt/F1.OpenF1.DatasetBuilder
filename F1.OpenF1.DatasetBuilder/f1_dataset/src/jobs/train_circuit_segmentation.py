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


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "circuit_segmentation.log"
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


def _build_meeting_dataset(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key"]
    working = df[df["lap_duration"].notna()].copy()

    agg_map: dict[str, str] = {
        "lap_duration": "mean",
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
    }

    available = {k: v for k, v in agg_map.items() if k in working.columns}
    meeting_df = working.groupby(group_cols).agg(available).reset_index()

    for col in ["meeting_name", "season"]:
        if col in working.columns:
            meeting_df[col] = working.groupby(group_cols)[col].first().values

    return meeting_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentacao de circuitos por comportamento agregado."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    df = load_consolidated()
    meeting_df = _build_meeting_dataset(df)

    numeric_cols = meeting_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "meeting_key"]
    feature_matrix = meeting_df[numeric_cols].fillna(meeting_df[numeric_cols].median())

    rows = len(meeting_df)
    effective_clusters = min(int(args.clusters), rows) if rows else 0

    if rows < 2 or effective_clusters < 2:
        meeting_df["cluster"] = 0
        labels = meeting_df["cluster"].to_numpy()
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
        meeting_df["cluster"] = labels

    clusters_logged = float(effective_clusters) if effective_clusters else 0.0
    metrics = {"clusters": clusters_logged, "rows": float(len(meeting_df))}
    if len(meeting_df) >= 2 and len(set(labels)) > 1:
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
        / "circuit_segmentation"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    meeting_df.to_csv(artifacts_dir / "circuit_clusters.csv", index=False)
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "features.json").write_text(
        json.dumps({"feature_cols": numeric_cols}, indent=2), encoding="utf-8"
    )

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="circuit_segmentation") as run:
            mlflow.set_tags({"task": "circuit_segmentation"})
            mlflow.log_params(
                {
                    "clusters": args.clusters,
                    "random_state": args.random_state,
                    "feature_count": len(numeric_cols),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "circuit_clusters.csv"))
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Segmentacao concluida. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
