from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

from config.settings import Settings


class MlflowClient:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.output.register_mlflow
        self.tracking_uri = settings.mlflow.tracking_uri
        self.experiment_name = settings.mlflow.experiment_name
        self.logger = logging.getLogger(__name__)

        if self.enabled:
            if not self.tracking_uri:
                tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
                tracking_path.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = f"file:{tracking_path}"
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[Path],
        tags: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
