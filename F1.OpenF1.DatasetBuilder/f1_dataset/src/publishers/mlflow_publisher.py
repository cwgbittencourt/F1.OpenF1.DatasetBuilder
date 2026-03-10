from __future__ import annotations

from pathlib import Path
from typing import Any

from clients.mlflow_client import MlflowClient


class MlflowPublisher:
    def __init__(self, client: MlflowClient) -> None:
        self.client = client

    def publish(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[Path],
    ) -> None:
        self.client.log_run(run_name=run_name, params=params, metrics=metrics, artifacts=artifacts)
