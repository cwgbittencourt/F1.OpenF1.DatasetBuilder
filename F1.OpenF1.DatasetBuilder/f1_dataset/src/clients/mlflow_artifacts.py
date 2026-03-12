from __future__ import annotations

from typing import Any

from mlflow.tracking import MlflowClient as TrackingClient


def get_tracking_client(
    tracking_uri: str,
    experiment_name: str,
    create_if_missing: bool = True,
) -> tuple[TrackingClient, str]:
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI vazio; configure o MLflow remoto.")
    client = TrackingClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        if not create_if_missing:
            raise RuntimeError(f"Experimento MLflow nao encontrado: {experiment_name}")
        exp_id = client.create_experiment(experiment_name)
        return client, exp_id
    return client, experiment.experiment_id


def find_latest_run(
    client: TrackingClient,
    experiment_id: str,
    tags: dict[str, Any],
) -> Any:
    filters = [f"tags.{key} = '{value}'" for key, value in tags.items()]
    filter_string = " and ".join(filters)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Nenhuma run encontrada para filtro: {filter_string}")
    return runs[0]


def artifact_uri(run: Any, filename: str) -> str:
    base_uri = run.info.artifact_uri.rstrip("/")
    return f"{base_uri}/{filename}"
