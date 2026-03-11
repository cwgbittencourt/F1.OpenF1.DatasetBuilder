from __future__ import annotations

import json
import os
import sys
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _api_get(base: str, path: str, params: dict | None = None) -> dict:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, method="GET")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(base: str, path: str, payload: dict) -> dict:
    url = base.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MLflow API error {exc.code}: {detail}") from exc


def _get_experiment_id(base: str, name: str) -> str:
    data = _api_get(
        base, "/api/2.0/mlflow/experiments/get-by-name", {"experiment_name": name}
    )
    exp = data.get("experiment") or {}
    exp_id = exp.get("experiment_id")
    if not exp_id:
        raise RuntimeError(f"Experimento nao encontrado: {name}")
    return str(exp_id)


def _search_runs(base: str, experiment_id: str, seasons: Iterable[int]) -> list[str]:
    run_ids: list[str] = []

    filters: list[str] = []
    for season in seasons:
        filters.append(f"params.season = '{season}'")
        filters.append(f"attributes.run_name LIKE 'season={season}__%'")

    for filter_expr in filters:
        page_token: str | None = None
        while True:
            payload = {
                "experiment_ids": [experiment_id],
                "filter": filter_expr,
                "max_results": 200,
            }
            if page_token:
                payload["page_token"] = page_token
            data = _api_post(base, "/api/2.0/mlflow/runs/search", payload)
            runs = data.get("runs") or []
            for run in runs:
                run_id = (run.get("info") or {}).get("run_id")
                if run_id:
                    run_ids.append(run_id)
            page_token = data.get("next_page_token")
            if not page_token:
                break

    return sorted(set(run_ids))


def _delete_runs(base: str, run_ids: list[str]) -> None:
    for run_id in run_ids:
        try:
            _api_post(base, "/api/2.0/mlflow/runs/delete", {"run_id": run_id})
        except HTTPError as exc:
            print(f"Falha ao deletar run {run_id}: {exc}", file=sys.stderr)


def _delete_s3_artifacts(bucket: str, experiment_id: str, run_ids: list[str]) -> None:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 nao disponivel no container.") from exc

    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Credenciais ou endpoint S3 nao configurados.")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    for run_id in run_ids:
        prefix = f"{experiment_id}/{run_id}/"
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                client.delete_object(Bucket=bucket, Key=item["Key"])


def main() -> None:
    base = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "OpenF1Dataset")
    seasons = [2023, 2024]
    bucket = "mlflow"

    exp_id = _get_experiment_id(base, experiment_name)
    run_ids = _search_runs(base, exp_id, seasons)
    print(f"Encontrados {len(run_ids)} runs para seasons {seasons}.")

    if not run_ids:
        return

    _delete_runs(base, run_ids)
    _delete_s3_artifacts(bucket, exp_id, run_ids)
    print("Runs e artefatos removidos.")


if __name__ == "__main__":
    main()
