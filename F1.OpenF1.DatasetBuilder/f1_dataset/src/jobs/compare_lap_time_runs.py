from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

from config.settings import ensure_paths, load_settings


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "lap_time_comparison.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _init_mlflow(tracking_uri: str | None, experiment_name: str) -> str:
    if not tracking_uri:
        raise RuntimeError("MLflow tracking URI nao configurado.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise RuntimeError(f"Experimento MLflow nao encontrado: {experiment_name}")
    return experiment.experiment_id


def _latest_run_for(experiment_id: str, task: str, exclude_sectors: bool) -> pd.Series:
    flag = str(exclude_sectors).lower()
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.task = '{task}' and tags.exclude_sectors = '{flag}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"Nenhum run encontrado para task={task} exclude_sectors={flag}"
        )
    return runs.iloc[0]


def _row_from_run(run: pd.Series, task: str, exclude_sectors: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "task": task,
        "exclude_sectors": exclude_sectors,
        "run_id": run.get("run_id"),
        "run_name": run.get("tags.mlflow.runName"),
        "start_time": run.get("start_time"),
        "status": run.get("status"),
    }

    for key, value in run.items():
        if isinstance(key, str) and key.startswith("metrics."):
            row[key] = value
        if isinstance(key, str) and key.startswith("params."):
            row[key] = value

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida comparativo de runs de lap time no MLflow."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    experiment_id = _init_mlflow(tracking_uri, settings.mlflow.experiment_name)

    targets = [
        ("lap_time_regression", True),
        ("lap_time_regression", False),
        ("lap_time_ranking", True),
        ("lap_time_ranking", False),
    ]

    rows: list[dict[str, object]] = []
    for task, exclude_sectors in targets:
        run = _latest_run_for(experiment_id, task, exclude_sectors)
        rows.append(_row_from_run(run, task, exclude_sectors))

    comparison_df = pd.DataFrame(rows)
    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "comparisons"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "lap_time_model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    summary = {
        "rows": len(comparison_df),
        "generated_at": run_timestamp,
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with mlflow.start_run(run_name="lap_time_comparison") as run:
        mlflow.set_tag("task", "lap_time_comparison")
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(artifacts_dir / "summary.json"))
        logging.getLogger(__name__).info(
            "Comparativo registrado no MLflow: %s", run.info.run_id
        )

    logging.getLogger(__name__).info(
        "CSV gerado: %s (%s linhas)", csv_path, len(comparison_df)
    )


if __name__ == "__main__":
    main()
