from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from config.settings import ensure_paths, load_settings


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_profiles_rankings.log"
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


def _find_latest_profiles(artifacts_dir: Path) -> Path:
    root = artifacts_dir / "modeling" / "driver_profiles"
    if not root.exists():
        raise FileNotFoundError(f"Nenhum relatorio encontrado em {root}")

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        reverse=True,
        key=lambda p: p.name,
    )
    for candidate in candidates:
        csv_path = candidate / "driver_profiles.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError("driver_profiles.csv nao encontrado.")


def _add_ranking(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    metric: str,
    direction: str,
    top_n: int,
) -> None:
    if metric not in df.columns:
        return
    subset = df[["driver_number", "driver_name", metric]].dropna()
    if subset.empty:
        return

    ascending = direction == "asc"
    ranked = subset.sort_values(metric, ascending=ascending).head(top_n)
    for idx, row in enumerate(ranked.itertuples(index=False), start=1):
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "rank": idx,
                "driver_number": row.driver_number,
                "driver_name": row.driver_name,
                "value": float(row._asdict()[metric]),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera rankings a partir do relatorio de pilotos."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    artifacts_root = Path(settings.paths.artifacts_dir)
    profiles_path = _find_latest_profiles(artifacts_root)
    df = pd.read_csv(profiles_path)

    if "delta_pace_median" in df.columns:
        df["delta_pace_median_abs"] = df["delta_pace_median"].abs()
    else:
        df["delta_pace_median_abs"] = df["delta_pace_mean"].abs()

    pace_metric = "lap_mean"
    if "lap_mean_delta_to_meeting_mean" in df.columns:
        pace_metric = "lap_mean_delta_to_meeting_mean"

    rows: list[dict[str, object]] = []
    metrics = [
        (pace_metric, "asc"),
        ("lap_std", "asc"),
        ("lap_quality_good_rate", "desc"),
        ("anomaly_rate", "asc"),
        ("degradation_mean", "asc"),
        ("degradation_slope", "asc"),
        ("delta_pace_median_abs", "asc"),
        ("rank_percentile_mean", "asc"),
        ("rank_percentile_median", "asc"),
        ("finish_rate", "desc"),
        ("lap_completion_mean", "desc"),
        ("points_total", "desc"),
    ]
    for metric, direction in metrics:
        _add_ranking(rows, df, metric, direction, args.top_n)

    rankings_df = pd.DataFrame(rows)

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "driver_profiles"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "driver_rankings.csv"
    rankings_df.to_csv(csv_path, index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="driver_profiles_rankings") as run:
            mlflow.set_tags({"task": "driver_profiles_rankings"})
            mlflow.log_params({"top_n": args.top_n, "source": str(profiles_path)})
            mlflow.log_artifact(str(csv_path))
            logging.getLogger(__name__).info(
                "Ranking registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Ranking gerado: %s", csv_path)


if __name__ == "__main__":
    main()
