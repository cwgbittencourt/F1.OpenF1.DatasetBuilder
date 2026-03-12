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

from clients.mlflow_tags import with_run_context
from config.settings import ensure_paths, load_settings


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_profiles_overall_ranking.log"
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


def _metric_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    ranked = series.rank(method="average")
    if len(series) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    pct = (ranked - 1) / (len(series) - 1)
    if higher_is_better:
        return pct
    return 1.0 - pct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera ranking geral dos pilotos a partir do driver_profiles.csv."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--top-n", type=int, default=10)
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
    elif "delta_pace_mean" in df.columns:
        df["delta_pace_median_abs"] = df["delta_pace_mean"].abs()

    pace_metric = "lap_mean"
    if "lap_mean_delta_to_meeting_mean" in df.columns:
        pace_metric = "lap_mean_delta_to_meeting_mean"

    metrics = {
        pace_metric: False,
        "lap_std": False,
        "lap_quality_good_rate": True,
        "anomaly_rate": False,
        "degradation_mean": False,
        "degradation_slope": False,
        "delta_pace_median_abs": False,
        "rank_percentile_mean": False,
        "rank_percentile_median": False,
        "points_total": True,
    }

    score_components: dict[str, pd.Series] = {}
    for metric, higher_is_better in metrics.items():
        if metric not in df.columns:
            continue
        series = df[metric].copy()
        if series.isna().all():
            continue
        score_components[metric] = _metric_score(series.fillna(series.median()), higher_is_better)

    if not score_components:
        raise RuntimeError("Nenhuma metrica disponivel para ranking geral.")

    score_df = pd.DataFrame(score_components)
    performance_score = score_df.mean(axis=1)

    if "finish_rate" in df.columns:
        finish_rate = df["finish_rate"].fillna(df["finish_rate"].median())
        if "lap_completion_mean" in df.columns:
            lap_completion = df["lap_completion_mean"].fillna(
                df["lap_completion_mean"].median()
            )
            reliability_score = 0.7 * finish_rate + 0.3 * lap_completion
        else:
            reliability_score = finish_rate
        df["overall_score"] = 0.65 * performance_score + 0.35 * reliability_score
    else:
        df["overall_score"] = performance_score
    df["overall_rank"] = df["overall_score"].rank(ascending=False, method="average")

    output_cols = [
        "driver_number",
        "driver_name",
        "team_name",
        "overall_score",
        "overall_rank",
    ] + list(score_components.keys())
    for extra in [
        "meeting_lap_mean_avg",
        "lap_mean_z_to_meeting_mean",
        "delta_pace_count",
        "finish_rate",
        "dnf_rate",
        "lap_completion_mean",
        "meetings_total",
        "stint_performance_delta_mean",
        "stint_performance_delta_slope",
        "tyre_wear_slope",
        "dominant_circuit_speed_class",
        "dominant_circuit_speed_class_pct",
        "track_temperature_mean",
        "track_temperature_min",
        "track_temperature_max",
        "track_temperature_std",
        "air_temperature_mean",
        "air_temperature_min",
        "air_temperature_max",
        "air_temperature_std",
    ]:
        if extra in df.columns and extra not in output_cols:
            output_cols.append(extra)
    if "delta_pace_count" in df.columns and "delta_pace_count" not in output_cols:
        output_cols.append("delta_pace_count")
    output_df = df[output_cols].sort_values("overall_rank")

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "driver_profiles"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "driver_overall_ranking.csv"
    output_df.to_csv(csv_path, index=False)

    summary = {
        "drivers": int(output_df.shape[0]),
        "generated_at": run_timestamp,
        "top_n": args.top_n,
        "metrics": metrics,
        "source": str(profiles_path),
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    top_df = output_df.head(args.top_n)
    top_df.to_csv(artifacts_dir / "driver_overall_top.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="driver_profiles_overall_ranking") as run:
            mlflow.set_tags(with_run_context({"task": "driver_profiles_overall_ranking"}))
            mlflow.log_params({"top_n": args.top_n, "source": str(profiles_path)})
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(artifacts_dir / "summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "driver_overall_top.csv"))
            logging.getLogger(__name__).info(
                "Ranking geral registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Ranking geral gerado: %s", csv_path)


if __name__ == "__main__":
    main()
