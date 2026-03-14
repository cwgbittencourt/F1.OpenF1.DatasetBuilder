from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.mlflow_metadata import build_mlflow_tags
from modeling.mlflow_registry import register_model_if_possible
from modeling.system_metrics import SystemMetrics
from modeling.utils import build_preprocessor, split_indices

MODEL_NAME = "stint_delta_pace"
MODEL_DESCRIPTION = "Prediz delta de ritmo entre stints."


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "stint_delta_pace.log"
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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    epsilon = 1e-9
    denom = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


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


def _build_stint_dataset(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df[df["lap_duration"].notna()].copy()

    agg_map: dict[str, str] = {
        "lap_duration": "mean",
        "lap_number": "count",
        "tyre_age_at_start": "first",
        "compound": "first",
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
        "telemetry_points": "mean",
        "trajectory_points": "mean",
    }

    available = {k: v for k, v in agg_map.items() if k in working.columns}
    stint_df = working.groupby(group_cols).agg(available).reset_index()
    stint_df = stint_df.rename(columns={"lap_number": "stint_length"})

    stint_df = stint_df.sort_values(group_cols)
    stint_df["prev_mean_lap"] = stint_df.groupby(
        ["meeting_key", "session_key", "driver_number"]
    )["lap_duration"].shift(1)
    stint_df["delta_pace"] = stint_df["lap_duration"] - stint_df["prev_mean_lap"]

    stint_df = stint_df[stint_df["delta_pace"].notna()]
    return stint_df


def _apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.season is not None:
        if "season" not in filtered.columns:
            raise ValueError("Coluna season nao encontrada no gold.")
        filtered = filtered[filtered["season"] == args.season]
    if args.meeting_key:
        if "meeting_key" not in filtered.columns:
            raise ValueError("Coluna meeting_key nao encontrada no gold.")
        filtered = filtered[filtered["meeting_key"].astype(str) == str(args.meeting_key)]
    if args.session_name and args.session_name.lower() != "all":
        if "session_name" not in filtered.columns:
            raise ValueError("Coluna session_name nao encontrada no gold.")
        filtered = filtered[
            filtered["session_name"].astype(str).str.strip().str.lower()
            == args.session_name.lower()
        ]
    if args.driver_number is not None:
        if "driver_number" not in filtered.columns:
            raise ValueError("Coluna driver_number nao encontrada no gold.")
        filtered = filtered[
            filtered["driver_number"].astype(str) == str(args.driver_number)
        ]
    if args.constructor:
        if "team_name" not in filtered.columns:
            raise ValueError("Coluna team_name nao encontrada no gold.")
        filtered = filtered[
            filtered["team_name"].astype(str).str.strip().str.lower()
            == str(args.constructor).strip().lower()
        ]
    if filtered.empty:
        raise ValueError("Nenhum dado encontrado no gold para os filtros informados.")
    return filtered


def _build_stint_lap_dataset(df: pd.DataFrame, baseline_laps: int) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df[df["lap_duration"].notna()].copy()
    if "lap_number" in working.columns:
        working = working.sort_values(group_cols + ["lap_number"])
    else:
        working = working.sort_values(group_cols)

    working["lap_in_stint"] = working.groupby(group_cols).cumcount() + 1
    baseline = (
        working[working["lap_in_stint"] <= baseline_laps]
        .groupby(group_cols)["lap_duration"]
        .mean()
        .rename("baseline_lap")
    )
    working = working.merge(baseline, on=group_cols, how="left")
    working = working[working["lap_in_stint"] > baseline_laps]
    working["delta_pace"] = working["lap_duration"] - working["baseline_lap"]
    working = working[working["delta_pace"].notna()]
    return working


def main() -> None:
    system_metrics = SystemMetrics.start()
    parser = argparse.ArgumentParser(
        description="Treina modelo para delta de ritmo entre stints."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument(
        "--group-col",
        default="meeting_key",
        help="Coluna usada para split por grupo (default: meeting_key).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--meeting-key", default=None)
    parser.add_argument("--session-name", default="all")
    parser.add_argument("--driver-number", type=int, default=None)
    parser.add_argument("--constructor", default=None)
    parser.add_argument(
        "--target-mode",
        choices=["prev_stint_mean", "stint_start_mean"],
        default="prev_stint_mean",
        help="Alvo: delta entre stints (prev_stint_mean) ou "
        "delta vs media das primeiras voltas do stint (stint_start_mean).",
    )
    parser.add_argument(
        "--baseline-laps",
        type=int,
        default=3,
        help="Numero de voltas iniciais do stint para baseline (stint_start_mean).",
    )
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    if not settings.output.register_mlflow:
        raise RuntimeError("REGISTER_MLFLOW=false; MLflow e obrigatorio para este treino.")
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    if args.baseline_laps < 1:
        raise ValueError("baseline_laps deve ser >= 1.")
    if args.session_name.lower() not in {"race", "sprint", "all"}:
        raise ValueError("session_name deve ser Race, Sprint ou all.")

    df = load_consolidated()
    df = _apply_filters(df, args)

    if args.target_mode == "stint_start_mean":
        stint_df = _build_stint_lap_dataset(df, args.baseline_laps)
        drop_cols = ["delta_pace", "baseline_lap", "lap_duration"]
    else:
        stint_df = _build_stint_dataset(df)
        drop_cols = ["delta_pace", "prev_mean_lap", "lap_duration"]

    target = stint_df["delta_pace"]
    features = stint_df.drop(columns=drop_cols, errors="ignore")

    train_idx, test_idx = split_indices(
        stint_df, args.group_col, args.test_size, args.random_state
    )
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    metrics = _regression_metrics(y_test.to_numpy(), preds)

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "stint_delta_pace"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "group_col": args.group_col,
        "test_size": args.test_size,
        "target_mode": args.target_mode,
        "baseline_laps": args.baseline_laps,
    }
    feature_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "features.json").write_text(
        json.dumps(feature_summary, indent=2), encoding="utf-8"
    )

    predictions_sample = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": preds,
        }
    ).head(200)
    predictions_sample.to_csv(artifacts_dir / "predictions_sample.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="stint_delta_pace", log_system_metrics=True) as run:
            tags = {"task": "stint_delta_pace", "target": "delta_pace"}
            tags.update(build_mlflow_tags(MODEL_NAME, MODEL_DESCRIPTION, run_timestamp))
            mlflow.set_tags(tags)
            params: dict[str, Any] = {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "group_col": args.group_col,
                "target_mode": args.target_mode,
                "baseline_laps": args.baseline_laps,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "feature_count": len(x_train.columns),
            }
            model_version = os.getenv("MODEL_VERSION")
            if model_version:
                params["model_version"] = model_version
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(system_metrics.collect())
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "split_summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "predictions_sample.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            register_model_if_possible(
                run_id=run.info.run_id,
                model_name=MODEL_NAME,
                model_version=model_version or run_timestamp,
                model_description=MODEL_DESCRIPTION,
            )
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Treino concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
