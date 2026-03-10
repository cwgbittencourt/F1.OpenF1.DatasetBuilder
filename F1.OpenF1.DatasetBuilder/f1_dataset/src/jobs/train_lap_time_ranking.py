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
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.pipeline import Pipeline

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.utils import TARGET_COLUMN, build_preprocessor, get_feature_frame, split_indices


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "lap_time_ranking.log"
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
    except Exception as exc:  # pragma: no cover - fallback safety
        raise RuntimeError(
            "skops nao disponivel para persistencia segura."
        ) from exc

    model_path = artifacts_dir / "model.skops"
    sio.dump(pipeline, model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.set_tag("model_format", "skops_artifact")


def _ranking_metrics(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    driver_col: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    ranking_df = meta_df[[group_col, driver_col]].copy()
    ranking_df["y_true"] = y_true
    ranking_df["y_pred"] = y_pred

    summary = (
        ranking_df.groupby([group_col, driver_col], as_index=False)
        .mean(numeric_only=True)
        .rename(columns={"y_true": "actual_mean_lap", "y_pred": "pred_mean_lap"})
    )

    group_metrics: list[dict[str, float]] = []
    for meeting_key, group in summary.groupby(group_col):
        if len(group) < 3:
            continue
        actual = group["actual_mean_lap"].to_numpy()
        pred = group["pred_mean_lap"].to_numpy()

        actual_rank = pd.Series(actual).rank(method="average").to_numpy()
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(actual_rank, pred_rank)[0, 1])

        relevance = 1.0 / (actual + 1e-6)
        scores = 1.0 / (pred + 1e-6)
        ndcg = float(ndcg_score([relevance], [scores]))

        group_metrics.append(
            {
                "meeting_key": float(meeting_key),
                "spearman": spearman,
                "ndcg": ndcg,
                "drivers": float(len(group)),
            }
        )

    metrics_df = pd.DataFrame(group_metrics)
    metrics = {
        "rank_meeting_count": float(len(metrics_df)),
        "rank_spearman_mean": float(metrics_df["spearman"].mean())
        if not metrics_df.empty
        else float("nan"),
        "rank_ndcg_mean": float(metrics_df["ndcg"].mean())
        if not metrics_df.empty
        else float("nan"),
        "rank_driver_mean": float(metrics_df["drivers"].mean())
        if not metrics_df.empty
        else float("nan"),
    }

    summary["actual_rank"] = summary.groupby(group_col)["actual_mean_lap"].rank(
        method="average"
    )
    summary["pred_rank"] = summary.groupby(group_col)["pred_mean_lap"].rank(
        method="average"
    )

    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lap time model and evaluate driver ranking."
    )
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument(
        "--include-sectors",
        action="store_true",
        help="Inclui features de setores (por padrao sao excluidas).",
    )
    parser.add_argument(
        "--group-col",
        default="meeting_key",
        help="Coluna usada para split por grupo (default: meeting_key).",
    )
    parser.add_argument(
        "--driver-col",
        default="driver_name",
        help="Coluna usada para ranking de pilotos (default: driver_name).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    exclude_sectors = not args.include_sectors

    df = load_consolidated()
    logging.getLogger(__name__).info(
        "Dataset carregado: %s linhas, %s colunas", len(df), len(df.columns)
    )

    features, target, dropped = get_feature_frame(
        df, target=TARGET_COLUMN, exclude_sectors=exclude_sectors
    )
    df_model = df.loc[features.index]

    train_idx, test_idx = split_indices(
        df_model, args.group_col, args.test_size, args.random_state
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

    rank_metrics, rank_summary = _ranking_metrics(
        df_model.loc[x_test.index],
        y_test.to_numpy(),
        preds,
        args.group_col,
        args.driver_col,
    )
    metrics.update(rank_metrics)

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "lap_time_ranking"
        / run_timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "group_col": args.group_col,
        "test_size": args.test_size,
    }
    feature_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dropped_columns": dropped,
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
    rank_summary.to_csv(artifacts_dir / "ranking_summary.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="lap_time_ranking") as run:
            mlflow.set_tags(
                {
                    "task": "lap_time_ranking",
                    "exclude_sectors": str(exclude_sectors).lower(),
                }
            )
            params: dict[str, Any] = {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "group_col": args.group_col,
                "driver_col": args.driver_col,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "feature_count": len(x_train.columns),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "split_summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "ranking_summary.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Treino concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
