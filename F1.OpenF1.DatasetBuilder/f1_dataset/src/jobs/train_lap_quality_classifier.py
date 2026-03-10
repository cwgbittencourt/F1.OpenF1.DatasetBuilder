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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config.settings import ensure_paths, load_settings
from modeling.dataset import load_consolidated
from modeling.utils import build_preprocessor, get_feature_frame, split_indices


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "lap_quality_classifier.log"
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


def _label_lap_quality(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["meeting_key", "session_key", "driver_number", "stint_number"]
    working = df.copy()
    working = working[working["lap_duration"].notna()]
    group_sizes = working.groupby(group_cols)["lap_duration"].transform("count")
    working = working[group_sizes >= 4]

    q25 = working.groupby(group_cols)["lap_duration"].transform(
        lambda s: s.quantile(0.25)
    )
    q75 = working.groupby(group_cols)["lap_duration"].transform(
        lambda s: s.quantile(0.75)
    )

    working["lap_quality"] = np.where(
        working["lap_duration"] <= q25,
        1,
        np.where(working["lap_duration"] >= q75, 0, np.nan),
    )
    working = working[working["lap_quality"].notna()]
    working["lap_quality"] = working["lap_quality"].astype(int)
    return working


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica qualidade de volta (boa vs ruim)."
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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    exclude_sectors = not args.include_sectors

    df = load_consolidated()
    df = _label_lap_quality(df)

    features, target, dropped = get_feature_frame(
        df, target="lap_quality", exclude_sectors=exclude_sectors
    )
    df_model = df.loc[features.index]

    train_idx, test_idx = split_indices(
        df_model, args.group_col, args.test_size, args.random_state
    )
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(settings.paths.artifacts_dir)
        / "modeling"
        / "lap_quality_classifier"
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

    predictions_sample = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": preds,
            "y_prob": proba,
        }
    ).head(500)
    predictions_sample.to_csv(artifacts_dir / "predictions_sample.csv", index=False)

    tracking_uri = settings.mlflow.tracking_uri
    if not tracking_uri and settings.output.register_mlflow:
        tracking_path = Path(settings.paths.artifacts_dir) / "mlruns"
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_path}"

    if settings.output.register_mlflow and _init_mlflow(
        tracking_uri, settings.mlflow.experiment_name
    ):
        with mlflow.start_run(run_name="lap_quality_classifier") as run:
            mlflow.set_tags(
                {
                    "task": "lap_quality_classifier",
                    "exclude_sectors": str(exclude_sectors).lower(),
                    "target": "lap_quality",
                }
            )
            params: dict[str, Any] = {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "group_col": args.group_col,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "feature_count": len(x_train.columns),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
            mlflow.log_artifact(str(artifacts_dir / "split_summary.json"))
            mlflow.log_artifact(str(artifacts_dir / "features.json"))
            mlflow.log_artifact(str(artifacts_dir / "predictions_sample.csv"))
            _log_model_safe(pipeline, artifacts_dir)
            logging.getLogger(__name__).info(
                "Run registrado no MLflow: %s", run.info.run_id
            )

    logging.getLogger(__name__).info("Treino concluido. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
