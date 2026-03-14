from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _registered_model_name(model_name: str) -> str:
    explicit = os.getenv("MLFLOW_REGISTER_MODEL_NAME")
    if explicit:
        return explicit
    prefix = os.getenv("MLFLOW_REGISTER_MODEL_PREFIX", "OpenF1")
    return f"{prefix}.{model_name}"


def register_model_if_possible(
    *,
    run_id: str,
    model_name: str,
    model_version: str | None,
    model_description: str | None = None,
) -> dict[str, Any] | None:
    if not _bool_env("MLFLOW_REGISTER_MODEL", True):
        return None
    registered_name = _registered_model_name(model_name)
    logger = logging.getLogger(__name__)

    try:
        result = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
    except Exception as exc:  # pragma: no cover - registry may be unavailable
        logger.warning("Falha ao registrar modelo '%s': %s", registered_name, exc)
        return None

    client = MlflowClient()
    try:
        if model_version:
            client.set_model_version_tag(
                registered_name, result.version, "model_version", model_version
            )
        if model_description:
            client.set_model_version_tag(
                registered_name, result.version, "model_description", model_description
            )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Falha ao setar tags da versao do modelo '%s': %s", registered_name, exc
        )

    return {"name": registered_name, "version": result.version}
