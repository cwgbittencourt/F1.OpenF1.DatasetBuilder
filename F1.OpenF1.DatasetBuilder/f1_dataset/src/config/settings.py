from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import yaml


@dataclass
class DriverFilter:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class MeetingsFilter:
    mode: str = "all"
    include: list[str] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    max_parallel_drivers: int = 1
    max_http_connections: int = 10
    min_request_interval_ms: int = 300
    retry_attempts: int = 4
    retry_backoff_seconds: float = 2.0
    rate_limit_cooldown_seconds: int = 30


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["parquet", "csv"])
    register_mlflow: bool = True


@dataclass
class PathsConfig:
    data_dir: str = "./f1_dataset/data"
    logs_dir: str = "./f1_dataset/data/logs"
    checkpoints_dir: str = "./f1_dataset/data/checkpoints"
    artifacts_dir: str = "./f1_dataset/data/artifacts"


@dataclass
class MlflowConfig:
    tracking_uri: str | None = None
    experiment_name: str = "OpenF1Dataset"


@dataclass
class ApiConfig:
    base_url: str = "https://api.openf1.org/v1"


@dataclass
class Settings:
    seasons: list[int] = field(default_factory=list)
    session_name: str = "Race"
    drivers: DriverFilter = field(default_factory=DriverFilter)
    meetings: MeetingsFilter = field(default_factory=MeetingsFilter)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def _merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    data_dir = os.getenv("DATA_DIR")
    logs_dir = os.getenv("LOG_DIR")
    checkpoints_dir = os.getenv("CHECKPOINT_DIR")
    artifacts_dir = os.getenv("ARTIFACTS_DIR")
    register_mlflow = os.getenv("REGISTER_MLFLOW")
    mlflow_tracking = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT")
    api_base_url = os.getenv("OPENF1_BASE_URL")

    if data_dir:
        raw.setdefault("paths", {})["data_dir"] = data_dir
    if logs_dir:
        raw.setdefault("paths", {})["logs_dir"] = logs_dir
    if checkpoints_dir:
        raw.setdefault("paths", {})["checkpoints_dir"] = checkpoints_dir
    if artifacts_dir:
        raw.setdefault("paths", {})["artifacts_dir"] = artifacts_dir
    if register_mlflow is not None:
        raw.setdefault("output", {})["register_mlflow"] = register_mlflow.lower() == "true"
    if mlflow_tracking:
        raw.setdefault("mlflow", {})["tracking_uri"] = mlflow_tracking
    if mlflow_experiment:
        raw.setdefault("mlflow", {})["experiment_name"] = mlflow_experiment
    if api_base_url:
        raw.setdefault("api", {})["base_url"] = api_base_url

    return raw


def _coerce_settings(raw: dict[str, Any]) -> Settings:
    drivers = DriverFilter(**raw.get("drivers", {}))
    meetings = MeetingsFilter(**raw.get("meetings", {}))
    execution = ExecutionConfig(**raw.get("execution", {}))
    output = OutputConfig(**raw.get("output", {}))
    paths = PathsConfig(**raw.get("paths", {}))
    mlflow = MlflowConfig(**raw.get("mlflow", {}))
    api = ApiConfig(**raw.get("api", {}))

    return Settings(
        seasons=raw.get("seasons", []),
        session_name=raw.get("session_name", "Race"),
        drivers=drivers,
        meetings=meetings,
        execution=execution,
        output=output,
        paths=paths,
        mlflow=mlflow,
        api=api,
    )


def load_settings(config_path: str | None) -> Settings:
    raw: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _apply_env_overrides(raw)
    return _coerce_settings(raw)


def ensure_paths(settings: Settings) -> None:
    for path in [
        settings.paths.data_dir,
        settings.paths.logs_dir,
        settings.paths.checkpoints_dir,
        settings.paths.artifacts_dir,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
