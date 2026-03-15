from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import difflib

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from clients.mlflow_artifacts import artifact_uri, find_latest_run, get_tracking_client
from clients.mlflow_client import MlflowClient
from clients.mlflow_tags import with_run_context
from config.settings import load_settings
from modeling.dataset import load_consolidated
from orchestration.artifacts_cleanup import cleanup_paths, should_cleanup
from orchestration.data_lake_sync import (
    download_data_lake,
    should_cleanup_data_lake,
    sync_data_lake,
)
from orchestration.import_utils import ensure_data, has_data_for_filter, latest_file, run_cmd


class DriverProfilesRequest(BaseModel):
    season: int = Field(..., description="Temporada, ex: 2023")
    meeting_key: str = Field(..., description="meeting_key da corrida")
    session_name: str = Field("Race", description="Race ou Sprint")
    include_llm: bool = Field(True, description="Gera texto via LLM")
    llm_endpoint: Optional[str] = Field(
        None, description="Override do endpoint do MLflow Gateway"
    )
    drivers_include: list[str] = Field(
        default_factory=list, description="Lista de pilotos para incluir (nome completo)"
    )
    drivers_exclude: list[str] = Field(
        default_factory=list, description="Lista de pilotos para excluir (nome completo)"
    )


class DriverProfilesResponse(BaseModel):
    status: str
    artifacts: dict[str, str]
    message: Optional[str] = None


class SeasonProfilesRequest(BaseModel):
    seasons: list[int] = Field(..., description="Lista de temporadas, ex: [2023, 2024]")
    session_names: list[str] = Field(
        default_factory=list,
        description="Lista de session_name (ex: Race, Sprint). Vazio = todas.",
    )
    include_llm: bool = Field(True, description="Gera texto via LLM")
    llm_endpoint: Optional[str] = Field(
        None, description="Override do endpoint do MLflow Gateway"
    )
    drivers_include: list[str] = Field(
        default_factory=list, description="Lista de pilotos para incluir (nome completo)"
    )
    drivers_exclude: list[str] = Field(
        default_factory=list, description="Lista de pilotos para excluir (nome completo)"
    )


class SeasonProfilesResponse(BaseModel):
    status: str
    seasons: list[int]
    session_names: list[str]
    artifacts: dict[str, dict[str, str]]
    summaries: dict[str, dict[str, object]]
    top_drivers: dict[str, list[dict[str, object]]]
    message: Optional[str] = None


class ImportSeasonRequest(BaseModel):
    season: int = Field(..., description="Temporada, ex: 2023")
    session_name: str = Field("Race", description="Race ou Sprint")
    include_llm: bool = Field(True, description="Gera texto via LLM ao fim de cada corrida")
    llm_endpoint: Optional[str] = Field(
        None, description="Override do endpoint do MLflow Gateway"
    )
    resume_job_id: Optional[str] = Field(
        None,
        description="Job anterior para retomar (pula meetings ja concluidos).",
    )


class ImportSeasonJobResponse(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None


class ImportSeasonResumeRequest(BaseModel):
    resume_job_id: str = Field(..., description="Job anterior para retomar.")
    include_llm: Optional[bool] = Field(
        None, description="Override do include_llm do job anterior."
    )
    llm_endpoint: Optional[str] = Field(
        None, description="Override do endpoint do MLflow Gateway"
    )


class TrainStintDeltaPaceRequest(BaseModel):
    season: Optional[int] = Field(None, description="Temporada, ex: 2023")
    meeting_key: Optional[str] = Field(None, description="meeting_key da corrida")
    session_name: str = Field("all", description="Race, Sprint ou all")
    driver_number: Optional[int] = Field(None, description="Numero do piloto (opcional)")
    constructor: Optional[str] = Field(
        None, description="Nome da construtora (team_name) (opcional)"
    )
    target_mode: Literal["prev_stint_mean", "stint_start_mean"] = Field(
        "prev_stint_mean", description="Modo do alvo do delta de ritmo."
    )
    baseline_laps: int = Field(3, description="Voltas iniciais do stint para baseline.")
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    max_depth: Optional[int] = Field(None, description="Profundidade maxima.")
    min_samples_leaf: int = Field(1, description="Minimo de amostras por folha.")


class TrainStintDeltaPaceJobResponse(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None


class TrainJobResponse(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None


class TrainLapTimeRegressionRequest(BaseModel):
    include_sectors: bool = Field(
        False, description="Inclui features de setores (por padrao sao excluidas)."
    )
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    max_depth: Optional[int] = Field(None, description="Profundidade maxima.")
    min_samples_leaf: int = Field(1, description="Minimo de amostras por folha.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainLapTimeRankingRequest(BaseModel):
    include_sectors: bool = Field(
        False, description="Inclui features de setores (por padrao sao excluidas)."
    )
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    driver_col: str = Field("driver_name", description="Coluna de ranking de pilotos.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    max_depth: Optional[int] = Field(None, description="Profundidade maxima.")
    min_samples_leaf: int = Field(1, description="Minimo de amostras por folha.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainRelativePositionRequest(BaseModel):
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    max_depth: Optional[int] = Field(None, description="Profundidade maxima.")
    min_samples_leaf: int = Field(1, description="Minimo de amostras por folha.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainTyreDegradationRequest(BaseModel):
    include_sectors: bool = Field(
        False, description="Inclui features de setores (por padrao sao excluidas)."
    )
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    max_depth: Optional[int] = Field(None, description="Profundidade maxima.")
    min_samples_leaf: int = Field(1, description="Minimo de amostras por folha.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainLapQualityClassifierRequest(BaseModel):
    include_sectors: bool = Field(
        False, description="Inclui features de setores (por padrao sao excluidas)."
    )
    group_col: str = Field("meeting_key", description="Coluna de split por grupo.")
    test_size: float = Field(0.2, description="Percentual de teste.")
    random_state: int = Field(42, description="Seed do split.")
    n_estimators: int = Field(300, description="Numero de arvores do RandomForest.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainLapAnomalyRequest(BaseModel):
    contamination: float = Field(0.02, description="Percentual esperado de anomalias.")
    n_estimators: int = Field(300, description="Numero de arvores do IsolationForest.")
    random_state: int = Field(42, description="Seed do modelo.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainDriverStyleClusteringRequest(BaseModel):
    clusters: int = Field(4, description="Numero de clusters.")
    random_state: int = Field(42, description="Seed do modelo.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class TrainCircuitSegmentationRequest(BaseModel):
    clusters: int = Field(3, description="Numero de clusters.")
    random_state: int = Field(42, description="Seed do modelo.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Versao do modelo para MLflow (opcional). "
            "Se omitido, sera gerada a partir dos parametros."
        ),
    )


class DataLakeSyncRequest(BaseModel):
    direction: Literal["upload", "download"] = Field(
        "upload", description="upload para MinIO ou download para local."
    )
    subdirs: list[str] = Field(
        default_factory=list, description="Subdirs: bronze, silver, gold."
    )
    cleanup_local: Optional[bool] = Field(
        None, description="Override para limpar dados locais apos upload."
    )
    only_if_missing: bool = Field(
        True, description="No download, baixa apenas se a pasta local estiver vazia."
    )


class DataLakeSyncResponse(BaseModel):
    status: str
    direction: str
    files: dict[str, int]
    message: Optional[str] = None


class GoldQuestionsRequest(BaseModel):
    question: str = Field(..., description="Pergunta em linguagem natural.")
    season: int = Field(..., description="Temporada, ex: 2023")
    meeting_key: Optional[str] = Field(None, description="meeting_key da corrida (opcional)")
    session_name: str = Field("all", description="Race, Sprint ou all")
    driver_name: Optional[str] = Field(None, description="Nome completo do piloto (opcional)")
    driver_number: Optional[int] = Field(None, description="Numero do piloto (opcional)")


class GoldQuestionsResponse(BaseModel):
    status: str
    answer: str
    summary: dict[str, Any]


class GoldMeetingItem(BaseModel):
    season: Optional[int] = None
    meeting_key: str
    meeting_name: Optional[str] = None
    sessions: list[str]


class GoldMeetingsResponse(BaseModel):
    status: str
    rows: int
    seasons: list[int]
    sessions: list[str]
    meetings: list[GoldMeetingItem]


class GoldLapDriversResponse(BaseModel):
    status: str
    rows: int
    columns: list[str]
    data: list[dict[str, Any]]
    fastest_lap: Optional[dict[str, Any]] = None


class GoldLapMaxResponse(BaseModel):
    status: str
    season: int
    session_name: str
    meeting_key: Optional[str] = None
    meeting_name: Optional[str] = None
    max_lap_number: int


app = FastAPI(title="OpenF1 Dataset API", version="1.0.0")


def _merge_llm(ranking_csv: Path, llm_csv: Path, base_dir: Path) -> Path:
    rank_df = pd.read_csv(ranking_csv)
    llm_df = pd.read_csv(llm_csv)
    merged = rank_df.merge(
        llm_df[["driver_number", "driver_name", "team_name", "llm_text", "error"]],
        on=["driver_number", "driver_name", "team_name"],
        how="left",
    )
    out_dir = base_dir / "llm_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "driver_overall_ranking_llm.csv"
    merged.to_csv(out_csv, index=False)
    return out_csv


def _normalize_list(values: list[str]) -> list[str]:
    normalized = [str(v).strip() for v in values if str(v).strip()]
    return normalized


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_simple(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace("", pd.NA)
    return cleaned


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return resp.getcode(), resp.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return 0, str(exc)


def _web_fallback_answer(question: str) -> Optional[str]:
    provider = os.environ.get("WEB_FALLBACK_PROVIDER", "duckduckgo").strip().lower()
    if provider in {"", "disabled", "off", "false", "0"}:
        return None
    if provider != "duckduckgo":
        return None
    query = question.strip()
    if not query:
        return None
    params = urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
        }
    )
    url = f"https://api.duckduckgo.com/?{params}"
    req = Request(url, headers={"User-Agent": "OpenF1-DatasetBuilder/web-fallback"})
    try:
        with urlopen(req, timeout=8) as resp:
            if resp.getcode() >= 400:
                return None
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    for key in ["AbstractText", "Answer"]:
        value = (data or {}).get(key)
        if value:
            return str(value).strip()
    related = (data or {}).get("RelatedTopics") or []
    for item in related:
        text = item.get("Text") if isinstance(item, dict) else None
        if text:
            return str(text).strip()
    return None


def _numeric_stats(series: pd.Series) -> Optional[dict[str, float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def _duration_seconds(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "f"}:
        return pd.to_numeric(series, errors="coerce")
    parsed = pd.to_timedelta(series, errors="coerce")
    return parsed.dt.total_seconds()


def _format_hhmmss(seconds: float | int | None) -> Optional[str]:
    if seconds is None or pd.isna(seconds):
        return None
    total_ms = int(round(float(seconds) * 1000))
    if total_ms < 0:
        total_ms = 0
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{ms:03d}"


def _format_mmss(seconds: float | int | None) -> Optional[str]:
    if seconds is None or pd.isna(seconds):
        return None
    total_ms = int(round(float(seconds) * 1000))
    if total_ms < 0:
        total_ms = 0
    minutes = total_ms // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{secs:02d}:{ms:03d}"


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _value_counts(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    counts = cleaned.value_counts().head(limit)
    return [{"value": key, "count": int(value)} for key, value in counts.items()]


def _value_counts_with_share(series: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    counts = cleaned.value_counts().head(limit)
    total = int(counts.sum())
    rows = []
    for key, value in counts.items():
        count = int(value)
        share = float(count / total) if total else 0.0
        rows.append({"value": key, "count": count, "share": share})
    return rows


def _lap_duration_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_timedelta(series, errors="coerce")
    parsed_seconds = parsed.dt.total_seconds()
    return numeric.where(numeric.notna(), parsed_seconds)


def _row_context(row: pd.Series, lap_seconds: float | None = None) -> dict[str, Any]:
    payload = {
        "meeting_name": _jsonable_value(row.get("meeting_name")),
        "meeting_key": _jsonable_value(row.get("meeting_key")),
        "meeting_date_start": _jsonable_value(row.get("meeting_date_start")),
        "session_name": _jsonable_value(row.get("session_name")),
        "session_key": _jsonable_value(row.get("session_key")),
        "driver_name": _jsonable_value(row.get("driver_name")),
        "driver_number": _jsonable_value(row.get("driver_number")),
        "team_name": _jsonable_value(row.get("team_name")),
        "lap_number": _jsonable_value(row.get("lap_number")),
        "lap_duration": _jsonable_value(row.get("lap_duration")),
    }
    if lap_seconds is not None:
        payload["lap_duration_seconds"] = _jsonable_value(float(lap_seconds))
        payload["lap_duration_min"] = _format_mmss(lap_seconds)
    return payload


def _best_row_for_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    mode: Literal["min", "max"],
) -> Optional[dict[str, Any]]:
    if metric not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    idx = values[valid].idxmin() if mode == "min" else values[valid].idxmax()
    row = df.loc[idx]
    payload = _row_context(row)
    payload["metric"] = metric
    payload["metric_value"] = _jsonable_value(float(values.loc[idx]))
    return payload


def _date_range(series: pd.Series) -> Optional[dict[str, str]]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return None
    return {
        "min": parsed.min().isoformat(),
        "max": parsed.max().isoformat(),
    }


def _filter_gold_dataset(
    df: pd.DataFrame,
    *,
    season: int,
    meeting_key: Optional[str],
    meeting_name: Optional[str],
    session_name: str,
) -> pd.DataFrame:
    if "season" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna season nao encontrada no gold.")
    filtered = df[df["season"] == season]

    if meeting_key is not None:
        if "meeting_key" not in filtered.columns:
            raise HTTPException(status_code=500, detail="Coluna meeting_key nao encontrada no gold.")
        filtered = filtered[filtered["meeting_key"].astype(str) == str(meeting_key)]
    elif meeting_name is not None:
        if "meeting_name" not in filtered.columns:
            raise HTTPException(status_code=500, detail="Coluna meeting_name nao encontrada no gold.")
        meeting_norm = _normalize_simple(meeting_name)
        meeting_series = (
            filtered["meeting_name"]
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        filtered = filtered[meeting_series == meeting_norm]

    if session_name.lower() != "all":
        if "session_name" not in filtered.columns:
            raise HTTPException(status_code=500, detail="Coluna session_name nao encontrada no gold.")
        filtered = filtered[
            filtered["session_name"].astype(str).str.strip().str.lower() == session_name.lower()
        ]

    return filtered


def _data_dir(env: dict[str, str], config_path: str) -> Path:
    settings = load_settings(config_path)
    return Path(env.get("DATA_DIR", settings.paths.data_dir))


def _rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except Exception:
        return str(path)


def _parse_partition_info(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for part in path.parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key and value:
            info[key] = value
    return info


def _ms_to_iso(value: int | None) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.utcfromtimestamp(value / 1000.0).isoformat() + "Z"
    except Exception:
        return None


def _now_iso_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parquet_profile(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"pyarrow indisponivel: {exc}") from exc

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    columns = list(schema.names)
    schema_out = [{"name": name, "dtype": str(schema.field(name).type)} for name in columns]
    rows = metadata.num_rows if metadata else None

    nulls: dict[str, int] = {}
    if metadata:
        for idx, name in enumerate(columns):
            total_nulls: Optional[int] = 0
            for rg in range(metadata.num_row_groups):
                col = metadata.row_group(rg).column(idx)
                stats = col.statistics
                if stats is None or stats.null_count is None:
                    total_nulls = None
                    break
                total_nulls += int(stats.null_count)
            if total_nulls is None:
                nulls = {}
                break
            if total_nulls:
                nulls[name] = total_nulls

    return {
        "rows": rows,
        "columns": len(columns),
        "schema": schema_out,
        "nulls": nulls or None,
    }


def _get_minio_client(env: dict[str, str]) -> tuple[object, str, str, str]:
    endpoint = env.get("DATA_LAKE_S3_ENDPOINT") or env.get("MLFLOW_S3_ENDPOINT_URL")
    access_key = env.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Credenciais S3 ou endpoint do data lake nao configurados.")
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"boto3 indisponivel: {exc}") from exc

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    bucket = env.get("DATA_LAKE_BUCKET", "openf1-datalake")
    prefix = env.get("DATA_LAKE_PREFIX", "openf1").strip("/")
    return client, bucket, prefix, endpoint


def _list_minio_objects(
    client: object,
    bucket: str,
    prefix: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key:
                continue
            items.append(
                {
                    "key": key,
                    "size": int(obj.get("Size", 0)),
                    "last_modified": obj.get("LastModified").isoformat()
                    if obj.get("LastModified")
                    else None,
                }
            )
            if limit and len(items) >= limit:
                return items
    return items

def _extract_driver_mentions(question: str, driver_names: list[str]) -> list[str]:
    if not question or not driver_names:
        return []
    normalized_question = _normalize_simple(question)
    tokens = re.findall(r"[a-z]+", normalized_question)
    matches: list[str] = []
    for name in driver_names:
        normalized_name = _normalize_simple(name)
        if normalized_name in normalized_question:
            matches.append(name)
            continue
        last = normalized_name.split()[-1]
        if re.search(rf"\b{re.escape(last)}\b", normalized_question):
            matches.append(name)
            continue
        for token in tokens:
            if difflib.SequenceMatcher(None, last, token).ratio() >= 0.86:
                matches.append(name)
                break
    seen: set[str] = set()
    unique = []
    for name in matches:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def _driver_aggregates(df: pd.DataFrame, drivers: list[str]) -> list[dict[str, Any]]:
    if "driver_name" not in df.columns or not drivers:
        return []
    data = df.copy()
    data = data[data["driver_name"].astype(str).isin(drivers)]
    if data.empty:
        return []
    if "lap_duration" in data.columns:
        data = data.copy()
        data["lap_duration_seconds"] = _duration_seconds(data["lap_duration"])

    group_cols = ["driver_name"]
    if "driver_number" in data.columns:
        group_cols.append("driver_number")

    agg_spec: dict[str, list[str]] = {}
    candidate_cols = {
        "lap_duration_seconds": ["mean", "median", "std"],
        "lap_number": ["max"],
        "avg_speed": ["mean"],
        "max_speed": ["mean"],
        "min_speed": ["mean"],
        "speed_std": ["mean"],
        "avg_rpm": ["mean"],
        "avg_throttle": ["mean"],
        "full_throttle_pct": ["mean"],
        "brake_pct": ["mean"],
        "drs_pct": ["mean"],
        "gear_changes": ["mean"],
        "distance_traveled": ["mean"],
        "trajectory_variation": ["mean"],
        "track_temperature": ["mean"],
        "air_temperature": ["mean"],
    }
    for col, funcs in candidate_cols.items():
        if col in data.columns:
            agg_spec[col] = funcs
    if not agg_spec:
        return []

    grouped = data.groupby(group_cols, dropna=False)
    stats = grouped.agg(agg_spec)
    stats.columns = ["_".join(col).strip() for col in stats.columns.to_flat_index()]
    stats["laps_total"] = grouped.size()
    stats = stats.reset_index()

    rows = []
    for _, row in stats.iterrows():
        payload = {}
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                payload[key] = None
            elif isinstance(value, (int, float)) and pd.notna(value):
                payload[key] = float(value) if isinstance(value, float) else int(value)
            else:
                payload[key] = value
        rows.append(payload)
    return rows


def _build_gold_summary(
    df: pd.DataFrame,
    payload: GoldQuestionsRequest,
    question: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "filters": {
            "season": payload.season,
            "meeting_key": payload.meeting_key,
            "session_name": payload.session_name,
            "driver_name": payload.driver_name,
            "driver_number": payload.driver_number,
        },
        "rows": int(len(df)),
    }
    summary["columns_available"] = sorted([str(c) for c in df.columns])

    if "driver_number" in df.columns:
        summary["drivers_total"] = int(df["driver_number"].nunique())
    elif "driver_name" in df.columns:
        summary["drivers_total"] = int(df["driver_name"].nunique())
    if "team_name" in df.columns:
        summary["teams_total"] = int(df["team_name"].nunique())
    if "meeting_key" in df.columns:
        summary["meetings_total"] = int(df["meeting_key"].nunique())
    if "session_name" in df.columns:
        summary["sessions"] = sorted(df["session_name"].dropna().astype(str).str.strip().unique())
    if "meeting_date_start" in df.columns:
        summary["meeting_date_start"] = _date_range(df["meeting_date_start"])

    numeric_summary: dict[str, Any] = {}
    lap_seconds: pd.Series | None = None
    if "lap_duration" in df.columns:
        lap_seconds = _lap_duration_seconds(df["lap_duration"])
        numeric_summary["lap_duration_seconds"] = _numeric_stats(lap_seconds)
        if lap_seconds.notna().any():
            summary["lap_duration_quantiles"] = {
                "p01": float(lap_seconds.quantile(0.01)),
                "p05": float(lap_seconds.quantile(0.05)),
                "p10": float(lap_seconds.quantile(0.10)),
                "p25": float(lap_seconds.quantile(0.25)),
                "p50": float(lap_seconds.quantile(0.50)),
                "p75": float(lap_seconds.quantile(0.75)),
                "p90": float(lap_seconds.quantile(0.90)),
                "p95": float(lap_seconds.quantile(0.95)),
                "p99": float(lap_seconds.quantile(0.99)),
            }
    for col in [
        "lap_number",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        "avg_speed",
        "max_speed",
        "min_speed",
        "speed_std",
        "avg_rpm",
        "max_rpm",
        "min_rpm",
        "rpm_std",
        "avg_throttle",
        "max_throttle",
        "min_throttle",
        "throttle_std",
        "full_throttle_pct",
        "brake_pct",
        "brake_events",
        "hard_brake_events",
        "drs_pct",
        "gear_changes",
        "distance_traveled",
        "trajectory_length",
        "trajectory_variation",
        "telemetry_points",
        "trajectory_points",
        "track_temperature",
        "air_temperature",
        "stint_number",
        "tyre_age_at_start",
        "tyre_age_at_lap",
    ]:
        if col in df.columns:
            stats = _numeric_stats(df[col])
            if stats:
                numeric_summary[col] = stats
    if numeric_summary:
        summary["numeric"] = numeric_summary

    if lap_seconds is not None:
        valid = lap_seconds.notna()
        if valid.any():
            fastest_idx = lap_seconds[valid].idxmin()
            fastest_row = df.loc[fastest_idx]
            summary["fastest_lap"] = _row_context(
                fastest_row, lap_seconds=float(lap_seconds.loc[fastest_idx])
            )
            slowest_idx = lap_seconds[valid].idxmax()
            slowest_row = df.loc[slowest_idx]
            summary["slowest_lap"] = _row_context(
                slowest_row, lap_seconds=float(lap_seconds.loc[slowest_idx])
            )

            ranked = (
                df.assign(_lap_seconds=lap_seconds)
                .loc[valid]
                .sort_values("_lap_seconds", ascending=True)
                .head(5)
            )
            summary["fastest_laps_top"] = [
                _row_context(row, lap_seconds=row.get("_lap_seconds"))
                for _, row in ranked.iterrows()
            ]
            slowest_ranked = (
                df.assign(_lap_seconds=lap_seconds)
                .loc[valid]
                .sort_values("_lap_seconds", ascending=False)
                .head(5)
            )
            summary["slowest_laps_top"] = [
                _row_context(row, lap_seconds=row.get("_lap_seconds"))
                for _, row in slowest_ranked.iterrows()
            ]

            if "meeting_key" in df.columns:
                group_cols = ["meeting_key"]
                if "meeting_name" in df.columns:
                    group_cols.append("meeting_name")
                per_meeting = (
                    df.assign(_lap_seconds=lap_seconds)
                    .loc[valid]
                    .groupby(group_cols, dropna=False)["_lap_seconds"]
                    .idxmin()
                )
                meeting_rows = df.loc[per_meeting].assign(
                    _lap_seconds=lap_seconds.loc[per_meeting]
                )
                summary["fastest_lap_by_meeting"] = [
                    _row_context(row, lap_seconds=row.get("_lap_seconds"))
                    for _, row in meeting_rows.iterrows()
                ]

            if "driver_name" in df.columns:
                group_cols = ["driver_name"]
                if "driver_number" in df.columns:
                    group_cols.append("driver_number")
                per_driver = (
                    df.assign(_lap_seconds=lap_seconds)
                    .loc[valid]
                    .groupby(group_cols, dropna=False)["_lap_seconds"]
                    .idxmin()
                )
                driver_rows = df.loc[per_driver].assign(
                    _lap_seconds=lap_seconds.loc[per_driver]
                )
                summary["fastest_lap_by_driver"] = [
                    _row_context(row, lap_seconds=row.get("_lap_seconds"))
                    for _, row in driver_rows.iterrows()
                ]

    records: dict[str, Any] = {}
    for metric in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        best = _best_row_for_metric(df, metric, mode="min")
        if best:
            records[f"best_{metric}"] = best
    for metric in ["st_speed", "i1_speed", "i2_speed", "avg_speed", "max_speed"]:
        best = _best_row_for_metric(df, metric, mode="max")
        if best:
            records[f"max_{metric}"] = best
    for metric in ["drs_pct", "full_throttle_pct", "brake_events", "hard_brake_events"]:
        best = _best_row_for_metric(df, metric, mode="max")
        if best:
            records[f"max_{metric}"] = best
    if records:
        summary["records"] = records

    if "meeting_key" in df.columns:
        counts = df.groupby("meeting_key", dropna=False).size()
        meeting_map = None
        if "meeting_name" in df.columns:
            meeting_map = (
                df[["meeting_key", "meeting_name"]]
                .drop_duplicates()
                .set_index("meeting_key")["meeting_name"]
            )
        meetings_top = []
        for key, count in counts.sort_values(ascending=False).head(10).items():
            meetings_top.append(
                {
                    "meeting_key": _jsonable_value(key),
                    "meeting_name": _jsonable_value(
                        meeting_map.get(key) if meeting_map is not None else None
                    ),
                    "laps_total": int(count),
                }
            )
        summary["laps_by_meeting_top"] = meetings_top

    if "driver_name" in df.columns:
        summary["laps_by_driver_top"] = _value_counts_with_share(df["driver_name"], limit=10)
    if "team_name" in df.columns:
        summary["laps_by_team_top"] = _value_counts_with_share(df["team_name"], limit=10)
    if "compound" in df.columns:
        summary["compound_usage"] = _value_counts_with_share(df["compound"], limit=8)

    if "is_pit_out_lap" in df.columns:
        pit = pd.to_numeric(df["is_pit_out_lap"], errors="coerce")
        if pit.notna().any():
            summary["pit_out_lap_rate"] = float(pit.mean())

    if "has_telemetry" in df.columns:
        tele = pd.to_numeric(df["has_telemetry"], errors="coerce")
        if tele.notna().any():
            summary["telemetry_coverage"] = {
                "rate": float(tele.mean()),
                "count": int(tele.sum()),
            }
    if "has_trajectory" in df.columns:
        traj = pd.to_numeric(df["has_trajectory"], errors="coerce")
        if traj.notna().any():
            summary["trajectory_coverage"] = {
                "rate": float(traj.mean()),
                "count": int(traj.sum()),
            }

    top_values: dict[str, Any] = {}
    for col in [
        "meeting_name",
        "meeting_key",
        "session_name",
        "driver_name",
        "driver_number",
        "team_name",
        "circuit_speed_class",
        "compound",
    ]:
        if col in df.columns:
            values = _value_counts(df[col])
            if values:
                top_values[col] = values
    if top_values:
        summary["top_values"] = top_values

        value_mappings = {
            "circuit_speed_class": {
                "low": "baixa velocidade",
                "medium": "media velocidade",
                "high": "alta velocidade",
            },
            "compound": {
                "SOFT": "macio",
                "MEDIUM": "medio",
                "HARD": "duro",
                "INTERMEDIATE": "intermediario",
                "WET": "chuva",
            },
        }

        def _map_value(col: str, value: str) -> Optional[str]:
            mapping = value_mappings.get(col, {})
            if col == "compound":
                key = str(value).upper().strip()
            else:
                key = str(value).lower().strip()
            return mapping.get(key)

        top_values_pt: dict[str, Any] = {}
        for col, values in top_values.items():
            if col not in value_mappings:
                continue
            mapped = []
            for item in values:
                mapped_item = dict(item)
                mapped_item["value_pt"] = _map_value(col, item.get("value", ""))
                mapped.append(mapped_item)
            top_values_pt[col] = mapped
        if top_values_pt:
            summary["top_values_pt"] = top_values_pt

        for col in ["circuit_speed_class", "compound"]:
            if col in top_values and top_values[col]:
                raw_value = top_values[col][0].get("value")
                label = _map_value(col, raw_value) if raw_value is not None else None
                summary[f"{col}_most_common"] = raw_value
                summary[f"{col}_most_common_pt"] = label

    if "driver_name" in df.columns:
        driver_names = (
            df["driver_name"].dropna().astype(str).str.strip().unique().tolist()
        )
    else:
        driver_names = []

    drivers_focus = _extract_driver_mentions(question, driver_names)
    if not drivers_focus and (payload.driver_name or payload.driver_number):
        drivers_focus = driver_names
    if not drivers_focus and driver_names:
        top_drivers = (
            df["driver_name"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .index.tolist()
        )
        drivers_focus = top_drivers
        summary["drivers_focus_reason"] = "top_by_laps"
    if drivers_focus:
        summary["drivers_focus"] = drivers_focus
        aggregates = _driver_aggregates(df, drivers_focus)
        if aggregates:
            summary["drivers_aggregates"] = aggregates

    return summary


def _build_gold_prompt(
    question: str,
    summary: dict[str, Any],
    strict_portuguese: bool = False,
) -> dict[str, Any]:
    glossary = {
        "season": "temporada (ano)",
        "meeting_key": "chave da corrida (meeting)",
        "meeting_name": "nome da corrida",
        "meeting_date_start": "data de inicio da corrida",
        "meeting_day": "dia do mes da corrida",
        "meeting_month": "mes da corrida",
        "session_key": "chave da sessao",
        "session_name": "tipo de sessao (Race ou Sprint)",
        "driver_number": "numero do piloto",
        "driver_name": "nome do piloto",
        "team_name": "equipe",
        "lap_number": "numero da volta",
        "lap_duration": "tempo de volta (formato original)",
        "lap_duration_seconds": "tempo de volta em segundos",
        "lap_duration_min": "tempo de volta em mm:ss:ms",
        "fastest_lap": "volta mais rapida no recorte (objeto com piloto, corrida e tempo)",
        "fastest_laps_top": "lista das voltas mais rapidas no recorte",
        "slowest_lap": "volta mais lenta no recorte",
        "slowest_laps_top": "lista das voltas mais lentas no recorte",
        "fastest_lap_by_meeting": "volta mais rapida por corrida (lista)",
        "fastest_lap_by_driver": "melhor volta de cada piloto (lista)",
        "records": "recordes por metrica (ex.: max_speed, best_sector_1)",
        "laps_by_meeting_top": "corridas com mais voltas registradas",
        "laps_by_driver_top": "pilotos com mais voltas registradas",
        "laps_by_team_top": "equipes com mais voltas registradas",
        "compound_usage": "uso de compostos (frequencia)",
        "pit_out_lap_rate": "taxa de voltas de saida dos boxes",
        "telemetry_coverage": "cobertura de telemetria",
        "trajectory_coverage": "cobertura de trajetoria",
        "duration_sector_1": "tempo no setor 1",
        "duration_sector_2": "tempo no setor 2",
        "duration_sector_3": "tempo no setor 3",
        "i1_speed": "velocidade no ponto i1",
        "i2_speed": "velocidade no ponto i2",
        "st_speed": "velocidade no speed trap",
        "is_pit_out_lap": "volta de saida dos boxes",
        "weather_date": "data/hora da medicao do clima",
        "track_temperature": "temperatura da pista (C)",
        "air_temperature": "temperatura do ar (C)",
        "circuit_speed_class": "classe de velocidade do circuito",
        "avg_speed": "velocidade media",
        "max_speed": "velocidade maxima",
        "min_speed": "velocidade minima",
        "speed_std": "variacao da velocidade",
        "avg_rpm": "rpm medio",
        "max_rpm": "rpm maximo",
        "min_rpm": "rpm minimo",
        "rpm_std": "variacao do rpm",
        "avg_throttle": "acelerador medio",
        "max_throttle": "acelerador maximo",
        "min_throttle": "acelerador minimo",
        "throttle_std": "variacao do acelerador",
        "full_throttle_pct": "percentual de acelerador total",
        "brake_pct": "percentual de uso do freio",
        "brake_events": "eventos de freada",
        "hard_brake_events": "eventos de freada forte",
        "drs_pct": "percentual de uso do DRS",
        "gear_changes": "trocas de marcha",
        "distance_traveled": "distancia percorrida",
        "trajectory_length": "comprimento da trajetoria",
        "trajectory_variation": "variacao da trajetoria",
        "telemetry_points": "pontos de telemetria",
        "trajectory_points": "pontos de trajetoria",
        "has_telemetry": "tem telemetria",
        "has_trajectory": "tem trajetoria",
        "stint_number": "numero do stint",
        "compound": "tipo de pneu",
        "stint_lap_start": "volta inicial do stint",
        "stint_lap_end": "volta final do stint",
        "tyre_age_at_start": "idade do pneu no inicio do stint (voltas)",
        "tyre_age_at_lap": "idade do pneu na volta (voltas)",
        "laps_total": "total de voltas",
        "suffixes": {
            "_mean": "media",
            "_median": "mediana",
            "_std": "desvio_padrao",
            "_max": "maximo",
            "_min": "minimo",
        },
        "value_mappings": {
            "circuit_speed_class": {
                "low": "baixa velocidade",
                "medium": "media velocidade",
                "high": "alta velocidade",
            },
            "compound": {
                "SOFT": "macio",
                "MEDIUM": "medio",
                "HARD": "duro",
                "INTERMEDIATE": "intermediario",
                "WET": "chuva",
            },
        },
    }
    system = (
        "Voce responde perguntas sobre F1 usando APENAS os dados do JSON fornecido. "
        "Nao use conhecimento externo. Responda sempre em portugues simples, "
        "com frases curtas e explique termos tecnicos em linguagem comum. "
        "Use o glossario para traduzir nomes de colunas e valores para portugues. "
        "Se a pergunta mencionar pneus/compostos, use a coluna 'compound'. "
        "Se mencionar velocidade do circuito, use 'circuit_speed_class' e traduza "
        "low/medium/high para baixa/media/alta velocidade. "
        "Se a pergunta mencionar volta mais rapida, use 'fastest_lap' ou 'fastest_laps_top'. "
        "Se mencionar volta mais lenta, use 'slowest_lap' ou 'slowest_laps_top'. "
        "Se mencionar recordes (velocidade, setor, DRS), use 'records'. "
        "Se existirem campos '*_pt' no JSON, use esses valores em portugues. "
        "Se a resposta nao puder ser obtida somente desses dados, "
        "responda exatamente: 'Sem dados no gold.'"
    )
    if strict_portuguese:
        system += (
            " Idioma obrigatorio: portugues do Brasil. "
            "Nao use palavras em ingles (exceto nomes proprios). "
            "Se sua resposta nao estiver em portugues, refaca a resposta."
        )
    user = (
        f"Pergunta: {question}\n"
        f"Glossario_pt: {json.dumps(glossary, ensure_ascii=False)}\n"
        f"dados_gold: {json.dumps(summary, ensure_ascii=False)}"
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }


def _parse_llm_answer(raw: str) -> str:
    data = json.loads(raw)
    return data["choices"][0]["message"]["content"].strip()


def _extract_json_payload(text: str) -> Optional[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\\w*\\s*|```$", "", cleaned, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"{.*}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _duckdb_enabled(env: dict[str, str]) -> bool:
    return env.get("GOLD_QUESTIONS_DUCKDB", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(env: dict[str, str], key: str, default: int, min_value: int = 1) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    return max(value, min_value)


def _duckdb_top_limits(env: dict[str, str]) -> tuple[int, int]:
    base = _env_int(env, "GOLD_QUESTIONS_TOP", 20)
    per_sector = _env_int(env, "GOLD_QUESTIONS_SECTOR_TOP", base)
    overall = _env_int(env, "GOLD_QUESTIONS_OVERALL_TOP", base)
    return per_sector, overall


def _duckdb_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith("int") or dtype.startswith("Int"):
            dtype = "integer"
        elif dtype.startswith("float"):
            dtype = "float"
        elif "datetime" in dtype or "date" in dtype:
            dtype = "timestamp"
        elif dtype == "bool":
            dtype = "boolean"
        else:
            dtype = "string"
        schema.append({"name": str(col), "type": dtype})
    return schema


def _build_duckdb_sql_prompt(
    question: str,
    filters: dict[str, Any],
    schema: list[dict[str, str]],
    *,
    wants_frequency: bool,
) -> dict[str, Any]:
    system = (
        "Voce gera SQL DuckDB para responder perguntas sobre F1. "
        "Use APENAS SELECT (pode usar WITH/CTE). "
        "Tabela disponivel: gold (ja filtrada pelos filtros informados). "
        "Nao use INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/PRAGMA/ATTACH/COPY. "
        "Evite subconsultas desnecessarias. "
        "Se nao for possivel responder com SQL, retorne use_sql=false."
    )
    frequency_hint = ""
    if wants_frequency:
        frequency_hint = (
            "- A pergunta pede frequencia/quantidade: inclua COUNT(*) com alias `count` "
            "e agrupe por piloto/equipe/corrida conforme fizer sentido.\n"
        )
    user = (
        "Retorne um JSON valido com as chaves: use_sql (bool) e sql (string). "
        "Se use_sql=false, inclua reason.\n"
        f"Pergunta: {question}\n"
        f"Filtros_aplicados: {json.dumps(filters, ensure_ascii=False)}\n"
        "Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "Dicas:\n"
        "- Use try_cast para converter lap_duration quando precisar comparar tempos.\n"
        "- Use LIMIT 50 no maximo.\n"
        "- Use apenas colunas existentes no schema.\n"
        f"{frequency_hint}"
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }


def _build_duckdb_answer_prompt(
    question: str,
    sql: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    system = (
        "Voce responde perguntas em portugues usando SOMENTE o resultado de uma consulta SQL. "
        "Se o resultado estiver vazio, responda exatamente: 'Sem dados no gold.' "
        "Se houver linhas, responda de forma objetiva e cite valores e chaves principais."
    )
    user = (
        f"Pergunta: {question}\n"
        f"SQL: {sql}\n"
        f"Resultado: {json.dumps(rows, ensure_ascii=False)}"
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }


def _sanitize_duckdb_sql(sql: str, limit: int = 50) -> str:
    cleaned = sql.strip().strip(";")
    cleaned = re.sub(r"^```\\w*\\s*|```$", "", cleaned, flags=re.DOTALL).strip()
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("SQL deve comecar com SELECT ou WITH.")
    forbidden = [
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "pragma",
        "attach",
        "copy",
        "export",
        "import",
        "call",
    ]
    for token in forbidden:
        if re.search(rf"\\b{token}\\b", lowered):
            raise ValueError("SQL contem comando nao permitido.")
    if not re.search(r"\\bfrom\\s+gold\\b", lowered):
        raise ValueError("SQL deve consultar apenas a tabela gold.")
    if re.search(r"\\blimit\\b", lowered) is None:
        cleaned = f"{cleaned} LIMIT {limit}"
    return cleaned


def _rows_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        payload = {}
        for key, value in row.items():
            payload[str(key)] = _jsonable_value(value)
        rows.append(payload)
    return rows


def _format_meetings_list(value: Any, limit: int = 6) -> Optional[str]:
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = value.tolist()
    except Exception:
        pass
    meetings: list[str]
    if isinstance(value, (list, tuple, set)):
        meetings = [str(v).strip() for v in value if str(v).strip()]
    else:
        text = str(value).strip()
        if not text:
            return None
        meetings = [text]
    if not meetings:
        return None
    # keep deterministic output while avoiding duplicates
    meetings = sorted(dict.fromkeys(meetings))
    if len(meetings) <= limit:
        return ", ".join(meetings)
    extra = len(meetings) - limit
    return ", ".join(meetings[:limit]) + f" (+{extra})"


def _format_duckdb_rows_answer(question: str, rows: list[dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    keys = {key for row in rows for key in row.keys()}
    if {"sector", "count"}.issubset(keys):
        sector_labels = {
            "sector_1": "Setor 1",
            "sector_2": "Setor 2",
            "sector_3": "Setor 3",
            "all": "Total",
        }
        by_sector: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            sector = str(row.get("sector"))
            by_sector.setdefault(sector, []).append(row)
        lines = []
        for sector in ["sector_1", "sector_2", "sector_3"]:
            if sector not in by_sector:
                continue
            label = sector_labels.get(sector, sector)
            total = by_sector[sector][0].get("total_opportunities")
            header = f"{label}:"
            if total is not None:
                header += f" (total {int(total)})"
            lines.append(header)
            for row in by_sector[sector]:
                driver = row.get("driver_name") or row.get("driver_number") or "--"
                team = row.get("team_name")
                count = row.get("count")
                pct = row.get("pct")
                detail = f"- {driver}"
                if team:
                    detail += f" ({team})"
                if count is not None:
                    detail += f" — {count} vezes"
                if isinstance(pct, (int, float)):
                    detail += f" ({pct * 100:.1f}%)"
                meetings = _format_meetings_list(row.get("meetings"))
                if meetings:
                    detail += f" | pistas: {meetings}"
                lines.append(detail)
        if "all" in by_sector:
            total = by_sector["all"][0].get("total_opportunities")
            header = "Total de setores ganhos (soma dos 3 setores):"
            if total is not None:
                header += f" (total {int(total)})"
            lines.append(header)
            for row in by_sector["all"]:
                driver = row.get("driver_name") or row.get("driver_number") or "--"
                team = row.get("team_name")
                count = row.get("count")
                pct = row.get("pct")
                detail = f"- {driver}"
                if team:
                    detail += f" ({team})"
                if count is not None:
                    detail += f" — {count} vezes"
                if isinstance(pct, (int, float)):
                    detail += f" ({pct * 100:.1f}%)"
                meetings = _format_meetings_list(row.get("meetings"))
                if meetings:
                    detail += f" | pistas: {meetings}"
                lines.append(detail)
        if lines:
            intro = "Ranking de melhores tempos por setor:"
            return f"{intro}\n" + "\n".join(lines)
    return None


def _question_wants_frequency(question: str) -> bool:
    tokens = _normalize_simple(question)
    keywords = [
        "quantos",
        "quantas",
        "vezes",
        "frequencia",
        "frequência",
        "quantidade",
        "numero de",
        "número de",
        "contagem",
    ]
    return any(key in tokens for key in keywords)


def _question_mentions_sector(question: str) -> bool:
    tokens = _normalize_simple(question)
    return "setor" in tokens or "sector" in tokens


def _question_mentions_best(question: str) -> bool:
    tokens = _normalize_simple(question)
    return any(
        key in tokens
        for key in [
            "melhor",
            "melhores",
            "mais rapido",
            "mais rápida",
            "mais rapida",
            "mais rápidas",
            "mais rapidas",
        ]
    )


def _sector_best_frequency_sql(limit_per_sector: int = 20, limit_overall: int = 20) -> str:
    return """
    WITH base AS (
      SELECT meeting_key, meeting_name, driver_name, team_name, duration_sector_1 AS sector_time, 'sector_1' AS sector
      FROM gold
      WHERE duration_sector_1 IS NOT NULL
      UNION ALL
      SELECT meeting_key, meeting_name, driver_name, team_name, duration_sector_2 AS sector_time, 'sector_2' AS sector
      FROM gold
      WHERE duration_sector_2 IS NOT NULL
      UNION ALL
      SELECT meeting_key, meeting_name, driver_name, team_name, duration_sector_3 AS sector_time, 'sector_3' AS sector
      FROM gold
      WHERE duration_sector_3 IS NOT NULL
    ),
    best_per_meeting AS (
      SELECT sector, meeting_key, MIN(sector_time) AS best_time
      FROM base
      GROUP BY sector, meeting_key
    ),
    best_rows AS (
      SELECT b.sector, b.meeting_key, b.meeting_name, b.driver_name, b.team_name, b.sector_time
      FROM base b
      JOIN best_per_meeting m
        ON b.sector = m.sector
       AND b.meeting_key = m.meeting_key
       AND b.sector_time = m.best_time
    ),
    meeting_lists AS (
      SELECT sector, driver_name, team_name,
             list(distinct meeting_name) AS meetings
      FROM best_rows
      GROUP BY sector, driver_name, team_name
    ),
    sector_totals AS (
      SELECT sector, COUNT(*) AS total_opportunities
      FROM best_rows
      GROUP BY sector
    ),
    counts AS (
      SELECT sector, driver_name, team_name, COUNT(*) AS count
      FROM best_rows
      GROUP BY sector, driver_name, team_name
    ),
    ranked_sector AS (
      SELECT c.*, s.total_opportunities,
             (c.count * 1.0 / s.total_opportunities) AS pct,
             ml.meetings,
             ROW_NUMBER() OVER (PARTITION BY c.sector ORDER BY c.count DESC, c.driver_name) AS rn
      FROM counts c
      JOIN sector_totals s
        ON c.sector = s.sector
      LEFT JOIN meeting_lists ml
        ON c.sector = ml.sector
       AND c.driver_name = ml.driver_name
       AND c.team_name = ml.team_name
    ),
    overall_total AS (
      SELECT COUNT(*) AS total_opportunities
      FROM best_rows
    ),
    overall_meetings AS (
      SELECT driver_name, team_name,
             list(distinct meeting_name) AS meetings
      FROM best_rows
      GROUP BY driver_name, team_name
    ),
    overall_counts AS (
      SELECT 'all' AS sector, driver_name, team_name,
             COUNT(*) AS count,
             (SELECT total_opportunities FROM overall_total) AS total_opportunities
      FROM best_rows
      GROUP BY driver_name, team_name
    ),
    ranked_overall AS (
      SELECT oc.*, om.meetings,
             (count * 1.0 / total_opportunities) AS pct,
             ROW_NUMBER() OVER (
               PARTITION BY oc.sector
               ORDER BY oc.count DESC, oc.driver_name
             ) AS rn
      FROM overall_counts oc
      LEFT JOIN overall_meetings om
        ON oc.driver_name = om.driver_name
       AND oc.team_name = om.team_name
    )
    SELECT sector, driver_name, team_name, count, total_opportunities, pct, meetings
    FROM ranked_sector
    WHERE rn <= {limit_per_sector}
    UNION ALL
    SELECT sector, driver_name, team_name, count, total_opportunities, pct, meetings
    FROM ranked_overall
    WHERE rn <= {limit_overall}
    ORDER BY sector, count DESC, driver_name
    """.format(limit_per_sector=limit_per_sector, limit_overall=limit_overall)


def _try_duckdb_answer(
    *,
    question: str,
    payload: GoldQuestionsRequest,
    filtered: pd.DataFrame,
    summary: dict[str, Any],
    env: dict[str, str],
) -> Optional[str]:
    if not _duckdb_enabled(env):
        return None
    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    wants_frequency = _question_wants_frequency(question)
    use_sector_best = _question_mentions_sector(question) and _question_mentions_best(question)

    filters = {
        "season": payload.season,
        "meeting_key": payload.meeting_key,
        "session_name": payload.session_name,
        "driver_name": payload.driver_name,
        "driver_number": payload.driver_number,
    }
    schema = _duckdb_schema(filtered)
    llm_endpoint = env.get(
        "MLFLOW_GATEWAY_ENDPOINT",
        "http://mlflow:5000/gateway/gemini/mlflow/invocations",
    )
    if use_sector_best and wants_frequency:
        limit_per_sector, limit_overall = _duckdb_top_limits(env)
        sql = _sector_best_frequency_sql(
            limit_per_sector=limit_per_sector,
            limit_overall=limit_overall,
        )
    else:
        llm_payload = _build_duckdb_sql_prompt(
            question, filters, schema, wants_frequency=wants_frequency
        )
        status, content = _post_json(llm_endpoint, llm_payload, timeout_s=60)
        if status != 200:
            return None
        sql_payload = _extract_json_payload(_parse_llm_answer(content))
        if not sql_payload or not sql_payload.get("use_sql"):
            return None
        sql = sql_payload.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return None
        try:
            sql = _sanitize_duckdb_sql(sql)
        except Exception:
            return None

    con = None
    try:
        con = duckdb.connect()
        con.register("gold", filtered)
        result_df = con.execute(sql).df()
    except Exception:
        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    if result_df.empty:
        return None

    rows = _rows_from_df(result_df)
    summary["duckdb_query"] = sql
    summary["duckdb_rows"] = rows

    if use_sector_best and wants_frequency:
        return _format_duckdb_rows_answer(question, rows)

    answer_payload = _build_duckdb_answer_prompt(question, sql, rows)
    status, content = _post_json(llm_endpoint, answer_payload, timeout_s=60)
    if status != 200:
        return None
    try:
        answer = _parse_llm_answer(content)
    except Exception:
        return None
    answer = _apply_pt_br_replacements(answer)
    if answer.strip().lower() == "sem dados no gold.":
        formatted = _format_duckdb_rows_answer(question, rows)
        return formatted
    if not _is_probably_portuguese(answer) or _contains_english_markers(answer):
        formatted = _format_duckdb_rows_answer(question, rows)
        if formatted:
            return formatted
        return None
    return answer


def _is_probably_portuguese(text: str) -> bool:
    if not text:
        return True
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return True
    en = {
        "the",
        "and",
        "is",
        "are",
        "this",
        "that",
        "these",
        "those",
        "with",
        "for",
        "from",
        "of",
        "to",
        "in",
        "on",
        "here",
        "summary",
        "average",
        "driver",
        "drivers",
        "lap",
        "speed",
        "data",
        "provides",
        "overview",
        "total",
        "totals",
        "key",
        "metrics",
    }
    pt = {
        "o",
        "a",
        "os",
        "as",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "e",
        "em",
        "para",
        "com",
        "sem",
        "por",
        "como",
        "que",
        "na",
        "no",
        "nas",
        "nos",
        "uma",
        "um",
        "resumo",
        "media",
        "piloto",
        "volta",
        "velocidade",
        "dados",
        "total",
        "totais",
        "metricas",
        "temperatura",
        "pista",
    }
    en_count = sum(1 for t in tokens if t in en)
    pt_count = sum(1 for t in tokens if t in pt)
    total = en_count + pt_count
    if total == 0:
        return True
    if en_count >= 3 and en_count / total > 0.6:
        return False
    return True


def _apply_pt_br_replacements(text: str) -> str:
    if not text:
        return text
    replacements = {
        r"\bHARD\b": "duro",
        r"\bMEDIUM\b": "médio",
        r"\bSOFT\b": "macio",
        r"\bINTERMEDIATE\b": "intermediário",
        r"\bWET\b": "chuva",
        r"\bRace\b": "corrida",
        r"\bSprints?\b": "sprint",
        r"\blow speed\b": "baixa velocidade",
        r"\bmedium speed\b": "média velocidade",
        r"\bhigh speed\b": "alta velocidade",
    }
    updated = text
    for pattern, replacement in replacements.items():
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    return updated


def _contains_english_markers(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    markers = {
        "hard",
        "medium",
        "soft",
        "intermediate",
        "wet",
        "average",
        "driver",
        "drivers",
        "lap",
        "laps",
        "speed",
        "session",
        "race",
        "summary",
    }
    return any(token in markers for token in tokens)


def _question_is_fastest_lap(question: str) -> bool:
    text = question.lower()
    return (
        "volta mais rápida" in text
        or "volta mais rapida" in text
        or "lap mais rapido" in text
        or "lap mais rápido" in text
        or "fastest lap" in text
    )


def _fallback_gold_summary_pt(summary: dict[str, Any]) -> str:
    filters = summary.get("filters", {}) if isinstance(summary, dict) else {}
    season = filters.get("season")
    meetings_total = summary.get("meetings_total")
    drivers_total = summary.get("drivers_total")
    teams_total = summary.get("teams_total")
    sessions = summary.get("sessions") or []
    date_range = summary.get("meeting_date_start") or {}
    date_min = date_range.get("min")
    date_max = date_range.get("max")
    circuit_class = summary.get("circuit_speed_class_most_common_pt")
    compound = summary.get("compound_most_common_pt")
    drivers_focus = summary.get("drivers_focus") or []
    fastest_lap = summary.get("fastest_lap")

    parts: list[str] = []
    if season:
        parts.append(f"Resumo da temporada {season}.")
    if meetings_total:
        parts.append(f"Total de corridas: {meetings_total}.")
    if sessions:
        parts.append(f"Sessoes consideradas: {', '.join(sessions)}.")
    if drivers_total:
        parts.append(f"Pilotos na base: {drivers_total}.")
    if teams_total:
        parts.append(f"Equipes na base: {teams_total}.")
    if circuit_class:
        parts.append(f"Classe de velocidade mais comum: {circuit_class}.")
    if compound:
        parts.append(f"Composto de pneu mais usado: {compound}.")
    if date_min and date_max:
        parts.append(f"Periodo coberto: {date_min} ate {date_max}.")
    if drivers_focus:
        drivers_str = ", ".join(str(d) for d in drivers_focus)
        parts.append(f"Pilotos com mais voltas registradas: {drivers_str}.")

    if not parts:
        if fastest_lap:
            driver = fastest_lap.get("driver_name") or fastest_lap.get("driver_number")
            meeting = fastest_lap.get("meeting_name") or fastest_lap.get("meeting_key")
            lap_number = fastest_lap.get("lap_number")
            lap_time = fastest_lap.get("lap_duration_min") or fastest_lap.get("lap_duration")
            return (
                f"Volta mais rápida: {lap_time} na {meeting}, "
                f"piloto {driver}, volta {lap_number}."
            )
        return "Sem dados no gold."
    return _apply_pt_br_replacements(" ".join(parts))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _check_mlflow_dependency(tracking_uri: str | None, experiment_name: str) -> dict[str, Any]:
    if not tracking_uri:
        return {
            "status": "not_configured",
            "message": "MLFLOW_TRACKING_URI vazio; MLflow remoto nao configurado.",
        }
    start = time.monotonic()
    try:
        get_tracking_client(tracking_uri, experiment_name, create_if_missing=False)
        return {
            "status": "ok",
            "tracking_uri": tracking_uri,
            "latency_ms": int((time.monotonic() - start) * 1000),
        }
    except RuntimeError as exc:
        message = str(exc)
        if "Experimento MLflow nao encontrado" in message:
            return {
                "status": "degraded",
                "tracking_uri": tracking_uri,
                "message": message,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        if "MLFLOW_TRACKING_URI vazio" in message:
            return {"status": "not_configured", "message": message}
        return {
            "status": "down",
            "tracking_uri": tracking_uri,
            "message": message,
        }
    except Exception as exc:
        return {
            "status": "down",
            "tracking_uri": tracking_uri,
            "message": str(exc),
        }


def _check_minio_dependency(env: dict[str, str]) -> dict[str, Any]:
    endpoint = env.get("DATA_LAKE_S3_ENDPOINT") or env.get("MLFLOW_S3_ENDPOINT_URL")
    access_key = env.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY")
    bucket = env.get("DATA_LAKE_BUCKET", "openf1-datalake")
    if not endpoint or not access_key or not secret_key:
        return {
            "status": "not_configured",
            "message": "Credenciais S3 ou endpoint do data lake nao configurados.",
        }
    start = time.monotonic()
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        client.head_bucket(Bucket=bucket)
        return {
            "status": "ok",
            "endpoint": endpoint,
            "bucket": bucket,
            "latency_ms": int((time.monotonic() - start) * 1000),
        }
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        status = "degraded" if code in {"404", "NoSuchBucket", "NotFound"} else "down"
        return {
            "status": status,
            "endpoint": endpoint,
            "bucket": bucket,
            "message": f"Erro no MinIO/S3: {code or 'ClientError'}",
            "latency_ms": int((time.monotonic() - start) * 1000),
        }
    except Exception as exc:
        return {
            "status": "down",
            "endpoint": endpoint,
            "bucket": bucket,
            "message": str(exc),
        }


def _check_openf1_dependency(base_url: str | None) -> dict[str, Any]:
    if not base_url:
        return {"status": "not_configured", "message": "OPENF1_BASE_URL nao configurado."}
    url = base_url.rstrip("/") + "/meetings?limit=1"
    start = time.monotonic()
    try:
        req = Request(url, headers={"User-Agent": "OpenF1-DatasetBuilder/health"})
        with urlopen(req, timeout=8) as resp:
            status_code = resp.getcode()
    except HTTPError as exc:
        status_code = exc.code
    except URLError as exc:
        return {
            "status": "down",
            "url": url,
            "message": f"Falha de rede: {exc.reason}",
        }
    except Exception as exc:
        return {"status": "down", "url": url, "message": str(exc)}

    latency_ms = int((time.monotonic() - start) * 1000)
    if 200 <= status_code < 300:
        return {"status": "ok", "url": url, "status_code": status_code, "latency_ms": latency_ms}
    if status_code in {401, 403, 429}:
        return {
            "status": "degraded",
            "url": url,
            "status_code": status_code,
            "message": "OpenF1 pode ficar indisponivel para nao-assinantes em horario de eventos.",
            "latency_ms": latency_ms,
        }
    if 400 <= status_code < 500:
        return {
            "status": "degraded",
            "url": url,
            "status_code": status_code,
            "message": "Resposta 4xx da OpenF1.",
            "latency_ms": latency_ms,
        }
    return {
        "status": "down",
        "url": url,
        "status_code": status_code,
        "message": "Resposta 5xx da OpenF1.",
        "latency_ms": latency_ms,
    }


@app.get("/health/dependencies")
def health_dependencies() -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)

    dependencies = {
        "mlflow": _check_mlflow_dependency(
            settings.mlflow.tracking_uri, settings.mlflow.experiment_name
        ),
        "minio": _check_minio_dependency(env),
        "openf1": _check_openf1_dependency(settings.api.base_url),
    }
    overall = "ok"
    for dep in dependencies.values():
        if dep.get("status") != "ok":
            overall = "degraded"
            break
    return {
        "status": overall,
        "dependencies": dependencies,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/catalog/bronze")
def catalog_bronze(
    limit: int = 500,
    check_sync: bool = False,
    season: Optional[int] = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    data_dir = _data_dir(env, config_path)
    root = data_dir / "bronze"
    items: list[dict[str, Any]] = []

    sync_mode = "not_checked"
    sync_error = None
    sync_keys: set[str] | None = None
    prefix = ""
    if check_sync:
        try:
            client, bucket, prefix, _ = _get_minio_client(env)
            prefix_key = f"{prefix}/bronze/".strip("/")
            objects = _list_minio_objects(client, bucket, prefix_key, limit=0)
            sync_keys = {obj["key"] for obj in objects}
            sync_mode = "checked"
        except Exception as exc:
            sync_mode = "error"
            sync_error = str(exc)

    if root.exists():
        for path in sorted(root.rglob("*.json")):
            rel = _rel_path(path, data_dir)
            rel_in_layer = _rel_path(path, root)
            info = _parse_partition_info(path)
            source = path.stem
            if season is not None and info.get("season") != str(season):
                continue
            item: dict[str, Any] = {
                "layer": "bronze",
                "raw": True,
                "source": source,
                "path": rel,
                **info,
            }
            if check_sync and sync_keys is not None:
                key = "/".join(part for part in [prefix, "bronze", rel_in_layer] if part)
                item["sync"] = key in sync_keys
            else:
                item["sync"] = None
            items.append(item)
            if limit and len(items) >= limit:
                break

    return {
        "status": "ok",
        "layer": "bronze",
        "season": season,
        "count": len(items),
        "sync_mode": sync_mode,
        "sync_error": sync_error,
        "items": items,
    }


@app.get("/catalog/silver")
def catalog_silver(limit: int = 200, season: Optional[int] = None) -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    data_dir = _data_dir(env, config_path)
    root = data_dir / "silver"
    items: list[dict[str, Any]] = []

    if root.exists():
        for path in sorted(root.rglob("*.parquet")):
            rel = _rel_path(path, data_dir)
            info = _parse_partition_info(path)
            if season is not None and info.get("season") != str(season):
                continue
            profile = _parquet_profile(path)
            items.append(
                {
                    "layer": "silver",
                    "normalized": True,
                    "path": rel,
                    **info,
                    **profile,
                }
            )
            if limit and len(items) >= limit:
                break

    return {
        "status": "ok",
        "layer": "silver",
        "season": season,
        "count": len(items),
        "items": items,
    }


@app.get("/catalog/gold")
def catalog_gold(
    limit: int = 200,
    include_schema: bool = False,
    season: Optional[int] = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    data_dir = _data_dir(env, config_path)
    root = data_dir / "gold"
    items: list[dict[str, Any]] = []

    if root.exists():
        for path in sorted(root.rglob("dataset.parquet")):
            rel = _rel_path(path, data_dir)
            info = _parse_partition_info(path)
            if season is not None and info.get("season") != str(season):
                continue
            profile = _parquet_profile(path)
            if not include_schema:
                profile = {
                    "rows": profile.get("rows"),
                    "columns": profile.get("columns"),
                }
            items.append(
                {
                    "layer": "gold",
                    "granularity": "lap",
                    "path": rel,
                    **info,
                    **profile,
                }
            )
            if limit and len(items) >= limit:
                break

        consolidated = root / "consolidated.parquet"
        if consolidated.exists() and season is None and (not limit or len(items) < limit):
            profile = _parquet_profile(consolidated)
            if not include_schema:
                profile = {
                    "rows": profile.get("rows"),
                    "columns": profile.get("columns"),
                }
            items.append(
                {
                    "layer": "gold",
                    "granularity": "lap",
                    "type": "consolidated",
                    "path": _rel_path(consolidated, data_dir),
                    **profile,
                }
            )

    return {
        "status": "ok",
        "layer": "gold",
        "season": season,
        "count": len(items),
        "items": items,
    }


@app.get("/jobs")
def list_jobs(limit: int = 50) -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)
    items: list[dict[str, Any]] = []

    status_files = sorted(
        jobs_dir.glob("*.status.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for status_file in status_files:
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id = payload.get("job_id") or status_file.stem.replace(".status", "")
        job_file = jobs_dir / f"{job_id}.json"
        job_type = payload.get("job_type")
        if not job_type and job_file.exists():
            try:
                job_payload = json.loads(job_file.read_text(encoding="utf-8"))
                job_type = job_payload.get("job_type") or job_payload.get("module")
            except Exception:
                job_type = None
        items.append(
            {
                "job_id": job_id,
                "status": payload.get("status"),
                "job_type": job_type,
                "created_at": payload.get("created_at"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "message": payload.get("message"),
            }
        )
        if limit and len(items) >= limit:
            break

    return {"status": "ok", "count": len(items), "items": items}


@app.get("/mlflow/runs")
def list_mlflow_runs(limit: int = 50) -> dict[str, Any]:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    tracking_uri = env.get("MLFLOW_TRACKING_URI") or settings.mlflow.tracking_uri
    experiment_name = env.get("MLFLOW_EXPERIMENT") or settings.mlflow.experiment_name
    if not tracking_uri:
        raise HTTPException(status_code=400, detail="MLFLOW_TRACKING_URI nao configurado.")

    try:
        client, experiment_id = get_tracking_client(
            tracking_uri, experiment_name, create_if_missing=False
        )
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    items: list[dict[str, Any]] = []
    for run in runs:
        run_id = run.info.run_id
        try:
            artifacts = [
                {"path": art.path, "is_dir": bool(art.is_dir)}
                for art in client.list_artifacts(run_id)
            ]
        except Exception:
            artifacts = []
        items.append(
            {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status,
                "start_time": _ms_to_iso(run.info.start_time),
                "end_time": _ms_to_iso(run.info.end_time),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "artifact_uri": run.info.artifact_uri,
                "artifacts": artifacts,
            }
        )

    return {
        "status": "ok",
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "count": len(items),
        "items": items,
    }


@app.get("/minio/objects")
def list_minio_objects(prefix: Optional[str] = None, limit: int = 200) -> dict[str, Any]:
    env = os.environ.copy()
    try:
        client, bucket, base_prefix, endpoint = _get_minio_client(env)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_prefix = prefix.strip("/") if prefix else base_prefix
    objects = _list_minio_objects(client, bucket, resolved_prefix, limit=limit)
    items: list[dict[str, Any]] = []
    for obj in objects:
        key = obj.get("key", "")
        layer = None
        for candidate in ("bronze", "silver", "gold"):
            if f"/{candidate}/" in key:
                layer = candidate
                break
        items.append(
            {
                "bucket": bucket,
                "key": key,
                "size": obj.get("size"),
                "layer": layer,
                "uri": f"s3://{bucket}/{key}",
                "last_modified": obj.get("last_modified"),
            }
        )

    return {
        "status": "ok",
        "endpoint": endpoint,
        "bucket": bucket,
        "prefix": resolved_prefix,
        "count": len(items),
        "items": items,
    }


@app.get("/ui/gold-lap", response_class=HTMLResponse)
def ui_gold_lap() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Gold Lap Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />
    <style>
      :root {
        --bg: #0f1518;
        --bg-soft: #151d22;
        --panel: #1b262c;
        --panel-strong: #23313a;
        --accent: #55d6be;
        --accent-2: #f7b267;
        --text: #f5f7f9;
        --muted: #9fb0bb;
        --danger: #ff6b6b;
        --border: rgba(255,255,255,0.08);
        --shadow: 0 10px 30px rgba(0,0,0,0.35);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
        color: var(--text);
        background: radial-gradient(1200px 600px at 10% -10%, rgba(85,214,190,0.25), transparent 60%),
                    radial-gradient(900px 500px at 90% 0%, rgba(247,178,103,0.2), transparent 55%),
                    var(--bg);
        min-height: 100vh;
      }
      header {
        padding: 28px 32px 8px;
      }
      header h1 {
        margin: 0 0 6px;
        font-size: 28px;
        letter-spacing: 0.3px;
      }
      header p {
        margin: 0;
        color: var(--muted);
      }
      .container {
        padding: 16px 32px 40px;
        display: grid;
        grid-template-columns: minmax(260px, 320px) 1fr;
        gap: 20px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 16px;
      }
      .panel h2 {
        margin: 0 0 12px;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--muted);
      }
      .form-grid {
        display: grid;
        gap: 12px;
      }
      label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--muted);
      }
      input, select, button {
        width: 100%;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
        color: var(--text);
        font-family: inherit;
        font-size: 14px;
      }
      input:focus, select:focus {
        outline: 2px solid rgba(85,214,190,0.5);
      }
      .inline {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .actions {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      button {
        cursor: pointer;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      button.primary {
        background: linear-gradient(120deg, var(--accent), #3fa7d6);
        color: #081114;
        border: none;
      }
      button.secondary {
        background: linear-gradient(120deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
      }
      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
      }
      .info {
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(85,214,190,0.08);
        border: 1px solid rgba(85,214,190,0.25);
        font-size: 13px;
      }
      .error {
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(255,107,107,0.12);
        border: 1px solid rgba(255,107,107,0.4);
        color: var(--danger);
        font-size: 13px;
      }
      .columns {
        max-height: 360px;
        overflow: auto;
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 10px;
        display: grid;
        gap: 6px;
      }
      .columns label {
        font-size: 12px;
        text-transform: none;
        letter-spacing: 0.2px;
        color: var(--text);
        display: flex;
        gap: 8px;
        align-items: center;
      }
      .grid-panel {
        display: grid;
        grid-template-rows: auto 1fr;
        gap: 12px;
      }
      .summary {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        font-size: 14px;
        color: var(--muted);
      }
      .summary span {
        color: var(--text);
        font-weight: 600;
      }
      .table-wrap {
        overflow: auto;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: rgba(12,16,18,0.6);
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      thead {
        position: sticky;
        top: 0;
        background: #0f181d;
        z-index: 1;
      }
      th, td {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        white-space: nowrap;
      }
      td.highlight {
        background: rgba(200, 170, 255, 0.25);
        box-shadow: inset 0 0 0 1px rgba(200, 170, 255, 0.45);
        border-bottom-color: rgba(200, 170, 255, 0.35);
      }
      td.highlight-fastest {
        background: rgba(120, 220, 160, 0.25);
        box-shadow: inset 0 0 0 1px rgba(120, 220, 160, 0.5);
        border-bottom-color: rgba(120, 220, 160, 0.4);
        color: #eafaf0;
      }
      th {
        text-align: left;
        font-weight: 600;
        color: var(--accent);
      }
      tbody tr:nth-child(odd) {
        background: rgba(255,255,255,0.02);
      }
      .mono { font-family: "JetBrains Mono", "Courier New", monospace; }
      .muted { color: var(--muted); }
      .fade-in {
        animation: fade 0.4s ease;
      }
      @keyframes fade {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @media (max-width: 980px) {
        .container {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>Gold Lap Viewer</h1>
      <p>Consulta por temporada, pista (meeting) e volta para listar dados por piloto.</p>
    </header>

    <section class="container">
      <aside class="panel">
        <h2>Parametros</h2>
        <div class="form-grid">
          <div>
            <label for="season">Season (obrigatorio)</label>
            <input id="season" type="number" min="2019" max="2100" step="1" value="2024" />
          </div>
          <input id="meeting_key" type="hidden" />
          <div>
            <label for="meeting_name">Meeting Name (obrigatorio)</label>
            <select id="meeting_name">
              <option value="">Selecione a etapa</option>
            </select>
          </div>
          <div>
            <label for="session_name">Session</label>
            <select id="session_name">
              <option value="Race">Race</option>
              <option value="Sprint">Sprint</option>
              <option value="all">all</option>
            </select>
          </div>
          <div class="inline">
            <div>
              <label for="lap_number">Lap Number (obrigatorio)</label>
              <input id="lap_number" list="lap-options" type="number" min="1" step="1" />
              <datalist id="lap-options"></datalist>
            </div>
            <div>
              <label>Max Lap</label>
              <div id="max-lap" class="info">--</div>
            </div>
          </div>
          <div class="actions">
            <button class="secondary" id="btn-max">Atualizar voltas</button>
            <button class="primary" id="btn-fetch">Buscar dados</button>
          </div>
          <div id="hint" class="info">Meeting name e carregado a partir da temporada selecionada.</div>
          <div id="error" class="error" style="display:none;"></div>
        </div>

        <h2 style="margin-top:18px;">Colunas</h2>
        <div class="actions" style="margin-bottom:10px;">
          <button class="secondary" id="btn-all">Selecionar tudo</button>
          <button class="secondary" id="btn-none">Limpar</button>
        </div>
        <div id="columns" class="columns"></div>

        <h2 style="margin-top:18px;">Ordenacao</h2>
        <div class="form-grid">
          <div>
            <label for="sort_col">Coluna</label>
            <select id="sort_col"></select>
          </div>
          <div class="inline">
            <button class="secondary" id="sort_dir" type="button">Ascendente</button>
            <button class="secondary" id="apply_sort" type="button">Aplicar</button>
          </div>
        </div>
      </aside>

      <main class="panel grid-panel">
        <div class="summary fade-in" id="summary">
          <div>Ano: <span id="summary-season">--</span></div>
          <div>Meeting Key: <span id="summary-key">--</span></div>
          <div>Meeting Name: <span id="summary-name">--</span> | Volta Rapida: <span id="summary-fastest">--</span></div>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="table-head"></thead>
            <tbody id="table-body"></tbody>
          </table>
        </div>
      </main>
    </section>

    <script>
      const state = {
        columns: [],
        data: [],
        selected: new Set(),
        sort: { column: null, dir: "asc" },
        maxLap: null,
      };

      const el = (id) => document.getElementById(id);
      const errorBox = el("error");

      const showError = (msg) => {
        if (!msg) {
          errorBox.style.display = "none";
          errorBox.textContent = "";
          return;
        }
        errorBox.textContent = msg;
        errorBox.style.display = "block";
      };

      const buildQuery = (silent = false) => {
        const season = el("season").value.trim();
        const meetingKey = el("meeting_key").value.trim();
        const meetingName = el("meeting_name").value.trim();
        const sessionName = el("session_name").value.trim();
        const lapNumber = el("lap_number").value.trim();

        if (!season || !lapNumber) {
          if (!silent) showError("Season e lap_number sao obrigatorios.");
          return null;
        }
        if (!meetingKey && !meetingName) {
          if (!silent) showError("Informe o meeting_name.");
          return null;
        }

        const params = new URLSearchParams();
        params.set("season", season);
        params.set("session_name", sessionName);
        params.set("lap_number", lapNumber);
        if (meetingKey) {
          params.set("meeting_key", meetingKey);
        } else {
          params.set("meeting_name", meetingName);
        }
        return params.toString();
      };

      const buildMaxQuery = (silent = false) => {
        const season = el("season").value.trim();
        const meetingKey = el("meeting_key").value.trim();
        const meetingName = el("meeting_name").value.trim();
        const sessionName = el("session_name").value.trim();
        if (!season) {
          if (!silent) showError("Season e obrigatorio.");
          return null;
        }
        if (!meetingKey && !meetingName) {
          if (!silent) showError("Informe o meeting_name.");
          return null;
        }
        const params = new URLSearchParams();
        params.set("season", season);
        params.set("session_name", sessionName);
        if (meetingKey) {
          params.set("meeting_key", meetingKey);
        } else {
          params.set("meeting_name", meetingName);
        }
        return params.toString();
      };

      const updateSummary = (row, fastest) => {
        const season = el("season").value.trim();
        const meetingKey = el("meeting_key").value.trim();
        const meetingName = el("meeting_name").selectedOptions?.[0]?.textContent || "";
        el("summary-season").textContent = row?.season ?? season ?? "--";
        el("summary-key").textContent = row?.meeting_key ?? meetingKey ?? "--";
        el("summary-name").textContent = row?.meeting_name ?? meetingName ?? "--";
        if (fastest) {
          const driver = fastest.driver_name || fastest.driver_number || "--";
          const time = fastest.lap_duration_min || "--";
          const lap = fastest.lap_number ?? "--";
          el("summary-fastest").textContent = `${driver} | ${time} | ${lap}`;
        } else {
          el("summary-fastest").textContent = "--";
        }
      };

      const updateColumnsPanel = () => {
        const wrapper = el("columns");
        wrapper.innerHTML = "";
        state.columns.forEach((col) => {
          const label = document.createElement("label");
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = state.selected.has(col);
          checkbox.addEventListener("change", () => {
            if (checkbox.checked) {
              state.selected.add(col);
            } else {
              state.selected.delete(col);
            }
            renderTable();
          });
          label.appendChild(checkbox);
          label.appendChild(document.createTextNode(col));
          wrapper.appendChild(label);
        });

        const sortSelect = el("sort_col");
        sortSelect.innerHTML = "";
        state.columns.forEach((col) => {
          const opt = document.createElement("option");
          opt.value = col;
          opt.textContent = col;
          sortSelect.appendChild(opt);
        });
        if (state.sort.column) {
          sortSelect.value = state.sort.column;
        }
      };

      const renderTable = () => {
        const head = el("table-head");
        const body = el("table-body");
        head.innerHTML = "";
        body.innerHTML = "";

        const cols = state.columns.filter((c) => state.selected.has(c));
        if (!cols.length) {
          head.innerHTML = "<tr><th class='muted'>Nenhuma coluna selecionada.</th></tr>";
          return;
        }
        head.innerHTML = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";

        const durationCols = new Set([
          "lap_duration_min",
          "lap_duration_total",
          "duration_sector_1",
          "duration_sector_2",
          "duration_sector_3",
        ]);
        const minCols = new Set([
          "lap_duration_min",
          "lap_duration_total",
          "duration_sector_1",
          "duration_sector_2",
          "duration_sector_3",
          "brake_events",
          "gear_changes",
        ]);
        const maxCols = new Set([
          "i1_speed",
          "i2_speed",
          "i3_speed",
          "st_speed",
          "avg_speed",
          "max_speed",
          "avg_rpm",
          "max_rpm",
        ]);

        const parseDurationSeconds = (value) => {
          if (value === null || value === undefined || value === "") return NaN;
          if (typeof value === "number") return value;
          const text = String(value).trim();
          if (!text) return NaN;
          if (text.includes("day")) {
            const parts = text.split(" ").filter(Boolean);
            const days = Number(parts[0]) || 0;
            const timePart = parts[parts.length - 1];
            const timeSecs = parseDurationSeconds(timePart);
            return days * 86400 + (Number.isNaN(timeSecs) ? 0 : timeSecs);
          }
          const timeParts = text.split(":").map((p) => p.trim());
          if (timeParts.length === 4) {
            const [h, m, s, ms] = timeParts;
            return Number(h) * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000;
          }
          if (timeParts.length === 3) {
            const [a, b, c] = timeParts;
            if (c.length === 3) {
              return Number(a) * 60 + Number(b) + Number(c) / 1000;
            }
            return Number(a) * 3600 + Number(b) * 60 + Number(c);
          }
          if (timeParts.length === 2) {
            const [m, s] = timeParts;
            return Number(m) * 60 + Number(s);
          }
          const asNum = Number(text.replace(",", "."));
          return Number.isNaN(asNum) ? NaN : asNum;
        };

        const valueToNumber = (col, value) => {
          if (durationCols.has(col)) {
            return parseDurationSeconds(value);
          }
          const num = Number(value);
          return Number.isNaN(num) ? NaN : num;
        };

        const extrema = {};
        cols.forEach((col) => {
          if (!minCols.has(col) && !maxCols.has(col)) return;
          const values = state.data
            .map((row) => valueToNumber(col, row[col]))
            .filter((val) => Number.isFinite(val));
          if (!values.length) return;
          extrema[col] = minCols.has(col)
            ? Math.min(...values)
            : Math.max(...values);
        });

        const rows = [...state.data];
        if (state.sort.column && cols.includes(state.sort.column)) {
          rows.sort((a, b) => {
            const av = a[state.sort.column];
            const bv = b[state.sort.column];
            if (av === null || av === undefined) return 1;
            if (bv === null || bv === undefined) return -1;
            const an = Number(av);
            const bn = Number(bv);
            let result;
            if (!Number.isNaN(an) && !Number.isNaN(bn)) {
              result = an - bn;
            } else {
              result = String(av).localeCompare(String(bv));
            }
            return state.sort.dir === "asc" ? result : -result;
          });
        }

        rows.forEach((row) => {
          const tr = document.createElement("tr");
          cols.forEach((c) => {
            const td = document.createElement("td");
            const value = row[c];
            if (typeof value === "number") {
              td.textContent = Number.isInteger(value) ? value : value.toFixed(3);
            } else {
              td.textContent = value ?? "";
            }
            if (c.includes("date") || c.includes("time")) td.classList.add("mono");
            const best = extrema[c];
            if (best !== undefined) {
              const numeric = valueToNumber(c, value);
              if (Number.isFinite(numeric) && Math.abs(numeric - best) < 1e-6) {
                if (c === "lap_duration_min") {
                  td.classList.add("highlight-fastest");
                } else {
                  td.classList.add("highlight");
                }
              }
            }
            tr.appendChild(td);
          });
          body.appendChild(tr);
        });
      };

      const fetchMaxLap = async (silent = false) => {
        showError("");
        const query = buildMaxQuery(silent);
        if (!query) return;
        try {
          const resp = await fetch(`/gold/laps/max?${query}`);
          if (!resp.ok) {
            const payload = await resp.json();
            throw new Error(payload.detail || "Falha ao buscar max lap.");
          }
          const payload = await resp.json();
          state.maxLap = payload.max_lap_number;
          el("max-lap").textContent = payload.max_lap_number ?? "--";
          updateSummary(payload, null);
          const list = el("lap-options");
          list.innerHTML = "";
          const max = payload.max_lap_number || 0;
          for (let i = 1; i <= max; i += 1) {
            const opt = document.createElement("option");
            opt.value = String(i);
            list.appendChild(opt);
          }
          return payload.max_lap_number;
        } catch (err) {
          showError(err.message);
          return null;
        }
      };

      const fetchMeetings = async (silent = false) => {
        const season = el("season").value.trim();
        const sessionName = el("session_name").value.trim();
        if (!season) return;
        try {
          const resp = await fetch(`/gold/meetings?season=${encodeURIComponent(season)}&session_name=${encodeURIComponent(sessionName)}`);
          if (!resp.ok) {
            const payload = await resp.json();
            throw new Error(payload.detail || "Falha ao carregar meetings.");
          }
          const payload = await resp.json();
          const select = el("meeting_name");
          const currentKey = el("meeting_key").value.trim();
          const currentName = select.selectedOptions?.[0]?.textContent || "";
          select.innerHTML = "<option value=''>Selecione a etapa</option>";
          (payload.meetings || []).forEach((meeting) => {
            const opt = document.createElement("option");
            opt.value = meeting.meeting_key || "";
            opt.textContent = meeting.meeting_name || meeting.meeting_key || "Etapa";
            select.appendChild(opt);
          });
          let matched = false;
          if (currentKey) {
            const opt = [...select.options].find((o) => o.value === currentKey);
            if (opt) {
              opt.selected = true;
              matched = true;
            }
          }
          if (!matched && currentName) {
            const opt = [...select.options].find((o) => o.textContent === currentName);
            if (opt) {
              opt.selected = true;
              matched = true;
            }
          }
          if (!matched && select.options.length > 1) {
            select.selectedIndex = 1;
          }
          const selectedKey = select.value;
          el("meeting_key").value = selectedKey;
          updateSummary({}, null);
          return payload.meetings || [];
        } catch (err) {
          if (!silent) showError(err.message);
          return [];
        }
      };

      const fetchData = async (silent = false) => {
        showError("");
        await fetchMeetings(true);
        const query = buildQuery(silent);
        if (!query) return;
        const maxLap = await fetchMaxLap(silent);
        const lapValue = Number(el("lap_number").value);
        if (maxLap && lapValue > maxLap) {
          if (!silent) showError(`Lap_number acima do maximo (${maxLap}).`);
          return;
        }
        try {
          const resp = await fetch(`/gold/lap?${query}`);
          if (!resp.ok) {
            const payload = await resp.json();
            throw new Error(payload.detail || "Falha ao buscar dados.");
          }
          const payload = await resp.json();
          const prevSelected = new Set(state.selected);
          const wasEmpty = state.columns.length === 0;
          state.columns = payload.columns || [];
          state.data = payload.data || [];
          if (wasEmpty) {
            state.selected = new Set(state.columns);
          } else {
            state.selected = new Set();
            state.columns.forEach((col) => {
              if (prevSelected.has(col)) state.selected.add(col);
            });
          }
          if (state.data[0]) updateSummary(state.data[0], payload.fastest_lap);
          updateColumnsPanel();
          renderTable();
        } catch (err) {
          if (!silent) showError(err.message);
        }
      };

      el("btn-fetch").addEventListener("click", () => fetchData(false));
      el("btn-max").addEventListener("click", () => fetchMaxLap(false));
      el("btn-all").addEventListener("click", () => {
        state.selected = new Set(state.columns);
        updateColumnsPanel();
        renderTable();
      });
      el("btn-none").addEventListener("click", () => {
        state.selected = new Set();
        updateColumnsPanel();
        renderTable();
      });
      el("sort_dir").addEventListener("click", () => {
        state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
        el("sort_dir").textContent = state.sort.dir === "asc" ? "Ascendente" : "Descendente";
      });
      el("apply_sort").addEventListener("click", () => {
        state.sort.column = el("sort_col").value || null;
        renderTable();
      });

      let lapTimer = null;
      el("lap_number").addEventListener("input", () => {
        if (lapTimer) clearTimeout(lapTimer);
        lapTimer = setTimeout(() => {
          fetchData(true);
        }, 350);
      });
      el("lap_number").addEventListener("change", () => fetchData(true));
      el("lap_number").addEventListener("keyup", (event) => {
        if (event.key === "Enter") fetchData(true);
      });

      const scheduleAutoFetch = (() => {
        let timer = null;
        return () => {
          if (timer) clearTimeout(timer);
          timer = setTimeout(() => fetchData(true), 350);
        };
      })();

      ["season", "meeting_name", "session_name"].forEach((id) => {
        const input = el(id);
        input.addEventListener("input", scheduleAutoFetch);
        input.addEventListener("change", () => fetchData(true));
        input.addEventListener("keyup", (event) => {
          if (event.key === "Enter") fetchData(true);
        });
      });

      el("meeting_name").addEventListener("change", () => {
        el("meeting_key").value = el("meeting_name").value;
        fetchData(true);
      });

      fetchMeetings(true);
    </script>
  </body>
</html>
    """
    return HTMLResponse(content=html)


@app.post("/data-lake/sync", response_model=DataLakeSyncResponse)
def data_lake_sync(payload: DataLakeSyncRequest) -> DataLakeSyncResponse:
    env = os.environ.copy()
    data_dir = Path(env.get("DATA_DIR", "/app/data"))
    subdirs = payload.subdirs or None
    try:
        if payload.direction == "upload":
            files = sync_data_lake(data_dir, env, subdirs=subdirs)
            cleanup_local = (
                payload.cleanup_local
                if payload.cleanup_local is not None
                else should_cleanup_data_lake(env)
            )
            if cleanup_local and files:
                cleanup_paths([data_dir / subdir for subdir in files.keys()])
            return DataLakeSyncResponse(
                status="ok", direction="upload", files=files
            )
        if payload.direction == "download":
            files = download_data_lake(
                data_dir, env, subdirs=subdirs, only_if_missing=payload.only_if_missing
            )
            return DataLakeSyncResponse(
                status="ok", direction="download", files=files
            )
        raise HTTPException(status_code=400, detail="direction invalida.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/gold/questions", response_model=GoldQuestionsResponse)
def perguntas_gold(payload: GoldQuestionsRequest) -> GoldQuestionsResponse:
    question = _normalize_text(payload.question)
    if not question:
        raise HTTPException(status_code=400, detail="question e obrigatoria.")

    session_name = (payload.session_name or "all").strip()
    if session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")

    driver_name = _normalize_text(payload.driver_name)
    meeting_key = _normalize_text(payload.meeting_key)
    driver_number = payload.driver_number

    try:
        df = load_consolidated()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filtered = df[df["season"] == payload.season] if "season" in df.columns else df
    if meeting_key and "meeting_key" in filtered.columns:
        filtered = filtered[filtered["meeting_key"].astype(str) == str(meeting_key)]
    if session_name.lower() != "all" and "session_name" in filtered.columns:
        filtered = filtered[
            filtered["session_name"].astype(str).str.strip().str.lower() == session_name.lower()
        ]
    if driver_number is not None and "driver_number" in filtered.columns:
        filtered = filtered[filtered["driver_number"].astype(str) == str(driver_number)]
    if driver_name and "driver_name" in filtered.columns:
        filtered = filtered[
            filtered["driver_name"].astype(str).str.strip().str.lower() == driver_name.lower()
        ]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado no gold para os filtros.")

    summary = _build_gold_summary(filtered, payload, question)

    if _question_is_fastest_lap(question):
        fastest = summary.get("fastest_lap")
        if fastest:
            driver = fastest.get("driver_name") or fastest.get("driver_number") or "--"
            meeting = fastest.get("meeting_name") or fastest.get("meeting_key") or "--"
            lap_number = fastest.get("lap_number")
            lap_time = fastest.get("lap_duration_min") or fastest.get("lap_duration") or "--"
            detail = f"{driver} | {lap_time}"
            if lap_number is not None:
                detail += f" | volta {lap_number}"
            answer = f"A volta mais rápida foi {detail} na {meeting}."
            return GoldQuestionsResponse(status="ok", answer=answer, summary=summary)

    env = os.environ.copy()
    duckdb_answer = _try_duckdb_answer(
        question=question,
        payload=payload,
        filtered=filtered,
        summary=summary,
        env=env,
    )
    if duckdb_answer:
        return GoldQuestionsResponse(status="ok", answer=duckdb_answer, summary=summary)
    llm_payload = _build_gold_prompt(question, summary)

    llm_endpoint = env.get(
        "MLFLOW_GATEWAY_ENDPOINT",
        "http://mlflow:5000/gateway/gemini/mlflow/invocations",
    )
    status, content = _post_json(llm_endpoint, llm_payload, timeout_s=60)
    if status != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Falha ao chamar LLM (status={status}): {content}",
        )
    try:
        answer = _parse_llm_answer(content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Resposta invalida do LLM: {exc}") from exc

    if not _is_probably_portuguese(answer):
        retry_payload = _build_gold_prompt(question, summary, strict_portuguese=True)
        status, content = _post_json(llm_endpoint, retry_payload, timeout_s=60)
        if status == 200:
            try:
                answer = _parse_llm_answer(content)
            except Exception:
                pass
    answer = _apply_pt_br_replacements(answer)
    normalized_answer = answer.strip().lower()
    if normalized_answer == "sem dados no gold." and summary.get("rows"):
        web_answer = _web_fallback_answer(question)
        if web_answer:
            answer = web_answer
        else:
            answer = _fallback_gold_summary_pt(summary)
    elif not _is_probably_portuguese(answer) or _contains_english_markers(answer):
        answer = _fallback_gold_summary_pt(summary)

    return GoldQuestionsResponse(status="ok", answer=answer, summary=summary)


@app.get("/gold/meetings", response_model=GoldMeetingsResponse)
def list_gold_meetings(
    season: Optional[int] = None,
    session_name: str = "all",
) -> GoldMeetingsResponse:
    session_name = (session_name or "all").strip()
    if session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")

    try:
        df = load_consolidated()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if season is not None:
        if "season" not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna season nao encontrada no gold.")
        df = df[df["season"] == season]

    if session_name.lower() != "all":
        if "session_name" not in df.columns:
            raise HTTPException(
                status_code=400, detail="Coluna session_name nao encontrada no gold."
            )
        df = df[
            df["session_name"].astype(str).str.strip().str.lower() == session_name.lower()
        ]

    if df.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado no gold para os filtros.")
    if "meeting_key" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna meeting_key nao encontrada no gold.")

    season_series = (
        pd.to_numeric(df["season"], errors="coerce") if "season" in df.columns else None
    )
    meeting_key_series = _clean_text_series(df["meeting_key"])
    meeting_name_series = (
        _clean_text_series(df["meeting_name"])
        if "meeting_name" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )
    session_series = (
        _clean_text_series(df["session_name"])
        if "session_name" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )

    data = pd.DataFrame(
        {
            "season": season_series if season_series is not None else pd.Series(
                [pd.NA] * len(df), index=df.index, dtype="Int64"
            ),
            "meeting_key": meeting_key_series,
            "meeting_name": meeting_name_series,
            "session_name": session_series,
        }
    )
    data = data.dropna(subset=["meeting_key"])
    data = data[data["meeting_key"].astype(str).str.strip() != ""]

    if data.empty:
        raise HTTPException(status_code=404, detail="Nenhum meeting encontrado no gold.")

    sessions_all = (
        sorted(
            s for s in data["session_name"].dropna().astype(str).unique() if str(s).strip()
        )
        if "session_name" in data.columns
        else []
    )
    seasons_all = (
        sorted(int(s) for s in data["season"].dropna().unique())
        if "season" in data.columns
        else []
    )

    meetings: list[GoldMeetingItem] = []
    grouped = data.groupby(["season", "meeting_key", "meeting_name"], dropna=False)
    for (season_val, meeting_key, meeting_name), group in grouped:
        sessions = sorted(
            s for s in group["session_name"].dropna().astype(str).unique() if str(s).strip()
        )
        meeting = GoldMeetingItem(
            season=int(season_val) if pd.notna(season_val) else None,
            meeting_key=str(meeting_key),
            meeting_name=str(meeting_name) if pd.notna(meeting_name) else None,
            sessions=sessions,
        )
        meetings.append(meeting)

    def _meeting_sort_key(item: GoldMeetingItem) -> tuple:
        season_key = item.season if item.season is not None else 10**9
        return (season_key, item.meeting_key)

    meetings.sort(key=_meeting_sort_key)

    return GoldMeetingsResponse(
        status="ok",
        rows=len(meetings),
        seasons=seasons_all,
        sessions=sessions_all,
        meetings=meetings,
    )


@app.get("/gold/lap", response_model=GoldLapDriversResponse)
def gold_lap_drivers(
    season: int,
    lap_number: int,
    meeting_key: Optional[str] = None,
    meeting_name: Optional[str] = None,
    session_name: str = "Race",
) -> GoldLapDriversResponse:
    if meeting_key is None and meeting_name is None:
        raise HTTPException(status_code=400, detail="Informe meeting_key ou meeting_name.")

    if session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")

    try:
        df = load_consolidated()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "lap_number" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna lap_number nao encontrada no gold.")

    filtered = _filter_gold_dataset(
        df,
        season=season,
        meeting_key=meeting_key,
        meeting_name=meeting_name,
        session_name=session_name,
    )

    filtered = filtered.copy()
    filtered["lap_number"] = pd.to_numeric(filtered["lap_number"], errors="coerce")
    filtered = filtered.dropna(subset=["lap_number"])

    if "lap_duration" in filtered.columns:
        filtered["lap_duration_seconds"] = _duration_seconds(filtered["lap_duration"])
    else:
        filtered["lap_duration_seconds"] = pd.NA

    driver_col = None
    for candidate in ["driver_number", "driver_name"]:
        if candidate in filtered.columns:
            driver_col = candidate
            break
    if not driver_col:
        raise HTTPException(status_code=500, detail="Coluna driver_number/driver_name nao encontrada no gold.")

    filtered = filtered.sort_values([driver_col, "lap_number"])
    filtered["lap_duration_total_seconds"] = (
        filtered.groupby(driver_col)["lap_duration_seconds"].cumsum()
    )

    fastest_lap = None
    up_to = filtered[filtered["lap_number"].astype(int) <= int(lap_number)]
    up_to = up_to[pd.to_numeric(up_to["lap_duration_seconds"], errors="coerce").notna()]
    if not up_to.empty:
        idx = up_to["lap_duration_seconds"].astype(float).idxmin()
        row = up_to.loc[idx]
        driver_name = row.get("driver_name") or row.get("driver_number")
        fastest_lap = {
            "driver_name": _jsonable_value(driver_name),
            "driver_number": _jsonable_value(row.get("driver_number")),
            "lap_number": _jsonable_value(
                int(row.get("lap_number")) if pd.notna(row.get("lap_number")) else None
            ),
            "lap_duration_min": _jsonable_value(
                _format_mmss(row.get("lap_duration_seconds"))
            ),
        }

    filtered = filtered[filtered["lap_number"].astype(int) == int(lap_number)]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado para os filtros.")

    filtered = filtered.copy()
    filtered["lap_duration_min"] = filtered["lap_duration_seconds"].map(_format_mmss)
    filtered["lap_duration_total"] = filtered["lap_duration_total_seconds"].map(_format_hhmmss)

    min_total = pd.to_numeric(filtered["lap_duration_total_seconds"], errors="coerce").min()
    if pd.isna(min_total):
        filtered["lap_duration_gap"] = None
    else:
        filtered["lap_duration_gap"] = (
            filtered["lap_duration_total_seconds"] - float(min_total)
        ).map(_format_hhmmss)

    filtered = filtered.sort_values("lap_duration_total_seconds", na_position="last")

    cols = list(filtered.columns)
    for col in ["lap_duration_min", "lap_duration_total", "lap_duration_gap"]:
        if col in cols:
            cols.remove(col)
    if "lap_duration" in cols:
        insert_at = cols.index("lap_duration") + 1
        for col in ["lap_duration_min", "lap_duration_total", "lap_duration_gap"]:
            cols.insert(insert_at, col)
            insert_at += 1
    else:
        cols.extend(["lap_duration_min", "lap_duration_total", "lap_duration_gap"])

    filtered = filtered[cols]
    filtered = filtered.drop(columns=["lap_duration_seconds", "lap_duration_total_seconds"], errors="ignore")

    data = filtered.to_dict(orient="records")
    return GoldLapDriversResponse(
        status="ok",
        rows=len(data),
        columns=list(filtered.columns),
        data=data,
        fastest_lap=fastest_lap,
    )


@app.get("/gold/laps/max", response_model=GoldLapMaxResponse)
def gold_lap_max(
    season: int,
    meeting_key: Optional[str] = None,
    meeting_name: Optional[str] = None,
    session_name: str = "Race",
) -> GoldLapMaxResponse:
    if meeting_key is None and meeting_name is None:
        raise HTTPException(status_code=400, detail="Informe meeting_key ou meeting_name.")

    if session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")

    try:
        df = load_consolidated()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "lap_number" not in df.columns:
        raise HTTPException(status_code=500, detail="Coluna lap_number nao encontrada no gold.")

    filtered = _filter_gold_dataset(
        df,
        season=season,
        meeting_key=meeting_key,
        meeting_name=meeting_name,
        session_name=session_name,
    )

    if filtered.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado para os filtros.")

    lap_series = pd.to_numeric(filtered["lap_number"], errors="coerce").dropna()
    if lap_series.empty:
        raise HTTPException(status_code=404, detail="Nenhuma volta encontrada para os filtros.")

    max_lap = int(lap_series.max())
    meeting_key_out = None
    meeting_name_out = None
    if "meeting_key" in filtered.columns:
        meeting_key_out = str(filtered["meeting_key"].iloc[0])
    if "meeting_name" in filtered.columns:
        meeting_name_out = str(filtered["meeting_name"].iloc[0])

    return GoldLapMaxResponse(
        status="ok",
        season=season,
        session_name=session_name,
        meeting_key=meeting_key_out,
        meeting_name=meeting_name_out,
        max_lap_number=max_lap,
    )


@app.post("/driver-profiles", response_model=DriverProfilesResponse)
def generate_driver_profiles(payload: DriverProfilesRequest) -> DriverProfilesResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_dir = Path(env.get("ARTIFACTS_DIR", "/app/artifacts"))
    data_dir = Path(env.get("DATA_DIR", "/app/data"))
    base_dir = artifacts_dir / "modeling" / "driver_profiles"
    run_group = str(uuid.uuid4())
    env["MLFLOW_RUN_GROUP"] = run_group
    env["RUN_SEASON"] = str(payload.season)
    env["RUN_MEETING_KEY"] = str(payload.meeting_key)
    env["RUN_SESSION_NAME"] = str(payload.session_name)
    os.environ["MLFLOW_RUN_GROUP"] = run_group
    os.environ["RUN_SEASON"] = str(payload.season)
    os.environ["RUN_MEETING_KEY"] = str(payload.meeting_key)
    os.environ["RUN_SESSION_NAME"] = str(payload.session_name)

    if not str(payload.meeting_key).strip():
        raise HTTPException(status_code=400, detail="meeting_key e obrigatorio.")
    if payload.session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")
    if not settings.output.register_mlflow:
        raise HTTPException(
            status_code=500,
            detail="REGISTER_MLFLOW=false; MLflow/MinIO e necessario para este endpoint.",
        )

    try:
        tracking_uri = env.get("MLFLOW_TRACKING_URI") or settings.mlflow.tracking_uri or ""
        experiment_name = env.get("MLFLOW_EXPERIMENT") or settings.mlflow.experiment_name
        create_exp = env.get("MLFLOW_CREATE_EXPERIMENT", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        tracking_client, experiment_id = get_tracking_client(
            tracking_uri, experiment_name, create_if_missing=create_exp
        )

        ensure_data(
            env,
            payload.season,
            payload.meeting_key,
            payload.session_name,
            config_path,
            data_dir,
            drivers_include=payload.drivers_include,
            drivers_exclude=payload.drivers_exclude,
        )
        required_columns = [
            "meeting_date_start",
            "weather_date",
            "track_temperature",
            "air_temperature",
            "circuit_speed_class",
        ]
        required_non_null = ["meeting_date_start"]
        if not has_data_for_filter(
            data_dir,
            payload.season,
            payload.meeting_key,
            payload.session_name,
            required_columns=required_columns,
            required_non_null=required_non_null,
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Nenhum dado encontrado apos importacao. "
                    "Verifique se a corrida/sessao existe na OpenF1."
                ),
            )

        report_cmd = [
            "python",
            "-m",
            "jobs.driver_profiles_report",
            "--config",
            config_path,
            "--season",
            str(payload.season),
            "--meeting-key",
            str(payload.meeting_key),
            "--session-name",
            payload.session_name,
        ]
        if payload.drivers_include:
            report_cmd += ["--drivers-include", ", ".join(payload.drivers_include)]
        if payload.drivers_exclude:
            report_cmd += ["--drivers-exclude", ", ".join(payload.drivers_exclude)]
        run_cmd(report_cmd, env)

        run_cmd(
            [
                "python",
                "-m",
                "jobs.driver_profiles_overall_ranking",
                "--config",
                config_path,
            ],
            env,
        )

        run_cmd(
            [
                "python",
                "-m",
                "jobs.driver_profiles_text_report",
                "--config",
                config_path,
            ],
            env,
        )

        profiles_csv = latest_file(base_dir, "driver_profiles.csv")
        ranking_csv = latest_file(base_dir, "driver_overall_ranking.csv")
        text_csv = latest_file(base_dir, "driver_profiles_text.csv")

        llm_csv = None
        merged_csv = None
        if payload.include_llm:
            llm_endpoint = payload.llm_endpoint or env.get(
                "MLFLOW_GATEWAY_ENDPOINT",
                "http://mlflow:5000/gateway/gemini/mlflow/invocations",
            )
            llm_output_dir = base_dir / "llm_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_cmd(
                [
                    "python",
                    "-m",
                    "jobs.generate_driver_performance_llm",
                    "--endpoint",
                    llm_endpoint,
                    "--ranking-csv",
                    str(ranking_csv),
                    "--profiles-text-csv",
                    str(text_csv),
                    "--output-dir",
                    str(llm_output_dir),
                ],
                env,
            )
            llm_csv = llm_output_dir / "driver_profiles_llm.csv"
            merged_csv = _merge_llm(ranking_csv, llm_csv, base_dir)
            mlflow_logger = MlflowClient(settings)
            if not mlflow_logger.enabled:
                raise HTTPException(
                    status_code=500,
                    detail="REGISTER_MLFLOW=false; MLflow e necessario para MinIO.",
                )
            llm_artifacts = [llm_csv, merged_csv, ranking_csv, text_csv, profiles_csv]
            mlflow_logger.log_run(
                run_name=(
                    f"driver_profiles_llm__season={payload.season}"
                    f"__meeting_key={payload.meeting_key}__session={payload.session_name}"
                ),
                params={
                    "season": payload.season,
                    "meeting_key": payload.meeting_key,
                    "session_name": payload.session_name,
                },
                metrics={},
                artifacts=llm_artifacts,
                tags=with_run_context(
                    {
                        "task": "driver_profiles_llm",
                        "season": str(payload.season),
                        "meeting_key": str(payload.meeting_key),
                        "session_name": str(payload.session_name),
                    }
                ),
            )

        tags = {
            "run_group": run_group,
            "season": str(payload.season),
            "meeting_key": str(payload.meeting_key),
            "session_name": str(payload.session_name),
        }
        ranking_run = find_latest_run(
            tracking_client,
            experiment_id,
            {**tags, "task": "driver_profiles_overall_ranking"},
        )
        text_run = find_latest_run(
            tracking_client,
            experiment_id,
            {**tags, "task": "driver_profiles_text_report"},
        )

        artifacts = {
            "driver_overall_ranking_csv": artifact_uri(ranking_run, "driver_overall_ranking.csv"),
            "driver_profiles_text_csv": artifact_uri(text_run, "driver_profiles_text.csv"),
        }
        if llm_csv:
            llm_run = find_latest_run(
                tracking_client,
                experiment_id,
                {**tags, "task": "driver_profiles_llm"},
            )
            artifacts["driver_profiles_llm_csv"] = artifact_uri(
                llm_run, "driver_profiles_llm.csv"
            )
        if merged_csv:
            llm_run = find_latest_run(
                tracking_client,
                experiment_id,
                {**tags, "task": "driver_profiles_llm"},
            )
            artifacts["driver_overall_ranking_llm_csv"] = artifact_uri(
                llm_run, "driver_overall_ranking_llm.csv"
            )

        if should_cleanup(env):
            cleanup_dirs = {profiles_csv.parent, ranking_csv.parent, text_csv.parent}
            if llm_csv:
                cleanup_dirs.add(llm_csv.parent)
            if merged_csv:
                cleanup_dirs.add(merged_csv.parent)
            cleanup_paths(cleanup_dirs)
        synced_dirs = sync_data_lake(data_dir, env)
        if synced_dirs and should_cleanup_data_lake(env):
            cleanup_paths([data_dir / subdir for subdir in synced_dirs.keys()])

        return DriverProfilesResponse(status="ok", artifacts=artifacts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/driver-profiles/season", response_model=SeasonProfilesResponse)
def generate_driver_profiles_by_season(
    payload: SeasonProfilesRequest,
) -> SeasonProfilesResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_dir = Path(env.get("ARTIFACTS_DIR", "/app/artifacts"))
    data_dir = Path(env.get("DATA_DIR", "/app/data"))
    base_dir = artifacts_dir / "modeling" / "driver_profiles"
    run_group = str(uuid.uuid4())

    seasons = sorted({int(s) for s in payload.seasons if str(s).strip()})
    session_names = _normalize_list(payload.session_names)
    if not seasons:
        raise HTTPException(status_code=400, detail="seasons deve ter ao menos 1 item.")
    if not settings.output.register_mlflow:
        raise HTTPException(
            status_code=500,
            detail="REGISTER_MLFLOW=false; MLflow/MinIO e necessario para este endpoint.",
        )

    try:
        tracking_uri = env.get("MLFLOW_TRACKING_URI") or settings.mlflow.tracking_uri or ""
        experiment_name = env.get("MLFLOW_EXPERIMENT") or settings.mlflow.experiment_name
        create_exp = env.get("MLFLOW_CREATE_EXPERIMENT", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        tracking_client, experiment_id = get_tracking_client(
            tracking_uri, experiment_name, create_if_missing=create_exp
        )
        env["MLFLOW_RUN_GROUP"] = run_group
        os.environ["MLFLOW_RUN_GROUP"] = run_group
        env["RUN_MEETING_KEY"] = ""
        os.environ["RUN_MEETING_KEY"] = ""

        downloaded_dirs = download_data_lake(data_dir, env, only_if_missing=True)

        required_columns = [
            "meeting_date_start",
            "weather_date",
            "track_temperature",
            "air_temperature",
            "circuit_speed_class",
        ]
        required_non_null = ["meeting_date_start"]
        normalized_sessions = {s.lower() for s in session_names}
        use_all_sessions = not normalized_sessions

        def _has_any_session_data(season: int) -> bool:
            if use_all_sessions or "all" in normalized_sessions:
                return has_data_for_filter(
                    data_dir,
                    season,
                    None,
                    "all",
                    required_columns=required_columns,
                    required_non_null=required_non_null,
                )
            for sess in session_names:
                if has_data_for_filter(
                    data_dir,
                    season,
                    None,
                    sess,
                    required_columns=required_columns,
                    required_non_null=required_non_null,
                ):
                    return True
            return False

        missing = [season for season in seasons if not _has_any_session_data(season)]
        if missing:
            missing_str = ", ".join(str(s) for s in missing)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Nenhum dado encontrado no gold para seasons: {missing_str}. "
                    "Execute a importacao da temporada para gerar os dados."
                ),
            )

        artifacts_by_season: dict[str, dict[str, str]] = {}
        summaries_by_season: dict[str, dict[str, object]] = {}
        top_by_season: dict[str, list[dict[str, object]]] = {}
        session_tag = ", ".join(session_names) if session_names else "all"

        for season in seasons:
            env["RUN_SEASON"] = str(season)
            env["RUN_SESSION_NAME"] = session_tag
            os.environ["RUN_SEASON"] = str(season)
            os.environ["RUN_SESSION_NAME"] = session_tag

            report_cmd = [
                "python",
                "-m",
                "jobs.driver_profiles_report",
                "--config",
                config_path,
                "--season",
                str(season),
            ]
            if session_names:
                report_cmd += ["--session-names", session_tag]
            else:
                report_cmd += ["--session-name", "all"]
            if payload.drivers_include:
                report_cmd += ["--drivers-include", ", ".join(payload.drivers_include)]
            if payload.drivers_exclude:
                report_cmd += ["--drivers-exclude", ", ".join(payload.drivers_exclude)]
            run_cmd(report_cmd, env)

            run_cmd(
                [
                    "python",
                    "-m",
                    "jobs.driver_profiles_overall_ranking",
                    "--config",
                    config_path,
                ],
                env,
            )

            run_cmd(
                [
                    "python",
                    "-m",
                    "jobs.driver_profiles_text_report",
                    "--config",
                    config_path,
                ],
                env,
            )

            profiles_csv = latest_file(base_dir, "driver_profiles.csv")
            ranking_csv = latest_file(base_dir, "driver_overall_ranking.csv")
            text_csv = latest_file(base_dir, "driver_profiles_text.csv")
            profiles_df = pd.read_csv(profiles_csv)
            ranking_df = pd.read_csv(ranking_csv)

            summary = {
                "drivers_total": int(profiles_df["driver_number"].nunique())
                if "driver_number" in profiles_df.columns
                else int(len(profiles_df)),
                "laps_total": int(profiles_df["laps_total"].sum())
                if "laps_total" in profiles_df.columns and not profiles_df["laps_total"].isna().all()
                else None,
                "meetings_total": int(profiles_df["meetings_total"].max())
                if "meetings_total" in profiles_df.columns and not profiles_df["meetings_total"].isna().all()
                else None,
                "sessions": session_names if session_names else ["all"],
                "season": season,
            }

            top_fields = [
                "driver_number",
                "driver_name",
                "team_name",
                "overall_score",
                "overall_rank",
                "lap_mean_delta_to_meeting_mean",
                "lap_std",
                "lap_quality_good_rate",
                "anomaly_rate",
                "stint_performance_delta_mean",
                "stint_performance_delta_slope",
                "tyre_wear_slope",
            ]
            available_fields = [c for c in top_fields if c in ranking_df.columns]
            top_df = ranking_df.sort_values("overall_rank").head(5)
            top_rows = top_df[available_fields].where(pd.notna(top_df), None).to_dict("records")

            llm_csv = None
            merged_csv = None
            if payload.include_llm:
                llm_endpoint = payload.llm_endpoint or env.get(
                    "MLFLOW_GATEWAY_ENDPOINT",
                    "http://mlflow:5000/gateway/gemini/mlflow/invocations",
                )
                llm_output_dir = (
                    base_dir / "llm_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                run_cmd(
                    [
                        "python",
                        "-m",
                        "jobs.generate_driver_performance_llm",
                        "--endpoint",
                        llm_endpoint,
                        "--ranking-csv",
                        str(ranking_csv),
                        "--profiles-text-csv",
                        str(text_csv),
                        "--output-dir",
                        str(llm_output_dir),
                    ],
                    env,
                )
                llm_csv = llm_output_dir / "driver_profiles_llm.csv"
                merged_csv = _merge_llm(ranking_csv, llm_csv, base_dir)
                mlflow_logger = MlflowClient(settings)
                if not mlflow_logger.enabled:
                    raise HTTPException(
                        status_code=500,
                        detail="REGISTER_MLFLOW=false; MLflow e necessario para MinIO.",
                    )
                llm_artifacts = [llm_csv, merged_csv, ranking_csv, text_csv, profiles_csv]
                mlflow_logger.log_run(
                    run_name=f"driver_profiles_llm__season={season}__sessions={session_tag}",
                    params={
                        "season": season,
                        "session_names": session_tag,
                    },
                    metrics={},
                    artifacts=llm_artifacts,
                    tags=with_run_context(
                        {
                            "task": "driver_profiles_llm",
                            "season": str(season),
                            "session_name": session_tag,
                        }
                    ),
                )

            tags = {
                "run_group": run_group,
                "season": str(season),
                "session_name": session_tag,
            }
            report_run = find_latest_run(
                tracking_client,
                experiment_id,
                {**tags, "task": "driver_profiles_report"},
            )
            ranking_run = find_latest_run(
                tracking_client,
                experiment_id,
                {**tags, "task": "driver_profiles_overall_ranking"},
            )
            text_run = find_latest_run(
                tracking_client,
                experiment_id,
                {**tags, "task": "driver_profiles_text_report"},
            )

            artifacts = {
                "driver_profiles_csv": artifact_uri(report_run, "driver_profiles.csv"),
                "driver_overall_ranking_csv": artifact_uri(
                    ranking_run, "driver_overall_ranking.csv"
                ),
                "driver_profiles_text_csv": artifact_uri(
                    text_run, "driver_profiles_text.csv"
                ),
            }
            if llm_csv:
                llm_run = find_latest_run(
                    tracking_client,
                    experiment_id,
                    {**tags, "task": "driver_profiles_llm"},
                )
                artifacts["driver_profiles_llm_csv"] = artifact_uri(
                    llm_run, "driver_profiles_llm.csv"
                )
            if merged_csv:
                llm_run = find_latest_run(
                    tracking_client,
                    experiment_id,
                    {**tags, "task": "driver_profiles_llm"},
                )
                artifacts["driver_overall_ranking_llm_csv"] = artifact_uri(
                    llm_run, "driver_overall_ranking_llm.csv"
                )

            artifacts_by_season[str(season)] = artifacts
            summaries_by_season[str(season)] = summary
            top_by_season[str(season)] = top_rows

            if should_cleanup(env):
                cleanup_dirs = {profiles_csv.parent, ranking_csv.parent, text_csv.parent}
                if llm_csv:
                    cleanup_dirs.add(llm_csv.parent)
                if merged_csv:
                    cleanup_dirs.add(merged_csv.parent)
                cleanup_paths(cleanup_dirs)

        if downloaded_dirs and should_cleanup_data_lake(env):
            cleanup_paths([data_dir / subdir for subdir in downloaded_dirs.keys()])

        return SeasonProfilesResponse(
            status="ok",
            seasons=seasons,
            session_names=session_names if session_names else ["all"],
            artifacts=artifacts_by_season,
            summaries=summaries_by_season,
            top_drivers=top_by_season,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _jobs_dir(env: dict[str, str], config_path: str) -> Path:
    settings = load_settings(config_path)
    default_dir = Path(settings.paths.logs_dir) / "jobs"
    jobs_dir = Path(env.get("JOBS_DIR", str(default_dir)))
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def _load_job_status(jobs_dir: Path, job_id: str) -> dict:
    status_file = jobs_dir / f"{job_id}.status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job nao encontrado.")
    return json.loads(status_file.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _spawn_job_process(
    job_file: Path,
    log_file: Path,
    env: dict[str, str],
    module: str = "jobs.import_season_job",
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("a", encoding="utf-8")
    cmd = [sys.executable, "-m", module, "--job-file", str(job_file)]
    kwargs: dict[str, object] = {
        "env": env,
        "stdout": log_handle,
        "stderr": log_handle,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)  # noqa: S603
    log_handle.close()


def _queue_ml_job(
    *,
    env: dict[str, str],
    config_path: str,
    job_type: str,
    module: str,
    args: list[str],
    params: dict[str, Any],
    artifacts_root: Path,
    filters: Optional[dict[str, Any]] = None,
) -> TrainJobResponse:
    jobs_dir = _jobs_dir(env, config_path)
    job_id = uuid.uuid4().hex
    job_file = jobs_dir / f"{job_id}.json"
    status_file = jobs_dir / f"{job_id}.status.json"
    log_file = jobs_dir / f"{job_id}.log"
    created_at = _now_iso_utc()

    job_payload: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "created_at": created_at,
        "status_file": str(status_file),
        "log_file": str(log_file),
        "config_path": config_path,
        "module": module,
        "args": args,
        "params": params,
        "artifacts_root": str(artifacts_root),
        "metrics_file": "metrics.json",
    }
    if filters is not None:
        job_payload["filters"] = filters

    status_payload: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "created_at": created_at,
        "params": params,
        "log_file": str(log_file),
        "status_file": str(status_file),
    }
    if filters is not None:
        status_payload["filters"] = filters

    _write_json_atomic(job_file, job_payload)
    _write_json_atomic(status_file, status_payload)
    _spawn_job_process(job_file, log_file, env, module="jobs.train_generic_job")

    return TrainJobResponse(status="queued", job_id=job_id)


def _tail_log(path: Path, lines: int) -> str:
    if lines <= 0:
        return ""
    block_size = 4096
    data = bytearray()
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        remaining = f.tell()
        while remaining > 0 and data.count(b"\n") <= lines:
            read_size = min(block_size, remaining)
            remaining -= read_size
            f.seek(remaining)
            data = f.read(read_size) + data
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _normalize_max_depth(max_depth: int | None) -> int | None:
    if max_depth is None:
        return None
    return max_depth if max_depth >= 1 else None


def _resolve_model_version(payload: BaseModel, prefix: str) -> str:
    provided = getattr(payload, "model_version", None)
    if provided:
        return str(provided)
    payload_dict = payload.dict()
    payload_dict.pop("model_version", None)
    raw = json.dumps(
        payload_dict,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


@app.post("/import-season", response_model=ImportSeasonJobResponse, status_code=202)
def import_season(payload: ImportSeasonRequest) -> ImportSeasonJobResponse:
    if payload.session_name.lower() not in {"race", "sprint"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race ou Sprint")

    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)

    job_id = uuid.uuid4().hex
    job_file = jobs_dir / f"{job_id}.json"
    status_file = jobs_dir / f"{job_id}.status.json"
    log_file = jobs_dir / f"{job_id}.log"
    created_at = _now_iso_utc()

    job_payload = {
        "job_id": job_id,
        "created_at": created_at,
        "status_file": str(status_file),
        "log_file": str(log_file),
        "season": payload.season,
        "session_name": payload.session_name,
        "include_llm": payload.include_llm,
        "llm_endpoint": payload.llm_endpoint,
        "config_path": config_path,
        "resume_job_id": payload.resume_job_id,
    }
    status_payload = {
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "season": payload.season,
        "session_name": payload.session_name,
        "include_llm": payload.include_llm,
        "log_file": str(log_file),
        "status_file": str(status_file),
        "resume_job_id": payload.resume_job_id,
    }

    _write_json_atomic(job_file, job_payload)
    _write_json_atomic(status_file, status_payload)
    _spawn_job_process(job_file, log_file, env)

    return ImportSeasonJobResponse(status="queued", job_id=job_id)


@app.post("/import-season/resume", response_model=ImportSeasonJobResponse, status_code=202)
def import_season_resume(payload: ImportSeasonResumeRequest) -> ImportSeasonJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)

    resume_status = _load_job_status(jobs_dir, payload.resume_job_id)
    resume_status_file = jobs_dir / f"{payload.resume_job_id}.status.json"
    season = resume_status.get("season")
    session_name = resume_status.get("session_name") or "Race"
    if not season:
        raise HTTPException(status_code=400, detail="Job anterior sem season valido.")
    if str(session_name).lower() not in {"race", "sprint"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race ou Sprint")

    include_llm = (
        payload.include_llm
        if payload.include_llm is not None
        else bool(resume_status.get("include_llm", True))
    )

    job_id = uuid.uuid4().hex
    job_file = jobs_dir / f"{job_id}.json"
    status_file = jobs_dir / f"{job_id}.status.json"
    log_file = jobs_dir / f"{job_id}.log"
    created_at = _now_iso_utc()

    job_payload = {
        "job_id": job_id,
        "created_at": created_at,
        "status_file": str(status_file),
        "log_file": str(log_file),
        "season": season,
        "session_name": session_name,
        "include_llm": include_llm,
        "llm_endpoint": payload.llm_endpoint,
        "config_path": config_path,
        "resume_job_id": payload.resume_job_id,
    }
    status_payload = {
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "season": season,
        "session_name": session_name,
        "include_llm": include_llm,
        "log_file": str(log_file),
        "status_file": str(status_file),
        "resume_job_id": payload.resume_job_id,
    }

    _write_json_atomic(job_file, job_payload)
    _write_json_atomic(status_file, status_payload)

    previous_status = str(resume_status.get("status") or "").lower()
    if previous_status not in {"completed", "failed"}:
        resume_status["status"] = "resumed"
        resume_status["resumed_job_id"] = job_id
        resume_status["finished_at"] = _now_iso_utc()
        resume_status["message"] = f"Retomado pelo job {job_id}."
        _write_json_atomic(resume_status_file, resume_status)

    _spawn_job_process(job_file, log_file, env)

    return ImportSeasonJobResponse(status="queued", job_id=job_id)


@app.post(
    "/train/stint-delta-pace",
    response_model=TrainStintDeltaPaceJobResponse,
    status_code=202,
    summary="Treino de delta de ritmo (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de delta de ritmo entre stints e "
        "retorna job_id para acompanhamento. Requer MLflow configurado. "
        "(Machine Learning)"
    ),
)
def train_stint_delta_pace_job(
    payload: TrainStintDeltaPaceRequest,
) -> TrainStintDeltaPaceJobResponse:
    if payload.session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")
    if payload.baseline_laps < 1:
        raise HTTPException(status_code=400, detail="baseline_laps deve ser >= 1")

    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)

    job_id = uuid.uuid4().hex
    job_file = jobs_dir / f"{job_id}.json"
    status_file = jobs_dir / f"{job_id}.status.json"
    log_file = jobs_dir / f"{job_id}.log"
    created_at = _now_iso_utc()

    max_depth = _normalize_max_depth(payload.max_depth)
    model_version = _resolve_model_version(payload, "stint_delta_pace")
    env["MODEL_VERSION"] = model_version
    job_payload = {
        "job_id": job_id,
        "created_at": created_at,
        "status_file": str(status_file),
        "log_file": str(log_file),
        "config_path": config_path,
        "filters": {
            "season": payload.season,
            "meeting_key": payload.meeting_key,
            "session_name": payload.session_name,
            "driver_number": payload.driver_number,
            "constructor": payload.constructor,
        },
        "params": {
            "target_mode": payload.target_mode,
            "baseline_laps": payload.baseline_laps,
            "group_col": payload.group_col,
            "test_size": payload.test_size,
            "random_state": payload.random_state,
            "n_estimators": payload.n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": payload.min_samples_leaf,
            "model_version": model_version,
        },
    }
    status_payload = {
        "job_id": job_id,
        "job_type": "train_stint_delta_pace",
        "status": "queued",
        "created_at": created_at,
        "filters": job_payload["filters"],
        "params": job_payload["params"],
        "log_file": str(log_file),
        "status_file": str(status_file),
    }

    _write_json_atomic(job_file, job_payload)
    _write_json_atomic(status_file, status_payload)
    _spawn_job_process(job_file, log_file, env, module="jobs.train_stint_delta_pace_job")

    return TrainStintDeltaPaceJobResponse(status="queued", job_id=job_id)


@app.post(
    "/train/lap-time-regression",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de lap time regression (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de regressao de tempo de volta. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_lap_time_regression_job(
    payload: TrainLapTimeRegressionRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "lap_time_regression"

    max_depth = _normalize_max_depth(payload.max_depth)
    model_version = _resolve_model_version(payload, "lap_time_regression")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--group-col",
        payload.group_col,
        "--test-size",
        str(payload.test_size),
        "--random-state",
        str(payload.random_state),
        "--n-estimators",
        str(payload.n_estimators),
        "--min-samples-leaf",
        str(payload.min_samples_leaf),
    ]
    if payload.include_sectors:
        args.append("--include-sectors")
    if max_depth is not None:
        args += ["--max-depth", str(max_depth)]

    params = payload.dict()
    params["max_depth"] = max_depth
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_lap_time_regression",
        module="jobs.train_lap_time_regression",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/lap-time-ranking",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de lap time ranking (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de ranking de lap time. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_lap_time_ranking_job(
    payload: TrainLapTimeRankingRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "lap_time_ranking"

    max_depth = _normalize_max_depth(payload.max_depth)
    model_version = _resolve_model_version(payload, "lap_time_ranking")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--group-col",
        payload.group_col,
        "--driver-col",
        payload.driver_col,
        "--test-size",
        str(payload.test_size),
        "--random-state",
        str(payload.random_state),
        "--n-estimators",
        str(payload.n_estimators),
        "--min-samples-leaf",
        str(payload.min_samples_leaf),
    ]
    if payload.include_sectors:
        args.append("--include-sectors")
    if max_depth is not None:
        args += ["--max-depth", str(max_depth)]

    params = payload.dict()
    params["max_depth"] = max_depth
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_lap_time_ranking",
        module="jobs.train_lap_time_ranking",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/relative-position",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de posicao relativa (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de posicao relativa. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_relative_position_job(
    payload: TrainRelativePositionRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "relative_position"

    max_depth = _normalize_max_depth(payload.max_depth)
    model_version = _resolve_model_version(payload, "relative_position")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--group-col",
        payload.group_col,
        "--test-size",
        str(payload.test_size),
        "--random-state",
        str(payload.random_state),
        "--n-estimators",
        str(payload.n_estimators),
        "--min-samples-leaf",
        str(payload.min_samples_leaf),
    ]
    if max_depth is not None:
        args += ["--max-depth", str(max_depth)]

    params = payload.dict()
    params["max_depth"] = max_depth
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_relative_position",
        module="jobs.train_relative_position",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/tyre-degradation",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de degradacao de pneus (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de degradacao de pneus. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_tyre_degradation_job(
    payload: TrainTyreDegradationRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "tyre_degradation"

    max_depth = _normalize_max_depth(payload.max_depth)
    model_version = _resolve_model_version(payload, "tyre_degradation")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--group-col",
        payload.group_col,
        "--test-size",
        str(payload.test_size),
        "--random-state",
        str(payload.random_state),
        "--n-estimators",
        str(payload.n_estimators),
        "--min-samples-leaf",
        str(payload.min_samples_leaf),
    ]
    if payload.include_sectors:
        args.append("--include-sectors")
    if max_depth is not None:
        args += ["--max-depth", str(max_depth)]

    params = payload.dict()
    params["max_depth"] = max_depth
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_tyre_degradation",
        module="jobs.train_tyre_degradation",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/lap-quality-classifier",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de qualidade de volta (Machine Learning)",
    description=(
        "Dispara treino assincrono do classificador de qualidade de volta. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_lap_quality_classifier_job(
    payload: TrainLapQualityClassifierRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "lap_quality_classifier"

    model_version = _resolve_model_version(payload, "lap_quality_classifier")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--group-col",
        payload.group_col,
        "--test-size",
        str(payload.test_size),
        "--random-state",
        str(payload.random_state),
        "--n-estimators",
        str(payload.n_estimators),
    ]
    if payload.include_sectors:
        args.append("--include-sectors")

    params = payload.dict()
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_lap_quality_classifier",
        module="jobs.train_lap_quality_classifier",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/lap-anomaly",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de anomalias de volta (Machine Learning)",
    description=(
        "Dispara treino assincrono do modelo de anomalias por volta. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_lap_anomaly_job(
    payload: TrainLapAnomalyRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "lap_anomaly"

    model_version = _resolve_model_version(payload, "lap_anomaly")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--contamination",
        str(payload.contamination),
        "--n-estimators",
        str(payload.n_estimators),
        "--random-state",
        str(payload.random_state),
    ]

    params = payload.dict()
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_lap_anomaly",
        module="jobs.train_lap_anomaly",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/driver-style-clustering",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de clustering de estilo (Machine Learning)",
    description=(
        "Dispara treino assincrono de clustering de estilo de pilotagem. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_driver_style_clustering_job(
    payload: TrainDriverStyleClusteringRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "driver_style_clustering"

    model_version = _resolve_model_version(payload, "driver_style_clustering")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--clusters",
        str(payload.clusters),
        "--random-state",
        str(payload.random_state),
    ]

    params = payload.dict()
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_driver_style_clustering",
        module="jobs.train_driver_style_clustering",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.post(
    "/train/circuit-segmentation",
    response_model=TrainJobResponse,
    status_code=202,
    summary="Treino de segmentacao de circuitos (Machine Learning)",
    description=(
        "Dispara treino assincrono de segmentacao de circuitos. "
        "Requer MLflow configurado. (Machine Learning)"
    ),
)
def train_circuit_segmentation_job(
    payload: TrainCircuitSegmentationRequest,
) -> TrainJobResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    settings = load_settings(config_path)
    artifacts_root = Path(settings.paths.artifacts_dir) / "modeling" / "circuit_segmentation"

    model_version = _resolve_model_version(payload, "circuit_segmentation")
    env["MODEL_VERSION"] = model_version
    args = [
        "--config",
        config_path,
        "--clusters",
        str(payload.clusters),
        "--random-state",
        str(payload.random_state),
    ]

    params = payload.dict()
    params["model_version"] = model_version

    return _queue_ml_job(
        env=env,
        config_path=config_path,
        job_type="train_circuit_segmentation",
        module="jobs.train_circuit_segmentation",
        args=args,
        params=params,
        artifacts_root=artifacts_root,
    )


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)
    status_file = jobs_dir / f"{job_id}.status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job nao encontrado.")
    return json.loads(status_file.read_text(encoding="utf-8"))


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, lines: int = 200) -> dict:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    jobs_dir = _jobs_dir(env, config_path)
    log_file = jobs_dir / f"{job_id}.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log nao encontrado.")
    safe_lines = max(1, min(int(lines), 1000))
    return {
        "job_id": job_id,
        "lines": safe_lines,
        "log": _tail_log(log_file, safe_lines),
    }
