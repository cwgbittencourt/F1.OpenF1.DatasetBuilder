from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import difflib

import pandas as pd
from fastapi import FastAPI, HTTPException
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


class ImportSeasonJobResponse(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None


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


def _value_counts(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return []
    counts = cleaned.value_counts().head(limit)
    return [{"value": key, "count": int(value)} for key, value in counts.items()]


def _date_range(series: pd.Series) -> Optional[dict[str, str]]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return None
    return {
        "min": parsed.min().isoformat(),
        "max": parsed.max().isoformat(),
    }

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
    if "lap_duration" in df.columns:
        numeric_summary["lap_duration_seconds"] = _numeric_stats(
            _duration_seconds(df["lap_duration"])
        )
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.post("/perguntas-gold", response_model=GoldQuestionsResponse)
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
    llm_payload = _build_gold_prompt(question, summary)

    env = os.environ.copy()
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
    def _parse_answer(raw: str) -> str:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"].strip()

    try:
        answer = _parse_answer(content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Resposta invalida do LLM: {exc}") from exc

    if not _is_probably_portuguese(answer):
        retry_payload = _build_gold_prompt(question, summary, strict_portuguese=True)
        status, content = _post_json(llm_endpoint, retry_payload, timeout_s=60)
        if status == 200:
            try:
                answer = _parse_answer(content)
            except Exception:
                pass

    return GoldQuestionsResponse(status="ok", answer=answer, summary=summary)


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


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _spawn_job_process(job_file: Path, log_file: Path, env: dict[str, str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("a", encoding="utf-8")
    cmd = [sys.executable, "-m", "jobs.import_season_job", "--job-file", str(job_file)]
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
    created_at = datetime.now().isoformat()

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
    }

    _write_json_atomic(job_file, job_payload)
    _write_json_atomic(status_file, status_payload)
    _spawn_job_process(job_file, log_file, env)

    return ImportSeasonJobResponse(status="queued", job_id=job_id)


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
