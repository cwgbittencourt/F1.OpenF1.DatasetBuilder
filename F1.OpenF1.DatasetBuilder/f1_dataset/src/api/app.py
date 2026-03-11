from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import load_settings
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/driver-profiles", response_model=DriverProfilesResponse)
def generate_driver_profiles(payload: DriverProfilesRequest) -> DriverProfilesResponse:
    env = os.environ.copy()
    config_path = env.get("CONFIG_PATH", "/app/config/config.yaml")
    artifacts_dir = Path(env.get("ARTIFACTS_DIR", "/app/artifacts"))
    data_dir = Path(env.get("DATA_DIR", "/app/data"))
    base_dir = artifacts_dir / "modeling" / "driver_profiles"

    if not str(payload.meeting_key).strip():
        raise HTTPException(status_code=400, detail="meeting_key e obrigatorio.")
    if payload.session_name.lower() not in {"race", "sprint", "all"}:
        raise HTTPException(status_code=400, detail="session_name deve ser Race, Sprint ou all")

    try:
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

        artifacts = {
            "driver_overall_ranking_csv": str(ranking_csv),
            "driver_profiles_text_csv": str(text_csv),
        }
        if llm_csv:
            artifacts["driver_profiles_llm_csv"] = str(llm_csv)
        if merged_csv:
            artifacts["driver_overall_ranking_llm_csv"] = str(merged_csv)

        return DriverProfilesResponse(status="ok", artifacts=artifacts)
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
