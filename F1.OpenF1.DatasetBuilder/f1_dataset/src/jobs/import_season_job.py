from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clients.mlflow_artifacts import artifact_uri, find_latest_run, get_tracking_client
from clients.mlflow_client import MlflowClient
from clients.mlflow_tags import with_run_context
from clients.openf1_client import OpenF1Client, RateLimiter
from config.settings import load_settings
from discovery.discovery import get_meetings_for_season, get_sessions_for_meeting, select_session
from orchestration.artifacts_cleanup import cleanup_paths, should_cleanup
from orchestration.data_lake_sync import should_cleanup_data_lake, sync_data_lake
from orchestration.import_utils import (
    ensure_data,
    has_data_for_filter,
    latest_file,
    meeting_start_date,
    run_cmd,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _init_status(job: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": status,
        "created_at": job.get("created_at"),
        "started_at": None,
        "finished_at": None,
        "season": job.get("season"),
        "session_name": job.get("session_name"),
        "include_llm": job.get("include_llm"),
        "total_meetings": 0,
        "processed_meetings": 0,
        "current_meeting": None,
        "meetings": [],
        "log_file": job.get("log_file"),
        "status_file": job.get("status_file"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Importar temporada via job.")
    parser.add_argument("--job-file", required=True, help="Caminho para o JSON do job.")
    args = parser.parse_args()

    job_file = Path(args.job_file)
    if not job_file.exists():
        raise SystemExit(f"Job file nao encontrado: {job_file}")

    job = json.loads(job_file.read_text(encoding="utf-8"))
    status_file = Path(job["status_file"])
    log_file = Path(job["log_file"])
    _setup_logging(log_file)
    logger = logging.getLogger(__name__)

    status_payload = _init_status(job, "running")
    status_payload["started_at"] = _now_iso()
    _write_json_atomic(status_file, status_payload)

    env = os.environ.copy()
    config_path = job.get("config_path") or env.get("CONFIG_PATH") or "./config/config.yaml"
    season = int(job.get("season"))
    session_name = str(job.get("session_name") or "Race")
    include_llm = bool(job.get("include_llm", True))
    llm_endpoint = job.get("llm_endpoint")

    try:
        settings = load_settings(config_path)
        if not settings.output.register_mlflow:
            raise RuntimeError("REGISTER_MLFLOW=false; MLflow/MinIO e necessario para este job.")
        rate_limiter = RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
        client = OpenF1Client(settings, rate_limiter=rate_limiter)
        mlflow_client = MlflowClient(settings)
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

        artifacts_dir = Path(env.get("ARTIFACTS_DIR", settings.paths.artifacts_dir))
        data_dir = Path(env.get("DATA_DIR", settings.paths.data_dir))
        base_dir = artifacts_dir / "modeling" / "driver_profiles"
        run_group = env.get("MLFLOW_RUN_GROUP") or str(job.get("job_id"))

        meetings = get_meetings_for_season(client, season)
        meetings = sorted(meetings, key=meeting_start_date)
        status_payload["total_meetings"] = len(meetings)
        _write_json_atomic(status_file, status_payload)

        if not meetings:
            status_payload["status"] = "failed"
            status_payload["finished_at"] = _now_iso()
            status_payload["message"] = "Nenhum meeting encontrado para a temporada."
            _write_json_atomic(status_file, status_payload)
            return

        for meeting in meetings:
            meeting_key = meeting.get("meeting_key")
            meeting_name = meeting.get("meeting_name", "")
            env["MLFLOW_RUN_GROUP"] = run_group
            env["RUN_SEASON"] = str(season)
            env["RUN_MEETING_KEY"] = str(meeting_key)
            env["RUN_SESSION_NAME"] = str(session_name)
            os.environ["MLFLOW_RUN_GROUP"] = run_group
            os.environ["RUN_SEASON"] = str(season)
            os.environ["RUN_MEETING_KEY"] = str(meeting_key)
            os.environ["RUN_SESSION_NAME"] = str(session_name)
            status_payload["current_meeting"] = {
                "meeting_key": str(meeting_key),
                "meeting_name": str(meeting_name),
            }
            _write_json_atomic(status_file, status_payload)

            session_key = None
            try:
                sessions = get_sessions_for_meeting(client, meeting_key)
                session = select_session(sessions, session_name)
                if not session:
                    status_payload["meetings"].append(
                        {
                            "meeting_key": str(meeting_key),
                            "meeting_name": str(meeting_name),
                            "status": "skipped",
                            "message": "Sessao nao encontrada para o meeting.",
                        }
                    )
                    _write_json_atomic(status_file, status_payload)
                    continue

                session_key = session.get("session_key")
                ensure_data(
                    env,
                    season,
                    meeting_key,
                    session_name,
                    config_path,
                    data_dir,
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
                    season,
                    meeting_key,
                    session_name,
                    required_columns=required_columns,
                    required_non_null=required_non_null,
                ):
                    status_payload["meetings"].append(
                        {
                            "meeting_key": str(meeting_key),
                            "meeting_name": str(meeting_name),
                            "session_key": str(session_key),
                            "status": "empty",
                            "message": "Nenhum dado encontrado apos importacao.",
                        }
                    )
                    _write_json_atomic(status_file, status_payload)
                    continue

                run_cmd(
                    [
                        "python",
                        "-m",
                        "jobs.driver_profiles_report",
                        "--config",
                        config_path,
                        "--season",
                        str(season),
                        "--meeting-key",
                        str(meeting_key),
                        "--session-name",
                        session_name,
                    ],
                    env,
                )

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
                profiles_csv = latest_file(base_dir, "driver_profiles.csv")

                llm_csv = None
                if include_llm:
                    resolved_endpoint = llm_endpoint or env.get(
                        "MLFLOW_GATEWAY_ENDPOINT",
                        "http://mlflow:5000/gateway/gemini/mlflow/invocations",
                    )
                    output_dir = (
                        base_dir
                        / "llm_reports"
                        / f"season{season}_meeting{meeting_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    run_cmd(
                        [
                            "python",
                            "-m",
                            "jobs.generate_driver_performance_llm",
                            "--endpoint",
                            resolved_endpoint,
                            "--ranking-csv",
                            str(ranking_csv),
                            "--profiles-text-csv",
                            str(text_csv),
                            "--output-dir",
                            str(output_dir),
                        ],
                        env,
                    )
                    llm_csv = output_dir / "driver_profiles_llm.csv"
                    if llm_csv.exists():
                        llm_artifacts = [
                            llm_csv,
                            profiles_csv,
                            ranking_csv,
                            text_csv,
                        ]
                        mlflow_client.log_run(
                            run_name=(
                                f"driver_profiles_llm__season={season}"
                                f"__meeting_key={meeting_key}__session={session_name}"
                            ),
                            params={
                                "season": season,
                                "meeting_key": meeting_key,
                                "meeting_name": meeting_name,
                                "session_name": session_name,
                            },
                            metrics={},
                            artifacts=llm_artifacts,
                            tags=with_run_context(
                                {
                                    "task": "driver_profiles_llm",
                                    "season": str(season),
                                    "meeting_key": str(meeting_key),
                                    "meeting_name": str(meeting_name),
                                    "session_name": str(session_name),
                                }
                            ),
                        )

                artifacts = {}
                tags = {
                    "run_group": run_group,
                    "season": str(season),
                    "meeting_key": str(meeting_key),
                    "session_name": str(session_name),
                }
                report_run = find_latest_run(
                    tracking_client,
                    experiment_id,
                    {**tags, "task": "driver_profiles_report"},
                )
                artifacts["driver_profiles_csv"] = artifact_uri(report_run, "driver_profiles.csv")
                ranking_run = find_latest_run(
                    tracking_client,
                    experiment_id,
                    {**tags, "task": "driver_profiles_overall_ranking"},
                )
                artifacts["driver_overall_ranking_csv"] = artifact_uri(
                    ranking_run, "driver_overall_ranking.csv"
                )
                text_run = find_latest_run(
                    tracking_client,
                    experiment_id,
                    {**tags, "task": "driver_profiles_text_report"},
                )
                artifacts["driver_profiles_text_csv"] = artifact_uri(
                    text_run, "driver_profiles_text.csv"
                )
                if llm_csv and llm_csv.exists():
                    llm_run = find_latest_run(
                        tracking_client,
                        experiment_id,
                        {**tags, "task": "driver_profiles_llm"},
                    )
                    artifacts["driver_profiles_llm_csv"] = artifact_uri(
                        llm_run, "driver_profiles_llm.csv"
                    )

                status_payload["meetings"].append(
                    {
                        "meeting_key": str(meeting_key),
                        "meeting_name": str(meeting_name),
                        "session_key": str(session_key),
                        "status": "ok",
                        "artifacts": artifacts,
                    }
                )
                if should_cleanup(env):
                    cleanup_dirs = {
                        profiles_csv.parent,
                        ranking_csv.parent,
                        text_csv.parent,
                    }
                    if llm_csv and llm_csv.exists():
                        cleanup_dirs.add(llm_csv.parent)
                    cleanup_paths(cleanup_dirs)
            except Exception as exc:
                logger.exception("Falha no meeting %s: %s", meeting_key, exc)
                status_payload["meetings"].append(
                    {
                        "meeting_key": str(meeting_key),
                        "meeting_name": str(meeting_name),
                        "session_key": str(session_key) if session_key is not None else None,
                        "status": "error",
                        "message": str(exc),
                    }
                )
            finally:
                status_payload["processed_meetings"] += 1
                _write_json_atomic(status_file, status_payload)

        status_payload["status"] = "completed"
        status_payload["finished_at"] = _now_iso()
        status_payload["current_meeting"] = None
        synced_dirs = sync_data_lake(data_dir, env)
        if synced_dirs and should_cleanup_data_lake(env):
            cleanup_paths([data_dir / subdir for subdir in synced_dirs.keys()])
        _write_json_atomic(status_file, status_payload)
        logger.info("Job concluido: %s", job.get("job_id"))
    except Exception as exc:
        logger.exception("Job falhou: %s", exc)
        status_payload["status"] = "failed"
        status_payload["finished_at"] = _now_iso()
        status_payload["message"] = str(exc)
        _write_json_atomic(status_file, status_payload)
        raise


if __name__ == "__main__":
    main()
