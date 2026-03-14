from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestration.import_utils import run_cmd


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


def _latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _init_status(job: dict[str, Any], status: str) -> dict[str, Any]:
    payload = {
        "job_id": job["job_id"],
        "job_type": job.get("job_type", "train_generic"),
        "status": status,
        "created_at": job.get("created_at"),
        "started_at": None,
        "finished_at": None,
        "params": job.get("params", {}),
        "artifacts_dir": None,
        "metrics": None,
        "message": None,
        "log_file": job.get("log_file"),
        "status_file": job.get("status_file"),
    }
    if "filters" in job:
        payload["filters"] = job.get("filters", {})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa um treino ML via job generico.")
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

    try:
        module = job.get("module")
        if not module:
            raise RuntimeError("Job sem module definido.")
        args_list = job.get("args", [])
        cmd = ["python", "-m", module, *args_list]

        run_cmd(cmd, env)

        artifacts_root = job.get("artifacts_root")
        metrics_file = job.get("metrics_file", "metrics.json")
        if artifacts_root:
            latest = _latest_dir(Path(artifacts_root))
            status_payload["artifacts_dir"] = str(latest) if latest else None
            metrics_path = latest / metrics_file if latest and metrics_file else None
            if metrics_path and metrics_path.exists():
                status_payload["metrics"] = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )

        status_payload["status"] = "completed"
        status_payload["finished_at"] = _now_iso()
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
