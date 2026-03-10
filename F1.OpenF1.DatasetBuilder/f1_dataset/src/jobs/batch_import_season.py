from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from clients.openf1_client import OpenF1Client, RateLimiter
from config.settings import load_settings


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _meeting_start_date(meeting: dict[str, Any]) -> datetime:
    raw = meeting.get("date_start") or meeting.get("meeting_start") or meeting.get("meeting_date")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.max


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("batch-size precisa ser > 0")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _write_config(base: dict[str, Any], season: int, meeting_keys: list[str], output_path: Path) -> None:
    cfg = dict(base)
    cfg["seasons"] = [season]
    cfg.setdefault("meetings", {})
    cfg["meetings"]["mode"] = "by_key"
    cfg["meetings"]["include"] = [str(k) for k in meeting_keys]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"batches": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"batches": {}}


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_pipeline(config_path: Path, log_path: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if ".\\f1_dataset\\src" not in existing and "./f1_dataset/src" not in existing:
        env["PYTHONPATH"] = ".\\f1_dataset\\src"
    cmd = ["python", "-m", "jobs.build_openf1_dataset", "--config", str(config_path)]
    logging.getLogger(__name__).info("Executando: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=log_file, text=True)


def _write_progress_summary(state_path: Path, summary_path: Path) -> None:
    state = _load_state(state_path)
    batches = state.get("batches", {})
    rows = []
    for batch_id, info in batches.items():
        rows.append(
            {
                "batch_id": batch_id,
                "status": info.get("status"),
                "started_at": info.get("started_at"),
                "finished_at": info.get("finished_at"),
                "config": info.get("config"),
                "meetings": ",".join([str(m) for m in info.get("meetings", [])]),
                "log": info.get("log"),
                "error": info.get("error"),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "batch_id",
        "status",
        "started_at",
        "finished_at",
        "config",
        "meetings",
        "log",
        "error",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join([str(row.get(col, "")) for col in header]))
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa temporada por batches de meetings."
    )
    parser.add_argument("--config", default="./config/config_2024.yaml")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--output-dir", default="./config/batches")
    parser.add_argument("--run", action="store_true", help="Executa o pipeline para cada batch.")
    parser.add_argument(
        "--sleep-between",
        type=int,
        default=0,
        help="Pausa (segundos) entre batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula batches marcados como completed no state file.",
    )
    args = parser.parse_args()

    _setup_logging()

    base_path = Path(args.config)
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    settings = load_settings(str(base_path))

    rate_limiter = RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
    client = OpenF1Client(settings, rate_limiter=rate_limiter)
    meetings = client.get("meetings", params={"year": args.season})
    meetings = sorted(meetings, key=_meeting_start_date)
    meeting_keys = [str(m.get("meeting_key")) for m in meetings if m.get("meeting_key") is not None]

    if not meeting_keys:
        raise RuntimeError(f"Nenhum meeting encontrado para season={args.season}")

    batches = _chunk(meeting_keys, args.batch_size)
    output_root = Path(args.output_dir) / f"season_{args.season}"
    state_path = output_root / "batches_state.json"
    state = _load_state(state_path)

    for idx, batch in enumerate(batches, start=1):
        output_path = output_root / f"batch_{idx:02d}.yaml"
        _write_config(base_cfg, args.season, batch, output_path)
        logging.getLogger(__name__).info(
            "Config batch %s criado com %s meetings: %s",
            output_path,
            len(batch),
            ", ".join(batch),
        )
        if args.run:
            batch_id = f"batch_{idx:02d}"
            batch_state = state.setdefault("batches", {}).get(batch_id, {})
            if args.resume and batch_state.get("status") == "completed":
                logging.getLogger(__name__).info(
                    "Batch %s ja concluido. Pulando.", batch_id
                )
                continue

            log_path = output_root / "logs" / f"{batch_id}.log"
            state["batches"][batch_id] = {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "config": str(output_path),
                "meetings": batch,
                "log": str(log_path),
            }
            _save_state(state_path, state)
            try:
                _run_pipeline(output_path, log_path)
                state["batches"][batch_id]["status"] = "completed"
                state["batches"][batch_id]["finished_at"] = datetime.now().isoformat()
            except subprocess.CalledProcessError as exc:
                state["batches"][batch_id]["status"] = "failed"
                state["batches"][batch_id]["finished_at"] = datetime.now().isoformat()
                state["batches"][batch_id]["error"] = str(exc)
                _save_state(state_path, state)
                _write_progress_summary(state_path, output_root / "batches_progress.csv")
                raise
            _save_state(state_path, state)
            _write_progress_summary(state_path, output_root / "batches_progress.csv")

            if args.sleep_between > 0:
                logging.getLogger(__name__).info(
                    "Pausando %s segundos antes do proximo batch.",
                    args.sleep_between,
                )
                time.sleep(args.sleep_between)

    _write_progress_summary(state_path, output_root / "batches_progress.csv")


if __name__ == "__main__":
    main()
