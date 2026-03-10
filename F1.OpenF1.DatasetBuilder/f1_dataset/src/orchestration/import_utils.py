from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def run_cmd(args: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def latest_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} nao encontrado em {root}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def has_data_for_filter(data_dir: Path, season: int, meeting_key: str, session_name: str) -> bool:
    consolidated = data_dir / "gold" / "consolidated.parquet"
    if not consolidated.exists():
        return False
    df = pd.read_parquet(consolidated, columns=["season", "meeting_key", "session_name"])
    df = df[df["season"] == season]
    df = df[df["meeting_key"].astype(str) == str(meeting_key)]
    if session_name.lower() != "all":
        df = df[df["session_name"].astype(str).str.lower() == session_name.lower()]
    return not df.empty


def write_temp_config(
    base_config_path: str,
    season: int,
    meeting_key: str,
    session_name: str,
    temp_dir: Path,
) -> Path:
    base: dict[str, Any] = {}
    base_path = Path(base_config_path)
    if base_path.exists():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    base["seasons"] = [season]
    base["session_name"] = session_name
    base.setdefault("meetings", {})
    base["meetings"]["mode"] = "by_key"
    base["meetings"]["include"] = [str(meeting_key)]

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"request_{season}_{meeting_key}_{session_name.lower()}.yaml"
    temp_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return temp_path


def ensure_data(
    env: dict[str, str],
    season: int,
    meeting_key: str,
    session_name: str,
    config_path: str,
    data_dir: Path,
) -> None:
    if has_data_for_filter(data_dir, season, meeting_key, session_name):
        return

    temp_dir = Path(env.get("CONFIG_DIR", "/app/config")) / "requests"

    def run_pipeline_for(name: str) -> None:
        temp_config = write_temp_config(config_path, season, meeting_key, name, temp_dir)
        run_cmd(
            [
                "python",
                "-m",
                "jobs.build_openf1_dataset",
                "--config",
                str(temp_config),
            ],
            env,
        )

    if session_name.lower() == "all":
        for name in ["Race", "Sprint"]:
            if not has_data_for_filter(data_dir, season, meeting_key, name):
                run_pipeline_for(name)
    else:
        run_pipeline_for(session_name)

    run_cmd(
        [
            "python",
            "-m",
            "jobs.consolidate_gold_dataset",
            "--output",
            str(data_dir / "gold" / "consolidated.parquet"),
        ],
        env,
    )


def meeting_start_date(meeting: dict[str, Any]) -> datetime:
    raw = meeting.get("date_start") or meeting.get("meeting_start") or meeting.get("meeting_date")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.max
