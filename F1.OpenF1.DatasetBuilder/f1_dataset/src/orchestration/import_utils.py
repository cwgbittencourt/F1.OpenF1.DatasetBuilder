from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from orchestration.data_lake_sync import download_data_lake, should_download_data_lake

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


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set()


def _clear_checkpoints(checkpoints_dir: Path, season: int, meeting_key: str | None) -> int:
    if not meeting_key or not checkpoints_dir.exists():
        return 0
    pattern = f"season={season}__meeting_key={meeting_key}__*.json"
    removed = 0
    for path in checkpoints_dir.glob(pattern):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    if removed:
        logging.getLogger(__name__).info(
            "Checkpoints removidos: %s (season=%s meeting_key=%s)",
            removed,
            season,
            meeting_key,
        )
    return removed


def has_data_for_filter(
    data_dir: Path,
    season: int,
    meeting_key: str | None,
    session_name: str,
    required_columns: list[str] | None = None,
    required_non_null: list[str] | None = None,
) -> bool:
    consolidated = data_dir / "gold" / "consolidated.parquet"
    if not consolidated.exists():
        return False
    required_columns = required_columns or []
    required_non_null = required_non_null or []
    base_cols = ["season", "meeting_key", "session_name"]
    available_cols = _parquet_columns(consolidated)
    if available_cols:
        missing = [col for col in required_columns if col not in available_cols]
        if missing:
            return False
        read_cols = list(dict.fromkeys(base_cols + required_columns + required_non_null))
    else:
        read_cols = list(dict.fromkeys(base_cols + required_columns + required_non_null))
    try:
        df = pd.read_parquet(consolidated, columns=read_cols)
    except Exception:
        return False
    df = df[df["season"] == season]
    if meeting_key:
        df = df[df["meeting_key"].astype(str) == str(meeting_key)]
    if session_name.lower() != "all":
        df = df[df["session_name"].astype(str).str.lower() == session_name.lower()]
    if df.empty:
        return False
    for col in required_columns:
        if col not in df.columns:
            return False
    for col in required_non_null:
        if col not in df.columns or df[col].dropna().empty:
            return False
    return True


def write_temp_config(
    base_config_path: str,
    season: int,
    meeting_key: str | None,
    session_name: str,
    temp_dir: Path,
    drivers_include: list[str] | None = None,
    drivers_exclude: list[str] | None = None,
) -> Path:
    base: dict[str, Any] = {}
    base_path = Path(base_config_path)
    if base_path.exists():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    base["seasons"] = [season]
    base["session_name"] = session_name
    if drivers_include is not None or drivers_exclude is not None:
        base.setdefault("drivers", {})
        if drivers_include is not None:
            base["drivers"]["include"] = drivers_include
        if drivers_exclude is not None:
            base["drivers"]["exclude"] = drivers_exclude
    base.setdefault("meetings", {})
    if meeting_key:
        base["meetings"]["mode"] = "by_key"
        base["meetings"]["include"] = [str(meeting_key)]
    else:
        base["meetings"]["mode"] = "all"
        base["meetings"]["include"] = []

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"request_{season}_{meeting_key}_{session_name.lower()}.yaml"
    temp_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    logging.getLogger(__name__).info(
        "Config temporaria: meetings.mode=%s include=%s session=%s season=%s",
        base["meetings"]["mode"],
        base["meetings"]["include"],
        session_name,
        season,
    )
    return temp_path


def ensure_data(
    env: dict[str, str],
    season: int,
    meeting_key: str | None,
    session_name: str,
    config_path: str,
    data_dir: Path,
    required_columns: list[str] | None = None,
    required_non_null: list[str] | None = None,
    drivers_include: list[str] | None = None,
    drivers_exclude: list[str] | None = None,
) -> None:
    if should_download_data_lake(env):
        download_data_lake(data_dir, env, only_if_missing=True)
    if required_columns is None:
        required_columns = [
            "meeting_date_start",
            "weather_date",
            "track_temperature",
            "air_temperature",
            "circuit_speed_class",
        ]
    if required_non_null is None:
        required_non_null = ["meeting_date_start"]

    if has_data_for_filter(
        data_dir,
        season,
        meeting_key,
        session_name,
        required_columns=required_columns,
        required_non_null=required_non_null,
    ):
        return

    temp_dir = Path(env.get("CONFIG_DIR", "/app/config")) / "requests"
    checkpoints_dir = Path(env.get("CHECKPOINT_DIR", str(Path(data_dir) / "checkpoints")))
    _clear_checkpoints(checkpoints_dir, season, meeting_key)

    def run_pipeline_for(name: str) -> None:
        temp_config = write_temp_config(
            config_path,
            season,
            meeting_key,
            name,
            temp_dir,
            drivers_include=drivers_include,
            drivers_exclude=drivers_exclude,
        )
        run_env = env.copy()
        run_env["SYNC_DATA_LAKE"] = "false"
        run_env["CLEANUP_LOCAL_DATA"] = "false"
        run_cmd(
            [
                "python",
                "-m",
                "jobs.build_openf1_dataset",
                "--config",
                str(temp_config),
            ],
            run_env,
        )

    if session_name.lower() == "all":
        for name in ["Race", "Sprint"]:
            if not has_data_for_filter(
                data_dir,
                season,
                meeting_key,
                name,
                required_columns=required_columns,
                required_non_null=required_non_null,
            ):
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
