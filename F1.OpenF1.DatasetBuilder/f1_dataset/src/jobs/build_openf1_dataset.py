from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from config.settings import ensure_paths, load_settings
from orchestration.artifacts_cleanup import cleanup_directory_contents, should_cleanup
from orchestration.data_lake_sync import should_cleanup_data_lake, sync_data_lake
from orchestration.runner import run_pipeline


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenF1 dataset pipeline")
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    ensure_paths(settings)
    _setup_logging(settings.paths.logs_dir)

    logging.getLogger(__name__).info("Iniciando pipeline com config: %s", config_path)
    logging.getLogger(__name__).info(
        "Meetings filter: mode=%s include=%s",
        settings.meetings.mode,
        settings.meetings.include,
    )
    run_pipeline(settings)
    if should_cleanup():
        cleanup_directory_contents(Path(settings.paths.artifacts_dir))
    synced_dirs = sync_data_lake(Path(settings.paths.data_dir))
    if synced_dirs and should_cleanup_data_lake():
        for subdir in synced_dirs.keys():
            cleanup_directory_contents(Path(settings.paths.data_dir) / subdir)


if __name__ == "__main__":
    main()
