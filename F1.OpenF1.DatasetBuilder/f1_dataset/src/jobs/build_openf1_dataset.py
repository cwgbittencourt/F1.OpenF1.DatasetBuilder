from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from config.settings import ensure_paths, load_settings
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
    run_pipeline(settings)


if __name__ == "__main__":
    main()
