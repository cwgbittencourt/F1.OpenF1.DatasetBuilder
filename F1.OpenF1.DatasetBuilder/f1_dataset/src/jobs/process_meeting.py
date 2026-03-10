from __future__ import annotations

import argparse
import logging
import os

from config.settings import ensure_paths, load_settings
from orchestration.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Processar um meeting especifico")
    parser.add_argument("--config", help="Caminho do arquivo de configuracao")
    parser.add_argument("--meeting-key", required=True, help="Meeting key alvo")
    args = parser.parse_args()

    config_path = args.config or os.getenv("CONFIG_PATH") or "./config/config.yaml"
    settings = load_settings(config_path)
    settings.meetings.mode = "by_key"
    settings.meetings.include = [args.meeting_key]
    ensure_paths(settings)

    logging.basicConfig(level=logging.INFO)
    run_pipeline(settings)


if __name__ == "__main__":
    main()
