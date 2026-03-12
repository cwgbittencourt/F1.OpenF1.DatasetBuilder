from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Iterable


def should_cleanup(env: dict[str, str] | None = None) -> bool:
    source = env or os.environ
    raw = source.get("CLEANUP_LOCAL_ARTIFACTS", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cleanup_paths(paths: Iterable[Path]) -> None:
    logger = logging.getLogger(__name__)
    for path in {p for p in paths if p}:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Falha ao limpar artefato %s: %s", path, exc)


def cleanup_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        cleanup_paths([child])
