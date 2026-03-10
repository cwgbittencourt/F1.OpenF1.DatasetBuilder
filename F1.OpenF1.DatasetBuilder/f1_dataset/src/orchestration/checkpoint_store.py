from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    status: str
    updated_at: str
    payload: dict[str, Any]


class CheckpointStore:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _file_path(self, unit_id: str) -> Path:
        return self.directory / f"{unit_id}.json"

    def get(self, unit_id: str) -> Checkpoint | None:
        path = self._file_path(unit_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint(**data)

    def set(self, unit_id: str, status: str, payload: dict[str, Any]) -> None:
        path = self._file_path(unit_id)
        checkpoint = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }
        path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
