from __future__ import annotations

import os
from typing import Any


def with_run_context(tags: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(tags)
    run_group = os.getenv("MLFLOW_RUN_GROUP")
    if run_group:
        enriched["run_group"] = run_group
    season = os.getenv("RUN_SEASON")
    if season:
        enriched["season"] = str(season)
    meeting_key = os.getenv("RUN_MEETING_KEY")
    if meeting_key:
        enriched["meeting_key"] = str(meeting_key)
    session_name = os.getenv("RUN_SESSION_NAME")
    if session_name:
        enriched["session_name"] = str(session_name)
    return enriched
