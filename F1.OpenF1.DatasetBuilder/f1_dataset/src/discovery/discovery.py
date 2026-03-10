from __future__ import annotations

from datetime import datetime
from typing import Any

from clients.openf1_client import OpenF1Client
from config.settings import MeetingsFilter


def get_meetings_for_season(client: OpenF1Client, season: int) -> list[dict[str, Any]]:
    return client.get("meetings", params={"year": season})


def select_meetings(meetings: list[dict[str, Any]], filter_cfg: MeetingsFilter) -> list[dict[str, Any]]:
    mode = (filter_cfg.mode or "all").lower()
    if not meetings:
        return []

    if mode == "first_of_season":
        return sorted(meetings, key=_meeting_start_date)

    if mode == "by_key":
        allowed = {str(item) for item in filter_cfg.include}
        return [m for m in meetings if str(m.get("meeting_key")) in allowed]

    if mode == "by_name":
        allowed = {str(item).lower() for item in filter_cfg.include}
        return [m for m in meetings if str(m.get("meeting_name", "")).lower() in allowed]

    return meetings


def get_sessions_for_meeting(client: OpenF1Client, meeting_key: int | str) -> list[dict[str, Any]]:
    return client.get("sessions", params={"meeting_key": meeting_key})


def select_session(sessions: list[dict[str, Any]], session_name: str) -> dict[str, Any] | None:
    name = session_name.lower()
    for session in sessions:
        if str(session.get("session_name", "")).lower() == name:
            return session
    return None


def get_drivers_for_session(client: OpenF1Client, session_key: int | str) -> list[dict[str, Any]]:
    return client.get("drivers", params={"session_key": session_key})


def _meeting_start_date(meeting: dict[str, Any]) -> datetime:
    raw = meeting.get("date_start") or meeting.get("meeting_start") or meeting.get("meeting_date")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.max
