from __future__ import annotations

from typing import Any

from clients.openf1_client import OpenF1Client


def collect_laps(client: OpenF1Client, session_key: int | str, driver_number: int | str) -> list[dict[str, Any]]:
    return client.get("laps", params={"session_key": session_key, "driver_number": driver_number})


def collect_car_data(client: OpenF1Client, session_key: int | str, driver_number: int | str) -> list[dict[str, Any]]:
    return client.get("car_data", params={"session_key": session_key, "driver_number": driver_number})


def collect_location(client: OpenF1Client, session_key: int | str, driver_number: int | str) -> list[dict[str, Any]]:
    return client.get("location", params={"session_key": session_key, "driver_number": driver_number})


def collect_stints(client: OpenF1Client, session_key: int | str, driver_number: int | str) -> list[dict[str, Any]]:
    return client.get("stints", params={"session_key": session_key, "driver_number": driver_number})
