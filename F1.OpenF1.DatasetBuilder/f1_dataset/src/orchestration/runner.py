from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd

from clients.openf1_client import OpenF1Client, RateLimiter
from clients.mlflow_client import MlflowClient
from collectors.collectors import (
    collect_car_data,
    collect_laps,
    collect_location,
    collect_stints,
    collect_weather,
)
from config.settings import Settings
from discovery.discovery import (
    get_drivers_for_session,
    get_meetings_for_season,
    get_sessions_for_meeting,
    select_meetings,
    select_session,
)
from feature_engineering.lap_features import LapContext, build_lap_dataset
from processors.normalize import normalize_frame
from publishers.data_writer import (
    write_bronze,
    write_bronze_session,
    write_gold,
    write_silver,
    write_silver_session,
)
from publishers.mlflow_publisher import MlflowPublisher
from validators.quality import validate_lap_dataset
from orchestration.checkpoint_store import CheckpointStore
from orchestration.artifacts_cleanup import cleanup_paths, should_cleanup


@dataclass(frozen=True)
class ProcessingUnit:
    season: int
    meeting_key: int | str
    meeting_name: str
    meeting_date_start: str | None
    session_key: int | str
    session_name: str
    driver_number: int | str
    driver_name: str
    team_name: str

    def unit_id(self) -> str:
        return (
            f"season={self.season}__meeting_key={self.meeting_key}__"
            f"session_key={self.session_key}__driver_number={self.driver_number}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "meeting_key": self.meeting_key,
            "meeting_name": self.meeting_name,
            "meeting_date_start": self.meeting_date_start,
            "session_key": self.session_key,
            "session_name": self.session_name,
            "driver_number": self.driver_number,
            "driver_name": self.driver_name,
            "team_name": self.team_name,
        }


def build_processing_units(settings: Settings, client: OpenF1Client) -> list[ProcessingUnit]:
    units: list[ProcessingUnit] = []
    for season in settings.seasons:
        meetings = get_meetings_for_season(client, season)
        selected_meetings = select_meetings(meetings, settings.meetings)
        for meeting in selected_meetings:
            meeting_key = meeting.get("meeting_key")
            meeting_name = meeting.get("meeting_name", "")
            sessions = get_sessions_for_meeting(client, meeting_key)
            session = select_session(sessions, settings.session_name)
            if not session:
                continue
            meeting_date_start = _parse_meeting_date(meeting, session)
            session_key = session.get("session_key")
            session_name = session.get("session_name", "")
            drivers = get_drivers_for_session(client, session_key)
            filtered = _filter_drivers(drivers, settings)
            for driver in filtered:
                units.append(
                    ProcessingUnit(
                        season=season,
                        meeting_key=meeting_key,
                        meeting_name=meeting_name,
                        meeting_date_start=meeting_date_start,
                        session_key=session_key,
                        session_name=session_name,
                        driver_number=driver.get("driver_number"),
                        driver_name=driver.get("full_name") or driver.get("name", ""),
                        team_name=driver.get("team_name", ""),
                    )
                )
            if (settings.meetings.mode or "").lower() == "first_of_season":
                break
    return units


def _filter_drivers(drivers: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    include = {name.lower() for name in settings.drivers.include}
    exclude = {name.lower() for name in settings.drivers.exclude}
    if not include and not exclude:
        return drivers

    filtered = []
    for driver in drivers:
        name = (driver.get("full_name") or driver.get("name") or "").lower()
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        filtered.append(driver)
    return filtered


def _parse_meeting_date(meeting: dict[str, Any], session: dict[str, Any] | None) -> str | None:
    raw = meeting.get("date_start") or meeting.get("meeting_start") or meeting.get("meeting_date")
    if not raw and session:
        raw = session.get("date_start") or session.get("session_start")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).isoformat()
    except ValueError:
        return str(raw)


def _weather_session_path(settings: Settings, unit: ProcessingUnit) -> Path:
    base_dir = Path(settings.paths.data_dir) / "silver"
    path = (
        base_dir
        / f"season={unit.season}"
        / f"meeting_key={unit.meeting_key}"
        / f"session_key={unit.session_key}"
    )
    return path / "weather.parquet"


def _load_weather_from_disk(settings: Settings, unit: ProcessingUnit) -> pd.DataFrame | None:
    path = _weather_session_path(settings, unit)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _ensure_weather_bronze(unit: ProcessingUnit, settings: Settings, weather_df: pd.DataFrame) -> None:
    if weather_df.empty:
        return
    base_dir = Path(settings.paths.data_dir) / "bronze"
    path = (
        base_dir
        / f"season={unit.season}"
        / f"meeting_key={unit.meeting_key}"
        / f"session_key={unit.session_key}"
        / "weather.json"
    )
    if path.exists():
        return
    write_bronze_session(weather_df.to_dict(orient="records"), base_dir, unit.as_dict(), "weather")


def _ensure_weather_silver(unit: ProcessingUnit, settings: Settings, weather_df: pd.DataFrame) -> None:
    if weather_df.empty:
        return
    path = _weather_session_path(settings, unit)
    if path.exists():
        return
    write_silver_session(weather_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "weather")


def _get_weather_for_session(
    unit: ProcessingUnit,
    settings: Settings,
    client: OpenF1Client,
    weather_cache: dict[str, pd.DataFrame],
    weather_events: dict[str, threading.Event],
    weather_lock: threading.Lock,
) -> pd.DataFrame:
    cache_key = f"{unit.season}:{unit.meeting_key}:{unit.session_key}"

    with weather_lock:
        cached = weather_cache.get(cache_key)
        if cached is not None:
            return cached
        event = weather_events.get(cache_key)
        if event is None:
            event = threading.Event()
            weather_events[cache_key] = event
            is_fetcher = True
        else:
            is_fetcher = False

    if not is_fetcher:
        event.wait()
        with weather_lock:
            return weather_cache.get(cache_key, pd.DataFrame())

    try:
        weather_df = _load_weather_from_disk(settings, unit)
        if weather_df is None:
            weather_raw = collect_weather(client, unit.session_key)
            weather_df = normalize_frame(weather_raw)
        weather_cache[cache_key] = weather_df
    finally:
        with weather_lock:
            event.set()
            weather_events.pop(cache_key, None)

    return weather_cache.get(cache_key, pd.DataFrame())


def run_pipeline(settings: Settings) -> None:
    logger = logging.getLogger(__name__)
    rate_limiter = RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
    discovery_client = OpenF1Client(settings, rate_limiter=rate_limiter)
    units = build_processing_units(settings, discovery_client)

    checkpoint_store = CheckpointStore(settings.paths.checkpoints_dir)
    mlflow_client = MlflowClient(settings)
    mlflow_publisher = MlflowPublisher(mlflow_client)
    weather_cache: dict[str, pd.DataFrame] = {}
    weather_events: dict[str, threading.Event] = {}
    weather_lock = threading.Lock()

    if settings.execution.max_parallel_drivers <= 1:
        for unit in units:
            _process_unit(
                unit,
                settings,
                checkpoint_store,
                mlflow_publisher,
                rate_limiter,
                weather_cache,
                weather_events,
                weather_lock,
            )
    else:
        with ThreadPoolExecutor(max_workers=settings.execution.max_parallel_drivers) as executor:
            futures = [
                executor.submit(
                    _process_unit,
                    unit,
                    settings,
                    checkpoint_store,
                    mlflow_publisher,
                    rate_limiter,
                    weather_cache,
                    weather_events,
                    weather_lock,
                )
                for unit in units
            ]
            for future in as_completed(futures):
                future.result()

    logger.info("Pipeline concluido. Unidades processadas: %s", len(units))


def _process_unit(
    unit: ProcessingUnit,
    settings: Settings,
    checkpoint_store: CheckpointStore,
    mlflow_publisher: MlflowPublisher,
    rate_limiter: RateLimiter,
    weather_cache: dict[str, pd.DataFrame],
    weather_events: dict[str, threading.Event],
    weather_lock: threading.Lock,
) -> None:
    logger = logging.getLogger(__name__)
    unit_id = unit.unit_id()
    existing = checkpoint_store.get(unit_id)
    if existing and existing.status == "completed":
        logger.info("Checkpoint encontrado. Pulando %s", unit_id)
        return

    checkpoint_store.set(unit_id, "running", unit.as_dict())
    try:
        client = OpenF1Client(settings, rate_limiter=rate_limiter)
        laps_raw = collect_laps(client, unit.session_key, unit.driver_number)
        car_raw = collect_car_data(client, unit.session_key, unit.driver_number)
        loc_raw = collect_location(client, unit.session_key, unit.driver_number)
        stints_raw = collect_stints(client, unit.session_key, unit.driver_number)
        weather_df = _get_weather_for_session(
            unit,
            settings,
            client,
            weather_cache,
            weather_events,
            weather_lock,
        )

        write_bronze(laps_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "laps")
        write_bronze(car_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "car_data")
        write_bronze(loc_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "location")
        write_bronze(stints_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "stints")
        _ensure_weather_bronze(unit, settings, weather_df)

        laps_df = normalize_frame(laps_raw)
        car_df = normalize_frame(car_raw)
        loc_df = normalize_frame(loc_raw)
        stints_df = normalize_frame(stints_raw)

        write_silver(laps_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "laps")
        write_silver(car_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "car_data")
        write_silver(loc_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "location")
        write_silver(stints_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "stints")
        _ensure_weather_silver(unit, settings, weather_df)

        context = LapContext(
            season=unit.season,
            meeting_key=unit.meeting_key,
            meeting_name=unit.meeting_name,
            meeting_date_start=unit.meeting_date_start,
            session_key=unit.session_key,
            session_name=unit.session_name,
            driver_number=unit.driver_number,
            driver_name=unit.driver_name,
            team_name=unit.team_name,
        )
        gold_df = build_lap_dataset(laps_df, car_df, loc_df, context, stints_df, weather_df)
        metrics = validate_lap_dataset(gold_df)

        gold_artifacts = write_gold(
            gold_df,
            Path(settings.paths.data_dir) / "gold",
            unit.as_dict(),
            settings.output.formats,
        )

        report_path = _write_quality_report(metrics, settings, unit)
        schema_path = _write_schema_report(gold_df, settings, unit)
        artifacts = gold_artifacts + [report_path, schema_path]

        mlflow_publisher.publish(
            run_name=unit_id,
            params={
                "season": unit.season,
                "meeting_key": unit.meeting_key,
                "session_key": unit.session_key,
                "driver_number": unit.driver_number,
                "driver_name": unit.driver_name,
            },
            metrics=metrics,
            artifacts=artifacts,
        )

        if should_cleanup():
            cleanup_paths([report_path, schema_path])

        checkpoint_store.set(unit_id, "completed", unit.as_dict())
    except Exception as exc:
        logger.exception("Falha ao processar %s: %s", unit_id, exc)
        checkpoint_store.set(unit_id, "failed", {"error": str(exc), **unit.as_dict()})
        raise


def _write_quality_report(metrics: dict[str, float], settings: Settings, unit: ProcessingUnit) -> Path:
    path = Path(settings.paths.artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"quality_{unit.unit_id()}.json"
    file_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def _write_schema_report(df: pd.DataFrame, settings: Settings, unit: ProcessingUnit) -> Path:
    path = Path(settings.paths.artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    schema = {
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
        ],
        "rows": int(len(df)),
    }
    file_path = path / f"schema_{unit.unit_id()}.json"
    file_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path
