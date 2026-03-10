from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from clients.openf1_client import OpenF1Client, RateLimiter
from clients.mlflow_client import MlflowClient
from collectors.collectors import collect_laps, collect_car_data, collect_location, collect_stints
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
from publishers.data_writer import write_bronze, write_silver, write_gold
from publishers.mlflow_publisher import MlflowPublisher
from validators.quality import validate_lap_dataset
from orchestration.checkpoint_store import CheckpointStore


@dataclass(frozen=True)
class ProcessingUnit:
    season: int
    meeting_key: int | str
    meeting_name: str
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


def run_pipeline(settings: Settings) -> None:
    logger = logging.getLogger(__name__)
    rate_limiter = RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
    discovery_client = OpenF1Client(settings, rate_limiter=rate_limiter)
    units = build_processing_units(settings, discovery_client)

    checkpoint_store = CheckpointStore(settings.paths.checkpoints_dir)
    mlflow_client = MlflowClient(settings)
    mlflow_publisher = MlflowPublisher(mlflow_client)

    if settings.execution.max_parallel_drivers <= 1:
        for unit in units:
            _process_unit(unit, settings, checkpoint_store, mlflow_publisher, rate_limiter)
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

        write_bronze(laps_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "laps")
        write_bronze(car_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "car_data")
        write_bronze(loc_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "location")
        write_bronze(stints_raw, Path(settings.paths.data_dir) / "bronze", unit.as_dict(), "stints")

        laps_df = normalize_frame(laps_raw)
        car_df = normalize_frame(car_raw)
        loc_df = normalize_frame(loc_raw)
        stints_df = normalize_frame(stints_raw)

        write_silver(laps_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "laps")
        write_silver(car_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "car_data")
        write_silver(loc_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "location")
        write_silver(stints_df, Path(settings.paths.data_dir) / "silver", unit.as_dict(), "stints")

        context = LapContext(
            season=unit.season,
            meeting_key=unit.meeting_key,
            meeting_name=unit.meeting_name,
            session_key=unit.session_key,
            session_name=unit.session_name,
            driver_number=unit.driver_number,
            driver_name=unit.driver_name,
            team_name=unit.team_name,
        )
        gold_df = build_lap_dataset(laps_df, car_df, loc_df, context, stints_df)
        metrics = validate_lap_dataset(gold_df)

        gold_artifacts = write_gold(
            gold_df,
            Path(settings.paths.data_dir) / "gold",
            unit.as_dict(),
            settings.output.formats,
        )

        report_path = _write_quality_report(metrics, settings, unit)
        artifacts = gold_artifacts + [report_path]

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
