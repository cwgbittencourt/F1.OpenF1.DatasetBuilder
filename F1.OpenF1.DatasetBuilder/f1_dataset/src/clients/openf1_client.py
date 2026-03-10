from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from config.settings import Settings


@dataclass
class RateLimiter:
    min_interval_seconds: float
    _lock: Lock = Lock()
    _last_request_time: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            self._last_request_time = time.monotonic()


class OpenF1Client:
    def __init__(self, settings: Settings, rate_limiter: RateLimiter | None = None) -> None:
        self.base_url = settings.api.base_url.rstrip("/")
        self.retry_attempts = settings.execution.retry_attempts
        self.retry_backoff_seconds = settings.execution.retry_backoff_seconds
        self.rate_limit_cooldown_seconds = settings.execution.rate_limit_cooldown_seconds
        self.rate_limiter = rate_limiter or RateLimiter(settings.execution.min_request_interval_ms / 1000.0)
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=settings.execution.max_http_connections,
                              pool_maxsize=settings.execution.max_http_connections)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.logger = logging.getLogger(__name__)

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(1, self.retry_attempts + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 429:
                    self.logger.warning("Rate limit detectado. Cooldown de %s segundos.", self.rate_limit_cooldown_seconds)
                    time.sleep(self.rate_limit_cooldown_seconds)
                    continue
                if response.status_code == 404:
                    self.logger.warning("Endpoint sem dados (404): %s params=%s", url, params)
                    return []
                if response.status_code >= 500:
                    self._backoff(attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                self.logger.warning("Falha na requisicao %s (tentativa %s/%s): %s", url, attempt, self.retry_attempts, exc)
                self._backoff(attempt)
        raise RuntimeError(f"Falha apos {self.retry_attempts} tentativas em {url}")

    def _backoff(self, attempt: int) -> None:
        sleep_for = self.retry_backoff_seconds * (2 ** (attempt - 1))
        time.sleep(sleep_for)
