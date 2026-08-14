"""HTTP client for STRI Panama Biota portal with caching and rate limiting."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = (
    "SeedLearn-TraitScraper/1.0 "
    "(academic research; https://github.com/mitchellxh/seedlearn)"
)


class STRIClient:
    """HTTP client with local HTML caching and polite rate limiting.

    Caches every response as a local HTML file so re-runs never re-fetch.
    Enforces a minimum delay between HTTP requests to be polite to the
    academic server.

    Args:
        cache_dir: Directory for cached HTML responses.
        delay_seconds: Minimum seconds between HTTP requests.
        timeout_seconds: Request timeout.
    """

    def __init__(
        self,
        cache_dir: Path,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _wait_for_rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def fetch_cached(
        self,
        url: str,
        cache_filename: str,
        force_refresh: bool = False,
    ) -> str:
        """Fetch URL with local file caching.

        Args:
            url: URL to fetch.
            cache_filename: Filename within cache_dir to store response.
            force_refresh: If True, ignore cache and re-fetch.

        Returns:
            HTML response text.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        cache_path = self.cache_dir / cache_filename
        if cache_path.exists() and not force_refresh:
            logger.debug("Cache hit: %s", cache_filename)
            return cache_path.read_text(encoding="utf-8")

        self._wait_for_rate_limit()
        logger.info("Fetching: %s", url)
        response = self._session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        self._last_request_time = time.monotonic()

        cache_path.write_text(response.text, encoding="utf-8")
        logger.debug("Cached: %s", cache_filename)
        return response.text
