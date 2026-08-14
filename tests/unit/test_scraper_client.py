"""Tests for STRI scraper HTTP client."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seedlearn.scraper.client import STRIClient


class TestSTRIClient:
    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        client = STRIClient(cache_dir=tmp_path / "cache")
        assert (tmp_path / "cache").is_dir()

    def test_cached_response_skips_http(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "test_page.html"
        cached_file.write_text("<html>cached</html>")

        client = STRIClient(cache_dir=cache_dir)
        result = client.fetch_cached("https://example.com", "test_page.html")
        assert result == "<html>cached</html>"

    @patch("seedlearn.scraper.client.requests.Session.get")
    def test_fetch_stores_to_cache(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>fresh</html>"
        mock_get.return_value = mock_response

        client = STRIClient(cache_dir=tmp_path / "cache", delay_seconds=0.0)
        result = client.fetch_cached("https://example.com", "new_page.html")
        assert result == "<html>fresh</html>"
        assert (tmp_path / "cache" / "new_page.html").read_text() == "<html>fresh</html>"

    @patch("seedlearn.scraper.client.requests.Session.get")
    def test_fetch_respects_delay(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response

        client = STRIClient(cache_dir=tmp_path, delay_seconds=0.1)
        start = time.monotonic()
        client.fetch_cached("https://a.com", "a.html")
        client.fetch_cached("https://b.com", "b.html")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.1  # At least one delay between requests

    @patch("seedlearn.scraper.client.requests.Session.get")
    def test_fetch_with_force_refresh(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "old.html").write_text("<html>old</html>")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>new</html>"
        mock_get.return_value = mock_response

        client = STRIClient(cache_dir=cache_dir, delay_seconds=0.0)
        result = client.fetch_cached(
            "https://example.com", "old.html", force_refresh=True,
        )
        assert result == "<html>new</html>"
        assert (cache_dir / "old.html").read_text() == "<html>new</html>"

    @patch("seedlearn.scraper.client.requests.Session.get")
    def test_user_agent_set(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response

        client = STRIClient(cache_dir=tmp_path, delay_seconds=0.0)
        client.fetch_cached("https://example.com", "test.html")

        # Session should have User-Agent header
        assert "SeedLearn" in client._session.headers.get("User-Agent", "")
