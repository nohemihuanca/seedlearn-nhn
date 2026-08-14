"""Tests for size-controlled thumbnail embedding (human.thumbnails)."""

import base64
import io

import pytest

from seedlearn.benchmarking.human import thumbnails as T

Image = pytest.importorskip("PIL.Image")


def _make_image(path, size=(500, 300), color=(120, 180, 90)):
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_thumbnail_data_uri_downscales_longest_edge(tmp_path):
    img = tmp_path / "v.jpg"
    _make_image(img, size=(500, 300))
    uri = T.thumbnail_data_uri(img, max_edge=160)
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        assert max(im.size) <= 160
        assert im.size == (160, 96)  # aspect preserved (500x300 -> 160x96)


def test_thumbnail_data_uri_missing_file_returns_none(tmp_path):
    assert T.thumbnail_data_uri(tmp_path / "nope.jpg") is None


def test_specimen_thumbnails_maps_and_totals(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _make_image(a)
    _make_image(b)
    result, total = T.specimen_thumbnails(
        {"SR1": [str(a), str(b)], "SR2": [str(tmp_path / "missing.jpg")]}
    )
    assert len(result["SR1"]) == 2
    assert result["SR2"] == []       # unreadable skipped, not fatal
    assert total > 0


def test_specimen_thumbnails_budget_is_soft(tmp_path, caplog):
    a = tmp_path / "a.jpg"
    _make_image(a)
    result, total = T.specimen_thumbnails({"SR1": [str(a)]}, budget_bytes=1)
    assert result["SR1"]  # still returned despite tiny budget
    assert total > 1


def test_load_specimen_image_paths(tmp_path):
    (tmp_path / "SR1.json").write_text('{"specimen_id": "SR1", "image_paths": ["/x/1.jpg", "/x/2.jpg"]}')
    (tmp_path / "run_metadata.json").write_text("{}")  # skipped
    paths = T.load_specimen_image_paths(tmp_path)
    assert paths == {"SR1": ["/x/1.jpg", "/x/2.jpg"]}
