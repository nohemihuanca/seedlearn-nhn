"""Tests for pipeline CLI argument parsing."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    """Ensure scripts/ is importable."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    yield
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)


class TestCLIParsing:
    def test_images_required_or_specimen(self):
        from run_pipeline import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_images_explicit(self):
        from run_pipeline import parse_args

        args = parse_args(["--images", "/a.jpg", "/b.jpg"])
        assert args.images == ["/a.jpg", "/b.jpg"]

    def test_specimen_arg(self):
        from run_pipeline import parse_args

        args = parse_args(["--specimen", "PP123"])
        assert args.specimen == "PP123"

    def test_config_override(self):
        from run_pipeline import parse_args

        args = parse_args(["--images", "/a.jpg", "--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_vlm_overrides(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--vlm-model", "custom/model",
            "--prompt-style", "sys2",
        ])
        assert args.vlm_model == "custom/model"
        assert args.prompt_style == "sys2"

    def test_classifier_overrides(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--rank", "species",
            "--k-shot", "20",
        ])
        assert args.rank == "species"
        assert args.k_shot == 20

    def test_skip_stages(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--skip", "trait_retrieval", "reasoning",
        ])
        assert args.skip == ["trait_retrieval", "reasoning"]

    def test_build_overrides_from_args(self):
        from run_pipeline import build_overrides, parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--vlm-model", "test/model",
            "--rank", "genus",
            "--device", "cpu",
        ])
        overrides = build_overrides(args)
        assert overrides["vlm.model"] == "test/model"
        assert overrides["classifier.rank"] == "genus"
        assert overrides["classifier.device"] == "cpu"

    def test_build_overrides_empty(self):
        from run_pipeline import build_overrides, parse_args

        args = parse_args(["--images", "/a.jpg"])
        overrides = build_overrides(args)
        assert overrides == {}

    def test_cache_dir_arg(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--cache-dir", "/path/to/cache",
        ])
        assert args.cache_dir == "/path/to/cache"

    def test_split_path_arg(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--split-path", "/path/to/split_seed42",
        ])
        assert args.split_path == "/path/to/split_seed42"

    def test_rag_index_arg(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--rag-index", "/path/to/rag_index",
        ])
        assert args.rag_index == "/path/to/rag_index"

    def test_catalog_arg(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--specimen", "PP123",
            "--catalog", "/path/to/catalog.csv",
        ])
        assert args.catalog == "/path/to/catalog.csv"
        assert args.specimen == "PP123"

    def test_all_data_artifacts_together(self):
        from run_pipeline import parse_args

        args = parse_args([
            "--images", "/a.jpg",
            "--cache-dir", "/cache",
            "--split-path", "/split",
            "--rag-index", "/rag",
        ])
        assert args.cache_dir == "/cache"
        assert args.split_path == "/split"
        assert args.rag_index == "/rag"
