"""Tests for Stage 1: VLM Morphological Extraction."""

import pytest
from unittest.mock import MagicMock, patch

from seedlearn.pipeline.config import VLMConfig
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.stages.morphology import MorphologyStage


class TestMorphologyStage:
    def test_name(self):
        stage = MorphologyStage(config=VLMConfig())
        assert stage.name == "morphology"

    def test_validate_input_requires_image_paths(self):
        stage = MorphologyStage(config=VLMConfig())
        errors = stage.validate_input({})
        assert any("image_paths" in e for e in errors)

    def test_validate_input_ok(self):
        stage = MorphologyStage(config=VLMConfig())
        errors = stage.validate_input({"image_paths": ["/a.jpg"]})
        assert errors == []

    def test_skip_returns_empty(self):
        stage = MorphologyStage(config=VLMConfig())
        result = stage.skip({})
        assert result.skipped is True
        assert result.data == {}

    @patch("seedlearn.pipeline.stages.morphology.InferenceClient")
    def test_run_calls_client(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.chat.return_value = MagicMock(
            content=(
                "1. Leaf relative position: alternate\n"
                "2. Leaf spacing: clustered"
            ),
            raw_content=(
                "1. Leaf relative position: alternate\n"
                "2. Leaf spacing: clustered"
            ),
            thinking=None,
            processing_time_ms=150.0,
        )

        stage = MorphologyStage(config=VLMConfig())
        stage.client = mock_instance

        result = stage.run({"image_paths": ["/img1.jpg", "/img2.jpg"]})
        assert result.stage_name == "morphology"
        assert result.skipped is False
        assert "traits" in result.data
        assert "raw_response" in result.data
        mock_instance.chat.assert_called_once()

    @patch("seedlearn.pipeline.stages.morphology.InferenceClient")
    def test_run_handles_client_error(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.chat.side_effect = Exception("Connection refused")

        stage = MorphologyStage(config=VLMConfig())
        stage.client = mock_instance

        result = stage.run({"image_paths": ["/img.jpg"]})
        assert result.error is not None
        assert "Connection refused" in result.error

    @patch("seedlearn.pipeline.stages.morphology.InferenceClient")
    def test_run_json_mode(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.chat.return_value = MagicMock(
            content='{"leaf_arrangement": {"relative_position": "alternate"}}',
            raw_content='{"leaf_arrangement": {"relative_position": "alternate"}}',
            thinking=None,
            processing_time_ms=100.0,
        )

        stage = MorphologyStage(config=VLMConfig(prompt_style="json"))
        stage.client = mock_instance

        result = stage.run({"image_paths": ["/img.jpg"]})
        assert isinstance(result.data["traits"], dict)
        assert "leaf_arrangement" in result.data["traits"]
