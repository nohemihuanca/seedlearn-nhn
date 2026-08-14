"""Tests for Stage 5: LLM Reasoning & Classification."""

import json
from unittest.mock import MagicMock

import pytest

from seedlearn.pipeline.config import ReasoningConfig
from seedlearn.pipeline.stages.reasoning import ReasoningStage


class TestReasoningStage:
    """Unit tests for ReasoningStage."""

    def test_name(self) -> None:
        stage = ReasoningStage(config=ReasoningConfig())
        assert stage.name == "reasoning"

    def test_validate_requires_evidence(self) -> None:
        stage = ReasoningStage(config=ReasoningConfig())
        errors = stage.validate_input({})
        assert any("evidence" in e.lower() for e in errors)

    def test_validate_rejects_empty_evidence(self) -> None:
        stage = ReasoningStage(config=ReasoningConfig())
        errors = stage.validate_input(
            {"evidence_synthesis": {"evidence_document": ""}}
        )
        assert len(errors) == 1

    def test_validate_ok_with_evidence(self) -> None:
        stage = ReasoningStage(config=ReasoningConfig())
        errors = stage.validate_input(
            {"evidence_synthesis": {"evidence_document": "test doc"}}
        )
        assert errors == []

    def test_skip(self) -> None:
        stage = ReasoningStage(config=ReasoningConfig())
        result = stage.skip({})
        assert result.skipped is True
        assert result.stage_name == "reasoning"

    def test_run_parses_json_response(self) -> None:
        response_data = {
            "predicted_family": "Fabaceae",
            "predicted_genus": "Inga",
            "predicted_species": "Inga edulis",
            "confidence": "high",
            "reasoning": "Compound leaves with pulvinus strongly suggest Fabaceae.",
            "supporting_features": ["compound leaves", "pulvinus"],
            "alternatives": [
                {"taxon": "Meliaceae", "reason": "similar leaves but no pulvinus"}
            ],
        }
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content=json.dumps(response_data),
            raw_content=json.dumps(response_data),
            thinking=None,
            processing_time_ms=500.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        context = {
            "evidence_synthesis": {
                "evidence_document": "Morphological Profile: compound leaves..."
            },
        }
        result = stage.run(context)

        assert result.error is None
        assert result.stage_name == "reasoning"
        assert "classification" in result.data
        clf = result.data["classification"]
        assert clf["predicted_family"] == "Fabaceae"
        assert clf["confidence"] == "high"
        assert clf["predicted_genus"] == "Inga"
        assert "reasoning" in clf

    def test_run_fallback_on_parse_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content="This is not valid JSON at all.",
            raw_content="This is not valid JSON at all.",
            thinking=None,
            processing_time_ms=200.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        context = {
            "evidence_synthesis": {"evidence_document": "..."},
            "classification": {
                "predictions": [{"rank_value": "Fabaceae", "softmax_score": 0.8}]
            },
        }
        result = stage.run(context)

        assert result.error is None
        clf = result.data["classification"]
        assert clf["predicted_family"] == "Fabaceae"
        assert clf["confidence"] == "low"
        assert "Fallback" in clf["reasoning"]

    def test_run_fallback_no_classification(self) -> None:
        """Fallback when neither parse nor Stage 2 data available."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content="garbage",
            raw_content="garbage",
            thinking=None,
            processing_time_ms=100.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        context = {"evidence_synthesis": {"evidence_document": "..."}}
        result = stage.run(context)

        clf = result.data["classification"]
        assert clf["predicted_family"] == "Unknown"
        assert clf["confidence"] == "low"
        assert clf["predicted_genus"] is None

    def test_no_images_in_messages(self) -> None:
        """Stage 5 is text-only -- no images in the messages."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content='{"predicted_family": "Fabaceae", "confidence": "high", '
            '"reasoning": "test"}',
            raw_content="...",
            thinking=None,
            processing_time_ms=100.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client
        stage.run({"evidence_synthesis": {"evidence_document": "test"}})

        call_args = mock_client.chat.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                assert all(c.get("type") != "image_url" for c in content)

    def test_client_error_returns_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Connection refused")

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        result = stage.run({"evidence_synthesis": {"evidence_document": "test"}})
        assert result.error is not None
        assert "Connection refused" in result.error
        assert result.data == {}

    def test_elapsed_ms_is_positive(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content='{"predicted_family": "Fabaceae", "confidence": "medium", '
            '"reasoning": "test"}',
            raw_content="...",
            thinking=None,
            processing_time_ms=300.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        result = stage.run({"evidence_synthesis": {"evidence_document": "test"}})
        assert result.elapsed_ms > 0

    def test_thinking_captured_in_result(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content='{"predicted_family": "Fabaceae", "confidence": "high", '
            '"reasoning": "compound leaves"}',
            raw_content="<think>analyzing evidence</think>...",
            thinking="analyzing evidence",
            processing_time_ms=400.0,
        )

        stage = ReasoningStage(config=ReasoningConfig())
        stage.client = mock_client

        result = stage.run({"evidence_synthesis": {"evidence_document": "test"}})
        assert result.data["thinking"] == "analyzing evidence"
        assert result.data["raw_response"] == "<think>analyzing evidence</think>..."
