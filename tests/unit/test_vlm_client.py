"""Tests for VLM client."""

from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, patch

from seedlearn.pipeline.vlm_client import (
    InferenceClient,
    InferenceConfig,
    InferenceResponse,
    build_messages,
    parse_json_response,
    strip_thinking,
)


class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.timeout == 172800
        assert cfg.image_mode == "file"

    def test_custom(self):
        cfg = InferenceConfig(base_url="http://remote:9000/v1", model="test/model")
        assert cfg.base_url == "http://remote:9000/v1"
        assert cfg.model == "test/model"


class TestStripThinking:
    def test_strips_think_tags(self):
        text = "<think>internal reasoning</think>The answer is 42."
        assert strip_thinking(text) == "The answer is 42."

    def test_no_think_tags(self):
        text = "Just a normal response."
        assert strip_thinking(text) == "Just a normal response."

    def test_multiline_thinking(self):
        text = "<think>\nline 1\nline 2\n</think>\nResult here."
        assert strip_thinking(text) == "\nResult here."

    def test_incomplete_thinking(self):
        text = "prefix</think>actual content"
        assert strip_thinking(text) == "actual content"


class TestParseJsonResponse:
    def test_clean_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_thinking(self):
        text = '<think>reasoning</think>```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_truncated_json_recovery(self):
        text = '{"key": "value", "nested": {"a": 1'
        result = parse_json_response(text)
        assert result is not None
        assert result["key"] == "value"

    def test_returns_none_on_garbage(self):
        result = parse_json_response("this is not json at all")
        assert result is None

    def test_json_mixed_with_text(self):
        text = 'Here is the result:\n{"predicted_family": "Fabaceae"}\nDone.'
        result = parse_json_response(text)
        assert result is not None
        assert result["predicted_family"] == "Fabaceae"


class TestBuildMessages:
    def test_single_image(self):
        msgs = build_messages(
            system_prompt="You are a botanist.",
            image_paths=["/path/to/img.jpg"],
            image_mode="file",
        )
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content = msgs[1]["content"]
        assert any(c["type"] == "image_url" for c in content)

    def test_multi_image(self):
        msgs = build_messages(
            system_prompt="Botanist.",
            image_paths=["/a.jpg", "/b.jpg", "/c.jpg"],
            image_mode="file",
        )
        user_content = msgs[1]["content"]
        image_items = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_items) == 3

    def test_with_user_text(self):
        msgs = build_messages(
            system_prompt="System.",
            image_paths=["/a.jpg"],
            image_mode="file",
            user_text="Describe this seedling.",
        )
        user_content = msgs[1]["content"]
        text_items = [c for c in user_content if c["type"] == "text"]
        assert len(text_items) == 1
        assert text_items[0]["text"] == "Describe this seedling."

    def test_text_only_no_images(self):
        msgs = build_messages(
            system_prompt="System.",
            image_paths=[],
            image_mode="file",
            user_text="Reason over this evidence.",
        )
        user_content = msgs[1]["content"]
        assert all(c.get("type") != "image_url" for c in user_content)

    def test_file_mode_urls(self):
        msgs = build_messages(
            system_prompt="S.",
            image_paths=["/abs/path/img.jpg"],
            image_mode="file",
        )
        url = msgs[1]["content"][0]["image_url"]["url"]
        assert url.startswith("file://")
