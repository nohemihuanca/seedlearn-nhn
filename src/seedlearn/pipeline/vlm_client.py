"""Server-agnostic VLM inference client and response utilities.

Extracted from benchmarks/run_vlm_stage1.py with JSON resilience from
the demo client. Provides InferenceConfig, InferenceResponse, and
InferenceClient for OpenAI-compatible vision-language model endpoints.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceClient",
    "InferenceConfig",
    "InferenceResponse",
    "build_messages",
    "parse_json_response",
    "strip_thinking",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Configuration for an OpenAI-compatible inference endpoint."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
    api_key: str = "EMPTY"
    timeout: float = 172800
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = -1.0
    image_mode: Literal["file", "base64"] = "file"


@dataclass
class InferenceResponse:
    """Structured response from an inference call."""

    content: str
    raw_content: str
    thinking: str | None = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from model output.

    Handles both complete and incomplete thinking blocks.  Preserves
    leading whitespace that follows the closing tag so callers can
    distinguish trailing newlines from content.

    Args:
        text: Raw model output that may contain thinking tags.

    Returns:
        Text with thinking blocks removed.
    """
    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle incomplete blocks where only </think> remains
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from model output with multi-pass recovery.

    Attempts progressively more aggressive extraction strategies:
    1. Strip thinking blocks and markdown fences, try ``json.loads``.
    2. Regex-extract the outermost ``{...}`` object.
    3. Truncation recovery -- close unmatched braces/brackets.
    4. Return ``None`` if all strategies fail.

    Args:
        text: Raw model output potentially containing JSON.

    Returns:
        Parsed dictionary, or ``None`` on total failure.
    """
    cleaned = strip_thinking(text).strip()

    # Handle markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    # Pass 1: direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Pass 2: extract outermost JSON object from mixed content
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Pass 3: truncation recovery
    json_text = cleaned
    brace_start = json_text.find("{")
    if brace_start == -1:
        return None
    json_text = json_text[brace_start:]

    open_braces = json_text.count("{") - json_text.count("}")
    open_brackets = json_text.count("[") - json_text.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Strip trailing incomplete value after last comma/colon
        trimmed = json_text.rstrip().rstrip(",")
        lines = trimmed.split("\n")
        while len(lines) > 1 and not lines[-1].rstrip().endswith(
            (",", "}", "]", '"', "'")
        ):
            lines.pop()
        json_text = "\n".join(lines).rstrip().rstrip(",")

        # For single-line truncation, strip trailing incomplete token
        # e.g. '{"key": "value", "nested": {"a": 1' -> strip ', "nested": {"a": 1'
        # Try progressively: just close brackets, then trim last kv pair
        for candidate in [json_text]:
            ob = candidate.count("{") - candidate.count("}")
            ol = candidate.count("[") - candidate.count("]")
            attempt = candidate + "]" * max(ol, 0) + "}" * max(ob, 0)
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                pass

        # Aggressive: strip back to last complete value, try again
        # Remove trailing partial key-value pairs character by character
        for i in range(len(json_text) - 1, 0, -1):
            ch = json_text[i]
            if ch in (",",):
                candidate = json_text[:i].rstrip()
                ob = candidate.count("{") - candidate.count("}")
                ol = candidate.count("[") - candidate.count("]")
                attempt = candidate + "]" * max(ol, 0) + "}" * max(ob, 0)
                try:
                    return json.loads(attempt)
                except (json.JSONDecodeError, ValueError):
                    continue

    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None


def _get_image_url(image_path: str | Path, image_mode: str) -> str:
    """Build an image URL for the OpenAI messages API.

    Args:
        image_path: Filesystem path to the image.
        image_mode: Either ``"file"`` (file URI) or ``"base64"`` (data URI).

    Returns:
        URL string suitable for the ``image_url`` content block.
    """
    path = Path(image_path).resolve()
    if image_mode == "file":
        return f"file://{path}"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{data}"


def _image_content(image_paths: list[str], image_mode: str) -> list[dict[str, Any]]:
    """Build image_url content blocks for a list of image paths."""
    return [
        {"type": "image_url", "image_url": {"url": _get_image_url(path, image_mode)}}
        for path in image_paths
    ]


def load_examples(examples_file: str | Path) -> list[dict[str, Any]]:
    """Load in-context few-shot exemplars from a JSON file.

    Accepts either a list of ``{"images": [paths], "answer": text}`` objects or
    the benchmark's ``{"idx": {"img_list": [paths], "target": text}}`` mapping,
    normalizing both to a list of ``{"images", "answer"}`` dicts.

    Args:
        examples_file: Path to the exemplar JSON.

    Returns:
        List of ``{"images": [...], "answer": str}`` exemplars.

    Raises:
        ValueError: If an exemplar is missing images or an answer/target.
    """
    raw = json.loads(Path(examples_file).read_text())
    items = raw.values() if isinstance(raw, dict) else raw
    out: list[dict[str, Any]] = []
    for item in items:
        images = item.get("images") or item.get("img_list")
        answer = item.get("answer") or item.get("target")
        if not images or not answer:
            raise ValueError(f"exemplar missing images or answer/target: {item!r}")
        entry: dict[str, Any] = {"images": list(images), "answer": str(answer)}
        if item.get("text"):
            entry["text"] = str(item["text"])
        out.append(entry)
    return out


def build_messages(
    system_prompt: str,
    image_paths: list[str],
    image_mode: str = "file",
    user_text: str | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build an OpenAI-compatible message array with optional images.

    Args:
        system_prompt: Text for the system message.
        image_paths: Zero or more filesystem paths to images.
        image_mode: ``"file"`` for file URIs, ``"base64"`` for data URIs.
        user_text: Optional user-supplied text appended after images.
        examples: Optional in-context exemplars, each
            ``{"images": [paths], "answer": text, "text": optional intro}``. When
            provided, each is emitted as a ``user`` (optional intro text + images)
            + ``assistant`` (answer) turn before the real query. Used here to
            attach reference-illustration turns for the few-shot condition. When
            ``None``/empty the output is unchanged.

    Returns:
        List of message dictionaries ready for ``chat.completions.create``.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]

    for ex in examples or []:
        ex_content: list[dict[str, Any]] = []
        if ex.get("text"):
            ex_content.append({"type": "text", "text": ex["text"]})
        ex_content.extend(_image_content(ex["images"], image_mode))
        messages.append({"role": "user", "content": ex_content})
        messages.append({"role": "assistant", "content": ex["answer"]})

    user_content: list[dict[str, Any]] = _image_content(image_paths, image_mode)

    if user_text:
        user_content.append({"type": "text", "text": user_text})

    if not user_content:
        user_content.append({"type": "text", "text": ""})

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------


class InferenceClient:
    """Lightweight client for OpenAI-compatible VLM endpoints.

    Wraps the OpenAI Python SDK to provide structured responses,
    automatic thinking-block extraction, and timing metadata.

    Args:
        config: Endpoint and generation configuration.  Uses defaults
            when ``None``.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        logger.info("InferenceClient initialized for %s", self.config.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        strip_thinking: bool = True,
    ) -> InferenceResponse:
        """Send a chat-completion request and return a structured response.

        Args:
            messages: OpenAI-format message list.
            strip_thinking: If ``True``, separate ``<think>`` blocks from
                the returned ``content`` field.

        Returns:
            Populated ``InferenceResponse``.
        """
        start = time.perf_counter()

        extra_body: dict[str, Any] = {}
        if self.config.top_k > 0:
            extra_body["top_k"] = self.config.top_k
        if self.config.min_p > 0:
            extra_body["min_p"] = self.config.min_p

        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body=extra_body if extra_body else None,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_content = completion.choices[0].message.content or ""

        content, thinking = raw_content, None
        if strip_thinking and "</think>" in raw_content:
            parts = raw_content.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()

        usage: dict[str, int] = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return InferenceResponse(
            content=content,
            raw_content=raw_content,
            thinking=thinking,
            model=completion.model,
            usage=usage,
            processing_time_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        """Check whether the inference server is responding.

        Returns:
            ``True`` if the server lists models successfully.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
