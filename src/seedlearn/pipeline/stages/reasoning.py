"""Stage 5: Text-only LLM reasoning and classification."""

from __future__ import annotations

import logging
import time
from typing import Any

from seedlearn.pipeline.config import ReasoningConfig, load_prompt
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.vlm_client import (
    InferenceClient,
    InferenceConfig,
    build_messages,
    parse_json_response,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a botanical classification expert. You are given a structured evidence \
summary about a tropical tree seedling specimen, compiled from morphological \
observations, visual embedding classification, and literature trait matching.

Your task:
1. Analyze ALL evidence carefully
2. Identify the most likely taxonomic family (and genus/species if sufficient evidence)
3. Assess your confidence level (high, medium, or low)
4. Explain your reasoning, citing specific traits and evidence
5. List plausible alternatives

Respond with a JSON object ONLY (no markdown, no explanation outside the JSON):
{
  "predicted_family": "<family name>",
  "predicted_genus": "<genus name or null>",
  "predicted_species": "<species binomial or null>",
  "confidence": "<high|medium|low>",
  "reasoning": "<1-3 sentences explaining your classification>",
  "supporting_features": ["<trait1>", "<trait2>"],
  "alternatives": [
    {"taxon": "<name>", "reason": "<why it's a plausible alternative>"}
  ]
}"""


class ReasoningStage:
    """Stage 5: Classify seedlings via text-only LLM reasoning over evidence.

    Uses a text-only LLM to reason about the assembled evidence document
    and produce a structured classification with confidence and alternatives.

    Args:
        config: Reasoning model configuration.
        prompt_file: Optional path to a custom system prompt file.  Falls back
            to the hardcoded ``_SYSTEM_PROMPT`` if the file is missing.
    """

    def __init__(self, config: ReasoningConfig, prompt_file: str | None = None) -> None:
        self._config = config
        self._prompt_file = prompt_file
        self._client: InferenceClient | None = None

    @property
    def name(self) -> str:
        """Short identifier for this stage."""
        return "reasoning"

    @property
    def client(self) -> InferenceClient:
        """Lazy-initialize and return the inference client."""
        if self._client is None:
            self._client = InferenceClient(
                InferenceConfig(
                    base_url=self._config.endpoint,
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    top_k=self._config.top_k,
                )
            )
        return self._client

    @client.setter
    def client(self, value: InferenceClient) -> None:
        """Allow injecting a client for testing."""
        self._client = value

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Check that the evidence document exists in context.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            List of error messages; empty when valid.
        """
        errors: list[str] = []
        evidence = context.get("evidence_synthesis", {})
        if not evidence.get("evidence_document"):
            errors.append(
                "Missing required 'evidence_synthesis.evidence_document' in context"
            )
        return errors

    def skip(self, context: dict[str, Any]) -> StageResult:
        """Return a no-op skipped result.

        Args:
            context: Pipeline state (unused).

        Returns:
            StageResult with ``skipped=True``.
        """
        return StageResult(stage_name=self.name, data={}, skipped=True)

    def _build_fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build fallback classification from Stage 2 predictions.

        Args:
            context: Pipeline state containing prior stage outputs.

        Returns:
            Minimal classification dict with low confidence.
        """
        clf_data = context.get("classification", {})
        predictions = clf_data.get("predictions", [])
        if predictions:
            top = predictions[0]
            return {
                "predicted_family": top.get("rank_value", "Unknown"),
                "predicted_genus": None,
                "predicted_species": None,
                "confidence": "low",
                "reasoning": "Fallback to visual classification (LLM reasoning failed).",
                "supporting_features": [],
                "alternatives": [],
            }
        return {
            "predicted_family": "Unknown",
            "predicted_genus": None,
            "predicted_species": None,
            "confidence": "low",
            "reasoning": "No classification data available.",
            "supporting_features": [],
            "alternatives": [],
        }

    def run(self, context: dict[str, Any]) -> StageResult:
        """Execute text-only LLM reasoning over the evidence document.

        Args:
            context: Must contain ``evidence_synthesis.evidence_document``.

        Returns:
            StageResult with classification, raw response, and timing.
        """
        start = time.perf_counter()
        try:
            evidence_doc = context["evidence_synthesis"]["evidence_document"]
            system_prompt = load_prompt(self._prompt_file, fallback=_SYSTEM_PROMPT)

            # Build text-only messages (no images)
            messages = build_messages(
                system_prompt=system_prompt,
                image_paths=[],
                user_text=evidence_doc,
            )

            response = self.client.chat(messages)

            classification = parse_json_response(response.content)

            if classification is None:
                logger.warning("Failed to parse LLM response, using fallback")
                classification = self._build_fallback(context)

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage_name=self.name,
                data={
                    "classification": classification,
                    "raw_response": response.raw_content,
                    "thinking": response.thinking,
                },
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Reasoning stage failed: %s", exc)
            return StageResult(
                stage_name=self.name,
                data={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
