"""Stage 1: VLM morphological extraction."""

from __future__ import annotations

import logging
import time
from typing import Any

from seedlearn.components.analyzers.parsers import FormParser
from seedlearn.components.analyzers.prompts import get_prompt, is_json_style
from seedlearn.pipeline.config import VLMConfig, load_prompt
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.vlm_client import (
    InferenceClient,
    InferenceConfig,
    build_messages,
    load_examples,
    parse_json_response,
)

logger = logging.getLogger(__name__)


class MorphologyStage:
    """Stage 1: Extract morphological traits from seedling images using a VLM.

    Args:
        config: VLM configuration including model, endpoint, and prompt style.
        prompt_file: Optional path to a custom prompt text file.  When set,
            this file is loaded instead of the prompt registry.  Falls back to
            the registry prompt (``config.prompt_style``) if the file is missing.
    """

    def __init__(self, config: VLMConfig, prompt_file: str | None = None) -> None:
        self._config = config
        self._prompt_file = prompt_file
        self._client: InferenceClient | None = None
        self._examples: list[dict[str, Any]] | None = None
        if config.examples_file:
            self._examples = load_examples(config.examples_file)

    @property
    def name(self) -> str:
        """Return stage name identifier."""
        return "morphology"

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
                    min_p=self._config.min_p,
                    image_mode=self._config.image_mode,
                )
            )
        return self._client

    @client.setter
    def client(self, value: InferenceClient) -> None:
        """Allow injecting a client for testing."""
        self._client = value

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Check that required context keys are present.

        Args:
            context: Pipeline context dictionary.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        if "image_paths" not in context or not context["image_paths"]:
            errors.append("Missing required 'image_paths' in context")
        return errors

    def skip(self, context: dict[str, Any]) -> StageResult:
        """Return a skipped result without performing inference.

        Args:
            context: Pipeline context dictionary (unused).

        Returns:
            StageResult marked as skipped.
        """
        return StageResult(stage_name=self.name, data={}, skipped=True)

    def _delivery_report(self, image_paths: list[str], prompt: str) -> dict[str, Any]:
        """Structured evidence of what Stage-1 actually sent to the model.

        Few-shot exemplars share the request's image slots with the specimen
        views (vLLM's ``--limit-mm-per-prompt``). Dropping specimen views to fit
        exemplars would silently break the one-factor-at-a-time design, so this
        records the exemplar/specimen image counts, the budget, whether the request
        is over budget (i.e. at risk of dropped views), and the prompt size — so a
        run dir carries provable delivery evidence rather than an inferred "the flag
        was set". Emits a warning on overflow (preserving prior behavior).
        """
        n_exemplar_images = sum(len(ex["images"]) for ex in (self._examples or []))
        n_specimen_images = len(image_paths)
        total = n_specimen_images + n_exemplar_images
        over_budget = total > self._config.max_images
        if over_budget:
            logger.warning(
                "image budget exceeded: %d specimen + %d exemplar images > max_images=%d; "
                "reduce exemplars or raise --limit-mm-per-prompt to avoid dropped views",
                n_specimen_images,
                n_exemplar_images,
                self._config.max_images,
            )
        return {
            "n_examples": len(self._examples or []),
            "n_exemplar_images": n_exemplar_images,
            "n_specimen_images": n_specimen_images,
            "max_images": self._config.max_images,
            "images_over_budget": over_budget,
            "prompt_chars": len(prompt),
        }

    def run(self, context: dict[str, Any]) -> StageResult:
        """Execute morphological extraction on the provided images.

        Args:
            context: Must contain 'image_paths' (list of file paths).

        Returns:
            StageResult with extracted traits, raw response, and timing.
        """
        start = time.perf_counter()
        try:
            image_paths = context["image_paths"]
            registry_prompt = get_prompt(self._config.prompt_style)
            prompt = load_prompt(self._prompt_file, fallback=registry_prompt)
            delivery = self._delivery_report(image_paths, prompt)
            messages = build_messages(
                system_prompt=prompt,
                image_paths=image_paths,
                image_mode=self._config.image_mode,
                examples=self._examples,
            )
            response = self.client.chat(messages)

            # Parse response based on prompt style
            if is_json_style(self._config.prompt_style):
                traits = parse_json_response(response.content) or {}
            else:
                traits = FormParser.parse(response.content)

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage_name=self.name,
                data={
                    "traits": traits,
                    "raw_response": response.raw_content,
                    "thinking": response.thinking,
                    "few_shot_delivery": delivery,
                },
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Morphology extraction failed: %s", exc)
            return StageResult(
                stage_name=self.name,
                data={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
