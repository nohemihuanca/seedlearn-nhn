"""Stage 3: Literature-based trait retrieval via RAG."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from seedlearn.pipeline.config import TraitRetrievalConfig, load_prompt
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.rag import RAGIndex

logger = logging.getLogger(__name__)

_DEFAULT_QUERY_TEMPLATE = "Tropical tree seedling with {traits}"
_DEFAULT_FALLBACK_QUERY = "tropical tree seedling"


def _compose_query(
    traits: dict[str, Any],
    template: str = _DEFAULT_QUERY_TEMPLATE,
    fallback: str = _DEFAULT_FALLBACK_QUERY,
) -> str:
    """Compose a natural language search query from extracted traits.

    Converts a trait dictionary (from Stage 1) into a descriptive sentence
    suitable for semantic search against botanical descriptions.

    Args:
        traits: Dictionary of trait name -> observed value.
        template: Query template with ``{traits}`` placeholder.
        fallback: Query to use when no usable traits are found.

    Returns:
        Natural language query string.
    """
    parts: list[str] = []
    for key, value in traits.items():
        if isinstance(value, dict):
            # Flatten nested section dicts (e.g. leaf_arrangement -> {relative_position: ...})
            for sub_key, sub_value in value.items():
                sub_str = str(sub_value).strip().lower()
                if sub_str and sub_str not in (
                    "unclear", "n/a", "not observed", "not visible",
                ):
                    parts.append(f"{sub_key}: {sub_str}")
        else:
            value_str = str(value).strip().lower()
            if value_str and value_str not in (
                "unclear",
                "n/a",
                "not observed",
                "not visible",
            ):
                parts.append(f"{key}: {value_str}")
    if not parts:
        return fallback
    return template.format(traits=", ".join(parts))


def _cross_reference(
    rag_matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-reference RAG matches against classification predictions.

    Identifies convergence (same taxon in both RAG and visual classification)
    and divergence (taxa appearing in only one source).  Comparisons are
    rank-aware: RAG matches tagged as "Family" are compared against
    family-level predictions, and so on.  When predictions use a single
    rank (e.g. family), only RAG matches of that rank participate.

    Args:
        rag_matches: Results from RAG search (each has ``taxon`` and ``rank``).
        predictions: Top-k predictions from Stage 2 classification
            (each has ``rank_value``).

    Returns:
        List of convergence/divergence annotations.
    """
    # Build lookup keyed on lowercase taxon name, keeping best score per taxon.
    # Include all ranks so family-level RAG matches can be found.
    rag_taxa: dict[str, dict[str, Any]] = {}
    for m in rag_matches:
        key = m["taxon"].lower()
        if key not in rag_taxa or m.get("score", 0) > rag_taxa[key].get("score", 0):
            rag_taxa[key] = m

    pred_taxa = {p["rank_value"].lower(): p for p in predictions}

    convergence: list[dict[str, Any]] = []

    # Check for taxa appearing in both sources
    shared = set(rag_taxa.keys()) & set(pred_taxa.keys())
    for taxon in shared:
        rag_entry = rag_taxa[taxon]
        pred_entry = pred_taxa[taxon]
        rag_score = rag_entry.get("score", 0)
        pred_conf = pred_entry.get("softmax_score", 0)

        if rag_score >= 0.5 and pred_conf >= 0.3:
            signal = "strong"
        else:
            signal = "moderate"

        convergence.append(
            {
                "taxon": rag_entry["taxon"],
                "signal": signal,
                "rag_score": rag_score,
                "visual_softmax_score": pred_conf,
                "source": "both",
            }
        )

    # RAG-only matches (top 3)
    rag_only = set(rag_taxa.keys()) - shared
    for taxon in list(rag_only)[:3]:
        entry = rag_taxa[taxon]
        convergence.append(
            {
                "taxon": entry["taxon"],
                "signal": "rag_only",
                "rag_score": entry.get("score", 0),
                "visual_softmax_score": 0.0,
                "source": "literature",
            }
        )

    # Prediction-only matches (top 3)
    pred_only = set(pred_taxa.keys()) - shared
    for taxon in list(pred_only)[:3]:
        entry = pred_taxa[taxon]
        convergence.append(
            {
                "taxon": entry["rank_value"],
                "signal": "visual_only",
                "rag_score": 0.0,
                "visual_softmax_score": entry.get("softmax_score", 0),
                "source": "visual",
            }
        )

    return convergence


class TraitRetrievalStage:
    """Stage 3: Retrieve literature trait descriptions matching observed morphology.

    Uses RAG (FAISS + sentence-transformers) to find botanical descriptions
    similar to the extracted traits, then cross-references with visual
    classification predictions to identify convergence signals.

    Args:
        config: Trait retrieval configuration.
        prompt_file: Optional path to a custom query template file.  The file
            should contain a template with a ``{traits}`` placeholder.  Falls
            back to the hardcoded default template if the file is missing.
    """

    def __init__(
        self, config: TraitRetrievalConfig, prompt_file: str | None = None
    ) -> None:
        self._config = config
        self._prompt_file = prompt_file
        self._rag_index: RAGIndex | None = None

    @property
    def name(self) -> str:
        """Return stage name identifier."""
        return "trait_retrieval"

    @property
    def rag_index(self) -> RAGIndex | None:
        """Lazy-load the RAG index from configured path if not already set."""
        if self._rag_index is None and self._config.index_path:
            self._rag_index = RAGIndex.load(Path(self._config.index_path))
        return self._rag_index

    @rag_index.setter
    def rag_index(self, value: RAGIndex) -> None:
        """Allow injecting a RAG index for testing or pre-loading."""
        self._rag_index = value

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Check that required context keys are present.

        Args:
            context: Pipeline context dictionary.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        if "morphology" not in context:
            errors.append("Missing required 'morphology' data from Stage 1")
        return errors

    def skip(self, context: dict[str, Any]) -> StageResult:
        """Return a skipped result without performing retrieval.

        Args:
            context: Pipeline context dictionary (unused).

        Returns:
            StageResult marked as skipped.
        """
        return StageResult(stage_name=self.name, data={}, skipped=True)

    def run(self, context: dict[str, Any]) -> StageResult:
        """Search RAG index for traits and cross-reference with classification.

        Args:
            context: Must contain 'morphology' from Stage 1. Optionally
                contains 'classification' from Stage 2.

        Returns:
            StageResult with RAG matches, query, and convergence annotations.
        """
        start = time.perf_counter()
        try:
            if self.rag_index is None:
                raise RuntimeError(
                    "RAG index not loaded. Set rag_index or configure index_path."
                )

            # Compose query from Stage 1 traits using configurable template
            morph_data = context.get("morphology", {})
            traits = morph_data.get("traits", {})
            query_template = load_prompt(
                self._prompt_file, fallback=_DEFAULT_QUERY_TEMPLATE
            )
            query = _compose_query(traits, template=query_template.strip())

            # Search RAG index
            rag_matches = self.rag_index.search(
                query,
                top_k=self._config.top_k,
                min_similarity=self._config.min_similarity,
            )

            # Cross-reference with Stage 2 predictions if available
            convergence: list[dict[str, Any]] = []
            if self._config.cross_reference:
                clf_data = context.get("classification", {})
                # Support both single-rank and multi-rank prediction formats
                predictions = clf_data.get("predictions", [])
                if not predictions:
                    predictions_by_rank = clf_data.get("predictions_by_rank", {})
                    # Collect predictions from all ranks for cross-referencing
                    for rank_preds in predictions_by_rank.values():
                        predictions.extend(rank_preds)
                if predictions:
                    convergence = _cross_reference(rag_matches, predictions)

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage_name=self.name,
                data={
                    "query": query,
                    "rag_matches": rag_matches,
                    "convergence": convergence,
                },
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Trait retrieval failed: %s", exc)
            return StageResult(
                stage_name=self.name,
                data={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
