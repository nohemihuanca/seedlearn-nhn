"""Stage 4: Deterministic evidence synthesis.

Assembles all evidence from Stages 1-3 into a structured text document
for downstream LLM reasoning. No machine learning — pure formatting
and rule-based quality annotations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from seedlearn.pipeline.config import EvidenceSynthesisConfig
from seedlearn.pipeline.protocol import StageResult

logger = logging.getLogger(__name__)

# Trait values considered uninformative for quality-flag purposes.
_UNCLEAR_VALUES = frozenset({"unclear", "n/a", "not observed", "not visible"})

# If this many or more traits are unclear, flag low morphological quality.
_UNCLEAR_THRESHOLD = 5


def _format_morphology_section(morph_data: dict[str, Any]) -> str:
    """Format morphological traits into a readable section.

    Handles both nested dicts (from FormParser/JSONParser, grouped by section)
    and flat dicts (legacy format with string values).

    Args:
        morph_data: Morphology stage output dict (expects a ``traits`` key).

    Returns:
        Markdown-formatted section string.
    """
    traits = morph_data.get("traits", {})
    if not traits:
        return "## Morphological Profile\nNo morphological data available.\n"

    lines = ["## Morphological Profile\n"]
    for key, value in traits.items():
        if isinstance(value, dict):
            # Nested section (e.g. "leaf_arrangement": {"relative_position": "alternate"})
            section_title = key.replace("_", " ").title()
            lines.append(f"\n### {section_title}\n")
            for field_name, field_value in value.items():
                label = field_name.replace("_", " ").title()
                lines.append(f"- **{label}**: {field_value}")
        elif key == "notes" and value:
            lines.append(f"\n### Notes\n\n{value}")
        else:
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


def _format_classification_section(clf_data: dict[str, Any]) -> str:
    """Format visual classification predictions.

    Args:
        clf_data: Classification stage output dict.

    Returns:
        Markdown-formatted section string.
    """
    # Multi-rank format
    if "predictions_by_rank" in clf_data:
        return _format_multirank_classification(clf_data)

    # Single-rank format (existing code below, unchanged)
    predictions = clf_data.get("predictions", [])
    if not predictions:
        return "## Visual Classification\nNo classification data available.\n"

    lines = ["## Visual Classification\n"]

    # Show margin when present
    margin = clf_data.get("margin")
    if margin is not None:
        lines.append(f"Decision margin: {margin:.3f}\n")

    for pred in predictions:
        name = pred.get("rank_value", "Unknown")
        conf = pred.get("softmax_score", 0)
        pos = pred.get("rank_position", "?")
        line = f"- #{pos}: **{name}** (similarity share: {conf:.1%})"
        l2 = pred.get("l2_distance")
        cos = pred.get("cosine_similarity")
        if l2 is not None and cos is not None:
            line += f" | L2: {l2:.3f}, cosine: {cos:.3f}"
        lines.append(line)

    # Per-image predictions subsection
    per_image = clf_data.get("per_image_predictions", [])
    if per_image:
        lines.append("")
        lines.append("### Per-Image Predictions\n")
        for entry in per_image:
            img = entry.get("image_path", "?")
            label = entry.get("top1_label", "?")
            conf = entry.get("top1_softmax_score", 0)
            lines.append(f"- {img}: **{label}** (similarity share: {conf:.1%})")

    # Nearest support images subsection
    nearest = clf_data.get("nearest_support", [])
    if nearest:
        lines.append("")
        lines.append("### Nearest Support Images\n")
        for entry in nearest:
            label = entry.get("label", "?")
            l2 = entry.get("l2_distance", 0)
            cos = entry.get("cosine_similarity", 0)
            line = f"- **{label}** (L2: {l2:.3f}, cosine: {cos:.3f})"
            img = entry.get("image_path")
            if img:
                line += f" [{img}]"
            lines.append(line)

    return "\n".join(lines) + "\n"


def _format_multirank_classification(clf_data: dict[str, Any]) -> str:
    """Format multi-rank classification predictions.

    Args:
        clf_data: Classification stage output with ``predictions_by_rank``.

    Returns:
        Markdown-formatted section string.
    """
    predictions_by_rank = clf_data.get("predictions_by_rank", {})
    margin_by_rank = clf_data.get("margin_by_rank", {})

    lines = ["## Visual Classification (Multi-Rank)\n"]

    for rank_name, predictions in predictions_by_rank.items():
        rank_title = rank_name.capitalize()
        lines.append(f"### {rank_title}\n")

        margin = margin_by_rank.get(rank_name)
        if margin is not None:
            lines.append(f"Decision margin: {margin:.3f}\n")

        for pred in predictions:
            name = pred.get("rank_value", "Unknown")
            conf = pred.get("softmax_score", 0)
            pos = pred.get("rank_position", "?")
            line = f"- #{pos}: **{name}** (similarity share: {conf:.1%})"
            l2 = pred.get("l2_distance")
            cos = pred.get("cosine_similarity")
            if l2 is not None and cos is not None:
                line += f" | L2: {l2:.3f}, cosine: {cos:.3f}"
            lines.append(line)

        lines.append("")

    # Hierarchical consistency
    consistency = clf_data.get("hierarchical_consistency")
    if consistency:
        status = "Consistent" if consistency.get("consistent") else "INCONSISTENT"
        lines.append(f"### Hierarchical Consistency: {status}\n")
        for note in consistency.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    # OOD confidence gate
    gate = clf_data.get("confidence_gate")
    if gate:
        flags = gate.get("flags", [])
        if flags:
            lines.append("### Out-of-Distribution Warning\n")
            for flag in flags:
                lines.append(f"- {flag}")
            lines.append("")

    # Per-image predictions
    per_image = clf_data.get("per_image_predictions", [])
    if per_image:
        lines.append("### Per-Image Predictions\n")
        for entry in per_image:
            img = entry.get("image_path", "?")
            label = entry.get("top1_label", "?")
            conf = entry.get("top1_softmax_score", 0)
            lines.append(f"- {img}: **{label}** (similarity share: {conf:.1%})")
        lines.append("")

    # Nearest support
    nearest = clf_data.get("nearest_support", [])
    if nearest:
        lines.append("### Nearest Support Images\n")
        for entry in nearest:
            label = entry.get("label", "?")
            l2 = entry.get("l2_distance", 0)
            cos = entry.get("cosine_similarity", 0)
            line = f"- **{label}** (L2: {l2:.3f}, cosine: {cos:.3f})"
            img = entry.get("image_path")
            if img:
                line += f" [{img}]"
            lines.append(line)

    return "\n".join(lines) + "\n"


def _format_literature_section(rag_data: dict[str, Any] | None) -> str:
    """Format RAG literature evidence.

    Args:
        rag_data: Trait-retrieval stage output dict, or ``None`` if skipped.

    Returns:
        Markdown-formatted section string.
    """
    if not rag_data:
        return "## Literature Evidence\nLiterature trait retrieval not available.\n"

    matches = rag_data.get("rag_matches", [])

    lines = ["## Literature Evidence\n"]
    if matches:
        lines.append("### Matching Taxa (by trait similarity)\n")
        for match in matches:
            taxon = match.get("taxon", "Unknown")
            rank = match.get("rank", "")
            score = match.get("score", 0)
            desc = match.get("description", "")
            lines.append(f"- **{taxon}** ({rank}, similarity: {score:.2f})")
            if desc:
                lines.append(f"  - {desc[:200]}")
    else:
        lines.append("No matching taxa found in literature.\n")

    return "\n".join(lines) + "\n"


def _format_convergence_section(rag_data: dict[str, Any] | None) -> str:
    """Format convergence/divergence analysis.

    Args:
        rag_data: Trait-retrieval stage output dict, or ``None`` if skipped.

    Returns:
        Markdown-formatted section string.
    """
    if not rag_data:
        return "## Convergence Analysis\nInsufficient data for convergence analysis.\n"

    convergence = rag_data.get("convergence", [])
    if not convergence:
        return "## Convergence Analysis\nNo cross-reference data available.\n"

    _signal_labels = {
        "strong": "STRONG CONVERGENCE",
        "moderate": "MODERATE CONVERGENCE",
        "rag_only": "Literature only",
        "visual_only": "Visual only",
    }

    lines = ["## Convergence Analysis\n"]
    for entry in convergence:
        taxon = entry.get("taxon", "Unknown")
        signal = entry.get("signal", "unknown")
        rag_score = entry.get("rag_score", 0)
        vis_conf = entry.get("visual_softmax_score", 0)
        marker = _signal_labels.get(signal, signal)

        lines.append(
            f"- **{taxon}** [{marker}] "
            f"(RAG: {rag_score:.2f}, Visual: {vis_conf:.1%})"
        )

    return "\n".join(lines) + "\n"


def _compute_quality_flags(
    morph_data: dict[str, Any],
    clf_data: dict[str, Any],
    config: EvidenceSynthesisConfig,
) -> list[str]:
    """Identify quality issues in the evidence.

    Args:
        morph_data: Morphology stage output.
        clf_data: Classification stage output.
        config: Synthesis config (``convergence_threshold`` used as the
            minimum acceptable top-prediction confidence).

    Returns:
        List of human-readable warning strings; empty when quality is fine.
    """
    flags: list[str] = []

    # Check for many unclear traits (flatten nested dicts to leaf values)
    traits = morph_data.get("traits", {})
    flat_values: list[str] = []
    for v in traits.values():
        if isinstance(v, dict):
            flat_values.extend(str(sv) for sv in v.values())
        elif isinstance(v, str) and v:
            flat_values.append(v)
    unclear_count = sum(
        1 for v in flat_values if v.strip().lower() in _UNCLEAR_VALUES
    )
    if unclear_count >= _UNCLEAR_THRESHOLD:
        flags.append(
            f"Low morphological quality: {unclear_count} of {len(flat_values)} "
            f"traits are unclear/not observed"
        )

    # Check for low classification softmax score
    predictions = clf_data.get("predictions", [])
    if not predictions:
        # Multi-rank: use first rank's predictions
        predictions_by_rank = clf_data.get("predictions_by_rank", {})
        if predictions_by_rank:
            first_rank = next(iter(predictions_by_rank))
            predictions = predictions_by_rank[first_rank]
    if predictions:
        top_conf = predictions[0].get("softmax_score", 0)
        threshold = config.convergence_threshold
        if top_conf < threshold:
            flags.append(
                f"Low classification similarity: top prediction captures only "
                f"{top_conf:.1%} of total similarity (threshold: {threshold:.1%})"
            )

    # Check decision margin
    margin = clf_data.get("margin")
    if margin is not None and margin < 0.1:
        flags.append(
            f"Low decision margin: {margin:.3f} between top-1 and top-2"
        )

    # Check per-image disagreement
    per_image = clf_data.get("per_image_predictions", [])
    if per_image:
        unique_labels = {e.get("top1_label") for e in per_image}
        if len(unique_labels) > 1:
            flags.append(
                f"Per-image disagreement: images predict "
                f"{len(unique_labels)} distinct labels "
                f"({', '.join(sorted(str(l) for l in unique_labels))})"
            )

    # Check hierarchical consistency (multi-rank)
    consistency = clf_data.get("hierarchical_consistency")
    if consistency and not consistency.get("consistent", True):
        notes = consistency.get("notes", [])
        if notes:
            flags.append(f"Hierarchical inconsistency: {notes[0]}")
        else:
            flags.append("Hierarchical inconsistency between taxonomy ranks")

    # Check OOD confidence gate (multi-rank)
    gate = clf_data.get("confidence_gate")
    if gate:
        gate_flags = gate.get("flags", [])
        for gf in gate_flags:
            flags.append(f"Out-of-distribution warning: {gf}")

    # Check for no data at all
    if not flat_values and not predictions:
        flags.append("No evidence available from any stage")

    return flags


class EvidenceSynthesisStage:
    """Stage 4: Assemble evidence from Stages 1-3 into a structured document.

    Deterministic — same inputs always produce the same output. No LLM
    or probabilistic computation.

    Args:
        config: Evidence synthesis configuration.
    """

    def __init__(self, config: EvidenceSynthesisConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """Short identifier for this stage."""
        return "evidence_synthesis"

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Verify that at least one upstream stage provided data.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            List of error messages; empty when valid.
        """
        errors: list[str] = []
        has_morphology = "morphology" in context
        has_classification = "classification" in context
        if not has_morphology and not has_classification:
            errors.append(
                "At least one of 'morphology' or 'classification' must be "
                "present in context"
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

    def run(self, context: dict[str, Any]) -> StageResult:
        """Synthesize evidence from all prior stages into a Markdown document.

        Args:
            context: Accumulated pipeline state containing upstream outputs.

        Returns:
            StageResult whose ``data`` dict contains ``evidence_document``
            (str) and ``quality_flags`` (list[str]).
        """
        start = time.perf_counter()
        try:
            morph_data = context.get("morphology", {})
            clf_data = context.get("classification", {})
            rag_data = context.get("trait_retrieval")

            sections = [
                "# Evidence Summary\n",
                _format_morphology_section(morph_data),
                _format_classification_section(clf_data),
                _format_literature_section(rag_data),
                _format_convergence_section(rag_data),
            ]
            evidence_document = "\n".join(sections)

            quality_flags = _compute_quality_flags(
                morph_data, clf_data, self._config
            )

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage_name=self.name,
                data={
                    "evidence_document": evidence_document,
                    "quality_flags": quality_flags,
                },
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Evidence synthesis failed: %s", exc)
            return StageResult(
                stage_name=self.name,
                data={},
                error=str(exc),
                elapsed_ms=elapsed,
            )
