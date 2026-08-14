"""Parsers for VLM morphological assessment responses.

Provides FormParser (for text-form output from SYS1-SYS4 prompts) and
JSONParser (for structured JSON output from the JSON prompt). Both produce
standardized dictionaries matching the 24-trait morphology schema and can
convert to MorphologyResult dataclasses for typed attribute access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes — structured morphology output
# ---------------------------------------------------------------------------


@dataclass
class LeafArrangement:
    """Traits 1-2: leaf arrangement and architecture."""

    relative_position: str = ""
    spacing: str = ""


@dataclass
class LeafComplexity:
    """Traits 3-6: leaf complexity."""

    type: str = ""
    compound_type: str = ""
    num_leaflets: int | str = ""
    leaflet_arrangement: str = ""


@dataclass
class LeafMorphology:
    """Traits 7-16: leaf morphology."""

    margin: str = ""
    shape: str = ""
    apex: str = ""
    base: str = ""
    venation: str = ""
    secondary_veins: str = ""
    surface_features: str = ""
    trichomes: str = ""
    petiole_length: str = ""
    petiole_features: str = ""


@dataclass
class StemTraits:
    """Traits 17-20: stem and shoot traits."""

    type: str = ""
    trichomes: str = ""
    color: str = ""
    texture: str = ""


@dataclass
class SpecialFeatures:
    """Traits 21-24: other visible seedling traits."""

    stipules: str = ""
    latex: str = ""
    pulvinus: str = ""
    tendrils: str = ""


@dataclass
class MorphologyResult:
    """Structured morphological assessment result.

    Attributes:
        image_id: Identifier for the source image.
        processing_time_ms: VLM inference time in milliseconds.
        leaf_arrangement: Traits 1-2 (position, spacing).
        leaf_complexity: Traits 3-6 (simple/compound, leaflet details).
        leaf_morphology: Traits 7-16 (margin, shape, venation, etc.).
        stem_traits: Traits 17-20 (type, color, texture).
        special_features: Traits 21-24 (stipules, latex, pulvinus, tendrils).
        notes: Free-text observations from section F.
    """

    image_id: str = ""
    processing_time_ms: float = 0.0
    leaf_arrangement: LeafArrangement = field(default_factory=LeafArrangement)
    leaf_complexity: LeafComplexity = field(default_factory=LeafComplexity)
    leaf_morphology: LeafMorphology = field(default_factory=LeafMorphology)
    stem_traits: StemTraits = field(default_factory=StemTraits)
    special_features: SpecialFeatures = field(default_factory=SpecialFeatures)
    notes: str = ""


# ---------------------------------------------------------------------------
# Trait mapping — numbered form fields to (section, field_name)
# ---------------------------------------------------------------------------

_TRAIT_MAP: dict[int, tuple[str, str]] = {
    1: ("leaf_arrangement", "relative_position"),
    2: ("leaf_arrangement", "spacing"),
    3: ("leaf_complexity", "type"),
    4: ("leaf_complexity", "compound_type"),
    5: ("leaf_complexity", "num_leaflets"),
    6: ("leaf_complexity", "leaflet_arrangement"),
    7: ("leaf_morphology", "margin"),
    8: ("leaf_morphology", "shape"),
    9: ("leaf_morphology", "apex"),
    10: ("leaf_morphology", "base"),
    11: ("leaf_morphology", "venation"),
    12: ("leaf_morphology", "secondary_veins"),
    13: ("leaf_morphology", "surface_features"),
    14: ("leaf_morphology", "trichomes"),
    15: ("leaf_morphology", "petiole_length"),
    16: ("leaf_morphology", "petiole_features"),
    17: ("stem_traits", "type"),
    18: ("stem_traits", "trichomes"),
    19: ("stem_traits", "color"),
    20: ("stem_traits", "texture"),
    21: ("special_features", "stipules"),
    22: ("special_features", "latex"),
    23: ("special_features", "pulvinus"),
    24: ("special_features", "tendrils"),
}

_SECTIONS = (
    "leaf_arrangement",
    "leaf_complexity",
    "leaf_morphology",
    "stem_traits",
    "special_features",
)

_SECTION_CLASSES: dict[str, type] = {
    "leaf_arrangement": LeafArrangement,
    "leaf_complexity": LeafComplexity,
    "leaf_morphology": LeafMorphology,
    "stem_traits": StemTraits,
    "special_features": SpecialFeatures,
}


def _strip_justification(value: str) -> str:
    """Remove parenthetical justification from a trait value.

    Args:
        value: Raw trait value, e.g. ``"whorled (leaves arranged...)"``.

    Returns:
        Clean value, e.g. ``"whorled"``.
    """
    return re.sub(r"\s*\(.*$", "", value).strip()


def _dict_to_morphology_result(
    data: dict[str, Any],
    image_id: str,
    time_ms: float,
) -> MorphologyResult:
    """Convert a structured trait dict to a MorphologyResult dataclass.

    Args:
        data: Nested dict with section keys mapping to field dicts.
        image_id: Image identifier.
        time_ms: Processing time in milliseconds.

    Returns:
        Populated MorphologyResult.
    """
    sections = {}
    for name, cls in _SECTION_CLASSES.items():
        sections[name] = cls(**data.get(name, {}))
    return MorphologyResult(
        image_id=image_id,
        processing_time_ms=time_ms,
        notes=data.get("notes", ""),
        **sections,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class FormParser:
    """Parser for text-form morphological assessment output (SYS1-SYS4 prompts).

    Parses the numbered form into a structured dict grouped by section,
    stripping parenthetical justifications from trait values.
    """

    @staticmethod
    def parse(text: str) -> dict[str, Any]:
        """Parse form output into a structured dictionary.

        Args:
            text: Raw VLM response in form format.

        Returns:
            Nested dict with keys: ``leaf_arrangement``, ``leaf_complexity``,
            ``leaf_morphology``, ``stem_traits``, ``special_features``,
            ``notes``.
        """
        result: dict[str, Any] = {section: {} for section in _SECTIONS}
        result["notes"] = ""

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped == "####":
                continue

            # Section F: free-text notes (no justification stripping)
            notes_match = re.match(r"^F\.\s*Notes:\s*(.*)", stripped)
            if notes_match:
                result["notes"] = notes_match.group(1).strip()
                continue

            # Numbered trait lines: "N. Trait name (options): value (justification)"
            trait_match = re.match(r"^(\d+)\.\s*", stripped)
            if not trait_match:
                continue
            number = int(trait_match.group(1))
            if number not in _TRAIT_MAP:
                continue

            rest = stripped[trait_match.end() :]
            _, _, value = rest.partition(":")
            value = _strip_justification(value)

            section, field_name = _TRAIT_MAP[number]
            result[section][field_name] = value

        return result

    @staticmethod
    def to_morphology_result(
        parsed: dict[str, Any],
        image_id: str = "",
        time_ms: float = 0.0,
    ) -> MorphologyResult:
        """Convert parsed dict to a MorphologyResult dataclass.

        Args:
            parsed: Output from :meth:`parse`.
            image_id: Image identifier.
            time_ms: Processing time in milliseconds.

        Returns:
            Populated MorphologyResult.
        """
        return _dict_to_morphology_result(parsed, image_id, time_ms)


class JSONParser:
    """Parser for JSON morphological assessment output (JSON prompt).

    Handles raw JSON and markdown-fenced JSON (````json ... ````).
    """

    @staticmethod
    def parse(text: str) -> dict[str, Any]:
        """Parse JSON output into a structured dictionary.

        Args:
            text: Raw VLM response containing JSON.

        Returns:
            Parsed dictionary matching the morphology schema.

        Raises:
            ValueError: If JSON cannot be extracted from the text.
        """
        cleaned = text.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = "\n".join(lines[1:])

        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract outermost JSON object
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"Cannot parse JSON from response: {text[:100]}...")

    @staticmethod
    def to_morphology_result(
        parsed: dict[str, Any],
        image_id: str = "",
        time_ms: float = 0.0,
    ) -> MorphologyResult:
        """Convert parsed dict to a MorphologyResult dataclass.

        Args:
            parsed: Output from :meth:`parse`.
            image_id: Image identifier.
            time_ms: Processing time in milliseconds.

        Returns:
            Populated MorphologyResult.
        """
        return _dict_to_morphology_result(parsed, image_id, time_ms)
