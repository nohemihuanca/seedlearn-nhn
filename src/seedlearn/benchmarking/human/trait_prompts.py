"""Prompt-facing description text for each morphological trait.

The 24 traits are presented to the Vision-LLM as numbered lines of a plain-text
Morphological Assessment Form inside ``SYSTEM_PROMPT_4`` (see
``seedlearn.components.analyzers.prompts``). That text is not otherwise structured,
so this module curates a ``trait_key -> verbatim wording`` map for the human-grading
report, which shows each trait's prompt wording (and its inline option list) next to
how the trait is actually graded. ``tests/unit/test_human_trait_prompts.py`` guards
against drift by asserting each value is still a substring of the live prompt.
"""

from __future__ import annotations

# trait_key -> the exact wording (with its inline option list) as written in the
# Morphological Assessment Form of SYSTEM_PROMPT_4. ``damage`` is intentionally absent:
# it is a human-only trait, not a numbered form item asked of the model.
PROMPT_TRAIT_TEXT: dict[str, str] = {
    "leaf_relative_position": "Leaf relative position (alternate / opposite / whorled)",
    "leaf_spacing": "Leaf spacing (clustered / distal)",
    "leaf_complexity_type": "Leaf complexity (simple / compound)",
    "compound_leaf_type": (
        "Compound leaf type, ONLY if leaf complexity is compound "
        "(odd-pinnate / even-pinnate)"
    ),
    "num_leaflets": "Number of leaflets (integer estimate 2, 4, 8, …)",
    "leaflet_arrangement": "Leaflet arrangement (opposite / alternate / subopposite)",
    "leaf_margin": "Leaf margin (entire / toothed) AND if toothed (dentate, serrate, etc.)",
    "leaf_shape": "Leaf shape (elliptic, obovate, lanceolate, etc.)",
    "leaf_apex": "Leaf apex (acute, obtuse, acuminate)",
    "leaf_base": "Leaf base (rounded, cordate, cuneate)",
    "venation": "Venation type (pinnate / palmate / parallel)",
    "secondary_veins": "Secondary veins (visibility, spacing, number)",
    "leaf_surface": "Leaf surface features (glabrous, shiny, dull, rugose)",
    "leaf_trichomes": "Leaf surface trichomes (present / absent)",
    "petiole_length": "Petiole length (short / long)",
    "petiole_features": "Petiole features (winged / grooved / terete)",
    "stem_type": "Stem type, may not be visible (woody / herbaceous)",
    "stem_trichomes": "Stem trichomes (present / absent)",
    "stem_color": "Stem color",
    "stem_texture": "Stem texture (smooth / ridged / lenticellate)",
    "stipules": "Stipules, sometimes visible only in close-up images (present / absent)",
    "latex": (
        "Latex, rarely visible but extremely diagnostic when present "
        "(present / not observed)"
    ),
    "pulvinus": "Pulvinus, for Fabaceae family only (present / absent)",
    "tendrils": "Tendrils (present / absent)",
}


def prompt_text_for(key: str) -> str | None:
    """Return the prompt wording for a trait key, or ``None`` if it is not a form item."""
    return PROMPT_TRAIT_TEXT.get(key)
