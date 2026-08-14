"""Tests for the curated prompt-wording map (human.trait_prompts)."""

import re

from seedlearn.benchmarking.human.trait_prompts import (
    PROMPT_TRAIT_TEXT,
    prompt_text_for,
)
from seedlearn.benchmarking.human.value_map import TRAIT_SPECS
from seedlearn.components.analyzers.prompts import SYSTEM_PROMPT_4


def _norm(text: str) -> str:
    """Collapse runs of whitespace to single spaces so substring checks are robust."""
    return re.sub(r"\s+", " ", text).strip()


def test_every_wording_is_a_substring_of_the_live_prompt():
    # Drift guard: if SYSTEM_PROMPT_4 is edited so a curated line no longer matches,
    # this fails loudly instead of the report silently showing stale text.
    prompt = _norm(SYSTEM_PROMPT_4)
    for key, text in PROMPT_TRAIT_TEXT.items():
        assert _norm(text) in prompt, f"{key}: wording not found in SYSTEM_PROMPT_4"


def test_covers_every_form_item_trait_key():
    # Every gradable/scored trait except the human-only ``damage`` is a numbered form
    # item and must have prompt wording; ``damage`` must not.
    form_keys = {s.key for s in TRAIT_SPECS if s.key != "damage"}
    assert set(PROMPT_TRAIT_TEXT) == form_keys
    assert "damage" not in PROMPT_TRAIT_TEXT


def test_prompt_text_for_lookup():
    assert prompt_text_for("leaf_margin").startswith("Leaf margin")
    assert prompt_text_for("damage") is None
    assert prompt_text_for("nonexistent") is None
