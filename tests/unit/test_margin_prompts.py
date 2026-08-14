"""Tests for the leaf-margin experiment prompt styles (margin_only, margin_rich)."""

from seedlearn.benchmarking.human.value_map import TRAIT_SPECS
from seedlearn.components.analyzers.parsers import FormParser
from seedlearn.components.analyzers.prompts import (
    PROMPTS,
    PromptStyle,
    get_prompt,
    is_multi_image_style,
    list_prompts,
)


def _canonical_margin_values() -> tuple[str, ...]:
    spec = next(s for s in TRAIT_SPECS if s.key == "leaf_margin")
    return tuple(spec.canonical_values)


def test_new_styles_registered_and_listed():
    for name in ("margin_only", "margin_rich"):
        assert get_prompt(name)
        assert name in list_prompts()
    assert PromptStyle.MARGIN_ONLY in PROMPTS
    assert PromptStyle.MARGIN_RICH in PROMPTS


def test_margin_prompts_keep_numbered_trait7_line():
    # The literal "7." numbered line must survive so FormParser maps it to margin.
    for name in ("margin_only", "margin_rich"):
        assert "7." in get_prompt(name) and "Leaf margin" in get_prompt(name)


def test_margin_only_response_parses_to_margin_field():
    # A response in the margin_only form parses to leaf_morphology.margin.
    response = (
        "=== Morphological Assessment Form ===\n"
        "C. Leaf Morphology\n"
        "    7.\tLeaf margin (entire / toothed): toothed (fine teeth along the edge)\n"
        "F. Notes: none\n####"
    )
    traits = FormParser.parse(response)
    assert traits["leaf_morphology"]["margin"] == "toothed"
    # Other sections stay empty rather than erroring.
    assert traits["leaf_arrangement"] == {}


def test_margin_rich_names_all_canonical_classes():
    # Drift guard: the enriched prompt must describe every graded canonical class,
    # so the prompt cannot silently diverge from the grader's vocabulary.
    prompt = get_prompt("margin_rich").lower()
    for value in _canonical_margin_values():
        assert value in prompt, value


def test_margin_styles_are_multi_image():
    assert is_multi_image_style("margin_only")
    assert is_multi_image_style("margin_rich")
