"""Tests for the bilingual trait/value map (benchmarking.human.value_map)."""

from seedlearn.benchmarking.human import value_map as vm
from seedlearn.benchmarking.human.value_map import (
    MISSING,
    TRAIT_SPECS,
    gradable_specs,
    model_value,
    normalize,
    spec_for_spanish_header,
    to_canonical,
    unmapped_values,
)


def _spec(key: str) -> vm.TraitSpec:
    return next(s for s in TRAIT_SPECS if s.key == key)


def test_english_and_spanish_map_to_same_token():
    spec = _spec("leaf_relative_position")
    assert to_canonical(spec, "whorled") == "whorled"
    assert to_canonical(spec, "verticilada") == "whorled"
    assert to_canonical(spec, "alterna") == to_canonical(spec, "alternate") == "alternate"


def test_margin_synonyms_collapse_to_toothed():
    spec = _spec("leaf_margin")
    assert to_canonical(spec, "entero") == "entire"
    assert to_canonical(spec, "entire") == "entire"
    for toothed in ("dentado", "serrado", "serrate", "crenate", "denticulate"):
        assert to_canonical(spec, toothed) == "toothed"


def test_bracket_placeholder_residue_is_stripped():
    # Some prompt templates present the answer field as ``[ ... ]`` and the model
    # echoes the markers, emitting ``"[entire"`` / ``"[entire]"`` / ``"[toothed"``.
    # These must normalize back to the bare token, not fail to map and drop out.
    spec = _spec("leaf_margin")
    assert normalize("[entire]") == "entire"
    assert normalize("[entire") == "entire"
    for residue in ("[entire", "[entire]", "( entire )", "{entire}"):
        assert to_canonical(spec, residue) == "entire", residue
    for residue in ("[toothed", "[toothed]"):
        assert to_canonical(spec, residue) == "toothed", residue


def test_accent_insensitive():
    spec = _spec("leaf_shape")
    assert normalize("elíptica") == "eliptica"
    assert to_canonical(spec, "eliptica") == "elliptic"
    assert to_canonical(spec, "elíptica") == "elliptic"
    assert to_canonical(spec, "elliptic") == "elliptic"


def test_missing_sentinels_both_languages():
    spec = _spec("leaf_apex")
    for missing in ("", "N/A", "no claro", "not observed", "no observado", "unclear", "  "):
        assert to_canonical(spec, missing) == MISSING


def test_latex_not_observed_is_missing_but_present_maps():
    spec = _spec("latex")
    assert to_canonical(spec, "no observado") == MISSING
    assert to_canonical(spec, "not observed") == MISSING
    assert to_canonical(spec, "presente") == "present"
    assert to_canonical(spec, "absent") == "absent"


def test_unmapped_value_is_missing_not_a_false_match():
    spec = _spec("leaf_apex")
    assert to_canonical(spec, "banana-shaped") == MISSING


def test_free_text_and_numeric_traits_not_gradable():
    non_gradable = {s.key for s in TRAIT_SPECS if not s.gradable}
    assert {"num_leaflets", "secondary_veins", "leaf_surface", "stem_color"} <= non_gradable


def test_gradable_specs_have_spanish_header_and_canonical_values():
    for spec in gradable_specs():
        assert spec.spanish_header_prefix, spec.key
        assert spec.canonical_values, spec.key
        # Every gradable trait has a model counterpart except documented
        # human-only traits (damage has no structured model field).
        if spec.key != "damage":
            assert spec.model_section and spec.model_field, spec.key
        # every canonical value is reachable as its own alias identity
        for token in spec.canonical_values:
            assert to_canonical(spec, token) == token, (spec.key, token)


def test_trait_keys_unique():
    keys = [s.key for s in TRAIT_SPECS]
    assert len(keys) == len(set(keys))


def test_model_value_extraction():
    traits = {
        "leaf_arrangement": {"relative_position": "whorled", "spacing": "clustered"},
        "special_features": {"stipules": "absent"},
    }
    assert model_value(traits, _spec("leaf_relative_position")) == "whorled"
    assert model_value(traits, _spec("stipules")) == "absent"
    # absent section -> None
    assert model_value(traits, _spec("venation")) is None
    # human-only trait (no model path) -> None
    assert model_value(traits, _spec("damage")) is None


def test_spec_for_spanish_header_matches_real_headers():
    cases = {
        "Posicion relativa de las hojas (alterna / opuesta / verticilada)": "leaf_relative_position",
        "Margen de la hoja (entero / dentado; si es dentado...)": "leaf_margin",
        "Tipo de tallo, puede no estar visible (lenoso / herbaceo)": "stem_type",
        "Tricomas en superficie foliar (presentes / ausentes)": "leaf_trichomes",
        "Tricomas en tallo (presentes / ausentes)": "stem_trichomes",
        "Zarcillos (presentes / ausentes)": "tendrils",
    }
    for header, expected_key in cases.items():
        spec = spec_for_spanish_header(header)
        assert spec is not None and spec.key == expected_key, header


def test_spec_for_spanish_header_unknown_returns_none():
    assert spec_for_spanish_header("identificador_unico") is None
    assert spec_for_spanish_header("") is None


def test_unmapped_values_surfaces_coverage_gaps():
    spec = _spec("leaf_apex")
    gaps = unmapped_values(spec, ["agudo", "weird-apex", "", "N/A", "obtuso"])
    assert gaps == ["weird-apex"]


def test_compound_margin_descriptor_resolves_when_tokens_agree():
    # Cloud models emit verbose margin values; all toothed-family words agree.
    spec = _spec("leaf_margin")
    for value in (
        "toothed, serrate",
        "toothed – serrate",
        "toothed, crenate",
        "toothed, shallowly crenate to undulate",
        "toothed-serrate",
    ):
        assert to_canonical(spec, value) == "toothed", value


def test_compound_descriptor_ambiguous_or_negated_stays_missing():
    spec = _spec("leaf_margin")
    # Two different canonicals disagree -> unresolved.
    assert to_canonical(spec, "entire to toothed") == MISSING
    # Negation must never be read as the token it negates.
    assert to_canonical(spec, "not toothed") == MISSING
    # No recognized token at all -> still missing (banana-shaped guard).
    assert to_canonical(spec, "banana shaped edge") == MISSING
