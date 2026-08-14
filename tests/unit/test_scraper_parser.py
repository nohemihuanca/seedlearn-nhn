"""Tests for STRI identification key HTML parser."""
from pathlib import Path

import pytest

from seedlearn.scraper.parser import (
    parse_filter_schema,
    parse_species_count,
    parse_species_list,
)
from seedlearn.scraper.schema import FilterCategory, SpeciesEntry


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_key_html() -> str:
    return (FIXTURE_DIR / "stri_key_sample.html").read_text()


class TestParseSpeciesList:
    def test_extracts_species_entries(self, sample_key_html: str) -> None:
        species = parse_species_list(sample_key_html)
        assert len(species) == 4
        assert all(isinstance(s, SpeciesEntry) for s in species)

    def test_species_have_taxon_ids(self, sample_key_html: str) -> None:
        species = parse_species_list(sample_key_html)
        assert all(s.taxon_id > 0 for s in species)
        ids = {s.taxon_id for s in species}
        assert 61651 in ids  # Anacardium excelsum
        assert 64534 in ids  # Erythrina fusca

    def test_species_have_families(self, sample_key_html: str) -> None:
        species = parse_species_list(sample_key_html)
        families = {s.family for s in species}
        assert families == {"Anacardiaceae", "Fabaceae"}

    def test_species_names_are_binomial(self, sample_key_html: str) -> None:
        species = parse_species_list(sample_key_html)
        assert all(" " in s.scientific_name for s in species)
        names = {s.scientific_name for s in species}
        assert "Anacardium excelsum" in names

    def test_family_assignment_correct(self, sample_key_html: str) -> None:
        species = parse_species_list(sample_key_html)
        by_id = {s.taxon_id: s for s in species}
        assert by_id[61651].family == "Anacardiaceae"
        assert by_id[64534].family == "Fabaceae"

    def test_empty_html_returns_empty(self) -> None:
        species = parse_species_list("<html><body></body></html>")
        assert species == []


class TestParseFilterSchema:
    def test_extracts_categories(self, sample_key_html: str) -> None:
        categories = parse_filter_schema(sample_key_html)
        assert len(categories) == 2
        assert all(isinstance(c, FilterCategory) for c in categories)

    def test_categories_have_options(self, sample_key_html: str) -> None:
        categories = parse_filter_schema(sample_key_html)
        assert all(len(c.options) >= 2 for c in categories)

    def test_option_attr_values_match_pattern(self, sample_key_html: str) -> None:
        import re

        categories = parse_filter_schema(sample_key_html)
        for cat in categories:
            for opt in cat.options:
                assert re.match(r"\d+-\d+", opt.attr_value)

    def test_category_names_discovered(self, sample_key_html: str) -> None:
        categories = parse_filter_schema(sample_key_html)
        names = {c.name for c in categories}
        assert "latex" in names
        assert "leaf_type" in names  # "Type" maps to known name via category_id=3

    def test_option_labels_parsed(self, sample_key_html: str) -> None:
        categories = parse_filter_schema(sample_key_html)
        all_labels = [opt.label for cat in categories for opt in cat.options]
        assert "present" in all_labels
        assert "simple" in all_labels

    def test_empty_html_returns_empty(self) -> None:
        categories = parse_filter_schema("<html><body></body></html>")
        assert categories == []


class TestParseSpeciesCount:
    def test_extracts_count(self, sample_key_html: str) -> None:
        count = parse_species_count(sample_key_html)
        assert count == 4

    def test_missing_count_returns_zero(self) -> None:
        count = parse_species_count("<html><body></body></html>")
        assert count == 0
