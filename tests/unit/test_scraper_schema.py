"""Tests for STRI scraper data schema."""
import pytest

from seedlearn.scraper.schema import (
    FilterCategory,
    FilterOption,
    IdentificationKey,
    SpeciesEntry,
    STRI_IDENTIFICATION_KEYS,
)


class TestFilterOption:
    def test_attr_value(self) -> None:
        opt = FilterOption(category_id=1, option_id=1, label="tree")
        assert opt.attr_value == "1-1"

    def test_column_name(self) -> None:
        opt = FilterOption(category_id=1, option_id=1, label="tree")
        assert opt.column_name("habit") == "habit__tree"

    def test_label_slugified(self) -> None:
        opt = FilterOption(category_id=1, option_id=5, label="epiphyte - hemiepiphyte")
        assert opt.column_name("habit") == "habit__epiphyte_hemiepiphyte"


class TestFilterCategory:
    def test_category_with_options(self) -> None:
        cat = FilterCategory(
            category_id=1,
            name="habit",
            options=[
                FilterOption(1, 1, "tree"),
                FilterOption(1, 2, "shrub"),
            ],
        )
        assert len(cat.options) == 2
        assert cat.column_names == ["habit__tree", "habit__shrub"]


class TestIdentificationKey:
    def test_key_properties(self) -> None:
        key = IdentificationKey(
            cl_id=59,
            name="Panama Dicots",
            slug="panama_dicots",
            project_id=10,
        )
        assert key.base_url == (
            "https://panamabiota.org/stri/ident/key.php"
            "?cl=59&proj=10&dynclid=0&taxon=All+Species"
        )
        assert key.directory_name == "cl59_panama_dicots"

    def test_filtered_url(self) -> None:
        key = IdentificationKey(
            cl_id=59, name="Panama Dicots",
            slug="panama_dicots", project_id=10,
        )
        url = key.filtered_url([FilterOption(1, 1, "tree")])
        assert "attr%5B%5D=1-1" in url or "attr[]=1-1" in url


class TestSpeciesEntry:
    def test_from_parsed_data(self) -> None:
        entry = SpeciesEntry(
            taxon_id=61885,
            scientific_name="Aphelandra arnoldii",
            family="Acanthaceae",
        )
        assert entry.taxon_id == 61885


class TestKeyRegistry:
    def test_all_keys_registered(self) -> None:
        assert len(STRI_IDENTIFICATION_KEYS) == 11

    def test_key_slugs_unique(self) -> None:
        slugs = [k.slug for k in STRI_IDENTIFICATION_KEYS]
        assert len(slugs) == len(set(slugs))
