"""STRI Panama Biota web scraper for morphological trait extraction."""

from seedlearn.scraper.client import STRIClient
from seedlearn.scraper.matrix import (
    build_trait_matrix,
    load_trait_matrix,
    save_trait_matrix,
)
from seedlearn.scraper.parser import (
    parse_filter_schema,
    parse_species_count,
    parse_species_list,
)
from seedlearn.scraper.schema import (
    FilterCategory,
    FilterOption,
    IdentificationKey,
    SpeciesEntry,
    STRI_IDENTIFICATION_KEYS,
)

__all__ = [
    "FilterCategory",
    "FilterOption",
    "IdentificationKey",
    "STRIClient",
    "SpeciesEntry",
    "STRI_IDENTIFICATION_KEYS",
    "build_trait_matrix",
    "load_trait_matrix",
    "parse_filter_schema",
    "parse_species_count",
    "parse_species_list",
    "save_trait_matrix",
]
