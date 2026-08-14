"""Data schema for STRI identification key scraping."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlencode


STRI_BASE_URL = "https://panamabiota.org/stri"


def _slugify(text: str) -> str:
    """Convert label to snake_case column-safe slug.

    Args:
        text: Raw label text (e.g. "epiphyte - hemiepiphyte").

    Returns:
        Slugified string (e.g. "epiphyte_hemiepiphyte").
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class FilterOption:
    """Single trait filter checkbox (e.g., habit=tree).

    Attributes:
        category_id: Numeric ID of the parent filter category.
        option_id: Numeric ID of this option within the category.
        label: Human-readable label from the page checkbox.
    """

    category_id: int
    option_id: int
    label: str

    @property
    def attr_value(self) -> str:
        """URL parameter value (e.g. '1-1' for attr[]=1-1)."""
        return f"{self.category_id}-{self.option_id}"

    def column_name(self, category_name: str) -> str:
        """CSV column name: {category}__{option} (e.g. 'habit__tree')."""
        return f"{_slugify(category_name)}__{_slugify(self.label)}"


@dataclass
class FilterCategory:
    """A trait category with its available options.

    Attributes:
        category_id: Numeric ID from the page (e.g. 1 for Habit).
        name: Human-readable category name (slugified for columns).
        options: Available filter options within this category.
    """

    category_id: int
    name: str
    options: list[FilterOption] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        """All column names for this category's options."""
        return [opt.column_name(self.name) for opt in self.options]


@dataclass(frozen=True)
class SpeciesEntry:
    """A species parsed from an identification key page.

    Attributes:
        taxon_id: Symbiota internal taxon ID (from URL parameter).
        scientific_name: Binomial name (genus + epithet).
        family: Taxonomic family.
    """

    taxon_id: int
    scientific_name: str
    family: str


@dataclass
class IdentificationKey:
    """An STRI identification key page with trait filters.

    Attributes:
        cl_id: Checklist ID (URL parameter cl=).
        name: Human-readable key name.
        slug: URL/directory-safe slug.
        project_id: Project ID (URL parameter proj=, default 10).
        categories: Filter categories discovered from the page.
    """

    cl_id: int
    name: str
    slug: str
    project_id: int = 10
    categories: list[FilterCategory] = field(default_factory=list)

    @property
    def directory_name(self) -> str:
        """Directory name: cl{id}_{slug}."""
        return f"cl{self.cl_id}_{self.slug}"

    @property
    def base_url(self) -> str:
        """Unfiltered key page URL (all species)."""
        return (
            f"{STRI_BASE_URL}/ident/key.php"
            f"?cl={self.cl_id}&proj={self.project_id}"
            f"&dynclid=0&taxon=All+Species"
        )

    def filtered_url(self, options: list[FilterOption]) -> str:
        """Key page URL with trait filter(s) applied.

        Args:
            options: Filter options to apply (typically one at a time).

        Returns:
            Full URL with attr[] parameters appended.
        """
        params = [("attr[]", opt.attr_value) for opt in options]
        return f"{self.base_url}&{urlencode(params)}"


STRI_IDENTIFICATION_KEYS: list[IdentificationKey] = [
    IdentificationKey(59, "Panama Dicots", "panama_dicots"),
    IdentificationKey(
        60, "Ferns and Allies of Panama", "ferns_and_allies_of_panama",
    ),
    IdentificationKey(61, "Monocots of Panama", "monocots_of_panama"),
    IdentificationKey(
        178,
        "BCI Eudicots, Magnoliids, and Basal Angiosperms",
        "bci_eudicots_magnoliids_basal_angiosperms",
    ),
    IdentificationKey(
        185, "Complete Tree Species of Panama", "complete_tree_species_of_panama",
    ),
    IdentificationKey(71, "CTFS Tree Atlas of Panama", "ctfs_tree_atlas_of_panama"),
    IdentificationKey(72, "CTFS Liana Atlas of Panama", "ctfs_liana_atlas_of_panama"),
    IdentificationKey(65, "Campana National Park", "campana_national_park"),
    IdentificationKey(66, "Myrtaceae of Panama", "myrtaceae_of_panama"),
    IdentificationKey(
        85, "Soberania National Park Plants", "soberania_national_park_plants",
    ),
    IdentificationKey(
        70,
        "Trees in the Vicinity of Gamboa and the Canal",
        "trees_gamboa_and_canal",
    ),
]
