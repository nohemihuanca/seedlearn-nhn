"""HTML parser for STRI identification key pages.

Parses species lists, filter schemas, and species counts from the Symbiota CMS
HTML structure used by the STRI Panama Biota portal.

Real HTML structure (verified against live pages):

Species list:
    <div id="key-taxa">
        <div class="family-div">FamilyName</div>
        <div class="taxon-div">
            <div class="sciname-div">
                <a href="../taxa/index.php?taxon=TAXON_ID&clid=CL_ID">
                    <i>Genus epithet</i>
                </a>
            </div>
        </div>
    </div>

Filter categories (container div ID varies per key: char15, char18, etc.):
    <div id="char{N}">
        <div class="dynam"><span class="dynamlang">Category Name</span></div>
        <div class="cs-div">
            <input name="attr[]" value="CAT_ID-OPT_ID"/>
            <span>option label</span>
        </div>
    </div>
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from seedlearn.scraper.schema import (
    FilterCategory,
    FilterOption,
    SpeciesEntry,
)

# Category ID → human-readable name mapping. Auto-discovered from page HTML,
# but we provide known fallbacks for stable column naming when the page label
# is ambiguous (e.g. "Type" → "leaf_type").
KNOWN_CATEGORY_NAMES: dict[int, str] = {
    1: "habit",
    2: "latex",
    3: "leaf_type",
    5: "leaf_arrangement",
    6: "leaf_margin",
    7: "stipules",
    10: "flower_symmetry",
    14: "ovary_placement",
    15: "carpel_number",
    17: "sex",
    18: "fruit",
    25: "armature",
    50: "leaf_punctations",
    60: "glands_on_blade_or_petiole",
}


def parse_species_list(html: str) -> list[SpeciesEntry]:
    """Parse species entries from an identification key page.

    Args:
        html: Raw HTML of the key page.

    Returns:
        List of SpeciesEntry with taxon_id, scientific_name, and family.
    """
    soup = BeautifulSoup(html, "html.parser")
    key_div = soup.find("div", id="key-taxa")
    if key_div is None:
        return []

    species: list[SpeciesEntry] = []
    current_family = ""

    for div in key_div.find_all("div", recursive=False):
        css_class = div.get("class", [])

        if "family-div" in css_class:
            current_family = div.get_text(strip=True)

        elif "taxon-div" in css_class:
            link = div.find("a", href=True)
            if link is None:
                continue
            href = str(link["href"])
            match = re.search(r"taxon=(\d+)", href)
            if match is None:
                continue
            taxon_id = int(match.group(1))

            # Scientific name is in <i> tag inside the link
            name_tag = link.find("i")
            name = (
                name_tag.get_text(strip=True) if name_tag
                else link.get_text(strip=True)
            )

            species.append(SpeciesEntry(
                taxon_id=taxon_id,
                scientific_name=name,
                family=current_family,
            ))

    return species


def parse_filter_schema(html: str) -> list[FilterCategory]:
    """Parse trait filter checkboxes from an identification key page.

    Auto-discovers which categories and options are available for this key.
    Category names are resolved from the ``dynamlang`` span within each
    ``char18`` group, with fallback to KNOWN_CATEGORY_NAMES for stable naming.

    Args:
        html: Raw HTML of the key page.

    Returns:
        List of FilterCategory, each with its FilterOption children.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Each category group is a <div id="char{N}"> (N varies per key: char15,
    # char18, etc.) containing dynamlang span + cs-div checkboxes. Match by
    # the id prefix pattern rather than a hardcoded ID.
    char_divs = soup.find_all("div", id=re.compile(r"^char\d+$"))

    categories: list[FilterCategory] = []

    for char_div in char_divs:
        # Extract category name from dynamlang span
        name_span = char_div.find("span", class_="dynamlang")
        if name_span is None:
            continue
        # Get text, strip info anchors and whitespace
        raw_name = name_span.get_text(strip=True).rstrip("?").strip()

        # Parse checkboxes within this category group
        checkboxes = char_div.find_all(
            "input", attrs={"type": "checkbox", "name": "attr[]"},
        )
        if not checkboxes:
            continue

        options: list[FilterOption] = []
        cat_id: int | None = None

        for cb in checkboxes:
            value = cb.get("value", "")
            match = re.match(r"(\d+)-(\d+)", value)
            if not match:
                continue
            this_cat_id = int(match.group(1))
            opt_id = int(match.group(2))

            if cat_id is None:
                cat_id = this_cat_id

            # Option label is in sibling <span> after the checkbox
            label_span = cb.find_next_sibling("span")
            if label_span is None:
                # Fallback: look in parent cs-div
                parent_div = cb.parent
                if parent_div:
                    label_span = parent_div.find("span")
            label = (
                label_span.get_text(strip=True) if label_span
                else f"option_{opt_id}"
            )

            options.append(FilterOption(
                category_id=this_cat_id,
                option_id=opt_id,
                label=label,
            ))

        if cat_id is not None and options:
            # Use known name if available, else slugify the page label
            name = KNOWN_CATEGORY_NAMES.get(cat_id, _slugify_category(raw_name))
            categories.append(FilterCategory(
                category_id=cat_id,
                name=name,
                options=options,
            ))

    return categories


def _slugify_category(name: str) -> str:
    """Slugify a category name from the page for column naming.

    Args:
        name: Raw category name (e.g. "Ovary Placement").

    Returns:
        Slugified name (e.g. "ovary_placement").
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def parse_species_count(html: str) -> int:
    """Extract the 'Species Count: N' value from a key page.

    Args:
        html: Raw HTML of the key page.

    Returns:
        Species count, or 0 if not found.
    """
    match = re.search(r"Species Count:\s*([\d,]+)", html)
    if match:
        return int(match.group(1).replace(",", ""))
    return 0
