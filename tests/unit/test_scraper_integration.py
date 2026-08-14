"""End-to-end integration test for STRI trait scraper with mocked HTTP."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from seedlearn.scraper.schema import (
    FilterOption,
    IdentificationKey,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Species subsets for simulated filter responses.
# Unfiltered page has 4 species: 61651, 62139, 64534, 66182.
FILTER_SPECIES: dict[str, list[int]] = {
    "2-1": [61651, 64534],          # latex: present → 2 species
    "2-2": [62139],                  # latex: absent → 1 species (66182 = uncoded)
    "3-1": [61651, 62139, 64534],   # leaf_type: simple → 3 species
    "3-2": [61651],                  # leaf_type: compound → 1 species (multi-label with simple)
}


def _build_filtered_html(taxon_ids: list[int]) -> str:
    """Build a minimal HTML response with only the given species.

    Uses the same structure as the full fixture but with a species subset.
    """
    species_data = {
        61651: ("Anacardiaceae", "Anacardium excelsum"),
        62139: ("Anacardiaceae", "Astronium graveolens"),
        64534: ("Fabaceae", "Erythrina fusca"),
        66182: ("Fabaceae", "Lonchocarpus velutinus"),
    }

    # Group by family
    by_family: dict[str, list[tuple[int, str]]] = {}
    for tid in taxon_ids:
        fam, name = species_data[tid]
        by_family.setdefault(fam, []).append((tid, name))

    species_html = ""
    for family, members in sorted(by_family.items()):
        species_html += f'<div class="family-div">{family}</div>\n'
        for tid, name in members:
            species_html += (
                f'<div class="taxon-div"><div class="sciname-div">'
                f'<a href="../taxa/index.php?taxon={tid}&amp;clid=70">'
                f'<i>{name}</i></a></div></div>\n'
            )

    return f"""<html><body>
<div id="key-taxa">
<div style="margin-bottom:15px;">Species Count: {len(taxon_ids)}</div>
{species_html}
</div>
</body></html>"""


def _mock_get(url: str, **kwargs: object) -> MagicMock:
    """Return mocked responses based on URL content."""
    response = MagicMock()
    response.status_code = 200

    # Check if this is a filtered request
    for attr_val, taxon_ids in FILTER_SPECIES.items():
        encoded = attr_val.replace("-", "-")
        if f"attr%5B%5D={encoded}" in url or f"attr[]={encoded}" in url:
            response.text = _build_filtered_html(taxon_ids)
            return response

    # Unfiltered: return full fixture
    response.text = (FIXTURE_DIR / "stri_key_sample.html").read_text()
    return response


class TestScraperIntegration:
    """End-to-end test: fetch → parse → build matrix → save → load → verify."""

    @patch("seedlearn.scraper.client.requests.Session.get", side_effect=_mock_get)
    def test_full_pipeline(self, mock_get: MagicMock, tmp_path: Path) -> None:
        from scripts.scrape_stri_traits import scrape_single_key

        key = IdentificationKey(
            cl_id=70,
            name="Test Key",
            slug="test_key",
            project_id=10,
        )

        csv_path = scrape_single_key(
            key, output_dir=tmp_path, delay_seconds=0.0,
        )

        # Verify CSV exists and loads
        assert csv_path.exists()
        df = pd.read_csv(csv_path)

        # Verify shape: 4 species, id cols + 4 traits + 2 uncoded = 9
        assert len(df) == 4
        assert "taxon_id" in df.columns
        assert "latex__present" in df.columns
        assert "leaf_type__simple" in df.columns
        assert "latex__uncoded" in df.columns
        assert "leaf_type__uncoded" in df.columns

        # Verify multi-label: species 61651 has both simple AND compound
        row_61651 = df[df["taxon_id"] == 61651].iloc[0]
        assert row_61651["leaf_type__simple"] == 1
        assert row_61651["leaf_type__compound"] == 1
        assert row_61651["latex__present"] == 1
        assert row_61651["latex__absent"] == 0
        assert row_61651["latex__uncoded"] == 0
        assert row_61651["leaf_type__uncoded"] == 0

        # Verify uncoded: species 66182 not in any latex filter → uncoded
        row_66182 = df[df["taxon_id"] == 66182].iloc[0]
        assert row_66182["latex__present"] == 0
        assert row_66182["latex__absent"] == 0
        assert row_66182["latex__uncoded"] == 1

        # Verify metadata JSON
        meta_path = (
            tmp_path / "per_key_trait_matrices" / "cl70_test_key_scrape_metadata.json"
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["species_count_reported"] == 4
        assert meta["species_count_scraped"] == 4
        assert len(meta["categories"]) == 2

        # Verify raw HTML cached
        html_dir = tmp_path / "raw_html" / "cl70_test_key"
        assert (html_dir / "unfiltered_all_species.html").exists()

    @patch("seedlearn.scraper.client.requests.Session.get", side_effect=_mock_get)
    def test_cached_html_skips_http(
        self, mock_get: MagicMock, tmp_path: Path,
    ) -> None:
        from scripts.scrape_stri_traits import scrape_single_key

        key = IdentificationKey(
            cl_id=70, name="Test Key", slug="test_key", project_id=10,
        )

        # First run: fetches from HTTP
        scrape_single_key(key, output_dir=tmp_path, delay_seconds=0.0)
        first_call_count = mock_get.call_count

        # Second run: should use cache (no additional HTTP calls)
        scrape_single_key(key, output_dir=tmp_path, delay_seconds=0.0)
        assert mock_get.call_count == first_call_count  # No new calls
