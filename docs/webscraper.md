# Web Scraper: STRI Panama Biota Trait Extraction

## Purpose

Extract morphological trait data from the Smithsonian Tropical Research Institute
(STRI) Panama Biota portal for use in SeedLearn's classification pipeline — specifically
to augment Stage 3 (Literature-Based Trait Retrieval) with structured trait vectors
and natural-language descriptions.

## Source: Panama Biota Portal

**Platform**: Symbiota CMS (open-source biodiversity data management)
**Base URL**: `https://panamabiota.org/stri/`

### Key Endpoints

| Endpoint | URL Pattern | Data Available |
|----------|-------------|----------------|
| **BCI Checklist** | `checklist.php?clid=180&pid=20` | 1,886 taxa (174 families, 850 genera, 1,869 species), paginated across 4 pages, voucher specimens |
| **Identification Key** | `ident/key.php?cl=59&proj=10&dynclid=0&taxon=All+Species` | 8,716 Panama dicot species with morphological trait filters |
| **Species Detail** | `taxa/index.php?taxon={taxon_id}&clid={checklist_id}` | Per-species descriptions, media, distribution maps |

### Available Morphological Trait Filters (Identification Key)

The identification key exposes **15 trait categories** as filterable attributes:

| Category | Options |
|----------|---------|
| **Plant Habit** | tree, shrub, terrestrial herb, vine-liana, epiphyte-hemiepiphyte, succulent, parasite-saprophyte, aquatic herb |
| **Latex** | present, absent |
| **Armature** | unarmed, stem or leaves armed |
| **Leaf Type** | simple, compound, pinnatifid-pinnatisect |
| **Leaf Arrangement** | alternate, opposite, whorled or clustered, fascicled, basal rosette |
| **Leaf Margin** | entire, toothed, lobed |
| **Stipules** | present, absent |
| **Leaf Punctations** | present, absent |
| **Glands on Blade/Petiole** | present, absent |
| **Flower Symmetry** | actinomorphic, zygomorphic |
| **Ovary Placement** | inferior, partially inferior, superior |
| **Carpel Number** | 1, 2, 3, 4, 5, 6, 7 or more |
| **Sex** | bisexual, unisexual |
| **Fruit** | dry, fleshy |

Filter mechanism: form-based submission with URL parameters (`attr[]=category-option`).
Applying filters returns a filtered species list grouped by family.

## Architecture

### Scraping Strategy

**Filter-inverted approach**: Instead of visiting thousands of individual species pages,
we fetch the identification key page once per filter option. Each filter request returns
all species matching that trait value. By collecting which species appear under each
filter, we build a complete presence/absence matrix without per-species crawling.

For a key with N filter options, this requires N+1 requests (N filtered + 1 unfiltered)
rather than S requests (one per species). For Panama Dicots: 45 requests vs 8,716.

### Package Structure

```
src/seedlearn/scraper/           # Installable package
├── __init__.py                  # Public API exports
├── schema.py                    # FilterOption, FilterCategory, IdentificationKey,
│                                # SpeciesEntry dataclasses + key registry
├── client.py                    # STRIClient: HTTP with caching + rate limiting
├── parser.py                    # HTML parsing for species lists and filter schemas
└── matrix.py                    # Trait matrix construction with uncoded detection

scripts/
├── scrape_stri_traits.py        # CLI: scrape one or all identification keys
└── merge_stri_trait_sources.py  # CLI: merge per-key matrices into unified database
```

### Data Layout (NFS via `data/` symlink)

```
data/traits/stri_web_keys/
├── raw_html/                                       # Cached raw HTML responses
│   ├── cl59_panama_dicots/
│   │   ├── unfiltered_all_species.html
│   │   ├── filtered_attr_1-1_habit_tree.html
│   │   └── ...
│   └── cl70_trees_gamboa_and_canal/
│       └── ...
│
├── per_key_trait_matrices/                          # One CSV + JSON per key
│   ├── cl59_panama_dicots_trait_matrix.csv
│   ├── cl59_panama_dicots_scrape_metadata.json
│   └── ...
│
└── merged/                                         # Final merged output
    ├── stri_all_sources_merged_trait_matrix.csv     # Source-tagged columns
    ├── stri_all_sources_consensus_trait_matrix.csv  # Consensus (any-true)
    └── merge_report.json                           # Merge statistics
```

### Target Identification Keys

| Key (cl) | Slug | Species | Trait Categories |
|----------|------|---------|------------------|
| 59 | `panama_dicots` | 8,716 | 15 (full set) |
| 60 | `ferns_and_allies_of_panama` | 990 | 1 (habit only) |
| 61 | `monocots_of_panama` | 2,324 | 1 (habit only) |
| 178 | `bci_eudicots_magnoliids_basal_angiosperms` | 1,189 | 13 |
| 185 | `complete_tree_species_of_panama` | 3,039 | 14 |
| 71 | `ctfs_tree_atlas_of_panama` | 2,688 | 13 |
| 72 | `ctfs_liana_atlas_of_panama` | 837 | 8 |
| 65 | `campana_national_park` | 759 | 7 |
| 85 | `soberania_national_park_plants` | 932 | 15 (full set) |
| 70 | `trees_gamboa_and_canal` | 196 | 10 |
| 66 | `myrtaceae_of_panama` | 129 | 11 (family-specific traits) |

Total estimated requests: ~320 at 1-second delay = ~6 minutes per full scrape.

## CLI Usage

### Scraping Traits

```bash
source .venv/bin/activate

# Scrape all 11 identification keys
python scripts/scrape_stri_traits.py --keys all

# Scrape specific keys
python scripts/scrape_stri_traits.py --keys 59 178 185

# Scrape smallest key first (for testing)
python scripts/scrape_stri_traits.py --keys 70 --verbose

# Force re-fetch (ignore cached HTML)
python scripts/scrape_stri_traits.py --keys 59 --force-refresh

# Custom delay between requests
python scripts/scrape_stri_traits.py --keys all --delay 2.0
```

### Merging Sources

```bash
# Merge all per-key matrices into unified database
python scripts/merge_stri_trait_sources.py --verbose

# Custom paths
python scripts/merge_stri_trait_sources.py \
    --input-dir data/traits/stri_web_keys/per_key_trait_matrices \
    --output-dir data/traits/stri_web_keys/merged
```

## Data Formats

### Per-Key Trait Matrix CSV

Each per-key CSV has one row per species and columns:

| Column Type | Pattern | Example | Values |
|-------------|---------|---------|--------|
| Identity | `taxon_id`, `family`, `scientific_name` | `61885, Acanthaceae, Aphelandra arnoldii` | — |
| Trait option | `{category}__{option}` | `habit__tree` | 0 or 1 (multi-label) |
| Uncoded flag | `{category}__uncoded` | `habit__uncoded` | 0 or 1 |

**Multi-label**: A species can be 1 for both `habit__tree` and `habit__shrub`.

**Uncoded detection**: If ALL options in a category are 0 for a species, the
`{category}__uncoded` column is set to 1. This means "data not entered" (not "absent
from all options"). A plant must have a habit — all-zero means nobody coded it.

### Merged Trait Matrix CSV

Source-tagged columns for cross-source comparison:

| Column Type | Pattern | Example |
|-------------|---------|---------|
| Source-specific trait | `{trait}__{source_slug}` | `habit__tree__cl59_panama_dicots` |
| Source-specific uncoded | `{category}__uncoded__{source_slug}` | `habit__uncoded__cl178_bci_...` |
| Consensus trait | `{trait}__consensus` | `habit__tree__consensus` |

**Consensus logic**: 1 if ANY source reports present; 0 if at least one source has
coded data and all say 0; NaN if no source has coded data for this taxon x trait.

### Scrape Metadata JSON

Each per-key matrix has a companion JSON with:
- `key_slug`, `species_count_reported`, `species_count_scraped`
- `trait_columns` list, `categories` with option details
- `scraped_at` UTC timestamp

## Prior Scraping Work

### Approach A: requests + BeautifulSoup (simple)

**Location**: `/nfs/roberts/project/pi_lsc4/shared/seedlearn/grace-bkp/scripts/si/`

| Script | Purpose |
|--------|---------|
| `append_si_text_to_new_data.py` | Scrape taxon descriptions from species detail pages; regex taxon ID extraction |
| `desc/create_species_descriptions.py` | Fetch tabbed descriptions (Bocas DB, Tree Atlas, BCI Flora) per species |
| `char/scrape.py` | Parse pre-downloaded HTML for binary leaf traits (arrangement, type) |
| `char/merge.py` | Join scraped traits with image metadata |
| `merge_leaf_characteristics.py` | Validate and merge binary leaf trait consistency |
| `merge_specific_descriptions.py` | Transfer descriptions across dataset versions |

**Limitation**: Only extracted 4 description types and 2 binary leaf traits. Did not
use the identification key's 15-category morphological filter system.

### Approach B: Playwright async + proxy (complex)

**Location**: `/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/auto-emma/src/emma/`

Designed for heavily-protected financial data sites. The STRI portal is an academic
Symbiota site with no apparent anti-scraping measures — this level of complexity is
unnecessary.

## Technical Notes

- **Politeness**: 1-second minimum delay between requests (academic resource)
- **Caching**: Raw HTML cached locally — re-runs never re-fetch
- **User-Agent**: `SeedLearn-TraitScraper/1.0` with academic research note
- **403 handling**: Site requires a User-Agent header (rejects bare requests)
- **HTML structure**: Symbiota CMS uses `div#key-taxa` for species lists,
  `div#char{N}` (N varies per key) with `span.dynamlang` for filter categories,
  `div.cs-div` with `span` labels for checkbox options
- **Output**: All data on NFS via `data/` symlink — never committed to git
