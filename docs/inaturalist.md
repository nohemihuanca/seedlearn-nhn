# iNaturalist Data Pipeline

**Navigation**: [SeedLearn](../README.md) > **iNaturalist Pipeline**

---

## Overview

The data pipeline acquires and prepares tropical seedling images from the [Yale-STRI AI Seedling Project](https://www.inaturalist.org/projects/yale-stri-ai-seedling-project) (Project 228504) in three stages:

```
iNaturalist API ──→ download_inaturalist.py ──→ raw observations
                                                      │
inat_metadata CSV ──→ convert_inaturalist.py ──→ species CSV
                                                      │
                    sort_inaturalist.py ──→ ML-ready hierarchical dataset
```

Scripts live in `scripts/`, species lists in `configs/species_lists/`:
```
scripts/
├── download_inaturalist.py     # Stage 1: API download
├── convert_inaturalist.py      # Stage 2: CSV conversion
└── sort_inaturalist.py         # Stage 3: Hierarchical sorting

configs/species_lists/
├── inat_metadata_FINAL_NHN_01_2025.csv   # Source metadata
├── YPS_seedling_spp_list_code_07_09_25.csv
├── YPS_seedling_spp_list_code_10_22_25.csv
└── YPS_seedling_spp_list_derived_01_29_26.csv
```

---

## Stage 1: Download (`download_inaturalist.py`)

Downloads all observations from Project 228504 with retry logic and progress tracking.

### Usage

```bash
python scripts/download_inaturalist.py
```

No CLI arguments — paths are configured in-script:
- **Source**: iNaturalist API v1 (`https://api.inaturalist.org/v1/observations`)
- **Output**: `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/inaturalist/project_228504/`

### Output Structure

```
project_228504/
└── {username}/
    └── {observation_id}/
        ├── observation_metadata.json     # Observation-level metadata
        ├── photo_{id}.jpg                # Image files
        └── photo_{id}_metadata.json      # Per-photo metadata
```

### Download Statistics (August 2025)

| Metric | Value |
|--------|-------|
| Total observations | 2,142 |
| Total photos | 12,641 |
| Downloaded | 9,896 (78.3%) |
| Skipped (already present) | 2,745 |
| Failed | 0 |

### Key Features

- **Idempotent**: Skips already-downloaded photos (safe to re-run)
- **Retry logic**: Up to 3 retries per photo, tries multiple URL sizes (original → large → medium)
- **Rate limiting**: 1-second delay between requests
- **Logging**: All activity logged to `download_log_{timestamp}.txt`

### Contributing Users

| Username | Data Volume |
|----------|------------|
| `mariagallegos` | 8.5 GB |
| `crono_secuencia_2` | 2.7 GB |
| `crono_secuencia_3` | 1.4 GB |
| `crono_secuencia_4` | 702 MB |
| `crono_secuencia5` | 251 MB |
| `biancolini23` | 127 MB |
| `nohemi_huanca_nunez` | 125 KB |

---

## Stage 2: Convert Metadata (`convert_inaturalist.py`)

Converts the photo-level iNaturalist metadata file to the individual-level legacy CSV format expected by `sort_inaturalist.py`.

### Usage

```bash
python scripts/convert_inaturalist.py \
    --input configs/species_lists/inat_metadata_FINAL_NHN_01_2025.csv \
    --output configs/species_lists/YPS_seedling_spp_list_derived_01_29_26.csv
```

### Conversion Process

1. **Drop nulls**: Removes ~7 rows with missing taxonomy
2. **Deduplicate**: Groups 12,640 photo rows → 2,132 unique individuals (keyed on `notes_code_clean2`)
3. **Map columns** to legacy format (fixed mapping, no conditional logic)

### Column Mapping

| Output (legacy) | Source Column | Notes |
|------------------|---------------|-------|
| `ID_YPS` | `notes_code_clean2` | Individual plant ID parsed from observation notes |
| `SPP` | `spp6_fixed` | 6-letter code after synonym correction |
| `GENUS` | `genus` | Parsed from `accepted_name` — verified exact match |
| `SPECIES` | `species` | Parsed from `accepted_name` — verified exact match |
| `FAMILY` | `accepted_family` | Accepted botanical family |
| `LIANA` | `habit` | `Climbing` → 1, `Freestanding` → 0 |
| `FOREST` | *(not in source)* | Hardcoded `"Unknown"` |

**Key**: `genus` and `species` are always an exact split of `accepted_name` across all rows. The `scientific_name_final2` column (field ID, possibly a synonym) is never used in output.

### Species CSV Format

| Column | Example | Description |
|--------|---------|-------------|
| `ID_YPS` | `PP123` | Individual plant ID |
| `SPP` | `INGSP` | 6-letter species code |
| `GENUS` | `Inga` | Genus (title case) |
| `SPECIES` | `spectabilis` | Species epithet (lowercase) |
| `FAMILY` | `Fabaceae` | Family (title case) |
| `LIANA` | `0` | 0=freestanding, 1=climbing |
| `FOREST` | `Unknown` | Forest site |

---

## Stage 3: Sort (`sort_inaturalist.py`)

Hierarchical ML-optimized sorting with strict taxonomy matching.

### Usage

```bash
# Default configuration
python scripts/sort_inaturalist.py

# Custom paths
python scripts/sort_inaturalist.py \
    --source /path/to/source \
    --output-base /path/to/output \
    --species-csv configs/species_lists/YPS_seedling_spp_list_derived_01_29_26.csv
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `.../inaturalist/project_228504` | Downloaded observations directory |
| `--output-base` | `.../data/raw` | Base output directory |
| `--species-csv` | `configs/species_lists/YPS_seedling_spp_list_derived_01_29_26.csv` | Species taxonomy mapping |

### Individual ID Extraction

The individual ID comes from the `description` field of `observation_metadata.json`:

```json
{
  "description": "PP123",
  "observed_on": "2024-01-10",
  "user": "bianco"
}
```

### Taxonomy Matching (Strict Only)

After the 2025-10-23 bug fix, matching uses **strict mode only**:

1. Case-insensitive exact match
2. Punctuation normalization (remove `.`, `-`, `_`) and retry
3. No fuzzy/substring matching — unmatched IDs logged as `MISSING_TAXONOMY`

This prevents the 6.57% data contamination caused by the previous substring matching bug.

### Taxonomy Normalization

| Field | Rule | Example |
|-------|------|---------|
| Family | Title case | `fabaceae` → `Fabaceae` |
| Genus | Title case | `inga` → `Inga` |
| Species | Lowercase | `Spectabilis` → `spectabilis` |
| Special chars | → underscore | `Inga.sp` → `Inga_sp` |

### Output Structure

See [Data Reference](data.md) for the full `sorted_12K/` tree layout.

### Validation Checks

| Check | Threshold | On Failure |
|-------|-----------|------------|
| Metadata exists | Required | `NO_METADATA` error |
| Individual ID present | Required | `NO_ID` error |
| Taxonomy mapping found | Required | `MISSING_TAXONOMY` error |
| Image files exist | >= 1 | `NO_IMAGES` error |
| Training images | >= 3 | `INSUFFICIENT_IMAGES` warning, skip |
| Image file size | 100 KB – 10 MB | `CORRUPTED_IMAGE` warning |

### Issue Tracking

All issues logged to `all_issues_v{date}_12K_{timestamp}.csv`:

| Issue Type | Severity | Meaning |
|------------|----------|---------|
| `NO_METADATA` | ERROR | Missing `observation_metadata.json` |
| `NO_ID` | ERROR | No individual ID in description |
| `MISSING_TAXONOMY` | ERROR | ID not in species CSV |
| `NO_IMAGES` | ERROR | No JPG files found |
| `INSUFFICIENT_IMAGES` | WARNING | < 3 training images |
| `CORRUPTED_IMAGE` | WARNING | Invalid image file |
| `UNEXPECTED_FILE_COUNT` | WARNING | Non-standard file count |
| `INCOMPLETE_SET` | INFO | Known 11-file variation |

### Latest Run (January 29, 2026)

| Metric | Value |
|--------|-------|
| Observations processed | 2,121 / 2,142 |
| Training images | 10,407 |
| Verification images | 2,121 |
| Unique families | 52 |
| Unique genera | 114 |
| Unique species | 164 |
| Unique individuals | 2,112 |
| Issues | 193 INFO, 41 WARNING, 19 ERROR |

---

## Re-Sort Procedure

When the species CSV is updated (new observations, taxonomy changes):

```bash
# 1. If metadata changed, regenerate the derived CSV
python scripts/convert_inaturalist.py \
    --input configs/species_lists/inat_metadata_FINAL_NHN_01_2025.csv \
    --output configs/species_lists/YPS_seedling_spp_list_derived_YYYY_MM_DD.csv

# 2. Run sort with new CSV
python scripts/sort_inaturalist.py \
    --species-csv configs/species_lists/YPS_seedling_spp_list_derived_YYYY_MM_DD.csv

# 3. Verify output
cat .../raw/YYYY-MM-DD/sorted_12K/metadata/processing_summary.json | python -m json.tool

# 4. Review issues
head .../raw/YYYY-MM-DD/sorted_12K/metadata/all_issues_v*.csv

# 5. Update DEFAULT_CATALOG in src/seedlearn/data/constants.py
# 6. Re-extract embeddings and re-run experiments
```

---

## Filename Conventions

**Training images**: `{Family}_{Genus}_{species}_{individual_id}_{NNN}.jpg`
- Example: `Fabaceae_Inga_spectabilis_PP123_001.jpg`

**Verification images**: `{Family}_{Genus}_{species}_{id}_verification_{user}_{position}.jpg`
- Example: `Fabaceae_Inga_spectabilis_PP123_verification_bianco_last.jpg`

**User tag positions**:
- **Tag first**: `crono_secuencia5`, `crono_secuencia_4`
- **Tag last**: `biancolini23`, `crono_secuencia_2`, `crono_secuencia_3`, `bianco`, `maria`
