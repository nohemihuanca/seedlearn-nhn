# Data Reference

**Navigation**: [SeedLearn](../README.md) > **Data Reference**

---

## NFS Layout

All project data lives on shared NFS storage at `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/` (~64 GB). The repo's `data/` directory is a symlink to this location.

```
/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/
├── raw/                    (50 GB)  ML-ready sorted images + catalogs
├── embeddings/             (48 MB)  BioCLIP 2 cached features (.npz)
├── splits/                 (24 MB)  Train/val/test partitions
├── traits/                (266 MB)  Literature-extracted morphological traits
├── experiments/                     SimpleShot experiment results only
├── inaturalist/            (14 GB)  Raw iNaturalist downloads
└── logs/                            Runtime logs (SLURM, pipeline)
```

**Repo symlink**: `seedlearn-dev/data` -> `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data`

---

## `raw/` — Sorted Image Datasets

Date-stamped directories, each containing a `sorted_12K/` tree produced by the [sort pipeline](inaturalist.md).

| Version | Size | Notes |
|---------|------|-------|
| `2025-08-18` | 13 GB | Initial 10K dataset |
| `2025-10-07` | 13 GB | First 12K sort |
| `2025-10-23` | 11 GB | Bug-fix re-sort (fuzzy matching removed) |
| `2026-01-29` | 14 GB | **Current** — accepted taxonomy, 164 species |

### `sorted_12K/` structure

```
sorted_12K/
├── training/
│   └── by_family/
│       └── {Family}/
│           └── by_genus/
│               └── {Genus}/
│                   └── by_species/
│                       └── {species}/
│                           └── {individual_id}/
│                               ├── {Family}_{Genus}_{species}_{id}_{NNN}.jpg
│                               └── individual_metadata.json
├── verification/
│   └── {Family}/{Genus}/{species}/{individual_id}/
│       └── {Family}_{Genus}_{species}_{id}_verification_{user}_{position}.jpg
└── metadata/
    ├── species_catalog_v{date}_12K_{timestamp}.csv   # Master catalog
    ├── all_issues_v{date}_12K_{timestamp}.csv        # QC issue log
    ├── processing_summary.json                        # Run statistics
    ├── sorting_log.json                               # Full audit trail
    └── normalization_log.csv                          # Taxonomy normalizations
```

### Species Catalog CSV Schema (21 columns)

The species catalog is the master reference for ML experiments:

| Column | Type | Description |
|--------|------|-------------|
| `data_version` | str | Version tag, e.g. `v2026-01-29_12K` |
| `sorting_timestamp` | str | Processing timestamp |
| `source_project` | str | Always `project_228504` |
| `ID_YPS` | str | Individual plant ID (e.g. `PP123`) |
| `SPP` | str | 6-letter species code (e.g. `INGSP`) |
| `FAMILY` | str | Botanical family (title case) |
| `GENUS` | str | Genus (title case) |
| `SPECIES` | str | Species epithet (lowercase) |
| `LIANA` | int | `1` = climbing, `0` = freestanding |
| `FOREST` | str | Forest site (currently `Unknown`) |
| `training_path` | str | Relative path to training images |
| `training_absolute_path` | str | Absolute NFS path |
| `verification_path` | str | Relative path to verification image |
| `verification_absolute_path` | str | Absolute NFS path |
| `verification_user` | str | Photographer who took verification image |
| `verification_position` | str | `first` or `last` (tag image position) |
| `training_image_count` | int | Number of training images |
| `verification_image_count` | int | Number of verification images |
| `total_image_count` | int | Total images for this individual |
| `has_minimum_images` | bool | `True` if >= 3 training images |
| `date_processed` | str | Processing date/time |

**Current catalog**: `data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv`

---

## `traits/` — Morphological Trait Extractions

Date-stamped directories produced by the literature extraction pipeline. A `latest` symlink always points to the active version.

```
traits/
├── 2025-11-10/                           # Original pt1 extraction
│   ├── _stats.json
│   ├── concatenated_output_traits.csv
│   ├── concatenated_output_nlp.csv
│   ├── cleaned_concatenated_output_traits.csv
│   ├── cleaned_compressed_concatenated_output_traits.csv
│   ├── concatenated_trait_metadata.csv
│   ├── covered_taxa_traits.csv
│   ├── uncovered_taxa_traits.csv
│   └── ...
├── 2026-02-12/                           # Updated extraction with synonyms
│   ├── results_pt2/
│   ├── results_combined/
│   └── results_combined_accepted_names_only/   # <- latest points here
│       ├── _stats.json
│       ├── concatenated_output_traits.csv
│       ├── concatenated_output_nlp.csv
│       ├── cleaned_compressed_concatenated_output_traits.csv
│       └── ...
└── latest -> 2026-02-12/results_combined_accepted_names_only
```

### Extraction Versions

| Version | Taxa Covered | Target Taxa | Coverage | Notes |
|---------|-------------|-------------|----------|-------|
| 2025-11-10 | 326 | 394 | 82.7% | Original pt1 extraction |
| 2026-02-12 | 351 | 422 | 83.2% | Updated with synonym handling |

### Trait CSV Schema (24 morphological traits + metadata)

```
rank, taxon, leaf relative position, leaf spacing, leaf complexity,
compound leaf type, number of leaflets, leaflet arrangement, leaf margin,
leaf shape, leaf apex, leaf base, venation type, secondary veins,
leaf surface features, leaf surface trichomes, petiole length,
petiole features, stem type, stem trichomes, stem color, stem texture,
stipules, latex, pulvinus, tendrils, other_traits
```

### NLP CSV Schema

```
rank, taxon, description, source, source-details, scraping-notes,
scraping-date-time, chunk
```

### Coverage Statistics (2026-02-12, accepted names only)

| Metric | Traits | NLP |
|--------|--------|-----|
| Extracted taxa | 351 | 362 |
| Target taxa | 422 | 422 |
| Coverage | 83.2% | 85.8% |
| Families covered | 52/64 (81.3%) | 62/64 (96.9%) |
| Genera covered | 130/149 (87.3%) | 128/149 (85.9%) |
| Species covered | 169/209 (80.9%) | 172/209 (82.3%) |

---

## `embeddings/` — BioCLIP 2 Features

Cached BioCLIP 2 image embeddings, versioned by catalog. Promoted from the old `experiments/simpleshot/.../cache/` location for discoverability.

```
embeddings/
└── 2026-01-29_v2026-01-29_12K/
    ├── features.npz              # Multi-rank cache (10407×768, L2-normalized)
    ├── features_meta.json        # Taxonomy cross-reference sidecar
    ├── family_features.npz       # Legacy single-rank caches
    ├── genus_features.npz
    └── species_features.npz
```

---

## `splits/` — Data Partitions

Train/val/test splits for few-shot evaluation, versioned by catalog. Promoted from the old `experiments/simpleshot/.../splits/` location.

```
splits/
└── 2026-01-29_v2026-01-29_12K/
    ├── family/
    │   ├── split_seed42.json
    │   └── ... (seeds 42-46)
    ├── genus/
    │   └── split_seed42.json ...
    └── species/
        └── split_seed42.json ...
```

### Split Configuration

- **Split type**: Individual-level via `GroupShuffleSplit` — splits by `ID_YPS` (individual plant ID), so all images of the same individual stay in the same partition. This prevents data leakage from multi-image specimens.
- **Ratios**: 70% train / 15% val / 15% test
- **Seeds**: 5 random seeds (42–46) for variance estimation
- **Key column**: `ID_YPS` — unique individual identifier used as the group key

### Per-Rank Summary

| Rank | Classes | Individuals | Images | Train | Val | Test | Max k-shot |
|------|---------|-------------|--------|-------|-----|------|------------|
| Family | 52 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 10 |
| Genus | 114 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 3 |
| Species | 164 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 3 |

Values shown as `individuals (images)`. Counts are for seed 42; other seeds have similar distributions due to the fixed 70/15/15 ratio.

### Per-Class Statistics (Seed 42)

| Rank | Split | Min samples/class | Max samples/class | Mean samples/class |
|------|-------|-------------------|--------------------|--------------------|
| Family | Train | 10 | 762 | 139.8 |
| Family | Val | 5 | 210 | — |
| Family | Test | 4 | 126 | — |
| Genus | Train | 3 | 540 | 63.8 |
| Genus | Val | 4 | 130 | — |
| Genus | Test | 3 | 85 | — |
| Species | Train | 3 | 287 | 44.3 |
| Species | Val | 4 | 63 | — |
| Species | Test | 3 | 64 | — |

### K-Shot Feasibility

**Max k-shot** is the largest value of k where *every* class in the rank is guaranteed to have at least k training samples. This determines which k-shot experiments can run without dropping any classes.

Recommended k-shot values per rank:

- **Family** (max k=10): `--k-shots 1 3 5 10`
- **Genus** (max k=3): `--k-shots 1 3`
- **Species** (max k=3): `--k-shots 1 3`

Higher k values are possible if you accept dropping low-sample classes from the evaluation.

### Class Warnings

Both genus and species ranks have one class with a low training sample count (seed 42):

- **Genus**: *Acalypha* — 3 train, 10 val, 14 test samples
- **Species**: *Acalypha diversifolia* — 3 train, 10 val, 14 test samples

This imbalance occurs because this individual has many images (27 total) but only one individual in the dataset. Since splitting is individual-level, all images land in a single partition, leaving the training set with the minimum allocation.

### Regenerating Splits

```bash
source .venv/bin/activate
python scripts/create_splits.py \
    --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_*.csv \
    --rank family --split-type individual --num-seeds 5 --start-seed 42
```

Repeat with `--rank genus` and `--rank species` for the other ranks.

---

## `experiments/` — Classification Results

Results from SimpleShot few-shot learning and baseline classifiers. Embeddings and splits have been promoted to their own top-level directories.

```
experiments/
├── blind/                          # Zero-shot BioCLIP classification
│   └── 2025-10-23_v2025-10-23_12K/
│       └── results/{family,species}/
├── closed_set/                     # Closed-set BioCLIP classification
│   └── 2025-10-23_v2025-10-23_12K/
│       └── results/{family,species}/
└── simpleshot/                     # Few-shot learning experiments
    └── 2026-01-29_v2026-01-29_12K/
        └── results/                # Experiment outputs
            └── {rank}/{k}_shot/split_seed{N}/
                ├── metrics.json
                ├── predictions.csv
                ├── per_class_metrics.csv
                ├── confusion_matrix.csv
                ├── support_set.json
                └── experiment_info.json
```

### Version auto-derivation

Artifact paths are automatically derived from the catalog filename by `get_catalog_version()` in `seedlearn.data.constants`, combined with per-artifact base path constants:

```
species_catalog_v2026-01-29_12K_20260129_*.csv
                 └── extracts: 2026-01-29_v2026-01-29_12K
```

| Artifact | Base constant | Resolved path |
|----------|---------------|---------------|
| Embeddings | `SHARED_EMBEDDINGS` | `data/embeddings/2026-01-29_v2026-01-29_12K/` |
| Splits | `SHARED_SPLITS` | `data/splits/2026-01-29_v2026-01-29_12K/` |
| Results | `SHARED_EXPERIMENTS` | `data/experiments/simpleshot/2026-01-29_v2026-01-29_12K/` |

---

## `inaturalist/` — Raw Downloads

Raw observations from iNaturalist Project 228504, organized by collector username.

```
inaturalist/
├── download_statistics.json
├── download_log_20250815_*.txt
└── project_228504/
    ├── biancolini23/       (127 MB)
    ├── crono_secuencia_2/  (2.7 GB)
    ├── crono_secuencia_3/  (1.4 GB)
    ├── crono_secuencia_4/  (702 MB)
    ├── crono_secuencia5/   (251 MB)
    ├── mariagallegos/      (8.5 GB)
    └── nohemi_huanca_nunez/ (125 KB)
```

Each observation directory contains:
```
{username}/{observation_id}/
├── observation_metadata.json
├── photo_{id}.jpg
└── photo_{id}_metadata.json
```

**Download stats** (August 2025): 2,142 observations, 12,641 photos (9,896 downloaded, 2,745 already present, 0 failures).

See [iNaturalist Pipeline](inaturalist.md) for download and sort workflow details.

---

## Benchmark Config Files

Ground truth and test configuration files for the VLM morphological extraction benchmark.
Located in `tests/benchmarks/configs/`.

| File | Description |
|------|-------------|
| `stage1_samples.json` | 21 benchmark specimens with 4-5 image paths each (103 total images). Keys are `{Family}_{Genus}_{species}[_N]` format. |
| `stage1_ground_truth.csv` | Full ground truth: 21 specimens × 24 traits with STRI source traceability. 5 traits auto-filled from STRI, 19 awaiting manual annotation. |
| `stage1_ground_truth_active.csv` | Active subset: 21 specimens × 5 STRI-covered traits (leaf_complexity, leaf_arrangement, leaf_margin, stipules, latex). Used by the scorer. |
| `stage1_trait_valid_values.csv` | Valid values reference for each of the 24 traits with annotation phase (stri/manual_priority/manual_backfill). |
| `stage1_test_results.json` | Synthetic predictions for integration testing the scorer. |

### Ground Truth Format

Each trait column has companion `{trait}_match_type` (exact or multi_label) and
`{trait}_stri_keys` (source STRI identification keys) columns. Multi-label entries
(e.g., `entire | toothed`) accept any matching value during scoring.

To add manual annotations: edit `stage1_ground_truth.csv`, fill in empty trait
cells, set `{trait}_source` to `manual`. The scorer operates on whatever ground
truth is available — empty cells are skipped.
