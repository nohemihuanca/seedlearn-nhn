# SeedLearn Project State, August 2026

This note records the handoff state used to initialize `nohemihuanca/seedlearn-nhn`.

## Repository State

- `seedlearn-nhn` was initialized from the former `dev` branch because that branch contains the packageized codebase (`src/seedlearn`), configs, docs, tests, trait-grading reports, and BioCLIP 2 defaults.
- The former `main` branch is older and smaller, but contains material that still needs review before porting.
- The new repository intentionally uses a clean single-commit history because GitHub push protection blocked the old history: several historical commits contained OpenAI API keys.
- Mitch's remote/repository should remain untouched and available only as a reference.

## Data State

The large project data are not stored in Git. The repository `data` entry is a symlink to:

```text
/nfs/roberts/project/pi_lsc4/shared/seedlearn/data
```

Verified cluster assets:

| Asset | Location | Status |
|-------|----------|--------|
| Current sorted image catalog | `data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv` | Current catalog documented in `docs/data.md` |
| BioCLIP 2 multi-rank embeddings | `data/embeddings/2026-01-29_v2026-01-29_12K/features.npz` | Verified `(10407, 768)` |
| BioCLIP 2 family cache | `data/embeddings/2026-01-29_v2026-01-29_12K/family_features.npz` | Verified `(10407, 768)` |
| Old BioCLIP 1 SimpleShot cache | `data/experiments/simpleshot/2025-10-23_v2025-10-23_12K/cache/family_features.npz` | Verified `(8377, 512)` |
| Old BioCLIP 1 SimpleShot results | `data/experiments/simpleshot/2025-10-23_v2025-10-23_12K/results/` | Metrics/reports exist for family and species k-shot runs |

The 2026-01-29 BioCLIP 2 embeddings exist, but no 2026-01-29 SimpleShot metrics were found under `data/experiments` during the August 2026 check.

## Modeling State

- BioCLIP 1 baselines/results exist for the 2025-10-23 dataset.
- BioCLIP 2 embeddings exist for the 2026-01-29 dataset.
- The next visual-classification milestone is to run SimpleShot on the BioCLIP 2 embeddings at family level first, then genus/species if time permits.
- The expected first report should compare old BioCLIP 1 SimpleShot results against BioCLIP 2 SimpleShot results using matching split/rank conventions where possible.

## Trait-Grading State

The repository includes trait-grading outputs from July 2026.

The human grading report at `trait_grading/reports/2026-07-06_153457/human_grading_report.html` summarizes:

| Axis | Macro rate | Macro kappa | Pairs compared |
|------|------------|-------------|----------------|
| model vs Roni | 0.68 | 0.10 | 1672 |
| model vs Carmen | 0.68 | 0.13 | 1870 |
| Roni vs Carmen | 0.79 | 0.45 | 1774 |

Interpretation: the model extracts some image-visible traits, but human-human agreement remains substantially stronger. A useful next analysis is a trait reliability table: traits humans agree on, traits the model matches well, and traits that are not reliable from the available photos.

## Immediate Priorities

1. Create a branch or issue set for careful old-`main` consolidation.
2. Review old `main` private validation files before porting; do not blindly publish `_PRIVATE` keys or raw human annotation files.
3. Run or submit a BioCLIP 2 SimpleShot job using `data/embeddings/2026-01-29_v2026-01-29_12K/features.npz`.
4. Generate a lightweight embedding visualization from BioCLIP 2 coordinates, preferably PCA/UMAP colored by family.
5. Check whether expert/student validation images overlap with the training images before using them as a locked test set.
