# Former `main` Branch Consolidation Inventory

This inventory captures material seen on the former `main` branch that was not ported automatically into the clean `seedlearn-nhn` snapshot.

## Port Carefully

These areas may contain useful project history or validation inputs, but should be reviewed before adding to `seedlearn-nhn`:

| Area | Examples | Review reason |
|------|----------|---------------|
| Human validation inputs | `human_validation/data/annotations/*.xlsx`, `human_validation/data/keys/*_PRIVATE.csv` | Potentially sensitive/private; confirm sharing rules and overlap with training data before publishing |
| Old workshop pipelines | `workshop_pipeline/step_1`, `workshop_pipeline/step_1_cloudbank`, `workshop_pipeline/step_2` | Useful prompt/prototype history, but likely superseded by `src/seedlearn` and `trait_grading` |
| Old BioCLIP/SimpleShot model scripts | `models/baseline/*`, `models/lightweight/simpleshot/*` | Some concepts are now represented under `src/seedlearn` and `scripts`; inspect before duplicating |
| Old iNaturalist/sort scripts | `iNaturalist/*` | Current repo has `scripts/download_inaturalist.py`, `scripts/sort_inaturalist.py`, and `docs/inaturalist.md`; compare for missing bug-fix notes |
| Scraper/literature outputs | `scrape_data/_results*`, notebooks, extraction scripts | May be historical source data; avoid importing large/generated outputs unless they are the canonical trait dataset |

## Do Not Port Blindly

- Files whose names include `_PRIVATE`.
- Generated result directories unless they are intentionally preserved as small benchmark artifacts.
- Historical scripts that hard-code local credentials, API keys, or old absolute paths.
- Large images or model artifacts; keep those on NFS and document paths instead.

## Old `main` Notes Worth Preserving

- The older README described SeedLearn as using iNaturalist citizen-science data. That wording is inaccurate for the core dataset: the seedling images were field-collected by the project and hosted/managed through iNaturalist. The new README uses field-collected image-set wording.
- Old `main` documented iNaturalist Project 228504 and the NFS root `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data`; the current docs retain those data pointers.
- Old `main` had an initial model overview naming closed-set, blind, and SimpleShot tracks. The current package keeps these as pipeline/clip concepts, but a future doc can map old script names to current commands.

## Recommended Next Consolidation Pass

1. Compare old `main` private validation keys against `trait_grading/keys` and decide which representation is canonical.
2. Compare old `workshop_pipeline` prompts/results against `configs/experiments` and `trait_grading/reports`.
3. Review old `iNaturalist/data-update-2025-10-23/FUZZY_MATCHING_BUG_REPORT.md` and port its lessons into `docs/inaturalist.md` if still relevant.
4. Archive or link old `models/lightweight/simpleshot` docs only if they contain workflow details missing from `docs/scripts.md` and `src/seedlearn/clip`.
