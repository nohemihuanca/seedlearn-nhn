# Trait Grading Study

Human-annotation grading of the pipeline's Vision-LLM morphological trait
extraction (Stage 1), plus grading of a botanist's photo-based species IDs.

See [`docs/human-grading.md`](../docs/human-grading.md) for the full methodology
and the reproducible end-to-end runbook.

## Layout

| Path | Tracked? | Contents |
|------|----------|----------|
| `annotations/roni_bianco.xlsx` | ✅ committed | Roni Bianco's per-view trait annotations **+** family/genus/species ID predictions |
| `annotations/carmen.xlsx` | ✅ committed | Carmen's per-view trait annotations (traits only) |
| `keys/curator_taxonomic_key.csv` | ✅ committed | `anonymous_id → specimen + true taxonomy` |
| `keys/image_key.csv` | ✅ committed | `anonymous_id + view → original image` |
| `model_run/<timestamp>/` | ✅ committed | Fresh Stage-1 model trait predictions for the 114 annotated specimens |
| `reports/` | ✅ committed | Generated CSV + HTML grading reports |

Everything here is tracked and shared with collaborators. The `anonymous_id →
specimen` mapping in `keys/` is known information for everyone working on this
repo, not something to hide.
