# Trait-Extraction Experiments (leaf-margin pilot)

Compare approaches to Stage-1 Vision-LLM trait extraction and measure which move
the needle, using **leaf margin** as the pilot trait. Each approach is a
**condition**; every condition is graded against the human-annotation backend
(`docs/human-grading.md`) and rolled up into one comparison report.

Plan of record: `docs/plans/2026-07-13-001-feat-leaf-margin-trait-experiments-plan.md`.

## What each lever tests

| ID | Model | Prompt / granularity | Few-shot | Isolates |
|----|-------|----------------------|----------|----------|
| **C0** | Qwen3-VL-32B (local) | `sys4`, all 24 traits | — | baseline (≈ production) |
| **C1** | Qwen3.6-35B-A3B (local) | `sys4` | — | model capability |
| **C2** | Qwen3-VL-32B | `margin_only` | — | focusing the prompt on one trait |
| **C3** | Qwen3-VL-32B | `margin_rich` | — | an enriched entire/toothed/lobed description |
| **C4** | Qwen3-VL-32B | `margin_rich` | 3 drawings | in-context visual exemplars |
| **K1** | GPT-5.4 (cloud) | all 24 traits | — | external: frontier model, all-traits |
| **K2** | GPT-5.1 (cloud) | one trait at a time | — | external: per-trait granularity |
| **K3** | GPT-5.1 (cloud) | one section at a time | — | external: per-section granularity |

C0–C4 are **local** (run through vLLM); K1–K3 are **external** collaborator runs
ingested from disk. The condition manifest is `configs/experiments/leaf_margin_ladder.yaml`.

Leaf margin is graded as a **coarse 3-way** (entire / toothed / lobed); serration
subtypes (serrate, dentate, crenate…) collapse to `toothed`. That is the level at
which gains register and it matches the production metric.

## Reproducible runbook

Activate the environment first: `source .venv/bin/activate`.

### 1. Ingest the external (Kaili) results

The cloud results live on `origin/main` under
`workshop_pipeline/step_1_cloudbank/results/`. Materialize them (a `main` worktree
is simplest) and adapt each into a `model_run/` dir:

```bash
git worktree add --detach /tmp/main-ro origin/main
R=/tmp/main-ro/workshop_pipeline/step_1_cloudbank/results
python scripts/ingest_workshop_results.py --source "$R/gpt-5.4/sys4_user1_results.json" \
    --out-dir trait_grading/model_run/K1_gpt-5.4_all-traits --label K1_gpt-5.4_all-traits \
    --model gpt-5.4 --granularity all_traits
python scripts/ingest_workshop_results.py --source "$R/gpt-5.1/per_trait/sys4_user1_trait7_results.json" \
    --out-dir trait_grading/model_run/K2_gpt-5.1_per-trait --label K2_gpt-5.1_per-trait \
    --model gpt-5.1 --granularity per_trait
python scripts/ingest_workshop_results.py --source "$R/gpt-5.1/section_C/sys4_user1_results.json" \
    --out-dir trait_grading/model_run/K3_gpt-5.1_per-section --label K3_gpt-5.1_per-section \
    --model gpt-5.1 --granularity per_section
```

The adapter parses each specimen's numbered-form answer with the pipeline's
`FormParser`, keys by the specimen id (the answer key's suffix == curator
`individual_code`), and writes the same per-specimen shape the grader reads —
no grader changes. (Trait 7 = leaf margin; section C = Leaf Morphology.)

### 2. Run the local conditions (GPU + vLLM required)

On a GPU node with a vLLM server up (`bash scripts/start_vllm.sh --model <id>`),
run each local condition as a **Stage-1-only** benchmark into a labeled dir. Only
Stage 1 is needed — grading reads only `stages.morphology.data.traits`:

```bash
CATALOG=data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv
python scripts/run_benchmark_pipeline.py \
    --catalog $CATALOG \
    --specimen-source trait_grading/keys/curator_taxonomic_key.csv \
    --skip classification trait_retrieval evidence_synthesis reasoning \
    --prompt-style margin_rich \
    --examples configs/experiments/leaf_margin_examples.json \
    --output-dir trait_grading/model_run/C4_image_fewshot_$(date +%Y%m%d_%H%M%S)
```

Swap `--prompt-style` / `--vlm-model` / `--examples` per the manifest. C1 needs the
upgraded model served (see the plan's U5); C4 needs the three exemplar drawings at
`trait_grading/exemplars/leaf_margin/{entire,toothed,lobed}.png` (referenced by
`configs/experiments/leaf_margin_examples.json`). Keep exemplars **single-view** —
3 exemplars + up to 6 specimen views must fit vLLM's `--limit-mm-per-prompt` of 10.

### 3. Compare and report

```bash
python scripts/compare_trait_experiments.py \
    --run "baseline=trait_grading/model_run/2026-07-06_134225" \
    --run "K1_gpt-5.4=trait_grading/model_run/K1_gpt-5.4_all-traits" \
    --run "K2_gpt-5.1_per-trait=trait_grading/model_run/K2_gpt-5.1_per-trait" \
    --run "K3_gpt-5.1_per-section=trait_grading/model_run/K3_gpt-5.1_per-section" \
    --baseline baseline \
    --out-dir trait_grading/reports/experiments/$(date +%Y-%m-%d_%H%M%S)
```

Outputs: `leaf_margin_per_axis.csv` (condition × κ-axis), `leaf_margin_summary.csv`
(one row per condition + STRI accuracy + provenance), and
`leaf_margin_comparison.html` (the synthesis report).

## How to read the report

- **Human ceiling.** Read model-vs-human agreement against the **Roni-vs-Carmen**
  ceiling (how well two trained botanists agree on margin), not against 100%.
- **Per-step deltas + McNemar.** The delta table is the real inferential object.
  Because every condition grades the same specimens, a paired **McNemar** test on
  per-specimen model-vs-Roni correctness is the correct significance test. A flat
  ranking would misattribute gains (C4 bundles three changes) — treat any ranking
  as descriptive only.
- **External-condition confound.** K1 is GPT-5.4 while K2/K3 are GPT-5.1, on
  *excerpted* prompts. Read granularity effects **within a model** (K2 vs K3), not
  across, and never read a cloud-vs-local gap as a lever effect.
- **κ colors.** Cells are colored by Cohen's κ (Landis & Koch bands). STRI is a
  separate match-any **accuracy** column (its multi-label reference makes a κ axis
  ill-defined).

## Current snapshot (baseline + Kaili)

Leaf margin, vs Roni; human ceiling = **92.9% / κ 0.796**:

| Condition | vs Roni | κ | STRI acc | Δ vs baseline (McNemar p) |
|-----------|---------|-----|----------|---------------------------|
| baseline Qwen3-VL-32B | 83.9% | 0.503 | 93.2% | — |
| K1 GPT-5.4 all-traits | 83.8% | 0.502 | 93.2% | 0.0% (p=1.00) |
| K2 GPT-5.1 per-trait | 81.1% | 0.395 | 94.5% | −2.7% (p=0.25) |
| K3 GPT-5.1 per-section | 80.2% | 0.358 | 94.6% | −3.6% (p=0.13) |

Reading: on leaf margin, GPT-5.4 all-traits is statistically identical to the local
Qwen baseline; GPT-5.1's per-trait/per-section granularity trends slightly worse
(not significant). All sit well below the human ceiling. The local prompt/model
levers (C1–C4) are pending GPU runs.

## Notes

- **Compound-value canonicalization.** Cloud models emit verbose margin values
  (e.g. `"toothed, serrate"`). The grader now resolves such a value to its single
  agreed canonical (negation-guarded) instead of dropping it to `MISSING`
  (`value_map.to_canonical`). This is symmetric across all conditions — it can only
  rescue previously-`MISSING` values.
- **Missing specimens.** `Euphorbiaceae_Maprounea_guianensis_ANMAPRGU4` has an empty
  answer in Kaili's GPT-5.4 file, so K1 has 113 specimens (the adapter skips it).
- **Adding a condition.** Add it to `configs/experiments/leaf_margin_ladder.yaml`
  and pass its `model_run` dir to `compare_trait_experiments.py`. Widening beyond
  leaf margin (stitching the full per-trait/per-section file sets) is future work.
