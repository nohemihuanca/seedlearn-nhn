# Human-Annotation Grading

Grade the pipeline's **visual component** — Stage 1 Vision-LLM morphological trait
extraction — against independent human annotators, and grade a botanist's
photo-based species identifications against the true taxonomy.

Code lives in `src/seedlearn/benchmarking/human/`; study data and results live in
the tracked `trait_grading/` directory.

> **Comparing extraction approaches?** See [`docs/trait-experiments.md`](trait-experiments.md)
> for the leaf-margin experiment ladder — comparing prompt/model/few-shot conditions
> (and external cloud runs) on the same human-grading backend.

## What it measures

Three trait-agreement axes over the gradable categorical traits, plus species ID:

| Axis | Question |
|------|----------|
| `model_vs_roni` | Does the Vision-LLM agree with Roni on each trait? |
| `model_vs_carmen` | Does the Vision-LLM agree with Carmen? |
| `roni_vs_carmen` | Do the two humans agree? (**inter-annotator ceiling**) |
| `* vs STRI` | Does each source (model, Roni, Carmen) match the STRI botanical trait matrix? |
| Roni species ID | How accurate are Roni's family/genus/species guesses from photos? |

`roni_vs_carmen` is the realistic ceiling: model-vs-human agreement should be read
against how well two trained humans agree, not against 100%. Each trait reports a
raw agreement rate **and** Cohen's κ (chance-corrected), so a skewed trait with a
high raw rate but low κ is visible.

The **STRI axis** scores each source against the STRI botanical trait matrix using
a match-any policy (the matrix is multi-label per species — a value is correct if it
is among the species' allowed STRI values). STRI codes only five traits
(leaf arrangement, leaf complexity, leaf margin, stipules, latex), so this axis
covers those. The headline metric is match-any **accuracy** over all comparable
specimens; because STRI is a multi-label reference, Cohen's κ is reported only over
the **single-label subset** (species STRI codes with exactly one value, where a
symmetric single-label comparison is well defined) and each STRI cell shows that
subset size as *κ n*. STRI is an independent botanical reference: a domain expert
(Roni) tracks it closely.

## Data layout (`trait_grading/`)

| Path | Tracked | Contents |
|------|---------|----------|
| `annotations/roni_bianco.xlsx` | yes | Roni's per-view trait annotations + species ID predictions |
| `annotations/carmen.xlsx` | yes | Carmen's per-view trait annotations (no IDs) |
| `keys/curator_taxonomic_key.csv` | yes | `anonymous_id → specimen_id + true taxonomy` |
| `keys/image_key.csv` | yes | `anonymous_id + view → original image` |
| `id_corrections.csv` | yes | Reviewable Roni-ID typo/variant/synonym corrections |
| `model_run/<timestamp>/` | yes | Fresh Stage-1 model trait JSONs for the 114 specimens |
| `reports/<timestamp>/` | yes | Generated CSV + HTML reports, one folder per run |

The keys are tracked and shared with collaborators — the de-anonymization mapping
is known information for everyone working on this repo, not something to hide.

## Methodology

- **Join.** Human annotations use blinded `anonymous_id` labels; the curator key
  maps each to a real `specimen_id` (e.g. `individual_001 → SRAPHEDE2`) and true
  family/genus/species.
- **Granularity.** Humans annotated traits **per view**; the model emits one pooled
  trait set **per specimen**. Each annotator's per-view values for a trait are
  collapsed to the **mode** (the value scored), with deterministic first-occurrence
  tie-breaking. The full per-view distribution is retained in
  `human_trait_distributions.csv` so noisy traits are visible behind their mode.
- **Vocabulary.** Model values are English (`whorled`), human values Spanish
  (`verticilada`); both collapse to shared canonical tokens via
  `human/value_map.py` before comparison. Blank / `no claro` / `not observed`
  values (either language) become `MISSING` and are excluded from scoring.
- **Gradable traits.** ~20 clean single-value categoricals are scored. Free-text /
  numeric traits (`num_leaflets`, `secondary_veins`, `leaf_surface`, `stem_color`)
  are reported descriptively, not scored. Unrecognized values are reported, never
  silently matched.
- **Species ID** reuses the same `compare_taxonomy()` core as the Stage-5 model-ID
  grader (case-insensitive, epithet-only fallback).

## Reproducible workflow

### 0. Install dependencies

`openpyxl` is required to read the annotation spreadsheets:

```bash
source .venv/bin/activate
uv sync          # installs openpyxl (now in pyproject.toml)
```

### 1. Run Stage 1 fresh on all 114 annotated specimens (GPU)

Start the vLLM server on a GPU node, then run the benchmark pipeline restricted to
the curator specimen set (this bypasses STRI overlap, so all 114 run — not just the
81 that match a STRI matrix):

```bash
# On a GPU node, with the vLLM server up (see docs/benchmarks.md):
python scripts/run_benchmark_pipeline.py \
    --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_*.csv \
    --cache-dir data/embeddings/2026-01-29_v2026-01-29_12K \
    --split-path data/splits/2026-01-29_v2026-01-29_12K/species/split_seed42 \
    --rag-index data/traits/latest/rag_index/ \
    --specimen-source trait_grading/keys/curator_taxonomic_key.csv \
    --output-dir trait_grading/model_run/$(date +%Y-%m-%d_%H%M%S)
```

`run_metadata.json` records the specimen count, source, timestamp, and model/prompt
version; any specimen missing from the catalog is listed in `missing_specimens`.

### 2. Grade and report

```bash
python scripts/grade_human_annotations.py \
    --results-dir trait_grading/model_run/<timestamp> \
    --html
```

Each run writes to its **own timestamped folder** `trait_grading/reports/<timestamp>/`
by default (pass `--out-dir` to override), so past runs are kept and can be compared.
Pass `--no-images` for a smaller/faster HTML, or `--no-stri` to skip the STRI axis.

Outputs in `trait_grading/reports/<timestamp>/`:

| File | Contents |
|------|----------|
| `trait_agreement_per_trait.csv` | rate + κ per trait per axis |
| `trait_agreement_overall.csv` | macro-averaged rate + κ per axis |
| `human_trait_distributions.csv` | per-specimen mode + every per-view value |
| `roni_id_accuracy.csv` / `roni_id_summary.json` | Roni's per-individual ID outcome + accuracy (raw **and** corrected columns) |
| `stri_accuracy.csv` | model/Roni/Carmen match-any accuracy vs STRI per trait, plus single-label-subset `n_kappa` + `cohen_kappa` |
| `human_grading_report.html` | per-trait table (inter-annotator ceiling + vs-STRI columns) + Roni ID section |

The STRI axis runs by default (`--stri-matrix` points at the cl185 matrix; pass
`--no-stri` to skip it).

## Report features

The HTML report (`human_grading_report.html`) surfaces:

- **Plain-language framing** — a top "What this report is" blurb states the report's purpose
  (how well the model reads traits from photos, judged against two botanists + STRI, plus a
  botanist's photo IDs), followed by a "How to read the tables" note explaining that the
  numbers are agreement rates and that Roni-vs-Carmen is the human ceiling.
- **Run provenance header** — when the report was generated, which model-run directory
  it used, and that run's model, prompt style, and specimen count (read from
  `run_metadata.json`). `run_benchmark_pipeline.py` records `prompt_style`, `model`, and
  `vlm_endpoint` so provenance is complete for new runs; older runs show "not recorded".
- **System prompt** — a collapsible section reproducing the exact system prompt sent to
  the Vision-LLM (resolved from the run's `prompt_style`; falls back to the `sys4`
  default, flagged, for runs predating provenance recording).
- **Overall-table explainer** — the macro-averaged summary is followed by a plain-language
  note defining *macro rate*, *macro k*, and *pairs compared* (macro = each trait weighted
  equally; pairs compared = a raw count of specimen×trait comparisons behind the axis).
- **Per-trait "what the model was asked" column** — the per-trait agreement table shows, next
  to each trait, the verbatim prompt wording it was given (from `human/trait_prompts.py`,
  drift-guarded against `SYSTEM_PROMPT_4`) plus the fuller graded canonical option set
  (`TraitSpec.canonical_values`), which can exceed the prompt's option list.
- **One consolidated κ explainer** — a single block explains the rate-vs-κ distinction and
  the Landis & Koch color bands together as a swatch legend (red = worse-than-chance … green
  = almost-perfect). Agreement cells are colored by κ. The `vs STRI` cells stay colored by
  **accuracy** (match-any) but also print a subset κ for context (single-label species only,
  with its own *κ n*).
- **Clickable drill-downs** — every colored cell (agreement **and** vs-STRI) opens a
  per-specimen/per-species breakdown; each row names the specimen, its species, and (when
  images are embedded) large example thumbnails, so you can see what each source was looking
  at. An affordance (inset shadow) and a hint make the cells' clickability obvious.
- **Bigger, zoomable example images** — every annotated view is embedded once as a
  size-controlled base64 thumbnail (≈320 px, JPEG) in a shared island, shown large (~180 px)
  in the drill-down modals and clickable to open a full-size lightbox (reusing the same
  embedded image, no extra bytes). The report stays a single shareable file (~17 MB for all
  567 views, ~3× the thumbnail-less size). Thumbnails are **not** shown in the species-ID
  table. Use `--no-images` to omit them entirely.

## Roni species-ID corrections (raw + corrected)

Roni's photo-based IDs contain obvious spelling typos, Latin orthographic variants, and
accepted taxonomic synonyms that a strict string match counts wrong. Rather than editing
the raw annotation data, an **auditable, editable file** credits these so a *corrected*
accuracy can be reported next to the untouched *raw* accuracy.

- **File:** `trait_grading/id_corrections.csv`, columns
  `specimen_id, rank, roni_original, canonical, category, note` where `rank ∈
  {family, genus, species}` and `category ∈ {typo, variant, synonym}`.
- **Crediting rule:** a rank is corrected-correct when raw-correct, **or** an entry's
  `roni_original` matches what Roni actually wrote (normalized) *and* its `canonical`
  matches the truth. Nothing is credited that is not listed; the raw data is never
  changed; Roni's original text stays visible in the report (blue cell + `category →
  canonical` note).
- **Review workflow:** to verify or change a correction, edit the CSV — add a row to
  credit a case, delete a row to withdraw it, or fix a `canonical`. Entries whose
  `roni_original` no longer matches the current annotation are treated as **stale** and
  credit nothing (surfaced by `stale_corrections`), so drift can't silently inflate the
  score. Re-run step 2 to regenerate both scores.
- **Effect on the seeded set:** crediting the verified typos, variants, and synonyms
  lifts the headline from raw family 92.1 / genus 88.6 / species 85.1 to corrected
  family 98.2 / genus 96.5 / species 93.9 (n=114). The species lift includes four
  epithet-level synonyms (Tontelea *ovalifolia*→*passiflora*, Coussarea
  *curvigemma*→*suaveolens*, Palicourea *capitata*→*violacea*, Piparea
  *commersoniana*→*dentata*).

### 3. Commit results for collaborators

The whole `trait_grading/` tree is tracked (annotations, keys, model runs, reports).
Commit the model run and reports so collaborators can review on the external repo.

## Validation

The full grading flow was validated end-to-end against the prior benchmark run
(`results/benchmarks/2026-03-04_181054/`, 81 overlapping specimens): inter-annotator
agreement (Roni vs Carmen) forms the expected ceiling above model-vs-human, and Roni's
ID accuracy decreases monotonically family → genus → species, as expected for
photo-based identification.

## Code map

| Module | Responsibility |
|--------|----------------|
| `human/value_map.py` | EN/ES → canonical tokens; trait gradability |
| `human/annotations.py` | Load xlsx, join blinded ids to specimens/taxonomy |
| `human/aggregate.py` | Per-view → per-specimen mode + distribution |
| `human/categorical_grader.py` | Per-trait agreement rate + Cohen's κ, three axes |
| `human/stri_compare.py` | Match-any accuracy + per-species detail vs the STRI matrix |
| `human/id_grading.py` | Roni species-ID grading (shared `compare_taxonomy` core), raw + corrected |
| `human/id_corrections.py` | Load + apply the reviewable ID corrections file |
| `human/thumbnails.py` | Downscale views to size-controlled base64 data URIs |
| `human/trait_prompts.py` | Curated `trait_key → prompt wording` map (drift-guarded vs `SYSTEM_PROMPT_4`) |
| `human/report.py` | CSV + HTML assembly (provenance, κ colors, drill-downs, images, prompt) |
| `scripts/grade_human_annotations.py` | Orchestration CLI |
