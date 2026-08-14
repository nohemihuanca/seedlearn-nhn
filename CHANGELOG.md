# Changelog

All notable changes to the SeedLearn project will be documented in this file.

## [Unreleased]

## [2026-07-17]

### Removed
- Development-workflow vestiges from the tracked tree: `docs/plans/` (19 design/plan docs), `docs/worktrees/` (5), `docs/superpowers/` (2), and `docs/restructure_plan.md`
- Untracked `CLAUDE.md` (`git rm --cached`; retained on disk, now gitignored per project rule — its production content already lives in `README.md`/`docs/`)

### Changed
- `.gitignore`: ignore tool caches (`.pytest_cache/`, `.ruff_cache/`) and reindex artifacts (`/index.md`, `/wiki/`); re-added the NEVER-COMMIT `CLAUDE.md` block

### Fixed
- `scripts/generate_report.py`: corrected the stale SimpleShot report output base (`REPO_BASE`) from the removed old-`main` `models/lightweight/simpleshot/outputs` path to `data/experiments/simpleshot`

## [2026-05-29]

### Changed
- Flattened `docs/paper/` to keep only research artifacts: `figures/` (8 figures + 3 specimen photos), `tables/` (8 JSON/CSV result files), and `results_narrative.md`. The submitted paper itself (Word/PDF) lives outside the repo.

### Removed
- Working drafts and scaffolding: `docs/paper/paper1/draft_01.md`, `voice_directive.md`, `paper1/results_narrative.md`, `paper1/figures/`, `paper1/tables/`
- Archived full-pipeline skeleton (`docs/paper/archive/skeleton_full_pipeline.md`) — deferred companion paper, scope now lives in repo history only
- Early section drafts: `docs/paper/skeleton.md`, `docs/paper/section_5_1.md`, `docs/paper/examples.md`

## [2026-05-10]

### Added
- Ablation experiment infrastructure (`experiments/ablation/`) — batch pipeline runner with sharding, SLURM array jobs for parallel H200 execution, baseline runner, and analysis scripts
- Paper 1 ablation results: 4-condition study (A: full pipeline, B: no RAG, C: visual only, D: baseline). Finding: RAG degrades accuracy when VLM trait extraction is imperfect — error cascade from wrong traits → wrong query → wrong retrieval

### Fixed
- Baseline runner sys.path resolution for cross-module import of `enumerate_test_individuals`

## [2026-04-03]

### Added
- Cross-model comparison report generator (`tests/benchmarks/generate_comparison_report.py`) — produces standalone HTML with side-by-side accuracy tables, per-trait confusion matrices for all models, per-specimen heatmap, and auto-generated key findings
- Gemma 4 31B-IT added to benchmark sweep (`sweep_vlm_models.sh`) with `--kv-cache-dtype fp8` and temperature 0.1

### Changed
- Upgraded vLLM from v0.18.1 to v0.19.1 — adds Gemma 4 architecture support
- Updated sweep model registry version comment to v0.19.1

## [2026-03-20]

### Added
- Concurrent single-image inference with `--workers` flag and GPU utilization tracking
- Per-model vLLM parameters in sweep script (temperature, extra flags, context limits)
- Hover tooltips on every metric in HTML scoring report

### Fixed
- Benchmark trait key mapping: `FORM_KEY_TO_TRAIT` now handles both short and full form keys (VLM reproduces parenthetical hints)
- Latex scoring: "not observed" treated as synonym for "absent" (via `SUBTYPE_MAP`), not as abstention — matches the prompt's `(present / not observed)` instruction
- `--prompt-file` flag added to `run_vlm_stage1.py` for custom prompt testing via `load_prompt()`
- vLLM startup timeout increased to 600s for Qwen3.5 flashinfer JIT kernel compilation
- Gemma 3 context limit fixed (8192→32768) to prevent max_tokens overflow

## [2026-03-19]

### Added
- Benchmark scoring engine (`tests/benchmarks/score_vlm_stage1.py`) for evaluating VLM morphological trait extraction against ground truth
- Scoring package (`tests/benchmarks/scoring/`) with matcher, loader, metrics, and HTML report modules
- Ground truth files for 21 benchmark specimens with STRI-derived trait values and source traceability
- `--mode single|multi|both` flag for `run_vlm_stage1.py` to compare per-image vs multi-image inference
- `SYS4_SINGLE` prompt variant for single-image morphological extraction
- HTML scoring report with confusion matrix heatmaps, per-trait accuracy, and multi-label transparency
- Synthetic test data (`stage1_test_results.json`) for scorer integration testing
- User-configurable prompt system: pipeline prompts can now be customized via external `.txt` files without editing source code
- `PromptsConfig` dataclass with `morphology`, `rag_query`, and `reasoning` fields for prompt file paths
- `load_prompt(path, fallback)` utility function with graceful fallback on missing/empty files
- Default prompt files in `configs/prompts/`: `stage1_morphology.txt`, `stage3_rag_query.txt`, `stage5_reasoning.txt`
- `prompts:` section in `configs/pipeline.yaml` for configuring prompt file paths
- Prompt customization documentation in `docs/pipeline.md` (PromptsConfig table, usage guide)
- 16 new tests in `tests/unit/test_prompt_loading.py` covering prompt loading, config integration, and stage wiring

### Changed
- Stage 1 (`morphology.py`), Stage 3 (`trait_retrieval.py`), and Stage 5 (`reasoning.py`) now accept optional `prompt_file` parameter — file path takes priority over hardcoded defaults
- `_compose_query()` in trait retrieval now accepts `template` and `fallback` arguments for configurable query construction

### Removed
- `TraitRetrievalConfig.query_style` field (was dead code, never used by Stage 3)
- `ReasoningConfig.prompt_template` field (was dead code, never wired to Stage 5; replaced by `PromptsConfig.reasoning`)

### Changed
- Renamed classification output key `"confidence"` → `"softmax_score"` and `"top1_confidence"` → `"top1_softmax_score"` for accuracy — the value is `softmax(−L2 distance)`, not a calibrated probability
- Renamed convergence key `"visual_confidence"` → `"visual_softmax_score"` in trait retrieval stage
- Updated all display labels in pipeline HTML report from "Confidence" to "Similarity Share" — plain-language term for the proportion of total closeness each candidate receives
- Updated evidence document text to use "similarity share" instead of "confidence" for Stage 2 metrics
- Added Glossary section to pipeline HTML report explaining all key metrics (similarity share, L2 distance, cosine similarity, decision margin, hierarchical consistency, RAG similarity, convergence signal, Stage 5 confidence)

## [2026-03-06]

### Added
- `docs/plans/user-config-prompts.md` — planning doc for user-configurable prompts via YAML config (non-technical audience, external `.txt` prompt files, deep-merge loading pattern)
- Dataset splits summary table in README with counts, distribution, k-shot feasibility
- Detailed per-class split statistics in `docs/data.md`
- `--random {test,val}` flag in `run_pipeline.py` for sampling random individuals from non-training splits
- `--random-seed N` flag for reproducible random selection
- `sample_individual_from_split()` function in `seedlearn.data.splits`
- `docs/slides/pipeline_full.svg` — two-row linear pipeline architecture diagram for presentations
