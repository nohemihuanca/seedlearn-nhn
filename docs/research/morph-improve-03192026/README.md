# Morphological Trait Extraction Improvement Research

> **Date**: 2026-03-19
> **Problem**: Stage 1 (Vision-LLM Morphology) produces inaccurate trait classifications —
> misidentifying leaf margins, shapes, and other fine-grained characteristics.
> **Scope**: Three parallel research tracks evaluated for improving extraction quality.

---

## Research Reports

| Report | Focus |
|--------|-------|
| [vlm_model_alternatives.md](vlm_model_alternatives.md) | Open-source VLM replacements for Qwen3-VL-32B |
| [prompt_engineering_vlm.md](prompt_engineering_vlm.md) | Prompt design, few-shot images, constrained decoding |
| [segmentation_activation_maps.md](segmentation_activation_maps.md) | Seedling isolation via segmentation and attention maps |

---

## Synthesis: Prioritized Improvement Opportunities

### Tier 1: Do First (high impact, low effort)

| # | Intervention | Track | Expected Impact | Effort | Evidence Quality |
|---|---|---|---|---|---|
| 1 | **Add botanical definitions + decision trees to prompts** | Prompt | +10-20% on ambiguous traits | 1-2 days | Moderate — peer-reviewed (Thielen 2024) |
| 2 | **Enable vLLM `guided_json` schema-constrained decoding** | Prompt | Eliminates parse errors, +5-10% on enum traits | Half day | Strong — vLLM production docs |
| 3 | **Swap to Qwen3-VL-32B-Thinking** | Model | +2-5% from chain-of-thought reasoning | 1 hour | Medium — arXiv, `strip_thinking()` already exists |
| 4 | **Lower temperature to 0.1** | Prompt | Reduces hallucinated trait values | 5 minutes | Well-established |

### Tier 2: Quick Experiments (medium impact, 1-3 days)

| # | Intervention | Track | Expected Impact | Effort | Evidence Quality |
|---|---|---|---|---|---|
| 5 | **BiRefNet background removal via `rembg`** | Segmentation | +5-15% by removing background noise | 1-2 days | High (Cloudflare benchmarks, CAAI 2024) |
| 6 | **GradCAM on BioCLIP 2 ViT** (diagnostic) | Segmentation | Diagnostic — identifies where model looks | 1 day | High (well-established, model already loaded) |
| 7 | **Qwen3.5-35B-A3B FP8** | Model | Potentially significant — MMMU +5.4, early-fusion vision | 1-2 days | Medium — self-reported, 1.76M downloads |

### Tier 3: Deeper Investments (high potential, more work)

| # | Intervention | Track | Expected Impact | Effort | Evidence Quality |
|---|---|---|---|---|---|
| 8 | **Grounded SAM 2** (text-prompted segmentation) | Segmentation | +10-15% with precise seedling isolation | 2-3 days | High (ECCV 2024, agricultural validation) |
| 9 | **Qwen3.5-27B dense** | Model | Most consistent structured output at this scale | 1-2 days | Medium — self-reported |
| 10 | **InternVL3.5-38B** | Model | 5.5B vision encoder may see finer morphological detail | 2-3 days | Medium-High — arXiv + independent leaderboard |
| 11 | **Targeted trait decomposition** (3-5 worst traits only) | Prompt | +5-10% on targeted traits | 2-3 days | Strong (Thielen 2024 direct comparison) |

### Tier 4: Experimental / Uncertain

| # | Intervention | Track | Expected Impact | Effort | Notes |
|---|---|---|---|---|---|
| 12 | Reference image grids (composite montage) | Prompt | Uncertain (0-15%) | 2-3 days | Santos 2025 says VLMs ignore visual demos |
| 13 | SAM 3 | Segmentation | High quality but 96GB VRAM, sequential loading | 3-5 days | ICLR 2026 submission, least field-tested |

---

## Key Findings

### 1. Textual prompt improvements outweigh model swaps

Adding botanical definitions and decision trees addresses the root cause: the model doesn't
know what "serrate" vs. "dentate" looks like from its pretraining alone. Explicit definitions
with discriminative criteria (tooth direction, shape, angle) give the model the domain
knowledge it lacks. This is the single highest-impact, lowest-risk intervention.

### 2. Segmentation and prompt engineering compound

Background removal eliminates noise (rulers, labels, other vegetation) so the VLM receives
a cleaner image. Better prompt definitions then guide the model to discriminate correctly
among valid trait options. These are complementary — neither alone is sufficient for
maximum improvement.

### 3. Few-shot image demonstrations are likely ineffective

Santos et al. (2025) showed VLMs primarily attend to textual cues in demonstrations and
largely ignore visual content — blacking out demo images caused no performance drop
(tested on 4-9B models). Composite image grids are an untested alternative worth
experimenting with, but textual definitions should be prioritized.

### 4. No VLM has been benchmarked on botanical trait extraction

All model comparisons use general benchmarks (MMMU, DocVQA, MathVista). Qwen3.5 models
dominate these but are 3 weeks old with zero independent validation. The safest path:
fix prompts first (certain improvement), then A/B test models on held-out specimens.

### 5. Qwen3.5 early-fusion architecture is architecturally significant

Unlike Qwen3-VL (vision bolted onto a language model), Qwen3.5 integrates vision into
base training. This means visual tokens participate in all training phases — a fundamental
advantage for fine-grained visual understanding. However, vLLM support has active bug
reports and needs testing before production use.

### 6. The current temperature (0.6) is too high for structured extraction

For deterministic trait classification requiring JSON output, temperature should be 0.1 or
lower. This is a 5-minute config change that likely reduces hallucinated trait values.

---

## Recommended Execution Order

**Week 1**: Items 1-4 — prompt improvements and config changes (zero model risk)
**Week 2**: Items 5-6 — segmentation baseline and attention diagnostics
**Week 3**: Item 7 — model swap experiment with A/B comparison on held-out specimens
**Ongoing**: Items 8-11 as time and results warrant

---

## Source Credibility Notes

Each research report marks claims with source type indicators:
- 📄 Peer-reviewed paper
- 📋 ArXiv preprint (not yet peer-reviewed)
- 🏆 Independent benchmark/leaderboard
- 👥 Community testing (HuggingFace, Reddit, GitHub)
- 🏢 Official model card / vendor claims
- ⚠️ Marketing / blog post

**Critical caveat across all tracks**: No technique or model in these reports has been
independently validated on tropical seedling field photographs. All expected impact
estimates are extrapolated from adjacent domains. Empirical validation on actual pipeline
data is required before committing to any approach.
