# VLM Model Alternatives for Botanical Trait Extraction

> **Purpose**: Evaluate open-source vision-language models to replace Qwen3-VL-32B (FP8) for
> extracting 24 morphological traits from tropical seedling images.
>
> **Hardware constraint**: Single NVIDIA H200 140 GB VRAM (SM 9.0, RHEL 8, glibc 2.28, CUDA 12.9).
>
> **Last verified**: 2026-03-19

---

## 1. Executive Summary

The VLM landscape has shifted dramatically since Qwen3-VL launched. Two developments
stand out: (a) the **Qwen3.5 family** (Feb 2026) unifies vision and language into a
single model with early-fusion multimodal training, eliminating the separate "-VL"
variant pattern; and (b) **InternVL3.5** (Aug 2025) introduced cascade reinforcement
learning and a Visual Resolution Router that delivers strong fine-grained perception
with efficient token usage.

### Top 4 Recommendations (ranked)

1. **Qwen3.5-35B-A3B (FP8)** -- New architecture with early-fusion vision, MoE
   (35B total / 3B active), fits easily on single H200. MMMU 81.4, OCRBench 91.0,
   HallusionBench 67.9. Only 3B active params means fast inference. Official FP8
   version available (1.76M downloads). Apache 2.0. vLLM support exists but has
   active bug reports -- test thoroughly before production.

2. **Qwen3.5-27B (BF16 or community FP8)** -- Dense hybrid architecture (27B),
   strongest per-model benchmarks in the Qwen3.5 family at sizes that fit single GPU.
   MMMU 82.3, MathVista 87.8, MMBench 92.6. ~56 GB BF16 weights fit H200 with headroom
   for KV cache at reduced context. No official FP8 yet. More stable than MoE variant
   for structured output.

3. **Qwen3-VL-32B-Thinking (FP8)** -- Zero-effort swap from current model. Same
   architecture, adds chain-of-thought reasoning. MMMU 78.1, MathVista 85.9.
   ~32 GB FP8. The `strip_thinking()` logic already exists in `vlm_client.py`.
   Try this first before any architecture change.

4. **InternVL3.5-38B** -- Independent architecture with a 5.5B-param vision encoder
   (largest among candidates). MMMU 76.9, strong multi-image support. ~77 GB BF16,
   ~39 GB FP8. Moderate integration effort. Good fallback if Qwen models underperform
   on fine-grained botanical features.

### Key Finding

The pipeline's OpenAI-compatible API client (`vlm_client.py`) means any vLLM-servable
model is near-drop-in. Changes needed: (a) `model` field in `configs/pipeline.yaml`,
(b) possible chat template adjustments (handled by vLLM), (c) prompt tuning.

---

## 2. Comprehensive Model Comparison Table

### Tier 1: Recommended Candidates

| Model | Total Params | Active Params | Arch | VRAM (BF16) | VRAM (FP8) | MMMU | MMMU-Pro | OCRBench | MathVista | MMBench | HallusBench | Multi-Img | vLLM | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-35B-A3B** | 35B | 3B | MoE | ~70 GB | ~36 GB | 81.4 | 75.1 | 91.0 | 86.2 | 91.5 | 67.9 | Yes | Yes* | Apache-2.0 |
| **Qwen3.5-27B** | 27B | 27B | Hybrid | ~56 GB | ~28 GB | 82.3 | 75.0 | -- | 87.8 | 92.6 | -- | Yes | Yes* | Apache-2.0 |
| **Qwen3-VL-32B-Think** | 32B | 32B | Dense | ~64 GB | ~32 GB | 78.1 | -- | 855 | 85.9 | 90.8 | 67.4 | Yes | Native | Apache-2.0 |
| **InternVL3.5-38B** | 38.4B | 38.4B | Dense | ~77 GB | ~39 GB | 76.9 | -- | 870 | 81.9 | 87.3 | 59.7 | Yes | Native | MIT |

### Tier 2: Worth Testing

| Model | Total Params | Active Params | Arch | VRAM (BF16) | VRAM (FP8) | MMMU | OCRBench | MathVista | MMBench | Multi-Img | vLLM | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** | 9B | 9B | Hybrid | ~20 GB | ~10 GB | 78.4 | 89.2 | 85.7 | 90.1 | Yes | Yes* | Apache-2.0 |
| **Qwen3.5-122B-A10B** | 122B | 10B | MoE | ~244 GB | ~122 GB | 83.9 | 92.1 | -- | 92.8 | Yes | Yes* | Apache-2.0 |
| **MiniCPM-V-4.5** | 8.7B | 8.7B | Dense | ~18 GB | ~9 GB | -- | Leading | -- | -- | Yes | Native | Apache-2.0 |
| **Kimi-K2.5** | 1T | 32B | MoE | ~2 TB | ~1 TB | -- | 92.3 | 90.1 | -- | Yes | Yes | MIT-mod |
| **GLM-4.1V-9B-Think** | 10B | 10B | Dense | ~20 GB | ~10 GB | -- | -- | -- | -- | Yes | Partial | MIT |
| **Mistral-Small-3.1-24B** | 24B | 24B | Dense | ~48 GB | ~24 GB | 64.0 | -- | 68.9 | -- | Yes | Native | Apache-2.0 |

### Tier 3: Not Recommended (see Section 6)

| Model | Total Params | Active Params | MMMU | Why Not |
|---|---|---|---|---|
| Qwen2.5-VL-72B | 72B | 72B | 70.2 | Older gen; needs INT4 to fit; MMMU lower than 32B-class |
| GLM-4.5V | 106B | 12B | -- | 106 GB FP8 leaves no KV cache room; limited community |
| Pixtral Large | 124B | 124B | -- | Deprecated by Mistral; does not fit |
| PaliGemma 2-28B | 28B | 28B | -- | No multi-image; no chat mode; research-only |
| Gemma 3-27B | 27B | 27B | 56.1 | Weak vision benchmarks; vLLM issues; no multi-image clarity |
| NVLM-D-72B | 72B | 72B | 58.7 | Non-commercial license; older (Sep 2024); needs quantization |
| Ovis2-34B | 35B | 35B | 66.7 | Low MMMU; limited adoption (207 downloads) |
| Llama 4 Scout | 109B | 17B | 73.4 | MoE too large for single GPU; knowledge cutoff Aug 2024 |
| CogVLM2-19B | 19B | 19B | 44.3 | Outdated (Aug 2024); low MMMU; 8K context |
| DeepSeek-VL2 | 28B | 4.5B | -- | 4.5B active too small; limited vLLM support |
| Molmo 2-8B | 8B | 8B | -- | Too small for 24-trait extraction |
| SmolVLM-2B | 2B | 2B | -- | Far too small |
| Reka Edge-7B | 7B | 7B | -- | Edge-optimized; too small; custom license |
| Kimi-VL-A3B | 16B | 2.8B | -- | Too small (2.8B active) |
| LLaVA-OneVision-72B | 72B | 72B | -- | Superseded; partial vLLM support |

**Notes:**
- VRAM estimates = weights only (2 bytes/param BF16, 1 byte/param FP8). Add 20-50% for KV cache + activations.
- `*` = vLLM support exists but has active bug reports as of March 2026.
- OCRBench scores: Qwen3.5 uses a different scale (0-100) vs older models (0-1000). Qwen3.5-35B's 91.0 ~ 910 on the old scale.
- Dashes (--) = score not found in available sources for that specific variant.

---

## 3. Detailed Per-Model Analysis

### 3.1 Qwen3.5-35B-A3B -- TOP RECOMMENDATION

**Architecture**: Hybrid Gated DeltaNet + sparse MoE. 40 layers, 256 total experts,
8 routed + 1 shared active per token. Early-fusion multimodal training integrates
vision directly into the base model (no separate "-VL" variant). 262K native context,
extensible to 1M tokens.

**Why it matters for botanical traits**: The early-fusion approach means visual tokens
participate in all training phases, not bolted on after language pre-training. OCRBench
91.0 demonstrates exceptional fine-grained visual-text alignment. HallusionBench 67.9
matches current Qwen3-VL-32B, reducing risk of trait hallucination. The MoE architecture
with only 3B active params means fast inference for batch processing specimens.

**Key benchmarks** (source: Qwen3.5 model card, self-reported):
- MMMU: 81.4 (+5.4 over Qwen3-VL-32B's 76.0) -- 🏢 Official model card
- MMMU-Pro: 75.1 -- 🏢 Official model card
- MathVista: 86.2 (+2.4 over Qwen3-VL-32B) -- 🏢 Official model card
- OCRBench: 91.0 (new scale; ~910 on old 1000-point scale) -- 🏢 Official model card
- HallusionBench: 67.9 -- 🏢 Official model card
- MMBench v1.1: 91.5 -- 🏢 Official model card
- AI2D: 92.6 -- 🏢 Official model card
- RealWorldQA: 84.1 -- 🏢 Official model card
- RefCOCO avg: 89.2 (spatial grounding) -- 🏢 Official model card
- Medical VQA (SLAKE: 78.7, PMC-VQA: 62.0) -- 🏢 Official model card

**Community adoption**:
- Downloads: 2.13M (base) + 1.76M (FP8) = ~3.9M total -- 👥 HuggingFace metrics
- Official FP8 version: `Qwen/Qwen3.5-35B-A3B-FP8` -- 🏢 Official release
- 58 community discussions on HuggingFace -- 👥 HuggingFace
- Extensive GGUF quantizations by unsloth (1.84M downloads) -- 👥 Community

**VRAM on H200**:
- BF16: ~70 GB total model (35B params, but MoE stores all experts). Fits with ~70 GB for KV cache at reduced context.
- FP8: ~36 GB. Fits comfortably with ample KV cache headroom.
- Official recommendation: 8 GPUs with TP for full 262K context. Single H200 viable with reduced context (128K-131K).

**vLLM support**: Listed in Qwen3.5 model card. However, there are active bug reports:
- Issue #36890: VRAM allocation bug on ROCm (not relevant for CUDA H200) -- 👥 GitHub
- Issue #36275: Qwen3.5-4B incompatibility -- 👥 GitHub
- PR #37562: Fix for GatedDeltaNet at TP>=2 (open, not merged) -- 👥 GitHub
- Conclusion: vLLM support works but is maturing. Test before production.

**Integration effort**: Moderate.
- Different architecture from Qwen3-VL (hybrid DeltaNet vs standard transformer).
- vLLM serves it via OpenAI-compatible API, so `vlm_client.py` works unchanged.
- Thinking mode is default; disable with `chat_template_kwargs: {"enable_thinking": false}` for deterministic structured output.
- Update `pipeline.yaml` model name + adjust prompts.

**Source credibility**: Medium. All benchmarks self-reported by Alibaba/Qwen team.
Model is 3 weeks old (Feb 2026) -- insufficient time for independent validation.
However, Qwen3-VL series benchmarks were broadly confirmed by community testing,
suggesting these numbers are plausible.

---

### 3.2 Qwen3.5-27B -- DENSE ALTERNATIVE

**Architecture**: Hybrid dense model with Gated DeltaNet + Gated Attention (NOT MoE).
64 layers, 5120 hidden dim. Same early-fusion multimodal training as the MoE variants
but all 27B params are active every forward pass.

**Why it matters for botanical traits**: Dense models generally produce more consistent
structured output than MoE models (no expert routing variance). MMMU 82.3 and MMBench
92.6 are the highest in the Qwen3.5 family below the 397B flagship. For a task
requiring reliable JSON output across 24 traits, consistency may trump speed.

**Key benchmarks** (source: Qwen3.5 model card, self-reported):
- MMMU: 82.3 -- 🏢 Official model card
- MMMU-Pro: 75.0 -- 🏢 Official model card
- MathVista: 87.8 -- 🏢 Official model card
- MathVision: 86.0 -- 🏢 Official model card
- MMBench v1.1: 92.6 -- 🏢 Official model card
- RealWorldQA: 83.7 -- 🏢 Official model card
- OmniDocBench1.5: 88.9 -- 🏢 Official model card
- CharXiv: 79.5 -- 🏢 Official model card
- IFEval: 95.0 (instruction following) -- 🏢 Official model card
- SWE-bench Verified: 72.4 -- 🏢 Official model card

**Community adoption**:
- Downloads: 1.78M -- 👥 HuggingFace metrics
- No official FP8 version yet -- 🏢 Official
- 104+ community quantizations available -- 👥 HuggingFace

**VRAM on H200**:
- BF16: ~56 GB weights. With 84 GB remaining for KV cache, fits comfortably at moderate context lengths (32K-64K). At 128K+ context, may be tight.
- FP8 (community quantization): ~28 GB. Excellent headroom.
- Single H200 viable with `--max-model-len 65536` or `131072` depending on batch size.

**vLLM support**: Same status as Qwen3.5-35B-A3B -- supported but with active bugs. The dense architecture may be more stable than MoE under vLLM since it avoids expert routing kernel issues.

**Integration effort**: Same as 3.1. Moderate.

**Caveats**:
- 27B dense is slower than 3B-active MoE per token.
- No official FP8 quantization -- must use community quants or vLLM on-the-fly quantization.
- Official docs recommend 8 GPUs for full 262K context. Single GPU requires context reduction.

**Source credibility**: Medium. Same caveats as Qwen3.5-35B-A3B.

---

### 3.3 Qwen3-VL-32B-Thinking -- ZERO-EFFORT FIRST STEP

**Architecture**: Dense 32B transformer with Qwen3 backbone. Standard ViT vision
encoder with window attention and dynamic resolution. This is the current model
architecture with thinking enabled.

**Why it matters**: The Thinking variant wraps reasoning in `<think>` blocks before
producing output. For morphological classification (palmate vs pinnate venation,
serrate vs entire margins), chain-of-thought should reduce misclassification.
The pipeline already handles `<think>` block stripping via `strip_thinking()`.

**Key benchmarks** (source: Qwen3-VL technical report):
- MMMU: 78.1 (+2.1 over Instruct) -- 📋 arXiv:2511.21631
- MathVista: 85.9 (+2.1) -- 📋 arXiv:2511.21631
- DocVQA: 96.1 -- 📋 arXiv:2511.21631
- OCRBench: 855 -- 📋 arXiv:2511.21631
- HallusionBench: 67.4 -- 📋 arXiv:2511.21631

**Community adoption**:
- Qwen3-VL-32B-Instruct: well-established, widely deployed -- 👥 Community
- Thinking variant: official FP8 at `Qwen/Qwen3-VL-32B-Thinking-FP8` -- 🏢 Official

**VRAM on H200**: ~32 GB FP8. Fits easily with full KV cache headroom.

**Integration effort**: Zero. Change `model` field in `pipeline.yaml`.

**Source credibility**: Medium. Benchmarks from arXiv:2511.21631. No independent
botanical benchmarks exist.

---

### 3.4 InternVL3.5-38B -- INDEPENDENT ARCHITECTURE

**Architecture**: ViT-MLP-LLM paradigm. InternViT-6B-448px (5.5B param vision encoder)
+ Qwen3-based LLM (32.8B params). The 5.5B vision encoder is the largest among all
candidates -- most VLMs use 300M-600M vision encoders. Features:
- Visual Resolution Router (ViR): dynamically adjusts visual token compression
  (256 or 64 tokens per patch based on semantic richness)
- Cascade RL training: offline MPO then online GSPO
- +16.0% reasoning improvement over InternVL3

**Why it matters for botanical traits**: The 5.5B vision encoder provides dedicated
capacity for fine-grained visual features (leaf margin serration, trichome patterns,
venation architecture). The ViR allocates more tokens to complex image regions --
exactly what's needed for subtle morphological features (stipules, lenticels) in
small image areas.

**Key benchmarks** (source: InternVL3.5 technical report + model card):
- MMMU: 76.9 -- 📋 arXiv:2508.18265
- OCRBench: 870 -- 🏢 Official model card
- MathVista: 81.9 -- 📋 arXiv:2508.18265
- MMBench v1.1: 87.3 -- 🏢 Official model card
- AI2D: 87.8 -- 🏢 Official model card
- HallusionBench: 59.7 -- 🏢 Official model card
- Multi-image and video supported -- 🏢 Official model card

**Community adoption**:
- Downloads: 34,404/month (38B variant) -- 👥 HuggingFace
- InternVL2-2B: 1.37M downloads (ecosystem is established) -- 👥 HuggingFace
- OpenGVLab organization: 1.88K followers -- 👥 HuggingFace
- InternVL series independently validated on OpenCompass and MMMU leaderboards -- 🏆 Independent
- 7 quantized variants available (GGUF, BNB) -- 👥 Community
- FP8 available: `ConfidentialMind/InternVL3-38B-FP8-Dynamic` -- 👥 Community
- Published paper with reproducible methodology -- 📋 arXiv:2508.18265
- vLLM recipe documented: `docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html` -- 🏢 Official

**VRAM on H200**:
- BF16: ~77 GB (fits with ~63 GB for KV cache)
- FP8: ~39 GB (ample headroom)

**vLLM support**: Native. Architecture `InternVLChatModel` listed in vLLM v0.8.5
supported models. Documented deployment recipe exists.

**Integration effort**: Moderate.
- Different chat template (InternVL-specific, but vLLM handles this).
- OpenAI-compatible API through vLLM means `vlm_client.py` works unchanged.
- Thinking mode available via system prompt, not native `<think>` tags.
- May need to adjust `image_mode` -- InternVL's dynamic tiling differs.

**Caveats**:
- HallusionBench 59.7 is lower than Qwen models (~67), meaning more trait hallucination risk.
- 38B dense model is slower than Qwen3.5-35B-A3B MoE.
- No published results on botanical/biological tasks.
- Lower downloads than Qwen models suggest smaller community.

**Source credibility**: Medium-High. Technical report published (arXiv:2508.18265).
InternVL series independently validated on multiple leaderboards. Cascade RL
methodology is well-documented.

---

### 3.5 Qwen3.5-9B -- LIGHTWEIGHT OPTION

**Architecture**: Hybrid Gated DeltaNet + Gated Attention (dense, not MoE). 32 layers,
4096 hidden dim. Same early-fusion multimodal training. 262K native context.

**Why it matters**: At 9B params, this model achieves MMMU 78.4 -- matching or exceeding
many 30B+ models from 2024-2025. If it can handle structured 24-trait JSON output
reliably, it would be the fastest and most resource-efficient option. Useful for rapid
iteration and prompt engineering experiments.

**Key benchmarks** (source: Qwen3.5 model card, self-reported):
- MMMU: 78.4 -- 🏢 Official model card
- MMMU-Pro: 70.1 -- 🏢 Official model card
- MathVista: 85.7 -- 🏢 Official model card
- MathVision: 78.9 -- 🏢 Official model card
- OCRBench: 89.2 -- 🏢 Official model card
- MMBench v1.1: 90.1 -- 🏢 Official model card
- OmniDocBench1.5: 87.7 -- 🏢 Official model card
- RefCOCO avg: 89.7 -- 🏢 Official model card

**Community adoption**: 2.59M downloads -- 👥 HuggingFace

**VRAM on H200**: ~20 GB BF16. Trivially fits with maximum KV cache headroom.

**vLLM support**: Same status as other Qwen3.5 models.

**Integration effort**: Same as 3.1.

**Caveats**:
- 9B may struggle with complex multi-attribute structured JSON extraction.
- Should be tested on a sample of specimens before committing.
- No official FP8 version (not needed at 9B).

**Source credibility**: Medium. Self-reported.

---

### 3.6 Qwen3.5-122B-A10B -- PREMIUM OPTION

**Architecture**: MoE with 122B total, 10B active (256 experts, 8+1 activated). 48
layers. Same early-fusion multimodal training.

**Key benchmarks**: MMMU 83.9, MMMU-Pro 76.9, OCRBench 92.1, MMBench 92.8 -- 🏢 Official

**VRAM on H200**: ~244 GB BF16 (does NOT fit). ~122 GB FP8 (fits, but leaves only
~18 GB for KV cache -- impractical for multi-image botanical analysis).

**Community adoption**: 544K downloads -- 👥 HuggingFace

**Verdict**: Benchmarks are excellent but single-H200 deployment is impractical.
Consider only if multi-GPU becomes available.

---

### 3.7 Qwen3.5-397B-A17B -- FLAGSHIP REFERENCE

**Architecture**: MoE with 397B total, 17B active (512 experts, 10+1 activated). 60
layers. The largest and best-performing Qwen3.5 model.

**Key benchmarks**: MMMU 85.0, MMMU-Pro 79.0, MathVista 90.3, OCRBench 93.1,
HallusionBench 71.4, MMBench 93.7 -- 🏢 Official

**VRAM on H200**: ~794 GB BF16 / ~397 GB FP8. Does NOT fit single GPU.

**Community adoption**: 1.8M downloads -- 👥 HuggingFace

**Verdict**: Provides an upper bound on Qwen3.5 family performance. Not deployable
on single H200. Reference only.

---

### 3.8 MiniCPM-V-4.5 -- EFFICIENT ALTERNATIVE

**Architecture**: Dense 8.7B model. Qwen3-8B backbone + SigLIP2-400M vision encoder.
Unified 3D-Resampler achieves 96x video token compression.

**Why it matters**: Claims OpenCompass average of 77.0, surpassing GPT-4o and
Qwen2.5-VL-72B. Exceptionally efficient: 28 GB GPU memory for video tasks,
7.5h total OpenCompass inference time (fastest among peers).

**Key benchmarks** (source: model card + arXiv:2509.18154):
- OpenCompass Average: 77.0 -- 🏢 Official model card
- OCRBench: "Leading, surpasses GPT-4o-latest" -- 🏢 Official model card
- OmniDocBench: SOTA among general MLLMs -- 🏢 Official model card
- Video-MME inference: 0.26h (11.7x faster than GLM-4.1V) -- 🏢 Official model card

**Community adoption**: 91.4K downloads -- 👥 HuggingFace. 1.07K likes.

**VRAM on H200**: ~18 GB BF16. Trivially fits.

**vLLM support**: Native. PR #23586 merged, available since v0.10.2 (Aug 2025).
Architecture `MiniCPMV` in supported models list.

**Caveats**:
- 8.7B may be too small for reliable 24-trait structured extraction.
- Specific vision benchmark numbers (MMMU, MathVista) not individually reported.
- Qwen3-8B backbone is a generation behind Qwen3.5.

**Source credibility**: Medium. Paper at arXiv:2509.18154. Individual benchmark
numbers not transparently reported (only "leading" claims and OpenCompass average).

---

### 3.9 Kimi-K2.5 -- FRONTIER MoE

**Architecture**: 1T total params, 32B active. 384 experts, 8+1 selected per token.
MoonViT-400M vision encoder. MLA (Multi-head Latent Attention). 256K context.

**Key benchmarks** (source: model card, self-reported):
- MMMU-Pro: 78.5 -- 🏢 Official model card
- MathVista: 90.1 -- 🏢 Official model card
- OCRBench: 92.3 -- 🏢 Official model card
- OmniDocBench1.5: 88.8 -- 🏢 Official model card
- AIME 2025: 96.1 -- 🏢 Official model card
- MathVision: 84.2 -- 🏢 Official model card

**Community adoption**: 3.38M downloads -- 👥 HuggingFace. 2.3K likes.

**VRAM on H200**: 1T total params = ~2 TB BF16. Does NOT fit. Even FP8 (~1 TB)
requires multi-node. The model is designed for API deployment or large clusters.

**vLLM support**: Yes. KimiVLForConditionalGeneration in supported models.
Also supports SGLang, KTransformers.

**Verdict**: Excellent benchmarks but completely impractical for single H200.
Consider via API if testing is needed.

---

### 3.10 GLM-4.1V-9B-Thinking

**Architecture**: 10B model based on GLM-4-9B-0414. Chain-of-thought reasoning via RL.
Supports up to 4K resolution images, arbitrary aspect ratios. 64K context.

**Why it matters**: Claims to outperform Qwen2.5-VL-72B on 29 benchmarks despite being
9B. RLCS (Reinforcement Learning with Curriculum Sampling) training methodology.

**Key benchmarks**: Specific numbers not available in text form (provided as benchmark
image only). Claims "best on 23/28 tasks at 10B scale" -- 🏢 Official model card.

**Community adoption**:
- Downloads: 405K -- 👥 HuggingFace
- 34 Spaces, 12 fine-tuned variants -- 👥 HuggingFace

**VRAM on H200**: ~20 GB BF16. Trivially fits.

**vLLM support**: Partial. `GLM4VForCausalLM` listed in vLLM but for `THUDM/glm-4v-9b`
(the older non-thinking variant). GLM-4.1V-Thinking may require `trust-remote-code`
and is not in the standard vLLM architecture list.

**Caveats**:
- Benchmark numbers not transparently reported (image-only).
- 9B may be insufficient for complex structured extraction.
- Bilingual (CN/EN) focus may not be optimal for botanical English terminology.
- vLLM support uncertain for the 4.1V-Thinking variant specifically.

**Source credibility**: Low-Medium. Paper at arXiv:2507.01006 covers the training
methodology but transparent benchmark tables are not publicly available in text form.

---

### 3.11 Mistral-Small-3.1-24B

**Architecture**: Dense 24B transformer. Tekken tokenizer (131K vocabulary). 128K context.

**Key benchmarks** (source: model card):
- MMMU: 64.0 -- 🏢 Official model card
- DocVQA: 94.08 -- 🏢 Official model card
- AI2D: 93.72 -- 🏢 Official model card
- ChartQA: 86.24 -- 🏢 Official model card
- MathVista: 68.91 -- 🏢 Official model card
- Multi-image: up to 10 images per prompt -- 🏢 Official model card

**Community adoption**: 367K downloads -- 👥 HuggingFace

**VRAM on H200**: ~48 GB BF16, ~24 GB FP8. Fits easily.

**vLLM support**: Native. `Mistral3ForConditionalGeneration` in supported models.

**Caveats**:
- MMMU 64.0 is significantly below Qwen3.5 models (78-82).
- MathVista 68.91 is 17+ points below Qwen3.5-35B.
- Strong on document understanding but weak on general visual reasoning.

**Verdict**: Not recommended as primary. Too far behind on visual reasoning benchmarks.

---

### 3.12 Llama 4 Scout-17B-16E

**Architecture**: MoE with 109B total, 17B active. 16 experts. Early-fusion
native multimodality. 10M token context window.

**Key benchmarks** (source: model card):
- MMMU: 73.4 -- 🏢 Official model card
- MathVista: 70.7 -- 🏢 Official model card
- DocVQA: 94.4 -- 🏢 Official model card
- ChartQA: 88.8 -- 🏢 Official model card

**Community adoption**: 240K downloads, 1.25K likes -- 👥 HuggingFace

**VRAM on H200**: ~218 GB BF16 (does NOT fit). ~109 GB FP8 (fits but minimal KV cache).

**vLLM support**: Native. `Llama4ForConditionalGeneration` in supported models.

**Caveats**: Knowledge cutoff Aug 2024. FP8 on single H200 leaves ~31 GB for
KV cache -- marginal for multi-image input.

**Verdict**: Interesting architecture but VRAM constraints and mediocre vision
benchmarks (MMMU 73.4) make it uncompetitive vs Qwen3.5.

---

## 4. Community Adoption Evidence

### Download Rankings (HuggingFace, image-text-to-text, March 2026)

| Rank | Model | Monthly Downloads |
|------|-------|-------------------|
| 1 | Qwen3-VL-2B-Instruct | 12.0M |
| 2 | Qwen3-VL-8B-Instruct | 8.94M |
| 3 | moondream2 | 5.32M |
| 4 | Qwen2.5-VL-7B-Instruct | 5.07M |
| 5 | llava-1.5-7b-hf | 4.63M |
| 6 | Qwen2.5-VL-3B-Instruct | 3.92M |
| 7 | Kimi-K2.5 | 3.38M |
| 8 | DeepSeek-OCR | 3.23M |
| 9 | Qwen3-VL-30B-A3B-Instruct | 3.0M |
| 10 | Qwen3.5-9B | 2.59M |
| 11 | gemma-3-4b-it | 2.21M |
| 12 | Qwen3.5-35B-A3B | 2.13M |
| 13 | gemma-3-12b-it | 2.10M |

Source: 👥 HuggingFace download metrics, March 2026.

**Key takeaways**:
- Qwen dominates (~60% of top 30 downloads). The ecosystem is mature and well-supported.
- Qwen3.5 models already at 2M+ downloads despite being only 3 weeks old.
- Gemma 3 has strong downloads (2M+) but for smaller variants; the 27B vision model scores poorly.
- InternVL series is established (InternVL2-2B at 1.37M) but the 38B model has lower adoption (34K).
- GLM-4.1V-9B at 405K downloads shows moderate community interest.

### Arena / Leaderboard Rankings

The LM Arena (formerly LMSYS Chatbot Arena) has migrated to arena.ai. The vision
leaderboard data was not programmatically extractable at time of research. However:
- Qwen models consistently rank among top open-source VLMs on OpenCompass -- 🏆 Independent
- InternVL series validated on OpenCompass and MMMU leaderboards -- 🏆 Independent
- No independent validation yet exists for Qwen3.5 models (too new) -- ⚠️ Gap

### GitHub Stars (inference frameworks)

| Framework | Stars | Qwen3.5 Support |
|-----------|-------|-----------------|
| vLLM | ~50K+ | Yes (with bugs) |
| SGLang | ~10K+ | Yes (recommended by Qwen) |
| KTransformers | Growing | Yes |

---

## 5. Implementation Effort and Pipeline Compatibility

### Phase 1: Zero-Effort Quick Win (< 1 hour)

Switch to Qwen3-VL-32B-Thinking:

```yaml
# configs/pipeline.yaml
vlm:
  model: "Qwen/Qwen3-VL-32B-Thinking-FP8"
```

The `strip_thinking()` logic in `vlm_client.py` already handles `<think>` blocks.
Also consider reducing temperature to 0.1 for more deterministic structured output.

### Phase 2: Qwen3.5-35B-A3B (1-2 days)

```bash
# Deploy with vLLM
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --trust-remote-code \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3
```

Changes needed:
1. Update `configs/pipeline.yaml` model name.
2. Decide on thinking mode: for structured JSON extraction, disable thinking:
   ```python
   extra_body={"chat_template_kwargs": {"enable_thinking": False}}
   ```
   Or keep thinking enabled and strip `<think>` blocks as with Qwen3-VL-Thinking.
3. Re-tune prompts -- Qwen3.5 uses different system prompt conventions.
4. Test vLLM stability under multi-image load (active bugs exist).
5. Run benchmark comparison on held-out specimens.

### Phase 3: Qwen3.5-27B Dense Alternative (1-2 days)

```bash
# Deploy with vLLM (BF16, reduced context)
vllm serve Qwen/Qwen3.5-27B \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3
```

Same integration steps as Phase 2. Dense model may produce more consistent
structured output than MoE.

### Phase 4: InternVL3.5-38B Fallback (2-3 days)

```bash
# Deploy with vLLM (FP8 for efficiency)
vllm serve ConfidentialMind/InternVL3-38B-FP8-Dynamic \
    --trust-remote-code \
    --max-model-len 32768

# Or BF16 for maximum quality
vllm serve OpenGVLab/InternVL3_5-38B \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

Additional changes needed:
- InternVL uses different chat template (handled by vLLM).
- Thinking mode via system prompt, not native tags.
- Different image preprocessing (dynamic tiling).
- Test multi-image handling: InternVL uses `Image-1: <image>\nImage-2: <image>` format.

---

## 6. Models Considered But Not Recommended

### PaliGemma 2-28B

- **Why considered**: Google's dedicated vision model, 28B with Gemma 2 backbone + SigLIP-So400m encoder.
- **Why rejected**: No multi-image support (single image + text only). Not a chat model -- uses prompt templates like "cap en", "detect X". Research-only license. DocVQA 76.1 (vs Qwen3.5's 89+). 591 downloads/month. Not suitable for conversational structured extraction.
- Source: 🏢 HuggingFace model card.

### Gemma 3-27B-IT

- **Why considered**: Google's latest open model, 1.2M downloads, multimodal.
- **Why rejected**: MMMU 56.1 (vs Qwen3.5-35B's 81.4 -- a 25-point gap). DocVQA 85.6 (vs 89+). vLLM has known compatibility issues (excessive VRAM, server errors, FP8 crashes -- per HuggingFace discussions). Multi-image support unclear. Vision encoder architecture not disclosed. Community reports model repetition issues.
- Source: 🏢 Model card; 👥 HuggingFace discussions (issues #70, #75, #90, #53).

### NVLM-D-72B (NVIDIA)

- **Why considered**: NVIDIA's VLM with InternViT-6B encoder + Qwen2-72B backbone.
- **Why rejected**: **CC-BY-NC-4.0 license (non-commercial)**. MMMU 58.7 (dated Sep 2024). Needs 144 GB BF16 (does not fit). Superseded by newer models.
- Source: 🏢 HuggingFace model card.

### Ovis2-34B

- **Why considered**: AIMv2-1B vision encoder + Qwen2.5-32B backbone.
- **Why rejected**: MMMU 66.7 (low). Only 207 downloads. Limited community adoption. No vLLM architecture listing.
- Source: 🏢 HuggingFace model card.

### GLM-4.5V (106B total, 12B active)

- **Why considered**: MoE with 3D-RoPE for spatial reasoning. Claims SOTA on 41 benchmarks.
- **Why rejected**: 106 GB FP8 leaves ~34 GB for KV cache on H200 -- insufficient for 10 seedling images. All benchmark claims self-reported by Zhipu AI with limited independent validation. Less community adoption than Qwen/InternVL.
- Source: 📋 arXiv:2507.01006; ⚠️ Zhipu AI blog.

### Qwen2.5-VL-72B-Instruct

- **Why considered**: Previous-generation flagship, strong on documents.
- **Why rejected**: MMMU 70.2 is lower than Qwen3-VL-32B's 76.0 (older architecture). Needs quantization to fit (INT4/AWQ at ~37 GB or FP8 at ~72 GB). Superseded by Qwen3.5 family.
- Source: 📋 arXiv:2502.13923; 🏆 LMSYS leaderboard.

### Pixtral Large (124B)

- **Why considered**: Strong LMSYS Arena ranking.
- **Why rejected**: **Deprecated by Mistral AI**. 124 GB FP8 impractical. MathVista 69.4 (low).
- Source: ⚠️ Mistral blog (deprecated notice).

### CogVLM2-19B (THUDM)

- **Why considered**: Part of the GLM/THUDM family.
- **Why rejected**: Outdated (Aug 2024). MMMU 44.3. 8K context limit. 2,685 downloads. Superseded by GLM-4.1V series.
- Source: 📋 arXiv:2408.16500.

### DeepSeek-VL2 (28B total, 4.5B active)

- **Why considered**: Efficient MoE, strong OCRBench.
- **Why rejected**: 4.5B active params insufficient for 24-trait extraction. Limited vLLM support (`DeepseekVLV2ForCausalLM` listed but community reports issues). No VL3 released.
- Source: 📋 arXiv:2412.10302.

### Llama 4 Scout-17B-16E

- **Why considered**: Meta's latest multimodal MoE, 10M context.
- **Why rejected**: 109B total params = ~109 GB FP8, leaving minimal KV cache. MMMU 73.4, MathVista 70.7 (below Qwen3.5). Knowledge cutoff Aug 2024.
- Source: 🏢 Official model card.

### SmolVLM / SmolVLM-2 (HuggingFace)

- **Why rejected**: 256M-2B params. Far too small for complex trait extraction.
- Source: 🏢 HuggingFace model cards.

### Reka Edge (7B)

- **Why rejected**: 7B edge-optimized model. Custom license with revenue cap. Efficient but too small.
- Source: 🏢 HuggingFace model card.

### LLaVA-OneVision-72B

- **Why rejected**: Older architecture. Partial vLLM support (GitHub issue #14290). Needs 144 GB BF16. Superseded.
- Source: 👥 vLLM GitHub issues.

### Phi-4-Reasoning-Vision-15B (Microsoft)

- **Why considered**: SigLIP-2 vision encoder, MIT license, thinking mode.
- **Why rejected**: MMMU 54.3 (weak). Max 3,600 visual tokens limits multi-image. 16K context too short for pipeline. 22.8K downloads (limited adoption). No explicit multi-image support documented.
- Source: 🏢 HuggingFace model card.

---

## 7. Factors Beyond Model Choice

The current trait extraction quality may not be solely a model problem:

1. **Prompt engineering**: Current `prompt_style: "sys4"` may not be optimal.
   Consider few-shot examples with verified trait annotations.

2. **Temperature/sampling**: Current `temperature: 0.6`. For structured extraction,
   try `temperature: 0.1` or `0.0` with `top_p: 1.0`.

3. **Structured output enforcement**: vLLM supports guided generation (JSON schema
   enforcement). This constrains outputs to valid trait values and prevents malformed JSON.

4. **Image quality/resolution**: Low-resolution or poorly-lit images defeat any model.

5. **Trait taxonomy ambiguity**: Some traits (e.g., "coriaceous" vs "chartaceous"
   leaf texture) may be inherently ambiguous from photographs.

6. **Thinking mode trade-offs**: For Qwen3.5, thinking mode is default. For structured
   JSON extraction, disabling thinking may produce more consistent output at the cost
   of reasoning quality.

---

## 8. Source Credibility Summary

| Model | Primary Source | Independent Validation | Confidence |
|---|---|---|---|
| Qwen3.5-35B-A3B | Model card (Feb 2026) | None yet (too new) | Medium |
| Qwen3.5-27B | Model card (Feb 2026) | None yet (too new) | Medium |
| Qwen3-VL-32B-Think | arXiv:2511.21631 | Limited community | Medium |
| InternVL3.5-38B | arXiv:2508.18265 | OpenCompass, MMMU leaderboard | Medium-High |
| Qwen3.5-9B | Model card (Feb 2026) | None yet (too new) | Medium |
| MiniCPM-V-4.5 | arXiv:2509.18154 | OpenCompass average | Medium |
| Kimi-K2.5 | Model card (2026) | None yet | Medium |
| GLM-4.1V-9B | arXiv:2507.01006 | Limited | Low-Medium |
| Gemma 3-27B | Model card (2025) | Community reports issues | Low-Medium |
| Mistral-Small-3.1 | Model card (2025) | Limited | Medium |

**Critical caveat**: No model has been independently benchmarked on botanical
morphological trait extraction from seedling photographs. All scores reflect
general-purpose visual understanding. Domain-specific performance may differ.

---

## 9. References

### Qwen3.5 Family
- [Qwen3.5-35B-A3B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) -- 🏢 Official model card
- [Qwen3.5-35B-A3B-FP8 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) -- 🏢 Official FP8 quantization
- [Qwen3.5-27B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B) -- 🏢 Official model card
- [Qwen3.5-9B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B) -- 🏢 Official model card
- [Qwen3.5-122B-A10B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) -- 🏢 Official model card
- [Qwen3.5-397B-A17B HuggingFace](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) -- 🏢 Official model card

### Qwen3-VL Family
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) -- 📋 ArXiv preprint
- [Qwen3-VL-32B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8) -- 🏢 Official
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) -- 📋 ArXiv preprint

### InternVL3.5
- [InternVL3.5 Paper](https://arxiv.org/abs/2508.18265) -- 📋 ArXiv preprint
- [InternVL3.5-38B HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-38B) -- 🏢 Official model card
- [InternVL3-38B-FP8-Dynamic](https://huggingface.co/ConfidentialMind/InternVL3-38B-FP8-Dynamic) -- 👥 Community FP8
- [vLLM InternVL3.5 Recipe](https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html) -- 🏢 vLLM docs

### GLM / THUDM Family
- [GLM-4.1V-9B-Thinking HuggingFace](https://huggingface.co/THUDM/glm-4.1v-9b-thinking) -- 🏢 Official model card
- [GLM-4.5V/4.6V Paper](https://arxiv.org/abs/2507.01006) -- 📋 ArXiv preprint
- [GLM-4.5V HuggingFace](https://huggingface.co/zai-org/GLM-4.5V) -- 🏢 Official model card
- [CogVLM2 Paper](https://arxiv.org/abs/2408.16500) -- 📋 ArXiv preprint

### Google / Gemma
- [Gemma 3-27B-IT HuggingFace](https://huggingface.co/google/gemma-3-27b-it) -- 🏢 Official model card
- [PaliGemma 2-28B HuggingFace](https://huggingface.co/google/paligemma2-28b-mix-448) -- 🏢 Official model card

### Other Models
- [MiniCPM-V-4.5 HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4_5) -- 🏢 Official model card
- [Kimi-K2.5 HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5) -- 🏢 Official model card
- [Mistral-Small-3.1-24B HuggingFace](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) -- 🏢 Official
- [Llama 4 Scout HuggingFace](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) -- 🏢 Official
- [NVLM-D-72B HuggingFace](https://huggingface.co/nvidia/NVLM-D-72B) -- 🏢 Official
- [Ovis2-34B HuggingFace](https://huggingface.co/AIDC-AI/Ovis2-34B) -- 🏢 Official
- [Phi-4-Reasoning-Vision-15B HuggingFace](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B) -- 🏢 Official
- [Reka Edge HuggingFace](https://huggingface.co/RekaAI/reka-edge-2603) -- 🏢 Official
- [Molmo 2 Blog](https://allenai.org/blog/molmo2) -- 🏢 Official
- [DeepSeek-VL2 Paper](https://arxiv.org/abs/2412.10302) -- 📋 ArXiv preprint

### Deployment & Infrastructure
- [vLLM Supported Models (v0.8.5)](https://docs.vllm.ai/en/v0.8.5/models/supported_models.html) -- 🏢 vLLM docs
- [vLLM FP8 Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/fp8/) -- 🏢 vLLM docs
- [vLLM Qwen3.5 Issues](https://github.com/vllm-project/vllm/issues?q=Qwen3.5) -- 👥 GitHub

### Community & Leaderboards
- [HuggingFace Trending VLMs](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending) -- 👥 Community
- [HuggingFace Most Downloaded VLMs](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=downloads) -- 👥 Community
- [OpenCompass VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) -- 🏆 Independent
- [Best Open-Source Multimodal Models (SiliconFlow)](https://www.siliconflow.com/articles/en/best-open-source-multimodal-models-2025) -- ⚠️ Blog

### Botanical VLM Research
- [LLMs for morphological data extraction from taxonomic descriptions](https://pmc.ncbi.nlm.nih.gov/articles/PMC12381580/) -- 📄 Peer-reviewed
- [VLMs for plant simulation configurations](https://arxiv.org/html/2603.08930) -- 📋 ArXiv
- [LeafNet: Plant disease VLM benchmark](https://arxiv.org/html/2602.13662) -- 📋 ArXiv
- [AgroBench: VLM benchmark in agriculture](https://arxiv.org/pdf/2507.20519) -- 📋 ArXiv
- [Automatic morphological trait extraction from the web](https://pmc.ncbi.nlm.nih.gov/articles/PMC12188617/) -- 📄 Peer-reviewed

### Source Credibility Key
- 📄 Peer-reviewed paper
- 📋 ArXiv preprint (not yet peer-reviewed)
- 🏆 Independent benchmark/leaderboard
- 👥 Community testing (HuggingFace, Reddit, GitHub)
- 🏢 Official model card / vendor claims
- ⚠️ Marketing / blog post
