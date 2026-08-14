# VLM VRAM Verification: H200 140GB Single-GPU Feasibility

Last verified: 2026-03-19

Task profile: Multi-image botanical trait extraction (1-10 images per request,
~640 visual tokens per image, ~2K text tokens, structured JSON output with 24 traits).

---

## 1. Qwen3.5-122B-A10B

### Model Identity

| Field | Value |
|-------|-------|
| HuggingFace | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) |
| FP8 variant | [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) |
| Downloads | ~545K/month (BF16), ~451K/month (FP8) |
| License | Apache 2.0 |
| Type | Vision-language MoE (image + video + text) |

### Architecture

| Parameter | Value |
|-----------|-------|
| Total params | 122B (125B with embeddings) |
| Active params/token | 10B (8 routed + 1 shared expert) |
| Experts | 256 total, 9 active per token |
| Layers | 48 (hybrid: 3x GatedDeltaNet + 1x GatedAttention per block) |
| Hidden dim | 3,072 |
| Attention heads (Q/KV) | 32 / 2 (GQA) |
| Head dim | 256 |
| Native context | 262,144 tokens (extensible to 1,010,000 via YaRN) |
| Architecture | Gated DeltaNet + Sparse MoE (early-fusion multimodal) |

### Weight Sizes

| Format | Size | Source |
|--------|------|--------|
| BF16 | ~234 GB | HuggingFace repo metadata |
| FP8 (official) | **127 GB** | 39 safetensors shards (36x 3.22GB + 4.73GB + 3.65GB + 2.80GB) |
| NVFP4 | ~75.6 GB | [NVIDIA forums](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-nvfp4-quantized-for-dgx-spark-234gb-75gb-runs-on-128gb/361819) |

### VRAM Budget on H200 140GB

#### Scenario A: FP8 (official Qwen weights)

```
Model weights (FP8):         127 GB
Available for KV + overhead:  13 GB
```

**Verdict: Tight but testable for single-image evaluation.** 127 GB of weights
leaves ~13 GB for KV cache, activations, and vLLM overhead. For production
multi-image serving (10 images), this is insufficient. However, for single-image
A/B quality testing with `--max-model-len 4096`, the KV cache requirement drops
to under 1 GB (due to 75% GatedDeltaNet layers using linear attention with no
KV cache — only the 25% full-attention layers need KV storage at ~24 KB/token).
This means single-image quality benchmarking is feasible at FP8 on a single H200,
even if production multi-image serving requires NVFP4 or 2x GPU.

#### Scenario B: NVFP4 quantization

```
Model weights (NVFP4):        75.6 GB
Available for KV + overhead:  64.4 GB
```

**Verdict: Fits comfortably.** 64 GB of headroom is more than enough for KV
cache at 10K-50K token context lengths. A practical `--max-model-len` of 32768
to 65536 would work well.

**However:** NVFP4 is an aggressive 4-bit quantization. Quality degradation for
structured JSON extraction with 24 botanical traits has not been validated.
The hybrid GatedDeltaNet architecture may be more sensitive to quantization
than standard transformers.

### KV Cache Calculation (for full-attention layers only)

The model uses GQA with 2 KV heads, head_dim=256, and only 25% of layers (12
of 48) use full attention. KV cache per token (FP16):

```
KV per token = 2 (K+V) x 12 layers x 2 KV_heads x 256 head_dim x 2 bytes
             = 2 x 12 x 2 x 256 x 2 = 24,576 bytes = 24 KB/token

For 10K tokens: 10,000 x 24 KB = 240 MB
For 50K tokens: 50,000 x 24 KB = 1.2 GB
```

This is extremely efficient due to GatedDeltaNet reducing KV cache needs by 75%.

### vLLM Support

- **Status:** Fully supported. Official vLLM serving commands in model card.
- **Caveats:**
  - Requires `--attention-backend triton --kv-cache-dtype bf16` (FP8 KV cache
    produces corrupt output per [sglang#19603](https://github.com/sgl-project/sglang/issues/19603))
  - `--disable-cuda-graph --disable-radix-cache` recommended
  - MTP speculative decoding supported (2.75x speedup reported)
- **Recommended command (single GPU, NVFP4):**
  ```bash
  vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
    --max-model-len 32768 \
    --attention-backend triton \
    --kv-cache-dtype bf16 \
    --reasoning-parser qwen3
  ```

### Vision Benchmarks

| Benchmark | Score |
|-----------|-------|
| MMMU | 83.9 |
| MMMU-Pro | 76.9 |
| MathVision | 86.2 |
| MMBench-EN | 92.8 |
| OmniDocBench | 89.8 |

### Bottom Line

**FP8 fits for single-image quality testing** with `--max-model-len 4096` (~13 GB
headroom is sufficient for single-request short-context inference due to efficient
GatedDeltaNet KV cache). **NVFP4 fits for multi-image production** (~76 GB weights,
~64 GB headroom) but 4-bit quality impact on structured botanical extraction is
unknown. For production FP8 multi-image serving, need 2x H200 or 2x H100.

---

## 2. Mistral-Small-3.1-24B-Instruct-2503

### Model Identity

| Field | Value |
|-------|-------|
| HuggingFace | [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| Downloads | ~367K/month |
| License | Apache 2.0 |
| Type | Vision-language dense model (24B, NOT MoE) |

### Architecture

| Parameter | Value |
|-----------|-------|
| Total params | 24B |
| Active params | 24B (dense, not MoE) |
| Context length | 128K tokens |
| Tokenizer | Tekken (131K vocab) |
| Multi-image | Yes, up to 10 images per prompt |

### Weight Sizes

| Format | Size | Source |
|--------|------|--------|
| BF16 | ~48 GB | HuggingFace (consolidated.safetensors = 48 GB, or 10 shards totaling ~48 GB) |
| FP8 (RedHatAI) | ~25 GB | [RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic) |

Note: The HuggingFace repo shows total size ~96 GB because it contains BOTH
`consolidated.safetensors` (48 GB, Mistral format) and split shards
(`model-00001-of-00010` through `-00010`, ~48 GB, HF format). You load one or
the other, not both.

### VRAM Budget on H200 140GB

#### BF16

```
Model weights (BF16):         48 GB
Available for KV + overhead:  92 GB
```

**Verdict: Fits easily.** 92 GB headroom supports long context and batching.

#### FP8

```
Model weights (FP8):          25 GB
Available for KV + overhead: 115 GB
```

**Verdict: Fits with massive headroom.** Could serve `--max-model-len 131072`
with room for concurrent requests.

### KV Cache Calculation

Assuming standard GQA with 8 KV heads (Mistral architecture), head_dim=128, 40 layers:

```
KV per token = 2 x 40 x 8 x 128 x 2 bytes = 163,840 bytes = 160 KB/token

For 10K tokens: 10,000 x 160 KB = 1.6 GB
For 50K tokens: 50,000 x 160 KB = 8.0 GB
For 128K tokens: 128,000 x 160 KB = 20.5 GB
```

Even at full 128K context in BF16, total VRAM = 48 + 20.5 + ~5 (overhead) = ~74 GB.
Well within 140 GB.

### vLLM Support

- **Status:** Fully supported (vLLM >= 0.8.1)
- **Multi-image:** Explicitly supported with `--limit_mm_per_prompt 'image=10'`
- **Recommended command:**
  ```bash
  vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --limit_mm_per_prompt 'image=10' \
    --max-model-len 32768
  ```

### Vision Benchmarks

| Benchmark | Score |
|-----------|-------|
| MMMU | 64.0 |
| MMMU-Pro | 49.3 |
| MathVista | 68.9 |
| ChartQA | 86.2 |
| DocVQA | 94.1 |
| AI2D | 93.7 |

### Bottom Line

**Easily fits on a single H200 at BF16.** The 24B model leaves ~90 GB headroom.
Multi-image (up to 10) is explicitly supported. Strong document/chart understanding
(DocVQA 94.1, AI2D 93.7) is relevant for structured extraction. Lower MMMU (64.0)
compared to Qwen3.5 (83.9) suggests weaker general visual reasoning.

### Regarding "Mistral-Small-3-110B"

**No model called "mistral-small-3-110B-2603" exists.** What does exist:

| Model | Params | Vision? | Notes |
|-------|--------|---------|-------|
| [Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | 119B (6.5B active, MoE) | Yes (image + text) | Released June 2026. 128 experts, 4 active. |
| [Pixtral-Large-Instruct-2411](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411) | 124B (123B decoder + 1B vision) | Yes, up to 30 images | Dense model. Requires ~300 GB VRAM. Deprecated tag on Mistral website. |
| [Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | 123B | Text only | No vision capability. |

**Mistral-Small-4-119B-2603** is the closest to "110B with vision" -- it is a
119B MoE with 6.5B active params and multimodal input. Only 5,358 downloads/month
(very new). Its FP8/NVFP4 variants may fit on a single H200 but this has not
been widely tested yet.

---

## 3. Phi-4-Reasoning-Vision-15B

### Model Identity

| Field | Value |
|-------|-------|
| HuggingFace | [microsoft/Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B) |
| Downloads | ~22.8K/month |
| License | MIT |
| Released | 2026-03-04 |
| Type | Vision-language dense model (mid-fusion) |

### Architecture

| Parameter | Value |
|-----------|-------|
| Total params | 15B |
| Vision encoder | SigLIP-2 (NaFlex variant) |
| Fusion | Mid-fusion (visual tokens projected into LM embedding space) |
| Context length | **16,384 tokens** |
| Max visual tokens | **3,600** (dynamic resolution) |
| Precision | BF16 |
| Attention | Bidirectional within images (intra-image) |

### Weight Sizes

| Format | Size | Source |
|--------|------|--------|
| BF16 | **30.2 GB** | 7 safetensors shards (5x ~4.9GB + 4.77GB + 4.77GB + 1.03GB) |
| GGUF | Various | [jamesburton/Phi-4-reasoning-vision-15B-GGUF](https://huggingface.co/jamesburton/Phi-4-reasoning-vision-15B-GGUF) |

### VRAM Budget on H200 140GB

```
Model weights (BF16):         30.2 GB
Available for KV + overhead: 109.8 GB
```

**Verdict: Trivially fits.** Massive headroom. Could serve many concurrent
requests.

### Multi-Image Assessment

- Training includes multi-image and sequential-image tasks (Stage 3)
- Max 3,600 visual tokens per request (dynamic resolution)
- With 10 images at ~640 tokens each = 6,400 visual tokens -- **exceeds the
  3,600 token visual budget**
- Practically limited to ~5 images at 640 tokens/image, or 10 images at
  reduced resolution (~360 tokens/image)

**This is a significant constraint for our 10-image task.**

### Context Length Constraint

The 16,384 token context is tight:

```
10 images x 640 tokens = 6,400 visual tokens (if supported)
System prompt + 24-trait schema = ~500 tokens
Reasoning/CoT output = ~2,000-4,000 tokens
JSON output = ~1,500 tokens
Total = ~10,400-12,400 tokens
```

This leaves minimal headroom. With 5 images it would be ~7,200-9,200 tokens,
which is more comfortable but still tight.

### Vision Benchmarks

| Benchmark | Score |
|-----------|-------|
| MMMU (VAL) | 54.3 |
| MathVista (MINI) | 75.2 |
| AI2D (TEST) | 84.8 |
| ChartQA (TEST) | 83.3 |
| ScreenSpot-V2 | 88.2 |
| OCRBench | 76.0 |

### vLLM Support

- **Status:** Supported (vLLM >= 0.15.2 per model card)
- **Known issues:**
  - Phi-4-reasoning-plus has a [known bug with repetitive reasoning](https://github.com/vllm-project/vllm/issues/18141) when using `--enable-reasoning --reasoning-parser deepseek_r1`
  - GGUF format has reported issues across vLLM versions ([#16510](https://github.com/vllm-project/vllm/issues/16510))
  - Phi-4-multimodal-instruct has open compatibility requests ([#13936](https://github.com/vllm-project/vllm/issues/13936))
- BF16 safetensors serving is recommended

### Bottom Line

**Fits trivially on H200 but has two critical limitations:**
1. **16K context length** is restrictive for 10-image multi-trait extraction
2. **3,600 visual token cap** cannot accommodate 10 high-resolution images
3. **MMMU 54.3** is significantly below Qwen3.5 (83.9) and even Mistral-Small-3.1 (64.0)

The 15B model size raises concerns about reliable 24-trait structured JSON
extraction. Smaller models tend to produce more hallucinated fields and
inconsistent JSON formatting. The low MMMU score (54.3) suggests weaker visual
understanding than alternatives.

**Not recommended** for our task due to context length, visual token budget,
and benchmark performance constraints.

---

## 4. Kimi-K2.5 (Moonshot AI)

### Model Identity

| Field | Value |
|-------|-------|
| HuggingFace | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) |
| NVFP4 variant | [nvidia/Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4) |
| Downloads | ~3.4M/month (base), ~377K/month (NVFP4) |
| License | Kimi-K2.5 Community License |
| Type | Vision-language MoE (image + video + text) |

### Architecture

| Parameter | Value |
|-----------|-------|
| Total params | **1T (1 trillion)** |
| Active params/token | **32B** |
| Experts | 384 total, 8 selected per token + 1 shared |
| Layers | 61 (including 1 dense layer) |
| Attention hidden dim | 7,168 |
| MoE hidden dim/expert | 2,048 |
| Attention heads | 64 |
| Attention mechanism | MLA (Multi-head Latent Attention) |
| Context length | 256K tokens |
| Vision encoder | MoonViT (400M params) |
| Native quantization | INT4 |

### Weight Sizes

| Format | Size | Source |
|--------|------|--------|
| INT4 (native/base) | **~595 GB** | 64 safetensors shards on HuggingFace |
| NVFP4 (NVIDIA) | **~591 GB** | 119 safetensors shards (119x ~5GB each) |
| GGUF Q2 (Unsloth) | ~240 GB | 1.8-bit dynamic quantization |
| GGUF Q4 | ~350-400 GB (est.) | Various community quantizations |

### VRAM Budget on H200 140GB

#### NVFP4

```
Model weights (NVFP4):       591 GB
Single H200 VRAM:            140 GB
```

**Verdict: Absolutely does NOT fit.** Even the most aggressive quantization
(NVFP4) is 591 GB -- over 4x the H200's capacity. The NVIDIA model card
recommends `--tensor-parallel-size 4` on B200 GPUs (each 192 GB = 768 GB total).

#### Why is NVFP4 still 591 GB?

Despite being "FP4", the model has 1 trillion parameters. At 4 bits (0.5 bytes)
per parameter: 1T x 0.5 = 500 GB for weights alone, plus metadata, scales,
and non-quantized layers (embeddings, norms). The NVFP4 quantization reduces
from the original BF16 size but the base model is simply too large.

### Available Quantizations

| Format | Provider | Size | Fits 140GB? |
|--------|----------|------|-------------|
| INT4 (native) | Moonshot AI | ~595 GB | No |
| NVFP4 | NVIDIA | ~591 GB | No |
| GGUF Q1.8 (dynamic) | Unsloth | ~240 GB | No |
| GGUF Q2 | Community | ~250 GB | No |
| AWQ | Not available | - | - |
| GPTQ | Not available | - | - |

**No quantization exists that fits on a single H200.**

### Minimum GPU Requirements

- **4x B200** (768 GB total) -- NVIDIA's tested configuration
- **8x H100** (640 GB total) -- may work with NVFP4
- **8x H200** (1,120 GB total) -- comfortable

### Smaller Variants

Moonshot AI has **not** released a smaller variant of K2.5. Related models:
- Kimi-K2-Instruct: Same 1T architecture (text-only, no vision)
- Kimi-K2-Thinking: Same 1T architecture
- No 7B/13B/32B distilled vision variant exists

### Vision Benchmarks

| Benchmark | Score |
|-----------|-------|
| MMMU-Pro | 78.5 |
| OCRBench | 92.3 |
| VideoMMMU | 86.6 |
| MathVista | 90.1 |

### vLLM Support

- **Status:** Supported (via DeepSeek V3 architecture backend)
- **NVFP4 command:**
  ```bash
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --tensor-parallel-size 4 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code
  ```

### Bottom Line

**Cannot fit on a single H200 under any quantization.** The 1T parameter count
makes this a multi-node model. Even the most aggressive community quantizations
(1.8-bit GGUF at 240 GB) exceed single-GPU capacity. No smaller distilled
variants exist.

---

## Summary Comparison

| Model | Weights (practical) | Fits H200 140GB? | Multi-image? | MMMU | Context | vLLM | Recommendation |
|-------|-------------------|-------------------|--------------|------|---------|------|----------------|
| Qwen3.5-122B-A10B (FP8) | 127 GB | **Yes** (single-image testing) | Yes | 83.9 | 262K | Yes | `--max-model-len 4096` for eval; NVFP4 or 2x GPU for production |
| Qwen3.5-122B-A10B (NVFP4) | 76 GB | **Yes** | Yes | TBD | 262K | Yes | Quality impact unknown |
| Mistral-Small-3.1-24B (BF16) | 48 GB | **Yes** | Yes (10 imgs) | 64.0 | 128K | Yes | Best single-GPU option |
| Mistral-Small-3.1-24B (FP8) | 25 GB | **Yes** | Yes (10 imgs) | 64.0 | 128K | Yes | Most headroom |
| Phi-4-Reasoning-Vision-15B | 30 GB | **Yes** | Limited | 54.3 | 16K | Yes* | Not recommended (context/quality) |
| Kimi-K2.5 (NVFP4) | 591 GB | **No** | Yes | 78.5 | 256K | Yes | Needs 4+ GPUs minimum |

### For our task (single H200 140GB):

**Tier 1 -- Testable at FP8 for single-image quality evaluation:**
- **Qwen3.5-122B-A10B (FP8)** at `--max-model-len 4096`. 127 GB weights + ~1 GB
  KV for single image fits within 140 GB. Best benchmarks (MMMU 83.9). Use this
  to establish a quality ceiling before committing to NVFP4 or multi-GPU.
- **Mistral-Small-3.1-24B** at BF16 or FP8. Proven vLLM support, explicit
  10-image support, 128K context, 92 GB headroom at BF16. Lower MMMU (64.0).

**Tier 2 -- Production multi-image serving:**
- **Qwen3.5-122B-A10B at NVFP4** (~76 GB). Fits with ~64 GB headroom for
  multi-image KV cache. 4-bit quality on structured JSON extraction unvalidated.
- **Mistral-Small-3.1-24B** at BF16 — already production-ready for 10 images.
- **Mistral-Small-4-119B-2603 (FP8)** ~126 GB weights — testable for
  single-image quality evaluation. MoE with 6.5B active, Pixtral vision encoder.
  3 days old, no vision benchmarks yet.

**Tier 3 -- Limited utility:**
- **Phi-4-Reasoning-Vision-15B**: Fits easily (30 GB) but 16K context and 3,600
  visual token cap are hard limitations. Testable for single-image evaluation but
  MMMU 54.3 is weak. May still be useful as a fast iteration model for prompt
  engineering experiments.
- **Kimi-K2.5**: Does not fit on any single GPU under any quantization.
