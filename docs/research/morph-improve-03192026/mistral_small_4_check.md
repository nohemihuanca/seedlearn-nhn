# Mistral Small 4 (119B) -- Model Evaluation

Last verified: 2026-03-19

## Summary

Mistral Small 4 is a **real, publicly available** model released on **2026-03-16** by Mistral AI.
It is a 119B-parameter Mixture-of-Experts model with **native vision capabilities** via a Pixtral
vision encoder. Released under Apache 2.0. Available on HuggingFace as
[`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603).

**Key question for SeedLearn**: Can it replace Qwen3-VL-32B as the Stage 1 vision-LLM?
**Answer**: Possibly, but it **cannot fit on a single H200 140GB** at FP8, and FP4 is tight.
Multi-GPU (TP=2) is the intended deployment. See VRAM analysis below.

---

## 1. Model Card and Naming

| Field | Value |
|-------|-------|
| Official name | `Mistral-Small-4-119B-2603` |
| HuggingFace URL | https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 |
| Announcement | https://mistral.ai/news/mistral-small-4 (2026-03-16) |
| License | Apache 2.0 |
| Model type | `mistral3` (multimodal, conditional generation) |

There is **no separate "-Instruct" variant**. This single model unifies instruct, reasoning
(formerly Magistral), and agentic (formerly Devstral) capabilities via a `reasoning_effort`
parameter (`"none"` for instruct, `"high"` for reasoning).

---

## 2. Architecture

**Mixture of Experts (MoE)**, not dense.

| Parameter | Value |
|-----------|-------|
| Total parameters | 119B |
| Active parameters per token | 6.5B (6B per Mistral's announcement) |
| Number of layers | 36 |
| Hidden size | 4,096 |
| Intermediate size (MLP) | 12,288 |
| Attention heads | 32 |
| KV heads | 32 (no GQA -- full MHA) |
| Head dimension | 128 |
| Routed experts | 128 |
| Shared experts | 1 |
| Experts active per token | 4 |
| MoE intermediate size | 2,048 |
| Vocabulary size | 131,072 |
| Max position embeddings | 1,048,576 (1M tokens) |
| Advertised context length | 256K tokens |
| RoPE | YaRN with scaling |
| Native dtype | bfloat16 |
| Architecture class | `Mistral3ForConditionalGeneration` |

### Vision Encoder (Pixtral)

| Parameter | Value |
|-----------|-------|
| Vision model type | pixtral |
| Image size | 1540 x 1540 pixels |
| Patch size | 14 |
| Vision layers | 24 |
| Vision hidden size | 1,024 |
| Vision intermediate size | 4,096 |
| Vision attention heads | 16 |
| Spatial merge size | 2 |
| Channels | 3 (RGB) |

---

## 3. Vision / Multimodal Capabilities

- **Yes, natively multimodal** -- accepts text + image inputs, text output.
- Vision encoder is **Pixtral** (Mistral's own vision architecture).
- Supports image understanding, document analysis, data extraction from images.
- **Multi-image support**: Not explicitly documented in the model card. The Pixtral architecture
  in previous Mistral models (Pixtral Large) did support multi-image, so it is likely supported
  but unconfirmed for this specific release.
- **Image resolution**: 1540x1540 native, patch size 14.

---

## 4. VRAM Requirements -- H200 140GB Feasibility

### Raw Weight Sizes

| Precision | Weight Size | Fits on 1x H200 (140GB)? |
|-----------|-------------|---------------------------|
| BF16 | ~238 GB | No |
| FP8 (E4M3) | ~126 GB | Barely fits weights only, no KV cache headroom |
| NVFP4 | ~65-70 GB | Yes for weights, but KV cache at 256K context is large |
| GGUF Q4_K_M | ~74 GB | Similar situation |
| GGUF Q3_K_M | ~57 GB | Plausible with limited context |
| GGUF IQ2_M | ~42 GB | Yes, but significant quality loss |

### Mistral's Official Hardware Recommendations

| Setup | Hardware |
|-------|----------|
| **Minimum** | 4x HGX H100, **2x HGX H200**, or 1x DGX B200 |
| **Recommended** | 4x HGX H100, 4x HGX H200, or 2x DGX B200 |

### Analysis for Single H200 (140GB)

**FP8**: The Q8_0 GGUF is 126 GB. With 140 GB VRAM, that leaves ~14 GB for KV cache, activations,
and CUDA overhead. At 256K context with 32 KV heads and head_dim=128, the KV cache alone would
consume tens of GB. **Verdict: Does not practically fit at FP8 on a single H200.**

**FP4 / NVFP4**: Weights are ~65-70 GB, leaving ~70 GB for KV cache and overhead. This is
workable for **short context** (a few thousand tokens, typical for image + prompt in SeedLearn
Stage 1). However:
- vLLM's NVFP4 serving example still uses `--tensor-parallel-size 2`
- Quality degradation at 4-bit for a 119B MoE model may be acceptable since only 6.5B params
  are active per token, but this needs benchmarking
- **Verdict: Technically possible at FP4 with short context, but not recommended by Mistral.**

**Bottom line**: This model is designed for **2+ GPU** deployment. For our single-H200 SeedLearn
setup, Qwen3-VL-32B remains more practical.

---

## 5. vLLM Support

**Yes, supported.** Mistral provides a custom Docker image and vLLM serving commands.

```bash
# Official recommended serving (requires 2+ GPUs)
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN_MLA \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --reasoning-parser mistral \
  --max_num_batched_tokens 16384 \
  --max_num_seqs 128 \
  --gpu_memory_utilization 0.8

# NVFP4 variant (still TP=2)
vllm serve mistralai/Mistral-Small-4-119B-2603-NVFP4 \
  --tensor-parallel-size 2 \
  --attention-backend TRITON_MLA
```

**Custom Docker**: `mistralllm/vllm-ms4:latest` -- includes fixes for tool calling and reasoning
parsing not yet in upstream vLLM.

**Note**: The model uses `FLASH_ATTN_MLA` (Multi-head Latent Attention) backend, which may require
a recent vLLM version or the custom Docker. Our current source-built vLLM may need updating.

---

## 6. Benchmarks

| Benchmark | Score | Notes |
|-----------|-------|-------|
| AIME 2025 (AA LCR) | 0.72 | Competitive with GPT-OSS 120B, 20% less output |
| LiveCodeBench | -- | Outperforms GPT-OSS 120B |
| MMLU-Pro | 78.0 | Community evaluation |
| GPQA Diamond | 71.2 | Community evaluation |
| **MMMU** | **Not reported** | No MMMU scores in model card or announcement |

### Key Performance Claims
- 40% latency reduction vs Mistral Small 3
- 3x throughput improvement vs Mistral Small 3
- Generates 1.6K character outputs vs 5.8-6.1K for competitors at similar accuracy
- Reasoning mode equivalent to previous Magistral models
- Non-reasoning mode equivalent to Mistral-Small-3.2-24B-Instruct-2506

**No vision-specific benchmarks** (MMMU, MathVista, DocVQA, etc.) are reported, which is a
notable omission for a multimodal model. This makes it hard to evaluate for SeedLearn Stage 1.

---

## 7. Multi-Image Support

**Not explicitly documented.** The Pixtral vision encoder architecture in previous Mistral models
(Pixtral Large 124B) supported multi-image input. The `Mistral3ForConditionalGeneration`
architecture class suggests continuity, but the model card does not confirm multi-image for this
release.

For SeedLearn Stage 1, we process single images per specimen, so this is not a blocker.

---

## 8. Download Counts and Community Adoption

| Metric | Value |
|--------|-------|
| Downloads (last month) | 5,358 |
| Likes | 240 |
| Community discussions | 15 |
| Release date | 2026-03-16 (3 days ago) |
| Total quantized variants | 25+ community uploads |

Adoption is early given the 3-day-old release. The 25+ community quantizations appearing within
days indicates strong interest. For comparison, Qwen3-VL-32B has hundreds of thousands of
downloads.

---

## 9. Quantized Variants

### Official (from mistralai)

| Variant | Downloads | Notes |
|---------|-----------|-------|
| `Mistral-Small-4-119B-2603` (BF16/FP8) | 5,358 | Base model, safetensors |
| `Mistral-Small-4-119B-2603-NVFP4` | 723 | 4-bit float, vLLM optimized |
| `Mistral-Small-4-119B-2603-eagle` | 188 | Speculative decoding variant |

### Community

| Variant | Organization | Downloads | Format |
|---------|-------------|-----------|--------|
| GGUF (multiple quants) | unsloth | 17,900 | Q2-Q8, BF16 |
| GGUF | bartowski | 4,800 | GGUF |
| GGUF | lmstudio-community | 4,310 | GGUF |
| GGUF | AaryanK | 4,010 | GGUF |
| AWQ 4-bit | cyankiwi | 94 | AWQ |
| MLX 4-bit | mlx-community | 825 | MLX |
| MLX 9-bit | inferencerlabs | 394 | MLX |
| MXFP4_MOE GGUF | noctrex | 296 | GGUF |

**GGUF quant sizes** (from unsloth):

| Quant | File Size |
|-------|-----------|
| IQ1_M | 29 GB |
| IQ2_M | 42 GB |
| Q3_K_M | 57 GB |
| Q4_K_M | 74 GB |
| Q5_K_M | 88 GB |
| Q6_K | 98 GB |
| Q8_0 | 126 GB |
| BF16 | 238 GB |

---

## 10. Relevance to SeedLearn

### Pros
- Native vision via Pixtral encoder (1540x1540 resolution -- good for seed morphology)
- Apache 2.0 license
- Unified instruct + reasoning in one model
- MoE efficiency: only 6.5B params active per token despite 119B total
- Function calling and structured output support

### Cons
- **Cannot fit on single H200 at acceptable quality** (FP8 too tight, FP4 unverified quality)
- No reported vision benchmarks (MMMU, etc.) -- cannot compare to Qwen3-VL-32B
- Very new (3 days old) -- limited community validation
- Requires custom vLLM Docker or fork for full feature support
- Multi-GPU requirement conflicts with our single-GPU pipeline setup

### Recommendation

**Do not adopt for SeedLearn Stage 1 at this time.** The model requires 2+ GPUs for practical
deployment, and vision-specific benchmarks are absent. Qwen3-VL-32B remains the better fit for
our single-H200 setup.

**Revisit if**:
1. Vision benchmarks (MMMU, DocVQA) are published and show meaningful improvement over Qwen3-VL
2. We gain access to multi-GPU nodes for pipeline serving
3. Community validates FP4 quality for vision tasks specifically
4. A smaller Mistral 4 variant is released (the non-reasoning mode claims parity with 24B model)
