# VLM Model vLLM Serving Parameters

Research compiled from HuggingFace model cards, generation_config.json files, and
vLLM documentation. Last verified: 2026-03-20.

## Current vLLM Status

| Item | Value |
|------|-------|
| **Latest stable** | v0.17.1 (2026-03-11) |
| **Our install** | v0.14.0rc2 (2026-01 source build, RHEL 8 glibc 2.28) |
| **Gap** | 3 major versions behind; v0.15.0, v0.16.0, v0.17.0 all released since |
| **Notable in v0.17.0** | PyTorch 2.10, FlashAttention 4, 699 commits from 272 contributors |

---

## 1. mistralai/Mistral-Small-4-119B-2603

| Parameter | Value |
|-----------|-------|
| **Architecture** | Dense 119B, multimodal (vision + text) |
| **Context window** | 256K tokens (generation_config max_length: 1,048,576) |
| **Precision** | BF16 (FP8 variant available as `-NVFP4`) |

### vLLM Serve Command

```bash
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
```

### Generation Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.1 | `reasoning_effort="none"` |
| temperature | 0.7 | `reasoning_effort="high"` |
| do_sample | true | |

generation_config.json: minimal -- only bos/eos/pad token IDs and max_length=1048576.

### Special Flags

- `--attention-backend FLASH_ATTN_MLA` -- optimized attention for this architecture
- `--tool-call-parser mistral` -- native function calling
- `--reasoning-parser mistral` -- reasoning mode with `reasoning_effort` per-request param
- `--enable-auto-tool-choice` -- automatic tool selection

### Minimum vLLM Version

**Not yet in mainline vLLM.** As of 2026-03-16, requires either:
1. Mistral's custom Docker: `docker pull mistralllm/vllm-ms4:latest`
2. Custom branch: `git clone --branch fix_mistral_parsing https://github.com/juliendenize/vllm.git`
3. Pending PR [#37081](https://github.com/vllm-project/vllm/pull/37081) expected to merge soon

Also requires: `mistral_common >= 1.10.0`, `transformers` from git main (5.3.0.dev0).

### Multi-Image Support

Supported via `image_url` in chat messages. No documented limit on image count.

### Known Issues

- FP8 weights not natively loadable via Transformers; BF16 workaround needed
- Requires custom vLLM fork until PR #37081 merges
- Speculative decoding variant available: `Mistral-Small-4-119B-2603-eagle`

### H200 VRAM Estimate (1x 140GB)

TP=2 required (2x H200). At BF16, ~238GB for weights alone. NVFP4 variant may fit on 1x H200.

---

## 2. google/gemma-3-27b-it

| Parameter | Value |
|-----------|-------|
| **Architecture** | Dense 27B, multimodal (vision + text) |
| **Context window** | 128K tokens |
| **Precision** | BF16 |

### vLLM Serve Command

No official vLLM serve command in the model card. Based on vLLM docs and community usage:

```bash
vllm serve google/gemma-3-27b-it \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

### Generation Defaults

generation_config.json (from RedHatAI FP8 variant, same base model):

| Parameter | Value |
|-----------|-------|
| cache_implementation | hybrid |
| bos_token_id | 2 |
| eos_token_id | [1, 106] |
| pad_token_id | 0 |

No sampling parameters (temperature, top_p, etc.) in generation_config.json -- all defaults.

### Special Flags

- `--trust-remote-code` -- may be needed depending on vLLM version
- Hybrid KV cache implementation (sliding window + full attention layers)
- `--enforce-eager` may be needed for stability (per RedHatAI evaluation config)

### Minimum vLLM Version

Gemma 3 support added in vLLM v0.7.x+ (early 2025). No specific minimum documented.
Our v0.14.0rc2 should support it.

### Multi-Image Support

Gemma 3 supports multiple images per conversation turn. Images passed via `image_url`
in OpenAI-compatible chat format.

### Known Issues

- No official vLLM recipe page exists (as of 2026-03-20)
- Hybrid cache (sliding window attention) can cause compatibility issues with some vLLM features
- BF16 model is large (~54GB); FP8 variant recommended for single-GPU deployment

### H200 VRAM Estimate (1x 140GB)

BF16: ~54GB weights. Fits on 1x H200 with room for large KV cache.

---

## 3. RedHatAI/gemma-3-27b-it-FP8-dynamic

| Parameter | Value |
|-----------|-------|
| **Architecture** | Dense 27B, multimodal (vision + text) |
| **Context window** | 128K tokens |
| **Precision** | FP8 dynamic quantization |
| **Base model** | google/gemma-3-27b-it |

### vLLM Serve Command (from model card evaluation config)

```bash
vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enforce-eager
```

### Python API Example

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="RedHatAI/gemma-3-27b-it-FP8-dynamic",
    trust_remote_code=True
)

sampling_params = SamplingParams(temperature=0.2, max_tokens=64)
```

### Generation Defaults

| Parameter | Value |
|-----------|-------|
| temperature | 0.2 | (from model card example) |
| max_tokens | 64 | (example only; increase for production) |

### Special Flags

- `--trust-remote-code` -- required
- `--enable-chunked-prefill` -- recommended
- `--enforce-eager` -- recommended for stability
- `add_bos_token=True` -- required per evaluation config
- `dtype=auto` -- auto-selects FP8

### Minimum vLLM Version

**>= v0.5.2** (explicitly stated in model card).

### Multi-Image Support

Same as base Gemma 3 -- supported via `multi_modal_data` dict with image list.

### Known Issues

- `max_model_len=4096` in eval config is very conservative; can increase for production
- FP8 dynamic quantization may have slightly different behavior than static FP8

### H200 VRAM Estimate (1x 140GB)

FP8: ~27GB weights. Easily fits on 1x H200 with large context window.

---

## 4. Qwen/Qwen3.5-27B-FP8

| Parameter | Value |
|-----------|-------|
| **Architecture** | Dense 27B, early-fusion multimodal (vision + video + text) |
| **Context window** | 262K tokens (extendable to 1,010K via YaRN) |
| **Precision** | FP8 (fine-grained, block size 128) |
| **Thinking mode** | Yes, enabled by default |

### vLLM Serve Command

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

With tool calling:
```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Text-only mode (disable vision encoder):
```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

### Generation Defaults

generation_config.json:

| Parameter | Value |
|-----------|-------|
| do_sample | true |
| temperature | 0.6 |
| top_k | 20 |
| top_p | 0.95 |

Model card recommended sampling (task-dependent):

| Mode | Task | temperature | top_p | top_k | presence_penalty |
|------|------|-------------|-------|-------|-----------------|
| Thinking | General | 1.0 | 0.95 | 20 | 1.5 |
| Thinking | Coding | 0.6 | 0.95 | 20 | 0.0 |
| Instruct | General | 0.7 | 0.8 | 20 | 1.5 |
| Instruct | Reasoning | 1.0 | 1.0 | 40 | 2.0 |

All modes: `repetition_penalty=1.0`, `min_p=0.0`.

### Special Flags

- `--reasoning-parser qwen3` -- required for thinking mode (`<think>...</think>` blocks)
- `--language-model-only` -- disables vision encoder for text-only use
- Thinking mode control via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- Multi-token prediction: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`

### Minimum vLLM Version

**Requires vLLM main branch (nightly).** Install via:
```bash
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

Qwen3.5 is too new for v0.17.1 stable -- needs nightly builds.

### Multi-Image Support

Native early-fusion multimodal: supports images and video directly.
- Images via `image_url` in chat messages
- Video via `video_url` with configurable frame sampling: `extra_body={"mm_processor_kwargs": {"fps": 2, "do_sample_frames": True}}`

### Known Issues

- TP=8 recommended in model card is for full 262K context; reduce max-model-len to use fewer GPUs
- Reduce `--max-model-len` if OOM; minimum ~128K recommended to preserve thinking quality
- YaRN static scaling may hurt performance on shorter texts
- `presence_penalty` > 1.5 can cause language mixing artifacts
- Multi-turn: only include final output in history (not thinking content)

### Recommended Max Output Tokens

- General: 32,768
- Complex math/coding: 81,920

### H200 VRAM Estimate (1x 140GB)

FP8: ~27GB weights. Fits on 1x H200 but TP=8 recommended for full 262K context due to KV cache.
For shorter contexts (8-32K), TP=1 should work.

---

## 5. Qwen/Qwen3.5-122B-A10B-FP8

| Parameter | Value |
|-----------|-------|
| **Architecture** | MoE 122B total / 10B active, early-fusion multimodal |
| **MoE config** | 48 layers, 256 experts, 8 routed + 1 shared per layer |
| **Context window** | 262K tokens (extendable to 1,010K via YaRN) |
| **Precision** | FP8 (fine-grained, block size 128) |
| **Thinking mode** | Yes, enabled by default |

### vLLM Serve Command

```bash
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

With tool calling:
```bash
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Text-only mode:
```bash
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

### Generation Defaults

Same as Qwen3.5-27B-FP8 (identical model card recommendations):

| Mode | Task | temperature | top_p | top_k | presence_penalty |
|------|------|-------------|-------|-------|-----------------|
| Thinking | General | 1.0 | 0.95 | 20 | 1.5 |
| Thinking | Coding | 0.6 | 0.95 | 20 | 0.0 |
| Instruct | General | 0.7 | 0.8 | 20 | 1.5 |
| Instruct | Reasoning | 1.0 | 1.0 | 40 | 2.0 |

### Special Flags

Same as Qwen3.5-27B-FP8, plus:
- MoE-specific: experts are handled internally by vLLM; no extra MoE flags needed for TP serving
- `--enable-expert-parallel` -- optional, distributes experts across GPUs (useful for MoE)

### Minimum vLLM Version

**Requires vLLM main branch (nightly).** Same as Qwen3.5-27B.

### Multi-Image Support

Same as Qwen3.5-27B-FP8 -- native early-fusion multimodal with image and video support.

### Known Issues

- Despite only 10B active params, all 122B params must be loaded into VRAM (MoE weight storage)
- FP8: ~61GB for weights alone; fits on 1x H200 but KV cache for long contexts needs more
- No official soft-switch for thinking mode (unlike Qwen3 base)
- `presence_penalty` 0-2 range; higher values may cause language mixing

### H200 VRAM Estimate (1x 140GB)

FP8: ~61GB weights. Fits on 1x H200 for moderate context lengths.
TP=2 recommended for full 262K context. TP=8 for production throughput.

---

## 6. Qwen/Qwen3-VL-32B-Thinking-FP8

| Parameter | Value |
|-----------|-------|
| **Architecture** | Dense 32B, VL model with thinking mode |
| **Context window** | 262K tokens |
| **Precision** | FP8 |
| **Thinking mode** | Yes, always on |
| **Note** | Transformers direct loading NOT supported; must use vLLM or SGLang |

### vLLM Serve Command

No explicit `vllm serve` command in model card. From the Python API example and
vLLM Qwen3-VL recipe docs:

```bash
vllm serve Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0
```

For H200 with more context:
```bash
vllm serve Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --async-scheduling \
  --limit-mm-per-prompt.video 0
```

### Python API Example (from model card)

```python
llm = LLM(
    model="Qwen/Qwen3-VL-32B-Thinking-FP8",
    trust_remote_code=True,
    gpu_memory_utilization=0.70,
    enforce_eager=False,
    tensor_parallel_size=torch.cuda.device_count(),
    seed=0
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1024,
    top_k=-1,
    stop_token_ids=[],
)
```

### Generation Defaults

generation_config.json:

| Parameter | Value |
|-----------|-------|
| do_sample | true |
| temperature | 0.8 |
| top_k | 20 |
| top_p | 0.95 |
| repetition_penalty | 1.0 |

Model card recommended (task-dependent):

| Task Type | temperature | top_p | top_k | presence_penalty | max_tokens |
|-----------|-------------|-------|-------|-----------------|------------|
| VL tasks | 1.0 | 0.95 | 20 | 0.0 | 40,960 |
| Text tasks | 1.0 | 0.95 | 20 | 1.5 | 32,768 |
| Hard math/code | 1.0 | 0.95 | 20 | 1.5 | 81,920 |

### Special Flags

- `--trust-remote-code` -- **required**
- `VLLM_WORKER_MULTIPROC_METHOD='spawn'` -- **required** environment variable
- `--limit-mm-per-prompt.video 0` -- disable video if image-only (saves memory)
- `--async-scheduling` -- recommended for throughput
- `--mm-encoder-tp-mode data` -- optimize vision encoder distribution (from Qwen3-VL recipe)
- Requires `qwen_vl_utils >= 0.0.14` for `process_vision_info()`

### Minimum vLLM Version

Not explicitly stated. Based on Qwen3-VL architecture support, likely requires v0.15.0+.
The Qwen3-VL recipe page exists in vLLM docs, suggesting mainline support.

### Multi-Image Support

Full multi-image and video support:
- Images via `image_url` in chat messages
- Video via `video_url` with frame sampling kwargs
- Uses `qwen_vl_utils.process_vision_info()` for preprocessing
- `multi_modal_data` dict accepts `{"image": [img1, img2, ...]}` for multiple images

### Known Issues

- **Cannot load via Transformers** -- vLLM or SGLang only
- `gpu_memory_utilization=0.70` in model card example is conservative; can increase to 0.90+
- Thinking mode is always on (no soft-switch to disable)
- For greedy decoding, model card uses `temperature=0` despite generation_config default of 0.8

### H200 VRAM Estimate (1x 140GB)

FP8: ~32GB weights + vision encoder. Fits easily on 1x H200.

---

## Comparison Summary

| Model | Params | Active | FP8 Weight Size | Min TP (H200) | Min vLLM | Thinking | Vision | Status |
|-------|--------|--------|----------------|---------------|----------|----------|--------|--------|
| Mistral-Small-4-119B | 119B | 119B | ~60GB | 2 (BF16) | Custom fork | Yes (per-request) | Yes | Needs custom fork |
| gemma-3-27b-it | 27B | 27B | N/A (BF16 ~54GB) | 1 | v0.7+ | No | Yes | Supported |
| gemma-3-27b-it-FP8 | 27B | 27B | ~27GB | 1 | v0.5.2+ | No | Yes | Supported |
| Qwen3.5-27B-FP8 | 27B | 27B | ~27GB | 1 (short ctx) | Nightly | Yes (default) | Yes (native) | Needs nightly |
| Qwen3.5-122B-A10B-FP8 | 122B | 10B | ~61GB | 1 (short ctx) | Nightly | Yes (default) | Yes (native) | Needs nightly |
| Qwen3-VL-32B-Thinking-FP8 | 32B | 32B | ~32GB | 1 | v0.15+ | Yes (always on) | Yes | Likely supported |

## Recommended Sampling for Morphological Trait Extraction

For our use case (structured JSON trait extraction from seedling images), the recommended
settings across models would be:

| Parameter | Recommended | Rationale |
|-----------|------------|-----------|
| temperature | 0.1-0.3 | Low temperature for deterministic, structured output |
| top_p | 0.95 | Standard nucleus sampling |
| top_k | 20 | Prevents low-probability hallucinations |
| repetition_penalty | 1.0 | No penalty needed for structured extraction |
| presence_penalty | 0.0 | Avoid for structured output (can cause omissions) |
| max_tokens | 2048-4096 | Sufficient for 24-trait JSON response |
| thinking mode | Disabled | Extra latency without benefit for extraction tasks |

## Upgrade Path for Our vLLM Install

Our current v0.14.0rc2 supports:
- gemma-3-27b-it (BF16 and FP8) -- **ready now**
- Qwen3-VL-32B-Thinking-FP8 -- **likely supported** (Qwen3-VL architecture)

Requires v0.17.1+ or nightly:
- Qwen3.5-27B-FP8 -- needs nightly
- Qwen3.5-122B-A10B-FP8 -- needs nightly
- Mistral-Small-4-119B -- needs custom fork (not even in nightly yet)

**Recommendation:** Rebuild vLLM to v0.17.1 stable to unlock Qwen3.5 models. Mistral
Small 4 requires waiting for PR #37081 to merge or using their custom Docker image.
