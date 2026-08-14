# Prompt Engineering for VLM Botanical Trait Extraction

> Research report on techniques for improving vision-language model accuracy on
> morphological trait extraction from tropical seedling photographs.
>
> Last verified: 2026-03-19

---

## Executive Summary

This report evaluates prompt engineering techniques for improving Qwen3-VL-32B
accuracy on 24-trait botanical morphological extraction. The key findings are:

1. **Few-shot image prompting is technically feasible but empirically unreliable.**
   Qwen3-VL supports multi-image input via vLLM, and the OpenAI-compatible API
   allows interleaved image-text messages. However, recent research (Santos et al.,
   2025) demonstrates that current VLMs largely *ignore visual content* in
   demonstration examples, relying instead on textual cues. This means reference
   images of "serrate vs. dentate" margins would likely be processed but not
   meaningfully used for trait discrimination. The VRAM cost (~250-1,000 tokens
   per reference image) makes this approach expensive for uncertain returns.

2. **The highest-ROI improvements are textual, not visual.** Three techniques
   have strong evidence and are immediately implementable:
   - **Botanical definitions in the prompt** (estimated +10-15% on ambiguous traits)
   - **Taxonomic decision trees** for confusable trait pairs (estimated +15-20% on
     leaf margin, apex, base discrimination)
   - **JSON schema-constrained decoding** via vLLM's structured output feature
     (eliminates parse failures, may improve trait accuracy by constraining the
     output space)

3. **Prompt decomposition** (one trait at a time) shows marginal accuracy gains
   (+0.7% F1) but at 10x token cost. Not recommended for 24 traits.

4. **Chain-of-thought prompting** has mixed evidence for VLM visual tasks. A
   lightweight variant (parenthetical justifications, already in SYS1) is likely
   near-optimal. Full CoT adds latency without proven gains for structured
   extraction.

**Recommended priority order:**
1. Add botanical definitions to existing prompts (low effort, high impact)
2. Add decision-tree logic for confusable traits (medium effort, high impact)
3. Enable vLLM JSON schema-constrained decoding (low effort, medium impact)
4. Experiment with reference image grids as a single composite (medium effort, uncertain impact)
5. Test prompt decomposition for the 3-5 most error-prone traits only (medium effort, marginal impact)

---

## 1. Few-Shot Image Prompting for VLMs

### 1.1 Technical Feasibility

**Can modern VLMs accept multiple reference images alongside a query image?**

Yes. Qwen3-VL (and its predecessors Qwen2-VL, Qwen2.5-VL) natively support
multi-image input. The vLLM OpenAI-compatible API accepts multiple `image_url`
content blocks in a single message:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Reference: serrate margin example"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Reference: dentate margin example"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Now classify this specimen:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    ],
}]
```

The vLLM server must be launched with `--limit-mm-per-prompt image=N` to allow
N images per request (default varies by model).

**Source:** [vLLM Qwen3-VL Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html),
[vLLM Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

### 1.2 State of the Art: Visual In-Context Learning (2024-2026)

The field has progressed through several key milestones:

| Model/Paper | Year | Approach | Key Finding |
|---|---|---|---|
| Flamingo (DeepMind) | 2022 | Interleaved image-text pretraining | First large-scale visual ICL; demonstrated few-shot learning from image demonstrations |
| VILA (NVIDIA) | 2024 | Interleaved pretraining analysis | Interleaved data essential for ICL; freezing LLM during pretraining destroys ICL capability |
| Santos et al. | 2025 | Attention analysis of 7 VLMs | **VLMs primarily use textual cues, not visual content, from demonstrations** |
| VL-ICL Bench (ICLR 2025) | 2025 | Comprehensive ICL benchmark | LLaVA-OneVision-72B and GPT-4V show best ICL; smaller models struggle |
| SAVs (ICCV 2025) | 2025 | Sparse Attention Vectors | 20 examples per label yield SOTA few-shot VL classification without fine-tuning |
| VLM-ICL Inspection | 2025 | Manufacturing defect detection | Single in-context example improved F1 from baseline to 0.95 on MVTec AD |

### 1.3 Critical Limitation: VLMs Do Not "See" Demonstration Images

The most important finding for our use case comes from Santos et al. (2025),
"What do vision-language models see in the context?":

> "Evaluated models do not 'see' images in the context; instead, their ICL
> ability is predominantly based on the textual modality."

Key experimental results:
- When demonstration images were **blacked out**, most models maintained similar
  performance
- When demonstration **text was removed**, performance degraded significantly
- LLaVA v1.5's performance drops to **zero** with 8+ demonstrations
- Only models trained on interleaved image-text data (Idefics2, OpenFlamingo)
  showed improvement with more demonstrations

**Implication for SeedLearn:** Providing reference images of "serrate margin"
alongside the query image is unlikely to help the model discriminate serrate
from dentate, because the model will primarily attend to the textual label
("serrate") rather than the visual features in the reference image.

**Source:** [Santos et al. 2025](https://arxiv.org/abs/2510.24331) (arXiv preprint,
7 models evaluated across 4 architectures, 3 benchmarks)

**Evidence quality:** Strong. Systematic study with attention analysis confirming
the mechanism. However, tested models are smaller (4B-9B) than Qwen3-VL-32B.
Larger models with interleaved pretraining *may* perform better, but this is
unverified.

### 1.4 VRAM and Context Length Implications

For Qwen-family VLMs, image tokenization follows:
- **28x28 pixels = 1 token** (Qwen2.5-VL); **32x32 pixels = 1 token** (Qwen3-VL)
- Range: 4 to 16,384 tokens per image
- Configurable via `min_pixels` and `max_pixels` parameters

Practical VRAM estimates for few-shot reference images:

| Configuration | Tokens per image | 4 reference images | Total added context |
|---|---|---|---|
| Low-res (256 tokens) | ~256 | ~1,024 | ~1,024 + text |
| Medium-res (640 tokens) | ~640 | ~2,560 | ~2,560 + text |
| High-res (1,280 tokens) | ~1,280 | ~5,120 | ~5,120 + text |
| Max-res (16,384 tokens) | ~16,384 | ~65,536 | Likely exceeds practical limits |

With Qwen3-VL's 262K context window, 4 medium-resolution reference images add
~2,560 tokens (~1% of context). This is feasible but adds inference latency
proportional to the additional KV-cache memory.

**Recommendation:** If testing few-shot images, use `min_pixels=256*28*28,
max_pixels=640*28*28` for reference images to minimize VRAM overhead while
maintaining enough detail for trait discrimination.

---

## 2. Prompt Engineering for Structured Trait Extraction

### 2.1 Botanical Definitions in Prompts

**Current state:** The SeedLearn prompts list trait options (e.g., "serrate,
dentate, etc.") but provide no definitions. The model must rely on its
pretraining knowledge of botanical terminology, which is unreliable for
fine-grained distinctions.

**Evidence:** Thielen et al. (2024) tested LLM-based botanical trait extraction
and found:
- Binary traits (2 values): >95% F1
- Moderate complexity (12 values): 70-80% F1
- High complexity with overlapping values (e.g., leaf apex, 7 values): **54% F1**

The authors specifically noted that "enhanced performance [is] possible through
descriptions of traits and their values" but did not implement this optimization.

**Proposed enhancement:** Add formal botanical definitions for commonly confused
trait values directly in the prompt. Example:

```
Leaf margin definitions:
- entire: smooth, without any teeth or lobes
- serrate: teeth point toward the leaf apex, like a saw blade
- dentate: teeth point outward, perpendicular to the margin
- crenate: teeth are rounded, scallop-shaped
- serrulate: finely serrate (teeth < 1mm)
- denticulate: finely dentate (teeth < 1mm)
```

**Expected impact:** +10-15% accuracy on traits with >4 possible values (margin,
shape, apex, base). Based on the Thielen et al. finding that trait complexity
directly correlates with error rate, and the CCAS work (below) showing that
disambiguating definitions improve classification.

**Evidence quality:** Moderate. Thielen et al. (2024) is a peer-reviewed study
in a botanical journal using Mistral-medium on text extraction. The transfer to
visual extraction is plausible but unverified. The principle that explicit
definitions reduce ambiguity is well-established in LLM prompt engineering.

**Source:** [Thielen et al. 2024](https://arxiv.org/abs/2409.17179)
(peer-reviewed, PMC12188617)

### 2.2 Taxonomic Decision Trees in Prompts

**Concept:** Replace flat lists of trait options with branching decision logic
that mirrors how botanists actually discriminate traits.

**Current prompt (leaf margin):**
```
Leaf margin (entire / toothed) AND if toothed (dentate, serrate, etc.): [BLANK]
```

**Proposed decision-tree prompt:**
```
Leaf margin — follow this decision path:
  1. Is the margin smooth with no projections? → "entire"
  2. If projections present:
     a. Are projections rounded? → "crenate"
     b. Are projections pointed?
        i.  Do teeth point toward the leaf tip (forward-angled)? → "serrate"
        ii. Do teeth point straight outward (perpendicular)? → "dentate"
     c. Are projections very fine (< 1mm)?
        → Use "serrulate" (forward) or "denticulate" (outward)
  3. If uncertain between options, report: "toothed (type unclear)"
```

**Evidence:** The GPTree framework (2024) demonstrated that tree-structured
prompting with LLMs eliminates the need for feature engineering and prompt
chaining, using dynamic splitting to improve classification. The CCAS approach
(2025) showed that explicitly modeling confusion between similar classes improved
detection AP by 112% in fine-grained scenarios. The SeedLearn SYS1 prompt
already uses a simple decision tree for leaf arrangement (alternate/opposite/
whorled), which could be extended to all ambiguous traits.

**Expected impact:** +15-20% on commonly confused trait pairs (serrate/dentate,
acute/acuminate, cuneate/attenuate, elliptic/obovate).

**Evidence quality:** Mixed. GPTree is an arxiv preprint (2024). CCAS is arxiv
(2025) but tested on object detection, not trait extraction. The SeedLearn
SYS1 prompt's existing decision tree for leaf arrangement is an internal
precedent. The technique is well-grounded in botanical practice (dichotomous
keys have been standard for 200+ years).

**Source:** [GPTree](https://arxiv.org/abs/2411.08257),
[CCAS](https://arxiv.org/abs/2505.09139)

### 2.3 Chain-of-Thought Prompting for Visual Tasks

**State of the art:** Apple's ACL 2025 paper on improving VLM CoT reasoning
showed significant gains through a two-stage strategy: (1) augmenting training
data with GPT-4o-generated rationales, and (2) DPO calibration using
correctness-based reward signals. However, these gains required model
fine-tuning, not just prompt changes.

For prompt-only CoT:
- Zero-shot CoT ("Let's think step by step") has shown mixed results for VLM
  visual tasks
- The ARGUS framework (CVPR 2025) proposed grounded CoT that links reasoning
  steps to image regions, but requires model modifications
- Medical VQA research (2025) showed zero-shot CoT "significantly improves both
  reasoning transparency and performance"

**Current SeedLearn approach:** SYS1 already requires "brief parenthetical
justifications" for each trait, which is a lightweight CoT variant.

**Recommendation:** The current SYS1 justification approach is likely
near-optimal for inference-time CoT. Full step-by-step reasoning chains would
increase token output 3-5x without proven accuracy gains for structured
extraction tasks. If Qwen3-VL supports a "thinking" mode (like Qwen3's
`enable_thinking`), this could be tested as it adds reasoning tokens that are
separate from the output.

**Expected impact:** Marginal (+2-5%) over current SYS1 justifications.

**Evidence quality:** The Apple paper (ACL 2025, peer-reviewed) shows gains but
requires fine-tuning. Prompt-only CoT evidence is weaker and mostly from
medical/general VQA domains, not fine-grained botanical extraction.

**Source:** [Apple CoT VLM](https://aclanthology.org/2025.acl-long.82/),
[ARGUS CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Man_Argus_Vision-Centric_Reasoning_with_Grounded_Chain-of-Thought_CVPR_2025_paper.pdf)

### 2.4 JSON Schema-Constrained Decoding

**Current state:** The SeedLearn JSON prompt (`PromptStyle.JSON`) requests JSON
output but does not enforce it at the decoding level. The model may produce
invalid JSON, missing fields, or extra prose.

**vLLM capability:** As of vLLM 0.8.5+, structured output is a core feature
supporting `guided_json` (JSON schema), `guided_choice` (enum values), and
`guided_regex`. The decoding engine applies logit masks to prevent invalid
tokens, guaranteeing schema-valid output from the first token.

**Implementation via OpenAI API:**
```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B",
    messages=messages,
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "leaf_arrangement": {
                    "type": "object",
                    "properties": {
                        "relative_position": {
                            "type": "string",
                            "enum": ["alternate", "opposite", "whorled", "unclear"]
                        },
                        ...
                    }
                }
            },
            "required": [...]
        }
    }
)
```

**Benefits:**
1. **Eliminates parse failures** — every response is valid JSON
2. **Constrains trait values** — enum fields prevent invented values
3. **May improve accuracy** — constraining the output space can focus the model's
   probability mass on valid options (observed in health IE tasks)
4. **Minimal overhead** — vLLM's xgrammar backend adds negligible latency

**Expected impact:** Eliminates 100% of format errors. Potential +5-10% accuracy
on enum-constrained traits by preventing hallucinated values.

**Evidence quality:** Strong for format reliability. The Red Hat (2025) and
vLLM blog (2025) document production use. Accuracy improvement from
constrained decoding is demonstrated in health information extraction
(LLMStructBench, 2025) but not specifically for VLM visual tasks.

**Source:** [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/),
[Red Hat 2025](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses),
[vLLM Structured Decoding Intro](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)

### 2.5 Prompt Decomposition: One Trait at a Time vs. All 24

**Evidence:** Thielen et al. (2024) directly compared batch vs. single-trait
prompting for botanical trait extraction:

| Approach | F1 Score | Coverage | Token Usage |
|---|---|---|---|
| All traits at once | 0.7643 | 55.0% | 150K tokens |
| Single trait per query | 0.7708 | 57.9% | 1.67M tokens |

The accuracy difference is **+0.7% F1** for single-trait prompting, at **11x
the token cost**.

**For SeedLearn:** Running 24 separate VLM inference calls (each with image
encoding) would be prohibitively expensive:
- ~24x latency increase (each call processes the full image)
- ~24x VRAM throughput reduction
- Marginal accuracy gain does not justify the cost

**Hybrid recommendation:** Decompose only the 3-5 most error-prone traits into
separate follow-up queries. For example, if leaf margin and leaf apex have the
highest error rates, run a targeted follow-up prompt:

```
Given this seedling image, focus specifically on the leaf margin.
Follow this decision path: [decision tree]
What is the leaf margin type?
```

**Expected impact:** +1-3% overall, +5-10% on targeted traits, at ~1.2x cost.

**Evidence quality:** Strong. Thielen et al. is peer-reviewed and directly
tests the comparison. The token cost numbers are from their actual experiments.

**Source:** [Thielen et al. 2024](https://arxiv.org/abs/2409.17179)

---

## 3. Visual Reference Techniques

### 3.1 Image Grid / Composite Montage

**Concept:** Instead of multi-image prompting (which VLMs may not leverage
visually), create a single composite image containing a grid of labeled
reference examples alongside the query image.

**Precedent:** The IG-VLM framework (2024) demonstrated that arranging multiple
video frames into a single image grid enables VLMs to process temporal
information without video-specific training, outperforming existing methods on
9 of 10 video QA benchmarks.

**Proposed approach for SeedLearn:**
```
┌─────────────────────────────────────┐
│  REFERENCE: Leaf Margin Types       │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│ │ ENT │ │ SER │ │ DEN │ │ CRE │   │
│ └─────┘ └─────┘ └─────┘ └─────┘   │
│ entire   serrate  dentate  crenate │
├─────────────────────────────────────┤
│  QUERY IMAGE                        │
│ ┌───────────────────────────┐       │
│ │     [specimen photo]      │       │
│ └───────────────────────────┘       │
└─────────────────────────────────────┘
```

**Advantages over multi-image few-shot:**
- Single image input (no `--limit-mm-per-prompt` configuration)
- VLMs process the entire composite as one visual field, so spatial
  relationships between reference and query are naturally encoded
- The model can visually compare the query leaf edge against labeled examples
  in the same visual field

**Disadvantages:**
- Reduces effective resolution of both reference and query images
- Requires preprocessing pipeline to generate composites
- Unknown whether the model would actually compare sub-images

**Expected impact:** Uncertain. No direct evidence for botanical trait
discrimination. The IG-VLM results are encouraging but for a different task
(video QA). This would require empirical testing.

**Evidence quality:** Weak for our specific use case. IG-VLM (2024, IEEE
Access) is peer-reviewed but for video understanding. The transfer to trait
reference grids is speculative.

**Source:** [IG-VLM](https://arxiv.org/abs/2403.18406)

### 3.2 Annotated Botanical Illustrations

**Concept:** Include high-quality botanical line drawings (e.g., from Flora
Neotropica or Tropicos) as reference images, with labeled diagnostic features.

**Rationale:** Botanical illustrations are specifically designed to highlight
diagnostic morphological features with less visual noise than photographs.
Line drawings of leaf margin types, venation patterns, and apex shapes are
standard tools in botanical education.

**Implementation:** Curate a small library of reference illustrations:
- Leaf margin types (entire, serrate, dentate, crenate, etc.)
- Leaf apex types (acute, acuminate, obtuse, rounded, emarginate)
- Leaf base types (cuneate, rounded, cordate, attenuate, oblique)
- Leaf shape types (elliptic, obovate, ovate, lanceolate, oblanceolate)
- Venation patterns (pinnate, palmate, parallel)

These could be provided as a composite image grid (Section 3.1) or as
individual reference images.

**Caveat:** Given the Santos et al. (2025) finding that VLMs ignore visual
demonstrations, botanical illustrations may be most effective when their
*content is described textually in the prompt* rather than shown as images.
The illustrations themselves might serve more as a prompt-engineering aid for
the *human* designing the text definitions.

**Expected impact:** Uncertain as visual input. High value as source material
for writing the textual definitions recommended in Section 2.1.

### 3.3 Contrastive Prompting for Confusable Traits

**Concept:** The CCAS framework (2025) demonstrated that prompts optimized to
*maximize distance from confounding classes* dramatically outperform generic
prompts. For the serrate/dentate confusion:

Instead of:
> "Leaf margin (entire / toothed) AND if toothed (dentate, serrate, etc.)"

Use contrastive definitions:
> "Serrate means teeth angle FORWARD toward the tip (like a saw blade cutting
> forward). Dentate means teeth point OUTWARD at right angles to the margin
> (like a castle battlement). If teeth angle forward: serrate. If teeth point
> straight out: dentate."

This explicitly models the confusion axis between the two classes.

**Evidence:** CCAS (2025) improved detection AP by 112% (goggles vs. glasses)
by optimizing prompts against confounding classes. The waste classification
study (2025) improved zero-shot accuracy from 82.7% to 90.5% with prompt
engineering including explicit category descriptions.

**Expected impact:** +10-20% on specific confusable trait pairs.

**Evidence quality:** Moderate. CCAS is an arxiv preprint testing detection,
not classification. The waste study is peer-reviewed (Waste Management, 2025).
Neither tests botanical traits specifically, but the principle of contrastive
disambiguation is well-established.

**Source:** [CCAS 2025](https://arxiv.org/abs/2505.09139),
[Waste Classification 2025](https://www.sciencedirect.com/science/article/pii/S0956053X25003502)

---

## 4. Technique Comparison Table

| Technique | Expected Accuracy Gain | Implementation Effort | VRAM/Latency Cost | Evidence Quality | Priority |
|---|---|---|---|---|---|
| **Botanical definitions in prompt** | +10-15% on ambiguous traits | Low (text edits to prompts.py) | None | Moderate (peer-reviewed botanical study + general PE literature) | **1 (highest)** |
| **Decision-tree prompting** | +15-20% on confusable pairs | Medium (rewrite trait sections) | None | Mixed (arxiv + 200yr botanical precedent) | **2** |
| **JSON schema-constrained decoding** | Eliminates parse errors; +5-10% on enum traits | Low (vLLM config change) | Negligible | Strong (vLLM production docs) | **3** |
| **Reference image grid (composite)** | Uncertain (0-15%) | Medium (image preprocessing) | +256-1280 tokens | Weak (different-domain precedent) | **4** |
| **Targeted trait decomposition** | +5-10% on 3-5 worst traits | Medium (pipeline changes) | +20-50% latency for targeted traits | Strong (peer-reviewed direct comparison) | **5** |
| **Few-shot image demonstrations** | Likely minimal (0-5%) | Medium (image selection + prompting) | +1K-5K tokens per example | Weak (evidence shows VLMs ignore visual demos) | **6** |
| **Full chain-of-thought** | +2-5% over current SYS1 | Low-Medium | +200-500% output tokens | Mixed (gains require fine-tuning, not prompt-only) | **7** |
| **Contrastive prompt optimization** | +10-20% on specific pairs | High (requires labeled eval set + iteration) | None | Moderate (arxiv, tested on detection) | **3** (combine with decision trees) |

---

## 5. Specific Recommendations for 24-Trait Botanical Extraction

### 5.1 Immediate Implementation (Week 1)

**A. Add botanical definitions to the JSON prompt:**

Modify `JSON_SYSTEM_PROMPT` in `src/seedlearn/components/analyzers/prompts.py`
to include definitions for all multi-valued traits. Focus on the traits most
prone to confusion:

- **Leaf margin** (serrate/dentate/crenate): tooth direction and shape
- **Leaf shape** (elliptic/obovate/ovate/lanceolate): width-to-length ratio
  and widest point location
- **Leaf apex** (acute/acuminate/obtuse): angle and taper characteristics
- **Leaf base** (cuneate/rounded/cordate/attenuate): symmetry and angle
- **Venation** (pinnate/palmate/parallel): secondary vein origin pattern

**B. Enable vLLM structured output:**

Add `guided_json` parameter to the VLM client when using `PromptStyle.JSON`.
Define strict enum values for all categorical traits. This requires changes to
`src/seedlearn/pipeline/vlm_client.py` to pass the schema in `extra_body`.

### 5.2 Short-Term Implementation (Weeks 2-3)

**C. Decision-tree prompts for confusable traits:**

Create a new `PromptStyle.SYS5` that adds decision-tree logic for the 5 most
error-prone traits. Evaluate against SYS1 on a held-out validation set.

**D. Test composite reference image grid:**

Create reference montage images for leaf margin types, apex types, and shape
types. Test whether including these as a single composite image alongside the
specimen photo improves discrimination. Compare:
- Baseline (SYS1, no reference)
- Text definitions only (Section 5.1A)
- Text definitions + reference grid
- Reference grid without text definitions

### 5.3 Medium-Term Experiments (Month 2)

**E. Targeted decomposition for worst-performing traits:**

After establishing baseline accuracy per trait, identify the 3-5 traits with
lowest accuracy. Create targeted follow-up prompts with decision trees for
these specific traits. Run as a second inference pass only when the first pass
returns "unclear" or when confidence is low.

**F. Evaluate Qwen3-VL thinking mode:**

If Qwen3-VL supports `enable_thinking=true` (as Qwen3 text models do), test
whether the internal reasoning tokens improve trait accuracy. This adds
latency but may improve discrimination on complex traits without changing the
prompt.

### 5.4 Traits Requiring Special Attention

Based on the trait list and botanical domain knowledge, these traits are most
likely to benefit from improved prompting:

| Trait | Confusion Risk | Recommended Technique |
|---|---|---|
| Leaf margin (serrate/dentate) | Very high | Decision tree + definitions |
| Leaf shape (elliptic/obovate) | High | Definition with measurement ratios |
| Leaf apex (acute/acuminate) | High | Decision tree (angle + taper) |
| Leaf base (cuneate/attenuate) | Moderate | Definition with angle criteria |
| Leaf arrangement (alternate/opposite) | Moderate | Already has decision tree in SYS1 |
| Stipules (present/absent) | High (hard to see) | Note "check leaf base and nodes" |
| Venation (pinnate/palmate) | Moderate | Definition + vein origin point |

---

## 6. References

### Peer-Reviewed Papers

1. Thielen et al. (2024). "Fully automatic extraction of morphological traits
   from the web: Utopia or reality?" *Methods in Ecology and Evolution* / PMC.
   [HTML](https://arxiv.org/html/2409.17179v1) |
   [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12188617/)
   — Direct comparison of single vs. batch trait extraction; F1 scores by trait
   complexity.

2. Apple ML Research (2025). "Improve Vision Language Model Chain-of-thought
   Reasoning." *ACL 2025*.
   [Paper](https://aclanthology.org/2025.acl-long.82/) |
   [Blog](https://machinelearning.apple.com/research/chain-of-thought)
   — Two-stage CoT improvement via data augmentation + DPO. Requires fine-tuning.

3. Man et al. (2025). "ARGUS: Vision-Centric Reasoning with Grounded
   Chain-of-Thought." *CVPR 2025*.
   [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Man_Argus_Vision-Centric_Reasoning_with_Grounded_Chain-of-Thought_CVPR_2025_paper.pdf)
   — Grounded CoT linking reasoning to image regions.

4. Lin et al. (2024). "VILA: On Pre-training for Visual Language Models."
   *CVPR 2024*.
   [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VILA_On_Pre-training_for_Visual_Language_Models_CVPR_2024_paper.pdf)
   — Interleaved pretraining essential for few-shot ICL capability.

5. Mitra et al. (2025). "Enhancing Few-Shot Vision-Language Classification
   with Large Multimodal Model Features." *ICCV 2025*.
   [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Mitra_Enhancing_Few-Shot_Vision-Language_Classification_with_Large_Multimodal_Model_Features_ICCV_2025_paper.pdf)
   — SAVs achieve SOTA few-shot classification with 20 examples per label.

6. CascadeVLM (2024). "Enhancing Fine-Grained Image Classifications via
   Cascaded Vision Language Models." *Findings of EMNLP 2024*.
   [Paper](https://aclanthology.org/2024.findings-emnlp.102/)
   — 92% accuracy on Stanford Cars via cascaded CLIP + LVLM.

7. Waste classification (2025). "Enhancing waste recognition with
   vision-language models: A prompt engineering approach."
   *Waste Management*.
   [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0956053X25003502)
   — Zero-shot VLM accuracy 82.7% → 90.5% with prompt engineering.

### Preprints and Technical Reports

8. Santos et al. (2025). "What do vision-language models see in the context?
   Investigating multimodal in-context learning." *arXiv 2510.24331*.
   [Paper](https://arxiv.org/abs/2510.24331)
   — **Critical finding:** VLMs ignore visual demonstrations, attend to text only.

9. VLM-ICL for Visual Inspection (2025). *arXiv 2502.09057*.
   [Paper](https://arxiv.org/abs/2502.09057)
   — One-shot visual ICL achieves F1=0.95 on MVTec AD using ViP-LLaVA.

10. CCAS (2025). "Beyond General Prompts: Automated Prompt Refinement using
    Contrastive Class Alignment Scores." *arXiv 2505.09139*.
    [Paper](https://arxiv.org/abs/2505.09139)
    — 112% AP improvement via contrastive prompt optimization.

11. GPTree (2024). "Towards Explainable Decision-Making via LLM-powered
    Decision Trees." *arXiv 2411.08257*.
    [Paper](https://arxiv.org/abs/2411.08257)
    — Tree-structured prompting for LLM classification.

12. IG-VLM (2024). "An Image Grid Can Be Worth a Video." *IEEE Access*.
    [Paper](https://arxiv.org/abs/2403.18406)
    — Image grid composites outperform multi-image input on 9/10 VQA benchmarks.

13. VL-ICL Bench (2025). *ICLR 2025*.
    [PDF](https://openreview.net/pdf?id=cpGPPLLYYx)
    — Comprehensive benchmark for vision-language in-context learning.

### Technical Documentation

14. vLLM Structured Outputs.
    [Docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
    — JSON schema, regex, grammar-constrained decoding.

15. vLLM Qwen3-VL Recipe.
    [Docs](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
    — Multi-image setup, context length, configuration.

16. NVIDIA VLM Prompt Engineering Guide (2025).
    [Blog](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
    — Practical multi-image prompting, structured output, specificity guidelines.

17. Qwen3-VL GitHub Repository.
    [GitHub](https://github.com/QwenLM/Qwen3-VL)
    — Model documentation, multi-image support, min/max pixels configuration.
