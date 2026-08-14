# Segmentation and Activation Maps for Seedling Isolation

Research report on techniques for isolating seedlings from background in botanical
photographs to improve downstream VLM trait extraction.

Last verified: 2026-03-19

---

## Executive Summary

The most promising approach for the seedling pipeline is a **two-stage detect-then-segment
strategy**: use Grounding DINO (or SAM 3's built-in text prompting) to locate the seedling,
then pass the cropped/masked image to the VLM for trait extraction. This avoids the need for
any training data and can be integrated as a preprocessing step before Stage 1 (morphology).

**Recommended first experiments, in priority order:**

1. **BiRefNet via rembg** -- simplest integration, strong general background removal, ~70ms
   per image on H200, 3.5 GB VRAM (FP16). Good baseline; may struggle with multiple plants
   in frame.
2. **Grounded SAM 2** (Grounding DINO + SAM 2) -- text-prompted detection ("seedling",
   "young plant") generates bounding boxes, then SAM 2 produces precise masks. More control
   than rembg, handles multi-object scenes. ~200-400ms per image, ~8-12 GB VRAM.
3. **SAM 3 with text prompts** -- newest option (Nov 2025), finds all instances of a concept
   exhaustively. 30ms on H200 but requires 96 GB VRAM (fits on H200 140GB). Less field-tested
   than Grounded SAM 2.
4. **GradCAM on BioCLIP 2 ViT-L/14** -- zero additional model cost since the encoder is
   already loaded. Produces attention heatmaps that can be thresholded into soft masks or
   used to generate bounding box crops. Quality is lower than dedicated segmentation but
   essentially free.

The VLM itself (Qwen3-VL) can then receive either: (a) a tightly cropped image, (b) the
original image with a visual bounding box overlay, or (c) a masked image with background
replaced by a neutral color. Cropping to the detected region is the simplest and most
effective approach based on current evidence.

---

## Technique Comparison Table

| Technique | Segmentation Quality | Speed (per image) | VRAM | Implementation Effort | Evidence Quality | Best For |
|---|---|---|---|---|---|---|
| **BiRefNet (rembg)** | High (IoU 0.87, Dice 0.92) | ~70ms FP16 | 3.5 GB | Low (pip install, 5 lines) | Production benchmarks (Cloudflare) | Single-subject background removal |
| **Grounded SAM 2** | Very high (IoU 0.94+) | 200-400ms | 8-12 GB | Medium (two models to load) | Peer-reviewed (ECCV, agriculture papers) | Text-prompted multi-object segmentation |
| **SAM 3** | Very high | ~30ms | ~96 GB | Medium (HF access request) | Meta benchmarks, ICLR 2026 submission | Exhaustive concept-based segmentation |
| **SAM 2 (point/box prompt)** | Very high (IoU 0.89-0.93) | ~25-33ms | 6-8 GB | Medium (needs prompt source) | Extensively validated | When you have box/point prompts |
| **GradCAM on BioCLIP 2** | Moderate (coarse heatmap) | <5ms | 0 GB extra | Low (pytorch-grad-cam) | Well-established method | Quick attention visualization, free |
| **CLIP Surgery** | Moderate | ~10ms | 0 GB extra | Medium | ArXiv 2023 | Zero-shot localization from text |
| **CRG (Contrastive Region Guidance)** | N/A (guides VLM) | ~2x inference | 0 GB extra | Medium | ECCV 2024 | Training-free VLM region focus |
| **Qwen3-VL self-grounding** | Moderate (box output) | ~500ms | Already loaded | Low (prompt engineering) | Community reports | Ask VLM to locate before describing |
| **U-Net / PlantSeg** | Variable | ~50-100ms | 2-4 GB | High (training required) | Domain-specific papers | When you have labeled training data |
| **MODNet** | High for portraits | ~100ms | 2-4 GB | Low | Peer-reviewed | Portrait/single-subject matting |
| **GrabCut + bbox** | Moderate | ~200ms (CPU) | 0 GB | Low (OpenCV) | Classical CV, well-understood | Quick prototype, CPU-only |

---

## Detailed Analysis

### 1. Dedicated Segmentation Models

#### 1.1 BiRefNet (via rembg)

**What it is:** Bilateral Reference Network for high-resolution dichotomous image
segmentation. The current state-of-the-art for general background removal.

**Architecture:** Swin-L backbone with a bilateral reference mechanism that passes
information bidirectionally, allowing pixel-level details to be informed by the larger scene
context. This is particularly relevant for botanical images where thin stems, leaf edges, and
fine structures need preservation.

**Performance:**
- IoU 0.87, Dice 0.92 (averaged across DIS5K and human benchmarks)
- Outperforms U2-Net (IoU 0.39-0.89 depending on dataset) and IS-Net (IoU 0.82)
- FP16 inference: 57.7ms on RTX 4090, ~70ms on A100; expect similar or better on H200
- VRAM: 3.5 GB FP16, 4.8 GB FP32 at 1024x1024

**Variants available:**
- `birefnet-general` -- recommended, best general accuracy
- `birefnet-general-lite` -- faster, slightly lower quality
- `BiRefNet_dynamic` -- handles 256x256 to 2304x2304 dynamically (released March 2025)
- `BiRefNet_HR-matting` -- alpha matting at 2048x2048 for soft edges

**Installation:**
```python
# Via rembg (simplest)
# uv pip install rembg[gpu]
from rembg import remove
from PIL import Image
output = remove(Image.open("seedling.jpg"))

# Via transformers (more control)
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained(
    "zhengpeng7/BiRefNet", trust_remote_code=True
)
```

**Botanical image assessment:** BiRefNet was benchmarked on general images, not specifically
on botanical field photographs with soil, leaf litter, and rulers. The bidirectional reference
mechanism should handle complex plant edges well, but validation on actual seedling images is
needed. Main risk: when multiple plants are present, BiRefNet will try to segment the single
most prominent foreground object, which may not be the target seedling.

**Source credibility:** Cloudflare production benchmarks (engineering blog, reproducible).
BiRefNet paper published in CAAI AIR 2024. Multiple independent evaluations confirm results.

**References:**
- [BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet)
- [Cloudflare Background Removal Evaluation](https://blog.cloudflare.com/background-removal/)
- [rembg GitHub](https://github.com/danielgatis/rembg)

---

#### 1.2 Grounded SAM 2 (Grounding DINO + SAM 2)

**What it is:** A pipeline combining Grounding DINO (open-set text-prompted object detector)
with SAM 2 (segment anything). You provide a text prompt like "seedling" or "young plant" and
get precise segmentation masks.

**How it works:**
1. Grounding DINO takes the image + text prompt -> produces bounding boxes with confidence
2. Bounding boxes are fed as prompts to SAM 2 -> produces pixel-level masks
3. Masks can be used to crop, mask background, or overlay on original image

**Performance:**
- Detection + segmentation quality depends on prompt engineering
- Baby kale study (2025): correlation 0.956 with manual annotations, zero training needed
- SAM 2 alone: IoU 0.89-0.93 on plant datasets when given good box prompts
- Inference: Grounding DINO ~100-200ms + SAM 2 ~25-33ms = ~200-400ms total
- VRAM: ~8-12 GB for both models loaded simultaneously

**Key advantage for this pipeline:** Text prompting means no training data needed. Can
distinguish between "seedling" and "ruler" or "label" in the same image. Can return multiple
objects, allowing selection of the most relevant.

**Limitations:**
- Grounding DINO may not understand "seedling" well since its training data is
  internet-scale, not botanical. May need to test prompts: "plant", "seedling", "young tree",
  "leaf", "green plant"
- Two separate models to load, configure, and maintain
- Slower than single-model approaches

**Validated on botanical images?** Yes, partially. Published studies on baby kale, herbarium
specimens (PlantSAM), and general agricultural datasets. Not specifically on tropical
seedlings in field conditions with complex backgrounds, but the closest validated approach.

**References:**
- [Grounded SAM 2 GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Grounding DINO (ECCV 2024)](https://github.com/IDEA-Research/GroundingDINO)
- [Baby Kale Detection Study (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S2772375525001364)
- [PlantSAM (Applications in Plant Sciences 2025)](https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.70034)

---

#### 1.3 SAM 3 (Segment Anything with Concepts)

**What it is:** Meta's latest segmentation model (November 2025), introducing Promptable
Concept Segmentation (PCS). Given a text phrase like "seedling", it exhaustively finds and
segments all matching instances.

**Key differences from SAM 2:**
- Text prompts are native (no need for Grounding DINO)
- Exhaustive: returns ALL instances of a concept, not just one
- Recognizes 270,000+ visual concepts
- ~30ms per image on H200 (very fast)
- But requires ~96 GB VRAM (fits on H200 140GB, but leaves limited headroom)

**Installation:**
```python
# Via Ultralytics
# uv pip install -U ultralytics
# Download sam3.pt from HuggingFace (requires access request)

from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(conf=0.25, task="segment", mode="predict",
                 model="sam3.pt", half=True, save=True)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("seedling.jpg")
results = predictor(text=["seedling", "young plant", "leaf"])
```

**Assessment:** Most powerful option but least field-tested. ICLR 2026 submission, not yet
peer-reviewed at time of writing. The 96 GB VRAM requirement is significant -- if the VLM
(Qwen3-VL-32B FP8, ~35 GB) is also loaded, total VRAM approaches the H200's 140 GB limit.
Sequential loading (segment first, unload, then run VLM) would work but adds complexity.

**References:**
- [SAM 3 Paper (ArXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [Meta AI Blog](https://ai.meta.com/blog/segment-anything-model-3/)
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/)

---

#### 1.4 Segment Any Plant (SAP)

**What it is:** A plant-specific framework built on SAM 2, designed for plant time-series
phenotyping (BioRxiv, March 2026).

**Performance:** Mean IoU 0.89-0.93, sub-pixel centerline precision. Tested on Arabidopsis,
root growth, sunflower gravitropism, confocal root microscopy.

**Relevance:** Primarily designed for controlled-environment time-series, not field
photographs. The prompting interface and SAM 2 adaptation may offer useful insights, but the
framework is more specialized than what this pipeline needs.

**Reference:**
- [SAP BioRxiv Preprint (March 2026)](https://www.biorxiv.org/content/10.64898/2026.03.11.711099v1)

---

### 2. Activation and Attention Map Techniques

#### 2.1 GradCAM on BioCLIP 2 ViT-L/14

**What it is:** Gradient-weighted Class Activation Mapping applied to the BioCLIP 2 vision
encoder that is already loaded in Stage 2 of the pipeline. Zero additional model cost.

**How it works with ViT:**
The ViT-L/14 produces patch embeddings of shape BATCH x 257 x 1024 (256 patches for 16x16
grid + 1 class token). GradCAM computes gradients of a target class score with respect to
activations at a chosen layer, producing a 16x16 heatmap that can be upscaled to the original
image resolution.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def reshape_transform(tensor, height=16, width=16):
    # ViT-L/14: skip class token, reshape patches to 2D
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Target the normalization layer before the last attention block
target_layers = [model.visual.transformer.resblocks[-1].ln_1]

cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)
grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
```

**Quality assessment:** GradCAM on ViT produces coarse 16x16 heatmaps (for ViT-L/14 at
224x224 input). After bilinear upscaling, the resolution is sufficient to identify the general
region of interest but not fine enough for precise segmentation. The heatmap can be
thresholded to create a rough bounding box or soft mask.

**Integration with pipeline:** Since BioCLIP 2 is already loaded in Stage 2, GradCAM
visualization can be computed with minimal overhead (<5ms). The resulting heatmap could:
- Generate a bounding box for the highest-activation region -> crop for VLM
- Produce a soft mask to weight image regions
- Serve as a diagnostic tool to verify the model attends to the seedling

**Limitations:**
- 16x16 resolution is coarse for precise segmentation
- Requires a target class/text to compute gradients against
- Quality depends on how well BioCLIP 2 was trained to attend to plant features

**Source credibility:** GradCAM is a well-established technique (ICCV 2017, 10,000+ citations).
pytorch-grad-cam is actively maintained with ViT-specific tutorials. CLIP + GradCAM
combination has been demonstrated in multiple papers and notebooks.

**References:**
- [pytorch-grad-cam GitHub](https://github.com/jacobgil/pytorch-grad-cam)
- [ViT Tutorial](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md)
- [CLIP GradCAM Notebook](https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb)
- [SigLIP 2 GradCAM Visualization (2025)](https://blogs.gwu.edu/pless/2025/04/21/siglip-2-gradcam-attention-visualization/)

---

#### 2.2 CLIP Surgery

**What it is:** A method to improve CLIP's explainability by modifying the self-attention
computation to produce cleaner activation maps. Operates on the same ViT architecture as
BioCLIP 2.

**How it works:** CLIP Surgery removes the query-key interaction from self-attention (which
tends to create noisy, class-agnostic patterns) and instead uses feature-feature similarity,
producing sharper localization maps that better highlight the target concept.

**Relevance:** Could be applied to BioCLIP 2's ViT-L/14 to get better localization than
vanilla GradCAM. The resulting maps are still patch-resolution (16x16 for ViT-L/14) but
tend to be sharper and more class-discriminative.

**Limitation:** ArXiv paper (April 2023), no peer review. The technique modifies the forward
pass, so it cannot be applied post-hoc to cached embeddings -- you need the model loaded.

**Reference:**
- [CLIP Surgery (ArXiv 2304.05653)](https://arxiv.org/abs/2304.05653)

---

#### 2.3 CLIP-ES (CLIP Efficient Segmenter)

**What it is:** CVPR 2023 paper that introduces a softmax function into GradCAM for CLIP and
uses a class-aware attention-based affinity (CAA) module on multi-head self-attention to
produce better zero-shot semantic segmentation from CLIP models.

**Relevance:** Bridges the gap between CLIP's image-level understanding and pixel-level
segmentation. Could theoretically produce usable masks from BioCLIP 2 without loading an
additional model.

**Limitation:** Requires adapting the implementation to open-clip / BioCLIP 2 architecture.
Implementation effort is moderate. Quality is below dedicated segmentation models.

**Reference:**
- [CLIP is Also an Efficient Segmenter (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_CLIP_Is_Also_an_Efficient_Segmenter_A_Text-Driven_Approach_for_CVPR_2023_paper.pdf)

---

#### 2.4 Qwen3-VL Attention Visualization

**What it is:** Extracting attention maps from the Qwen3-VL model to see where it "looks"
when extracting traits.

**Current state:** Qwen3-VL uses windowed attention in its vision encoder, which makes
attention visualization complex. The pooling across three dimensions (height, width, temporal)
means mapping attention back to the original image requires careful dimensional reshaping.
Community discussions on GitHub indicate this is possible but not straightforward.

**Practical assessment:** Not recommended as a primary segmentation strategy. The windowed
attention architecture was designed for efficiency, not interpretability. However, it could
serve as a diagnostic tool to understand *after the fact* what the VLM attended to, helping
identify cases where background elements were misidentified.

**References:**
- [Qwen3-VL Attention Visualization Issue](https://github.com/QwenLM/Qwen3-VL/issues/1097)
- [Qwen3-VL Vision Attention Masks Issue](https://github.com/QwenLM/Qwen3-VL/issues/753)

---

### 3. Background Suppression and Preprocessing

#### 3.1 RMBG 2.0 (BRIA AI)

**What it is:** Commercial background removal model built on BiRefNet architecture, trained
on BRIA's proprietary dataset. Available on HuggingFace with a commercial-use license.

**Performance:** Comparable to or slightly better than BiRefNet-general on diverse images.
Handles multiple objects, complex textures, and detailed edges.

**Assessment:** Strong alternative to rembg if licensing permits. Same architectural
advantages as BiRefNet for botanical images.

**Reference:**
- [RMBG 2.0 HuggingFace](https://huggingface.co/briaai/RMBG-2.0)

---

#### 3.2 GrabCut with Automated Bounding Box

**What it is:** Classical CV technique (OpenCV) that iteratively refines a foreground mask
given an initial bounding box. Can be combined with any object detector to automate the box.

**Usage:**
```python
import cv2
import numpy as np

img = cv2.imread("seedling.jpg")
mask = np.zeros(img.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
# rect from detector: (x, y, w, h)
rect = (50, 50, 400, 500)
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
```

**Assessment:** Fast (~200ms CPU), no GPU needed, no model to load. Quality is moderate --
works well when background and foreground have distinct color distributions (green plant on
brown soil). Struggles with green-on-green (seedling in leaf litter). Good for quick
prototyping but not production quality.

---

#### 3.3 Alpha Matting (BiRefNet_HR-matting)

**What it is:** Produces soft alpha mattes rather than hard binary masks. Preserves
transparency at edges, useful for fine structures like leaf margins and trichomes.

**Assessment:** Overkill for VLM preprocessing (the VLM does not need transparent edges). More
relevant if creating training datasets or visual outputs. BiRefNet_HR-matting variant handles
this at 2048x2048 resolution.

---

### 4. VLM Region-of-Interest Techniques

#### 4.1 Crop-Then-Infer (Recommended)

**What it is:** The simplest approach -- detect the seedling region, crop the image to that
region (with configurable padding), and pass only the cropped image to the VLM.

**Evidence:**
- R-VLM (ACL 2025): 13% improvement in grounding accuracy when using zoomed-in crops
- Multiple studies confirm VLMs perform better on cropped regions for fine-grained tasks
- Reduces visual noise from rulers, labels, other vegetation
- Qwen2.5-VL specifically benefits from higher effective resolution on the target region

**Implementation:**
```python
def crop_with_padding(image, bbox, padding_ratio=0.1):
    """Crop image to bbox with padding."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * padding_ratio), int(h * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)
    return image.crop((x1, y1, x2, y2))
```

**Pipeline integration:** Insert as a preprocessing step between image loading and Stage 1.
Both the cropped image (for VLM) and the original image (for BioCLIP embedding, which is
already trained on full images) can be passed through the pipeline.

---

#### 4.2 Visual Bounding Box Overlay

**What it is:** Draw a colored bounding box or circle on the original image around the
seedling, then pass the annotated image to the VLM with a text prompt like "Describe the
morphological traits of the plant inside the red bounding box."

**Evidence:** Mixed. Set-of-Marks (SoM) prompting works well with GPT-4V but does not
transfer reliably to open-source VLMs. Qwen3-VL may understand drawn annotations to some
degree, but this has not been rigorously tested.

**Assessment:** Easy to implement, worth testing, but crop-then-infer is more robust.

---

#### 4.3 Contrastive Region Guidance (CRG)

**What it is:** Training-free method (ECCV 2024) that guides a VLM to focus on specific
regions by contrasting model outputs with and without the region visible. Works by masking
regions and measuring the change in output distribution.

**How it works:**
1. Run the VLM on the original image -> get logit distribution
2. Black out the target region -> run again -> get second distribution
3. Contrast the distributions to amplify information from the target region

**Performance:** Up to 11.1% accuracy improvement on ViP-Bench tasks. The implementation
exists for LLaVA-based models; adapting to Qwen3-VL via vLLM would require intercepting
logits.

**Assessment:** Elegant but doubles inference time (two VLM passes per image). For a pipeline
already running a 32B VLM, this adds significant cost. More suitable as a research experiment
than a production approach.

**Reference:**
- [CRG Paper (ECCV 2024)](https://arxiv.org/abs/2403.02325)
- [CRG GitHub](https://github.com/meetdavidwan/crg)

---

#### 4.4 Qwen3-VL Self-Grounding

**What it is:** Ask the VLM itself to first locate the seedling before describing it. Qwen3-VL
(inheriting from Qwen2.5-VL) supports bounding box output in absolute coordinates.

**Two-pass approach:**
1. First prompt: "Identify the main seedling in this image. Return a bounding box in the
   format [x1, y1, x2, y2]."
2. Crop to the returned bounding box
3. Second prompt: "Describe the morphological traits of this seedling." (on cropped image)

**Assessment:** No additional model needed, but doubles VLM inference time. Quality of
bounding box depends on Qwen3-VL's grounding ability for "seedling" -- untested on this
specific domain. If the VLM is already confused by the background, its bounding box may also
be unreliable.

---

### 5. Specialized Plant Segmentation

#### 5.1 PlantSAM

**What it is:** YOLOv10 for plant detection + SAM 2 for segmentation, specifically designed
for herbarium specimens. Achieved IoU 0.94, Dice 0.97.

**Relevance:** Herbarium specimens are pressed, dried plants on white backgrounds -- very
different from field photographs of living seedlings. The architecture (detection + SAM) is
the same as Grounded SAM but with a plant-specific detector.

**Assessment:** The YOLOv10 component would need retraining for seedling images. The SAM 2
component is directly reusable.

**Reference:**
- [PlantSAM (ArXiv 2507.16506)](https://arxiv.org/abs/2507.16506)

---

#### 5.2 EMSAM (Enhanced Multi-Scale SAM)

**What it is:** SAM variant enhanced with multi-scale feature extraction for leaf disease
segmentation. Published in Frontiers in Plant Science (2025).

**Relevance:** Addresses plant-specific segmentation challenges but focused on disease
lesions, not whole-plant extraction from backgrounds.

**Reference:**
- [EMSAM (Frontiers in Plant Science 2025)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1564079/full)

---

## Integration Recommendations for the Seedling Pipeline

### Proposed Architecture: Stage 0 (Preprocessing)

Insert a new preprocessing stage before the existing Stage 1 (Morphology):

```
Stage 0: Seedling Detection & Segmentation (NEW)
    Input:  image_paths from context
    Output: cropped_image_paths, segmentation_masks, bounding_boxes

Stage 1: VLM Morphology (existing, receives cropped images)
Stage 2: BioCLIP Classification (existing, receives original images*)
Stage 3: RAG Trait Retrieval (existing, unchanged)
Stage 4: Evidence Synthesis (existing, unchanged)
Stage 5: LLM Reasoning (existing, unchanged)
```

*Note: BioCLIP 2 embeddings should likely use the original (uncropped) images since the
model was trained on full images and embeddings may lose discriminative information if
cropped. This should be tested empirically.

### Implementation Plan

**Phase 1 -- Baseline (1-2 days):**
- Install rembg with BiRefNet backend
- Add background removal as an optional preprocessing step
- Compare VLM trait extraction accuracy on original vs. background-removed images
- Measure: trait extraction agreement with human annotations, false positive rate for
  background-derived traits

**Phase 2 -- Text-Prompted Detection (2-3 days):**
- Set up Grounded SAM 2 (Grounding DINO + SAM 2)
- Test text prompts: "seedling", "young plant", "plant", "leaf"
- Evaluate: detection accuracy on a sample of 50-100 images with manual bounding boxes
- Compare cropped VLM results vs. Phase 1

**Phase 3 -- BioCLIP Attention Maps (1 day):**
- Implement GradCAM on the BioCLIP 2 ViT-L/14 already loaded in Stage 2
- Generate heatmaps for a sample of images
- Evaluate whether heatmaps correlate with seedling location
- If yes, use as a lightweight alternative or fallback when segmentation models are unavailable

**Phase 4 -- Pipeline Integration (2-3 days):**
- Add `SegmentationConfig` to `pipeline/config.py` with method selection
- Implement `SegmentationStage` in `pipeline/stages/segmentation.py`
- Wire into pipeline runner with optional enable/disable
- Add unit tests

### VRAM Budget on H200 (140 GB)

| Component | VRAM | Notes |
|---|---|---|
| Qwen3-VL-32B FP8 (vLLM) | ~35 GB | Stage 1 |
| BioCLIP 2 ViT-L/14 | ~1.5 GB | Stage 2 |
| BiRefNet FP16 | ~3.5 GB | Stage 0 option A |
| Grounded SAM 2 | ~8-12 GB | Stage 0 option B |
| SAM 3 | ~96 GB | Stage 0 option C (cannot coexist with VLM) |
| **Total (options A or B)** | **~40-50 GB** | Comfortable fit |

Options A (BiRefNet) and B (Grounded SAM 2) can coexist with all other pipeline models.
Option C (SAM 3) requires sequential loading -- segment all images first, unload, then run
VLM -- which may be acceptable for batch processing.

### Config Example

```yaml
segmentation:
  enabled: true
  method: "grounded_sam2"  # or "birefnet", "sam3", "gradcam", "none"
  text_prompt: "seedling"
  confidence_threshold: 0.3
  padding_ratio: 0.1
  apply_to_vlm: true       # crop for Stage 1
  apply_to_bioclip: false   # keep original for Stage 2
```

---

## Summary of Source Credibility

| Source | Type | Credibility |
|---|---|---|
| BiRefNet (CAAI AIR 2024) | Peer-reviewed | High |
| Cloudflare model evaluation | Production benchmark | High |
| Grounded SAM 2 (ECCV 2024) | Peer-reviewed | High |
| SAM 3 (ICLR 2026 submission) | Under review | Medium-high |
| SAP (BioRxiv March 2026) | Preprint | Medium |
| Baby kale Grounding DINO study | Peer-reviewed (Elsevier) | High |
| PlantSAM (APS 2025) | Peer-reviewed | High |
| pytorch-grad-cam | Open-source, 9k+ stars | High (well-maintained) |
| CRG (ECCV 2024) | Peer-reviewed | High |
| CLIP Surgery (ArXiv 2023) | Preprint | Medium |
| Qwen3-VL attention discussions | GitHub issues | Low (community anecdotes) |

**Critical gap:** None of these techniques have been specifically validated on tropical
seedling field photographs with the exact types of backgrounds in this pipeline's images
(soil, leaf litter, rulers, labels, mixed vegetation). Phase 1-2 experiments above are
designed to empirically validate on the actual data.

---

## References

1. [BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet) -- Bilateral Reference Network for high-resolution dichotomous image segmentation
2. [rembg GitHub](https://github.com/danielgatis/rembg) -- Background removal tool with BiRefNet backend
3. [Cloudflare Background Removal Evaluation](https://blog.cloudflare.com/background-removal/) -- Production benchmarking of segmentation models
4. [Grounded SAM 2 GitHub](https://github.com/IDEA-Research/Grounded-SAM-2) -- Grounding DINO + SAM 2 pipeline
5. [Grounding DINO GitHub (ECCV 2024)](https://github.com/IDEA-Research/GroundingDINO) -- Open-set object detection
6. [SAM 3 (ArXiv 2511.16719)](https://arxiv.org/abs/2511.16719) -- Segment Anything with Concepts
7. [SAM 3 Ultralytics Docs](https://docs.ultralytics.com/models/sam-3/) -- SAM 3 usage guide with model specs
8. [SAM 3 GitHub](https://github.com/facebookresearch/sam3) -- Official SAM 3 repository
9. [Segment Any Plant (BioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.03.11.711099v1) -- Plant-specific SAM 2 framework
10. [PlantSAM (APS 2025)](https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.70034) -- Herbarium specimen segmentation
11. [EMSAM (Frontiers 2025)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1564079/full) -- Enhanced multi-scale SAM for leaf disease
12. [pytorch-grad-cam GitHub](https://github.com/jacobgil/pytorch-grad-cam) -- GradCAM for ViT and CNN models
13. [pytorch-grad-cam ViT Tutorial](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md) -- Vision Transformer implementation guide
14. [CLIP GradCAM Notebook](https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb) -- CLIP + GradCAM visualization
15. [SigLIP 2 GradCAM (2025)](https://blogs.gwu.edu/pless/2025/04/21/siglip-2-gradcam-attention-visualization/) -- Recent GradCAM on vision-language encoders
16. [CLIP Surgery (ArXiv 2304.05653)](https://arxiv.org/abs/2304.05653) -- Improved CLIP explainability
17. [CLIP-ES (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_CLIP_Is_Also_an_Efficient_Segmenter_A_Text-Driven_Approach_for_CVPR_2023_paper.pdf) -- Zero-shot segmentation from CLIP
18. [CRG (ECCV 2024)](https://arxiv.org/abs/2403.02325) -- Contrastive Region Guidance for VLMs
19. [CRG GitHub](https://github.com/meetdavidwan/crg) -- PyTorch implementation
20. [R-VLM (ACL 2025)](https://aclanthology.org/2025.findings-acl.501.pdf) -- Region-aware VLM for precise grounding
21. [RMBG 2.0 (HuggingFace)](https://huggingface.co/briaai/RMBG-2.0) -- BRIA AI background removal model
22. [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) -- Multimodal vision-language model
23. [Qwen2.5-VL Technical Report (ArXiv)](https://arxiv.org/abs/2502.13923) -- Visual grounding capabilities
24. [Baby Kale Grounding DINO + SAM Study](https://www.sciencedirect.com/science/article/pii/S2772375525001364) -- Agricultural application
25. [SAM Orchestration for Agriculture (Frontiers 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1748468/full) -- SAM annotation pipelines for agricultural datasets
