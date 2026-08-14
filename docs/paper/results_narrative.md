# Results Narrative & Emerging Story

> **Status**: Living document. Updated as ablation results accumulate.
> **Last updated**: 2026-05-10 20:00 (n=200 A, n=172 B, n=200 C, n=50 D)

---

## The Original Hypothesis (What We Expected)

RAG augmentation would improve classification accuracy by grounding
VLM-extracted morphological traits in published botanical literature.

## What The Data Actually Shows

### Full Results Table

**Family-Level Accuracy:**

| Condition | Final | Vis@1 | Vis@3 | Vis@5 | N |
|-----------|:-:|:-:|:-:|:-:|:-:|
| A (full pipeline) | 88.0% | 65.5% | 88.0% | 94.0% | 200 |
| B (no RAG) | 94.2% | 67.4% | 87.8% | 94.2% | 172 |
| C (visual only → reasoning) | **95.5%** | 65.5% | 88.0% | 94.0% | 200 |
| D (baseline, Stage 2 top-1) | 62.0% | 62.0% | 84.0% | 94.0% | 50 |

**Multi-Rank Accuracy (Pipeline Final Prediction):**

| Condition | Family | Genus | Species | N |
|-----------|:-:|:-:|:-:|:-:|
| A (full pipeline) | 88.0% | 87.0% | 76.5% | 200 |
| B (no RAG) | 94.2% | 94.2% | 71.5% | 172 |
| C (visual only) | **95.5%** | 95.5% | 87.5% | 200 |
| D (baseline) | 62.0% | — | — | 50 |

*Note: Condition D genus/species numbers (86%) are artifacts of independent
per-rank SimpleShot classifiers producing hierarchically inconsistent
predictions. Only family is meaningful for D.*

**Significance Tests (McNemar's, family level):**

| Comparison | chi2 | p-value | Discordant | Direction |
|-----------|:-:|:-:|:-:|:-:|
| A vs B | 9.09 | 0.0026** | 11 | B better |
| A vs C | 11.53 | 0.0007*** | 17 | C better |
| A vs D | 12.50 | 0.0004*** | 18 | A better |
| A vs B (species) | 1.44 | 0.23 ns | 34 | — |

### Finding 1: LLM reasoning is the dominant effect (+33%)

| Source | Family Accuracy |
|--------|:-:|
| D: Visual classifier top-1 (no reasoning) | 62.0% |
| C: Visual → LLM reasoning (no morphology, no RAG) | **95.5%** |

The reasoning LLM boosts family accuracy by **33.5 percentage points** over
the raw visual classifier. This is the paper's strongest result. The LLM
takes Stage 2's ranked visual predictions — where the correct family appears
in top-5 94% of the time but at top-1 only 62% — and selects the right
candidate. No morphological traits, no literature, no RAG needed.

### Finding 2: Morphological traits (Stage 1) hurt performance

| Condition | Family Accuracy | Has Morphology |
|-----------|:-:|:-:|
| C (visual only) | **95.5%** | No |
| B (visual + morphology) | 94.2% | Yes |
| A (visual + morphology + RAG) | 88.0% | Yes |

Adding Stage 1 morphological traits reduces accuracy from 95.5% to 94.2%
(-1.3%). Adding RAG on top of morphology further reduces to 88.0% (-6.2%).

**Why?** The VLM's morphological extraction has ~77% mean accuracy across
traits (Table 2, Section 5.1). The 23% error rate introduces incorrect trait
information into the evidence document. The reasoning LLM trusts these
traits — "leaf arrangement: alternate" — even when they're wrong, and this
misleads its classification. The visual classifier's ranked predictions
already encode the correct answer (top-5 = 94%), so adding noisy
morphological text can only hurt.

### Finding 3: RAG amplifies morphology errors

RAG compounds the morphology problem. When Stage 1 extracts incorrect traits
(e.g., "alternate" for a truly "opposite" plant), the RAG query becomes:
"Tropical tree seedling with relative_position: alternate..." — retrieving
literature descriptions for the *wrong* trait profile. The LLM then sees
morphology evidence, visual evidence, AND literature evidence all pointing
in different directions, and the literature can tip it wrong.

Evidence: RAG helped 0 specimens and hurt 11 at family level.

### Finding 4: Species-level RAG effect has weakened

Early results (n=42 paired) showed A: 82.4% vs B: 66.7% species accuracy
(+15.7%, p=0.041*). With more data (n=172 paired), this narrowed to
A: 76.5% vs B: 71.5% (+5.0%, p=0.23 ns). The effect may be real but is
not statistically significant at current sample size.

However, condition C's species accuracy (87.5%) is the highest of all
conditions — again suggesting that morphological traits are the bottleneck,
not RAG availability.

### Finding 5: Convergence signals don't predict accuracy

| Signal | N | Accuracy |
|--------|:-:|:-:|
| Moderate convergence | 94 (47%) | 84.0% |
| Divergent | 106 (53%) | 91.5% |
| Strong convergence | 0 (0%) | — |

Specimens with convergence are *less* accurate than those without. The
convergence mechanism identifies when RAG and visual agree — but agreement
doesn't mean correctness. Both sources can converge on the wrong answer.

---

## The Revised Paper Narrative

### The story is NOT about RAG improving accuracy

The data does not support the original thesis. RAG augmentation does not
improve classification accuracy — it reduces it, significantly (p=0.003).

### The story IS about three things:

**1. LLM reasoning over visual evidence is transformative.**
A text-only LLM reading ranked visual predictions achieves 95.5% family
accuracy from a 62% visual baseline — a +33.5% boost. This requires no
morphological extraction, no literature retrieval, no RAG. The pipeline's
architectural contribution is separating vision from reasoning and letting
the LLM select from a ranked candidate list rather than accepting the
classifier's single top-1 answer.

**2. VLM morphological extraction is the bottleneck, not the solution.**
At ~77% trait accuracy, Stage 1 introduces more noise than signal. The
reasoning LLM performs better WITHOUT morphological traits (C: 95.5%) than
WITH them (B: 94.2%). This suggests that current VLMs are not reliable
enough for structured trait extraction to improve downstream classification.
The traits help the LLM reason *about* plants but can mislead it when wrong.

**3. RAG over imperfect traits amplifies errors.**
RAG retrieves literature matching the *extracted* traits, not the *true*
traits. When extraction is wrong (23% of the time), RAG retrieves the wrong
literature, adding a second source of misinformation. The convergence
mechanism cannot catch this because it detects agreement between sources,
not correctness — and both sources can agree on the wrong answer when the
underlying query is wrong.

### Implications

- **The pipeline's value** is in the Stage 2→4→5 path (visual → evidence
  doc → reasoning), not the full 5-stage path. The simpler pipeline is
  both faster and more accurate.

- **RAG becomes valuable when Stage 1 improves.** If morphological extraction
  reaches ~95% accuracy (per the research roadmap in
  `docs/research/morph-improve-03192026/`), the RAG query composition will
  produce accurate queries, and literature retrieval will add genuine signal
  rather than amplifying errors.

- **The paper should present this as an honest ablation** showing where each
  component adds vs. subtracts value, rather than as a RAG success story.
  The LLM reasoning result (62%→95.5%) is strong enough to carry the paper
  without needing RAG to "work."

---

## Data Caveats

1. **Condition D sample size** is only 50 (vs 200 for A/C). The 62% family
   accuracy may not be stable.

2. **Condition D genus/species** (86%) is an artifact — independent per-rank
   classifiers produce hierarchically inconsistent predictions (family=Lauraceae
   but genus=Amaioua, which is Rubiaceae). Only family is valid for D.

3. **Paired McNemar tests** are limited to specimens appearing in both
   conditions. A has 200, B has 172 — overlap depends on which specimens
   each shard processed. 2 B shards are still running.

4. **Species matching** is case-insensitive exact string comparison of
   predicted binomial vs ground truth binomial. Synonyms, misspellings, or
   alternative binomials would register as mismatches.

---

## Updated Figures Plan

1. **Bar chart**: Family accuracy by condition (D → C → B → A) showing
   the reasoning boost and the morphology/RAG degradation
2. **Error cascade diagram**: Stage 1 trait error → bad RAG query → wrong
   literature → wrong classification (with a concrete example)
3. **Visual top-k selection**: Example where Stage 2 top-1 is wrong but
   top-3 contains the correct family, and the LLM selects correctly
4. **Accuracy vs trait extraction quality**: Hypothetical curve showing
   at what Stage 1 accuracy RAG becomes beneficial (future work framing)
