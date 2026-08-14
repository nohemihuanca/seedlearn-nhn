# Why Cohen's κ was zero — and which zeros were real

**Date:** 2026-07-16
**Data:** C0 baseline (`Qwen3-VL-32B`) and C1 upgraded (`Qwen3.6-35B`) all-trait runs,
114 curator specimens graded against Roni.

The 2026-07-06 human-grading report showed κ = 0.000 for many traits
(`leaf_margin`, `stem_type`, `stipules`, `pulvinus`, `tendrils`, `petiole_features`)
and near-zero for others (`leaf_apex` −0.057, `leaf_relative_position` 0.064). The
single-trait experiment narrowed to leaf margin on the premise that "leaf margin does
poorly." **That premise was half right and half an artifact.** A κ of zero here has
**three** distinct causes, and only one of them is a bug.

---

## Mechanism A — canonicalization drop (a real grading bug, now fixed)

**This is what happened to leaf margin.** The baseline model routinely describes a
toothed margin with a *compound* descriptor — `"toothed, serrate"`, `"toothed,
dentate"`. The pre-fix grader matched values only by exact alias, so every compound
value fell to `MISSING` and left the comparison. What remained comparable was almost
entirely the clean single word `"entire"` — so the model looked like a **constant
"entire" rater**, and κ collapsed to exactly 0.000 despite ~84% raw agreement.

**14 specimens** were silently dropped this way (n fell 112 → 97). Concrete examples:

| Specimen | Species | Model said | Dropped pre-fix → |
|----------|---------|-----------|-------------------|
| BAR0299 | *Eugenia nesiotica* | `toothed, serrate` | MISSING |
| PPPIT1RU4 | *Cojoba rufescens* | `toothed, dentate` | MISSING |
| ANROUPMO6 | *Roupala montana* | `toothed, serrate` | MISSING |
| BVTET1PO6 | *Tetracera portobellensis* | `toothed, serrate` | MISSING |
| CHARDIPE7 | *Ardisia pellucida* | `toothed, serrate` | MISSING |
| SHCORDBI3 | *Cordia bicolor* | `toothed, dentate` | MISSING |
| … | (14 total) | `toothed, {serrate,dentate}` | MISSING |

Commit `2fe6a65` fixed this: compound descriptors whose recognized tokens agree now
resolve to the single canonical (`"toothed, serrate"` → `toothed`), negation-guarded.
After the fix, leaf-margin κ vs Roni is **0.469** (baseline), not 0 — and the upgraded
model lifts it to **0.591**. Leaf margin never "did poorly"; the grader was discarding
every case where the model said anything more specific than *entire*.

**Only leaf margin suffered Mechanism A** among the zeros — it is the only trait whose
values are routinely compound. The other zeros are *not* this bug.

---

## Mechanism B — a constant rater (real, not a bug, unfixable by a better model)

κ corrects for chance agreement. When **one rater assigns a single class to every
specimen**, there is no variance to correct against and κ is 0 (or undefined) no matter
how the other rater behaves. For these traits the *human* annotator is the constant one,
because the trait is near-invariant across tropical **tree seedlings**:

| Trait | Roni's labels | Model's labels | Why κ = 0 |
|-------|---------------|----------------|-----------|
| `stem_type` | woody × 109 (only) | woody 68, herbaceous 46 | Roni calls every seedling woody → no variance |
| `tendrils` | absent × 100 (only) | absent 112, present 2 | tendrils absent in nearly all → no variance |
| `stipules`, `pulvinus`, `petiole_features` | one class dominates | — | same degenerate-marginal pattern |

This is a **legitimate** zero: the model isn't wrong (all these seedlings *are* woody),
there's simply nothing for κ to measure. A stronger model cannot fix it — and indeed the
upgraded model leaves `stem_type` and `tendrils` at 0.000. (Where the human labels do
carry a little variance — `petiole_features`, `pulvinus` — the upgraded model *does* move
κ off zero: 0.000 → 0.188 and 0.000 → 0.037.)

> Note: `compound_leaf_type` reads κ=0 for both Qwen models but **1.000 for GPT-5.1** —
> on the ~17 compound-leaf specimens the Qwen models are degenerate while GPT is not; that
> zero is model behavior, not annotation.

---

## Mechanism C — genuine disagreement (real signal, NOT an artifact)

This is the key answer to *"is the same thing happening with leaf relative position and
leaf apex?"* — **No.** Both raters vary, κ is well-defined, and it is low because the
model and Roni genuinely disagree. Nothing was dropped; no rater is constant.

**`leaf_apex`** (κ ≈ 0.06, *not* zero):

| | acute | acuminate | obtuse |
|---|------|-----------|--------|
| Model | **96** | 6 | 12 |
| Roni | 53 | **59** | 2 |

The model over-calls **acute** where Roni sees **acuminate** — a genuinely subtle
distinction (a drawn-out vs. merely pointed tip), not a grading fault. The upgraded model
improves it modestly (0.061 → 0.107).

**`leaf_relative_position`** (κ ≈ 0.06, *not* zero):

| | alternate | opposite | whorled |
|---|-----------|----------|---------|
| Model | 43 | 18 | **53** |
| Roni | 66 | 44 | **0** |

The model calls 53 specimens **whorled**; Roni uses *whorled* **zero times** — she reads
the same clustered-at-node arrangement as *opposite*. This is a real definitional/judgment
gap, not the canonicalization artifact. The upgraded model barely moves it
(0.061 → 0.074), consistent with a genuine disagreement a better model doesn't resolve.

---

## Summary

| Trait(s) | Mechanism | Was it an artifact? | Fixed / improved? |
|----------|-----------|---------------------|-------------------|
| `leaf_margin` | A — canonicalization drop | **Yes (bug)** | Fixed → κ 0.47, upgrade → 0.59 |
| `stem_type`, `tendrils`, `stipules`, `pulvinus` | B — constant human rater | No (legitimate) | Not fixable; some move slightly |
| `leaf_apex`, `leaf_relative_position` | C — genuine disagreement | **No** | Real; upgrade helps modestly |

**Takeaways.** (1) Narrowing to leaf margin was based on an artifact — leaf margin was
fine once the grader stopped dropping compound values. (2) Leaf apex and leaf relative
position are *not* the same artifact; their low κ is a true model–human gap. (3) The
exactly-0.000 binary traits are class-imbalance, not a bug, and no model change fixes them.
(4) Across all 21 gradable traits, the upgraded model still improves **12, worsens 3, ties
3** vs baseline — the upgrade helps broadly, and separately from the leaf-margin fix.
