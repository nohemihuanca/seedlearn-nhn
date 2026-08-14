# SeedLearn Monday Prep Notes

Date: 2026-08-14

These are internal working notes for preparing the Monday project-status talk and workshop plan.

## Current Project Framing

SeedLearn develops AI-assisted tools for identifying tropical tree seedlings from field-collected image sets. The current project has two main active threads:

- BioCLIP 2 embeddings plus SimpleShot/few-shot classification for family, genus, and species identification.
- Vision-language-model trait extraction and human grading for morphology traits.

The GitHub repo to use going forward is Nohemi's independent clean copy:

- `https://github.com/nohemihuanca/seedlearn-nhn`

Do not push to or modify Mitch's original repo unless explicitly requested.

## BioCLIP 2 Data Status

The newest embedding data found on Bouchet is:

```text
/home/nh525/project_pi_lsc4/shared/seedlearn/data/embeddings/2026-01-29_v2026-01-29_12K/features.npz
```

This file contains:

- 10,407 images
- 768-dimensional BioCLIP 2 features
- `family_labels`, `genus_labels`, `species_labels`
- `individual_ids`
- `image_paths`

The matching split folder is:

```text
/home/nh525/project_pi_lsc4/shared/seedlearn/data/splits/2026-01-29_v2026-01-29_12K/
```

The split summary from the repo says:

| Rank | Classes | Individuals | Images | Max k-shot |
|---|---:|---:|---:|---:|
| Family | 52 | 2,112 | 10,407 | 10 |
| Genus | 114 | 2,112 | 10,407 | 3 |
| Species | 164 | 2,112 | 10,407 | 3 |

Important: because species max k-shot is 3, a full all-species BioCLIP 2 SimpleShot run should use `k=3`, not `k=5`, unless rare species are filtered out.

## Older BioCLIP Results

The older downloaded reports are from the previous evaluation stage:

- Dataset: Oct 23, 2025 clean data
- 8,377 seedling images
- 49 family classes
- 170 species classes

Old species SimpleShot 5-shot result:

- Top-1 accuracy: 48.21%
- Top-5 accuracy: 73.03%

These are useful as historical context, but should not be presented as the new BioCLIP 2 result.

Current honest phrasing:

> Previous BioCLIP SimpleShot species benchmark reached 48% top-1 and 73% top-5 accuracy on the older 8,377-image dataset. We now have BioCLIP 2 embeddings for 10,407 images, and the next step is to rerun the equivalent SimpleShot benchmark on the new embeddings.

## Next BioCLIP 2 SimpleShot Run

Do not run this interactively on the login node. Submit as a SLURM job from Nohemi's writable folder.

Suggested run directory:

```text
/home/nh525/seedlearn_runs/bioclip2_simpleshot/
```

Recommended first run:

- Rank: species
- k-shot: 3
- Split: `split_seed42`
- Device: CPU
- Output: Nohemi's home folder

The job should:

1. Read the shared BioCLIP 2 multi-rank `features.npz`.
2. Write a local `species_features.npz` cache using `species_labels`.
3. Run `scripts/run_simpleshot.py` with `--rank species --k-shot 3`.
4. Save `metrics.json`, `predictions.csv`, `support_set.json`, and `experiment_info.json` under Nohemi's run directory.

After it finishes, use `metrics.json` to make an updated figure in the style of the old species identification accuracy plot.

Potential updated figure bars:

- Random chance: `1 / 164 = 0.61%`
- BioCLIP2 zero-shot: pending or omitted unless rerun
- BioCLIP2 SimpleShot 3-shot top-1: from new `metrics.json`
- BioCLIP2 SimpleShot 3-shot top-5: from new `metrics.json`

## Trait Grading Status

Human trait grading report:

```text
/Users/nohemihuancanunez/Downloads/human_grading_report.html
```

Repo copy:

```text
trait_grading/reports/2026-07-06_153457/human_grading_report.html
```

Model run:

- `Qwen/Qwen3-VL-32B-Instruct-FP8`
- Prompt style: `sys4`
- Specimens: 114

Overall macro agreement:

- Model vs Roni: 0.68
- Model vs Carmen: 0.68
- Roni vs Carmen: 0.79

Overall macro kappa:

- Model vs Roni: 0.10
- Model vs Carmen: 0.13
- Roni vs Carmen: 0.45

Interpretation: raw agreement looks moderate, but low model-human kappa means much of the apparent agreement may come from common/default categories. Kappa is chance-corrected agreement.

Per-trait extracted report rows:

- 21 trait rows total
- 19 traits include model-vs-human comparisons

Useful examples for Monday:

- Leaf complexity is strong:
  - Model vs Roni: 92% agreement, kappa 0.74
  - Model vs Carmen: 89% agreement, kappa 0.65
  - Roni vs Carmen: 96% agreement, kappa 0.88
- Leaf relative position / leaf arrangement is weak:
  - Model vs Roni: 34% agreement, kappa 0.06
  - Model vs Carmen: 40% agreement, kappa 0.17
  - Roni vs Carmen: 76% agreement, kappa 0.58

## Local Monday Figures

Generated local figures are in:

```text
/Users/nohemihuancanunez/Downloads/seedlearn_monday_figures/
```

Important files:

- `human_trait_benchmark_model_agreement_original_style_larger_text.png`
- `human_trait_benchmark_model_agreement_original_style_larger_text.pdf`
- `human_trait_model_vs_roni_summary.png`
- `human_trait_model_vs_roni_summary.pdf`
- `human_trait_grading_per_trait.csv`
- `human_trait_grading_overall.csv`

Repo figures already committed/pushed:

- `docs/clip/figures/bioclip2_pca_family.png`
- `docs/clip/figures/bioclip2_pca_family_ellipses.png`
- `docs/clip/figures/seedlearn_family_distribution.png`

## Monday Talk Suggested Story

1. Project consolidation:
   - Clean independent repo created as `seedlearn-nhn`.
   - Mitch's work preserved; Nohemi now has a working repo for collaborators.

2. BioCLIP 2 status:
   - New embeddings exist for 10,407 images.
   - Current PCA/distribution figures summarize the new training set.
   - Updated BioCLIP 2 SimpleShot evaluation still needs to be run or located.

3. Trait grading status:
   - VLM trait extraction has a human grading benchmark.
   - Some traits are promising, especially leaf complexity.
   - Some traits remain difficult, especially leaf arrangement/relative position.

4. Workshop plan:
   - Run BioCLIP 2 SimpleShot baselines.
   - Decide whether to report all-species 3-shot or filtered-species 5-shot.
   - Improve trait prompts where model-human kappa is low.
   - Prepare collaborator-friendly repo instructions and issues.
