# BioCLIP 2 SimpleShot Baseline

**Date:** 2026-08-14  
**Status:** Completed family-level visual baseline  
**Script:** [`scripts/run_alltrain_individual_numpy.py`](../../scripts/run_alltrain_individual_numpy.py)

This baseline evaluates frozen BioCLIP 2 image embeddings with a lightweight
SimpleShot-style nearest-centroid classifier. It is not BioCLIP 2 fine-tuning:
BioCLIP 2 stays frozen, every image is converted into a 768-dimensional visual
embedding, and the classifier averages training embeddings into one centroid per
family.

## Matched Baseline Result

Cluster output:

```text
/home/nh525/seedlearn_runs/bioclip2_simpleshot/results/family_alltrain_seed42_individual/metrics.json
```

| Metric | Value |
| --- | ---: |
| Top-1 family accuracy | 64.7% |
| Top-5 family accuracy | 91.5% |
| Test individuals | 317 |
| Test images | 1,577 |
| Avg. test images per individual | 4.97 |
| Training individuals | 1,478 |
| Training images | 7,270 |
| Families | 52 |
| Classifier | all training-image centroids |
| Test unit | individual mean embedding |

## Split Details

| Rank | Classes | Individuals | Images | Train | Val | Test | Max k-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Family | 52 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 10 |
| Genus | 114 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 3 |
| Species | 164 | 2,112 | 10,407 | 1,478 (7,270) | 317 (1,560) | 317 (1,577) | 3 |

Splits are individual-level grouped splits: 70% train, 15% validation, and 15%
test. All photos of the same individual stay in the same partition to avoid
train/test leakage.

## How It Works

1. Convert every training image to a frozen BioCLIP 2 embedding.
2. For each family, average all training image embeddings from that family.
3. Treat that average as the family centroid.
4. For each test individual, average all of its image embeddings into one
   individual-level embedding.
5. Compare that embedding to every family centroid.
6. Top-1 is the closest family; Top-5 counts the result correct if the true
   family is anywhere in the five closest families.

This is the comparison to use for Mitch-style visual candidate generation. It
differs from quick k-shot image-level runs, which sample a fixed number of
training images per class and evaluate each test image independently.

## Cluster Paths

```text
/home/nh525/project_pi_lsc4/shared/seedlearn/data/embeddings/2026-01-29_v2026-01-29_12K/features.npz
/home/nh525/project_pi_lsc4/shared/seedlearn/data/splits/2026-01-29_v2026-01-29_12K/family/split_seed42
/home/nh525/seedlearn_runs/bioclip2_simpleshot/results/family_alltrain_seed42_individual
```

## Interpretation

BioCLIP 2 + SimpleShot is a strong family-level visual candidate generator: the
correct family appears in the top five about 91.5% of the time. The next
constraint is likely downstream trait extraction, retrieval, and reasoning, not
whether frozen BioCLIP 2 embeddings can retrieve plausible family candidates.
