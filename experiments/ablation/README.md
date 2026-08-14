# Ablation Experiment: RAG Augmentation

Demonstrates that RAG augmentation improves tropical seedling classification
accuracy over raw VLM output. Supports the results in Section 5 of the paper.

## Conditions

| ID | Name | Stages Used | What It Tests |
|----|------|-------------|---------------|
| A | `full_pipeline` | 1+2+3→4→5 | All evidence including RAG + convergence |
| B | `no_rag` | 1+2→4→5 | Morphology + visual, no literature |
| C | `visual_only` | 2→4→5 | Visual predictions only → LLM reasoning |
| D | `baseline` | 2 only | SimpleShot top-1 (no LLM reasoning) |

## Workflow

### 1. Submit

```bash
cd /nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev
bash experiments/ablation/slurm/launch_all.sh
```

Submits 13 GPU jobs: 12 array tasks (3 conditions × 4 shards) + 1 baseline.

### 2. Monitor

```bash
# Quick count
python experiments/ablation/analysis/monitor.py

# Validate all outputs (checks JSON integrity + stage errors)
python experiments/ablation/analysis/monitor.py --validate

# Show error details
python experiments/ablation/analysis/monitor.py --errors

# Auto-refresh until complete
python experiments/ablation/analysis/monitor.py --watch --validate
```

### 3. Validate

After all jobs complete, run full validation:

```bash
python experiments/ablation/analysis/monitor.py --validate --errors
```

Expected: 317 valid outputs per condition, 1,268 total.

If specimens are missing (preemption, timeout), resubmit — the runner
skips existing outputs automatically.

### 4. Analyze

```bash
python experiments/ablation/analysis/compute_metrics.py
```

Produces:
- `outputs/tables/accuracy_by_condition.csv` — top-1/3/5 per condition
- `outputs/tables/per_family_accuracy.json` — per-family breakdown
- `outputs/tables/mcnemar_tests.json` — paired significance tests
- `outputs/tables/rag_precision.json` — RAG retrieval quality (Exp 2)
- `outputs/tables/convergence_analysis.json` — convergence vs accuracy (Exp 3)
- `outputs/tables/rag_impact_cases.json` — specimens where RAG changed outcome

### 5. Save provenance

```bash
python experiments/ablation/analysis/monitor.py --provenance
```

Saves `outputs/provenance.json` with git commit, package versions, SLURM job
IDs, and completion counts.

## Recovery

If jobs are preempted or fail partway:

```bash
# Check what's missing
python experiments/ablation/analysis/monitor.py --validate --errors

# Resubmit — runners skip completed specimens automatically
bash experiments/ablation/slurm/launch_all.sh
```

## Directory Structure

```
experiments/ablation/
├── config.yaml           # Central paths + conditions
├── README.md             # This file
├── runners/
│   ├── batch_runner.py   # Conditions A/B/C (GPU + vLLM)
│   └── baseline_runner.py# Condition D (GPU, no vLLM)
├── analysis/
│   ├── monitor.py        # Progress tracking + validation
│   └── compute_metrics.py# Accuracy tables + significance tests
├── slurm/
│   ├── launch_all.sh     # Master launcher
│   ├── ablation_gpu.sbatch
│   └── baseline_gpu.sbatch
└── outputs/              # Created at runtime
    ├── condition_{A,B,C,D}/  # Per-specimen JSONs
    ├── logs/                  # SLURM stdout/stderr
    ├── tables/                # Analysis output
    └── provenance.json        # Reproducibility metadata
```
