#!/usr/bin/env python
"""Batch pipeline runner for ablation experiments.

Runs the seedlearn pipeline on all test-partition specimens under a specified
ablation condition (A/B/C), with sharding support for parallel SLURM execution.

Usage:
    python batch_runner.py --condition A --shard-index 0 --num-shards 4 \
        --vlm-endpoint http://localhost:35789/v1

Conditions:
    A: Full pipeline (all stages)
    B: No RAG (skip trait_retrieval)
    C: Visual only (skip morphology + trait_retrieval)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root: experiments/ablation/runners/ -> seedlearn-dev/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Condition -> skip_stages mapping
CONDITIONS = {
    "A": [],
    "B": ["trait_retrieval"],
    "C": ["morphology", "trait_retrieval"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch ablation runner.")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["A", "B", "C"],
        help="Ablation condition (A=full, B=no_rag, C=visual_only).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard of individuals to process (0-indexed).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--vlm-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for result JSONs (default: outputs/condition_X).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config.yaml (default: auto-detect).",
    )
    parser.add_argument(
        "--max-specimens",
        type=int,
        default=None,
        help="Limit number of specimens (for testing).",
    )
    return parser.parse_args()


def load_experiment_config(config_path: Path | None) -> dict:
    """Load the experiment config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "experiments" / "ablation" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def enumerate_test_individuals(
    catalog_path: Path,
    splits_dir: Path,
    split_seed: int,
) -> list[tuple[str, str, list[str]]]:
    """Enumerate all test-partition individuals with their images.

    Returns:
        List of (individual_id, ground_truth_family, [image_paths]).
    """
    from seedlearn.data.catalog import load_dataset
    from seedlearn.data.splits import load_split

    records, _ = load_dataset(catalog_path, rank="family")
    split_path = splits_dir / "family" / f"split_seed{split_seed}"
    split = load_split(split_path)

    # Build individual -> images lookup from test partition
    individual_images: dict[str, list[str]] = defaultdict(list)
    individual_family: dict[str, str] = {}
    for idx in split.test_indices:
        rec = records[idx]
        individual_images[rec.individual_id].append(str(rec.image_path))
        individual_family[rec.individual_id] = rec.family

    result = []
    for indiv_id in sorted(individual_images.keys()):
        result.append((
            indiv_id,
            individual_family[indiv_id],
            sorted(individual_images[indiv_id]),
        ))
    return result


def build_pipeline(
    exp_config: dict,
    condition: str,
    vlm_endpoint: str,
):
    """Build and return the PipelineRunner with all stages initialized.

    Mirrors scripts/run_pipeline.py lines 282-365.
    """
    from seedlearn.pipeline.config import load_config
    from seedlearn.pipeline.rag import RAGIndex
    from seedlearn.pipeline.runner import PipelineRunner
    from seedlearn.pipeline.stages.classification import ClassificationStage
    from seedlearn.pipeline.stages.evidence import EvidenceSynthesisStage
    from seedlearn.pipeline.stages.morphology import MorphologyStage
    from seedlearn.pipeline.stages.reasoning import ReasoningStage
    from seedlearn.pipeline.stages.trait_retrieval import TraitRetrievalStage

    skip_stages = CONDITIONS[condition]
    skip_set = set(skip_stages)

    # Load pipeline config with endpoint override
    pipeline_yaml = PROJECT_ROOT / exp_config["pipeline_config"]
    overrides = {
        "vlm.endpoint": vlm_endpoint,
        "reasoning.endpoint": vlm_endpoint,
        "skip_stages": json.dumps(skip_stages),
    }
    config = load_config(str(pipeline_yaml), overrides=overrides)

    data = exp_config["data"]
    cache_dir = PROJECT_ROOT / data["cache_dir"]
    splits_dir = PROJECT_ROOT / data["splits_dir"]
    rag_index_dir = PROJECT_ROOT / data["rag_index"]
    split_seed = data["split_seed"]

    # Stage 1: VLM Morphology
    morphology_stage = MorphologyStage(config.vlm, prompt_file=config.prompts.morphology)

    # Stage 2: Classification — load multi-rank cache
    classification_stage = ClassificationStage(config.classifier)
    if "classification" not in skip_set:
        split_name = f"split_seed{split_seed}"
        split_paths: dict[str, Path] = {}
        for rank in ("family", "genus", "species"):
            rank_split = splits_dir / rank / split_name
            if rank_split.with_suffix(".npz").exists():
                split_paths[rank] = rank_split
        if split_paths:
            logger.info("Loading multi-rank classifier (%d ranks)...", len(split_paths))
            classification_stage.load_from_multirank_cache(cache_dir, split_paths)
        else:
            logger.error("No split files found in %s", splits_dir)
            sys.exit(1)

    # Stage 3: Trait Retrieval (RAG)
    trait_stage = TraitRetrievalStage(
        config.trait_retrieval, prompt_file=config.prompts.rag_query
    )
    if "trait_retrieval" not in skip_set:
        logger.info("Loading RAG index from %s...", rag_index_dir)
        trait_stage.rag_index = RAGIndex.load(rag_index_dir)

    # Stage 4: Evidence Synthesis
    evidence_stage = EvidenceSynthesisStage(config.evidence_synthesis)

    # Stage 5: Reasoning
    reasoning_stage = ReasoningStage(
        config.reasoning, prompt_file=config.prompts.reasoning
    )

    # Wire into runner
    runner = PipelineRunner(config)
    runner.stages = [
        morphology_stage,
        classification_stage,
        trait_stage,
        evidence_stage,
        reasoning_stage,
    ]
    return runner


def main() -> None:
    """Run batch ablation experiment."""
    args = parse_args()
    exp_config = load_experiment_config(
        Path(args.config) if args.config else None
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            PROJECT_ROOT / "experiments" / "ablation" / "outputs"
            / f"condition_{args.condition}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate test individuals
    data = exp_config["data"]
    logger.info("Loading catalog and splits...")
    individuals = enumerate_test_individuals(
        catalog_path=PROJECT_ROOT / data["catalog"],
        splits_dir=PROJECT_ROOT / data["splits_dir"],
        split_seed=data["split_seed"],
    )
    logger.info("Total test individuals: %d", len(individuals))

    # Apply sharding
    shard = individuals[args.shard_index :: args.num_shards]
    logger.info(
        "Shard %d/%d: %d individuals",
        args.shard_index, args.num_shards, len(shard),
    )

    # Apply max-specimens limit
    if args.max_specimens:
        shard = shard[: args.max_specimens]
        logger.info("Limited to %d specimens (--max-specimens)", len(shard))

    # Build pipeline (expensive init — done once)
    logger.info("Building pipeline (condition %s, skip=%s)...", args.condition, CONDITIONS[args.condition])
    runner = build_pipeline(exp_config, args.condition, args.vlm_endpoint)

    # Process specimens
    completed = 0
    skipped = 0
    errors = 0
    total = len(shard)
    batch_start = time.perf_counter()

    for i, (indiv_id, gt_family, image_paths) in enumerate(shard, 1):
        out_path = output_dir / f"{indiv_id}.json"

        # Resume: skip if output exists
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            logger.info("[%d/%d] SKIP %s (already exists)", i, total, indiv_id)
            continue

        start = time.perf_counter()
        result = runner.run(specimen_id=indiv_id, image_paths=image_paths)
        elapsed = time.perf_counter() - start

        # Extract predicted family for logging
        result_dict = result.to_dict()
        predicted = "?"
        reasoning_data = result.get_stage_data("reasoning")
        if reasoning_data:
            clf = reasoning_data.get("classification", {})
            predicted = clf.get("predicted_family", "?")
        else:
            clf_data = result.get_stage_data("classification")
            preds = clf_data.get("predictions_by_rank", {}).get("family", [])
            if preds:
                predicted = preds[0].get("rank_value", "?")

        # Check for errors
        has_error = any(
            sr.error for sr in result.stage_results.values()
            if not sr.skipped
        )
        if has_error:
            errors += 1
            status = "ERROR"
        else:
            completed += 1
            status = "OK" if predicted == gt_family else "WRONG"

        logger.info(
            "[%d/%d] %s %s -> %s (gt=%s) %.1fs",
            i, total, status, indiv_id, predicted, gt_family, elapsed,
        )

        # Add ground truth to output for analysis convenience
        result_dict["ground_truth"] = {
            "family": gt_family,
        }
        result_dict["condition"] = args.condition

        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    batch_elapsed = time.perf_counter() - batch_start
    logger.info(
        "Done. Completed=%d, Skipped=%d, Errors=%d, Total=%.1fs (%.1fs/specimen)",
        completed, skipped, errors, batch_elapsed,
        batch_elapsed / max(completed, 1),
    )


if __name__ == "__main__":
    main()
