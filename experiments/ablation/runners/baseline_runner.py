#!/usr/bin/env python
"""Baseline runner for ablation condition D.

Runs Stage 2 (BioCLIP classification) only — no vLLM needed. Extracts the
top-1 prediction per rank and saves results in PipelineResult-compatible
JSON format so analysis scripts work uniformly across all conditions.

Usage:
    python baseline_runner.py --output-dir outputs/condition_D --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Baseline (condition D) runner.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for result JSONs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device for BioCLIP feature extraction.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config.yaml.",
    )
    parser.add_argument(
        "--max-specimens",
        type=int,
        default=None,
        help="Limit number of specimens (for testing).",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline classification for all test individuals."""
    args = parse_args()

    config_path = Path(args.config) if args.config else (
        PROJECT_ROOT / "experiments" / "ablation" / "config.yaml"
    )
    with open(config_path) as f:
        exp_config = yaml.safe_load(f)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "experiments" / "ablation" / "outputs" / "condition_D"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = exp_config["data"]

    # Ensure project root is importable
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from experiments.ablation.runners.batch_runner import enumerate_test_individuals

    from seedlearn.pipeline.config import load_config
    from seedlearn.pipeline.runner import PipelineRunner
    from seedlearn.pipeline.stages.classification import ClassificationStage
    from seedlearn.pipeline.stages.evidence import EvidenceSynthesisStage
    from seedlearn.pipeline.stages.morphology import MorphologyStage
    from seedlearn.pipeline.stages.reasoning import ReasoningStage
    from seedlearn.pipeline.stages.trait_retrieval import TraitRetrievalStage

    # Enumerate test individuals
    logger.info("Loading catalog and splits...")
    individuals = enumerate_test_individuals(
        catalog_path=PROJECT_ROOT / data["catalog"],
        splits_dir=PROJECT_ROOT / data["splits_dir"],
        split_seed=data["split_seed"],
    )
    logger.info("Total test individuals: %d", len(individuals))

    if args.max_specimens:
        individuals = individuals[: args.max_specimens]

    # Build pipeline with all non-classification stages skipped
    skip_stages = ["morphology", "trait_retrieval", "evidence_synthesis", "reasoning"]
    pipeline_yaml = PROJECT_ROOT / exp_config["pipeline_config"]
    overrides = {
        "classifier.device": args.device,
        "skip_stages": json.dumps(skip_stages),
    }
    config = load_config(str(pipeline_yaml), overrides=overrides)

    cache_dir = PROJECT_ROOT / data["cache_dir"]
    splits_dir = PROJECT_ROOT / data["splits_dir"]
    split_seed = data["split_seed"]

    classification_stage = ClassificationStage(config.classifier)
    split_name = f"split_seed{split_seed}"
    split_paths: dict[str, Path] = {}
    for rank in ("family", "genus", "species"):
        rank_split = splits_dir / rank / split_name
        if rank_split.with_suffix(".npz").exists():
            split_paths[rank] = rank_split

    logger.info("Loading multi-rank classifier...")
    classification_stage.load_from_multirank_cache(cache_dir, split_paths)

    # Create runner with all stages (skipped ones are no-ops)
    morphology_stage = MorphologyStage(config.vlm)
    trait_stage = TraitRetrievalStage(config.trait_retrieval)
    evidence_stage = EvidenceSynthesisStage(config.evidence_synthesis)
    reasoning_stage = ReasoningStage(config.reasoning)

    runner = PipelineRunner(config)
    runner.stages = [
        morphology_stage,
        classification_stage,
        trait_stage,
        evidence_stage,
        reasoning_stage,
    ]

    # Process specimens
    completed = 0
    total = len(individuals)
    batch_start = time.perf_counter()

    for i, (indiv_id, gt_family, image_paths) in enumerate(individuals, 1):
        out_path = output_dir / f"{indiv_id}.json"
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info("[%d/%d] SKIP %s", i, total, indiv_id)
            continue

        result = runner.run(specimen_id=indiv_id, image_paths=image_paths)
        result_dict = result.to_dict()

        # Extract top-1 prediction
        clf_data = result.get_stage_data("classification")
        preds = clf_data.get("predictions_by_rank", {}).get("family", [])
        predicted = preds[0].get("rank_value", "?") if preds else "?"

        result_dict["ground_truth"] = {"family": gt_family}
        result_dict["condition"] = "D"

        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        completed += 1
        status = "OK" if predicted == gt_family else "WRONG"
        logger.info("[%d/%d] %s %s -> %s (gt=%s)", i, total, status, indiv_id, predicted, gt_family)

    elapsed = time.perf_counter() - batch_start
    logger.info("Done. %d specimens in %.1fs", completed, elapsed)


if __name__ == "__main__":
    main()
