#!/usr/bin/env python
"""Run the seedlearn 5-stage classification pipeline.

Usage:
    # Explicit images
    python scripts/run_pipeline.py --images /path/to/img1.jpg /path/to/img2.jpg \\
        --cache-dir data/experiments/simpleshot/.../features \\
        --split-path data/experiments/simpleshot/.../splits/family/split_seed42 \\
        --rag-index data/traits/latest/rag_index/

    # Specimen lookup from catalog (resolves images automatically)
    python scripts/run_pipeline.py --specimen PP123 --catalog $CATALOG \\
        --cache-dir ... --split-path ... --rag-index ...

    # Skip VLM stages (CPU-only test of Stages 2-4)
    python scripts/run_pipeline.py --images /a.jpg --skip morphology reasoning \\
        --cache-dir ... --split-path ... --rag-index ... --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from seedlearn.pipeline.config import load_config
from seedlearn.pipeline.runner import PipelineRunner
from seedlearn.pipeline.stages import (
    ClassificationStage,
    EvidenceSynthesisStage,
    MorphologyStage,
    ReasoningStage,
    TraitRetrievalStage,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (None = sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the seedlearn 5-stage classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input ---
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--images",
        nargs="+",
        type=str,
        default=None,
        help="Image file paths for the specimen.",
    )
    input_group.add_argument(
        "--specimen",
        type=str,
        default=None,
        help="Specimen ID (used for output naming; images resolved from catalog if --images not given).",
    )
    input_group.add_argument(
        "--random",
        type=str,
        choices=["test", "val"],
        default=None,
        metavar="PARTITION",
        help="Sample a random individual from the test or val split.",
    )
    input_group.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for --random selection (printed in output for reproducibility).",
    )

    # --- Data artifacts ---
    data_group = parser.add_argument_group("Data artifacts")
    data_group.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to species catalog CSV (required for --specimen lookup).",
    )
    data_group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory containing cached BioCLIP 2 feature .npz files.",
    )
    data_group.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to split files (without extension, e.g. .../split_seed42).",
    )
    data_group.add_argument(
        "--rag-index",
        type=str,
        default=None,
        help="Directory containing pre-built FAISS RAG index.",
    )

    # --- Config ---
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline YAML config file.",
    )
    config_group.add_argument(
        "--skip",
        nargs="+",
        type=str,
        default=None,
        help="Stage names to skip (e.g. trait_retrieval reasoning).",
    )

    # --- VLM overrides ---
    vlm_group = parser.add_argument_group("VLM overrides (Stage 1)")
    vlm_group.add_argument("--vlm-model", type=str, default=None)
    vlm_group.add_argument("--vlm-endpoint", type=str, default=None)
    vlm_group.add_argument("--prompt-style", type=str, default=None)
    vlm_group.add_argument("--image-mode", type=str, default=None)

    # --- Classifier overrides ---
    clf_group = parser.add_argument_group("Classifier overrides (Stage 2)")
    clf_group.add_argument("--rank", type=str, default=None)
    clf_group.add_argument("--k-shot", type=int, default=None)
    clf_group.add_argument("--top-k", type=int, default=None)
    clf_group.add_argument("--device", type=str, default=None)

    # --- Reasoning overrides ---
    rea_group = parser.add_argument_group("Reasoning overrides (Stage 5)")
    rea_group.add_argument("--reasoning-model", type=str, default=None)
    rea_group.add_argument("--reasoning-endpoint", type=str, default=None)

    # --- Output ---
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default=None)
    out_group.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate HTML report alongside JSON output.",
    )
    out_group.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(argv)

    if not args.images and not args.specimen and not args.random:
        parser.error("At least one of --images, --specimen, or --random is required.")

    if args.random and (args.images or args.specimen):
        parser.error("--random is mutually exclusive with --images and --specimen.")

    if args.random and not (args.split_path and args.catalog):
        parser.error("--random requires both --split-path and --catalog.")

    return args


def build_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Convert CLI arguments to dot-notation config overrides.

    Only includes overrides for arguments that were explicitly provided.

    Args:
        args: Parsed argument namespace.

    Returns:
        Dictionary of dot-notation key -> string value.
    """
    mapping = {
        "vlm_model": "vlm.model",
        "vlm_endpoint": "vlm.endpoint",
        "prompt_style": "vlm.prompt_style",
        "image_mode": "vlm.image_mode",
        "rank": "classifier.rank",
        "k_shot": "classifier.k_shot",
        "top_k": "classifier.top_k",
        "device": "classifier.device",
        "reasoning_model": "reasoning.model",
        "reasoning_endpoint": "reasoning.endpoint",
        "output_dir": "output.directory",
    }

    overrides: dict[str, str] = {}
    for attr, dotkey in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[dotkey] = str(value)

    return overrides


def resolve_specimen_images(
    specimen_id: str,
    catalog_path: str,
) -> list[str]:
    """Look up all image paths for a specimen from the catalog.

    Args:
        specimen_id: Individual plant ID (ID_YPS column value).
        catalog_path: Path to the species catalog CSV.

    Returns:
        List of image path strings for the specimen.

    Raises:
        SystemExit: If the catalog cannot be loaded or the specimen is not found.
    """
    from seedlearn.data.catalog import load_dataset

    records, _ = load_dataset(Path(catalog_path))
    matches = [str(r.image_path) for r in records if r.individual_id == specimen_id]
    if not matches:
        logger.error(
            "Specimen '%s' not found in catalog. Check ID_YPS values in %s",
            specimen_id,
            catalog_path,
        )
        sys.exit(1)
    return matches


def main() -> None:
    """Run the pipeline from CLI arguments."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Load config with overrides
    overrides = build_overrides(args)
    if args.skip:
        overrides["skip_stages"] = json.dumps(args.skip)
    config = load_config(args.config, overrides=overrides if overrides else None)

    skip_set = set(config.skip_stages)

    # --- Resolve random specimen if requested ---
    if args.random:
        from seedlearn.data.catalog import load_dataset
        from seedlearn.data.splits import load_split, sample_individual_from_split

        logger.info("Sampling random %s individual...", args.random)
        rand_records, _ = load_dataset(
            Path(args.catalog),
            rank=config.classifier.rank,
        )
        rand_split = load_split(Path(args.split_path))
        individual_id, used_seed = sample_individual_from_split(
            rand_records,
            rand_split,
            partition=args.random,
            seed=args.random_seed,
        )
        indiv_record = next(r for r in rand_records if r.individual_id == individual_id)
        logger.info(
            "Selected individual: %s (%s, seed: %d)",
            individual_id,
            indiv_record.label,
            used_seed,
        )
        logger.info(
            "  Reproduce with: --specimen %s  (or --random-seed %d)",
            individual_id,
            used_seed,
        )
        args.specimen = individual_id

    # --- Build stages and initialize with data artifacts ---

    # Stage 1: VLM Morphology
    morphology_stage = MorphologyStage(config.vlm, prompt_file=config.prompts.morphology)

    # Stage 2: Classification (requires cached features + split)
    classification_stage = ClassificationStage(config.classifier)
    if "classification" not in skip_set:
        if args.cache_dir and args.split_path:
            cache_dir = Path(args.cache_dir)
            multirank_cache = cache_dir / "features.npz"

            if multirank_cache.exists():
                # Multi-rank v2 cache: auto-discover rank splits
                logger.info("Detected multi-rank cache: %s", multirank_cache)
                split_base = Path(args.split_path)
                # args.split_path is like .../splits/family/split_seed42
                # Discover sibling rank directories
                splits_root = split_base.parent.parent  # .../splits/
                split_name = split_base.name  # split_seed42

                split_paths: dict[str, Path] = {}
                for rank in ("family", "genus", "species"):
                    rank_split = splits_root / rank / split_name
                    # load_split expects path without extension
                    if (rank_split.with_suffix(".npz")).exists():
                        split_paths[rank] = rank_split
                        logger.info("  Found %s split: %s", rank, rank_split)
                    else:
                        logger.info(
                            "  No %s split found (looked for %s.npz)", rank, rank_split
                        )

                if split_paths:
                    classification_stage.load_from_multirank_cache(
                        cache_dir, split_paths
                    )
                else:
                    logger.warning(
                        "Multi-rank cache found but no rank splits discovered."
                    )
            else:
                # Legacy single-rank cache
                logger.info("Fitting classifier from cache: %s", args.cache_dir)
                classification_stage.load_from_cache(args.cache_dir, args.split_path)
        else:
            logger.warning(
                "Stage 2 (classification) requires --cache-dir and --split-path. "
                "It will fail at runtime without them. Use --skip classification to skip."
            )

    # Stage 3: Trait Retrieval (requires RAG index)
    trait_stage = TraitRetrievalStage(
        config.trait_retrieval, prompt_file=config.prompts.rag_query
    )
    if "trait_retrieval" not in skip_set:
        if args.rag_index:
            from seedlearn.pipeline.rag import RAGIndex

            logger.info("Loading RAG index from: %s", args.rag_index)
            trait_stage.rag_index = RAGIndex.load(args.rag_index)
        else:
            logger.warning(
                "Stage 3 (trait_retrieval) requires --rag-index. "
                "It will fail at runtime without it. Use --skip trait_retrieval to skip."
            )

    # Stage 4: Evidence Synthesis (deterministic, no data needed)
    evidence_stage = EvidenceSynthesisStage(config.evidence_synthesis)

    # Stage 5: Reasoning
    reasoning_stage = ReasoningStage(
        config.reasoning, prompt_file=config.prompts.reasoning
    )

    # Wire stages into runner
    runner = PipelineRunner(config)
    runner.stages = [
        morphology_stage,
        classification_stage,
        trait_stage,
        evidence_stage,
        reasoning_stage,
    ]

    # --- Resolve input images ---

    image_paths = args.images or []
    specimen_id = args.specimen or "specimen"

    if not image_paths and args.specimen:
        if not args.catalog:
            logger.error("--specimen requires --catalog to look up image paths.")
            sys.exit(1)
        logger.info("Resolving images for specimen '%s' from catalog", args.specimen)
        image_paths = resolve_specimen_images(args.specimen, args.catalog)
        logger.info(
            "Found %d images for specimen '%s'", len(image_paths), args.specimen
        )

    if not image_paths:
        logger.error("No images provided. Use --images or --specimen with --catalog.")
        sys.exit(1)

    logger.info(
        "Running pipeline for specimen '%s' with %d images",
        specimen_id,
        len(image_paths),
    )

    # --- Execute ---

    result = runner.run(specimen_id=specimen_id, image_paths=image_paths)

    # --- Output ---

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{specimen_id}.json"

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Total elapsed: %.1f ms", result.total_elapsed_ms)

    if args.report:
        from seedlearn.reporting.pipeline_html import generate_pipeline_report

        html_path = output_path.with_suffix(".html")
        html = generate_pipeline_report(result.to_dict())
        html_path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", html_path)

    # Print summary
    for name, sr in result.stage_results.items():
        status = "SKIP" if sr.skipped else ("ERROR" if sr.error else "OK")
        print(f"  {name}: {status} ({sr.elapsed_ms:.1f} ms)")


if __name__ == "__main__":
    main()
