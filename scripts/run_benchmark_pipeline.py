#!/usr/bin/env python
"""Run the seedlearn pipeline on all STRI-overlapping specimens for benchmarking.

Initializes the PipelineRunner once, then loops over all catalog specimens
that overlap with the specified STRI trait matrix. Writes per-specimen JSON
outputs for downstream grading by ``scripts/grade_benchmark.py``.

Usage:
    # Quick test with 3 specimens
    python scripts/run_benchmark_pipeline.py \\
        --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_*.csv \\
        --cache-dir data/embeddings/2026-01-29_v2026-01-29_12K \\
        --split-path data/splits/2026-01-29_v2026-01-29_12K/family/split_seed42 \\
        --rag-index data/traits/latest/rag_index/ \\
        --limit 3

    # Resume after interruption
    python scripts/run_benchmark_pipeline.py \\
        --catalog ... --cache-dir ... --split-path ... --rag-index ... \\
        --output-dir data/benchmarks/2026-03-04/ --resume

    # Filter to specific species
    python scripts/run_benchmark_pipeline.py \\
        --catalog ... --cache-dir ... --split-path ... --rag-index ... \\
        --species-filter "Anacardium excelsum,Protium tenuifolium"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from seedlearn.benchmarking.overlap import (
    load_overlap_specimens,
    load_specimens_by_id,
)
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

DEFAULT_STRI_MATRIX = (
    "/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/traits/"
    "stri_web_keys/per_key_trait_matrices/"
    "cl185_complete_tree_species_of_panama_trait_matrix.csv"
)
DEFAULT_SYNONYM_TABLE = "configs/species_lists/inat_metadata_FINAL_NHN_01_2025.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (None = sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Batch-run the seedlearn pipeline for trait benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Data artifacts ---
    data_group = parser.add_argument_group("Data artifacts")
    data_group.add_argument(
        "--catalog", type=str, required=True,
        help="Path to species catalog CSV.",
    )
    data_group.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory containing cached BioCLIP 2 feature .npz files.",
    )
    data_group.add_argument(
        "--split-path", type=str, default=None,
        help="Path to split files (without extension).",
    )
    data_group.add_argument(
        "--rag-index", type=str, default=None,
        help="Directory containing pre-built FAISS RAG index.",
    )

    # --- Benchmark-specific ---
    bench_group = parser.add_argument_group("Benchmark options")
    bench_group.add_argument(
        "--stri-matrix", type=str, default=DEFAULT_STRI_MATRIX,
        help="Path to STRI trait matrix CSV (default: cl185).",
    )
    bench_group.add_argument(
        "--synonym-table", type=str, default=DEFAULT_SYNONYM_TABLE,
        help="Path to inat_metadata CSV with synonym columns.",
    )
    bench_group.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N specimens (for testing).",
    )
    bench_group.add_argument(
        "--species-filter", type=str, default=None,
        help="Comma-separated species names to process (case-insensitive).",
    )
    bench_group.add_argument(
        "--specimen-source", type=str, default=None,
        help="Select an explicit specimen set directly from the catalog "
             "(curator key CSV with an individual_code column, or a one-id-per-line "
             "file). Bypasses STRI overlap so every listed specimen is run.",
    )
    bench_group.add_argument(
        "--resume", action="store_true", default=False,
        help="Skip specimens whose output JSON already exists.",
    )
    bench_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for per-specimen JSON outputs (default: data/benchmarks/<date>/).",
    )

    # --- Config ---
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", type=str, default=None,
        help="Path to pipeline YAML config file.",
    )
    config_group.add_argument(
        "--skip", nargs="+", type=str, default=None,
        help="Stage names to skip (e.g. trait_retrieval reasoning).",
    )

    # --- VLM overrides ---
    vlm_group = parser.add_argument_group("VLM overrides (Stage 1)")
    vlm_group.add_argument("--vlm-model", type=str, default=None)
    vlm_group.add_argument("--vlm-endpoint", type=str, default=None)
    vlm_group.add_argument("--prompt-style", type=str, default=None)
    vlm_group.add_argument("--image-mode", type=str, default=None)
    vlm_group.add_argument(
        "--examples", type=str, default=None,
        help="Path to an in-context few-shot exemplar JSON (Stage-1 few-shot images).",
    )

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
    out_group.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Convert CLI arguments to dot-notation config overrides.

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
        "examples": "vlm.examples_file",
        "rank": "classifier.rank",
        "k_shot": "classifier.k_shot",
        "top_k": "classifier.top_k",
        "device": "classifier.device",
        "reasoning_model": "reasoning.model",
        "reasoning_endpoint": "reasoning.endpoint",
    }

    overrides: dict[str, str] = {}
    for attr, dotkey in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[dotkey] = str(value)
    return overrides


def build_runner(args: argparse.Namespace) -> PipelineRunner:
    """Initialize PipelineRunner with all stages loaded.

    Mirrors the initialization in scripts/run_pipeline.py but separated
    for reuse in batch mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Configured PipelineRunner ready for .run() calls.
    """
    overrides = build_overrides(args)
    if args.skip:
        overrides["skip_stages"] = json.dumps(args.skip)
    config = load_config(args.config, overrides=overrides if overrides else None)
    skip_set = set(config.skip_stages)

    # Stage 1: VLM Morphology
    morphology_stage = MorphologyStage(config.vlm)

    # Stage 2: Classification
    classification_stage = ClassificationStage(config.classifier)
    if "classification" not in skip_set:
        if args.cache_dir and args.split_path:
            cache_dir = Path(args.cache_dir)
            multirank_cache = cache_dir / "features.npz"

            if multirank_cache.exists():
                logger.info("Detected multi-rank cache: %s", multirank_cache)
                split_base = Path(args.split_path)
                splits_root = split_base.parent.parent
                split_name = split_base.name

                split_paths: dict[str, Path] = {}
                for rank in ("family", "genus", "species"):
                    rank_split = splits_root / rank / split_name
                    if rank_split.with_suffix(".npz").exists():
                        split_paths[rank] = rank_split
                        logger.info("  Found %s split: %s", rank, rank_split)

                if split_paths:
                    classification_stage.load_from_multirank_cache(
                        cache_dir, split_paths
                    )
            else:
                classification_stage.load_from_cache(args.cache_dir, args.split_path)
        else:
            logger.warning(
                "Stage 2 requires --cache-dir and --split-path. "
                "Use --skip classification to skip."
            )

    # Stage 3: Trait Retrieval
    trait_stage = TraitRetrievalStage(config.trait_retrieval)
    if "trait_retrieval" not in skip_set:
        if args.rag_index:
            from seedlearn.pipeline.rag import RAGIndex

            logger.info("Loading RAG index from: %s", args.rag_index)
            trait_stage.rag_index = RAGIndex.load(args.rag_index)
        else:
            logger.warning(
                "Stage 3 requires --rag-index. Use --skip trait_retrieval to skip."
            )

    # Stage 4: Evidence Synthesis
    evidence_stage = EvidenceSynthesisStage(config.evidence_synthesis)

    # Stage 5: Reasoning
    reasoning_stage = ReasoningStage(config.reasoning)

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
    """Run the benchmark pipeline batch."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("results/benchmarks") / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select specimens: explicit id list (curator key) or STRI overlap.
    missing_specimens: list[str] = []
    if args.specimen_source:
        specimens, missing_specimens = load_specimens_by_id(
            catalog_path=Path(args.catalog),
            specimen_source=Path(args.specimen_source),
        )
        if missing_specimens:
            logger.warning(
                "%d requested specimens missing from catalog: %s",
                len(missing_specimens), missing_specimens,
            )
    else:
        synonym_path = Path(args.synonym_table) if args.synonym_table else None
        specimens = load_overlap_specimens(
            catalog_path=Path(args.catalog),
            stri_matrix_path=Path(args.stri_matrix),
            synonym_path=synonym_path,
        )

    # Apply filters
    if args.species_filter:
        filter_set = {s.strip().lower() for s in args.species_filter.split(",")}
        specimens = [
            s for s in specimens if s.scientific_name.lower() in filter_set
        ]
        logger.info("Filtered to %d specimens matching species filter", len(specimens))

    if args.limit:
        specimens = specimens[: args.limit]
        logger.info("Limited to first %d specimens", len(specimens))

    if not specimens:
        logger.error("No overlap specimens found. Check catalog and STRI matrix paths.")
        sys.exit(1)

    logger.info(
        "Processing %d specimens from %d unique species",
        len(specimens),
        len({s.scientific_name.lower() for s in specimens}),
    )

    # Resolve VLM config so the run records exactly which model + prompt produced
    # these traits (provenance for the grading report). Cheap: config only, no model load.
    _overrides = build_overrides(args)
    _resolved = load_config(args.config, overrides=_overrides if _overrides else None)

    # Save run metadata
    run_meta = {
        "catalog": args.catalog,
        "stri_matrix": args.stri_matrix,
        "synonym_table": args.synonym_table,
        "n_specimens": len(specimens),
        "n_species": len({s.scientific_name.lower() for s in specimens}),
        "limit": args.limit,
        "species_filter": args.species_filter,
        "specimen_source": args.specimen_source,
        "missing_specimens": missing_specimens,
        "skip_stages": args.skip,
        "prompt_style": _resolved.vlm.prompt_style,
        "model": _resolved.vlm.model,
        "vlm_endpoint": _resolved.vlm.endpoint,
        "examples_file": _resolved.vlm.examples_file,
        "started_at": datetime.now().isoformat(),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Build pipeline runner (expensive one-time init)
    logger.info("Initializing pipeline runner...")
    runner = build_runner(args)
    logger.info("Pipeline runner ready.")

    # Process specimens
    n_success = 0
    n_error = 0
    total_start = time.perf_counter()

    for i, specimen in enumerate(specimens, 1):
        out_path = output_dir / f"{specimen.specimen_id}.json"

        if args.resume and out_path.exists():
            logger.info(
                "[%d/%d] Skipping %s (already exists)",
                i, len(specimens), specimen.specimen_id,
            )
            n_success += 1
            continue

        logger.info(
            "[%d/%d] Processing %s (%s, %d images)",
            i,
            len(specimens),
            specimen.specimen_id,
            specimen.scientific_name,
            len(specimen.image_paths),
        )

        try:
            result = runner.run(
                specimen_id=specimen.specimen_id,
                image_paths=specimen.image_paths,
            )
            result_dict = result.to_dict()

            # Add benchmark metadata to the output
            result_dict["benchmark_metadata"] = {
                "scientific_name": specimen.scientific_name,
                "family": specimen.family,
                "genus": specimen.genus,
                "species_epithet": specimen.species_epithet,
                "stri_match_name": specimen.stri_match_name,
                "match_method": specimen.match_method,
            }

            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2)

            elapsed = result.total_elapsed_ms
            logger.info(
                "  -> Saved %s (%.1f ms)", out_path.name, elapsed
            )
            n_success += 1

        except Exception as exc:
            logger.error("  -> FAILED: %s", exc)
            error_dict = {
                "specimen_id": specimen.specimen_id,
                "error": str(exc),
                "benchmark_metadata": {
                    "scientific_name": specimen.scientific_name,
                    "family": specimen.family,
                    "genus": specimen.genus,
                    "species_epithet": specimen.species_epithet,
                    "stri_match_name": specimen.stri_match_name,
                    "match_method": specimen.match_method,
                },
            }
            with open(out_path, "w") as f:
                json.dump(error_dict, f, indent=2)
            n_error += 1

    total_elapsed = time.perf_counter() - total_start

    # Summary
    print(f"\nBenchmark pipeline complete:")
    print(f"  Output directory: {output_dir}")
    print(f"  Specimens: {n_success} success, {n_error} errors")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}m)")
    if n_success + n_error > 0:
        avg = total_elapsed / (n_success + n_error)
        print(f"  Avg per specimen: {avg:.1f}s")


if __name__ == "__main__":
    main()
