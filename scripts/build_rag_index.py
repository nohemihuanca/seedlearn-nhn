#!/usr/bin/env python
"""Build a FAISS RAG index from botanical NLP trait descriptions.

Usage:
    python scripts/build_rag_index.py \\
        --descriptions data/traits/latest/concatenated_output_nlp.csv \\
        --output data/traits/latest/rag_index/

One-time offline step. The resulting index is loaded by the trait retrieval
pipeline stage at runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from seedlearn.pipeline.rag import RAGIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments, load CSV, build and save RAG index."""
    parser = argparse.ArgumentParser(
        description="Build FAISS RAG index from NLP trait descriptions.",
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        required=True,
        help="Path to concatenated NLP descriptions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save the RAG index.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--taxon-col",
        type=str,
        default="taxon",
        help="Column name for taxon identifier (default: 'taxon').",
    )
    parser.add_argument(
        "--rank-col",
        type=str,
        default="rank",
        help="Column name for taxonomic rank (default: 'rank').",
    )
    parser.add_argument(
        "--description-col",
        type=str,
        default="description",
        help="Column name for NLP description text (default: 'description').",
    )
    args = parser.parse_args()

    if not args.descriptions.exists():
        logger.error("File not found: %s", args.descriptions)
        sys.exit(1)

    logger.info("Reading descriptions from %s", args.descriptions)
    df = pd.read_csv(args.descriptions)

    required = {args.taxon_col, args.rank_col, args.description_col}
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "Missing columns: %s. Available: %s", missing, list(df.columns),
        )
        sys.exit(1)

    # Drop rows with empty descriptions
    df = df.dropna(subset=[args.description_col])
    df = df[df[args.description_col].str.strip().astype(bool)]

    descriptions = [
        {
            "taxon": str(row[args.taxon_col]),
            "rank": str(row[args.rank_col]),
            "description": str(row[args.description_col]),
        }
        for _, row in df.iterrows()
    ]

    logger.info(
        "Building index from %d descriptions with model '%s'",
        len(descriptions),
        args.model,
    )
    index = RAGIndex.build(descriptions, model_name=args.model)
    index.save(args.output)
    logger.info("Done. Index saved to %s", args.output)


if __name__ == "__main__":
    main()
