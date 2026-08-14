#!/usr/bin/env python
"""Adapt externally-produced (cloud) Stage-1 results into the ``model_run`` shape.

Collaborator runs (e.g. Kaili's GPT-5.4 / GPT-5.1 workshop results under
``workshop_pipeline/step_1_cloudbank/results/``) are stored in the
``run_vlm_stage1.py`` benchmark shape: a single JSON with an ``answers`` dict
keyed by ``Family_Genus_species_<SPECIMENID>`` whose values are the standard
numbered assessment-form text. The human-annotation grader
(``scripts/grade_human_annotations.py`` / ``experiment_compare.py``) instead reads
per-specimen JSON files shaped like the pipeline's ``model_run`` output —
``stages.morphology.data.traits`` keyed by ``specimen_id``.

This adapter bridges the two: it parses each specimen's answer text with the same
``FormParser`` the pipeline uses and writes one ``{specimen_id}.json`` per specimen,
plus a ``run_metadata.json`` recording provenance, so an ingested directory grades
exactly like a locally-produced condition — with no grader changes.

Usage::

    # Ingest one collaborator result file as a labeled condition
    python scripts/ingest_workshop_results.py \\
        --source workshop_pipeline/step_1_cloudbank/results/gpt-5.4/sys4_user1_results.json \\
        --label K1_gpt-5.4_all-traits \\
        --model gpt-5.4 --granularity all_traits \\
        --out-dir trait_grading/model_run/K1_gpt-5.4_all-traits

The source path may live in another worktree/branch — see ``docs/trait-experiments.md``
for materializing ``origin/main`` results the ``dev`` tree does not carry.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from seedlearn.components.analyzers.parsers import FormParser

logger = logging.getLogger(__name__)


def specimen_id_from_key(answer_key: str) -> str:
    """Extract the specimen id (curator ``individual_code``) from an answer key.

    Workshop answer keys are ``Family_Genus_species_<SPECIMENID>`` (e.g.
    ``Acanthaceae_Aphelandra_scabra_SRAPHEDE2``); the specimen id is the final
    underscore-delimited token.

    Args:
        answer_key: The ``answers`` dict key from a workshop result JSON.

    Returns:
        The specimen id, e.g. ``SRAPHEDE2``.
    """
    return answer_key.rsplit("_", 1)[-1]


def _answer_text(key: str, answers: dict, processed: dict) -> str | None:
    """Resolve a specimen's answer text, preferring ``answers`` then ``processed_results``."""
    text = answers.get(key)
    if isinstance(text, str) and text.strip():
        return text
    proc = processed.get(key)
    if isinstance(proc, dict):
        alt = proc.get("answer")
        if isinstance(alt, str) and alt.strip():
            return alt
    return None


def load_answers(source_json: Path) -> dict[str, str]:
    """Load the specimen -> answer-text map from a workshop result JSON.

    Args:
        source_json: Path to a ``*_results.json`` file in the benchmark shape.

    Returns:
        Mapping of answer key -> raw numbered-form answer text (only entries
        with usable text).

    Raises:
        ValueError: If the file has no ``answers`` (or usable ``processed_results``).
    """
    data = json.loads(source_json.read_text())
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(f"{source_json}: no 'answers' dict found")
    processed = data.get("processed_results") or {}
    processed = processed if isinstance(processed, dict) else {}
    out: dict[str, str] = {}
    for key in answers:
        text = _answer_text(key, answers, processed)
        if text is not None:
            out[key] = text
    if not out:
        raise ValueError(f"{source_json}: no usable answer text in 'answers'")
    return out


def load_merged_answers(sources: Sequence[Path]) -> dict[str, str]:
    """Load and merge specimen -> answer-text across one or more result files.

    A decomposed run asks the model in several calls — one per trait, or one per
    form section — and writes a separate result file per excerpt. Each file holds
    the same specimens but only its excerpt's numbered form lines. Because
    :class:`FormParser` keys each line on its trait *number* (not on the section
    header), concatenating a specimen's texts across the excerpt files
    reconstitutes that specimen's full form and parses exactly like a
    single-call run.

    Args:
        sources: Result JSONs to merge, in form order (their order only affects
            which file's free-text ``F. Notes`` survives parsing; notes are not
            a gradable trait).

    Returns:
        Mapping of answer key -> concatenated answer text.
    """
    merged: dict[str, list[str]] = {}
    for src in sources:
        for key, text in load_answers(src).items():
            merged.setdefault(key, []).append(text)
    return {key: "\n".join(parts) for key, parts in merged.items()}


def ingest(
    source_json: Path | Sequence[Path],
    out_dir: Path,
    *,
    label: str,
    model: str,
    granularity: str,
    prompt_style: str = "sys4",
    prompt_file: str | None = None,
) -> dict[str, Any]:
    """Convert workshop result JSON(s) into a ``model_run``-shaped directory.

    Args:
        source_json: The collaborator's ``*_results.json`` file, or several files
            from one decomposed run (per-trait / per-section) to merge into a
            single condition — see :func:`load_merged_answers`.
        out_dir: Destination directory for per-specimen JSONs + ``run_metadata.json``.
        label: Condition label recorded in provenance (e.g. ``K1_gpt-5.4_all-traits``).
        model: Prediction model name (e.g. ``gpt-5.4``).
        granularity: Prompt granularity (``all_traits`` / ``per_trait`` / ``per_section``).
        prompt_style: Base prompt style the source was excerpted from.
        prompt_file: Optional path to the exact as-run system prompt text, recorded in
            provenance so the report can show what the cloud model actually saw.

    Returns:
        The provenance dict written to ``run_metadata.json``.
    """
    sources = [source_json] if isinstance(source_json, Path) else list(source_json)
    answers = load_merged_answers(sources)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_ids: set[str] = set()
    for key, text in answers.items():
        specimen_id = specimen_id_from_key(key)
        if specimen_id in seen_ids:
            logger.warning("duplicate specimen id %s (key %s) — overwriting", specimen_id, key)
        seen_ids.add(specimen_id)
        traits = FormParser.parse(text)
        record = {
            "specimen_id": specimen_id,
            "source_answer_key": key,
            "stages": {"morphology": {"data": {"traits": traits}}},
        }
        (out_dir / f"{specimen_id}.json").write_text(json.dumps(record, indent=2))
        written += 1

    metadata = {
        "external": True,
        "label": label,
        "model": model,
        "granularity": granularity,
        "prompt_style": prompt_style,
        "source_json": str(sources[0]) if len(sources) == 1 else [str(s) for s in sources],
        "n_specimens": written,
    }
    if prompt_file:
        metadata["prompt_file"] = str(prompt_file)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("ingested %d specimens from %s -> %s", written, source_json, out_dir)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, action="append", dest="source",
                   help="Workshop *_results.json file. Repeat to merge the excerpt files "
                        "of one decomposed run (per-trait / per-section) into one condition.")
    p.add_argument("--out-dir", type=Path, required=True, help="Destination model_run dir.")
    p.add_argument("--label", type=str, required=True, help="Condition label (provenance).")
    p.add_argument("--model", type=str, required=True, help="Prediction model, e.g. gpt-5.4.")
    p.add_argument(
        "--granularity",
        type=str,
        default="all_traits",
        choices=["all_traits", "per_trait", "per_section"],
        help="Prompt granularity of the source run.",
    )
    p.add_argument("--prompt-style", type=str, default="sys4", help="Base prompt style excerpted from.")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="Path to the exact as-run system prompt text (recorded in provenance).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    missing = [s for s in args.source if not s.exists()]
    if missing:
        for s in missing:
            logger.error("source not found: %s", s)
        return 1
    meta = ingest(
        args.source,
        args.out_dir,
        label=args.label,
        model=args.model,
        granularity=args.granularity,
        prompt_style=args.prompt_style,
        prompt_file=args.prompt_file,
    )
    logger.info("wrote %d specimens + run_metadata.json to %s", meta["n_specimens"], args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
