"""Load ground truth CSVs and benchmark result directories."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAIT_COLUMNS: list[str] = [
    "leaf_complexity",
    "leaf_arrangement",
    "leaf_margin",
    "stipules",
    "latex",
]


@dataclass
class GroundTruthEntry:
    """Ground truth for one specimen."""

    specimen_key: str
    specimen_id: str
    family: str
    scientific_name: str
    num_images: int
    traits: dict[str, str]
    match_types: dict[str, str]
    multi_label_count: int = 0


@dataclass
class ResultEntry:
    """VLM prediction result for one specimen (or one image)."""

    specimen_key: str
    image_id: str = ""
    traits: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""
    thinking: str = ""


def load_ground_truth(path: Path) -> dict[str, GroundTruthEntry]:
    """Load ground truth CSV into a dict keyed by specimen_key.

    Args:
        path: Path to the ground truth CSV file. Expected columns include
            specimen_key, specimen_id, family, scientific_name, num_images,
            and per-trait columns with corresponding ``_match_type`` suffixes.

    Returns:
        Dictionary mapping specimen_key to its ``GroundTruthEntry``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    entries: dict[str, GroundTruthEntry] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            traits: dict[str, str] = {}
            match_types: dict[str, str] = {}
            for trait in TRAIT_COLUMNS:
                val = row.get(trait, "")
                mt = row.get(f"{trait}_match_type", "exact")
                if val:
                    traits[trait] = val
                    match_types[trait] = mt

            entry = GroundTruthEntry(
                specimen_key=row["specimen_key"],
                specimen_id=row.get("specimen_id", ""),
                family=row.get("family", ""),
                scientific_name=row.get("scientific_name", ""),
                num_images=int(row.get("num_images", 0)),
                traits=traits,
                match_types=match_types,
                multi_label_count=int(row.get("multi_label_count", 0)),
            )
            entries[entry.specimen_key] = entry

    logger.info("Loaded %d ground truth entries from %s", len(entries), path)
    return entries


def _parse_result_json(path: Path) -> ResultEntry:
    """Parse a single result JSON file into a ResultEntry.

    Args:
        path: Path to a JSON file containing keys ``specimen_key``,
            ``image_id``, ``traits``, ``raw_response``, and ``thinking``.

    Returns:
        Populated ``ResultEntry``.
    """
    with open(path) as f:
        data = json.load(f)

    return ResultEntry(
        specimen_key=data.get("specimen_key", path.stem),
        image_id=data.get("image_id", ""),
        traits=data.get("traits", {}),
        raw_response=data.get("raw_response", ""),
        thinking=data.get("thinking", ""),
    )


def _parse_legacy_result_json(path: Path) -> dict[str, ResultEntry]:
    """Parse legacy ``run_vlm_stage1.py`` result format.

    The existing format stores answers as
    ``{"answers": {"specimen_key": "answer_text", ...}, ...}``.

    Args:
        path: Path to the legacy JSON results file.

    Returns:
        Dictionary mapping specimen_key to ``ResultEntry``.
    """
    from tests.benchmarks.run_vlm_stage1 import parse_answer

    # Mapping from legacy EXPECTED_KEYS to our trait column names
    key_to_trait = {
        "leaf complexity": "leaf_complexity",
        "leaf relative position": "leaf_arrangement",
        "leaf margin": "leaf_margin",
        "stipules, sometimes visible only in close-up images": "stipules",
        "latex, rarely visible but extremely diagnostic when present": "latex",
    }

    with open(path) as f:
        data = json.load(f)

    entries: dict[str, ResultEntry] = {}
    answers = data.get("answers", {})
    cots = data.get("cots", {})

    for specimen_key, answer_text in answers.items():
        parsed = parse_answer(answer_text)
        traits: dict[str, str] = {}
        for old_key, trait_col in key_to_trait.items():
            if old_key in parsed:
                traits[trait_col] = parsed[old_key]
        entries[specimen_key] = ResultEntry(
            specimen_key=specimen_key,
            traits=traits,
            raw_response=answer_text,
            thinking=cots.get(specimen_key, ""),
        )

    return entries


def load_result_dir(
    path: Path,
    mode: str = "multi",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a benchmark result directory.

    Args:
        path: Path to result directory.
        mode: ``"multi"`` (one prediction per specimen) or ``"single"``
            (one prediction per image).

    Returns:
        Tuple of ``(config_dict, results_dict)``.
        For multi: results maps ``specimen_key -> ResultEntry``.
        For single: results maps ``specimen_key -> list[ResultEntry]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result directory not found: {path}")

    config_path = path / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    results: dict[str, Any] = {}

    if mode == "multi":
        multi_dir = path / "multi"
        if multi_dir.exists():
            for rfile in sorted(multi_dir.glob("*.json")):
                entry = _parse_result_json(rfile)
                results[entry.specimen_key] = entry
        else:
            # Try legacy format
            for rfile in sorted(path.glob("*.json")):
                if rfile.name == "config.json":
                    continue
                legacy = _parse_legacy_result_json(rfile)
                results.update(legacy)
                break
    elif mode == "single":
        single_dir = path / "single"
        if single_dir.exists():
            for specimen_dir in sorted(single_dir.iterdir()):
                if not specimen_dir.is_dir():
                    continue
                specimen_results: list[ResultEntry] = []
                for img_file in sorted(specimen_dir.glob("img_*.json")):
                    entry = _parse_result_json(img_file)
                    specimen_results.append(entry)
                if specimen_results:
                    results[specimen_dir.name] = specimen_results
        else:
            logger.warning(
                "No single/ subdirectory in %s. "
                "Did you run inference with --mode single or --mode both?",
                path,
            )

    logger.info("Loaded %d results from %s (mode=%s)", len(results), path, mode)
    return config, results
