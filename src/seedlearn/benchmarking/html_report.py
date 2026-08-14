"""Generate browsable HTML report for benchmark grading results.

Produces a self-contained HTML file with per-specimen cards showing images,
VLM predictions vs STRI ground truth, and color-coded verdicts.
"""

from __future__ import annotations

import base64
import html as html_module
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from seedlearn.benchmarking.trait_mapping import (
    TRAIT_RULES,
    get_raw_vlm_values,
    map_prediction,
)

logger = logging.getLogger(__name__)

# Trait categories to display, in order
DISPLAY_CATEGORIES = [
    "leaf_arrangement",
    "leaf_type",
    "leaf_margin",
    "stipules",
    "latex",
]

# Human-readable category labels
CATEGORY_LABELS = {
    "leaf_arrangement": "Leaf Arrangement",
    "leaf_type": "Leaf Type",
    "leaf_margin": "Leaf Margin",
    "stipules": "Stipules",
    "latex": "Latex",
}


def _encode_image_thumbnail(path: str, max_size: int = 400) -> str | None:
    """Read an image and return a base64 data URI for embedding in HTML.

    Args:
        path: Absolute path to image file.
        max_size: Maximum thumbnail dimension in pixels.

    Returns:
        Data URI string, or None if the image cannot be loaded.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(path)
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _get_stri_ground_truth(
    stri_row: pd.Series,
    column_suffix: str = "",
) -> dict[str, dict[str, Any]]:
    """Extract ground truth values organized by category.

    Args:
        stri_row: STRI matrix row for this species.
        column_suffix: Column suffix (e.g., "__consensus").

    Returns:
        Dict of category -> {option: value, uncoded: bool}.
    """
    result: dict[str, dict[str, Any]] = {}

    for cat in DISPLAY_CATEGORIES:
        cat_rules = [r for r in TRAIT_RULES if r.category == cat]
        options: dict[str, float | None] = {}

        for rule in cat_rules:
            col = rule.stri_column + column_suffix
            if col in stri_row.index:
                val = stri_row[col]
                if isinstance(val, float) and math.isnan(val):
                    options[rule.stri_column] = None
                else:
                    options[rule.stri_column] = float(val)
            else:
                options[rule.stri_column] = None

        # Check uncoded
        uncoded_col = f"{cat}__uncoded{column_suffix}"
        uncoded = False
        if uncoded_col in stri_row.index:
            uv = stri_row[uncoded_col]
            if not (isinstance(uv, float) and math.isnan(uv)):
                uncoded = bool(int(uv) == 1)

        # Also uncoded if all values are None
        if all(v is None for v in options.values()):
            uncoded = True

        result[cat] = {"options": options, "uncoded": uncoded}

    return result


def _verdict_class(predicted: int | None, gt: float | None, uncoded: bool) -> str:
    """Return CSS class for a trait cell.

    Args:
        predicted: Pipeline binary prediction.
        gt: Ground truth binary value.
        uncoded: Whether the category is uncoded.

    Returns:
        CSS class name.
    """
    if uncoded or gt is None:
        return "uncoded"
    if predicted is None:
        return "no-pred"
    if predicted == 1 and gt == 1.0:
        return "correct"
    if predicted == 1 and gt == 0.0:
        return "incorrect"
    return "neutral"


def generate_benchmark_html(
    results_dir: Path,
    stri_path: Path,
    synonym_path: Path | None = None,
    column_suffix: str = "",
    max_thumbnails: int = 3,
) -> str:
    """Generate a self-contained HTML report for benchmark results.

    Args:
        results_dir: Directory containing per-specimen JSON files.
        stri_path: Path to STRI trait matrix CSV.
        synonym_path: Path to synonym table (for STRI name resolution).
        column_suffix: Column suffix for STRI columns.
        max_thumbnails: Maximum number of image thumbnails per specimen.

    Returns:
        HTML string.
    """
    stri_df = pd.read_csv(stri_path, dtype={"taxon_id": str})

    # Build name index
    stri_name_index: dict[str, int] = {}
    for idx, name in enumerate(stri_df["scientific_name"].values):
        if isinstance(name, str):
            stri_name_index[name.strip().lower()] = idx

    # Build synonym lookup if provided
    synonym_lookup: dict[str, list[str]] = {}
    if synonym_path and synonym_path.exists():
        syn_df = pd.read_csv(synonym_path, dtype=str).fillna("")
        for _, row in syn_df.drop_duplicates(subset=["accepted_name"]).iterrows():
            accepted = row.get("accepted_name", "").strip().lower()
            if not accepted:
                continue
            syns: list[str] = []
            raw = row.get("synonyms", "").strip()
            if raw and raw.upper() != "NA":
                for s in raw.split(","):
                    s = s.strip().lower()
                    if s and s != accepted:
                        syns.append(s)
            sf = row.get("scientific_name_final2", "").strip().lower()
            if sf and sf != accepted and sf not in syns:
                syns.append(sf)
            if syns:
                synonym_lookup[accepted] = syns

    def find_stri_row(name: str) -> pd.Series | None:
        key = name.lower()
        if key in stri_name_index:
            return stri_df.iloc[stri_name_index[key]]
        for syn in synonym_lookup.get(key, []):
            if syn in stri_name_index:
                return stri_df.iloc[stri_name_index[syn]]
        return None

    # Load all results
    json_files = sorted(
        f for f in results_dir.glob("*.json") if f.name != "run_metadata.json"
    )

    specimens: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "by_category": {cat: {"correct": 0, "incorrect": 0, "skipped": 0} for cat in DISPLAY_CATEGORIES},
    }
    # Per-STRI-column support tracking: count of graded specimens with gt=1
    column_support: dict[str, int] = {r.stri_column: 0 for r in TRAIT_RULES}
    # Per-STRI-column TP/FP tracking
    column_tp: dict[str, int] = {r.stri_column: 0 for r in TRAIT_RULES}
    column_fp: dict[str, int] = {r.stri_column: 0 for r in TRAIT_RULES}
    # Total graded specimens (those with at least one correct/incorrect verdict)
    graded_specimen_ids: set[str] = set()
    # Per-species tracking: species -> category -> list of verdicts
    species_verdicts: dict[str, dict[str, list[str]]] = {}
    # Per-species tracking: species -> category -> list of predicted options
    species_predictions: dict[str, dict[str, list[str | None]]] = {}

    for jf in json_files:
        with open(jf) as f:
            result = json.load(f)

        if "stages" not in result:
            continue

        meta = result.get("benchmark_metadata", {})
        specimen_id = result.get("specimen_id", jf.stem)
        scientific_name = meta.get("scientific_name", "")
        family = meta.get("family", "")
        stri_match_name = meta.get("stri_match_name", scientific_name)

        morphology = result.get("stages", {}).get("morphology", {})
        traits = morphology.get("data", {}).get("traits", {})
        raw_response = morphology.get("data", {}).get("raw_response", "")

        # Stage 5 species ID
        reasoning = result.get("stages", {}).get("reasoning", {})
        classification = reasoning.get("data", {}).get("classification", {})
        pred_species = classification.get("predicted_species", "")
        pred_confidence = classification.get("confidence", "")

        # Get STRI row
        stri_row = find_stri_row(scientific_name)
        if stri_row is None:
            stri_row = find_stri_row(stri_match_name)

        # Map predictions
        predictions = map_prediction(traits, column_suffix=column_suffix)
        vlm_raw = get_raw_vlm_values(traits)

        # Get ground truth
        gt_data: dict[str, dict[str, Any]] = {}
        if stri_row is not None:
            gt_data = _get_stri_ground_truth(stri_row, column_suffix)

        # Build per-category grading
        categories_html: list[dict[str, Any]] = []
        for cat in DISPLAY_CATEGORIES:
            gt_cat = gt_data.get(cat, {"options": {}, "uncoded": True})
            uncoded = gt_cat["uncoded"]
            options = gt_cat["options"]

            vlm_value = vlm_raw.get(cat, "")

            # Which option did the pipeline predict?
            pred_col = None
            pred_match_rule = ""
            for rule in TRAIT_RULES:
                if rule.category == cat:
                    col = rule.stri_column + column_suffix
                    if predictions.get(col) == 1:
                        pred_col = rule.stri_column
                        pred_match_rule = " | ".join(sorted(rule.match_values))
                        break

            # Ground truth active options
            gt_active = [
                col.replace(f"{cat}__", "")
                for col, val in options.items()
                if val is not None and val == 1.0
            ]

            # Determine verdict
            if uncoded or stri_row is None:
                verdict = "skipped"
                stats["by_category"][cat]["skipped"] += 1
            elif pred_col is None and "not observed" in vlm_value.lower():
                verdict = "not_observed"
                stats["by_category"][cat]["skipped"] += 1
            elif pred_col is None:
                verdict = "no_pred"
                stats["by_category"][cat]["skipped"] += 1
            elif options.get(pred_col, 0) == 1.0:
                verdict = "correct"
                stats["by_category"][cat]["correct"] += 1
            else:
                verdict = "incorrect"
                stats["by_category"][cat]["incorrect"] += 1

            # Track per-column F1 stats (only for graded specimens)
            if verdict in ("correct", "incorrect") and pred_col is not None:
                graded_specimen_ids.add(specimen_id)
                gt_val = options.get(pred_col, 0)
                if gt_val == 1.0:
                    column_tp[pred_col] += 1
                else:
                    column_fp[pred_col] += 1
            # Track support: for each option column with gt=1 in graded specimens
            if verdict in ("correct", "incorrect") and not uncoded:
                for opt_col, opt_val in options.items():
                    if opt_val is not None and opt_val == 1.0:
                        column_support[opt_col] += 1

            # Track per-species stats
            sp_key = scientific_name
            if sp_key not in species_verdicts:
                species_verdicts[sp_key] = {c: [] for c in DISPLAY_CATEGORIES}
                species_predictions[sp_key] = {c: [] for c in DISPLAY_CATEGORIES}
            species_verdicts[sp_key][cat].append(verdict)
            species_predictions[sp_key][cat].append(pred_col)

            categories_html.append({
                "category": cat,
                "label": CATEGORY_LABELS.get(cat, cat),
                "vlm_value": vlm_value,
                "pred_option": pred_col.replace(f"{cat}__", "") if pred_col else "",
                "match_rule": pred_match_rule,
                "gt_active": gt_active,
                "uncoded": uncoded,
                "verdict": verdict,
            })

        # Image thumbnails
        image_paths = result.get("image_paths", [])
        thumbnails: list[str] = []
        for ip in image_paths[:max_thumbnails]:
            uri = _encode_image_thumbnail(ip)
            if uri:
                thumbnails.append(uri)

        specimens.append({
            "specimen_id": specimen_id,
            "scientific_name": scientific_name,
            "family": family,
            "pred_species": pred_species,
            "pred_confidence": pred_confidence,
            "categories": categories_html,
            "thumbnails": thumbnails,
            "n_images": len(image_paths),
            "raw_response": raw_response,
        })
        stats["total"] += 1

    # Import compute_binary_metrics for F1 calculation
    from seedlearn.benchmarking.report import compute_binary_metrics

    n_total_graded = len(graded_specimen_ids)

    # Compute accuracy summary with F1
    summary_rows: list[str] = []
    for cat in DISPLAY_CATEGORIES:
        s = stats["by_category"][cat]
        n_graded = s["correct"] + s["incorrect"]
        acc = s["correct"] / n_graded if n_graded > 0 else None
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"

        # Compute macro F1 for this category: mean of per-option F1s
        cat_rules = [r for r in TRAIT_RULES if r.category == cat]
        option_f1s: list[float] = []
        for rule in cat_rules:
            col = rule.stri_column
            metrics = compute_binary_metrics(
                tp=column_tp[col],
                fp=column_fp[col],
                support=column_support[col],
                n_graded_specimens=n_total_graded,
            )
            if metrics["f1"] is not None:
                option_f1s.append(metrics["f1"])
        macro_f1 = sum(option_f1s) / len(option_f1s) if option_f1s else None
        f1_str = f"{macro_f1:.3f}" if macro_f1 is not None else "N/A"

        # Per-option GT distribution: show % of graded specimens with gt=1
        gt_parts: list[str] = []
        for rule in cat_rules:
            option_label = rule.stri_column.replace(f"{cat}__", "")
            sup = column_support[rule.stri_column]
            if n_total_graded > 0:
                pct = sup / n_total_graded
                gt_parts.append(f"{option_label}: {pct:.0%}")
            else:
                gt_parts.append(f"{option_label}: N/A")
        gt_dist_str = ", ".join(gt_parts) if gt_parts else "N/A"

        summary_rows.append(
            f'<tr><td>{CATEGORY_LABELS.get(cat, cat)}</td>'
            f'<td class="num">{acc_str}</td>'
            f'<td class="num">{f1_str}</td>'
            f'<td class="gt-dist">{gt_dist_str}</td>'
            f'<td class="num">{s["correct"]}</td>'
            f'<td class="num">{s["incorrect"]}</td>'
            f'<td class="num">{s["skipped"]}</td></tr>'
        )

    # Build per-species accuracy table
    species_accuracy_rows: list[str] = []
    for sp in sorted(species_verdicts):
        cells = f'<td class="species-cell">{sp}</td>'
        for cat in DISPLAY_CATEGORIES:
            v_list = species_verdicts[sp][cat]
            n_correct = v_list.count("correct")
            n_incorrect = v_list.count("incorrect")
            n_graded = n_correct + n_incorrect
            if n_graded > 0:
                acc = n_correct / n_graded
                cls = "acc-high" if acc >= 0.8 else ("acc-mid" if acc >= 0.5 else "acc-low")
                cells += f'<td class="num {cls}">{acc:.0%} <span class="sub">({n_correct}/{n_graded})</span></td>'
            else:
                cells += '<td class="num acc-na">N/A</td>'
        species_accuracy_rows.append(f'<tr>{cells}</tr>')

    # Build within-species agreement table
    species_agreement_rows: list[str] = []
    for sp in sorted(species_predictions):
        cells = f'<td class="species-cell">{sp}</td>'
        for cat in DISPLAY_CATEGORIES:
            p_list = species_predictions[sp][cat]
            with_pred = [p for p in p_list if p is not None]
            n_total = len(p_list)
            n_with = len(with_pred)
            if n_with > 0:
                counts = Counter(with_pred)
                mode_val, mode_count = counts.most_common(1)[0]
                agreement = mode_count / n_with
                cls = "acc-high" if agreement >= 0.8 else ("acc-mid" if agreement >= 0.5 else "acc-low")
                cells += f'<td class="num {cls}">{agreement:.0%} <span class="sub">({mode_count}/{n_with})</span></td>'
            else:
                cells += '<td class="num acc-na">N/A</td>'
        species_agreement_rows.append(f'<tr>{cells}</tr>')

    # Build specimen cards
    cards_html: list[str] = []
    for spec in specimens:
        # Thumbnails
        imgs = "".join(
            f'<img src="{t}" class="thumb" />' for t in spec["thumbnails"]
        )
        if spec["n_images"] > len(spec["thumbnails"]):
            imgs += f'<span class="more-imgs">+{spec["n_images"] - len(spec["thumbnails"])} more</span>'

        # Category rows
        cat_rows: list[str] = []
        for c in spec["categories"]:
            verdict_cls = c["verdict"]
            gt_str = ", ".join(c["gt_active"]) if c["gt_active"] else ("uncoded" if c["uncoded"] else "none")
            pred_str = c["pred_option"] if c["pred_option"] else ("—" if c["vlm_value"] else "no output")
            vlm_str = c["vlm_value"] if c["vlm_value"] else "—"
            rule_str = c.get("match_rule", "")

            cat_rows.append(
                f'<tr class="{verdict_cls}">'
                f'<td>{c["label"]}</td>'
                f'<td class="vlm-val">{vlm_str}</td>'
                f'<td class="match-rule">{rule_str}</td>'
                f'<td class="pred-val">{pred_str}</td>'
                f'<td class="gt-val">{gt_str}</td>'
                f'<td class="verdict-cell">{verdict_cls.upper()}</td>'
                f'</tr>'
            )

        # Overall specimen verdict
        verdicts = [c["verdict"] for c in spec["categories"]]
        has_incorrect = "incorrect" in verdicts
        all_correct = all(v in ("correct", "skipped", "no_pred") for v in verdicts)
        card_class = "card-incorrect" if has_incorrect else ("card-correct" if all_correct else "card-mixed")

        cards_html.append(f'''
        <div class="card {card_class}" data-verdicts="{' '.join(verdicts)}"
             data-species="{spec['scientific_name'].lower()}"
             data-family="{spec['family'].lower()}">
            <div class="card-header">
                <h3>{spec['specimen_id']}</h3>
                <span class="species-name">{spec['scientific_name']}</span>
                <span class="family-name">({spec['family']})</span>
                {f'<span class="pred-id">Stage 5: {spec["pred_species"]} [{spec["pred_confidence"]}]</span>' if spec["pred_species"] else ''}
            </div>
            <div class="card-body">
                <div class="images">{imgs}</div>
                <table class="traits-table">
                    <thead>
                        <tr><th>Category</th><th>VLM Raw</th><th>Match Rule</th><th>Mapped</th><th>Ground Truth</th><th>Verdict</th></tr>
                    </thead>
                    <tbody>{''.join(cat_rows)}</tbody>
                </table>
                {f'<details class="vlm-response"><summary>VLM Raw Response</summary><pre>{html_module.escape(spec["raw_response"])}</pre></details>' if spec.get("raw_response") else ''}
            </div>
        </div>
        ''')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Grading Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f5; color: #333; padding: 20px; }}
h1 {{ margin-bottom: 10px; }}
.subtitle {{ color: #666; margin-bottom: 20px; }}

/* Summary */
.summary {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.summary table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
.summary th, .summary td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
.summary .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.summary .gt-dist {{ font-size: 0.85em; color: #555; white-space: nowrap; }}

/* Filters */
.filters {{ background: white; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: flex; gap: 15px; align-items: center;
            flex-wrap: wrap; }}
.filters label {{ font-weight: 600; font-size: 0.9em; }}
.filters select, .filters input {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; }}
.filters input {{ width: 250px; }}

/* Cards */
.card {{ background: white; border-radius: 8px; margin-bottom: 12px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #ccc; overflow: hidden; }}
.card-correct {{ border-left-color: #22c55e; }}
.card-incorrect {{ border-left-color: #ef4444; }}
.card-mixed {{ border-left-color: #f59e0b; }}
.card.hidden {{ display: none; }}

.card-header {{ padding: 12px 16px; display: flex; align-items: center; gap: 12px;
                cursor: pointer; flex-wrap: wrap; }}
.card-header h3 {{ font-size: 1em; font-family: monospace; }}
.species-name {{ font-style: italic; font-weight: 600; }}
.family-name {{ color: #666; font-size: 0.9em; }}
.pred-id {{ font-size: 0.85em; color: #666; margin-left: auto; }}

.card-body {{ padding: 0 16px 16px; }}
.images {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
.thumb {{ width: 300px; height: 300px; object-fit: cover; border-radius: 4px; border: 1px solid #eee; }}
.more-imgs {{ color: #888; font-size: 0.9em; }}

.traits-table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
.traits-table th {{ text-align: left; padding: 6px 10px; background: #f8f8f8;
                    border-bottom: 2px solid #ddd; }}
.traits-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
.vlm-val {{ font-family: monospace; color: #555; }}
.match-rule {{ font-family: monospace; color: #888; font-size: 0.85em; }}
.pred-val {{ font-weight: 600; }}
.gt-val {{ font-weight: 600; }}
.verdict-cell {{ font-size: 0.8em; font-weight: 700; }}

/* Collapsible VLM response */
.vlm-response {{ margin-top: 12px; }}
.vlm-response summary {{ cursor: pointer; font-weight: 600; font-size: 0.9em; color: #555;
                          padding: 6px 0; }}
.vlm-response pre {{ background: #f8f8f8; border: 1px solid #e5e5e5; border-radius: 4px;
                     padding: 12px; font-size: 0.85em; white-space: pre-wrap;
                     word-wrap: break-word; max-height: 400px; overflow-y: auto;
                     margin-top: 6px; }}

tr.correct {{ background: #f0fdf4; }}
tr.correct .verdict-cell {{ color: #16a34a; }}
tr.correct .pred-val {{ color: #16a34a; }}

tr.incorrect {{ background: #fef2f2; }}
tr.incorrect .verdict-cell {{ color: #dc2626; }}
tr.incorrect .pred-val {{ color: #dc2626; }}

tr.skipped {{ background: #f8f8f8; color: #999; }}
tr.skipped .verdict-cell {{ color: #999; }}

tr.no_pred {{ background: #fffbeb; }}
tr.no_pred .verdict-cell {{ color: #d97706; }}

tr.not_observed {{ background: #fffbeb; }}
tr.not_observed .verdict-cell {{ color: #d97706; }}

/* Per-species tables */
.species-table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
.species-table th {{ padding: 8px 12px; text-align: right; background: #f8f8f8;
                     border-bottom: 2px solid #ddd; }}
.species-table th:first-child {{ text-align: left; }}
.species-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
.species-cell {{ font-style: italic; font-weight: 600; text-align: left !important; white-space: nowrap; }}
.acc-high {{ color: #16a34a; }}
.acc-mid {{ color: #d97706; }}
.acc-low {{ color: #dc2626; }}
.acc-na {{ color: #999; }}
.sub {{ font-size: 0.8em; color: #888; font-weight: normal; }}

.count-display {{ background: white; border-radius: 8px; padding: 10px 20px;
                  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                  font-size: 0.9em; color: #666; }}
</style>
</head>
<body>

<h1>Benchmark Grading Report</h1>
<p class="subtitle">{stats['total']} specimens graded against STRI trait matrix</p>

<div class="summary">
<h2>Accuracy Summary</h2>
<table>
<thead><tr><th>Category</th><th class="num">Accuracy</th><th class="num">Macro F1</th><th>GT Distribution</th><th class="num">Correct</th><th class="num">Incorrect</th><th class="num">Skipped</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody>
</table>
</div>

<div class="summary">
<h2>Trait Accuracy by Species</h2>
<table class="species-table">
<thead><tr><th>Species</th>{''.join(f'<th>{CATEGORY_LABELS.get(cat, cat)}</th>' for cat in DISPLAY_CATEGORIES)}</tr></thead>
<tbody>{''.join(species_accuracy_rows)}</tbody>
</table>
</div>

<div class="summary">
<h2>Within-Species Agreement (VLM Consistency)</h2>
<table class="species-table">
<thead><tr><th>Species</th>{''.join(f'<th>{CATEGORY_LABELS.get(cat, cat)}</th>' for cat in DISPLAY_CATEGORIES)}</tr></thead>
<tbody>{''.join(species_agreement_rows)}</tbody>
</table>
</div>

<div class="filters">
    <label>Filter:</label>
    <select id="verdict-filter" onchange="applyFilters()">
        <option value="all">All verdicts</option>
        <option value="incorrect">Has incorrect</option>
        <option value="correct">All correct</option>
    </select>
    <select id="category-filter" onchange="applyFilters()">
        <option value="all">All categories</option>
        {''.join(f'<option value="{cat}">{CATEGORY_LABELS.get(cat, cat)}</option>' for cat in DISPLAY_CATEGORIES)}
    </select>
    <input type="text" id="search" placeholder="Search species or specimen ID..." oninput="applyFilters()" />
</div>

<div class="count-display" id="count-display">Showing {stats['total']} of {stats['total']} specimens</div>

{''.join(cards_html)}

<script>
function applyFilters() {{
    const verdict = document.getElementById('verdict-filter').value;
    const category = document.getElementById('category-filter').value;
    const search = document.getElementById('search').value.toLowerCase();
    const cards = document.querySelectorAll('.card');
    let shown = 0;

    cards.forEach(card => {{
        let show = true;
        const verdicts = card.dataset.verdicts.split(' ');
        const species = card.dataset.species;
        const family = card.dataset.family;
        const specId = card.querySelector('h3').textContent.toLowerCase();

        // Verdict filter
        if (verdict === 'incorrect') {{
            show = verdicts.includes('incorrect');
        }} else if (verdict === 'correct') {{
            show = !verdicts.includes('incorrect') && !verdicts.includes('no_pred');
        }}

        // Category filter - show only cards where that specific category is incorrect
        if (show && category !== 'all') {{
            const rows = card.querySelectorAll('.traits-table tbody tr');
            let catFound = false;
            rows.forEach(row => {{
                const catCell = row.cells[0].textContent;
                const catKey = Object.entries({json.dumps(CATEGORY_LABELS)}).find(
                    ([k, v]) => v === catCell
                );
                if (catKey && catKey[0] === category) {{
                    catFound = row.classList.contains('incorrect');
                }}
            }});
            show = catFound;
        }}

        // Search filter
        if (show && search) {{
            show = species.includes(search) || specId.includes(search) || family.includes(search);
        }}

        card.classList.toggle('hidden', !show);
        if (show) shown++;
    }});

    document.getElementById('count-display').textContent =
        `Showing ${{shown}} of {stats['total']} specimens`;
}}
</script>

</body>
</html>'''

    return html
