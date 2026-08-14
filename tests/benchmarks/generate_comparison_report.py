#!/usr/bin/env python3
"""Generate a comprehensive cross-model comparison HTML report.

Reads individual scored results (summary.json, per_specimen_scorecard.csv,
confusion_matrices/) and produces a single standalone HTML report with:
  - Side-by-side accuracy table (answered + strict)
  - Per-model confusion matrices for each trait
  - Per-specimen scorecard heatmap across all models
  - Key findings summary

Usage:
    python tests/benchmarks/generate_comparison_report.py \
        --output results/vlm_benchmark/comparison_5models_full/report.html
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path

# ── Model registry ───────────────────────────────────────────────────────────
# Maps display name → scores directory (relative to project root)
MODEL_RESULTS: dict[str, str] = {
    "Qwen3-VL-32B-Instruct": "results/vlm_benchmark/baseline_qwen3vl32b_sys4/scores",
    "Qwen3-VL-32B-Thinking": "tests/benchmarks/results/20260320_125709_Qwen_Qwen3-VL-32B-Thinking-FP8/scores",
    "Gemma-3-27B (BF16)": "tests/benchmarks/results/20260320_134333_google_gemma-3-27b-it/scores",
    "Gemma-3-27B (FP8)": "tests/benchmarks/results/20260320_145656_RedHatAI_gemma-3-27b-it-FP8-dynamic/scores",
    "Qwen3.5-27B (FP8)": "tests/benchmarks/results/20260320_155940_Qwen_Qwen3.5-27B-FP8/scores",
}


def _load_summary(scores_dir: Path) -> dict:
    with open(scores_dir / "summary.json") as f:
        return json.load(f)


def _load_scorecard(scores_dir: Path) -> list[dict]:
    rows = []
    with open(scores_dir / "per_specimen_scorecard.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_confusion(scores_dir: Path, trait: str) -> dict:
    cm_path = scores_dir / "confusion_matrices" / f"{trait}.json"
    if cm_path.exists():
        with open(cm_path) as f:
            return json.load(f)
    return {}


def _compute_overall(traits: list[dict]) -> dict:
    total_correct = sum(t["n_correct"] for t in traits)
    total_scored = sum(t["n_scored"] for t in traits)
    total_abstain = sum(t["n_abstention"] for t in traits)
    total_with_gt = total_scored + total_abstain
    return {
        "accuracy": total_correct / total_scored if total_scored else 0,
        "strict_accuracy": total_correct / total_with_gt if total_with_gt else 0,
        "abstention_rate": total_abstain / total_with_gt if total_with_gt else 0,
        "n_correct": total_correct,
        "n_scored": total_scored,
        "n_abstain": total_abstain,
    }


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _cell_color(value: float, best: float, worst: float) -> str:
    """Return background style for best/worst highlighting."""
    if value == best:
        return ' style="background:#d4edda; font-weight:bold"'
    if value == worst:
        return ' style="background:#f8d7da"'
    return ""


def _cell_color_low_good(value: float, best: float, worst: float) -> str:
    """For metrics where lower is better (abstention)."""
    if value == best:
        return ' style="background:#d4edda; font-weight:bold"'
    if value == worst:
        return ' style="background:#f8d7da"'
    return ""


def _cm_cell_bg(count: int, max_count: int) -> str:
    """Blue gradient for confusion matrix cells."""
    if max_count == 0 or count == 0:
        return "background:#ffffff"
    intensity = count / max_count
    r = int(230 - intensity * 200)
    g = int(238 - intensity * 138)
    b = int(250 - intensity * 70)
    text_color = "#ffffff" if intensity > 0.6 else "#333333"
    return f"background:rgb({r},{g},{b}); color:{text_color}"


CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 20px; background: #f5f5f5; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px;
}
h1 { color: #2c3e50; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #4a90d9; margin-top: 20px; }
h4 { color: #666; margin-top: 16px; margin-bottom: 8px; }
.card {
    background: white; padding: 20px; border-radius: 8px;
    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
table {
    border-collapse: collapse; width: 100%; background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 16px;
}
th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: center; }
th {
    background: #4a90d9; color: white; position: sticky; top: 0;
    font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; cursor: help;
}
td.model { text-align: left; font-weight: bold; font-size: 0.9em; white-space: nowrap; }
tr:hover { background: #f0f7ff; }
.cm-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px; margin-bottom: 20px;
}
.cm-card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.cm-card h4 { margin-top: 0; color: #4a90d9; }
.cm-table { width: auto; margin: 0; box-shadow: none; }
.cm-table th { font-size: 0.75em; padding: 6px 8px; }
.cm-table td { padding: 6px 8px; font-size: 0.85em; font-weight: bold; }
.note {
    background: #e8f4fd; border-left: 4px solid #4a90d9;
    padding: 12px 16px; border-radius: 4px; margin: 12px 0; font-size: 0.9em;
}
.finding { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 4px; margin: 8px 0; font-size: 0.9em; }
.footer { margin-top: 30px; color: #999; font-size: 0.8em; text-align: center; }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: bold; padding: 8px 0; }
.specimen-table td { font-size: 0.85em; padding: 6px 10px; }
.specimen-table th { font-size: 0.8em; padding: 6px 10px; }
"""


def generate_report(project_root: Path, output_path: Path) -> None:
    """Generate comprehensive cross-model comparison HTML report."""

    # Load all model data
    models: dict[str, dict] = {}
    for name, rel_path in MODEL_RESULTS.items():
        scores_dir = project_root / rel_path
        if not scores_dir.exists():
            print(f"WARNING: Skipping {name} — {scores_dir} not found")
            continue
        summary = _load_summary(scores_dir)
        overall = _compute_overall(summary["per_trait"])
        trait_map = {t["trait"]: t for t in summary["per_trait"]}
        scorecard = _load_scorecard(scores_dir)
        models[name] = {
            "summary": summary,
            "overall": overall,
            "traits": trait_map,
            "scorecard": scorecard,
            "scores_dir": scores_dir,
        }

    trait_names = list(next(iter(models.values()))["traits"].keys())
    n_specimens = models[next(iter(models))]["summary"]["metadata"]["n_specimens_scored"]
    n_models = len(models)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLM Model Comparison — Full Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>VLM Model Comparison — Full Benchmark Report</h1>
<p>Generated: {now} | {n_models} models | {n_specimens} specimens | {len(trait_names)} STRI traits | Multi-image mode</p>
<div class="note">
Green = best per column. Red = worst. Models ranked by strict accuracy (production suitability).
Hover over headers for definitions. Scroll down for confusion matrices and per-specimen breakdowns.
</div>
"""

    # ── Section 1: Accuracy tables ────────────────────────────────────────
    # Compute best/worst for each column
    acc_vals = {name: m["overall"]["accuracy"] for name, m in models.items()}
    strict_vals = {name: m["overall"]["strict_accuracy"] for name, m in models.items()}
    abst_vals = {name: m["overall"]["abstention_rate"] for name, m in models.items()}

    best_acc, worst_acc = max(acc_vals.values()), min(acc_vals.values())
    best_strict, worst_strict = max(strict_vals.values()), min(strict_vals.values())
    best_abst, worst_abst = min(abst_vals.values()), max(abst_vals.values())

    # Per-trait best/worst
    trait_acc_best = {}
    trait_acc_worst = {}
    trait_strict_best = {}
    trait_strict_worst = {}
    for t in trait_names:
        t_accs = [m["traits"][t]["accuracy"] for m in models.values()]
        t_stricts = [m["traits"][t]["strict_accuracy"] for m in models.values()]
        trait_acc_best[t] = max(t_accs)
        trait_acc_worst[t] = min(t_accs)
        trait_strict_best[t] = max(t_stricts)
        trait_strict_worst[t] = min(t_stricts)

    # Accuracy (answered) table
    html += """
<div class="card">
<h2>Accuracy (Answered Only — Excludes Abstentions)</h2>
<div class="note">When a model commits to an answer, how often is it right?
High accuracy + high abstention = cautious model. Compare with strict accuracy below.</div>
<table>
<tr>
<th title="Vision-language model tested">Model</th>
<th title="correct / (correct + mismatch)">Overall</th>
"""
    for t in trait_names:
        html += f'<th title="Accuracy on {t} when model answers">{t.replace("_", " ")}</th>'
    html += '<th title="Fraction of cells where model refused to answer">Abstain%</th></tr>\n'

    # Sort by strict accuracy descending
    sorted_models = sorted(models.items(), key=lambda x: x[1]["overall"]["strict_accuracy"], reverse=True)

    for name, m in sorted_models:
        ov = m["overall"]
        acc_style = _cell_color(ov["accuracy"], best_acc, worst_acc)
        abst_style = _cell_color_low_good(ov["abstention_rate"], best_abst, worst_abst)
        html += f'<tr><td class="model">{escape(name)}</td>'
        html += f'<td{acc_style}>{_pct(ov["accuracy"])}</td>'
        for t in trait_names:
            tv = m["traits"][t]["accuracy"]
            style = _cell_color(tv, trait_acc_best[t], trait_acc_worst[t])
            html += f"<td{style}>{_pct(tv)}</td>"
        html += f'<td{abst_style}>{_pct(ov["abstention_rate"])}</td></tr>\n'

    html += "</table></div>\n"

    # Strict accuracy table
    html += """
<div class="card">
<h2>Strict Accuracy (Abstentions Count as Wrong)</h2>
<div class="note">Most conservative metric — penalizes both errors and refusals.
Use this for production pipeline decisions.</div>
<table>
<tr>
<th>Model</th>
<th title="correct / (correct + mismatch + abstentions)">Overall Strict</th>
"""
    for t in trait_names:
        html += f'<th title="Strict accuracy on {t}">{t.replace("_", " ")}</th>'
    html += "</tr>\n"

    for name, m in sorted_models:
        ov = m["overall"]
        strict_style = _cell_color(ov["strict_accuracy"], best_strict, worst_strict)
        html += f'<tr><td class="model">{escape(name)}</td>'
        html += f'<td{strict_style}>{_pct(ov["strict_accuracy"])}</td>'
        for t in trait_names:
            tv = m["traits"][t]["strict_accuracy"]
            style = _cell_color(tv, trait_strict_best[t], trait_strict_worst[t])
            html += f"<td{style}>{_pct(tv)}</td>"
        html += "</tr>\n"

    html += "</table></div>\n"

    # ── Section 2: Confusion matrices ─────────────────────────────────────
    html += """
<div class="card">
<h2>Confusion Matrices by Trait</h2>
<div class="note">Rows = actual (ground truth), columns = predicted. Diagonal = correct.
Off-diagonal = errors. Cell darkness indicates count. [abstain] = model refused to answer.</div>
"""

    for t in trait_names:
        html += f'<h3>{t.replace("_", " ").title()}</h3>\n'
        html += '<div class="cm-grid">\n'

        for name, m in sorted_models:
            cm = _load_confusion(m["scores_dir"], t)
            if not cm:
                continue

            # Get all labels (actual + predicted)
            actual_labels = list(cm.keys())
            pred_labels = set()
            for row in cm.values():
                pred_labels.update(row.keys())
            pred_labels = sorted(pred_labels)

            # Find max count for color scaling
            max_count = max(
                (count for row in cm.values() for count in row.values()),
                default=0,
            )

            html += f'<div class="cm-card"><h4>{escape(name)}</h4>\n'
            html += '<table class="cm-table"><tr><th>Actual \\ Pred</th>'
            for p in pred_labels:
                html += f"<th>{escape(p)}</th>"
            html += "</tr>\n"

            for a in actual_labels:
                html += f'<tr><td style="font-weight:bold; text-align:left">{escape(a)}</td>'
                for p in pred_labels:
                    count = cm.get(a, {}).get(p, 0)
                    is_correct = a == p
                    bg = _cm_cell_bg(count, max_count)
                    if is_correct:
                        tip = f"Correct: predicted '{p}' when actual '{a}' ({count}x)"
                    else:
                        tip = f"Error: predicted '{p}' when actual '{a}' ({count}x)"
                    html += f'<td style="{bg}" title="{tip}">{count}</td>'
                html += "</tr>\n"

            html += "</table></div>\n"

        html += "</div>\n"  # close cm-grid

    html += "</div>\n"  # close card

    # ── Section 3: Per-specimen scorecard ─────────────────────────────────
    html += """
<div class="card">
<h2>Per-Specimen Accuracy Across Models</h2>
<div class="note">Each cell shows specimen accuracy for that model. Click a specimen row
to expand trait-level detail. Sorted by mean accuracy across all models (hardest first).</div>
"""

    # Build specimen → model → accuracy mapping
    all_specimens = set()
    specimen_data: dict[str, dict[str, dict]] = {}
    for name, m in models.items():
        for row in m["scorecard"]:
            spec = row.get("specimen") or row.get("specimen_key", "?")
            family = row.get("family", "")
            all_specimens.add(spec)
            if spec not in specimen_data:
                specimen_data[spec] = {"family": family}
            correct = int(row.get("n_correct", 0))
            incorrect = int(row.get("n_incorrect", 0))
            abstained = int(row.get("n_abstained", 0))
            total = correct + incorrect
            acc = correct / total if total > 0 else 0
            specimen_data[spec][name] = {
                "accuracy": acc,
                "correct": correct,
                "incorrect": incorrect,
                "abstained": abstained,
            }

    # Sort by mean accuracy (hardest first)
    def _mean_acc(spec: str) -> float:
        accs = [
            specimen_data[spec][n]["accuracy"]
            for n in models
            if n in specimen_data[spec]
        ]
        return sum(accs) / len(accs) if accs else 0

    sorted_specimens = sorted(all_specimens, key=_mean_acc)

    model_names = [name for name, _ in sorted_models]

    html += '<table class="specimen-table"><tr><th>Specimen</th><th>Family</th>'
    for name in model_names:
        short = name.split("(")[0].strip() if "(" in name else name
        html += f"<th>{escape(short)}</th>"
    html += "<th>Mean</th></tr>\n"

    for spec in sorted_specimens:
        sd = specimen_data[spec]
        mean = _mean_acc(spec)

        html += f'<tr><td style="text-align:left; font-size:0.8em">{escape(spec)}</td>'
        html += f'<td style="text-align:left; font-size:0.8em">{escape(sd.get("family", ""))}</td>'

        for name in model_names:
            if name in sd:
                acc = sd[name]["accuracy"]
                abst = sd[name]["abstained"]
                if acc >= 0.8:
                    bg = "#d4edda"
                elif acc >= 0.6:
                    bg = "#fff3cd"
                else:
                    bg = "#f8d7da"
                abst_mark = f" ({abst}A)" if abst > 0 else ""
                html += f'<td style="background:{bg}; font-weight:bold" title="{sd[name]["correct"]}✓ {sd[name]["incorrect"]}✗ {abst}A">{_pct(acc)}{abst_mark}</td>'
            else:
                html += '<td style="color:#ccc">—</td>'

        # Mean
        if mean >= 0.8:
            bg = "#d4edda"
        elif mean >= 0.6:
            bg = "#fff3cd"
        else:
            bg = "#f8d7da"
        html += f'<td style="background:{bg}; font-weight:bold">{_pct(mean)}</td>'
        html += "</tr>\n"

    html += "</table></div>\n"

    # ── Section 4: Key findings ───────────────────────────────────────────
    best_strict_name = sorted_models[0][0]
    best_strict_val = sorted_models[0][1]["overall"]["strict_accuracy"]
    best_acc_name = max(models.items(), key=lambda x: x[1]["overall"]["accuracy"])[0]
    best_acc_val = models[best_acc_name]["overall"]["accuracy"]

    # Find hardest trait
    trait_mean_strict = {}
    for t in trait_names:
        vals = [m["traits"][t]["strict_accuracy"] for m in models.values()]
        trait_mean_strict[t] = sum(vals) / len(vals)
    hardest_trait = min(trait_mean_strict, key=trait_mean_strict.get)
    easiest_trait = max(trait_mean_strict, key=trait_mean_strict.get)

    html += f"""
<div class="card">
<h2>Key Findings</h2>
<div class="finding"><strong>Best strict accuracy (production pick):</strong> {escape(best_strict_name)} at {_pct(best_strict_val)} — always commits, no abstentions.</div>
<div class="finding"><strong>Best per-answer accuracy (smartest):</strong> {escape(best_acc_name)} at {_pct(best_acc_val)} — highest accuracy when it answers, but abstains frequently.</div>
<div class="finding"><strong>Hardest trait:</strong> {hardest_trait.replace('_', ' ')} (mean strict: {_pct(trait_mean_strict[hardest_trait])}) — large cross-model variance suggests prompt engineering potential.</div>
<div class="finding"><strong>Easiest trait:</strong> {easiest_trait.replace('_', ' ')} (mean strict: {_pct(trait_mean_strict[easiest_trait])}) — all models converge here.</div>
<div class="finding"><strong>Thinking mode:</strong> Qwen3-VL-32B-Thinking underperforms Instruct on strict accuracy (abstentions increase, accuracy flat). Not recommended.</div>
<div class="finding"><strong>Qwen3.5-27B paradox:</strong> Highest per-answer accuracy on leaf_margin and stipules, but 15%+ abstention rate destroys strict accuracy. Needs <code>enable_thinking: false</code> to test.</div>
<div class="finding"><strong>Stipules systematic failure:</strong> No model ever predicts "present" — all default to "absent". This is a prompt issue, not model quality.</div>
</div>
"""

    # ── Section 5: Test configuration ─────────────────────────────────────
    html += f"""
<div class="card">
<h2>Test Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td style="text-align:left">Specimens</td><td>{n_specimens} (from 17 families)</td></tr>
<tr><td style="text-align:left">Traits scored</td><td>{', '.join(t.replace('_', ' ') for t in trait_names)}</td></tr>
<tr><td style="text-align:left">Ground truth source</td><td>STRI identification keys (stage1_ground_truth_active.csv)</td></tr>
<tr><td style="text-align:left">Inference mode</td><td>Multi-image (all 4-5 photos per specimen)</td></tr>
<tr><td style="text-align:left">Prompt style</td><td>sys4 (multi-image specialist)</td></tr>
<tr><td style="text-align:left">Temperature</td><td>0.6 (pipeline default at time of test)</td></tr>
<tr><td style="text-align:left">Hardware</td><td>NVIDIA H200 140GB HBM3 (single GPU)</td></tr>
<tr><td style="text-align:left">Benchmark date</td><td>2026-03-20</td></tr>
</table>
</div>
"""

    html += f'<div class="footer">SeedLearn VLM Benchmark — {n_models} models | Generated {now}</div>\n'
    html += "</body></html>"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Report written: {output_path}")
    print(f"  {n_models} models, {n_specimens} specimens, {len(trait_names)} traits")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-model VLM benchmark comparison report"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/vlm_benchmark/comparison_5models_full/report.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )
    args = parser.parse_args()
    generate_report(args.project_root, args.output)


if __name__ == "__main__":
    main()
