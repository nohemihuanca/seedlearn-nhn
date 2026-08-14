"""HTML report generator for VLM Stage 1 benchmark scoring.

Produces a self-contained HTML report with inline CSS covering:
- Summary dashboard with per-trait accuracy
- Multi-label transparency breakdown
- Confusion matrix heatmaps per trait
- Per-specimen scorecard table
- Multi-vs-single comparison (when available)

Every metric includes a hover tooltip (title attribute) explaining what
it measures and how to interpret it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _accuracy_color(acc: float) -> str:
    """Return CSS background color based on accuracy threshold."""
    if acc >= 0.8:
        return "#d4edda"  # green
    elif acc >= 0.6:
        return "#fff3cd"  # yellow
    return "#f8d7da"  # red


def _heatmap_bg(count: int, max_count: int) -> str:
    """Return CSS background color for confusion matrix cell intensity."""
    if max_count == 0 or count == 0:
        return "#ffffff"
    ratio = count / max_count
    r = int(240 - 210 * ratio)
    g = int(248 - 148 * ratio)
    b = int(255 - 75 * ratio)
    return f"rgb({r},{g},{b})"


def _text_color_for_bg(count: int, max_count: int) -> str:
    """Return white text for dark cells, black for light."""
    if max_count > 0 and count / max_count > 0.6:
        return "#ffffff"
    return "#333333"


# ---------------------------------------------------------------------------
# Tooltip definitions — single source of truth for all metric explanations
# ---------------------------------------------------------------------------

_TIPS = {
    # Summary dashboard metric boxes
    "accuracy_answered": (
        "Accuracy (answered only): Of predictions where the model gave a definite "
        "answer (not 'unclear' or abstained), what fraction matched the ground truth? "
        "Excludes abstentions from both numerator and denominator. "
        "High accuracy + high abstention = cautious model."
    ),
    "strict_accuracy": (
        "Strict accuracy: Fraction of ALL ground-truth cells answered correctly. "
        "Abstentions count as wrong. This is the most conservative metric — "
        "it penalizes both errors and refusals to answer."
    ),
    "abstention_rate": (
        "Abstention rate: Fraction of ground-truth cells where the model refused "
        "to answer (e.g., 'unclear', 'not visible', empty). "
        "The gap between Accuracy and Strict Accuracy is explained by this rate. "
        "High abstention means the model is overly cautious."
    ),
    "specimens_scored": (
        "Number of specimens in the ground truth file that were also present "
        "in the benchmark results (matched by specimen_key)."
    ),
    "traits_scored": (
        "Number of morphological traits with ground truth values available "
        "for scoring. Traits without ground truth are skipped."
    ),
    "correct_answered": (
        "Correct / Answered: Number of correct predictions out of total non-abstained "
        "predictions. This is the numerator and denominator of the Accuracy metric."
    ),
    # Per-trait table columns
    "col_trait": (
        "Morphological trait being scored. Each trait is evaluated independently "
        "across all specimens."
    ),
    "col_accuracy": (
        "Accuracy: correct / (correct + mismatch). Excludes abstentions. "
        "Answers the question: 'When the model commits to an answer, how often is it right?'"
    ),
    "col_strict_acc": (
        "Strict accuracy: correct / total. Abstentions count as wrong. "
        "Answers: 'Across all specimens, what fraction are correctly identified?'"
    ),
    "col_abstention_pct": (
        "Percentage of specimens where the model abstained (answered 'unclear', "
        "'not visible', etc.) instead of committing to a trait value."
    ),
    "col_correct": (
        "Number of specimens where the prediction matched the ground truth "
        "(exact match or valid parent-type match, e.g., 'serrate' matches 'toothed')."
    ),
    "col_mismatch": (
        "Number of specimens where the model gave a definite answer that did NOT "
        "match any ground truth value. These are the actual errors."
    ),
    "col_n": (
        "Total number of specimens with ground truth for this trait. "
        "Equals correct + mismatch + abstentions."
    ),
    # Multi-label transparency columns
    "col_exact_gt_cells": (
        "Number of ground-truth cells with a single valid value (e.g., 'simple'). "
        "These provide the cleanest evaluation signal."
    ),
    "col_exact_gt_acc": (
        "Accuracy on single-value ground truth cells only. More stringent than "
        "overall accuracy because the model must predict the exact right value."
    ),
    "col_multi_label_cells": (
        "Number of ground-truth cells with multiple valid values (e.g., 'entire | toothed'). "
        "These are more permissive — the model gets credit for matching ANY valid value."
    ),
    "col_multi_label_acc": (
        "Accuracy on multi-label ground truth cells. Typically higher than exact-GT "
        "accuracy because multiple answers are accepted. Compare with exact-GT accuracy "
        "to assess whether the overall score is inflated by permissive matching."
    ),
    # Multi-label note
    "multi_label_note": (
        "Multi-label entries come from STRI identification keys that code species-level "
        "trait variability. For example, a species may exhibit both 'entire' and 'toothed' "
        "leaf margins across its life stages. The model gets credit for matching any valid "
        "value, which makes scoring more permissive. Compare exact-GT vs multi-label "
        "accuracy to gauge this effect."
    ),
    # Confusion matrix
    "cm_note": (
        "Confusion matrix: Rows are the actual (ground truth) values, columns are "
        "what the model predicted. Diagonal cells (top-left to bottom-right) are correct "
        "predictions. Off-diagonal cells are errors — read across a row to see what the "
        "model predicts when the actual value is X. Cell darkness indicates count. "
        "[abstain] means the model refused to answer."
    ),
    # Specimen scorecard columns
    "col_specimen": "Specimen identifier from the benchmark sample set.",
    "col_family": "Botanical family of the specimen.",
    "col_specimen_acc": (
        "Accuracy for this specimen: correct traits / (correct + incorrect). "
        "Abstained traits are excluded. Low accuracy means the model struggles "
        "with this particular species or individual."
    ),
    "col_specimen_correct": ("Number of traits correctly predicted for this specimen."),
    "col_specimen_incorrect": (
        "Number of traits where the model gave a wrong answer for this specimen."
    ),
    "col_specimen_abstained": (
        "Number of traits where the model abstained ('unclear', 'not visible') "
        "for this specimen. High abstention may indicate poor image quality or "
        "a species the model is unfamiliar with."
    ),
    # Multi-vs-single columns
    "col_multi_acc": (
        "Multi-image accuracy: All N images of the specimen sent together in one "
        "VLM call. This is the default pipeline mode. The model can cross-reference "
        "angles to resolve ambiguities."
    ),
    "col_single_mean_acc": (
        "Single-image mean accuracy: Each image sent independently, accuracy averaged "
        "across all images. Uses strict accuracy (abstentions count as wrong) to match "
        "the multi-image denominator. Lower than multi suggests multiple angles help."
    ),
    "col_majority_vote": (
        "Majority vote accuracy: For each specimen, take the most common prediction "
        "across all single-image runs, then score that prediction against ground truth. "
        "If this exceeds multi-image accuracy, simple voting outperforms the VLM's own "
        "multi-image reasoning."
    ),
    "col_consistency": (
        "Mean consistency: Average fraction of images that agree on the same answer "
        "per specimen. 1.0 = all images give the same answer. Low consistency means "
        "the trait is angle-dependent or the model is uncertain."
    ),
    "col_resolution_effect": (
        "Resolution effect: multi_accuracy − single_mean_accuracy. "
        "Positive = multi-image helps (model benefits from multiple angles). "
        "Negative = multi-image hurts (model gets confused by extra images). "
        "Near zero = no difference."
    ),
    "col_mvs_n": (
        "Number of specimens with ground truth for this trait, used in both "
        "multi and single scoring."
    ),
}


def _tip(key: str) -> str:
    """Return a title attribute string for a tooltip."""
    text = _TIPS.get(key, "")
    if text:
        return f' title="{_escape(text)}"'
    return ""


_CSS = """\
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 20px; background: #f5f5f5; color: #333;
}
h1 { color: #2c3e50; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #4a90d9; }
.card {
    background: white; padding: 20px; border-radius: 8px;
    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 16px;
}
.metric-box {
    background: #f8f9fa; border-radius: 6px; padding: 12px; text-align: center;
    border-left: 4px solid #4a90d9; cursor: help;
}
.metric-box .value { font-size: 1.6em; font-weight: bold; color: #2c3e50; }
.metric-box .label { font-size: 0.85em; color: #7f8c8d; }
table {
    border-collapse: collapse; width: 100%; background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 16px;
}
th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
th {
    background: #4a90d9; color: white; position: sticky; top: 0;
    font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px;
    cursor: help;
}
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #f0f7ff; }
.cm-table { width: auto; margin: 0 0 24px 0; }
.cm-table th { font-size: 0.8em; padding: 6px 10px; }
.cm-table td { padding: 6px 10px; text-align: center; font-size: 0.9em; }
.note {
    background: #e8f4fd; border-left: 4px solid #4a90d9;
    padding: 10px 14px; border-radius: 4px; margin: 10px 0;
    font-size: 0.9em; cursor: help;
}
details { margin: 5px 0; }
summary { cursor: pointer; font-weight: bold; }
.footer { margin-top: 30px; color: #999; font-size: 0.8em; text-align: center; }
"""


def _render_summary_dashboard(summary: dict) -> str:
    """Render the top-level summary dashboard section."""
    meta = summary.get("metadata", {})
    traits = summary.get("per_trait", [])

    n_specimens = meta.get("n_specimens_scored", meta.get("n_specimens_gt", 0))
    n_traits = len(traits)

    total_correct = sum(t.get("n_correct", 0) for t in traits)
    total_answered = sum(t.get("n_correct", 0) + t.get("n_mismatch", 0) for t in traits)
    total_scored = sum(t.get("n_scored", 0) for t in traits)
    total_abstained = sum(t.get("n_abstention", 0) for t in traits)

    overall_acc = total_correct / total_answered if total_answered else 0.0
    overall_strict = total_correct / total_scored if total_scored else 0.0
    abstention_rate = total_abstained / total_scored if total_scored else 0.0

    html = '<div class="card">\n<h2>Summary Dashboard</h2>\n'
    html += '<div class="metric-grid">\n'
    metrics = [
        (f"{overall_acc:.1%}", "Accuracy (answered)", "accuracy_answered"),
        (f"{overall_strict:.1%}", "Strict Accuracy", "strict_accuracy"),
        (f"{abstention_rate:.1%}", "Abstention Rate", "abstention_rate"),
        (str(n_specimens), "Specimens Scored", "specimens_scored"),
        (str(n_traits), "Traits Scored", "traits_scored"),
        (f"{total_correct}/{total_answered}", "Correct / Answered", "correct_answered"),
    ]
    for value, label, tip_key in metrics:
        html += (
            f'<div class="metric-box"{_tip(tip_key)}>'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>\n'
        )
    html += "</div>\n"

    # Per-trait table
    html += "<h3>Per-Trait Accuracy</h3>\n"
    html += "<table>\n<tr>"
    cols = [
        ("Trait", "col_trait"),
        ("Accuracy", "col_accuracy"),
        ("Strict Acc", "col_strict_acc"),
        ("Abstention %", "col_abstention_pct"),
        ("Correct", "col_correct"),
        ("Mismatch", "col_mismatch"),
        ("N", "col_n"),
    ]
    for col_name, tip_key in cols:
        html += f"<th{_tip(tip_key)}>{col_name}</th>"
    html += "</tr>\n"
    for t in traits:
        acc = t.get("accuracy", 0.0)
        bg = _accuracy_color(acc)
        html += "<tr>"
        html += f"<td>{_escape(t['trait'])}</td>"
        html += f'<td style="background:{bg}; font-weight:bold">{acc:.1%}</td>'
        html += f"<td>{t.get('strict_accuracy', 0.0):.1%}</td>"
        html += f"<td>{t.get('abstention_rate', 0.0):.1%}</td>"
        html += f"<td>{t.get('n_correct', 0)}</td>"
        html += f"<td>{t.get('n_mismatch', 0)}</td>"
        html += f"<td>{t.get('n_scored', 0)}</td>"
        html += "</tr>\n"
    html += "</table>\n</div>\n"
    return html


def _render_multi_label_transparency(summary: dict) -> str:
    """Render the multi-label transparency breakdown section."""
    traits = summary.get("per_trait", [])
    has_multi = any(t.get("n_multi_label_gt", 0) > 0 for t in traits)
    if not has_multi:
        return ""

    html = '<div class="card">\n<h2>Multi-Label Transparency</h2>\n'
    html += f'<div class="note"{_tip("multi_label_note")}>'
    html += (
        "Multi-label ground truth cells accept any of several "
        "valid values. Accuracy on these cells is typically higher because the "
        "matching criteria are more permissive. Hover over column headers for details."
    )
    html += "</div>\n"
    html += "<table>\n<tr>"
    cols = [
        ("Trait", "col_trait"),
        ("Exact-GT Cells", "col_exact_gt_cells"),
        ("Exact-GT Acc", "col_exact_gt_acc"),
        ("Multi-Label Cells", "col_multi_label_cells"),
        ("Multi-Label Acc", "col_multi_label_acc"),
    ]
    for col_name, tip_key in cols:
        html += f"<th{_tip(tip_key)}>{col_name}</th>"
    html += "</tr>\n"

    for t in traits:
        n_exact = t.get("n_exact_gt", 0)
        n_multi = t.get("n_multi_label_gt", 0)
        exact_acc = t.get("exact_gt_accuracy", 0.0)
        multi_acc = t.get("multi_label_gt_accuracy", 0.0)
        html += "<tr>"
        html += f"<td>{_escape(t['trait'])}</td>"
        html += f"<td>{n_exact}</td>"
        html += (
            f'<td style="background:{_accuracy_color(exact_acc)}">{exact_acc:.1%}</td>'
        )
        html += f"<td>{n_multi}</td>"
        html += (
            f'<td style="background:{_accuracy_color(multi_acc)}">{multi_acc:.1%}</td>'
        )
        html += "</tr>\n"

    html += "</table>\n</div>\n"
    return html


def _render_confusion_matrices(confusion_matrices: dict) -> str:
    """Render per-trait confusion matrix heatmap tables."""
    if not confusion_matrices:
        return ""

    html = '<div class="card">\n<h2>Confusion Matrices</h2>\n'
    html += f'<div class="note"{_tip("cm_note")}>'
    html += (
        "Rows = actual (ground truth), columns = predicted. "
        "Diagonal cells are correct predictions. Off-diagonal cells are errors. "
        "Cell darkness indicates count. Hover for details."
    )
    html += "</div>\n"

    for trait, matrix in confusion_matrices.items():
        html += f"<h3>{_escape(trait)}</h3>\n"

        pred_labels = sorted(matrix.keys())
        actual_labels = sorted({a for pred_row in matrix.values() for a in pred_row})

        max_count = max(
            (matrix.get(p, {}).get(a, 0) for p in pred_labels for a in actual_labels),
            default=0,
        )

        html += '<table class="cm-table">\n'
        html += '<tr><th title="Row = actual ground truth value, Column = model prediction">'
        html += "Actual \\ Predicted</th>"
        for p in pred_labels:
            html += f'<th title="Model predicted: {_escape(p)}">{_escape(p)}</th>'
        html += "</tr>\n"

        for a in actual_labels:
            html += "<tr><td style='font-weight:bold'"
            html += f' title="Ground truth value: {_escape(a)}"'
            html += f">{_escape(a)}</td>"
            for p in pred_labels:
                count = matrix.get(p, {}).get(a, 0)
                bg = _heatmap_bg(count, max_count)
                fg = _text_color_for_bg(count, max_count)
                if p == a:
                    cell_tip = f"Correct: model predicted '{p}' when actual was '{a}' ({count} times)"
                else:
                    cell_tip = (
                        f"Error: model predicted '{p}' when actual was '{a}' ({count} times)"
                        if count > 0
                        else f"No cases of predicting '{p}' when actual was '{a}'"
                    )
                html += (
                    f'<td style="background:{bg}; color:{fg}; font-weight:bold"'
                    f' title="{_escape(cell_tip)}">{count}</td>'
                )
            html += "</tr>\n"
        html += "</table>\n"

    html += "</div>\n"
    return html


def _render_specimen_scorecard(summary: dict) -> str:
    """Render the per-specimen scorecard table."""
    specimens = summary.get("per_specimen", [])
    if not specimens:
        return ""

    sorted_specimens = sorted(specimens, key=lambda s: s.get("accuracy", 0.0))

    html = '<div class="card">\n<h2>Per-Specimen Scorecard</h2>\n'
    html += "<table>\n<tr>"
    cols = [
        ("Specimen", "col_specimen"),
        ("Family", "col_family"),
        ("Accuracy", "col_specimen_acc"),
        ("Correct", "col_specimen_correct"),
        ("Incorrect", "col_specimen_incorrect"),
        ("Abstained", "col_specimen_abstained"),
    ]
    for col_name, tip_key in cols:
        html += f"<th{_tip(tip_key)}>{col_name}</th>"
    html += "</tr>\n"

    for s in sorted_specimens:
        acc = s.get("accuracy", 0.0)
        bg = _accuracy_color(acc)
        html += "<tr>"
        html += f"<td>{_escape(s.get('specimen_key', ''))}</td>"
        html += f"<td>{_escape(s.get('family', ''))}</td>"
        html += f'<td style="background:{bg}; font-weight:bold">{acc:.1%}</td>'
        html += f"<td>{s.get('traits_correct', 0)}</td>"
        html += f"<td>{s.get('traits_incorrect', 0)}</td>"
        html += f"<td>{s.get('traits_abstained', 0)}</td>"
        html += "</tr>\n"

    html += "</table>\n</div>\n"
    return html


def _render_multi_vs_single(summary: dict) -> str:
    """Render multi-vs-single comparison table if data exists."""
    mvs = summary.get("multi_vs_single")
    if not mvs:
        return ""

    html = '<div class="card">\n<h2>Multi vs Single Image Comparison</h2>\n'
    html += "<table>\n<tr>"
    cols = [
        ("Trait", "col_trait"),
        ("Multi Acc", "col_multi_acc"),
        ("Single Mean Acc", "col_single_mean_acc"),
        ("Majority Vote", "col_majority_vote"),
        ("Consistency", "col_consistency"),
        ("Resolution Effect", "col_resolution_effect"),
        ("N", "col_mvs_n"),
    ]
    for col_name, tip_key in cols:
        html += f"<th{_tip(tip_key)}>{col_name}</th>"
    html += "</tr>\n"

    for trait, m in mvs.items():
        delta = m.get("resolution_effect", 0.0)
        delta_color = "#28a745" if delta > 0 else "#dc3545" if delta < 0 else "#333"
        html += "<tr>"
        html += f"<td>{_escape(trait)}</td>"
        html += f'<td style="background:{_accuracy_color(m["multi_accuracy"])}">'
        html += f"{m['multi_accuracy']:.1%}</td>"
        html += f'<td style="background:{_accuracy_color(m["single_mean_accuracy"])}">'
        html += f"{m['single_mean_accuracy']:.1%}</td>"
        html += f"<td>{m['majority_vote_accuracy']:.1%}</td>"
        html += f"<td>{m['mean_consistency']:.1%}</td>"
        html += f'<td style="color:{delta_color}; font-weight:bold"'
        html += f"{_tip('col_resolution_effect')}>{delta:+.1%}</td>"
        html += f"<td>{m['n_specimens']}</td>"
        html += "</tr>\n"

    html += "</table>\n</div>\n"
    return html


def generate_score_report_html(
    summary: dict[str, Any],
    confusion_matrices: dict[str, dict[str, dict[str, int]]],
) -> str:
    """Generate a self-contained HTML scoring report.

    Args:
        summary: Full summary dict as written to summary.json (includes
            ``metadata``, ``per_trait``, ``per_specimen``, and optionally
            ``multi_vs_single`` keys).
        confusion_matrices: Per-trait confusion matrices where
            ``cm[predicted][actual] = count``.

    Returns:
        Complete HTML string ready to write to a file.
    """
    meta = summary.get("metadata", {})
    mode = meta.get("mode", "multi")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources = meta.get("result_sources", [])
    source_str = ", ".join(str(s) for s in sources) if sources else "N/A"

    body = _render_summary_dashboard(summary)
    body += _render_multi_label_transparency(summary)
    body += _render_confusion_matrices(confusion_matrices)
    body += _render_specimen_scorecard(summary)
    body += _render_multi_vs_single(summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Stage 1 Scoring Report</title>
    <style>
{_CSS}
    </style>
</head>
<body>
    <h1>VLM Stage 1 Scoring Report</h1>
    <p>Generated: {timestamp} | Mode: {_escape(mode)} | Sources: {_escape(source_str)}</p>
{body}
    <div class="footer">SeedLearn VLM Benchmark Scorer</div>
</body>
</html>
"""
    return html
