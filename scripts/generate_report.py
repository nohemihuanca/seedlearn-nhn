#!/usr/bin/env python3
"""Generate visual HTML reports for SimpleShot experiments and pipeline results.

Subcommands:
    single          Generate report for a single experiment
    learning-curve  Generate learning curve comparison across k-shots
    pipeline        Generate HTML report from a pipeline result JSON file

Examples:
    python scripts/generate_report.py single \\
        --experiment-dir /path/to/results/family/5_shot/split_seed42

    python scripts/generate_report.py learning-curve \\
        --rank family --baseline-blind 0.2387 --baseline-closed 0.1349

    python scripts/generate_report.py pipeline \\
        --result-json results/pipeline/SRAPHEDE2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats as sp_stats

from seedlearn.reporting import html as report_html

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =========================================================================
# Single-experiment report
# =========================================================================

REPO_BASE = Path("data/experiments/simpleshot")


def _extract_dated_subdir(experiment_dir: Path) -> str | None:
    for part in experiment_dir.parts:
        if re.match(r"\d{4}-\d{2}-\d{2}_v\d{4}-\d{2}-\d{2}_\d+K", part):
            return part
    return None


def _copy_html_to_repo(html_path: Path, experiment_dir: Path, rank: str) -> Path | None:
    dated_subdir = _extract_dated_subdir(experiment_dir)
    if dated_subdir is None:
        return None
    repo_path = REPO_BASE / dated_subdir / rank / "visuals" / "simpleshot_report.html"
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_path, repo_path)
    return repo_path


def _load_experiment_data(experiment_dir: Path):
    metrics_path = experiment_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {experiment_dir}")
    with open(metrics_path) as f:
        metrics = json.load(f)

    predictions = pd.read_csv(experiment_dir / "predictions.csv")

    per_class_metrics = pd.read_csv(experiment_dir / "per_class_metrics.csv")
    if per_class_metrics.columns[0] != "label":
        per_class_metrics = per_class_metrics.rename(columns={per_class_metrics.columns[0]: "label"})
    per_class_metrics = per_class_metrics.set_index("label")

    confusion_matrix = pd.read_csv(experiment_dir / "confusion_matrix.csv", index_col=0)

    with open(experiment_dir / "support_set.json") as f:
        support_set = json.load(f)

    return metrics, predictions, per_class_metrics, confusion_matrix, support_set


def _auto_detect_rank(experiment_dir: Path) -> str:
    info_path = experiment_dir / "experiment_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"experiment_info.json not found. Specify --rank explicitly.")
    with open(info_path) as f:
        return json.load(f).get("rank", "family")


def _generate_single_report(
    experiment_dir: Path, rank: str, output_dir: Path,
    top_k: int, baseline_blind: float | None, baseline_closed: float | None,
) -> None:
    metrics, predictions, per_class_metrics, confusion_matrix, support_set = _load_experiment_data(experiment_dir)
    k_shot = support_set.get("k_shot", 0)
    num_classes = support_set.get("num_classes", len(per_class_metrics))
    total_images = len(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_parts = []

    html_parts.append(report_html.get_html_header(rank, k_shot))
    html_parts.append(report_html.get_evaluation_context(rank=rank, k_shot=k_shot, metrics=metrics, num_classes=num_classes, total_images=total_images))

    fig_summary = report_html.build_summary_table(metrics, k_shot)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_summary, include_plotlyjs="cdn", div_id="summary-table"), chart_type="summary", k_shot=k_shot, rank=rank))

    html_parts.append(report_html.get_support_set_section(k_shot=k_shot, num_classes=num_classes, support_seed=support_set.get("support_seed", 42)))

    fig_support = report_html.build_support_set_distribution(support_set)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_support, include_plotlyjs=False, div_id="support-distribution"), chart_type="support_dist", k_shot=k_shot, rank=rank))

    fig_label = report_html.build_label_support_chart(predictions, top_k)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_label, include_plotlyjs=False, div_id="label-support"), chart_type="label_support", k_shot=k_shot, rank=rank))

    fig_pc = report_html.build_per_label_metrics(per_class_metrics, top_k)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_pc, include_plotlyjs=False, div_id="per-class-metrics"), chart_type="per_class", k_shot=k_shot, rank=rank))

    fig_cm = report_html.build_confusion_matrix(confusion_matrix, top_k)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_cm, include_plotlyjs=False, div_id="confusion-matrix"), chart_type="confusion", k_shot=k_shot, rank=rank))

    fig_errors = report_html.build_top_errors(predictions, top_n=15)
    html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_errors, include_plotlyjs=False, div_id="top-errors"), chart_type="errors", k_shot=k_shot, rank=rank))

    support_classes = list(support_set.get("per_class_support", {}).keys())
    if support_classes:
        fig_sp = report_html.build_support_vs_performance(per_class_metrics, support_classes)
        html_parts.append(report_html.wrap_plotly_div(pio.to_html(fig_sp, include_plotlyjs=False, div_id="support-performance"), chart_type="support_perf", k_shot=k_shot, rank=rank))

    html_parts.append(report_html.get_conclusions_section(rank=rank, k_shot=k_shot, accuracy=metrics.get("accuracy", 0.0), top5_accuracy=metrics.get("top5_accuracy", 0.0), baseline_blind_accuracy=baseline_blind, baseline_closed_accuracy=baseline_closed))
    html_parts.append(report_html.get_html_footer())

    output_path = output_dir / "simpleshot_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logging.info("Report saved to %s (%.2f MB)", output_path, output_path.stat().st_size / 1024 / 1024)
    repo_copy = _copy_html_to_repo(output_path, experiment_dir, rank)
    if repo_copy:
        logging.info("Repo copy saved to %s", repo_copy)


# =========================================================================
# Learning-curve report
# =========================================================================

def _auto_discover_batch_summary(rank: str) -> Path:
    base = Path("/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/experiments/simpleshot")
    batch_files = list(base.rglob(f"batch_summary_{rank}_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch summary files found for rank '{rank}'")
    return max(batch_files, key=lambda p: p.stat().st_mtime)


def _load_batch_results(batch_summary_path: Path) -> dict:
    with open(batch_summary_path) as f:
        batch_data = json.load(f)

    experiments, k_shots_set, seeds_set = [], set(), set()
    for result in batch_data.get("results", []):
        if not result.get("success"):
            continue
        output_dir = Path(result["output_dir"])
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        k_shot, seed = result.get("k_shot"), result.get("split_seed")
        if k_shot is None or seed is None:
            continue
        k_shots_set.add(k_shot)
        seeds_set.add(seed)
        experiments.append({"k_shot": k_shot, "seed": seed, "metrics": metrics, "output_dir": str(output_dir)})

    return {"experiments": experiments, "k_shots": sorted(k_shots_set), "seeds": sorted(seeds_set)}


def _aggregate_across_seeds(experiments: list[dict], k_shots: list[int]) -> pd.DataFrame:
    rows = []
    for k in k_shots:
        accs = [e["metrics"]["accuracy"] for e in experiments if e["k_shot"] == k]
        t5s = [e["metrics"]["top5_accuracy"] for e in experiments if e["k_shot"] == k]
        if not accs:
            continue
        acc_mean, acc_std = np.mean(accs), (np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
        t5_mean, t5_std = np.mean(t5s), (np.std(t5s, ddof=1) if len(t5s) > 1 else 0.0)
        acc_ci = (sp_stats.t.interval(0.95, len(accs) - 1, loc=acc_mean, scale=sp_stats.sem(accs))[1] - acc_mean) if len(accs) > 1 else 0.0
        t5_ci = (sp_stats.t.interval(0.95, len(t5s) - 1, loc=t5_mean, scale=sp_stats.sem(t5s))[1] - t5_mean) if len(t5s) > 1 else 0.0
        rows.append({"k_shot": k, "accuracy_mean": acc_mean, "accuracy_std": acc_std, "accuracy_ci": acc_ci, "top5_accuracy_mean": t5_mean, "top5_accuracy_std": t5_std, "top5_accuracy_ci": t5_ci, "num_seeds": len(accs)})
    return pd.DataFrame(rows)


def _compute_marginal(agg: pd.DataFrame) -> pd.DataFrame:
    if len(agg) < 2:
        return pd.DataFrame()
    df = agg.sort_values("k_shot").reset_index(drop=True)
    rows = []
    for i in range(len(df) - 1):
        imp = df.loc[i + 1, "accuracy_mean"] - df.loc[i, "accuracy_mean"]
        rows.append({"k_from": df.loc[i, "k_shot"], "k_to": df.loc[i + 1, "k_shot"], "improvement": imp})
    return pd.DataFrame(rows)


def _build_learning_curve_fig(agg: pd.DataFrame, rank: str, bl_blind=None, bl_closed=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["k_shot"], y=agg["accuracy_mean"], error_y=dict(type="data", array=agg["accuracy_ci"]), mode="lines+markers", name="Top-1", line=dict(color="#3b82f6", width=3), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=agg["k_shot"], y=agg["top5_accuracy_mean"], error_y=dict(type="data", array=agg["top5_accuracy_ci"]), mode="lines+markers", name="Top-5", line=dict(color="#10b981", width=2, dash="dash"), marker=dict(size=8, symbol="diamond")))
    if bl_blind is not None:
        fig.add_trace(go.Scatter(x=[agg["k_shot"].min(), agg["k_shot"].max()], y=[bl_blind] * 2, mode="lines", name="Zero-Shot", line=dict(color="#ef4444", width=2, dash="dot")))
    if bl_closed is not None:
        fig.add_trace(go.Scatter(x=[agg["k_shot"].min(), agg["k_shot"].max()], y=[bl_closed] * 2, mode="lines", name="Closed-Set", line=dict(color="#f59e0b", width=2, dash="dot")))
    fig.update_layout(title=f"Learning Curve - {rank.capitalize()}", xaxis_title="k-shot", yaxis_title="Accuracy", yaxis_tickformat=".0%", template="plotly_white", height=500)
    return fig


def _generate_learning_curve_report(
    agg: pd.DataFrame, marginal: pd.DataFrame, rank: str, output_dir: Path,
    baseline_blind=None, baseline_closed=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_lc = _build_learning_curve_fig(agg, rank, baseline_blind, baseline_closed)
    lc_html = pio.to_html(fig_lc, include_plotlyjs="cdn", div_id="learning-curve")

    # Marginal improvement bar chart
    marg_html = ""
    if len(marginal) > 0:
        fig_m = go.Figure()
        labels = [f"{int(r['k_from'])}->{int(r['k_to'])}" for _, r in marginal.iterrows()]
        fig_m.add_trace(go.Bar(x=labels, y=marginal["improvement"], marker_color="#3b82f6"))
        fig_m.update_layout(title=f"Marginal Improvement - {rank.capitalize()}", xaxis_title="Transition", yaxis_title="Accuracy Gain", yaxis_tickformat=".1%", template="plotly_white", height=400)
        marg_html = pio.to_html(fig_m, include_plotlyjs=False, div_id="marginal-improvement")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Learning Curve - {rank.capitalize()}</title>
<style>body{{font-family:'Helvetica Neue',Arial,sans-serif;margin:0;padding:0;background:#f7f7f7;}}
header{{background:#111827;color:#fff;padding:24px;}}h1{{margin:0;font-size:28px;}}
section{{margin:24px auto;max-width:1200px;background:#fff;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.08);border-radius:12px;}}</style>
</head><body><header><h1>Learning Curve Analysis - {rank.capitalize()}</h1></header><main>
<section>{lc_html}</section>
{"<section>" + marg_html + "</section>" if marg_html else ""}
</main></body></html>"""

    output_path = output_dir / "learning_curve_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info("Report saved to %s", output_path)


# =========================================================================
# CLI
# =========================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # single subcommand
    sp_single = subparsers.add_parser("single", help="Generate report for a single experiment")
    sp_single.add_argument("--experiment-dir", type=Path, required=True, help="Directory containing experiment outputs")
    sp_single.add_argument("--rank", choices=["family", "genus", "species"], default=None, help="Taxonomic rank")
    sp_single.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    sp_single.add_argument("--top-k", type=int, default=0, help="Show only top-k labels (0=all)")
    sp_single.add_argument("--baseline-blind", type=float, default=None)
    sp_single.add_argument("--baseline-closed", type=float, default=None)

    # learning-curve subcommand
    sp_lc = subparsers.add_parser("learning-curve", help="Generate learning curve report")
    sp_lc.add_argument("--rank", choices=["family", "genus", "species"], required=True, help="Taxonomic rank")
    sp_lc.add_argument("--batch-summary", type=Path, default=None, help="Path to batch summary JSON")
    sp_lc.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    sp_lc.add_argument("--baseline-blind", type=float, default=None)
    sp_lc.add_argument("--baseline-closed", type=float, default=None)

    # pipeline subcommand
    sp_pipe = subparsers.add_parser("pipeline", help="Generate HTML report from pipeline result JSON")
    sp_pipe.add_argument("--result-json", type=Path, required=True, help="Path to pipeline result JSON file")
    sp_pipe.add_argument("--output", type=Path, default=None, help="Output HTML path (default: same dir as JSON)")

    args = parser.parse_args()

    if args.command == "single":
        if args.rank is None:
            args.rank = _auto_detect_rank(args.experiment_dir)
        if args.output_dir is None:
            args.output_dir = args.experiment_dir / "visual"
        try:
            _generate_single_report(args.experiment_dir, args.rank, args.output_dir, args.top_k, args.baseline_blind, args.baseline_closed)
        except FileNotFoundError as e:
            logging.error("Error: %s", e)
            return 1

    elif args.command == "learning-curve":
        if args.batch_summary is None:
            try:
                args.batch_summary = _auto_discover_batch_summary(args.rank)
            except FileNotFoundError as e:
                logging.error("%s", e)
                return 1

        if args.output_dir is None:
            args.output_dir = args.batch_summary.parent / "results" / args.rank / "learning_curves"

        batch = _load_batch_results(args.batch_summary)
        if not batch["experiments"]:
            logging.error("No successful experiments found")
            return 1

        agg = _aggregate_across_seeds(batch["experiments"], batch["k_shots"])
        marginal = _compute_marginal(agg)
        _generate_learning_curve_report(agg, marginal, args.rank, args.output_dir, args.baseline_blind, args.baseline_closed)

    elif args.command == "pipeline":
        result_path = args.result_json
        if not result_path.exists():
            logging.error("Result file not found: %s", result_path)
            return 1

        result = json.loads(result_path.read_text())

        from seedlearn.reporting.pipeline_html import generate_pipeline_report

        html = generate_pipeline_report(result)

        output_path = args.output or result_path.with_suffix(".html")
        output_path.write_text(html, encoding="utf-8")
        logging.info("Pipeline report saved to %s (%.2f MB)", output_path, output_path.stat().st_size / 1024 / 1024)

    return 0


if __name__ == "__main__":
    sys.exit(main())
