"""HTML report generation for SimpleShot few-shot learning experiments.

This module consolidates chart builders (Plotly), HTML/CSS templates, and
interpretation text into a single reporting surface.  It merges the original
``visualization/plotly_utils.py``, ``visualization/html_templates.py``, and
``visualization/interpretation.py`` files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =========================================================================
# Plotly chart builders
# =========================================================================

def build_summary_table(metrics: dict[str, Any], k_shot: int) -> go.Figure:
    """Create summary metrics table for experiment overview.

    Args:
        metrics: Dictionary with accuracy, F1, timing, etc.
        k_shot: Number of shots per class (for display).

    Returns:
        Plotly table figure.
    """
    keys = [
        "accuracy",
        "top5_accuracy",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "num_test_images",
        "num_classes",
        "k_shot",
        "support_samples",
        "support_seed",
        "split_seed",
    ]
    summary_rows = []
    for key in keys:
        if key == "k_shot":
            value = k_shot
        elif key in metrics:
            value = metrics[key]
        else:
            continue
        summary_rows.append({"Metric": key, "Value": value})

    df = pd.DataFrame(summary_rows)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[col] for col in df.columns]),
            )
        ]
    )
    fig.update_layout(
        title=f"Experiment Summary - {k_shot}-shot Few-Shot Learning",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_support_set_distribution(support_set: dict[str, Any]) -> go.Figure:
    """Create bar chart showing support set class distribution.

    Args:
        support_set: Support set metadata from support_set.json.

    Returns:
        Plotly bar chart figure.
    """
    per_class = support_set.get("per_class_support", {})
    class_counts = {label: len(paths) for label, paths in per_class.items()}

    df = pd.DataFrame([
        {"label": label, "count": count}
        for label, count in sorted(class_counts.items(), key=lambda x: -x[1])
    ])

    fig = px.bar(
        df,
        x="count",
        y="label",
        orientation="h",
        title=f"Support Set Distribution ({support_set.get('k_shot', '?')}-shot)",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=40, t=60, b=40),
        xaxis=dict(title="Samples per Class"),
        yaxis_title="Class Label",
    )
    fig.update_traces(marker_color="#00cc96")
    return fig


def build_label_support_chart(predictions: pd.DataFrame, top_k: int = 0) -> go.Figure:
    """Create bar chart showing test set label distribution.

    Args:
        predictions: DataFrame with target_label column.
        top_k: Show only top-k labels by support (0 = all).

    Returns:
        Plotly bar chart figure.
    """
    label_counts = (
        predictions.groupby("target_label", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    label_counts.insert(0, "rank", label_counts.index + 1)
    top_counts = label_counts if top_k <= 0 else label_counts.head(top_k)
    title_suffix = "All Labels" if top_k <= 0 else f"Top {top_k} Labels"

    fig = px.bar(
        top_counts,
        x="count",
        y="target_label",
        orientation="h",
        title=f"Test Set Label Support \u2014 {title_suffix}",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=40, t=60, b=40),
    )
    return fig


def build_per_label_metrics(report_df: pd.DataFrame, top_k: int = 0) -> go.Figure:
    """Create grouped bar chart with precision/recall/F1 per class.

    Args:
        report_df: DataFrame with precision, recall, f1 columns indexed by label.
        top_k: Show only top-k labels by support (0 = all).

    Returns:
        Plotly grouped bar chart figure.
    """
    per_label = report_df[
        ~report_df.index.isin(["accuracy", "macro avg", "weighted avg", "micro avg"])
    ].copy()
    per_label = per_label.sort_values("support", ascending=False)

    if top_k > 0:
        per_label = per_label.head(top_k)

    fig = go.Figure()
    for metric in ["precision", "recall", "f1"]:
        display_name = "F1-score" if metric == "f1" else metric.capitalize()
        fig.add_bar(x=per_label[metric], y=per_label.index, name=display_name, orientation="h")

    title_suffix = f" (Top {top_k} by Support)" if top_k > 0 else ""
    fig.update_layout(
        title=f"Per-Class Metrics{title_suffix}",
        barmode="group",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=40, t=60, b=40),
    )
    return fig


def build_confusion_matrix(cm_df: pd.DataFrame, top_k: int = 0) -> go.Figure:
    """Create heatmap visualization of confusion matrix.

    Args:
        cm_df: Confusion matrix as DataFrame with labels as index/columns.
        top_k: Show only top-k labels by support (0 = all).

    Returns:
        Plotly heatmap figure.
    """
    if top_k > 0:
        row_sums = cm_df.sum(axis=1).sort_values(ascending=False)
        top_labels = row_sums.head(top_k).index
        cm_df = cm_df.loc[top_labels, top_labels]

    fig = px.imshow(
        cm_df.values,
        x=cm_df.columns,
        y=cm_df.index,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix" + (f" (Top {top_k} Classes)" if top_k > 0 else ""),
    )
    fig.update_layout(
        xaxis=dict(side="bottom"),
        margin=dict(l=160, r=40, t=60, b=100),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def build_top_errors(predictions: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create bar chart showing most common error pairs.

    Args:
        predictions: DataFrame with target_label and prediction_label columns.
        top_n: Number of top error pairs to show.

    Returns:
        Plotly bar chart figure.
    """
    errors = predictions[~predictions["is_correct"]].copy()

    error_counts = (
        errors.groupby(["target_label", "prediction_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    error_counts["error_pair"] = (
        error_counts["target_label"] + " \u2192 " + error_counts["prediction_label"]
    )

    fig = px.bar(
        error_counts,
        x="count",
        y="error_pair",
        orientation="h",
        title=f"Top {top_n} Most Common Error Pairs",
        labels={"count": "Error Count", "error_pair": "True \u2192 Predicted"},
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=240, r=40, t=60, b=40),
    )
    fig.update_traces(marker_color="#EF553B")
    return fig


def build_support_vs_performance(
    per_class_metrics: pd.DataFrame,
    support_set_classes: list[str],
) -> go.Figure:
    """Create scatter plot showing support set size vs performance per class.

    Args:
        per_class_metrics: DataFrame with f1 and support indexed by label.
        support_set_classes: List of classes included in support set.

    Returns:
        Plotly scatter plot figure.
    """
    metrics = per_class_metrics.loc[
        per_class_metrics.index.isin(support_set_classes)
    ].copy()

    fig = px.scatter(
        metrics,
        x="support",
        y="f1",
        hover_data=["precision", "recall"],
        title="Support vs Performance Analysis",
        labels={"support": "Test Set Support", "f1": "F1-Score"},
    )

    fig.update_traces(marker=dict(size=10, opacity=0.6))

    for idx, row in metrics.iterrows():
        fig.add_annotation(
            x=row["support"],
            y=row["f1"],
            text=str(idx),
            showarrow=False,
            yshift=10,
            font=dict(size=8),
        )

    fig.update_layout(margin=dict(l=60, r=40, t=60, b=60))
    return fig


# =========================================================================
# HTML / CSS templates
# =========================================================================

def get_html_header(rank: str, k_shot: int) -> str:
    """Generate HTML header with title and CSS styling.

    Args:
        rank: Taxonomic rank (family, genus, species).
        k_shot: Number of shots per class.

    Returns:
        HTML header string with embedded CSS.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SimpleShot {k_shot}-shot Report - {rank.capitalize()}</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; background-color: #f7f7f7; }}
    header {{ background-color: #111827; color: #fff; padding: 24px; }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ font-size: 22px; color: #111827; margin-top: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
    section {{ margin: 24px auto; max-width: 1200px; background: #fff; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); border-radius: 12px; }}
    .interpretation, .conclusions {{ margin-bottom: 32px; padding: 20px; background-color: #f9fafb; border-left: 4px solid #3b82f6; border-radius: 8px; }}
    .interpretation p, .conclusions p {{ line-height: 1.6; margin: 12px 0; }}
    .chart-help {{ margin: 16px 0; padding: 12px 16px; background-color: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 6px; font-size: 14px; line-height: 1.5; }}
    .chart-help em {{ color: #92400e; font-weight: 600; font-style: normal; }}
    .figure {{ margin-bottom: 32px; }}
    .highlight {{ background-color: #fef3c7; padding: 2px 6px; border-radius: 3px; }}
    .metric-good {{ color: #059669; font-weight: bold; }}
    .metric-poor {{ color: #dc2626; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>SimpleShot {k_shot}-shot Few-Shot Learning Report - {rank.capitalize()} Level</h1>
    <p>Nearest-prototype classification with BioCLIP embeddings</p>
  </header>
  <main>
"""


def get_html_footer() -> str:
    """Generate HTML footer closing tags.

    Returns:
        HTML footer string.
    """
    return """
  </main>
</body>
</html>
"""


def get_evaluation_context(
    rank: str,
    k_shot: int,
    metrics: dict[str, Any],
    num_classes: int,
    total_images: int,
) -> str:
    """Generate contextual interpretation of the experiment setup.

    Args:
        rank: Taxonomic rank.
        k_shot: Number of shots per class.
        metrics: Dictionary with accuracy and other metrics.
        num_classes: Number of classes.
        total_images: Total number of test images.

    Returns:
        HTML string with interpretation section.
    """
    accuracy = metrics.get("accuracy", 0.0)
    top5_accuracy = metrics.get("top5_accuracy", 0.0)
    random_baseline = 1.0 / num_classes if num_classes > 0 else 0.0
    improvement_vs_random = accuracy / random_baseline if random_baseline > 0 else 0.0

    rank_description = {
        "family": "Family-level classification groups species into broad taxonomic families (e.g., Fabaceae, Rubiaceae). This is the coarsest taxonomic level tested.",
        "genus": "Genus-level classification is finer than family but coarser than species (e.g., Solanum, Piper). Moderate taxonomic specificity.",
        "species": "Species-level classification is the finest taxonomic level (e.g., Solanum lycopersicum). This is the most challenging classification task.",
    }

    if accuracy > 0.5:
        performance_class = "metric-good"
        performance_text = "strong performance"
    elif accuracy > 0.3:
        performance_class = "metric-good"
        performance_text = "moderate performance"
    else:
        performance_class = "metric-poor"
        performance_text = "below expectations"

    return f"""
    <section>
      <div class="interpretation">
        <h2>Evaluation Context</h2>
        <p><strong>Taxonomic Rank:</strong> {rank.capitalize()}</p>
        <p>{rank_description.get(rank, "Custom taxonomic rank.")}</p>

        <p><strong>Few-Shot Learning Setup:</strong> This is a <em>{k_shot}-shot</em> evaluation where SimpleShot learns
        to classify seedlings using only <span class="highlight">{k_shot} examples per class</span> for training. The model
        computes class prototypes (centroids) from BioCLIP embeddings and classifies test images by finding the nearest prototype.</p>

        <p><strong>Dataset:</strong> {total_images:,} seedling images across {num_classes} {rank}-level classes.</p>

        <p><strong>Random Baseline:</strong> {random_baseline:.2%} (1/{num_classes} classes) - what we'd expect from random guessing.</p>

        <p><strong>Model Performance:</strong> <span class="{performance_class}">{accuracy:.2%} Top-1 accuracy</span> ({improvement_vs_random:.1f}x better than random),
        <span class="{performance_class}">{top5_accuracy:.2%} Top-5 accuracy</span>. This represents <em>{performance_text}</em> for {k_shot}-shot learning.</p>
      </div>
    </section>
"""


def get_support_set_section(k_shot: int, num_classes: int, support_seed: int) -> str:
    """Generate interpretation section for support set analysis.

    Args:
        k_shot: Number of shots per class.
        num_classes: Number of classes.
        support_seed: Random seed used for support set sampling.

    Returns:
        HTML string with support set interpretation.
    """
    total_support = k_shot * num_classes

    return f"""
    <section>
      <div class="interpretation">
        <h2>Support Set Analysis</h2>
        <p><strong>Support Set Size:</strong> {total_support:,} images ({k_shot} per class x {num_classes} classes)</p>

        <p><strong>Sampling Strategy:</strong> Support examples were randomly sampled from the training set using
        seed {support_seed} to ensure reproducibility. Each class received exactly {k_shot} example(s).</p>

        <p><strong>Key Insight:</strong> The support set distribution chart below verifies that all classes received
        their allocated {k_shot} example(s). Any deviation would indicate a sampling issue requiring investigation.</p>

        <p><strong>Impact on Performance:</strong> In few-shot learning, the quality and representativeness of support
        examples significantly impact classification accuracy. Classes with more diverse or prototypical support examples
        tend to perform better.</p>
      </div>
    </section>
"""


def get_conclusions_section(
    rank: str,
    k_shot: int,
    accuracy: float,
    top5_accuracy: float,
    baseline_blind_accuracy: float | None = None,
    baseline_closed_accuracy: float | None = None,
) -> str:
    """Generate conclusions section with actionable recommendations.

    Args:
        rank: Taxonomic rank.
        k_shot: Number of shots per class.
        accuracy: Top-1 accuracy achieved.
        top5_accuracy: Top-5 accuracy achieved.
        baseline_blind_accuracy: Blind (zero-shot) accuracy for comparison.
        baseline_closed_accuracy: Closed-set accuracy for comparison.

    Returns:
        HTML string with conclusions section.
    """
    improvements = []
    if baseline_blind_accuracy is not None:
        improvement = ((accuracy - baseline_blind_accuracy) / baseline_blind_accuracy * 100)
        if accuracy > baseline_blind_accuracy:
            improvements.append(f"<li><strong>{improvement:.1f}% improvement</strong> over zero-shot baseline ({baseline_blind_accuracy:.2%})</li>")

    if baseline_closed_accuracy is not None:
        improvement = ((accuracy - baseline_closed_accuracy) / baseline_closed_accuracy * 100)
        if accuracy > baseline_closed_accuracy:
            improvements.append(f"<li><strong>{improvement:.1f}% improvement</strong> over closed-set baseline ({baseline_closed_accuracy:.2%})</li>")

    improvements_text = "<ul>" + "".join(improvements) + "</ul>" if improvements else "<p>No baseline comparisons available.</p>"

    if accuracy < 0.3:
        recommendations = """
        <p><strong>Recommendations for Improvement:</strong></p>
        <ul>
          <li>Increase k-shot value (try k=10, 20, or 50 if data allows)</li>
          <li>Review support set quality - ensure representative examples</li>
          <li>Consider feature normalization techniques</li>
          <li>Explore advanced few-shot methods (Prototypical Networks, MAML)</li>
        </ul>
        """
    elif accuracy < 0.5:
        recommendations = """
        <p><strong>Next Steps:</strong></p>
        <ul>
          <li>Experiment with higher k-shot values to find performance plateau</li>
          <li>Analyze per-class performance to identify challenging classes</li>
          <li>Consider ensemble methods or metric learning approaches</li>
        </ul>
        """
    else:
        recommendations = f"""
        <p><strong>Success Factors:</strong></p>
        <ul>
          <li>{k_shot}-shot learning achieved strong performance for {rank}-level classification</li>
          <li>BioCLIP embeddings provide effective feature representations for seedlings</li>
          <li>SimpleShot's nearest-prototype approach is well-suited to this task</li>
        </ul>
        """

    return f"""
    <section>
      <div class="conclusions">
        <h2>Conclusions & Recommendations</h2>
        <p><strong>Baseline Comparison:</strong></p>
        {improvements_text}

        <p><strong>Top-5 Accuracy ({top5_accuracy:.2%}):</strong> This metric shows that the correct class appears in
        the top 5 predictions more frequently, suggesting the model captures meaningful similarities even when the
        top prediction is incorrect.</p>

        {recommendations}

        <p><strong>Reproducibility:</strong> This experiment can be reproduced using the support_set.json file which
        contains the exact images used for each class prototype.</p>
      </div>
    </section>
"""


def get_chart_interpretation(chart_type: str, k_shot: int = 5, rank: str = "family") -> str:
    """Generate concise interpretation text for each chart type.

    Args:
        chart_type: Type of chart.
        k_shot: Number of shots per class.
        rank: Taxonomic rank.

    Returns:
        HTML string with interpretation paragraph.
    """
    interpretations = {
        "summary": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This table summarizes the experiment's key metrics. <strong>Top-1 accuracy</strong> is
            the primary performance measure (correct prediction on first try). <strong>Top-5 accuracy</strong> shows if the
            correct class appears in the top 5 predictions. <strong>k_shot={k_shot}</strong> indicates we used {k_shot} training
            example(s) per class. <strong>Support samples</strong> is the total number of training examples used
            ({k_shot} x number of classes).</p>
            </div>
        """,
        "support_dist": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This chart verifies the support set sampling. <strong>Each class should have exactly
            {k_shot} sample(s)</strong> (horizontal bars should all be the same length). Any deviation indicates a sampling error.
            This is a few-shot learning specific check to ensure fair comparison across classes.</p>
            </div>
        """,
        "label_support": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This shows <strong>how many test images</strong> exist for each {rank}. Longer bars
            mean more test examples for that class. This helps identify if poor performance on a class might be due to limited
            test data (though performance should ideally be independent of test set size in few-shot learning).</p>
            </div>
        """,
        "per_class": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This grouped bar chart shows three metrics per class: <strong style="color:#636EFA;">Precision</strong>
            (of all predictions for this class, how many were correct?), <strong style="color:#EF553B;">Recall</strong> (of all
            true instances of this class, how many did we find?), and <strong style="color:#00CC96;">F1-score</strong> (harmonic
            mean of precision and recall). <strong>Longer bars = better performance</strong>. Look for classes with consistently
            short bars to identify challenging classification cases.</p>
            </div>
        """,
        "confusion": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> Each cell shows how many times class Y (row) was predicted as class X (column).
            <strong>Bright diagonal cells = correct predictions</strong> (good!). <strong>Bright off-diagonal cells = common
            confusions</strong> (bad). This reveals which {rank}s are visually similar to the model. Hover over cells to see
            exact counts.</p>
            </div>
        """,
        "errors": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This shows the <strong>most frequent misclassifications</strong>. Format: "True Class
            -> Predicted Class (Count)". These error pairs reveal systematic confusion patterns. Investigate the longest bars to
            understand which {rank}s the model struggles to distinguish. This can inform data collection or feature engineering
            priorities.</p>
            </div>
        """,
        "support_perf": f"""
            <div class="chart-help">
            <p><em>How to interpret:</em> This scatter plot correlates <strong>test set size</strong> (x-axis) with
            <strong>F1-score</strong> (y-axis) for each class. In ideal few-shot learning, performance should be independent
            of test set size (no correlation). <strong>Positive correlation</strong> suggests the model benefits from more
            examples. <strong>Negative correlation</strong> is unusual and may indicate rare classes have more distinctive
            features. Hover over points to see class names and exact metrics.</p>
            </div>
        """,
    }

    return interpretations.get(chart_type, "")


def wrap_plotly_div(
    plotly_html: str,
    section_title: str | None = None,
    chart_type: str | None = None,
    k_shot: int = 5,
    rank: str = "family",
) -> str:
    """Wrap a Plotly figure HTML in a section container with interpretation.

    Args:
        plotly_html: HTML string from fig.to_html().
        section_title: Optional section title.
        chart_type: Optional chart type for auto-generated interpretation.
        k_shot: Number of shots per class (for interpretation).
        rank: Taxonomic rank (for interpretation).

    Returns:
        HTML string with wrapped figure and interpretation.
    """
    title_html = f"<h2>{section_title}</h2>" if section_title else ""
    interpretation = get_chart_interpretation(chart_type, k_shot, rank) if chart_type else ""

    return f"""
    <section>
      {title_html}
      {interpretation}
      <div class="figure">
        {plotly_html}
      </div>
    </section>
"""


# =========================================================================
# Interpretation text generators
# =========================================================================

def interpret_accuracy(
    accuracy: float,
    k_shot: int,
    rank: str,
    num_classes: int,
) -> str:
    """Generate natural language interpretation of accuracy results.

    Args:
        accuracy: Top-1 accuracy (0-1).
        k_shot: Number of shots per class.
        rank: Taxonomic rank.
        num_classes: Number of classes.

    Returns:
        Human-readable interpretation string.
    """
    random_baseline = 1.0 / num_classes
    improvement = accuracy / random_baseline if random_baseline > 0 else 0.0

    if k_shot >= 20:
        excellent, good, moderate = 0.60, 0.45, 0.30
    elif k_shot >= 10:
        excellent, good, moderate = 0.55, 0.40, 0.25
    elif k_shot >= 5:
        excellent, good, moderate = 0.50, 0.35, 0.20
    else:
        excellent, good, moderate = 0.40, 0.25, 0.15

    if accuracy >= excellent:
        assessment = "excellent"
        implication = f"The model demonstrates strong few-shot learning capability with only {k_shot} examples per class."
    elif accuracy >= good:
        assessment = "good"
        implication = f"The model shows promising performance for {k_shot}-shot learning, though there's room for improvement."
    elif accuracy >= moderate:
        assessment = "moderate"
        implication = f"The model performs reasonably given the limited training data ({k_shot} examples per class)."
    else:
        assessment = "below expectations"
        implication = f"Performance is lower than expected for {k_shot}-shot learning. Consider increasing k or reviewing support set quality."

    return (
        f"With {accuracy:.2%} Top-1 accuracy ({improvement:.1f}x better than random guessing), the model achieves "
        f"<strong>{assessment}</strong> performance for {rank}-level classification. {implication}"
    )


def interpret_top5_accuracy(top1_accuracy: float, top5_accuracy: float) -> str:
    """Generate interpretation of Top-5 accuracy relative to Top-1.

    Args:
        top1_accuracy: Top-1 accuracy (0-1).
        top5_accuracy: Top-5 accuracy (0-1).

    Returns:
        Human-readable interpretation string.
    """
    if top5_accuracy == 0:
        return "Top-5 accuracy data not available."

    gap = top5_accuracy - top1_accuracy
    relative_improvement = (gap / top1_accuracy * 100) if top1_accuracy > 0 else 0

    if gap > 0.25:
        assessment = "significant gap"
        implication = "This suggests the model captures meaningful class similarities, but struggles to rank the correct class first. The large Top-5 improvement indicates the correct answer is often in the top predictions."
    elif gap > 0.15:
        assessment = "moderate gap"
        implication = "The model frequently places the correct class in the top 5 predictions even when not ranked first. This is typical for challenging few-shot classification tasks."
    else:
        assessment = "small gap"
        implication = "When the model is confident enough to place a class in the top 5, it's usually the top prediction. This indicates strong discriminative power."

    return (
        f"The Top-5 accuracy ({top5_accuracy:.2%}) shows a <strong>{assessment}</strong> compared to Top-1 ({top1_accuracy:.2%}), "
        f"with a {gap:.1%} difference (+{relative_improvement:.1f}%). {implication}"
    )


def interpret_confusion_patterns(
    top_errors: list[tuple[str, str, int]],
    rank: str,
) -> str:
    """Generate interpretation of top confusion patterns.

    Args:
        top_errors: List of (true_label, predicted_label, count) tuples.
        rank: Taxonomic rank.

    Returns:
        Human-readable interpretation string.
    """
    if not top_errors:
        return "No significant error patterns detected - predictions are well-distributed."

    most_common = top_errors[0]
    true_label, pred_label, count = most_common

    total_top_errors = sum(err[2] for err in top_errors[:5])

    if len(top_errors) > 1 and top_errors[0][2] == top_errors[1][2]:
        pattern = "multiple confusion pairs with similar frequencies"
        implication = f"No single dominant error pattern. The model's mistakes are distributed across several {rank}-level confusions."
    else:
        pattern = f"{true_label} -> {pred_label} ({count} occurrences)"
        implication = f"This is the most common confusion, suggesting visual similarity or insufficient support examples for distinguishing these {rank}s."

    return (
        f"<p>The most frequent error pattern is <strong>{pattern}</strong>. {implication}</p>"
        f"<p>The top 5 error pairs account for {total_top_errors} total misclassifications, providing insight into "
        f"which {rank}-level distinctions are most challenging for the model.</p>"
    )


def interpret_per_class_performance(
    best_classes: list[tuple[str, float]],
    worst_classes: list[tuple[str, float]],
    rank: str,
) -> str:
    """Generate interpretation of per-class performance variation.

    Args:
        best_classes: List of (class_name, f1_score) for top performers.
        worst_classes: List of (class_name, f1_score) for bottom performers.
        rank: Taxonomic rank.

    Returns:
        Human-readable interpretation string.
    """
    if not best_classes or not worst_classes:
        return "Per-class performance data not available."

    best_avg = sum(f1 for _, f1 in best_classes) / len(best_classes)
    worst_avg = sum(f1 for _, f1 in worst_classes) / len(worst_classes)
    performance_gap = best_avg - worst_avg

    best_names = ", ".join([name for name, _ in best_classes[:3]])
    worst_names = ", ".join([name for name, _ in worst_classes[:3]])

    if performance_gap > 0.4:
        assessment = "large performance variation"
        implication = "This suggests some classes have highly distinctive features while others are more challenging to classify. Consider reviewing support set quality for underperforming classes."
    elif performance_gap > 0.2:
        assessment = "moderate performance variation"
        implication = "Some classes perform notably better than others, which is typical in few-shot learning. This may reflect inherent visual similarity among certain taxonomic groups."
    else:
        assessment = "consistent performance"
        implication = "Performance is relatively uniform across classes, indicating the model generalizes well and support set quality is consistent."

    return (
        f"<p><strong>Best performing {rank}s:</strong> {best_names} (avg F1: {best_avg:.2%})</p>"
        f"<p><strong>Worst performing {rank}s:</strong> {worst_names} (avg F1: {worst_avg:.2%})</p>"
        f"<p>There is <strong>{assessment}</strong> across classes (gap: {performance_gap:.1%}). {implication}</p>"
    )


def interpret_support_vs_performance(
    correlation: float | None,
    k_shot: int,
) -> str:
    """Generate interpretation of support set size vs performance correlation.

    Args:
        correlation: Pearson correlation between test support and F1-score (or None).
        k_shot: Number of shots per class.

    Returns:
        Human-readable interpretation string.
    """
    if correlation is None:
        return "Support vs performance correlation data not available."

    if correlation > 0.3:
        assessment = "positive correlation"
        implication = f"Classes with more test examples tend to perform better, which is expected. However, with only {k_shot} training examples per class, performance should ideally be less dependent on test set size."
    elif correlation > 0.1:
        assessment = "weak positive correlation"
        implication = "Performance is only slightly influenced by test set size, suggesting the model generalizes reasonably well across classes regardless of how many test examples exist."
    elif correlation > -0.1:
        assessment = "no correlation"
        implication = f"Performance is independent of test set size, which is ideal for {k_shot}-shot learning. The model generalizes equally well to rare and common classes."
    else:
        assessment = "weak negative correlation"
        implication = "Interestingly, some rare classes (fewer test examples) perform better. This may indicate that rare classes have more distinctive features or better support set quality."

    return (
        f"The correlation between test set support and F1-score is <strong>{correlation:.3f}</strong>, indicating "
        f"<strong>{assessment}</strong>. {implication}"
    )


def generate_key_findings(
    accuracy: float,
    top5_accuracy: float,
    k_shot: int,
    rank: str,
    num_classes: int,
    baseline_comparisons: dict[str, float] | None = None,
) -> list[str]:
    """Generate a list of key findings for executive summary.

    Args:
        accuracy: Top-1 accuracy.
        top5_accuracy: Top-5 accuracy.
        k_shot: Number of shots per class.
        rank: Taxonomic rank.
        num_classes: Number of classes.
        baseline_comparisons: Optional dict with 'blind' and 'closed_set' accuracies.

    Returns:
        List of finding strings (HTML formatted).
    """
    findings = []

    random_baseline = 1.0 / num_classes
    improvement = accuracy / random_baseline
    findings.append(
        f"<strong>{k_shot}-shot learning achieved {accuracy:.2%} Top-1 accuracy</strong> "
        f"({improvement:.1f}x better than random guessing)"
    )

    if top5_accuracy > 0:
        findings.append(
            f"<strong>Top-5 accuracy reached {top5_accuracy:.2%}</strong>, showing the model "
            f"captures meaningful class similarities"
        )

    if baseline_comparisons:
        if "blind" in baseline_comparisons:
            blind_acc = baseline_comparisons["blind"]
            if accuracy > blind_acc:
                improvement_pct = (accuracy - blind_acc) / blind_acc * 100
                findings.append(
                    f"<strong>Outperformed zero-shot baseline by {improvement_pct:.1f}%</strong> "
                    f"({blind_acc:.2%} -> {accuracy:.2%})"
                )

        if "closed_set" in baseline_comparisons:
            closed_acc = baseline_comparisons["closed_set"]
            if accuracy > closed_acc:
                improvement_pct = (accuracy - closed_acc) / closed_acc * 100
                findings.append(
                    f"<strong>Outperformed closed-set baseline by {improvement_pct:.1f}%</strong> "
                    f"({closed_acc:.2%} -> {accuracy:.2%})"
                )

    findings.append(
        f"Achieved this performance using only <strong>{k_shot} examples per class</strong> "
        f"({k_shot * num_classes:,} total training images)"
    )

    return findings
