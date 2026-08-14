"""HTML report generator for the 5-stage classification pipeline.

Produces an fmriprep-inspired per-specimen visual report with interactive
Plotly charts, stage-by-stage evidence sections, and a timing waterfall.

Follows the same patterns as ``html.py``: embedded CSS, string-based HTML
assembly, Plotly CDN for the first chart + ``include_plotlyjs=False`` for
the rest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


# =========================================================================
# Public API
# =========================================================================


def generate_pipeline_report(result_dict: dict[str, Any]) -> str:
    """Generate a complete HTML report from a pipeline result dict.

    Args:
        result_dict: Pipeline result dictionary (from JSON), with keys
            ``specimen_id``, ``image_paths``, ``stages``, ``total_elapsed_ms``.

    Returns:
        Complete HTML document string.
    """
    parts: list[str] = []
    stages = result_dict.get("stages", {})

    parts.append(_pipeline_css(result_dict))
    parts.append(_header_section(result_dict))
    parts.append(_morphology_section(stages.get("morphology", {})))
    parts.append(_classification_section(stages.get("classification", {})))
    parts.append(_trait_retrieval_section(stages.get("trait_retrieval", {})))
    parts.append(_evidence_section(stages.get("evidence_synthesis", {})))
    parts.append(_reasoning_section(stages.get("reasoning", {})))
    parts.append(_timing_section(result_dict))
    parts.append(_glossary_section())
    parts.append(_footer())

    return "\n".join(parts)


# =========================================================================
# CSS
# =========================================================================


def _pipeline_css(result: dict[str, Any]) -> str:
    """Generate HTML head with embedded CSS."""
    specimen = result.get("specimen_id", "Unknown")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Pipeline Report — {specimen}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; background-color: #f7f7f7; color: #1f2937; }}
    header {{ background-color: #111827; color: #fff; padding: 28px 32px; }}
    header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    header .meta {{ font-size: 14px; color: #9ca3af; }}
    header .meta span {{ margin-right: 24px; }}
    .badges {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600; }}
    .badge-ok {{ background: #065f46; color: #a7f3d0; }}
    .badge-skip {{ background: #374151; color: #9ca3af; }}
    .badge-error {{ background: #7f1d1d; color: #fca5a5; }}
    section {{ margin: 24px auto; max-width: 1200px; background: #fff; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); border-radius: 12px; }}
    h2 {{ font-size: 22px; color: #111827; margin-top: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
    h3 {{ font-size: 17px; color: #374151; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
    th {{ background: #f9fafb; font-weight: 600; color: #374151; }}
    tr:hover {{ background: #f3f4f6; }}
    .interpretation {{ margin: 16px 0; padding: 16px 20px; background-color: #f9fafb; border-left: 4px solid #3b82f6; border-radius: 8px; font-size: 14px; line-height: 1.65; }}
    .warning-box {{ margin: 12px 0; padding: 14px 18px; background-color: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px; font-size: 14px; }}
    .warning-box strong {{ color: #92400e; }}
    .result-card {{ margin: 16px 0; padding: 20px 24px; background: linear-gradient(135deg, #f0fdf4, #ecfdf5); border: 2px solid #86efac; border-radius: 12px; }}
    .result-card .taxon {{ font-size: 22px; font-weight: 700; color: #065f46; }}
    .result-card .detail {{ font-size: 15px; color: #374151; margin-top: 4px; }}
    .confidence-high {{ color: #059669; font-weight: bold; }}
    .confidence-medium {{ color: #d97706; font-weight: bold; }}
    .confidence-low {{ color: #dc2626; font-weight: bold; }}
    .margin-green {{ background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
    .margin-yellow {{ background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
    .margin-red {{ background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
    .skipped {{ padding: 32px; text-align: center; color: #9ca3af; background: #f3f4f6; border-radius: 8px; }}
    .figure {{ margin: 20px 0; }}
    details {{ margin: 12px 0; }}
    details summary {{ cursor: pointer; font-weight: 600; color: #374151; padding: 8px 0; }}
    details pre {{ background: #f9fafb; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }}
    .highlight-row {{ background-color: #fef3c7 !important; }}
    footer {{ text-align: center; padding: 24px; color: #9ca3af; font-size: 13px; }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 4px 0; }}
    .warning {{ margin: 12px 0; padding: 14px 18px; background-color: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px; font-size: 14px; }}
    .warning strong {{ color: #92400e; }}
    .metric-row {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin: 10px 0;
    }}
    .metric-badge {{
      display: inline-block;
      padding: 4px 12px;
      background: #f3f4f6;
      border-radius: 6px;
      font-size: 0.9em;
    }}
  </style>
</head>
<body>
"""


# =========================================================================
# Header
# =========================================================================


def _header_section(result: dict[str, Any]) -> str:
    """Banner with specimen ID, image count, timing, and stage badges."""
    specimen = result.get("specimen_id", "Unknown")
    n_images = len(result.get("image_paths", []))
    elapsed = result.get("total_elapsed_ms", 0)

    stages = result.get("stages", {})
    stage_names = [
        ("morphology", "Stage 1: Morphology"),
        ("classification", "Stage 2: Classification"),
        ("trait_retrieval", "Stage 3: Trait Retrieval"),
        ("evidence_synthesis", "Stage 4: Evidence Synthesis"),
        ("reasoning", "Stage 5: Reasoning"),
    ]

    badges = []
    for key, label in stage_names:
        sr = stages.get(key, {})
        if sr.get("skipped"):
            badges.append(f'<span class="badge badge-skip">{label}: SKIP</span>')
        elif sr.get("error"):
            badges.append(f'<span class="badge badge-error">{label}: ERROR</span>')
        else:
            badges.append(f'<span class="badge badge-ok">{label}: OK</span>')

    return f"""  <header>
    <h1>Pipeline Report — {specimen}</h1>
    <div class="meta">
      <span>{n_images} images</span>
      <span>Total: {elapsed / 1000:.1f}s</span>
    </div>
    <div class="badges">
      {"".join(badges)}
    </div>
  </header>
  <main>
"""


# =========================================================================
# Stage 1 — Morphology
# =========================================================================


def _morphology_section(stage: dict[str, Any]) -> str:
    """Morphological profile table with 24 traits."""
    if stage.get("skipped"):
        return _skipped_section("Morphological Profile (Stage 1)", "Stage 1 was skipped — no VLM morphology extraction performed.")

    if stage.get("error"):
        return _error_section("Morphological Profile (Stage 1)", stage["error"])

    data = stage.get("data", {})
    traits = data.get("traits", {})

    # Separate notes from trait table
    notes = traits.get("notes", "")
    trait_items = {k: v for k, v in traits.items() if k != "notes"}

    rows = []
    for trait_name, value in trait_items.items():
        # Extract the short value before the parenthetical
        short_val = _extract_short_value(str(value))
        rows.append(f"      <tr><td>{_escape(trait_name)}</td><td>{_escape(short_val)}</td></tr>")

    notes_html = ""
    if notes:
        notes_html = f"""
    <div class="interpretation">
      <strong>Observer Notes:</strong> {_escape(notes)}
    </div>"""

    raw = data.get("raw_response", "")
    raw_html = ""
    if raw:
        raw_html = f"""
    <details>
      <summary>Raw VLM Response</summary>
      <pre>{_escape(raw)}</pre>
    </details>"""

    return f"""
  <section>
    <h2>Morphological Profile (Stage 1)</h2>
    <div class="interpretation">
      A vision-language model examined {_stage_image_count(stage)} specimen images and extracted
      {len(trait_items)} morphological traits using a standardized botanical assessment form.
    </div>
    <table>
      <thead><tr><th>Trait</th><th>Observed Value</th></tr></thead>
      <tbody>
{"chr(10)".join(rows) if rows else ""}
      </tbody>
    </table>
{notes_html}
{raw_html}
  </section>
"""


# =========================================================================
# Stage 2 — Classification
# =========================================================================


def _multirank_classification_section(stage: dict[str, Any]) -> str:
    """Multi-rank visual classification: grouped chart, consistency, OOD gate."""
    data = stage.get("data", {})
    predictions_by_rank = data.get("predictions_by_rank", {})
    margin_by_rank = data.get("margin_by_rank", {})
    consistency = data.get("hierarchical_consistency", {})
    gate = data.get("confidence_gate", {})
    per_image = data.get("per_image_predictions", [])
    nearest = data.get("nearest_support", [])

    # --- Grouped bar chart: top-1 per rank ---
    chart_html = ""
    rank_colors = {"family": "#3b82f6", "genus": "#8b5cf6", "species": "#10b981"}
    if predictions_by_rank:
        ranks: list[str] = []
        top1_names: list[str] = []
        top1_confs: list[float] = []
        colors: list[str] = []
        for rank_name in ("family", "genus", "species"):
            preds = predictions_by_rank.get(rank_name, [])
            if preds:
                ranks.append(rank_name.capitalize())
                top1_names.append(preds[0]["rank_value"])
                top1_confs.append(preds[0]["softmax_score"] * 100)
                colors.append(rank_colors.get(rank_name, "#6b7280"))

        if ranks:
            fig = go.Figure(go.Bar(
                x=top1_confs, y=ranks, orientation="h",
                marker_color=colors,
                text=[f"{n} ({c:.1f}%)" for n, c in zip(top1_names, top1_confs)],
                textposition="outside",
            ))
            fig.update_layout(
                title="Multi-Rank Classification (Top-1 per Rank)",
                xaxis_title="Similarity Share (%)",
                xaxis=dict(range=[0, max(top1_confs) * 1.3]),
                margin=dict(l=100, r=180, t=50, b=40),
                height=250,
                template="plotly_white",
            )
            chart_html = f'<div class="figure">{pio.to_html(fig, include_plotlyjs="cdn", div_id="multirank-chart", full_html=False)}</div>'

    # --- Margin badges per rank ---
    margin_html = '<div class="metric-row">'
    for rank_name in ("family", "genus", "species"):
        m = margin_by_rank.get(rank_name)
        if m is not None:
            if m > 0.1:
                cls = "margin-green"
            elif m >= 0.03:
                cls = "margin-yellow"
            else:
                cls = "margin-red"
            margin_html += f'<span class="metric-badge">{rank_name.capitalize()} margin: <span class="{cls}">{m:.4f}</span></span> '
    margin_html += "</div>"

    # --- Consistency badge ---
    consistency_html = ""
    if consistency:
        is_consistent = consistency.get("consistent", True)
        if is_consistent:
            consistency_html = '<p><span class="badge badge-ok">Hierarchical Consistency: OK</span></p>'
        else:
            notes = consistency.get("notes", [])
            notes_html = "".join(f"<li>{_escape(n)}</li>" for n in notes)
            consistency_html = f"""
    <div class="warning">
      <strong>Hierarchical Inconsistency Detected</strong>
      <ul>{notes_html}</ul>
    </div>"""

    # --- OOD warning ---
    ood_html = ""
    gate_flags = gate.get("flags", [])
    if gate_flags:
        flag_items = "".join(f"<li>{_escape(f)}</li>" for f in gate_flags)
        ood_html = f"""
    <div class="warning">
      <strong>Out-of-Distribution Warning</strong>
      <p>Distance-based confidence gating detected potential OOD query:</p>
      <ul>{flag_items}</ul>
    </div>"""

    # --- Per-image table (same as single-rank) ---
    per_image_html = ""
    if per_image:
        all_same = len({p["top1_label"] for p in per_image}) == 1
        img_rows = []
        for p in per_image:
            fname = Path(p["image_path"]).name
            conf = p["top1_softmax_score"] * 100
            img_rows.append(f"      <tr><td>{_escape(fname)}</td><td>{_escape(p['top1_label'])}</td><td>{conf:.1f}%</td></tr>")
        agreement = "All images agree on top-1 prediction." if all_same else '<span class="confidence-low">Images disagree on top-1 prediction!</span>'
        per_image_html = f"""
    <h3>Per-Image Predictions</h3>
    <p>{agreement}</p>
    <table>
      <thead><tr><th>Image</th><th>Top-1 Label</th><th>Similarity Share</th></tr></thead>
      <tbody>
{chr(10).join(img_rows)}
      </tbody>
    </table>"""

    # --- Nearest support (same as single-rank) ---
    nearest_html = ""
    if nearest:
        nn_rows = []
        for n in nearest:
            specimen_id = _extract_specimen_id(n.get("image_path", ""))
            nn_rows.append(
                f"      <tr><td>{_escape(n['label'])}</td>"
                f"<td>{n['l2_distance']:.3f}</td>"
                f"<td>{n['cosine_similarity']:.3f}</td>"
                f"<td>{_escape(specimen_id)}</td></tr>"
            )
        nearest_html = f"""
    <h3>Nearest Support Images (k-NN)</h3>
    <table>
      <thead><tr><th>Label</th><th>L2 Distance</th><th>Cosine Sim.</th><th>Specimen</th></tr></thead>
      <tbody>
{chr(10).join(nn_rows)}
      </tbody>
    </table>"""

    # --- Per-rank detail tables ---
    rank_detail_html = ""
    for rank_name in ("family", "genus", "species"):
        preds = predictions_by_rank.get(rank_name, [])
        if len(preds) > 1:
            rows = []
            for p in preds:
                rows.append(
                    f"      <tr><td>#{p['rank_position']}</td>"
                    f"<td>{_escape(p['rank_value'])}</td>"
                    f"<td>{p['softmax_score']*100:.1f}%</td>"
                    f"<td>{p.get('l2_distance', 0):.3f}</td>"
                    f"<td>{p.get('cosine_similarity', 0):.3f}</td></tr>"
                )
            rank_detail_html += f"""
    <details>
      <summary>{rank_name.capitalize()} — all top-k predictions</summary>
      <table>
        <thead><tr><th>#</th><th>Taxon</th><th>Similarity Share</th><th>L2</th><th>Cosine</th></tr></thead>
        <tbody>
{chr(10).join(rows)}
        </tbody>
      </table>
    </details>"""

    return f"""
  <section>
    <h2>Visual Classification — Multi-Rank (Stage 2)</h2>
    <div class="interpretation">
      BioCLIP 2 embeddings classified simultaneously at family, genus, and species level.
      Each rank uses an independent SimpleShot classifier on the same 768-dim embedding.
      <strong>Similarity share</strong> divides 100% of total closeness across all candidate
      classes — the top class gets the largest slice, but values are low when candidates
      are visually similar. Focus on <em>rank ordering</em> and <em>decision margin</em>,
      not the absolute percentage.
    </div>
{chart_html}
{margin_html}
{consistency_html}
{ood_html}
{rank_detail_html}
{per_image_html}
{nearest_html}
  </section>
"""


def _classification_section(stage: dict[str, Any]) -> str:
    """Visual classification: top-k bar chart, margin, per-image, k-NN."""
    if stage.get("skipped"):
        return _skipped_section("Visual Classification (Stage 2)", "Stage 2 was skipped.")

    if stage.get("error"):
        return _error_section("Visual Classification (Stage 2)", stage["error"])

    data = stage.get("data", {})

    # Multi-rank dispatch
    if "predictions_by_rank" in data:
        return _multirank_classification_section(stage)

    # Single-rank (existing code continues below)
    predictions = data.get("predictions", [])
    margin = data.get("margin", 0)
    per_image = data.get("per_image_predictions", [])
    nearest = data.get("nearest_support", [])

    # --- Top-k bar chart ---
    chart_html = ""
    if predictions:
        labels = [p["rank_value"] for p in reversed(predictions)]
        confidences = [p["softmax_score"] * 100 for p in reversed(predictions)]
        colors = ["#3b82f6" if i == len(predictions) - 1 else "#93c5fd"
                  for i in range(len(predictions))]
        colors.reverse()

        fig = go.Figure(go.Bar(
            x=confidences, y=labels, orientation="h",
            marker_color=list(reversed(colors)),
            text=[f"{c:.1f}%" for c in reversed(confidences)],
            textposition="outside",
        ))
        fig.update_layout(
            title="Top-k Predictions (Similarity Share)",
            xaxis_title="Similarity Share (%)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=160, r=60, t=50, b=40),
            height=max(250, len(predictions) * 50 + 80),
            template="plotly_white",
        )
        chart_html = f'<div class="figure">{pio.to_html(fig, include_plotlyjs="cdn", div_id="topk-chart", full_html=False)}</div>'

    # --- Margin badge ---
    if margin > 0.1:
        margin_cls = "margin-green"
    elif margin >= 0.03:
        margin_cls = "margin-yellow"
    else:
        margin_cls = "margin-red"
    margin_html = f'<p>Decision margin: <span class="{margin_cls}">{margin:.4f}</span></p>'

    # --- Per-image table ---
    per_image_html = ""
    if per_image:
        all_same = len({p["top1_label"] for p in per_image}) == 1
        img_rows = []
        for p in per_image:
            fname = Path(p["image_path"]).name
            conf = p["top1_softmax_score"] * 100
            img_rows.append(f"      <tr><td>{_escape(fname)}</td><td>{_escape(p['top1_label'])}</td><td>{conf:.1f}%</td></tr>")

        agreement = "All images agree on top-1 prediction." if all_same else '<span class="confidence-low">Images disagree on top-1 prediction!</span>'
        per_image_html = f"""
    <h3>Per-Image Predictions</h3>
    <p>{agreement}</p>
    <table>
      <thead><tr><th>Image</th><th>Top-1 Label</th><th>Similarity Share</th></tr></thead>
      <tbody>
{chr(10).join(img_rows)}
      </tbody>
    </table>"""

    # --- Nearest support table ---
    nearest_html = ""
    if nearest:
        nn_rows = []
        for n in nearest:
            specimen_id = _extract_specimen_id(n.get("image_path", ""))
            nn_rows.append(
                f"      <tr><td>{_escape(n['label'])}</td>"
                f"<td>{n['l2_distance']:.3f}</td>"
                f"<td>{n['cosine_similarity']:.3f}</td>"
                f"<td>{_escape(specimen_id)}</td></tr>"
            )
        nearest_html = f"""
    <h3>Nearest Support Images (k-NN)</h3>
    <table>
      <thead><tr><th>Label</th><th>L2 Distance</th><th>Cosine Sim.</th><th>Specimen</th></tr></thead>
      <tbody>
{chr(10).join(nn_rows)}
      </tbody>
    </table>"""

    return f"""
  <section>
    <h2>Visual Classification (Stage 2)</h2>
    <div class="interpretation">
      BioCLIP 2 image embeddings (768-dim) were pooled across specimen images and classified
      via SimpleShot nearest-centroid against the support set. <strong>Similarity share</strong>
      divides 100% of total closeness across all candidate classes. The <em>decision margin</em>
      is the difference in similarity share between the #1 and #2 predictions — larger = stronger separation.
    </div>
{chart_html}
{margin_html}
{per_image_html}
{nearest_html}
  </section>
"""


# =========================================================================
# Stage 3 — Trait Retrieval
# =========================================================================


def _trait_retrieval_section(stage: dict[str, Any]) -> str:
    """RAG matches table and convergence analysis."""
    if stage.get("skipped"):
        return _skipped_section("Literature Evidence (Stage 3)", "Stage 3 was skipped.")

    if stage.get("error"):
        return _error_section("Literature Evidence (Stage 3)", stage["error"])

    data = stage.get("data", {})
    rag_matches = data.get("rag_matches", [])
    convergence = data.get("convergence", [])

    # --- RAG matches bar chart (top 10) ---
    chart_html = ""
    top_matches = rag_matches[:10]
    if top_matches:
        labels = [f"{m['taxon']} ({m['rank']})" for m in reversed(top_matches)]
        scores = [m["score"] for m in reversed(top_matches)]

        fig = go.Figure(go.Bar(
            x=scores, y=labels, orientation="h",
            marker_color="#8b5cf6",
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        ))
        fig.update_layout(
            title="Top RAG Matches (Trait Similarity)",
            xaxis_title="Similarity Score",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=220, r=60, t=50, b=40),
            height=max(300, len(top_matches) * 40 + 80),
            template="plotly_white",
        )
        chart_html = f'<div class="figure">{pio.to_html(fig, include_plotlyjs=False, div_id="rag-chart", full_html=False)}</div>'

    # --- Convergence table ---
    conv_html = ""
    if convergence:
        conv_rows = []
        for c in convergence:
            signal = c.get("signal", "")
            if signal == "both":
                signal_display = "Visual + Literature"
            elif signal == "visual_only":
                signal_display = "Visual only"
            elif signal == "rag_only":
                signal_display = "Literature only"
            else:
                signal_display = signal

            rag_s = f"{c['rag_score']:.3f}" if c.get("rag_score", 0) > 0 else "—"
            vis_c = f"{c['visual_softmax_score'] * 100:.1f}%" if c.get("visual_softmax_score", 0) > 0 else "—"
            conv_rows.append(
                f"      <tr><td>{_escape(c['taxon'])}</td>"
                f"<td>{signal_display}</td>"
                f"<td>{rag_s}</td>"
                f"<td>{vis_c}</td></tr>"
            )
        conv_html = f"""
    <h3>Convergence Analysis</h3>
    <div class="interpretation">
      Convergence compares which taxa appear in visual classification (Stage 2) vs. literature
      retrieval (Stage 3). Taxa flagged as <em>both</em> have independent support from two
      evidence channels. Taxa flagged as single-source may indicate the visual model sees
      something the literature doesn't describe, or vice versa.
    </div>
    <table>
      <thead><tr><th>Taxon</th><th>Signal</th><th>RAG Score</th><th>Visual Similarity</th></tr></thead>
      <tbody>
{chr(10).join(conv_rows)}
      </tbody>
    </table>"""

    return f"""
  <section>
    <h2>Literature Evidence (Stage 3)</h2>
    <div class="interpretation">
      Morphological traits from Stage 1 were encoded via sentence-transformers and searched
      against a FAISS index of ~3,490 botanical NLP descriptions. Matches are ranked by
      semantic similarity — higher scores indicate the literature description closely matches
      the observed morphology.
    </div>
{chart_html}
{conv_html}
  </section>
"""


# =========================================================================
# Stage 4 — Evidence Synthesis
# =========================================================================


def _evidence_section(stage: dict[str, Any]) -> str:
    """Rendered evidence document and quality flags."""
    if stage.get("skipped"):
        return _skipped_section("Evidence Synthesis (Stage 4)", "Stage 4 was skipped.")

    if stage.get("error"):
        return _error_section("Evidence Synthesis (Stage 4)", stage["error"])

    data = stage.get("data", {})
    evidence_md = data.get("evidence_document", "")
    quality_flags = data.get("quality_flags", [])

    # Convert markdown to HTML (limited subset: headers, bold, bullets, lists)
    evidence_html = _markdown_to_html(evidence_md) if evidence_md else "<p>No evidence document generated.</p>"

    flags_html = ""
    if quality_flags:
        flag_items = "\n".join(
            f'    <div class="warning-box"><strong>Quality Flag:</strong> {_escape(f)}</div>'
            for f in quality_flags
        )
        flags_html = f"""
    <h3>Quality Flags</h3>
{flag_items}"""

    return f"""
  <section>
    <h2>Evidence Synthesis (Stage 4)</h2>
    <div class="interpretation">
      This stage deterministically assembles evidence from Stages 1-3 into a structured
      document for the reasoning LLM. No ML inference occurs here — it's a formatting and
      quality-checking step. Quality flags highlight potential issues.
    </div>
{flags_html}
    <details>
      <summary>Full Evidence Document</summary>
      <div style="padding: 16px; background: #f9fafb; border-radius: 8px; font-size: 14px; line-height: 1.65;">
        {evidence_html}
      </div>
    </details>
  </section>
"""


# =========================================================================
# Stage 5 — Reasoning
# =========================================================================


def _reasoning_section(stage: dict[str, Any]) -> str:
    """Final classification card, reasoning, alternatives."""
    if stage.get("skipped"):
        return _skipped_section("Final Classification (Stage 5)", "Stage 5 was skipped — no LLM reasoning performed.")

    if stage.get("error"):
        return _error_section("Final Classification (Stage 5)", stage["error"])

    data = stage.get("data", {})
    clf = data.get("classification", {})

    family = clf.get("predicted_family", "—")
    genus = clf.get("predicted_genus", "—")
    species = clf.get("predicted_species", "—")
    confidence = clf.get("confidence", "unknown")
    reasoning = clf.get("reasoning", "")
    features = clf.get("supporting_features", [])
    alternatives = clf.get("alternatives", [])

    conf_cls = {
        "high": "confidence-high",
        "medium": "confidence-medium",
        "low": "confidence-low",
    }.get(confidence, "confidence-low")

    # --- Result card ---
    result_card = f"""
    <div class="result-card">
      <div class="taxon">{_escape(family)}</div>
      <div class="detail">{_escape(genus)} {_escape(species)}</div>
      <div class="detail">Confidence: <span class="{conf_cls}">{confidence.upper()}</span></div>
    </div>"""

    # --- Reasoning narrative ---
    reasoning_html = f"""
    <h3>Reasoning</h3>
    <p>{_escape(reasoning)}</p>""" if reasoning else ""

    # --- Supporting features ---
    features_html = ""
    if features:
        items = "\n".join(f"      <li>{_escape(f)}</li>" for f in features)
        features_html = f"""
    <h3>Supporting Features</h3>
    <ul>
{items}
    </ul>"""

    # --- Alternatives table ---
    alts_html = ""
    if alternatives:
        alt_rows = []
        for a in alternatives:
            alt_rows.append(
                f"      <tr><td>{_escape(a['taxon'])}</td>"
                f"<td>{_escape(a['reason'])}</td></tr>"
            )
        alts_html = f"""
    <h3>Alternatives Considered</h3>
    <table>
      <thead><tr><th>Taxon</th><th>Reason for Exclusion</th></tr></thead>
      <tbody>
{chr(10).join(alt_rows)}
      </tbody>
    </table>"""

    return f"""
  <section>
    <h2>Final Classification (Stage 5)</h2>
    <div class="interpretation">
      A text-only LLM reviewed the assembled evidence document and produced a final
      classification with reasoning. The model sees no images — only the structured evidence
      from Stages 1-4.
    </div>
{result_card}
{reasoning_html}
{features_html}
{alts_html}
  </section>
"""


# =========================================================================
# Timing waterfall
# =========================================================================


def _timing_section(result: dict[str, Any]) -> str:
    """Horizontal bar chart of per-stage elapsed time."""
    stages = result.get("stages", {})
    stage_order = [
        ("morphology", "Stage 1: Morphology"),
        ("classification", "Stage 2: Classification"),
        ("trait_retrieval", "Stage 3: Trait Retrieval"),
        ("evidence_synthesis", "Stage 4: Evidence Synthesis"),
        ("reasoning", "Stage 5: Reasoning"),
    ]

    labels = []
    times_sec = []
    colors = []
    for key, label in stage_order:
        sr = stages.get(key, {})
        elapsed = sr.get("elapsed_ms", 0) / 1000
        labels.append(label)
        times_sec.append(elapsed)
        if sr.get("skipped"):
            colors.append("#9ca3af")
        elif sr.get("error"):
            colors.append("#ef4444")
        else:
            colors.append("#3b82f6")

    fig = go.Figure(go.Bar(
        x=times_sec, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{t:.1f}s" for t in times_sec],
        textposition="outside",
    ))
    fig.update_layout(
        title="Stage Timing",
        xaxis_title="Elapsed (seconds)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, r=60, t=50, b=40),
        height=280,
        template="plotly_white",
    )
    chart_html = pio.to_html(fig, include_plotlyjs=False, div_id="timing-chart", full_html=False)

    total = result.get("total_elapsed_ms", 0) / 1000

    return f"""
  <section>
    <h2>Timing</h2>
    <p>Total pipeline elapsed: <strong>{total:.1f}s</strong></p>
    <div class="figure">{chart_html}</div>
  </section>
"""


# =========================================================================
# Glossary
# =========================================================================


def _glossary_section() -> str:
    """Collapsible glossary of metrics and scores used in the report."""
    return """
  <section>
    <h2>Glossary</h2>
    <div class="interpretation">
      Definitions for metrics and scores used throughout this report.
    </div>
    <details open>
      <summary>Stage 2 &mdash; Visual Classification</summary>
      <table>
        <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
        <tbody>
          <tr><td><strong>Similarity Share</strong></td><td>Each candidate class gets a slice of 100% based on how close it is to the query specimen. Closer classes get larger slices, farther classes get smaller slices. Computed as softmax(&minus;L2&nbsp;distance) across all centroids. Because all candidates are visually similar tropical seedlings, the slices are close in size &mdash; a top score of 2&ndash;3% is typical and <em>does not</em> mean low confidence. With N classes, the uniform baseline is 1/N (e.g., 1.9% for 52 families). Focus on the <em>rank ordering</em> and <em>decision margin</em>, not the absolute percentage.</td></tr>
          <tr><td><strong>L2 Distance</strong></td><td>Euclidean distance between the query embedding and a class centroid in 768-dim BioCLIP&nbsp;2 space, after mean-centering and L2 normalization. Lower = more similar. All vectors lie on the unit hypersphere, so L2 ranges from 0 to 2.</td></tr>
          <tr><td><strong>Cosine Similarity</strong></td><td>Cosine of the angle between the query embedding and a class centroid. Higher = more similar. Ranges from &minus;1 to 1.</td></tr>
          <tr><td><strong>Decision Margin</strong></td><td>Difference in similarity share between the #1 and #2 predictions. Larger margin = stronger separation between the top candidate and the runner-up. Color-coded: green (&gt;0.10), yellow (0.03&ndash;0.10), red (&lt;0.03).</td></tr>
          <tr><td><strong>Hierarchical Consistency</strong></td><td>Whether the top-1 predictions at family, genus, and species level form a valid taxonomic lineage (e.g., predicted species belongs to the predicted genus and family).</td></tr>
        </tbody>
      </table>
    </details>
    <details>
      <summary>Stage 3 &mdash; Literature Evidence</summary>
      <table>
        <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
        <tbody>
          <tr><td><strong>RAG Similarity</strong></td><td>Cosine similarity between a sentence-transformer encoding of the observed morphological traits and botanical literature descriptions in the FAISS index. Higher = closer semantic match.</td></tr>
          <tr><td><strong>Convergence Signal</strong></td><td>Whether a taxon appears in visual classification (Stage 2), literature retrieval (Stage 3), or both. &ldquo;Both&rdquo; indicates independent support from two evidence channels.</td></tr>
        </tbody>
      </table>
    </details>
    <details>
      <summary>Stage 5 &mdash; Final Classification</summary>
      <table>
        <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
        <tbody>
          <tr><td><strong>Confidence (High / Medium / Low)</strong></td><td>The reasoning LLM&rsquo;s self-assessed certainty in its final classification, based on the strength and convergence of evidence from Stages 1&ndash;4. This is a qualitative judgment &mdash; entirely distinct from the numerical similarity share in Stage 2.</td></tr>
        </tbody>
      </table>
    </details>
  </section>
"""


# =========================================================================
# Footer
# =========================================================================


def _footer() -> str:
    """Closing HTML."""
    return """  </main>
  <footer>Generated by SeedLearn</footer>
</body>
</html>
"""


# =========================================================================
# Helpers
# =========================================================================


def _skipped_section(title: str, message: str) -> str:
    """Gray placeholder for a skipped stage."""
    return f"""
  <section>
    <h2>{title}</h2>
    <div class="skipped">{_escape(message)}</div>
  </section>
"""


def _error_section(title: str, error: str) -> str:
    """Red warning box for a stage that errored."""
    return f"""
  <section>
    <h2>{title}</h2>
    <div class="warning-box"><strong>Error:</strong> {_escape(error)}</div>
  </section>
"""


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _extract_short_value(value: str) -> str:
    """Extract the value before the first parenthetical explanation.

    Example: "whorled (multiple leaves...)" → "whorled"
    Falls back to the full string if no parenthetical found.
    """
    if value in ("N/A", ""):
        return value
    match = re.match(r"^([^(]+?)(?:\s*\()", value)
    return match.group(1).strip() if match else value.strip()


def _extract_specimen_id(image_path: str) -> str:
    """Extract specimen ID from an image path.

    Path pattern: .../SPECIMEN_ID/Family_Genus_species_SPECIMEN_ID_NNN.jpg
    """
    parts = Path(image_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return Path(image_path).stem


def _stage_image_count(stage: dict[str, Any]) -> str:
    """Infer image count text from stage data."""
    data = stage.get("data", {})
    n = data.get("num_images_pooled") or data.get("num_images")
    if n:
        return f"{n}"
    return "the"


def _markdown_to_html(md: str) -> str:
    """Convert a limited Markdown subset to HTML.

    Handles: ``# headings``, ``**bold**``, ``- bullets``, blank lines → ``<p>``.
    Sufficient for the evidence document format.
    """
    lines = md.split("\n")
    html_parts: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Close list if we're leaving a bullet block
        if in_list and not stripped.startswith("- "):
            html_parts.append("</ul>")
            in_list = False

        if not stripped:
            continue
        elif stripped.startswith("### "):
            html_parts.append(f"<h4>{_md_inline(stripped[4:])}</h4>")
        elif stripped.startswith("## "):
            html_parts.append(f"<h3>{_md_inline(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            html_parts.append(f"<h3>{_md_inline(stripped[2:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"  <li>{_md_inline(stripped[2:])}</li>")
        else:
            html_parts.append(f"<p>{_md_inline(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def _md_inline(text: str) -> str:
    """Convert inline Markdown (bold, code) to HTML."""
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # `code`
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # [text](url) — escape but keep links
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text
