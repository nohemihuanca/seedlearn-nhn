#!/usr/bin/env python
"""Compare trait-extraction conditions across ALL gradable traits.

Grades each condition's ``model_run`` dir with the shared human-annotation grader,
keeps every gradable trait (not just leaf margin), and writes a report built to
support a modeling decision: a recommendation panel (mean κ over the traits κ can
actually judge), a trait × condition agreement matrix coloured by the per-trait human
ceiling, and a trait-decision table that groups traits by whether the gap is a model
gap, already at human level, or undecidable — each row showing Roni's class
distribution so lopsided traits are visible. Every cell drills down to per-model
per-species predictions.

    python scripts/compare_all_traits.py \\
        --run "C0_baseline=trait_grading/model_run/C0_baseline_20260713_161841" \\
        --run "C1_upgraded_model=trait_grading/model_run/C1_upgraded_model_20260713_215241" \\
        --run "K1_gpt-5.4=trait_grading/model_run/K1_gpt-5.4_all-traits" \\
        --out-dir trait_grading/reports/experiments/all_traits_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from html import escape
from pathlib import Path

from seedlearn.benchmarking.human.experiment_compare import (
    VERDICT_AT_HUMAN_LEVEL,
    VERDICT_MODEL_GAP,
    VERDICT_UNDECIDABLE,
    AllTraitConditionMetrics,
    Ceiling,
    build_trait_comparison,
    class_distribution,
    distinct_species,
    grade_condition_all_traits,
    majority_fraction,
    model_display_names,
    species_map_from_curator,
    triage_trait,
)
from seedlearn.benchmarking.human.value_map import MISSING, gradable_specs

logger = logging.getLogger(__name__)

_TG = Path("trait_grading")
_MODEL_AXIS = "model_vs_roni"


def _kappa_color(k: float | None) -> str:
    """Landis & Koch band color for a κ value (shared palette with the margin report)."""
    if k is None:
        return "#eee"
    if k < 0.0:
        return "#d73027"
    if k < 0.2:
        return "#fc8d59"
    if k < 0.4:
        return "#fee08b"
    if k < 0.6:
        return "#d9ef8b"
    if k < 0.8:
        return "#91cf60"
    return "#1a9850"


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


_SPEC_BY_KEY = {s.key: s for s in gradable_specs()}

# Order + human-readable copy for the three trait-triage groups.
_VERDICT_GROUPS = [
    (VERDICT_MODEL_GAP, "Model gaps — humans agree, models don't",
     "The annotators agree here but the models fall short, so this is where model "
     "work pays off."),
    (VERDICT_AT_HUMAN_LEVEL, "At human level — model already agrees with Roni as well as Carmen does",
     "The best model's κ meets or beats the human-human ceiling; the bottleneck is "
     "annotation, not the model."),
    (VERDICT_UNDECIDABLE, "Undecidable by κ — κ can't score these",
     "κ ≡ 0 or undefined here regardless of model accuracy — the reference uses one "
     "class, has no annotations, or lacks a human ceiling. Excluded from the "
     "recommendation mean."),
]


def _roni_cells_for(metrics, trait) -> list:
    """Any condition's cells for ``trait`` — Roni's side is annotator-invariant."""
    for m in metrics:
        cells = m.cells.get(trait, {}).get(_MODEL_AXIS)
        if cells:
            return cells
    return []


def _best_model(metrics, trait) -> tuple[float | None, str | None]:
    """Highest model-vs-Roni κ for ``trait`` and the internal label that scored it."""
    best_k: float | None = None
    best_label: str | None = None
    for m in metrics:
        am = m.axes.get(trait, {}).get(_MODEL_AXIS)
        if am is None or not am.n or am.kappa is None:
            continue
        if best_k is None or am.kappa > best_k:
            best_k, best_label = am.kappa, m.label
    return best_k, best_label


def analyze_traits(metrics, ceiling) -> dict[str, dict]:
    """Per-trait triage facts shared by the panel, table, CSV, and drill-down.

    Returns ``{trait: {roni_dist, roni_majority, ceiling_kappa, best_kappa,
    best_label, verdict}}``. Roni's distribution and majority come from the trait's
    canonical vocabulary (zero-filled), so lopsided traits read the same everywhere.
    """
    out: dict[str, dict] = {}
    for trait in sorted_traits(metrics):
        spec = _SPEC_BY_KEY.get(trait)
        cells = _roni_cells_for(metrics, trait)
        roni_dist = class_distribution(cells, spec, side="roni") if spec else {}
        ceil = ceiling.get(trait)
        ceil_k = ceil.kappa if ceil else None
        best_k, best_label = _best_model(metrics, trait)
        out[trait] = {
            "roni_dist": roni_dist,
            "roni_majority": majority_fraction(roni_dist),
            "ceiling_kappa": ceil_k,
            "best_kappa": best_k,
            "best_label": best_label,
            "verdict": triage_trait(roni_dist, ceil_k, best_k),
        }
    return out


def _dist_inline(dist: dict[str, int]) -> str:
    """Compact ``class n | class n`` string over the classes actually used."""
    used = [(k, v) for k, v in dist.items() if v]
    return " | ".join(f"{escape(k)} {v}" for k, v in used) if used else "—"


def _dist_bar(dist: dict[str, int]) -> str:
    """A thin stacked bar of the class proportions (majority class darkest)."""
    total = sum(dist.values())
    if not total:
        return ""
    used = sorted(((k, v) for k, v in dist.items() if v), key=lambda kv: -kv[1])
    shades = ["#3949ab", "#7986cb", "#c5cae9", "#e8eaf6"]
    segs = "".join(
        f"<span style='display:inline-block;height:9px;width:{v / total * 100:.1f}%;"
        f"background:{shades[min(i, len(shades) - 1)]}' title='{escape(k)}: {v}'></span>"
        for i, (k, v) in enumerate(used)
    )
    return f"<span class='distbar'>{segs}</span>"


def parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
    """Parse ``label=path`` CLI pairs into (label, dir) tuples."""
    out: list[tuple[str, Path]] = []
    for spec in run_args:
        if "=" not in spec:
            raise ValueError(f"--run must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def _embed_json(obj) -> str:
    """JSON for an inline <script> block, neutralizing any ``</`` sequence."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def sorted_traits(metrics: list[AllTraitConditionMetrics]) -> list[str]:
    """Every trait any condition graded, in a stable order."""
    traits: dict[str, None] = {}
    for m in metrics:
        for t in m.axes:
            traits.setdefault(t, None)
    return list(traits)


def write_summary_csv(path, metrics, species_by_cell, analysis) -> None:
    """One row per (trait, condition): model-vs-Roni metrics + trait triage facts.

    Carries the ceiling κ, Roni's class balance, and the trait verdict as columns so
    the CSV holds everything the HTML shows — the ceiling used to live in the HTML
    only, forcing anyone who wanted it to scrape the page.
    """
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trait", "condition", "model", "axis", "agreement_rate",
                    "cohen_kappa", "n_compared", "n_species", "ceiling_kappa",
                    "roni_majority_frac", "roni_distribution", "trait_verdict"])
        for m in metrics:
            for trait, by_axis in sorted(m.axes.items()):
                am = by_axis.get(_MODEL_AXIS)
                if am is None:
                    continue
                a = analysis.get(trait, {})
                ceil_k = a.get("ceiling_kappa")
                maj = a.get("roni_majority")
                roni_dist = ";".join(f"{k}:{v}" for k, v in a.get("roni_dist", {}).items() if v)
                w.writerow([trait, m.label, m.model, _MODEL_AXIS,
                            "" if am.rate is None else f"{am.rate:.4f}",
                            "" if am.kappa is None else f"{am.kappa:.4f}", am.n,
                            species_by_cell.get((trait, m.label), ""),
                            "" if ceil_k is None else f"{ceil_k:.4f}",
                            "" if maj is None else f"{maj:.4f}",
                            roni_dist, a.get("verdict", "")])


def _matrix_html(metrics, traits, ceiling, display, species_by_cell) -> str:
    """Trait × model κ matrix; columns are model names; every cell opens its trait."""
    # The internal condition label lives in a tooltip (provenance) — not shown, so the
    # report reads by model name only (no "C0 baseline" text).
    header = "".join(
        f"<th title='{escape(m.label)}'>{escape(display[m.label])}"
        f"{' <span class=\"ext\">ext</span>' if m.external else ''}"
        f"<br><a href='#' class='promptlink' data-label='{escape(m.label)}'>prompt ▸</a></th>"
        for m in metrics
    )
    rows = []
    for trait in traits:
        ceil = ceiling.get(trait)
        cells = [
            f"<td class='drill' data-trait='{escape(trait)}'><b>{escape(trait)}</b></td>",
            f"<td class='drill' data-trait='{escape(trait)}' style='background:{_kappa_color(ceil.kappa if ceil else None)}'>"
            f"{_fmt(ceil.kappa if ceil else None)}</td>",
        ]
        for m in metrics:
            am = m.axes.get(trait, {}).get(_MODEL_AXIS)
            # A margin-only condition never asked most traits: no graded pairs is
            # "outside this prompt's scope", not "scored zero" — say so rather than
            # rendering an empty-looking κ that reads as a real (bad) result.
            if am is None or not am.n:
                cells.append(
                    f"<td class='drill na' data-trait='{escape(trait)}' "
                    f"title='not asked in this condition&#39;s prompt'>·</td>"
                )
                continue
            k = am.kappa
            nsp = species_by_cell.get((trait, m.label), 0)
            txt = f"{_fmt(am.rate, True)}<br>κ {_fmt(k)} · {nsp} sp"
            cells.append(
                f"<td class='drill' data-trait='{escape(trait)}' style='background:{_kappa_color(k)}'>{txt}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<table><tr><th>Trait</th><th>Ceiling<br>(κ R·C)</th>{header}</tr>"
        f"{''.join(rows)}</table>"
    )


def _kappa_legend_html() -> str:
    """Landis & Koch κ interpretation bands, colored with the report's palette."""
    bands = [
        (-0.01, "&lt; 0.00", "poor"),
        (0.10, "0.00–0.20", "slight"),
        (0.30, "0.20–0.40", "fair"),
        (0.50, "0.40–0.60", "moderate"),
        (0.70, "0.60–0.80", "substantial"),
        (0.90, "0.80–1.00", "almost perfect"),
    ]
    cells = "".join(
        f"<td style='background:{_kappa_color(v)};text-align:center'>{rng}<br>"
        f"<small>{lab}</small></td>"
        for v, rng, lab in bands
    )
    return (
        "<table class='legend'><tr><th style='background:#fff;border:none'>"
        "Cohen's κ<br><small>(Landis &amp; Koch)</small></th>"
        f"{cells}</tr></table>"
    )


def _undecidable_reason(a: dict) -> str:
    """Why κ can't judge a trait: no annotations / single class / no ceiling / unasked."""
    maj = a["roni_majority"]
    if maj is None:
        return "no gradable annotations"
    if maj >= 1.0:
        return "single class"
    if a["best_kappa"] is None:
        return "no model predictions"
    if a["ceiling_kappa"] is None:
        return "no human ceiling"
    return "undecidable"


def _recommendation_html(metrics, analysis, display) -> str:
    """Top panel: mean κ over decidable traits + who wins, with a plain verdict line.

    Only full-coverage conditions (those that answered at least half the decidable
    traits) are ranked — a single-trait prompt experiment would otherwise post a high
    mean on one easy trait and win spuriously. Narrow conditions are simply left out
    of the panel; they remain visible in the trait × model matrix.
    """
    undecidable = [t for t, a in analysis.items() if a["verdict"] == VERDICT_UNDECIDABLE]
    decidable = [t for t, a in analysis.items() if a["verdict"] != VERDICT_UNDECIDABLE]
    wins: dict[str, int] = {}
    for t in decidable:
        bl = analysis[t]["best_label"]
        if bl is not None:
            wins[bl] = wins.get(bl, 0) + 1

    def _answered(m, t):
        am = m.axes.get(t, {}).get(_MODEL_AXIS)
        return am is not None and am.n and am.kappa is not None

    rows = []
    for m in metrics:
        ks = [m.axes[t][_MODEL_AXIS].kappa for t in decidable if _answered(m, t)]
        mean_k = sum(ks) / len(ks) if ks else None
        rows.append((mean_k, m.label, wins.get(m.label, 0), len(ks)))
    # Only rank conditions that answered at least half the decidable traits.
    threshold = max(1, len(decidable) // 2)
    full = sorted((r for r in rows if r[3] >= threshold),
                  key=lambda r: (r[0] is not None, r[0] or 0), reverse=True)

    body = "".join(
        f"<tr><td><b>{escape(display[label])}</b></td>"
        f"<td style='background:{_kappa_color(mk)};text-align:center'><b>{_fmt(mk)}</b></td>"
        f"<td style='text-align:center'>{w}</td>"
        f"<td style='text-align:center'>{cov}/{len(decidable)}</td></tr>"
        for mk, label, w, cov in full
    )
    winner = display[full[0][1]] if full and full[0][0] is not None else "—"
    excluded = ", ".join(f"{escape(t)} ({_undecidable_reason(analysis[t])})"
                         for t in undecidable)
    return (
        f"<p class='rec'><b>Recommendation:</b> {escape(winner)} — highest mean κ across the "
        f"{len(decidable)} traits κ can judge.</p>"
        f"<table><tr><th>Model</th><th>Mean κ<br>({len(decidable)} decidable traits)</th>"
        "<th>Best on<br>#traits</th><th>Decidable<br>answered</th></tr>"
        f"{body}</table>"
        f"<p class='note'>{len(undecidable)} traits are excluded from the mean because κ can't "
        f"judge them: {excluded}. Including them would drag every model toward zero equally.</p>"
    )


def _decision_table_html(metrics, analysis, display) -> str:
    """Traits grouped by triage verdict, each row showing Roni's class distribution."""
    parts = []
    for verdict, title, blurb in _VERDICT_GROUPS:
        traits = [t for t, a in analysis.items() if a["verdict"] == verdict]
        if not traits:
            continue
        traits.sort(key=lambda t: (analysis[t]["ceiling_kappa"] is not None,
                                    analysis[t]["ceiling_kappa"] or 0), reverse=True)
        rows = []
        for t in traits:
            a = analysis[t]
            n = sum(a["roni_dist"].values())
            maj = a["roni_majority"]
            ceil_k, best_k = a["ceiling_kappa"], a["best_kappa"]
            frac = (f"{best_k / ceil_k * 100:.0f}%"
                    if (best_k is not None and ceil_k and ceil_k > 0) else "—")
            # Undecidable rows have no meaningful %-of-ceiling; show why κ can't judge them.
            if verdict == VERDICT_UNDECIDABLE:
                frac = f"<i>{escape(_undecidable_reason(a))}</i>"
            best_name = display.get(a["best_label"], "—") if a["best_label"] else "—"
            rows.append(
                f"<tr><td class='drill' data-trait='{escape(t)}'><b>{escape(t)}</b></td>"
                f"<td>{_dist_bar(a['roni_dist'])}<br><small>{_dist_inline(a['roni_dist'])}</small></td>"
                f"<td style='text-align:right'>{n}</td>"
                f"<td style='text-align:right'>{'—' if maj is None else f'{maj * 100:.0f}%'}</td>"
                f"<td style='text-align:right;background:{_kappa_color(ceil_k)}'>{_fmt(ceil_k)}</td>"
                f"<td style='text-align:right;background:{_kappa_color(best_k)}'>{_fmt(best_k)}</td>"
                f"<td>{escape(best_name)}</td>"
                f"<td style='text-align:right'>{frac}</td></tr>"
            )
        parts.append(
            f"<h3 class='vg'>{escape(title)} <span class='vgn'>({len(traits)})</span></h3>"
            f"<p class='note'>{escape(blurb)}</p>"
            "<table><tr><th>Trait</th><th>Roni class distribution</th><th>n</th>"
            "<th>majority</th><th>ceiling κ</th><th>best model κ</th><th>best model</th>"
            "<th>% of ceiling</th></tr>"
            f"{''.join(rows)}</table>"
        )
    return "".join(parts)


def _distribution_payload(metrics, analysis, display) -> dict:
    """Per-trait Roni + per-model predicted class counts for the drill-down header."""
    payload: dict[str, dict] = {}
    for trait, a in analysis.items():
        spec = _SPEC_BY_KEY.get(trait)
        if spec is None:
            continue
        vocab = [v for v in spec.canonical_values]
        # include any extra observed tokens (defensive; usually none)
        models = {}
        for m in metrics:
            cells = m.cells.get(trait, {}).get(_MODEL_AXIS)
            if not cells:
                continue
            models[display[m.label]] = class_distribution(cells, spec, side="model")
        for d in models.values():
            for k in d:
                if k not in vocab:
                    vocab.append(k)
        payload[trait] = {"vocab": vocab, "roni": a["roni_dist"], "models": models}
    return payload


def write_html(path, metrics, ceiling, species_map) -> None:
    """Render the all-trait report: recommendation panel, matrix, decision table, drill-down."""
    traits = sorted_traits(metrics)
    display = model_display_names(metrics)
    analysis = analyze_traits(metrics, ceiling)
    # Distinct species behind each (trait, condition) model-vs-Roni cell.
    species_by_cell = {}
    for m in metrics:
        for trait, by_axis in m.cells.items():
            cells = by_axis.get(_MODEL_AXIS)
            if cells is not None:
                species_by_cell[(trait, m.label)] = distinct_species(cells, species_map)
    # Unified per-trait payload: every model's prediction per specimen + humans.
    trait_payload = build_trait_comparison(metrics, display, species_map)
    model_order = [display[m.label] for m in metrics]
    prompts = {
        m.label: {
            "style": (m.prompt.style if m.prompt else None),
            "model": (m.prompt.model if m.prompt else m.model),
            "examples_file": (m.prompt.examples_file if m.prompt else None),
            "external": (m.prompt.external if m.prompt else m.external),
            "text": (m.prompt.text if m.prompt else None),
            "unavailable_reason": (m.prompt.unavailable_reason if m.prompt else "no prompt info"),
        }
        for m in metrics
    }
    dists = _distribution_payload(metrics, analysis, display)
    html = f"""<!doctype html><meta charset="utf-8">
<title>All-trait extraction comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1250px}}
table{{border-collapse:collapse;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.35rem .5rem;text-align:left;font-size:.82rem;vertical-align:top}}
th{{background:#f4f4f4}}
.ext{{background:#e8eaf6;color:#3949ab;font-size:.65rem;padding:0 .25rem;border-radius:3px}}
.clabel{{color:#999;font-size:.62rem;font-weight:normal}}
.note{{color:#555;font-size:.85rem}}
.rec{{font-size:1rem;background:#e8f5e9;border-left:4px solid #43a047;padding:.6rem .9rem;border-radius:4px}}
.drill{{cursor:pointer}}
.drill:hover{{outline:2px solid #3949ab;outline-offset:-2px}}
.vg{{margin:1.4rem 0 .2rem;font-size:1rem}}
.vgn{{color:#888;font-weight:normal}}
.distbar{{display:inline-block;width:120px;border:1px solid #ccc;line-height:0;white-space:nowrap;vertical-align:middle}}
/* Trait outside a condition's prompt scope — muted so it never reads as a low score. */
.na{{color:#ccc;background:repeating-linear-gradient(45deg,#fafafa,#fafafa 4px,#f2f2f2 4px,#f2f2f2 8px)}}
.promptlink{{font-size:.7rem;color:#3949ab;text-decoration:none;font-weight:normal}}
dialog{{max-width:1100px;width:94%;border:1px solid #bbb;border-radius:8px;padding:0}}
dialog::backdrop{{background:rgba(0,0,0,.4)}}
.dhead{{display:flex;justify-content:space-between;align-items:center;padding:.6rem 1rem;border-bottom:1px solid #ddd;background:#f4f4f4}}
.dbody{{padding:.4rem 1rem 1rem;max-height:74vh;overflow:auto}}
.dbody table{{font-size:.76rem}}
.distsum{{background:#fafafa}} .distsum td,.distsum th{{text-align:right}} .distsum td:first-child{{text-align:left;font-weight:600}}
.miss{{color:#999;font-style:italic}}
td.ok{{background:#d7f0c8}} td.bad{{background:#fbd6d2}} td.na{{background:#eee;color:#999}}
.roni{{background:#e8eaf6;font-weight:600}}
.close{{cursor:pointer;border:none;background:#e0e0e0;border-radius:4px;padding:.2rem .6rem}}
pre.prompt{{white-space:pre-wrap;background:#f4f6f8;border:1px solid #cfd8dc;border-radius:6px;padding:12px 16px;font-size:12px}}
table.legend{{margin:.4rem 0}}
table.legend td{{font-size:.72rem;padding:.25rem .5rem}}
</style>
<h1>All-trait extraction — model comparison</h1>
<h2>Recommendation</h2>
{_recommendation_html(metrics, analysis, display)}
{_kappa_legend_html()}
<h2>Trait decisions — where to invest</h2>
<p class="note">Traits sorted into three regimes. κ only measures a model fairly when the two
annotators themselves agree (high ceiling) and Roni used more than one class; the distribution
column shows how lopsided each trait is. <b>Click a trait</b> to see every model's per-species
prediction and its predicted class counts.</p>
{_decision_table_html(metrics, analysis, display)}
<h2>Trait × model agreement (vs Roni)</h2>
<p class="note">Each cell is the model's <b>agreement with Roni</b> (rate + Cohen's κ + distinct
species), coloured by κ. The <b>Ceiling</b> column is the Roni-vs-Carmen human κ per trait — read
model κ against it, not against 1.0. <b>Click any cell</b> to drill into per-species predictions.
Where Roni recorded no value (∅ not visible) that species is excluded from κ and shown grey.</p>
{_matrix_html(metrics, traits, ceiling, display, species_by_cell)}

<dialog id="modal">
  <div class="dhead"><b id="mtitle"></b><button class="close" onclick="document.getElementById('modal').close()">close ✕</button></div>
  <div class="dbody" id="mbody"></div>
</dialog>
<script type="application/json" id="traits">{_embed_json(trait_payload)}</script>
<script type="application/json" id="models">{_embed_json(model_order)}</script>
<script type="application/json" id="prompts">{_embed_json(prompts)}</script>
<script type="application/json" id="dists">{_embed_json(dists)}</script>
<script>
(function() {{
  var MISSING = {json.dumps(MISSING)};
  var TRAITS = JSON.parse(document.getElementById('traits').textContent);
  var MODELS = JSON.parse(document.getElementById('models').textContent);
  var PROMPTS = JSON.parse(document.getElementById('prompts').textContent);
  var DISTS = JSON.parse(document.getElementById('dists').textContent);
  var modal = document.getElementById('modal');
  function esc(s) {{ return String(s==null?'':s).replace(/[&<>"]/g,function(c){{return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c];}}); }}
  function canon(v) {{ return v===MISSING ? '<span class="miss">∅</span>' : esc(v); }}
  // Agreement class of a value vs Roni: grey when either side has no value (not judged).
  function agreeCls(v, roni) {{ return (roni===MISSING || v===MISSING) ? 'na' : (v===roni ? 'ok' : 'bad'); }}
  // Predicted class counts: Roni's reference beside each model's, over the trait vocabulary.
  function distTable(trait) {{
    var d = DISTS[trait];
    if (!d) return '';
    var h = '<table class="distsum"><tr><th>predicted class counts</th>';
    d.vocab.forEach(function(v){{ h += '<th>'+esc(v)+'</th>'; }});
    h += '</tr><tr><td>Roni (reference)</td>';
    d.vocab.forEach(function(v){{ h += '<td>'+(d.roni[v]||0)+'</td>'; }});
    h += '</tr>';
    Object.keys(d.models).forEach(function(m){{
      h += '<tr><td>'+esc(m)+'</td>';
      d.vocab.forEach(function(v){{ h += '<td>'+(d.models[m][v]||0)+'</td>'; }});
      h += '</tr>';
    }});
    return h + '</table>';
  }}
  function openTrait(trait) {{
    var rows = TRAITS[trait] || [];
    document.getElementById('mtitle').textContent = trait + '  —  ' + rows.length + ' species (coloured vs Roni)';
    var h = distTable(trait)
          + '<table><tr><th>Species</th><th>Roni</th><th>Carmen</th>';
    MODELS.forEach(function(m){{ h += '<th>'+esc(m)+'</th>'; }});
    h += '</tr>';
    rows.forEach(function(r) {{
      var roniTxt = r.roni===MISSING ? '<span class="miss">∅ not visible</span>' : esc(r.roni);
      h += '<tr><td><i>'+esc(r.species)+'</i></td>'
         + '<td class="roni">'+roniTxt+'</td>'
         + '<td class="'+agreeCls(r.carmen, r.roni)+'">'+canon(r.carmen)+'</td>';
      MODELS.forEach(function(m) {{
        var cell = r.models[m];
        if (!cell) {{ h += '<td class="na">—</td>'; return; }}
        var title = cell.raw ? ' title="'+esc(cell.raw)+'"' : '';
        h += '<td class="'+agreeCls(cell.canonical, r.roni)+'"'+title+'>'+canon(cell.canonical)+'</td>';
      }});
      h += '</tr>';
    }});
    document.getElementById('mbody').innerHTML = h + '</table>';
    if (modal.showModal) modal.showModal(); else modal.setAttribute('open','open');
  }}
  function openPrompt(label) {{
    var p = PROMPTS[label] || {{}};
    document.getElementById('mtitle').textContent = 'Prompt · ' + label;
    var meta = '<p class="note">style: <code>'+esc(p.style||'—')+'</code> · model: <code>'+esc(p.model||'—')+'</code>'
             + (p.examples_file?' · few-shot: <code>'+esc(p.examples_file)+'</code>':'')
             + (p.external?' · <b>external (cloud)</b>':'') + '</p>';
    var body = p.text ? '<pre class="prompt">'+esc(p.text)+'</pre>'
                      : '<p class="note">⚠ as-run prompt unavailable: '+esc(p.unavailable_reason||'unknown')+'</p>';
    document.getElementById('mbody').innerHTML = meta + body;
    if (modal.showModal) modal.showModal(); else modal.setAttribute('open','open');
  }}
  document.querySelectorAll('.drill').forEach(function(el){{ el.addEventListener('click',function(){{openTrait(el.getAttribute('data-trait'));}}); }});
  document.querySelectorAll('.promptlink').forEach(function(el){{ el.addEventListener('click',function(e){{e.stopPropagation();e.preventDefault();openPrompt(el.getAttribute('data-label'));}}); }});
}})();
</script>
"""
    path.write_text(html)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], required=True, help="label=path (repeatable).")
    p.add_argument("--out-dir", type=Path, required=True, help="Report output directory.")
    p.add_argument("--roni-xlsx", default=str(_TG / "annotations/roni_bianco.xlsx"))
    p.add_argument("--carmen-xlsx", default=str(_TG / "annotations/carmen.xlsx"))
    p.add_argument("--curator-key", default=str(_TG / "keys/curator_taxonomic_key.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    runs = parse_runs(args.run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    species_map = species_map_from_curator(args.curator_key)

    metrics: list[AllTraitConditionMetrics] = []
    ceiling: dict[str, Ceiling] = {}
    for label, run_dir in runs:
        if not run_dir.exists():
            logger.warning("skipping %s: %s not found", label, run_dir)
            continue
        m, ceil = grade_condition_all_traits(
            run_dir, roni_xlsx=args.roni_xlsx, carmen_xlsx=args.carmen_xlsx,
            curator_key=args.curator_key, label=label, cell_axes=(_MODEL_AXIS,),
        )
        metrics.append(m)
        ceiling = ceiling or ceil
        logger.info("%-28s graded %d traits", label, len(m.axes))
    if not metrics:
        logger.error("no conditions graded")
        return 1

    analysis = analyze_traits(metrics, ceiling)
    write_summary_csv(args.out_dir / "all_trait_summary.csv", metrics,
                      {(t, m.label): distinct_species(m.cells.get(t, {}).get(_MODEL_AXIS, []), species_map)
                       for m in metrics for t in m.axes},
                      analysis)
    write_html(args.out_dir / "all_trait_comparison.html", metrics, ceiling, species_map)
    logger.info("wrote CSV + HTML to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
