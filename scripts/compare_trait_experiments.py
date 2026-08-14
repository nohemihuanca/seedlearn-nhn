#!/usr/bin/env python
"""Compare leaf-margin extraction across experiment conditions.

Grades each condition's ``model_run`` directory with the shared human-annotation
grader, extracts leaf-margin metrics, and writes a comparison CSV + synthesis HTML
that ranks conditions against the Roni-vs-Carmen human ceiling, with paired
McNemar significance for each condition against a chosen baseline.

Conditions are given as ``label=path`` pairs (local or external-adapted dirs both
work — they are the same ``model_run`` shape)::

    python scripts/compare_trait_experiments.py \\
        --run "baseline=trait_grading/model_run/2026-07-06_134225" \\
        --run "K1_gpt-5.4=trait_grading/model_run/K1_gpt-5.4_all-traits" \\
        --run "K2_gpt-5.1_per-trait=trait_grading/model_run/K2_gpt-5.1_per-trait" \\
        --run "K3_gpt-5.1_per-section=trait_grading/model_run/K3_gpt-5.1_per-section" \\
        --baseline baseline \\
        --out-dir trait_grading/reports/experiments/$(date +%Y-%m-%d_%H%M%S)
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
    ConditionMetrics,
    grade_condition,
    paired_mcnemar,
)
from seedlearn.benchmarking.human.value_map import MISSING

logger = logging.getLogger(__name__)

_TG = Path("trait_grading")
DEFAULT_STRI_MATRIX = (
    "/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/traits/"
    "stri_web_keys/per_key_trait_matrices/"
    "cl185_complete_tree_species_of_panama_trait_matrix.csv"
)


def _kappa_color(k: float | None) -> str:
    """Landis & Koch band color for a κ value (matches the report's palette)."""
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


def parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
    """Parse ``label=path`` CLI pairs into (label, dir) tuples."""
    out: list[tuple[str, Path]] = []
    for spec in run_args:
        if "=" not in spec:
            raise ValueError(f"--run must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def write_long_csv(path: Path, metrics: list[ConditionMetrics]) -> None:
    """One row per (condition, κ-axis)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "model", "granularity", "external", "axis", "agreement_rate", "cohen_kappa", "n_compared"])
        for m in metrics:
            for axis, am in (("model_vs_roni", m.vs_roni), ("model_vs_carmen", m.vs_carmen)):
                w.writerow([m.label, m.model, m.granularity, m.external, axis,
                            "" if am.rate is None else f"{am.rate:.4f}",
                            "" if am.kappa is None else f"{am.kappa:.4f}", am.n])


def write_wide_csv(path: Path, metrics: list[ConditionMetrics]) -> None:
    """One row per condition, with the STRI accuracy column + provenance."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "model", "granularity", "external", "n_model_specimens",
                    "roni_rate", "roni_kappa", "roni_n", "carmen_rate", "carmen_kappa",
                    "carmen_n", "stri_accuracy", "stri_n"])
        for m in metrics:
            w.writerow([m.label, m.model, m.granularity, m.external, m.n_model_specimens,
                        "" if m.vs_roni.rate is None else f"{m.vs_roni.rate:.4f}",
                        "" if m.vs_roni.kappa is None else f"{m.vs_roni.kappa:.4f}", m.vs_roni.n,
                        "" if m.vs_carmen.rate is None else f"{m.vs_carmen.rate:.4f}",
                        "" if m.vs_carmen.kappa is None else f"{m.vs_carmen.kappa:.4f}", m.vs_carmen.n,
                        "" if m.stri_accuracy is None else f"{m.stri_accuracy:.4f}", m.stri_n])


def _cell_payload(metrics: list[ConditionMetrics], ceiling_label: str | None) -> dict:
    """Serialize per-(label, axis) drill-down rows for the report's inline JSON."""
    payload: dict[str, list[dict]] = {}
    for m in metrics:
        for axis, rows in m.cells.items():
            # Roni-vs-Carmen is human-only and identical across conditions; expose it
            # once under the ceiling banner's key rather than per condition.
            if axis == "roni_vs_carmen" and m.label != ceiling_label:
                continue
            key = "ceiling|roni_vs_carmen" if axis == "roni_vs_carmen" else f"{m.label}|{axis}"
            payload[key] = [
                {
                    "id": c.specimen_id,
                    "mr": c.model_raw or "",
                    "mc": c.model_canonical,
                    "md": c.model_dropped,
                    "rv": c.roni_views,
                    "rc": c.roni_canonical,
                    "cv": c.carmen_views,
                    "cc": c.carmen_canonical,
                    "counted": c.counted,
                    "agree": c.agree,
                }
                for c in rows
            ]
    return payload


def _prompt_payload(metrics: list[ConditionMetrics]) -> dict:
    """Serialize each condition's reconstructed prompt provenance for the modal."""
    out: dict[str, dict] = {}
    for m in metrics:
        p = m.prompt
        out[m.label] = {
            "style": (p.style if p else None),
            "model": (p.model if p else m.model),
            "examples_file": (p.examples_file if p else None),
            "external": (p.external if p else m.external),
            "text": (p.text if p else None),
            "unavailable_reason": (p.unavailable_reason if p else "no prompt info"),
        }
    return out


def _embed_json(obj: dict) -> str:
    """JSON for an inline <script> block, neutralizing any ``</`` sequence."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def write_html(path: Path, metrics: list[ConditionMetrics], ceiling, deltas) -> None:
    """Render the synthesis report: condition table + delta/McNemar table + ceiling.

    Each model-vs-Roni / model-vs-Carmen cell and the ceiling banner are clickable,
    opening a per-specimen drill-down (raw model value, its canonicalization incl.
    dropped-to-MISSING rows, and both annotators' raw+canonical values). Each row's
    "prompt" link opens the condition's reconstructed inference prompt.
    """
    ceiling_label = metrics[0].label if metrics else None
    rows = []
    for m in metrics:
        tag = " <span class='ext'>external</span>" if m.external else ""
        rows.append(
            f"<tr><td>{escape(m.label)}{tag}<br>"
            f"<a href='#' class='promptlink' data-label='{escape(m.label)}'>prompt ▸</a></td>"
            f"<td>{escape(m.model)}</td>"
            f"<td>{escape(m.granularity)}</td><td>{m.n_model_specimens}</td>"
            f"<td class='drill' data-key='{escape(m.label)}|model_vs_roni' "
            f"style='background:{_kappa_color(m.vs_roni.kappa)}'>{_fmt(m.vs_roni.rate, True)} "
            f"(κ {_fmt(m.vs_roni.kappa)}, n={m.vs_roni.n})</td>"
            f"<td class='drill' data-key='{escape(m.label)}|model_vs_carmen' "
            f"style='background:{_kappa_color(m.vs_carmen.kappa)}'>{_fmt(m.vs_carmen.rate, True)} "
            f"(κ {_fmt(m.vs_carmen.kappa)}, n={m.vs_carmen.n})</td>"
            f"<td>{_fmt(m.stri_accuracy, True)} (n={m.stri_n})</td></tr>"
        )
    drows = []
    for d in deltas:
        sig = "" if d.p_value is None else f"{d.p_value:.3f}"
        drows.append(
            f"<tr><td>{escape(d.label_b)} vs {escape(d.label_a)}</td><td>{d.n_paired}</td>"
            f"<td>{_fmt(d.rate_delta, True)}</td><td>+{d.b_only_correct} / −{d.a_only_correct}</td>"
            f"<td>{sig}</td></tr>"
        )
    ceil_txt = f"Roni vs Carmen (human ceiling): {_fmt(ceiling.rate, True)} agreement, κ {_fmt(ceiling.kappa)} (n={ceiling.n})"
    cells_json = _embed_json(_cell_payload(metrics, ceiling_label))
    prompts_json = _embed_json(_prompt_payload(metrics))
    html = f"""<!doctype html><meta charset="utf-8">
<title>Leaf-margin experiment comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f4f4f4}}
.ext{{background:#e8eaf6;color:#3949ab;font-size:.7rem;padding:.1rem .3rem;border-radius:3px}}
.ceiling{{background:#fff8e1;border-left:4px solid #f9a825;padding:.6rem 1rem;margin:1rem 0}}
.note{{color:#555;font-size:.85rem}}
.drill,.clickceil{{cursor:pointer}}
.drill:hover{{outline:2px solid #3949ab;outline-offset:-2px}}
.promptlink{{font-size:.75rem;color:#3949ab;text-decoration:none}}
.clickhint{{font-size:.7rem;color:#3949ab;font-weight:normal}}
dialog{{max-width:960px;width:92%;border:1px solid #bbb;border-radius:8px;padding:0}}
dialog::backdrop{{background:rgba(0,0,0,.4)}}
.dhead{{display:flex;justify-content:space-between;align-items:center;
  padding:.6rem 1rem;border-bottom:1px solid #ddd;background:#f4f4f4}}
.dbody{{padding:.4rem 1rem 1rem;max-height:72vh;overflow:auto}}
.dbody table{{font-size:.8rem}}
.miss{{color:#999;font-style:italic}}
.badge{{background:#d73027;color:#fff;font-size:.65rem;padding:.05rem .35rem;border-radius:3px;margin-left:.3rem}}
tr.dropped{{background:#fdecea}}
tr.disagree{{background:#fff7e6}}
.close{{cursor:pointer;border:none;background:#e0e0e0;border-radius:4px;padding:.2rem .6rem}}
pre.prompt{{white-space:pre-wrap;background:#f4f6f8;border:1px solid #cfd8dc;border-radius:6px;padding:12px 16px;font-size:12px}}
</style>
<h1>Leaf-margin extraction — condition comparison</h1>
<div class="ceiling clickceil" data-key="ceiling|roni_vs_carmen"><b>{escape(ceil_txt)}</b>
<span class="clickhint">— click to inspect per specimen</span><br>
<span class="note">Model-vs-human agreement should be read against this human ceiling, not against 100%.
Leaf margin is graded as a coarse 3-way (entire / toothed / lobed); serration subtypes collapse to "toothed".</span></div>
<h2>Per-condition leaf-margin agreement</h2>
<p class="note">Click any <b>vs Roni</b> / <b>vs Carmen</b> cell to see the per-specimen raw value,
how it canonicalized (dropped-to-MISSING rows flagged), and both annotators' values.
Click <b>prompt ▸</b> to see the exact prompt that condition ran.</p>
<table><tr><th>Condition</th><th>Model</th><th>Granularity</th><th>n</th>
<th>vs Roni</th><th>vs Carmen</th><th>vs STRI (acc)</th></tr>
{''.join(rows)}
</table>
<p class="note">Cells colored by Cohen's κ (Landis &amp; Koch bands): red = worse-than-chance … green = almost-perfect.
External (cloud GPT) conditions are labeled; read granularity effects <b>within a model</b> (e.g. K2 vs K3, both GPT-5.1),
not across models.</p>
<h2>Paired deltas vs baseline (McNemar, model-vs-Roni)</h2>
<table><tr><th>Comparison</th><th>n paired</th><th>Δ agreement rate</th>
<th>newly-correct / newly-wrong</th><th>McNemar p</th></tr>
{''.join(drows)}
</table>
<p class="note">Δ and the McNemar test are computed on the specimens both conditions scored (paired design).
"newly-correct" = specimens the condition gets right that the baseline gets wrong (and vice-versa).</p>

<dialog id="modal">
  <div class="dhead"><b id="mtitle"></b><button class="close" onclick="document.getElementById('modal').close()">close ✕</button></div>
  <div class="dbody" id="mbody"></div>
</dialog>
<script type="application/json" id="cells">{cells_json}</script>
<script type="application/json" id="prompts">{prompts_json}</script>
<script>
(function() {{
  var MISSING = {json.dumps(MISSING)};
  var CELLS = JSON.parse(document.getElementById('cells').textContent);
  var PROMPTS = JSON.parse(document.getElementById('prompts').textContent);
  var modal = document.getElementById('modal');
  function esc(s) {{
    return String(s == null ? '' : s).replace(/[&<>"]/g, function(ch) {{
      return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[ch];
    }});
  }}
  function canon(v) {{ return v === MISSING ? '<span class="miss">∅ missing</span>' : esc(v); }}
  function views(vs) {{ return (vs && vs.length) ? esc(vs.join(' · ')) : '<span class="miss">—</span>'; }}
  function openDrill(key) {{
    var rows = CELLS[key] || [];
    var dropped = rows.filter(function(r) {{ return r.md; }}).length;
    var counted = rows.filter(function(r) {{ return r.counted; }}).length;
    document.getElementById('mtitle').textContent =
      key.replace('|', '  ·  ') + '   (' + counted + ' counted, ' + rows.length + ' shown, ' + dropped + ' dropped)';
    var h = '<table><tr><th>Specimen</th><th>Model raw</th><th>Model → canonical</th>' +
            '<th>Roni views → canon</th><th>Carmen views → canon</th><th>counted</th><th>agree</th></tr>';
    rows.forEach(function(r) {{
      var cls = r.md ? 'dropped' : (r.counted && !r.agree ? 'disagree' : '');
      var drop = r.md ? '<span class="badge">DROPPED</span>' : '';
      h += '<tr class="' + cls + '"><td>' + esc(r.id) + '</td>' +
           '<td>' + (r.mr ? esc(r.mr) : '<span class="miss">—</span>') + '</td>' +
           '<td>' + canon(r.mc) + drop + '</td>' +
           '<td>' + views(r.rv) + ' → ' + canon(r.rc) + '</td>' +
           '<td>' + views(r.cv) + ' → ' + canon(r.cc) + '</td>' +
           '<td>' + (r.counted ? '✓' : '·') + '</td>' +
           '<td>' + (r.counted ? (r.agree ? '✓' : '✗') : '·') + '</td></tr>';
    }});
    h += '</table>';
    document.getElementById('mbody').innerHTML = h;
    if (modal.showModal) modal.showModal(); else modal.setAttribute('open', 'open');
  }}
  function openPrompt(label) {{
    var p = PROMPTS[label] || {{}};
    document.getElementById('mtitle').textContent = 'Prompt · ' + label;
    var meta = '<p class="note">style: <code>' + esc(p.style || '—') + '</code> · model: <code>' +
               esc(p.model || '—') + '</code>' +
               (p.examples_file ? ' · few-shot: <code>' + esc(p.examples_file) + '</code>' : '') +
               (p.external ? ' · <b>external (cloud)</b>' : '') + '</p>';
    var body = p.text
      ? '<pre class="prompt">' + esc(p.text) + '</pre>'
      : '<p class="note">⚠ as-run prompt unavailable: ' + esc(p.unavailable_reason || 'unknown') + '</p>';
    document.getElementById('mbody').innerHTML = meta + body;
    if (modal.showModal) modal.showModal(); else modal.setAttribute('open', 'open');
  }}
  document.querySelectorAll('.drill, .clickceil').forEach(function(el) {{
    el.addEventListener('click', function() {{ openDrill(el.getAttribute('data-key')); }});
  }});
  document.querySelectorAll('.promptlink').forEach(function(el) {{
    el.addEventListener('click', function(e) {{ e.preventDefault(); openPrompt(el.getAttribute('data-label')); }});
  }});
}})();
</script>
"""
    path.write_text(html)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], required=True, help="label=path (repeatable).")
    p.add_argument("--baseline", type=str, default=None, help="Label to compute deltas against (default: first run).")
    p.add_argument("--out-dir", type=Path, required=True, help="Report output directory.")
    p.add_argument("--roni-xlsx", default=str(_TG / "annotations/roni_bianco.xlsx"))
    p.add_argument("--carmen-xlsx", default=str(_TG / "annotations/carmen.xlsx"))
    p.add_argument("--curator-key", default=str(_TG / "keys/curator_taxonomic_key.csv"))
    p.add_argument("--stri-matrix", default=DEFAULT_STRI_MATRIX)
    p.add_argument("--no-stri", action="store_true", help="Skip the STRI axis.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    runs = parse_runs(args.run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stri_matrix = None if args.no_stri else args.stri_matrix

    metrics: list[ConditionMetrics] = []
    ceiling = None
    for label, run_dir in runs:
        if not run_dir.exists():
            logger.warning("skipping %s: %s not found", label, run_dir)
            continue
        m, ceil = grade_condition(
            run_dir, roni_xlsx=args.roni_xlsx, carmen_xlsx=args.carmen_xlsx,
            curator_key=args.curator_key, stri_matrix=stri_matrix, label=label,
        )
        metrics.append(m)
        ceiling = ceiling or ceil
        logger.info("%-28s vs Roni: %.1f%% (κ %.3f, n=%d)  STRI: %s",
                    label, (m.vs_roni.rate or 0) * 100, m.vs_roni.kappa or 0.0, m.vs_roni.n,
                    "—" if m.stri_accuracy is None else f"{m.stri_accuracy*100:.1f}%")
    if not metrics:
        logger.error("no conditions graded")
        return 1

    base_label = args.baseline or metrics[0].label
    base = next((m for m in metrics if m.label == base_label), metrics[0])
    deltas = [paired_mcnemar(base, m) for m in metrics if m.label != base.label]

    write_long_csv(args.out_dir / "leaf_margin_per_axis.csv", metrics)
    write_wide_csv(args.out_dir / "leaf_margin_summary.csv", metrics)
    write_html(args.out_dir / "leaf_margin_comparison.html", metrics, ceiling, deltas)
    logger.info("wrote CSV + HTML to %s", args.out_dir)
    if ceiling:
        logger.info("human ceiling (Roni vs Carmen): %.1f%% (κ %.3f, n=%d)",
                    (ceiling.rate or 0) * 100, ceiling.kappa or 0.0, ceiling.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
