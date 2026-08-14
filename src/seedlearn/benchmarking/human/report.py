"""Assemble human-grading results into CSV + HTML reports.

Orchestrates the full flow -- load annotations, join, aggregate to per-specimen
modes, grade the three agreement axes, grade Roni's species IDs -- and writes
CSV tables plus a self-contained HTML report into the output directory.
"""

from __future__ import annotations

import html as _html
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from seedlearn.benchmarking.human.aggregate import SpecimenAggregate, aggregate_records
from seedlearn.benchmarking.human.annotations import load_annotations, load_curator_key
from seedlearn.benchmarking.human.categorical_grader import (
    PairDetail,
    TraitAgreement,
    grade_all_axes,
    human_lookup,
    human_view_lookup,
    load_model_traits,
    model_lookup,
    overall_by_axis,
    pair_details,
)
from seedlearn.benchmarking.human.id_corrections import load_corrections
from seedlearn.benchmarking.human.id_grading import (
    HumanIDRecord,
    grade_human_ids,
    id_accuracy,
)
from seedlearn.benchmarking.human.stri_compare import (
    STRIAgreement,
    STRIPairDetail,
    accuracy_vs_stri,
    build_stri_lookup,
    load_stri_matrix,
    stri_pair_details,
)
from seedlearn.benchmarking.human.thumbnails import (
    load_specimen_image_paths,
    specimen_thumbnails,
)
from seedlearn.benchmarking.human.trait_prompts import PROMPT_TRAIT_TEXT
from seedlearn.benchmarking.human.value_map import TRAIT_SPECS

AXES = ("model_vs_roni", "model_vs_carmen", "roni_vs_carmen")
STRI_SOURCES = ("model", "roni", "carmen")

# trait_key -> TraitSpec, for the per-trait table's "what the model was asked" column.
_SPECS_BY_KEY = {s.key: s for s in TRAIT_SPECS}

# Embed example views at this size so the click-to-zoom lightbox stays crisp; each view
# is embedded once and reused by every drill-down modal. Chosen to keep the whole report
# near ~3x the thumbnail-less size while staying legible when zoomed.
THUMB_MAX_EDGE = 320
THUMB_QUALITY = 72

# Shared correct/incorrect cell colors (green matches the high-agreement shade).
COLOR_CORRECT = "#c8e6c9"
COLOR_WRONG = "#ffcdd2"
# Credited via a reviewable correction (typo/variant/synonym) -- distinct from raw green.
COLOR_CORRECTED = "#b3e5fc"


@dataclass
class ReportBundle:
    """Everything a report needs, assembled once."""

    agreements: list[TraitAgreement]
    overall: dict
    aggregates: list[SpecimenAggregate]
    id_records: list[HumanIDRecord]
    id_acc: dict
    n_model_specimens: int
    n_annotated_individuals: int
    stri_results: list[STRIAgreement] = field(default_factory=list)
    n_stri_matched: int = 0
    # Per-specimen drill-down detail keyed by trait_key -> axis -> [PairDetail].
    pair_details: dict[str, dict[str, list[PairDetail]]] = field(default_factory=dict)
    # Per-specimen STRI drill-down detail keyed by trait_key -> source -> [STRIPairDetail].
    stri_details: dict[str, dict[str, list[STRIPairDetail]]] = field(default_factory=dict)
    # Provenance: the source model run's metadata + where it lives, and when this
    # report was generated. Empty dict when the run recorded no run_metadata.json.
    run_metadata: dict = field(default_factory=dict)
    results_dir: str = ""
    generated_at: str = ""
    # specimen_id -> [base64 thumbnail data URI] for every annotated view (may be empty).
    thumbnails: dict[str, list[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble(
    results_dir: str | Path,
    roni_xlsx: str | Path,
    carmen_xlsx: str | Path | None,
    curator_key: str | Path,
    stri_matrix: str | Path | None = None,
    corrections_file: str | Path | None = "trait_grading/id_corrections.csv",
    embed_images: bool = True,
) -> ReportBundle:
    """Load all inputs and compute every grading result.

    When ``stri_matrix`` is provided, each source (model, Roni, Carmen) is also
    scored against the STRI trait matrix (match-any accuracy on the five
    STRI-coded traits). When ``corrections_file`` exists, Roni's species IDs are
    additionally graded corrected (typo / variant / synonym crediting). When
    ``embed_images`` is set, every annotated view is embedded as a base64 thumbnail.
    """
    run_metadata = _load_run_metadata(results_dir)
    corrections = load_corrections(corrections_file) if corrections_file else {}
    curator = load_curator_key(curator_key)
    records = []
    roni_records, _ = load_annotations(roni_xlsx, "roni", curator)
    records += roni_records
    if carmen_xlsx and Path(carmen_xlsx).exists():
        carmen_records, _ = load_annotations(carmen_xlsx, "carmen", curator)
        records += carmen_records

    aggregates = aggregate_records(records)
    model_traits = load_model_traits(results_dir)
    agreements = grade_all_axes(model_traits, aggregates)
    overall = overall_by_axis(agreements)
    id_records = grade_human_ids(roni_records, "roni", corrections)
    n_individuals = len({r.anonymous_id for r in records})

    # Source lookups, computed once for both the drill-down detail and STRI.
    model_lk = model_lookup(model_traits)
    roni_lk = human_lookup(aggregates, "roni")
    carmen_lk = human_lookup(aggregates, "carmen")
    roni_views = human_view_lookup(aggregates, "roni")
    carmen_views = human_view_lookup(aggregates, "carmen")

    # Per-specimen comparison detail behind every (trait, axis) cell. Only the
    # human side of each axis carries per-view raw values (the model has one).
    axis_details = {
        "model_vs_roni": pair_details(model_lk, roni_lk, "model_vs_roni", b_views=roni_views),
        "model_vs_carmen": pair_details(model_lk, carmen_lk, "model_vs_carmen", b_views=carmen_views),
        "roni_vs_carmen": pair_details(
            roni_lk, carmen_lk, "roni_vs_carmen", a_views=roni_views, b_views=carmen_views
        ),
    }
    details_by_trait: dict[str, dict[str, list[PairDetail]]] = defaultdict(dict)
    for axis, per_trait in axis_details.items():
        for trait_key, rows in per_trait.items():
            details_by_trait[trait_key][axis] = rows

    stri_results: list[STRIAgreement] = []
    stri_details: dict[str, dict[str, list[STRIPairDetail]]] = defaultdict(dict)
    n_stri_matched = 0
    if stri_matrix and Path(stri_matrix).exists():
        specimen_to_species = {
            e.specimen_id: f"{e.genus} {e.species}"
            for e in curator.values()
            if e.specimen_id
        }
        stri_rows = load_stri_matrix(str(stri_matrix))
        stri_lookup, n_stri_matched = build_stri_lookup(specimen_to_species, stri_rows)
        sources = {"model": model_lk, "roni": roni_lk, "carmen": carmen_lk}
        for name, lk in sources.items():
            stri_results += accuracy_vs_stri(lk, stri_lookup, name)
            for trait_key, rows in stri_pair_details(
                lk, stri_lookup, name, specimen_to_species
            ).items():
                stri_details[trait_key][name] = rows

    thumbnails: dict[str, list[str]] = {}
    if embed_images:
        thumbnails, _ = specimen_thumbnails(
            load_specimen_image_paths(results_dir),
            max_edge=THUMB_MAX_EDGE,
            quality=THUMB_QUALITY,
        )

    return ReportBundle(
        agreements=agreements,
        overall=overall,
        aggregates=aggregates,
        id_records=id_records,
        id_acc=id_accuracy(id_records),
        n_model_specimens=len(model_traits),
        n_annotated_individuals=n_individuals,
        stri_results=stri_results,
        n_stri_matched=n_stri_matched,
        pair_details=dict(details_by_trait),
        stri_details={k: dict(v) for k, v in stri_details.items()},
        run_metadata=run_metadata,
        results_dir=str(results_dir),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        thumbnails=thumbnails,
    )


def _load_run_metadata(results_dir: str | Path) -> dict:
    """Load ``run_metadata.json`` from the model-run dir; ``{}`` when absent."""
    path = Path(results_dir) / "run_metadata.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_trait_agreement_csv(path: Path, agreements: list[TraitAgreement]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trait_key", "axis", "n_compared", "n_agree", "agreement_rate", "cohen_kappa"])
        for a in agreements:
            w.writerow(
                [a.trait_key, a.axis, a.n_compared, a.n_agree, _fmt(a.agreement_rate), _fmt(a.cohen_kappa)]
            )


def write_overall_csv(path: Path, overall: dict) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "n_traits", "total_compared", "macro_agreement_rate", "macro_cohen_kappa"])
        for axis, s in overall.items():
            w.writerow(
                [axis, s["n_traits"], s["total_compared"], _fmt(s["macro_agreement_rate"]), _fmt(s["macro_cohen_kappa"])]
            )


def write_distributions_csv(path: Path, aggregates: list[SpecimenAggregate]) -> None:
    """One row per (specimen, annotator, trait): the mode used + every per-view value."""
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["anonymous_id", "specimen_id", "annotator", "trait_key", "mode",
             "n_views", "n_present", "per_view_values"]
        )
        for agg in aggregates:
            for key, t in agg.traits.items():
                w.writerow(
                    [agg.anonymous_id, agg.specimen_id or "", agg.annotator, key, t.mode,
                     t.n_views, t.n_present, " | ".join(t.raw_values)]
                )


def write_stri_csv(path: Path, stri_results: list[STRIAgreement]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["trait_key", "source", "n_compared", "n_correct", "accuracy_vs_stri",
             "n_kappa", "cohen_kappa"]
        )
        for r in stri_results:
            w.writerow(
                [r.trait_key, r.source, r.n_compared, r.n_correct, _fmt(r.accuracy),
                 r.n_kappa, _fmt(r.cohen_kappa)]
            )


def write_id_csv(path: Path, id_records: list[HumanIDRecord]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["anonymous_id", "specimen_id", "annotator", "true_family", "true_genus", "true_species",
             "pred_family", "pred_genus", "pred_species",
             "family_correct", "genus_correct", "species_correct",
             "family_corrected", "genus_corrected", "species_corrected",
             "correction_category"]
        )
        for r in id_records:
            cats = sorted(
                {c.category for c in (r.family_correction, r.genus_correction, r.species_correction)
                 if c is not None}
            )
            w.writerow(
                [r.anonymous_id, r.specimen_id or "", r.annotator, r.true_family, r.true_genus, r.true_species,
                 r.pred_family or "", r.pred_genus or "", r.pred_species or "",
                 int(r.family_correct), int(r.genus_correct), int(r.species_correct),
                 int(r.family_corrected), int(r.genus_corrected), int(r.species_corrected),
                 "|".join(cats)]
            )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _pivot(agreements: list[TraitAgreement]) -> dict[str, dict[str, TraitAgreement]]:
    pivot: dict[str, dict[str, TraitAgreement]] = defaultdict(dict)
    for a in agreements:
        pivot[a.trait_key][a.axis] = a
    return pivot


def _rate_color(rate: float | None) -> str:
    if rate is None:
        return "#eee"
    if rate >= 0.8:
        return "#c8e6c9"
    if rate >= 0.5:
        return "#fff3c4"
    return "#ffcdd2"


# Landis & Koch kappa bands -> color ramp (red = worse-than-chance ... green = almost
# perfect). Neutral grey when kappa is undefined. The agreement cells are colored by
# THIS (chance-corrected) rather than the raw rate; see _KAPPA_LEGEND.
_KAPPA_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.00, "#ef9a9a", "worse than chance (&kappa; &lt; 0)"),
    (0.20, "#ffcc80", "slight (0.00 - 0.20)"),
    (0.40, "#fff59d", "fair (0.21 - 0.40)"),
    (0.60, "#e6ee9c", "moderate (0.41 - 0.60)"),
    (0.80, "#c5e1a5", "substantial (0.61 - 0.80)"),
    (1.01, "#81c784", "almost perfect (0.81 - 1.00)"),
)
_KAPPA_UNDEFINED = "#eee"


def _kappa_color(kappa: float | None) -> str:
    """Color for a Cohen's kappa value using the Landis & Koch bands."""
    if kappa is None:
        return _KAPPA_UNDEFINED
    for upper, color, _label in _KAPPA_BANDS:
        if kappa < upper:
            return color
    return _KAPPA_BANDS[-1][1]


def _kappa_explainer() -> str:
    """The single, consolidated Cohen's kappa explanation for the report.

    Combines the plain-language "rate vs chance-corrected kappa" prose with the
    Landis & Koch color-swatch legend, so kappa is explained in exactly one place and
    the swatch color sits next to its interpretation.
    """
    swatches = "".join(
        f'<tr><td style="background:{color};width:34px"></td>'
        f'<td style="text-align:left">{label}</td></tr>'
        for _upper, color, label in _KAPPA_BANDS
    )
    swatches += (
        f'<tr><td style="background:{_KAPPA_UNDEFINED};width:34px"></td>'
        f'<td style="text-align:left">&kappa; undefined (&lt; 2 comparable, or one value only)</td></tr>'
    )
    return (
        '<div style="background:#f4f6f8;border:1px solid #cfd8dc;border-radius:6px;'
        'padding:12px 16px;max-width:760px;margin:14px 0">'
        "<b>How to read the agreement cells (rate &amp; Cohen&rsquo;s &kappa;)</b>"
        '<p class="note" style="margin:6px 0">'
        "The <b>agreement rate</b> is simply the fraction of specimens where two sources "
        "gave the same value. It can look high purely because one answer dominates a trait "
        '(if 90% of plants are &quot;simple&quot; leaves, two sources agree 90% of the time '
        "just by both saying &quot;simple&quot;). <b>Cohen&rsquo;s &kappa;</b> corrects for "
        "that chance agreement: it asks how much the sources agree <i>beyond</i> what you'd "
        "expect at random. So a trait with a high rate but a low &kappa; is &quot;easy&quot; "
        "(one answer dominates), while a high &kappa; means genuine agreement on a hard, "
        "varied trait. <b>The agreement cells below are colored by &kappa;</b> using these "
        "Landis &amp; Koch bands:</p>"
        '<table style="margin:6px 0">'
        "<tr><th>Color</th><th style='text-align:left'>&kappa; band (Landis &amp; Koch)</th></tr>"
        + swatches
        + "</table></div>"
    )


def _cell(a: TraitAgreement | None, trait_key: str = "", axis: str = "") -> str:
    if a is None or a.n_compared == 0:
        return '<td style="background:#eee;color:#999">--</td>'
    kappa = "" if a.cohen_kappa is None else f" / k={a.cohen_kappa:.2f}"
    key = _html.escape(f"{trait_key}|{axis}", quote=True)
    return (
        f'<td class="cell-click" data-detail-key="{key}" '
        f'style="background:{_kappa_color(a.cohen_kappa)};cursor:pointer" '
        f'title="click for the per-specimen comparison">'
        f"{a.agreement_rate:.2f}{kappa}<br><small>n={a.n_compared}</small></td>"
    )


def _prompt_cell(trait_key: str) -> str:
    """Left-aligned cell: the trait's prompt wording plus its graded canonical options.

    The wording comes from :data:`PROMPT_TRAIT_TEXT` (verbatim from the Vision-LLM's
    prompt form); the options come from the trait's :class:`TraitSpec.canonical_values`
    (the fuller set actually used for grading, which can exceed the prompt's list).
    """
    wording = PROMPT_TRAIT_TEXT.get(trait_key)
    asked = _html.escape(wording) if wording else "<i>not a sys4 prompt item</i>"
    spec = _SPECS_BY_KEY.get(trait_key)
    opts = ", ".join(spec.canonical_values) if spec and spec.canonical_values else ""
    graded = (
        f'<br><small>graded options: {_html.escape(opts)}</small>' if opts else ""
    )
    return f'<td style="text-align:left;max-width:340px">{asked}{graded}</td>'


def _stri_pivot(results: list[STRIAgreement]) -> dict[str, dict[str, STRIAgreement]]:
    pivot: dict[str, dict[str, STRIAgreement]] = defaultdict(dict)
    for r in results:
        pivot[r.trait_key][r.source] = r
    return pivot


def _stri_cell(r: STRIAgreement | None, trait_key: str = "", source: str = "") -> str:
    if r is None or r.n_compared == 0:
        return '<td style="background:#eee;color:#999">--</td>'
    key = _html.escape(f"stri|{trait_key}|{source}", quote=True)
    # Color stays keyed to the match-any accuracy (the headline for STRI); kappa is
    # shown for context, computed over the single-label subset only (n_kappa).
    if r.cohen_kappa is None:
        kappa = ""
        subset = ""
    else:
        kappa = f" / k={r.cohen_kappa:.2f}"
        subset = f", &kappa; n={r.n_kappa}"
    return (
        f'<td class="cell-click" data-detail-key="{key}" '
        f'style="background:{_rate_color(r.accuracy)};cursor:pointer" '
        f'title="click for the per-species comparison">'
        f"{r.accuracy:.2f}{kappa}<br><small>n={r.n_compared}{subset}</small></td>"
    )


# Self-contained drill-down modal: a shared <dialog> filled on cell click from
# the JSON data island. Plain string (not an f-string) so its braces are literal.
_DRILL_SCRIPT = """
<dialog id="drill">
  <div id="drill-head"></div>
  <div id="drill-body"></div>
  <form method="dialog" style="margin-top:10px"><button>close</button></form>
</dialog>
<dialog id="lightbox" class="lightbox"><img id="lightbox-img" alt="enlarged view"></dialog>
<script>
(function(){
  var el = document.getElementById('drill-data');
  var data = el ? JSON.parse(el.textContent) : {};
  var dlg = document.getElementById('drill');
  function cap(t){ return t ? t.charAt(0).toUpperCase() + t.slice(1) : t; }
  function esc(s){ var d = document.createElement('div'); d.textContent = (s == null ? '' : s); return d.innerHTML; }
  function views(v){ return (v && v.length) ? esc(v.join(' | ')) : ''; }
  var species = {};
  var spEl = document.getElementById('species-map');
  if (spEl) { try { species = JSON.parse(spEl.textContent); } catch (e) { species = {}; } }
  function spName(sid){ return species[sid] || ''; }

  // Example thumbnails are embedded once in the thumbs-map island and reused across
  // every drill-down modal (no base64 duplicated per trait). They render larger here
  // and each opens full-size in the lightbox on click.
  var thumbs = {};
  var thEl = document.getElementById('thumbs-map');
  if (thEl) { try { thumbs = JSON.parse(thEl.textContent); } catch (e) { thumbs = {}; } }
  var hasThumbs = false;
  for (var tk in thumbs) { if (thumbs.hasOwnProperty(tk)) { hasThumbs = true; break; } }
  function thumbImgs(sid, h){
    var arr = thumbs[sid] || [];
    var out = '';
    for (var i = 0; i < arr.length; i++){
      out += '<img class="thumb" src="' + arr[i] + '" loading="lazy" alt="view" ' +
        'title="click to enlarge" style="height:' + h + 'px">';
    }
    return out;
  }

  // Click any drill-down thumbnail to open it full-size in the lightbox (reuses the
  // already-embedded data URI). Delegated because modal content is built on the fly.
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lightbox-img');
  var body = document.getElementById('drill-body');
  if (body) body.addEventListener('click', function(ev){
    var t = ev.target;
    if (t && t.classList && t.classList.contains('thumb')){
      lbImg.src = t.src;
      if (typeof lb.showModal === 'function') lb.showModal(); else lb.setAttribute('open', 'open');
    }
  });
  if (lb) lb.addEventListener('click', function(){
    if (typeof lb.close === 'function') lb.close(); else lb.removeAttribute('open');
  });

  function renderStri(rows, trait, source){
    document.getElementById('drill-head').innerHTML =
      '<b>' + esc(trait) + '</b> &middot; ' + cap(source) + ' vs STRI' +
      ' <small>(' + rows.length + ' species, mismatches first)</small>';
    if (!rows.length) return '<p>No compared species for this trait and source.</p>';
    var html = '<table><tr><th>specimen</th><th>species</th><th>' + cap(source) +
      ' value</th><th>allowed STRI values</th>' + (hasThumbs ? '<th>views</th>' : '') + '</tr>';
    rows.forEach(function(r){
      var cls = r.g ? '' : ' class="disagree"';
      html += '<tr' + cls + '><td>' + esc(r.s) + '</td><td><i>' + esc(spName(r.s) || r.sp) +
        '</i></td><td>' + esc(r.v) + '</td><td>' + views(r.al) + '</td>' +
        (hasThumbs ? '<td style="text-align:left">' + thumbImgs(r.s, 180) + '</td>' : '') +
        '</tr>';
    });
    return html + '</table>';
  }

  function renderAgreement(rows, key){
    var parts = key.split('|');
    var trait = parts[0], sides = (parts[1] || '').split('_vs_');
    var la = cap(sides[0]), lb2 = cap(sides[1]);
    document.getElementById('drill-head').innerHTML =
      '<b>' + esc(trait) + '</b> &middot; ' + esc(la) + ' vs ' + esc(lb2) +
      ' <small>(' + rows.length + ' compared, disagreements first)</small>';
    if (!rows.length) return '<p>No compared specimens for this trait and axis.</p>';
    // Give each human side its own per-view column (so Roni vs Carmen shows
    // both); the model has no per-view values, so its column is omitted.
    var hasA = rows.some(function(r){ return r.av && r.av.length; });
    var hasB = rows.some(function(r){ return r.bv && r.bv.length; });
    var head = '<tr><th>specimen</th><th>species</th><th>' + esc(la) + '</th><th>' + esc(lb2) + '</th>';
    if (hasA) head += '<th>' + esc(la) + ' per-view</th>';
    if (hasB) head += '<th>' + esc(lb2) + ' per-view</th>';
    if (hasThumbs) head += '<th>views</th>';
    head += '</tr>';
    var html = '<table>' + head;
    rows.forEach(function(r){
      var cls = r.g ? '' : ' class="disagree"';
      html += '<tr' + cls + '><td>' + esc(r.s) + '</td><td><i>' + esc(spName(r.s)) +
        '</i></td><td>' + esc(r.a) + '</td><td>' + esc(r.b) + '</td>';
      if (hasA) html += '<td>' + views(r.av) + '</td>';
      if (hasB) html += '<td>' + views(r.bv) + '</td>';
      if (hasThumbs) html += '<td style="text-align:left">' + thumbImgs(r.s, 180) + '</td>';
      html += '</tr>';
    });
    return html + '</table>';
  }

  document.querySelectorAll('.cell-click').forEach(function(td){
    td.addEventListener('click', function(){
      var key = td.getAttribute('data-detail-key');
      var rows = data[key] || [];
      var parts = key.split('|');
      var html = (parts[0] === 'stri')
        ? renderStri(rows, parts[1], parts[2])
        : renderAgreement(rows, key);
      document.getElementById('drill-body').innerHTML = html;
      if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', 'open');
    });
  });
})();
</script>
"""


def _system_prompt_block(bundle: ReportBundle) -> str:
    """Collapsible section reproducing the exact system prompt sent to the model.

    Uses the run's recorded ``prompt_style``; for older runs that did not record it,
    falls back to the registry default (``sys4``) and says so explicitly.
    """
    from seedlearn.components.analyzers.prompts import get_prompt

    recorded = bundle.run_metadata.get("prompt_style")
    style = recorded or "sys4"
    inferred = recorded in (None, "")
    try:
        text = get_prompt(style)
    except (KeyError, ValueError):
        return (
            f'<p class="note">System prompt unavailable: unknown prompt style '
            f"<code>{_html.escape(str(style))}</code>.</p>"
        )
    note = (
        ' <small style="color:#b26a00">(style not recorded in this run — showing the '
        "registry default; newer runs record the exact style)</small>"
        if inferred else ""
    )
    return (
        "<details style='margin:14px 0;max-width:900px'>"
        f"<summary><b>System prompt sent to the model</b> "
        f"(style: <code>{_html.escape(style)}</code>){note}</summary>"
        "<pre style='white-space:pre-wrap;background:#f4f6f8;border:1px solid #cfd8dc;"
        f"border-radius:6px;padding:12px 16px;font-size:13px'>{_html.escape(text)}</pre>"
        "</details>"
    )


def _provenance_block(bundle: ReportBundle) -> str:
    """Render a metadata header: when the report ran and which model run it used."""
    m = bundle.run_metadata

    def field_row(label: str, value: object) -> str:
        shown = _html.escape(str(value)) if value not in (None, "") else "<i>not recorded</i>"
        return f"<tr><td style='text-align:left'><b>{label}</b></td><td style='text-align:left'>{shown}</td></tr>"

    rows = [
        field_row("Report generated", bundle.generated_at or "unknown"),
        field_row("Model run dir", bundle.results_dir or "unknown"),
        field_row("Model run started", m.get("started_at")),
        field_row("Model", m.get("model")),
        field_row("Prompt style", m.get("prompt_style")),
        field_row("Specimens in run", m.get("n_specimens")),
    ]
    return (
        '<table style="margin:12px 0"><caption>Run provenance</caption>'
        + "".join(rows)
        + "</table>"
    )


def generate_html(path: Path, bundle: ReportBundle) -> None:
    pivot = _pivot(bundle.agreements)
    has_stri = bool(bundle.stri_results)
    stri_pivot = _stri_pivot(bundle.stri_results)
    rows = []
    for trait_key in sorted(pivot):
        cells = "".join(_cell(pivot[trait_key].get(ax), trait_key, ax) for ax in AXES)
        if has_stri:
            cells += "".join(
                _stri_cell(stri_pivot.get(trait_key, {}).get(src), trait_key, src)
                for src in STRI_SOURCES
            )
        rows.append(
            f"<tr><td><b>{_html.escape(trait_key)}</b></td>"
            f"{_prompt_cell(trait_key)}{cells}</tr>"
        )

    stri_headers = (
        "<th>model vs STRI</th><th>Roni vs STRI</th><th>Carmen vs STRI</th>"
        if has_stri else ""
    )
    stri_note = (
        f'<p class="note">The <b>vs STRI</b> columns score each source against the '
        f"STRI botanical trait matrix (match-any: a value is correct if it is among "
        f"the species' allowed STRI values). STRI only codes five traits, so the other "
        f"rows are blank. Each cell also shows Cohen's &kappa; &mdash; but because STRI "
        f"is a <i>multi-label</i> reference (a species can allow several values), &kappa; "
        f"is computed only over the <b>single-label subset</b> (species STRI codes with "
        f"exactly one value; that subset size is shown as <i>&kappa; n</i>), where a "
        f"symmetric single-label comparison is well defined. Based on "
        f"{bundle.n_stri_matched} specimens whose species matched a STRI row.</p>"
        if has_stri else ""
    )

    # Per-(trait, axis) drill-down payload, embedded as a JSON data island. Short
    # keys keep it compact; the escape stops a raw "</" from closing the script.
    detail_payload: dict[str, list[dict]] = {}
    for trait_key, axis_map in bundle.pair_details.items():
        for axis, detail_rows in axis_map.items():
            detail_payload[f"{trait_key}|{axis}"] = [
                {"s": d.specimen_id, "a": d.value_a, "b": d.value_b,
                 "g": d.agree, "av": d.a_views, "bv": d.b_views}
                for d in detail_rows
            ]
    # STRI drill-down rows share the island under a "stri|trait|source" namespace.
    for trait_key, src_map in bundle.stri_details.items():
        for source, detail_rows in src_map.items():
            detail_payload[f"stri|{trait_key}|{source}"] = [
                {"s": d.specimen_id, "sp": d.species, "v": d.value,
                 "al": d.allowed, "g": d.correct}
                for d in detail_rows
            ]
    detail_json = json.dumps(detail_payload).replace("</", "<\\/")
    drill_data_island = (
        f'<script type="application/json" id="drill-data">{detail_json}</script>'
    )

    # specimen_id -> "Genus species" so every drill-down row can show the species.
    species_map = {
        r.specimen_id: f"{r.true_genus} {r.true_species}".strip()
        for r in bundle.id_records
        if r.specimen_id
    }
    species_json = json.dumps(species_map).replace("</", "<\\/")
    species_island = (
        f'<script type="application/json" id="species-map">{species_json}</script>'
    )

    overall_rows = "".join(
        f"<tr><td>{ax}</td><td>{_fmt(bundle.overall.get(ax, {}).get('macro_agreement_rate'), 2)}</td>"
        f"<td>{_fmt(bundle.overall.get(ax, {}).get('macro_cohen_kappa'), 2)}</td>"
        f"<td>{bundle.overall.get(ax, {}).get('total_compared', 0)}</td></tr>"
        for ax in AXES
    )

    acc = bundle.id_acc
    if acc.get("n_graded"):
        n = acc.get("n_graded", 0)
        id_summary = (
            f"<b>Raw:</b> family {acc.get('family_accuracy', 0):.1%}, "
            f"genus {acc.get('genus_accuracy', 0):.1%}, "
            f"species {acc.get('species_accuracy', 0):.1%} &nbsp;&middot;&nbsp; "
            f"<b>Corrected</b> (typo/variant/synonym credited): "
            f"family {acc.get('corrected_family_accuracy', 0):.1%}, "
            f"genus {acc.get('corrected_genus_accuracy', 0):.1%}, "
            f"species {acc.get('corrected_species_accuracy', 0):.1%} &nbsp;(n={n})"
        )
    else:
        id_summary = "no Roni IDs graded"

    def _id_cell(pred: str | None, true_value: str, correct: bool,
                 corrected: bool, correction) -> str:
        # One of Roni's rank predictions. Green when raw-correct; blue + a
        # "<category> -> <canonical>" note when credited by a reviewable
        # correction (original text kept); red + the true value when still wrong.
        val = _html.escape(pred or "--")
        if correct:
            return f'<td style="background:{COLOR_CORRECT}">{val}</td>'
        if corrected and correction is not None:
            note = f"{correction.category} &rarr; {_html.escape(correction.canonical)}"
            return (
                f'<td style="background:{COLOR_CORRECTED}">{val}'
                f"<br><small>{note}</small></td>"
            )
        return (
            f'<td style="background:{COLOR_WRONG}">{val}'
            f"<br><small>true: {_html.escape(true_value)}</small></td>"
        )

    has_images = bool(bundle.thumbnails)

    # Every embedded view, once, keyed by specimen -> consumed by the drill-down modals
    # and the click-to-zoom lightbox (the species-ID table no longer shows thumbnails).
    thumbs_island = ""
    if has_images:
        thumbs_json = json.dumps(bundle.thumbnails).replace("</", "<\\/")
        thumbs_island = (
            f'<script type="application/json" id="thumbs-map">{thumbs_json}</script>'
        )

    id_rows = "".join(
        f"<tr><td>{_html.escape(r.specimen_id or r.anonymous_id)}</td>"
        f"<td>{_html.escape(r.true_genus)} {_html.escape(r.true_species)}</td>"
        f"{_id_cell(r.pred_family, r.true_family, r.family_correct, r.family_corrected, r.family_correction)}"
        f"{_id_cell(r.pred_genus, r.true_genus, r.genus_correct, r.genus_corrected, r.genus_correction)}"
        f"{_id_cell(r.pred_species, r.true_species, r.species_correct, r.species_corrected, r.species_correction)}"
        f"</tr>"
        for r in bundle.id_records
    )

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Human trait grading report</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;margin:12px 0}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:center;font-size:14px}}
th{{background:#37474f;color:#fff}}
caption{{font-weight:bold;text-align:left;margin-bottom:6px}}
small{{color:#555}}
.note{{color:#555;max-width:760px}}
.cell-click{{cursor:pointer;box-shadow:inset 0 -3px 0 rgba(55,71,79,0.28)}}
.cell-click:hover{{outline:2px solid #37474f}}
.click-hint{{font-size:15px;color:#37474f}}
dialog{{border:1px solid #90a4ae;border-radius:8px;max-width:90vw;max-height:85vh;
       overflow:auto;padding:18px 20px}}
dialog::backdrop{{background:rgba(0,0,0,0.35)}}
#drill-body table{{margin:8px 0}}
#drill-body tr.disagree td{{background:{COLOR_WRONG}}}
.thumb{{margin:2px;border:1px solid #ccc;border-radius:2px;vertical-align:middle;cursor:zoom-in}}
.thumb:hover{{outline:2px solid #37474f}}
.lightbox{{border:none;background:transparent;padding:0;max-width:96vw;max-height:96vh;overflow:visible}}
.lightbox::backdrop{{background:rgba(0,0,0,0.8)}}
#lightbox-img{{max-width:92vw;max-height:92vh;display:block;cursor:zoom-out;
             box-shadow:0 0 24px rgba(0,0,0,0.6)}}
</style></head><body>
<h1>Human trait-grading report</h1>
<p class="note" style="max-width:820px"><b>What this report is.</b> It checks how well the
AI vision model &quot;reads&quot; a seedling's leaf and stem traits from photos. For each
trait, the model's answer is compared against two trained botanists (Roni and Carmen) and,
where available, an independent botanical reference (the STRI trait database). It also
checks how accurately one botanist (Roni) named the plants from those same photos, against
the true species. Higher agreement is better — and because even two experts don't always
agree, the <b>Roni-vs-Carmen</b> column is the realistic bar to judge the model against,
not 100%.</p>
{_provenance_block(bundle)}
{_system_prompt_block(bundle)}
<p class="note" style="max-width:820px"><b>How to read the tables.</b>
{bundle.n_model_specimens} specimens were assessed by the model, with
{bundle.n_annotated_individuals} sets of human annotations. Every number below is an
<b>agreement rate</b> — how often two sources gave the same answer for a trait. A botanist
labels each trait once per photo, and those per-photo labels are reduced to their single
most common value before comparing (explained in detail below). The
<b>Roni vs Carmen</b> column is how often the two humans agree with each other — the ceiling
the model's columns should be read against.</p>

<table><caption>Overall (macro-averaged across traits)</caption>
<tr><th>Axis</th><th>Macro rate</th><th>Macro k</th><th>Pairs compared</th></tr>
{overall_rows}
</table>
<p class="note" style="margin:6px 0 14px">
<b>What this table means.</b> Each row is one comparison axis, summarized across
<i>all</i> gradable traits at once. <b>Macro</b> means every trait counts equally:
we compute the metric per trait, then take the plain average over the traits that
had at least one comparison — so a common trait and a rare one carry the same
weight (this is different from a specimen-weighted average, where trait sizes would
dominate). <b>Macro rate</b> = the mean of the per-trait agreement rates.
<b>Macro k</b> = the mean of the per-trait Cohen's &kappa; (traits whose &kappa; is
undefined are skipped). <b>Pairs compared</b> = the total number of specimen&times;trait
comparisons behind the axis (a raw count summed over traits, <i>not</i> averaged) —
it tells you how much data the row rests on. The per-trait breakdown is in the
<b>Per-trait agreement</b> table below.</p>

{_kappa_explainer()}

{stri_note}
<div style="background:#f4f6f8;border:1px solid #cfd8dc;border-radius:6px;
            padding:12px 16px;max-width:760px;margin:14px 0">
<b>How a human's trait value is determined across views</b>
<p class="note" style="margin:6px 0">
Each annotator labels every trait <b>once per view</b> (a specimen has several
view images). To compare against the model's single pooled value, each
annotator's per-view values are canonicalized (Spanish/English &rarr; shared
tokens), blank / <i>no claro</i> / <i>not observed</i> values are treated as
missing and dropped, and the remaining values are collapsed to the <b>modal
value</b> (the most frequent; ties broken by first view order). That mode is what
is scored. <b>Click any agreement cell below</b> to see the per-specimen
comparison, including each human's per-view values behind the mode; the full
distribution is also in <code>human_trait_distributions.csv</code>.</p>
</div>
<p class="note click-hint">&#128070; <b>Click any colored cell</b> (agreement or vs-STRI)
to see the per-specimen breakdown behind it — including example photos you can click to
enlarge.</p>
{'<p class="note">The <b>vs STRI</b> cells are colored by <b>accuracy</b> (match-any): '
 'green &ge; 0.80, yellow &ge; 0.50, red &lt; 0.50. The &kappa; printed on each STRI cell '
 'is over the single-label subset only, so it is shown for context and is not the basis '
 'for the color.</p>'
 if has_stri else ''}
<table><caption>Per-trait agreement</caption>
<tr><th>Trait</th><th>What the model was asked<br><small style="color:#cfd8dc">(prompt wording &amp; graded options)</small></th><th>model vs Roni</th><th>model vs Carmen</th><th>Roni vs Carmen (ceiling)</th>{stri_headers}</tr>
{''.join(rows)}
</table>

<h2>Roni species ID vs truth</h2>
<p class="note">{id_summary}</p>
<p class="note">Each cell shows Roni's original prediction. <span style="background:{COLOR_CORRECT};
padding:1px 5px">green</span> = raw-correct; <span style="background:{COLOR_CORRECTED};
padding:1px 5px">blue</span> = credited by a reviewable correction (the
<i>category &rarr; canonical</i> is shown beneath, and the original text is kept);
<span style="background:{COLOR_WRONG};padding:1px 5px">red</span> = still wrong (true value
beneath). Corrections are curated in <code>trait_grading/id_corrections.csv</code>; the raw
data is never edited.</p>
<table><caption>Per-individual ID outcome</caption>
<tr><th>Specimen</th><th>True (genus species)</th><th>Roni family</th>
<th>Roni genus</th><th>Roni species</th></tr>
{id_rows}
</table>

{drill_data_island}
{species_island}
{thumbs_island}
{_DRILL_SCRIPT}
</body></html>"""
    path.write_text(doc)


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run_report(
    results_dir: str | Path,
    roni_xlsx: str | Path,
    carmen_xlsx: str | Path | None,
    curator_key: str | Path,
    out_dir: str | Path,
    html: bool = True,
    stri_matrix: str | Path | None = None,
    corrections_file: str | Path | None = "trait_grading/id_corrections.csv",
    embed_images: bool = True,
) -> ReportBundle:
    """Assemble grading results and write all CSVs (+ optional HTML) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = assemble(
        results_dir, roni_xlsx, carmen_xlsx, curator_key, stri_matrix,
        corrections_file, embed_images,
    )

    write_trait_agreement_csv(out / "trait_agreement_per_trait.csv", bundle.agreements)
    write_overall_csv(out / "trait_agreement_overall.csv", bundle.overall)
    write_distributions_csv(out / "human_trait_distributions.csv", bundle.aggregates)
    write_id_csv(out / "roni_id_accuracy.csv", bundle.id_records)
    (out / "roni_id_summary.json").write_text(json.dumps(bundle.id_acc, indent=2))
    if bundle.stri_results:
        write_stri_csv(out / "stri_accuracy.csv", bundle.stri_results)
    if html:
        generate_html(out / "human_grading_report.html", bundle)
    return bundle
