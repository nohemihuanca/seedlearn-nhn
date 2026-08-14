#!/usr/bin/env python3
"""Grade Vision-LLM trait predictions against human annotations.

Loads the human annotation spreadsheets (Roni, Carmen), joins blinded IDs to
specimens and true taxonomy via the curator key, aggregates per-view annotations
to per-specimen modal values, and grades three agreement axes
(model-vs-Roni, model-vs-Carmen, Roni-vs-Carmen) plus Roni's species IDs.

Writes CSV tables and a self-contained HTML report.

Example:
    python scripts/grade_human_annotations.py \
        --results-dir trait_grading/model_run/<timestamp> \
        --roni-xlsx trait_grading/annotations/roni_bianco.xlsx \
        --carmen-xlsx trait_grading/annotations/carmen.xlsx \
        --curator-key trait_grading/keys/curator_taxonomic_key.csv \
        --out-dir trait_grading/reports \
        --html
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from seedlearn.benchmarking.human.report import run_report

_TG = Path("trait_grading")
_DEFAULT_STRI = (
    "data/traits/stri_web_keys/per_key_trait_matrices/"
    "cl185_complete_tree_species_of_panama_trait_matrix.csv"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory of per-specimen pipeline JSONs (e.g. trait_grading/model_run/<ts>)",
    )
    p.add_argument("--roni-xlsx", default=str(_TG / "annotations/roni_bianco.xlsx"))
    p.add_argument("--carmen-xlsx", default=str(_TG / "annotations/carmen.xlsx"))
    p.add_argument("--curator-key", default=str(_TG / "keys/curator_taxonomic_key.csv"))
    p.add_argument(
        "--out-dir",
        default=None,
        help="Report output directory (default: trait_grading/reports/<timestamp>/, "
        "so each run is kept as a distinct version).",
    )
    p.add_argument("--html", action="store_true", help="Also write the HTML report")
    p.add_argument(
        "--stri-matrix", default=_DEFAULT_STRI,
        help="STRI trait matrix CSV; adds model/Roni/Carmen vs-STRI accuracy columns.",
    )
    p.add_argument(
        "--no-stri", action="store_true", help="Skip the STRI comparison axis.",
    )
    p.add_argument(
        "--no-images", action="store_true",
        help="Skip embedding per-view thumbnails (smaller, faster HTML).",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(_TG / "reports" / datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    bundle = run_report(
        results_dir=args.results_dir,
        roni_xlsx=args.roni_xlsx,
        carmen_xlsx=args.carmen_xlsx,
        curator_key=args.curator_key,
        out_dir=out_dir,
        html=args.html,
        stri_matrix=None if args.no_stri else args.stri_matrix,
        embed_images=not args.no_images,
    )

    print(f"Graded {bundle.n_model_specimens} model specimens vs "
          f"{bundle.n_annotated_individuals} annotated individuals.")
    for axis, s in bundle.overall.items():
        rate = s["macro_agreement_rate"]
        kappa = s["macro_cohen_kappa"]
        rate_s = "n/a" if rate is None else f"{rate:.2f}"
        kappa_s = "n/a" if kappa is None else f"{kappa:.2f}"
        print(f"  {axis:16s} macro_rate={rate_s} macro_kappa={kappa_s} pairs={s['total_compared']}")
    if bundle.stri_results:
        by_src: dict[str, list[float]] = {}
        for r in bundle.stri_results:
            if r.accuracy is not None:
                by_src.setdefault(r.source, []).append(r.accuracy)
        print(f"  vs STRI ({bundle.n_stri_matched} species matched, 5 traits):")
        for src, accs in by_src.items():
            macro = sum(accs) / len(accs) if accs else float("nan")
            print(f"    {src:8s} macro_accuracy={macro:.2f}")
    acc = bundle.id_acc
    if acc.get("n_graded"):
        print(f"  Roni ID: family={acc['family_accuracy']:.2f} "
              f"genus={acc['genus_accuracy']:.2f} species={acc['species_accuracy']:.2f}")
    print(f"Reports written to {out_dir}/")


if __name__ == "__main__":
    main()
