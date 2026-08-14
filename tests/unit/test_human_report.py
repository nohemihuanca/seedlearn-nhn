"""Tests for human-grading report assembly + writers (human.report)."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.aggregate import SpecimenAggregate, SpecimenTraitAgg
from seedlearn.benchmarking.human.categorical_grader import PairDetail, TraitAgreement
from seedlearn.benchmarking.human.id_corrections import Correction
from seedlearn.benchmarking.human.id_grading import HumanIDRecord
from seedlearn.benchmarking.human.stri_compare import STRIAgreement, STRIPairDetail
from seedlearn.benchmarking.human import report as R

REPO = Path(__file__).resolve().parents[2]
RONI = REPO / "trait_grading/annotations/roni_bianco.xlsx"
CARMEN = REPO / "trait_grading/annotations/carmen.xlsx"
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"
PRIOR_RUN = REPO / "results/benchmarks/2026-03-04_181054"
STRI_MATRIX = REPO / (
    "data/traits/stri_web_keys/per_key_trait_matrices/"
    "cl185_complete_tree_species_of_panama_trait_matrix.csv"
)


def _bundle():
    aggs = [
        SpecimenAggregate(
            "i1", "SR1", "roni",
            {"leaf_margin": SpecimenTraitAgg("leaf_margin", "entire",
                                             ["entire", "entire"], ["entero", "entero"], 2, 2)},
        )
    ]
    agreements = [
        TraitAgreement("leaf_margin", "model_vs_roni", 10, 8, 0.8, 0.6),
        TraitAgreement("leaf_margin", "model_vs_carmen", 0, 0, None, None),
        TraitAgreement("leaf_margin", "roni_vs_carmen", 9, 8, 0.89, 0.7),
    ]
    id_records = [
        HumanIDRecord("roni", "i1", "SR1", "Acanthaceae", "Aphelandra", "scabra",
                      "Acanthaceae", "Aphelandra", "scabra", True, True, True)
    ]
    details = {
        "leaf_margin": {
            "model_vs_roni": [
                PairDetail("leaf_margin", "model_vs_roni", "SR1", "entire", "entire",
                           True, [], ["entero", "entero"]),
                PairDetail("leaf_margin", "model_vs_roni", "SR2", "toothed", "entire",
                           False, [], ["entero", "dentado"]),
            ]
        }
    }
    return R.ReportBundle(
        agreements=agreements,
        overall={"model_vs_roni": {"n_traits": 1, "total_compared": 10,
                                   "macro_agreement_rate": 0.8, "macro_cohen_kappa": 0.6}},
        aggregates=aggs,
        id_records=id_records,
        id_acc={"n_graded": 1, "family_accuracy": 1.0, "genus_accuracy": 1.0, "species_accuracy": 1.0},
        n_model_specimens=1,
        n_annotated_individuals=1,
        pair_details=details,
    )


def test_csv_writers(tmp_path):
    b = _bundle()
    R.write_trait_agreement_csv(tmp_path / "agg.csv", b.agreements)
    R.write_distributions_csv(tmp_path / "dist.csv", b.aggregates)
    R.write_id_csv(tmp_path / "id.csv", b.id_records)
    agg = (tmp_path / "agg.csv").read_text()
    assert "leaf_margin,model_vs_roni,10,8,0.800,0.600" in agg
    dist = (tmp_path / "dist.csv").read_text()
    # mode + per-view distribution both present
    assert "entire" in dist and "entero | entero" in dist
    assert "1,1,1" in (tmp_path / "id.csv").read_text()  # all three correct


def test_generate_html_renders_axes_and_id(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "model vs Roni" in doc
    assert "Roni vs Carmen (ceiling)" in doc
    assert "--" in doc  # the n_compared==0 model_vs_carmen cell
    assert "Aphelandra" in doc  # ID section


def test_generate_html_has_kappa_guide_and_id_columns(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # Cohen's kappa interpretation guide (consolidated explainer)
    assert "How to read the agreement cells" in doc
    assert "almost perfect" in doc
    assert "Landis" in doc
    # Split Roni ID columns (family / genus / species)
    assert "Roni family" in doc
    assert "Roni genus" in doc
    assert "Roni species" in doc
    assert "Acanthaceae" in doc
    # The old combined column + OK/X correctness columns are gone.
    assert "Roni predicted" not in doc


def test_correct_id_cells_are_green(tmp_path):
    b = _bundle()  # _bundle's record is all-correct
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # The correct family/genus/species cells carry the green background; no true: text.
    assert f"background:{R.COLOR_CORRECT}" in doc
    assert "true:" not in doc


def test_id_cells_show_true_value_when_wrong(tmp_path):
    b = _bundle()
    b.id_records[0] = HumanIDRecord(
        "roni", "i1", "SR1", "Acanthaceae", "Aphelandra", "scabra",
        "Fabaceae", "Inga", "fulgida", False, False, False,
    )
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert f"background:{R.COLOR_WRONG}" in doc
    assert "true: Acanthaceae" in doc   # wrong family
    assert "true: Aphelandra" in doc    # wrong genus, shown in its own cell
    assert "true: scabra" in doc        # wrong species, shown in its own cell


def test_id_genus_right_species_wrong_colors_independently(tmp_path):
    b = _bundle()
    b.id_records[0] = HumanIDRecord(
        "roni", "i1", "SR1", "Acanthaceae", "Aphelandra", "scabra",
        "Acanthaceae", "Aphelandra", "fulgida", True, True, False,
    )
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # species miss shows its true value; genus (correct) does not add a true: line
    assert "true: scabra" in doc
    assert "true: Aphelandra" not in doc


def test_corrected_id_cell_is_blue_with_note_and_original(tmp_path):
    b = _bundle()
    corr = Correction("SR1", "species", "hayessi", "hayesii", "typo")
    b.id_records[0] = HumanIDRecord(
        "roni", "i1", "SR1", "Annonaceae", "Annona", "hayesii",
        "Annonaceae", "Annona", "hayessi",
        family_correct=True, genus_correct=True, species_correct=False,
        family_corrected=True, genus_corrected=True, species_corrected=True,
        species_correction=corr,
    )
    b.id_acc = {
        "n_graded": 1, "family_accuracy": 1.0, "genus_accuracy": 1.0, "species_accuracy": 0.0,
        "corrected_family_accuracy": 1.0, "corrected_genus_accuracy": 1.0,
        "corrected_species_accuracy": 1.0,
    }
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert f"background:{R.COLOR_CORRECTED}" in doc      # blue correction shade
    assert "hayessi" in doc                               # original text preserved
    assert "typo &rarr; hayesii" in doc                   # category -> canonical note
    # Headline carries both raw and corrected triples
    assert "Raw:" in doc and "Corrected" in doc
    assert "species 0.0%" in doc and "species 100.0%" in doc


def test_id_csv_has_corrected_columns(tmp_path):
    b = _bundle()
    corr = Correction("SR1", "genus", "Monteverdia", "Maytenus", "synonym")
    b.id_records[0] = HumanIDRecord(
        "roni", "i1", "SR1", "Celastraceae", "Maytenus", "schippii",
        "Celastraceae", "Monteverdia", "schippii",
        family_correct=True, genus_correct=False, species_correct=True,
        family_corrected=True, genus_corrected=True, species_corrected=True,
        genus_correction=corr,
    )
    R.write_id_csv(tmp_path / "id.csv", b.id_records)
    text = (tmp_path / "id.csv").read_text()
    assert "family_corrected,genus_corrected,species_corrected,correction_category" in text
    assert "synonym" in text


def test_thumbnails_embedded_once_for_modals_with_lightbox(tmp_path):
    b = _bundle()
    b.thumbnails = {"SR1": ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]}
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # Thumbnails live once in the shared island (consumed by the drill-down modals +
    # lightbox), so each base64 view appears exactly once -- no per-trait duplication.
    assert 'id="thumbs-map"' in doc
    assert doc.count("data:image/jpeg;base64,") == 2
    assert 'class="thumb"' in doc                 # img built by the drill script
    assert 'id="lightbox"' in doc                 # click-to-zoom overlay present
    # No thumbnails in the species-ID table anymore.
    assert "<th>Views</th>" not in doc
    assert 'class="thumbs"' not in doc and 'data-specimen=' not in doc


def test_id_table_never_shows_views_column(tmp_path):
    b = _bundle()
    b.thumbnails = {"SR1": ["data:image/jpeg;base64,AAA"]}
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "<th>Views</th>" not in doc            # removed from the ID table
    assert 'id="thumbs-map"' in doc               # but still embedded for the modals


def test_drilldown_data_island_and_clickable_cells(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # JSON data island carries the per-specimen detail
    assert 'id="drill-data"' in doc
    assert "SR2" in doc  # disagreeing specimen from the detail payload
    # Cells are clickable and the modal scaffold + script are present
    assert "cell-click" in doc
    assert 'data-detail-key="leaf_margin|model_vs_roni"' in doc
    assert "<dialog id=\"drill\">" in doc


def test_report_is_self_contained(tmp_path):
    b = _bundle()
    b.thumbnails = {"SR1": ["data:image/jpeg;base64,AAA"]}
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # No external/network asset references (script is inline; images are data URIs
    # embedded in the thumbs-map island and rendered as <img src="data:..."> at runtime).
    assert "http://" not in doc and "https://" not in doc
    assert 'src="http' not in doc                    # no remote images
    assert "data:image/jpeg;base64,AAA" in doc       # thumbnail embedded inline


def test_methods_note_explains_per_view_aggregation(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "most common value" in doc
    assert "once per view" in doc


def test_missing_model_cell_does_not_crash(tmp_path):
    # axis with no comparable specimens still renders
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    assert (tmp_path / "r.html").stat().st_size > 0


def test_kappa_color_bands():
    assert R._kappa_color(None) == R._KAPPA_UNDEFINED
    assert R._kappa_color(-0.1) == "#ef9a9a"   # worse than chance
    assert R._kappa_color(0.1) == "#ffcc80"    # slight
    assert R._kappa_color(0.5) == "#e6ee9c"    # moderate
    assert R._kappa_color(0.95) == "#81c784"   # almost perfect
    assert R._kappa_color(1.0) == "#81c784"    # boundary


def test_agreement_cells_colored_by_kappa_with_legend(tmp_path):
    b = _bundle()  # leaf_margin model_vs_roni has kappa 0.6 -> moderate band
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "background:#e6ee9c" in doc                       # kappa 0.6 -> moderate color
    assert "How to read the agreement cells" in doc          # consolidated explainer
    assert "Landis &amp; Koch" in doc
    assert "Click any colored cell" in doc                   # affordance hint
    assert "cell-click" in doc


def test_kappa_explained_once(tmp_path):
    # Guard against the old duplicate: there must be exactly one band legend (the swatch
    # table), and the old separate prose interpretation table must be gone.
    b = _bundle()
    doc = (tmp_path / "r.html")
    R.generate_html(doc, b)
    text = doc.read_text()
    assert text.count("&kappa; band (Landis &amp; Koch)") == 1     # single swatch legend
    assert "Interpretation (Landis &amp; Koch)" not in text        # old prose table gone
    assert "colored by &kappa;" in text                            # prose + swatches merged


def test_top_blurb_and_plain_intro_present(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "What this report is" in doc                      # top blurb
    assert "How to read the tables" in doc                   # plain-language intro
    # Old jargon intro is gone.
    assert "Vision-LLM Stage-1 trait predictions vs human annotators" not in doc


def test_per_trait_table_shows_prompt_column(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "What the model was asked" in doc                 # new column header
    # leaf_margin's prompt wording + its graded canonical options both render.
    assert "Leaf margin (entire / toothed)" in doc
    assert "graded options: entire, toothed, lobed" in doc


def test_stri_scale_note_only_with_stri(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    assert "colored by <b>accuracy</b>" not in (tmp_path / "r.html").read_text()
    b.stri_results = [STRIAgreement("leaf_margin", "model", 50, 40, 0.8)]
    R.generate_html(tmp_path / "r2.html", b)
    assert "colored by <b>accuracy</b>" in (tmp_path / "r2.html").read_text()


def test_system_prompt_section_uses_recorded_style(tmp_path):
    b = _bundle()
    b.run_metadata = {"prompt_style": "sys4"}
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "System prompt sent to the model" in doc
    assert "style: <code>sys4</code>" in doc
    assert "botanical expert" in doc            # actual SYS4 prompt text
    assert "<details" in doc and "<pre" in doc  # collapsible + preformatted
    assert "not recorded in this run" not in doc  # style WAS recorded


def test_system_prompt_section_infers_default_when_missing(tmp_path):
    b = _bundle()  # run_metadata {} -> no prompt_style
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "style: <code>sys4</code>" in doc
    assert "not recorded in this run" in doc     # inferred note


def test_provenance_block_shows_run_metadata(tmp_path):
    b = _bundle()
    b.run_metadata = {
        "started_at": "2026-06-26T14:03:29",
        "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
        "prompt_style": "sys4",
        "n_specimens": 114,
    }
    b.results_dir = "trait_grading/model_run/2026-06-26_140321"
    b.generated_at = "2026-07-06T10:00:00"
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "Run provenance" in doc
    assert "trait_grading/model_run/2026-06-26_140321" in doc
    assert "sys4" in doc
    assert "2026-07-06T10:00:00" in doc
    assert "Qwen/Qwen3-VL-32B-Instruct-FP8" in doc


def test_provenance_block_handles_missing_metadata(tmp_path):
    b = _bundle()  # run_metadata defaults to {}
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "Run provenance" in doc
    assert "not recorded" in doc  # missing model/prompt fields


def test_load_run_metadata_reads_and_tolerates_missing(tmp_path):
    (tmp_path / "run_metadata.json").write_text('{"prompt_style": "sys4"}')
    assert R._load_run_metadata(tmp_path) == {"prompt_style": "sys4"}
    assert R._load_run_metadata(tmp_path / "nope") == {}


def test_stri_columns_and_csv(tmp_path):
    b = _bundle()
    b.stri_results = [
        STRIAgreement("leaf_margin", "model", 50, 40, 0.8),
        STRIAgreement("leaf_margin", "roni", 48, 44, 0.917),
        STRIAgreement("leaf_margin", "carmen", 47, 41, 0.872),
    ]
    b.n_stri_matched = 76
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "model vs STRI" in doc
    assert "Roni vs STRI" in doc
    assert "Carmen vs STRI" in doc
    assert "76 specimens" in doc  # STRI note with matched count
    R.write_stri_csv(tmp_path / "stri.csv", b.stri_results)
    csv_text = (tmp_path / "stri.csv").read_text()
    assert "n_kappa,cohen_kappa" in csv_text
    assert "leaf_margin,model,50,40,0.800" in csv_text


def test_stri_cell_shows_subset_kappa(tmp_path):
    b = _bundle()
    b.stri_results = [STRIAgreement("leaf_margin", "roni", 40, 34, 0.85, 30, 0.55)]
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "k=0.55" in doc                # subset kappa printed on the STRI cell
    assert "&kappa; n=30" in doc          # single-label subset size shown
    assert "single-label subset" in doc   # explanation in the STRI note


def test_overall_table_has_plain_language_explanation(tmp_path):
    b = _bundle()
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert "What this table means" in doc
    assert "Macro rate" in doc
    assert "Pairs compared" in doc


def test_no_stri_means_no_stri_columns(tmp_path):
    b = _bundle()  # stri_results defaults to empty
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    # No STRI table headers / cells when there is no STRI data (the drill-down
    # script always defines renderStri, so we check the column markup, not a substring).
    assert "<th>Roni vs STRI</th>" not in doc
    assert "data-detail-key=\"stri|" not in doc


def test_stri_cells_clickable_with_detail_island(tmp_path):
    b = _bundle()
    b.stri_results = [STRIAgreement("leaf_margin", "roni", 48, 44, 0.917)]
    b.stri_details = {
        "leaf_margin": {
            "roni": [
                STRIPairDetail("leaf_margin", "roni", "SR2", "Ruellia fulgida",
                               "lobed", ["entire"], False),
                STRIPairDetail("leaf_margin", "roni", "SR1", "Aphelandra scabra",
                               "toothed", ["entire", "toothed"], True),
            ]
        }
    }
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert 'data-detail-key="stri|leaf_margin|roni"' in doc  # clickable STRI cell
    assert '"stri|leaf_margin|roni"' in doc                   # island key
    assert "renderStri" in doc                                 # modal branch
    assert "Ruellia fulgida" in doc                            # per-species detail


def test_species_map_island_and_agreement_species_column(tmp_path):
    b = _bundle()  # id record specimen SR1 -> Aphelandra scabra
    R.generate_html(tmp_path / "r.html", b)
    doc = (tmp_path / "r.html").read_text()
    assert 'id="species-map"' in doc
    assert '"SR1": "Aphelandra scabra"' in doc
    # agreement modal renders a species column via spName()
    assert "spName" in doc
    assert "renderAgreement" in doc


# --------------------------------------------------------------------------- #
# End-to-end on real data + the prior benchmark run.
# --------------------------------------------------------------------------- #

_have = RONI.exists() and CURATOR.exists() and PRIOR_RUN.exists()


@pytest.mark.skipif(not _have, reason="study data / prior run absent")
def test_run_report_end_to_end(tmp_path):
    bundle = R.run_report(
        results_dir=PRIOR_RUN,
        roni_xlsx=RONI,
        carmen_xlsx=CARMEN if CARMEN.exists() else None,
        curator_key=CURATOR,
        out_dir=tmp_path,
        html=True,
        stri_matrix=STRI_MATRIX if STRI_MATRIX.exists() else None,
        embed_images=False,  # image embedding covered by test_human_thumbnails
    )
    assert bundle.n_annotated_individuals == 114
    expected = [
        "trait_agreement_per_trait.csv",
        "trait_agreement_overall.csv",
        "human_trait_distributions.csv",
        "roni_id_accuracy.csv",
        "roni_id_summary.json",
        "human_grading_report.html",
    ]
    if STRI_MATRIX.exists():
        assert bundle.n_stri_matched > 0
        assert bundle.stri_results  # STRI axis computed
        expected.append("stri_accuracy.csv")
    for name in expected:
        assert (tmp_path / name).exists(), name
        assert (tmp_path / name).stat().st_size > 0
    # roni_vs_carmen ceiling should generally exceed model_vs_roni overall
    roni_carmen = bundle.overall["roni_vs_carmen"]["macro_agreement_rate"]
    model_roni = bundle.overall["model_vs_roni"]["macro_agreement_rate"]
    assert roni_carmen >= model_roni


@pytest.mark.skipif(not _have, reason="study data / prior run absent")
def test_run_report_no_html(tmp_path):
    R.run_report(PRIOR_RUN, RONI, CARMEN if CARMEN.exists() else None, CURATOR, tmp_path,
                 html=False, embed_images=False)
    assert not (tmp_path / "human_grading_report.html").exists()
    assert (tmp_path / "trait_agreement_per_trait.csv").exists()
