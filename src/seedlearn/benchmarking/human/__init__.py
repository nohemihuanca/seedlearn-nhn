"""Human-annotation grading for Vision-LLM morphological traits.

This subpackage grades the pipeline's Stage 1 (Vision-LLM morphology) trait
predictions against independent human annotators, and grades a botanist's
photo-based species identifications against the true taxonomy.

Modules
-------
value_map
    Bilingual (English model / Spanish human) trait + value mapping to shared
    canonical tokens, and per-trait gradability classification.
annotations
    Load the annotators' spreadsheets and join blinded IDs to real specimens
    and true taxonomy via the curator/image keys.
aggregate
    Collapse per-view human annotations to a per-specimen modal value while
    retaining the full per-view distribution.
categorical_grader
    Per-trait agreement (rate + Cohen's kappa) for model-vs-human and
    human-vs-human comparisons.
report
    CSV + HTML reporting over the grading results.
"""

from seedlearn.benchmarking.human.aggregate import (
    SpecimenAggregate,
    SpecimenTraitAgg,
    aggregate_records,
    modal_value,
)
from seedlearn.benchmarking.human.categorical_grader import (
    TraitAgreement,
    grade_all_axes,
    load_model_traits,
    overall_by_axis,
)
from seedlearn.benchmarking.human.id_grading import (
    HumanIDRecord,
    grade_human_ids,
    id_accuracy,
)
from seedlearn.benchmarking.human.report import ReportBundle, assemble, run_report
from seedlearn.benchmarking.human.stri_compare import (
    STRI_TRAITS,
    STRIAgreement,
    accuracy_vs_stri,
    build_stri_lookup,
    load_stri_matrix,
)
from seedlearn.benchmarking.human.annotations import (
    AnnotationRecord,
    CuratorEntry,
    join_specimens,
    load_annotations,
    load_curator_key,
    parse_annotation_rows,
    parse_curator_rows,
)
from seedlearn.benchmarking.human.value_map import (
    MISSING,
    TRAIT_SPECS,
    TraitSpec,
    gradable_specs,
    model_value,
    normalize,
    spec_for_spanish_header,
    to_canonical,
)

__all__ = [
    "MISSING",
    "TRAIT_SPECS",
    "AnnotationRecord",
    "CuratorEntry",
    "HumanIDRecord",
    "ReportBundle",
    "STRI_TRAITS",
    "STRIAgreement",
    "SpecimenAggregate",
    "accuracy_vs_stri",
    "build_stri_lookup",
    "load_stri_matrix",
    "SpecimenTraitAgg",
    "TraitAgreement",
    "TraitSpec",
    "aggregate_records",
    "assemble",
    "grade_all_axes",
    "grade_human_ids",
    "id_accuracy",
    "load_model_traits",
    "overall_by_axis",
    "run_report",
    "gradable_specs",
    "join_specimens",
    "load_annotations",
    "load_curator_key",
    "modal_value",
    "model_value",
    "normalize",
    "parse_annotation_rows",
    "parse_curator_rows",
    "spec_for_spanish_header",
    "to_canonical",
]
