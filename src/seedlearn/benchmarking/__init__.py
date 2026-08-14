"""Benchmarking modules for evaluating pipeline trait extraction and species ID."""

from seedlearn.benchmarking.id_grader import IDGradeRecord, grade_specimen_id
from seedlearn.benchmarking.overlap import OverlapSpecimen, load_overlap_specimens
from seedlearn.benchmarking.trait_grader import (
    TraitGradeRecord,
    TraitVerdict,
    grade_specimen_traits,
)
from seedlearn.benchmarking.trait_mapping import (
    TRAIT_RULES,
    TraitRule,
    get_raw_vlm_values,
    map_prediction,
)

__all__ = [
    "IDGradeRecord",
    "OverlapSpecimen",
    "TRAIT_RULES",
    "TraitGradeRecord",
    "TraitRule",
    "TraitVerdict",
    "get_raw_vlm_values",
    "grade_specimen_id",
    "grade_specimen_traits",
    "load_overlap_specimens",
    "map_prediction",
]
