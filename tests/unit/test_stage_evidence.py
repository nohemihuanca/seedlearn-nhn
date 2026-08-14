"""Tests for Stage 4: Deterministic Evidence Synthesis."""

import pytest

from seedlearn.pipeline.config import EvidenceSynthesisConfig
from seedlearn.pipeline.stages.evidence import (
    EvidenceSynthesisStage,
    _compute_quality_flags,
    _format_classification_section,
    _format_convergence_section,
    _format_literature_section,
    _format_morphology_section,
)


class TestFormatSections:
    def test_morphology_with_traits(self) -> None:
        data = {
            "traits": {
                "leaf_complexity": {"type": "compound"},
                "special_features": {"stipules": "present"},
            }
        }
        section = _format_morphology_section(data)
        assert "Morphological Profile" in section
        assert "compound" in section
        assert "present" in section

    def test_morphology_empty(self) -> None:
        section = _format_morphology_section({})
        assert "No morphological data" in section

    def test_classification_with_predictions(self) -> None:
        data = {
            "predictions": [
                {"rank_value": "Moraceae", "softmax_score": 0.65, "rank_position": 1},
                {"rank_value": "Sapotaceae", "softmax_score": 0.20, "rank_position": 2},
            ]
        }
        section = _format_classification_section(data)
        assert "Visual Classification" in section
        assert "Moraceae" in section
        assert "65.0%" in section

    def test_classification_empty(self) -> None:
        section = _format_classification_section({})
        assert "No classification data" in section

    def test_literature_with_matches(self) -> None:
        data = {
            "rag_matches": [
                {
                    "taxon": "Moraceae",
                    "rank": "family",
                    "score": 0.88,
                    "description": "latex trees",
                },
            ]
        }
        section = _format_literature_section(data)
        assert "Literature Evidence" in section
        assert "Moraceae" in section

    def test_literature_none(self) -> None:
        section = _format_literature_section(None)
        assert "not available" in section

    def test_convergence_strong(self) -> None:
        data = {
            "convergence": [
                {
                    "taxon": "Moraceae",
                    "signal": "strong",
                    "rag_score": 0.88,
                    "visual_softmax_score": 0.65,
                    "source": "both",
                },
            ]
        }
        section = _format_convergence_section(data)
        assert "Convergence" in section
        assert "STRONG" in section
        assert "Moraceae" in section

    def test_convergence_none(self) -> None:
        section = _format_convergence_section(None)
        assert "Insufficient" in section


class TestEnrichedClassification:
    """Tests for enriched classification output formatting."""

    def test_classification_with_distances(self) -> None:
        data = {
            "predictions": [
                {
                    "rank_value": "Moraceae",
                    "softmax_score": 0.65,
                    "rank_position": 1,
                    "l2_distance": 0.43,
                    "cosine_similarity": 0.91,
                },
            ]
        }
        section = _format_classification_section(data)
        assert "L2: 0.430" in section
        assert "cosine: 0.910" in section

    def test_classification_with_margin(self) -> None:
        data = {
            "predictions": [
                {"rank_value": "A", "softmax_score": 0.7, "rank_position": 1},
            ],
            "margin": 0.52,
        }
        section = _format_classification_section(data)
        assert "Decision margin: 0.520" in section

    def test_classification_backward_compatible(self) -> None:
        """Old-format data (no distances, no margin) should still render."""
        data = {
            "predictions": [
                {"rank_value": "Moraceae", "softmax_score": 0.65, "rank_position": 1},
            ]
        }
        section = _format_classification_section(data)
        assert "Moraceae" in section
        assert "65.0%" in section
        assert "L2:" not in section
        assert "margin" not in section.lower()

    def test_per_image_formatted(self) -> None:
        data = {
            "predictions": [
                {"rank_value": "A", "softmax_score": 0.7, "rank_position": 1},
            ],
            "per_image_predictions": [
                {"image_path": "/img1.jpg", "top1_label": "A", "top1_softmax_score": 0.68},
                {"image_path": "/img2.jpg", "top1_label": "B", "top1_softmax_score": 0.55},
            ],
        }
        section = _format_classification_section(data)
        assert "Per-Image Predictions" in section
        assert "/img1.jpg" in section
        assert "/img2.jpg" in section

    def test_nearest_support_formatted(self) -> None:
        data = {
            "predictions": [
                {"rank_value": "A", "softmax_score": 0.7, "rank_position": 1},
            ],
            "nearest_support": [
                {
                    "label": "A",
                    "l2_distance": 0.31,
                    "cosine_similarity": 0.95,
                    "image_path": "/support/img_0.jpg",
                },
                {
                    "label": "A",
                    "l2_distance": 0.35,
                    "cosine_similarity": 0.93,
                },
            ],
        }
        section = _format_classification_section(data)
        assert "Nearest Support Images" in section
        assert "/support/img_0.jpg" in section


class TestQualityFlags:
    def test_many_unclear_traits(self) -> None:
        morph = {
            "traits": {
                "leaf_arrangement": {"relative_position": "unclear", "spacing": "unclear"},
                "leaf_complexity": {"type": "unclear"},
                "special_features": {"stipules": "unclear", "latex": "unclear"},
            }
        }
        flags = _compute_quality_flags(morph, {}, EvidenceSynthesisConfig())
        assert any("unclear" in f.lower() for f in flags)

    def test_low_similarity(self) -> None:
        clf = {"predictions": [{"rank_value": "X", "softmax_score": 0.15}]}
        flags = _compute_quality_flags({}, clf, EvidenceSynthesisConfig())
        assert any("similarity" in f.lower() for f in flags)

    def test_no_flags_when_good(self) -> None:
        morph = {"traits": {"leaf_complexity": {"type": "compound"}}}
        clf = {"predictions": [{"rank_value": "Fabaceae", "softmax_score": 0.8}]}
        flags = _compute_quality_flags(morph, clf, EvidenceSynthesisConfig())
        assert flags == []

    def test_quality_flag_low_margin(self) -> None:
        clf = {
            "predictions": [{"rank_value": "A", "softmax_score": 0.8}],
            "margin": 0.05,
        }
        flags = _compute_quality_flags({}, clf, EvidenceSynthesisConfig())
        assert any("margin" in f.lower() for f in flags)

    def test_quality_flag_no_margin_flag_when_high(self) -> None:
        clf = {
            "predictions": [{"rank_value": "A", "softmax_score": 0.8}],
            "margin": 0.5,
        }
        flags = _compute_quality_flags({}, clf, EvidenceSynthesisConfig())
        assert not any("margin" in f.lower() for f in flags)

    def test_quality_flag_per_image_disagreement(self) -> None:
        clf = {
            "predictions": [{"rank_value": "A", "softmax_score": 0.6}],
            "per_image_predictions": [
                {"image_path": "/a.jpg", "top1_label": "A", "top1_softmax_score": 0.6},
                {"image_path": "/b.jpg", "top1_label": "B", "top1_softmax_score": 0.5},
            ],
        }
        flags = _compute_quality_flags({}, clf, EvidenceSynthesisConfig())
        assert any("disagreement" in f.lower() for f in flags)

    def test_quality_flag_no_disagreement_when_unanimous(self) -> None:
        clf = {
            "predictions": [{"rank_value": "A", "softmax_score": 0.8}],
            "per_image_predictions": [
                {"image_path": "/a.jpg", "top1_label": "A", "top1_softmax_score": 0.8},
                {"image_path": "/b.jpg", "top1_label": "A", "top1_softmax_score": 0.7},
            ],
        }
        flags = _compute_quality_flags({}, clf, EvidenceSynthesisConfig())
        assert not any("disagreement" in f.lower() for f in flags)


class TestEvidenceSynthesisStage:
    def test_name(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        assert stage.name == "evidence_synthesis"

    def test_validate_requires_at_least_one_input(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        errors = stage.validate_input({})
        assert len(errors) > 0

    def test_validate_ok_with_morphology(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        errors = stage.validate_input({"morphology": {"traits": {}}})
        assert errors == []

    def test_validate_ok_with_classification(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        errors = stage.validate_input({"classification": {"predictions": []}})
        assert errors == []

    def test_skip(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        result = stage.skip({})
        assert result.skipped is True

    def test_run_with_all_stages(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        context = {
            "morphology": {
                "traits": {
                    "leaf_arrangement": {"relative_position": "alternate"},
                    "leaf_complexity": {"type": "simple"},
                    "leaf_morphology": {"margin": "entire"},
                    "special_features": {"latex": "present"},
                },
            },
            "classification": {
                "predictions": [
                    {"rank_value": "Moraceae", "softmax_score": 0.65, "rank_position": 1},
                    {"rank_value": "Sapotaceae", "softmax_score": 0.20, "rank_position": 2},
                ],
            },
            "trait_retrieval": {
                "rag_matches": [
                    {
                        "taxon": "Moraceae",
                        "rank": "family",
                        "score": 0.88,
                        "description": "latex trees",
                    },
                ],
                "convergence": [
                    {
                        "taxon": "Moraceae",
                        "signal": "strong",
                        "rag_score": 0.88,
                        "visual_softmax_score": 0.65,
                        "source": "both",
                    },
                ],
            },
        }
        result = stage.run(context)
        assert "evidence_document" in result.data
        doc = result.data["evidence_document"]
        assert isinstance(doc, str)
        assert "Morphological Profile" in doc
        assert "Visual Classification" in doc
        assert "Literature Evidence" in doc
        assert "Convergence" in doc
        assert "alternate" in doc
        assert "Moraceae" in doc
        assert result.error is None

    def test_run_with_skipped_trait_retrieval(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        context = {
            "morphology": {"traits": {"leaf_complexity": {"type": "compound"}}},
            "classification": {
                "predictions": [
                    {"rank_value": "Fabaceae", "softmax_score": 0.9, "rank_position": 1}
                ]
            },
        }
        result = stage.run(context)
        doc = result.data["evidence_document"]
        assert "Morphological Profile" in doc
        assert "Visual Classification" in doc
        assert "Literature Evidence" in doc

    def test_evidence_is_deterministic(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        context = {
            "morphology": {"traits": {"leaf_morphology": {"shape": "elliptic"}}},
            "classification": {
                "predictions": [
                    {"rank_value": "Fabaceae", "softmax_score": 0.5, "rank_position": 1}
                ]
            },
        }
        doc1 = stage.run(context).data["evidence_document"]
        doc2 = stage.run(context).data["evidence_document"]
        assert doc1 == doc2

    def test_quality_flags(self) -> None:
        stage = EvidenceSynthesisStage(config=EvidenceSynthesisConfig())
        context = {
            "morphology": {
                "traits": {
                    "leaf_arrangement": {"relative_position": "unclear", "spacing": "unclear"},
                    "leaf_complexity": {"type": "unclear"},
                    "special_features": {"stipules": "unclear", "latex": "unclear"},
                },
            },
            "classification": {
                "predictions": [
                    {"rank_value": "Unknown", "softmax_score": 0.15, "rank_position": 1}
                ],
            },
        }
        result = stage.run(context)
        assert "quality_flags" in result.data
        flags = result.data["quality_flags"]
        assert len(flags) > 0


class TestMultiRankEvidenceFormatting:
    """Tests for multi-rank classification evidence formatting."""

    def test_format_multirank_classification(self) -> None:
        clf_data = {
            "predictions_by_rank": {
                "family": [{"rank_value": "Fabaceae", "softmax_score": 0.42, "rank_position": 1,
                            "l2_distance": 0.3, "cosine_similarity": 0.85}],
                "genus": [{"rank_value": "Inga", "softmax_score": 0.31, "rank_position": 1,
                           "l2_distance": 0.45, "cosine_similarity": 0.77}],
            },
            "margin_by_rank": {"family": 0.24, "genus": 0.12},
        }
        html = _format_classification_section(clf_data)
        assert "Fabaceae" in html
        assert "Inga" in html
        assert "Family" in html or "family" in html

    def test_quality_flags_hierarchical_inconsistency(self) -> None:
        clf_data = {
            "hierarchical_consistency": {"consistent": False, "notes": ["genus disagrees with family"]},
        }
        flags = _compute_quality_flags({}, clf_data, EvidenceSynthesisConfig())
        assert any("hierarchical" in f.lower() or "inconsisten" in f.lower() for f in flags)

    def test_quality_flags_ood_detection(self) -> None:
        clf_data = {
            "confidence_gate": {
                "flags": ["species_distance_exceeds_threshold"],
                "family_in_distribution": True,
                "species_in_distribution": False,
            },
        }
        flags = _compute_quality_flags({}, clf_data, EvidenceSynthesisConfig())
        assert any("distribution" in f.lower() or "distance" in f.lower() for f in flags)
