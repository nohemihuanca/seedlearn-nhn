"""Tests for Stage 3: Literature-Based Trait Retrieval."""

import pytest
from unittest.mock import MagicMock

from seedlearn.pipeline.stages.trait_retrieval import (
    TraitRetrievalStage,
    _compose_query,
    _cross_reference,
)
from seedlearn.pipeline.config import TraitRetrievalConfig


class TestComposeQuery:
    def test_basic_traits(self):
        traits = {"leaf complexity": "compound", "stipules": "present"}
        query = _compose_query(traits)
        assert "compound" in query
        assert "stipules" in query

    def test_skips_unclear(self):
        traits = {"leaf complexity": "unclear", "stipules": "present"}
        query = _compose_query(traits)
        assert "unclear" not in query
        assert "stipules" in query

    def test_empty_traits(self):
        query = _compose_query({})
        assert "tropical" in query.lower()

    def test_skips_na(self):
        traits = {"leaf complexity": "N/A", "margin": "entire"}
        query = _compose_query(traits)
        assert "N/A" not in query
        assert "entire" in query


class TestCrossReference:
    def test_convergence_detected(self):
        rag = [{"taxon": "Fabaceae", "score": 0.85, "description": "..."}]
        preds = [{"rank_value": "Fabaceae", "softmax_score": 0.8}]
        result = _cross_reference(rag, preds)
        shared = [c for c in result if c["source"] == "both"]
        assert len(shared) == 1
        assert shared[0]["taxon"] == "Fabaceae"
        assert shared[0]["signal"] == "strong"

    def test_divergence_detected(self):
        rag = [{"taxon": "Fabaceae", "score": 0.85, "description": "..."}]
        preds = [{"rank_value": "Meliaceae", "softmax_score": 0.8}]
        result = _cross_reference(rag, preds)
        rag_only = [c for c in result if c["source"] == "literature"]
        vis_only = [c for c in result if c["source"] == "visual"]
        assert len(rag_only) >= 1
        assert len(vis_only) >= 1

    def test_case_insensitive(self):
        rag = [{"taxon": "fabaceae", "score": 0.8, "description": "..."}]
        preds = [{"rank_value": "Fabaceae", "softmax_score": 0.7}]
        result = _cross_reference(rag, preds)
        shared = [c for c in result if c["source"] == "both"]
        assert len(shared) == 1


class TestTraitRetrievalStage:
    def test_name(self):
        stage = TraitRetrievalStage(config=TraitRetrievalConfig())
        assert stage.name == "trait_retrieval"

    def test_validate_requires_morphology(self):
        stage = TraitRetrievalStage(config=TraitRetrievalConfig())
        errors = stage.validate_input({})
        assert any("morphology" in e for e in errors)

    def test_validate_ok_with_morphology(self):
        stage = TraitRetrievalStage(config=TraitRetrievalConfig())
        errors = stage.validate_input(
            {"morphology": {"traits": {"leaf_shape": "elliptic"}}}
        )
        assert errors == []

    def test_skip_returns_empty(self):
        stage = TraitRetrievalStage(config=TraitRetrievalConfig())
        result = stage.skip({})
        assert result.skipped is True

    def test_run_without_index_returns_error(self):
        stage = TraitRetrievalStage(config=TraitRetrievalConfig())
        result = stage.run({"morphology": {"traits": {}}})
        assert result.error is not None
        assert "index" in result.error.lower()

    def test_run_with_mock_rag(self):
        mock_index = MagicMock()
        mock_index.search.return_value = [
            {
                "taxon": "Fabaceae",
                "rank": "family",
                "score": 0.87,
                "description": "compound leaves, stipules present",
            },
            {
                "taxon": "Inga",
                "rank": "genus",
                "score": 0.84,
                "description": "opposite pinnately compound",
            },
        ]

        config = TraitRetrievalConfig(top_k=5, cross_reference=True)
        stage = TraitRetrievalStage(config=config)
        stage.rag_index = mock_index

        context = {
            "morphology": {
                "traits": {"leaf complexity": "compound", "stipules": "present"},
            },
            "classification": {
                "predictions": [
                    {"rank_value": "Fabaceae", "softmax_score": 0.8},
                    {"rank_value": "Meliaceae", "softmax_score": 0.15},
                ],
            },
        }
        result = stage.run(context)
        assert result.error is None
        assert "rag_matches" in result.data
        assert "convergence" in result.data
        assert "query" in result.data
        assert len(result.data["rag_matches"]) == 2

    def test_cross_reference_convergence(self):
        mock_index = MagicMock()
        mock_index.search.return_value = [
            {"taxon": "Fabaceae", "rank": "family", "score": 0.9, "description": "..."},
        ]

        config = TraitRetrievalConfig(cross_reference=True)
        stage = TraitRetrievalStage(config=config)
        stage.rag_index = mock_index

        context = {
            "morphology": {"traits": {}},
            "classification": {
                "predictions": [{"rank_value": "Fabaceae", "softmax_score": 0.8}],
            },
        }
        result = stage.run(context)
        conv = result.data["convergence"]
        assert any(
            c["taxon"] == "Fabaceae" and c["signal"] == "strong" for c in conv
        )

    def test_run_without_classification(self):
        """Stage works even if Stage 2 was skipped."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            {"taxon": "Fabaceae", "rank": "family", "score": 0.8, "description": "..."},
        ]

        config = TraitRetrievalConfig(cross_reference=True)
        stage = TraitRetrievalStage(config=config)
        stage.rag_index = mock_index

        context = {"morphology": {"traits": {"leaf complexity": "compound"}}}
        result = stage.run(context)
        assert result.error is None
        assert result.data["convergence"] == []
