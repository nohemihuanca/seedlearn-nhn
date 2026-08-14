"""Tests for RAG index building and search."""

import pytest

from seedlearn.pipeline.rag import RAGIndex


@pytest.fixture
def sample_descriptions():
    """Minimal botanical description set."""
    return [
        {
            "taxon": "Fabaceae",
            "rank": "family",
            "description": (
                "Trees with compound leaves, stipules present, "
                "pulvinus at leaf base."
            ),
        },
        {
            "taxon": "Inga",
            "rank": "genus",
            "description": (
                "Genus of legumes with opposite pinnately compound "
                "leaves and winged rachis."
            ),
        },
        {
            "taxon": "Meliaceae",
            "rank": "family",
            "description": (
                "Trees with pinnately compound leaves, no stipules, "
                "bark often bitter."
            ),
        },
        {
            "taxon": "Moraceae",
            "rank": "family",
            "description": (
                "Trees and shrubs with simple alternate leaves, "
                "latex present, stipules present."
            ),
        },
    ]


class TestRAGIndex:
    """Test suite for RAGIndex build, search, save, and load."""

    def test_build_from_descriptions(self, sample_descriptions):
        """Index contains all provided descriptions."""
        index = RAGIndex.build(sample_descriptions, model_name="all-MiniLM-L6-v2")
        assert index.size == 4

    def test_search_returns_ranked(self, sample_descriptions):
        """Search results are ranked by descending similarity."""
        index = RAGIndex.build(sample_descriptions, model_name="all-MiniLM-L6-v2")
        results = index.search("compound leaves with stipules and pulvinus", top_k=3)
        assert len(results) <= 3
        assert all("taxon" in r for r in results)
        assert all("score" in r for r in results)
        assert all("description" in r for r in results)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(self, sample_descriptions, tmp_path):
        """Round-trip save/load preserves index contents."""
        index = RAGIndex.build(sample_descriptions, model_name="all-MiniLM-L6-v2")
        save_dir = tmp_path / "rag_index"
        index.save(save_dir)

        loaded = RAGIndex.load(save_dir)
        assert loaded.size == 4

        results = loaded.search("compound leaves", top_k=2)
        assert len(results) == 2

    def test_search_with_min_similarity(self, sample_descriptions):
        """High min_similarity filters out low-relevance results."""
        index = RAGIndex.build(sample_descriptions, model_name="all-MiniLM-L6-v2")
        results = index.search(
            "completely unrelated electronics topic",
            top_k=4,
            min_similarity=0.9,
        )
        assert len(results) < 4

    def test_load_missing_raises(self, tmp_path):
        """Loading from a nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RAGIndex.load(tmp_path / "nonexistent")

    def test_search_respects_top_k(self, sample_descriptions):
        """Search never returns more than top_k results."""
        index = RAGIndex.build(sample_descriptions, model_name="all-MiniLM-L6-v2")
        results = index.search("leaves", top_k=2)
        assert len(results) <= 2
