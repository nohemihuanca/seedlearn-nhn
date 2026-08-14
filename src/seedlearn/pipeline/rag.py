"""RAG index for botanical trait descriptions.

Build a FAISS index from NLP trait descriptions, then search by semantic
similarity to find taxa whose literature traits match observed morphology.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGIndex:
    """FAISS-backed semantic search over botanical trait descriptions.

    Use ``build()`` to create from descriptions, or ``load()`` to restore
    a previously saved index.

    Args:
        index: FAISS inner-product index.
        metadata: List of dicts, one per indexed description. Each dict
            must contain at least ``taxon``, ``rank``, and ``description``.
        model_name: Sentence-transformer model used for embedding.
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        metadata: list[dict[str, Any]],
        model_name: str,
    ) -> None:
        self._index = index
        self._metadata = metadata
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def size(self) -> int:
        """Number of indexed descriptions."""
        return self._index.ntotal

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @classmethod
    def build(
        cls,
        descriptions: list[dict[str, str]],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> RAGIndex:
        """Build a FAISS index from botanical trait descriptions.

        Args:
            descriptions: List of dicts with keys ``taxon``, ``rank``,
                and ``description``.
            model_name: Sentence-transformers model for embedding.

        Returns:
            Populated RAGIndex ready for search.
        """
        model = SentenceTransformer(model_name)
        texts = [d["description"] for d in descriptions]
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info(
            "Built RAG index: %d descriptions, %d-dim", len(descriptions), dim,
        )

        instance = cls(index=index, metadata=descriptions, model_name=model_name)
        instance._model = model
        return instance

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search the index for descriptions matching a query.

        Args:
            query: Natural language query (e.g., trait description).
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            Ranked list of dicts with ``taxon``, ``rank``, ``description``,
            and ``score`` keys.
        """
        model = self._get_model()
        query_vec = model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self._index.search(query_vec, min(top_k, self.size))

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_similarity:
                continue
            entry = dict(self._metadata[int(idx)])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save(self, directory: Path | str) -> None:
        """Save index and metadata to disk.

        Args:
            directory: Directory to save into (created if needed).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w") as f:
            json.dump(
                {
                    "model_name": self._model_name,
                    "entries": self._metadata,
                },
                f,
                indent=2,
            )

        logger.info("Saved RAG index to %s (%d entries)", directory, self.size)

    @classmethod
    def load(cls, directory: Path | str) -> RAGIndex:
        """Load a previously saved index.

        Args:
            directory: Directory containing index.faiss and metadata.json.

        Returns:
            Restored RAGIndex.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        directory = Path(directory)
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            data = json.load(f)

        logger.info(
            "Loaded RAG index from %s (%d entries)", directory, index.ntotal,
        )
        return cls(
            index=index,
            metadata=data["entries"],
            model_name=data["model_name"],
        )
