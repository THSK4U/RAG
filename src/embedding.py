import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

from .models import config


class Embedding:
    """Embedding and indexing helper for lexical and semantic retrieval."""

    def __init__(self) -> None:
        """Initialize sentence transformer model and processed directory."""
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        self.processed = Path(config.processed_dir)

    def lexical(self, corpus: list[list[str]]) -> None:
        """Build and save BM25 lexical search index.

        Args:
            corpus: List of tokenized documents.
        """
        bm25 = BM25Plus(corpus)

        with open(self.processed / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

    def Semantic_embidding(self, cropus: list[str]) -> None:
        """Build and save normalized dense embeddings for semantic search.

        Args:
            cropus: List of raw text chunks to embed.
        """
        embeddings = self.model.encode(
            cropus,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(self.processed / "embeddings", embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a normalized vector.

        Args:
            query: The search query string.

        Returns:
            np.ndarray: Normalized dense embedding vector.
        """
        return self.model.encode(query, normalize_embeddings=True)
