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

    def Semantic_embidding(
        self,
        all_texts: list[str],
        new_texts: list[str],
        keep_indices: list[int] | None = None,
    ) -> None:
        """Build and save normalized dense embeddings for semantic search.

        Uses incremental mode when keep_indices is provided and an existing
        embeddings file is present: keeps old vectors for unchanged chunks,
        encodes only the new ones, then concatenates.

        Args:
            all_texts: Full list of raw text chunks (used for full rebuild).
            new_texts: Only the newly added/modified chunks to encode.
            keep_indices: Indices into the old embeddings array to preserve.
        """
        embeddings_path = self.processed / "embeddings.npy"
        if keep_indices is not None and embeddings_path.exists():
            old_embeddings = np.load(embeddings_path)
            if keep_indices and max(keep_indices) >= len(old_embeddings):
                final = self.model.encode(
                    all_texts,
                    batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
            else:
                kept = old_embeddings[keep_indices]
                if new_texts:
                    new_embs = self.model.encode(
                        new_texts,
                        batch_size=64,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                    )
                    final = np.vstack([kept, new_embs])
                else:
                    final = kept
        else:
            final = self.model.encode(
                all_texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        np.save(self.processed / "embeddings", final)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a normalized vector.

        Args:
            query: The search query string.

        Returns:
            np.ndarray: Normalized dense embedding vector.
        """
        return self.model.encode(query, normalize_embeddings=True)
