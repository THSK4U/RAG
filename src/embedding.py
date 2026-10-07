import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.models import StudentSearchResults

from .models import config


class Embedding:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        self.processed = Path(config.processed_dir)

    def lexical(self, corpus):
        """Build and save BM25 index."""
        bm25 = BM25Plus(corpus)

        with open(self.processed / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

    def Semantic_embidding(self, cropus):
        """Build and save normalized dense embeddings for semantic search."""
        embeddings = self.model.encode(
            cropus, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        np.save(self.processed / "embeddings.npy", embeddings)
