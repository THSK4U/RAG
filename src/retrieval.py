import json
import pickle

from .indexer import tokenize
from .models import MinimalSource


class Retriever:
    def __init__(self):
        with open("data/processed/bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open("data/processed/chunks.json") as f:
            self.chunks = json.load(f)

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            results.append(
                MinimalSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"],
                )
            )

        return results
