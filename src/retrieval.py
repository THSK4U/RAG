from .models import MinimalSource
import pickle
import json
from .indexer import tokenize


def search(query: str, k: int = 5) -> list[MinimalSource]:
    with open("data/processed/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open("data/processed/chunks.json") as f:
        chunks = json.load(f)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_k_indices = scores.argsort()[::-1][:k]

    results = []
    for idx in top_k_indices:
        chunk = chunks[idx]
        results.append(
            MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            ))

    return results
