import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .embedding import Embedding
from .indexer import tokenize
from .models import (
    FullSource,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    config,
)


class Retriever:
    def __init__(self):
        self.processed = Path(config.processed_dir)
        self.embedder = Embedding().encode_query

        with open(self.processed / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(self.processed / "chunks.json") as f:
            self.chunks = json.load(f)

        with open(self.processed / "query_cache.json", "r") as f:
            self.query_cache = json.load(f)

        self.semantic = np.load(self.processed / "embeddings.npy")

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:

        cache_key = f"{query.strip().lower()}"
        if cache_key in self.query_cache\
            and self.query_cache[cache_key][0]["k"] >= k:
                return [FullSource(**item) for item in self.query_cache[cache_key][1:k + 1]]

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = scores.argsort()[::-1][:k]

        query_embedding = self.embedder(query=query)
        semantic_score = self.semantic @ query_embedding
        top_k_semantic = semantic_score.argsort()[::-1][k]
        

        rrf_scores = {}

        for rank, i in enumerate(top_k_indices):
            rrf_scores[i] =  rrf_scores.get(i, 0.0) + (1.0 / k + rank)
        for rank, i in enumerate(top_k_semantic):
            rrf_scores[i] =  rrf_scores.get(i, 0.0) + (1.0 / k + rank)
        exit()

        top_k_final = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:k]

        results = []
        for idx in top_k_final:
            chunk = self.chunks[idx]
            results.append(
                FullSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"],
                    metadata=chunk["metadata"],
                )
            )
        print(results)
        exit()

        self.query_cache[cache_key] = [r.model_dump() for r in results]
        self.query_cache[cache_key].insert(0, {"k": k})
        with open(self.processed / "query_cache.json", "w") as f:
            json.dump(self.query_cache, f, indent=2)

        print(" Query cache Done! processed/query_cache.json", flush=True)
        return results

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:

        with open(dataset_path) as f:
            dataset = RagDataset.model_validate_json(f.read())

        all_results = []
        for q in tqdm(dataset.rag_questions, desc="Searching"):
            sources = self.search(q.question, k=k)

            all_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(
            search_results=all_results,
            k=k,
        )

        import os

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(dataset_path)
        output_path = os.path.join(save_directory, filename)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Search dataset DONE! {output_path}")
