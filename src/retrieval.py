import json
import pickle
from pathlib import Path

from tqdm import tqdm

from .indexer import tokenize
from .models import (
    FullSource,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Retriever:
    def __init__(self):
        with open("data/processed/bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open("data/processed/chunks.json") as f:
            self.chunks = json.load(f)

        try:
            with open("data/processed/query_cache.json", "r") as f:
                self.query_cache = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError):
            processed = Path("data/processed")
            processed.mkdir(parents=True, exist_ok=True)

            self.query_cache = {}

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:

        cache_key = f"{query.strip().lower()}_{k}"

        if cache_key in self.query_cache:
            return [FullSource(**item) for item in self.query_cache[cache_key]]
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            results.append(
                FullSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"],
                    metadata=chunk["metadata"]
                )
            )

        self.query_cache[cache_key] = [r.model_dump() for r in results]
        with open("data/processed/query_cache.json", "w") as f:
            json.dump(self.query_cache, f, indent=2)

        print(" Query cache Done! data/processed/query_cache.json")
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
