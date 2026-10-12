# from .models import config

from pathlib import Path
from typing import Any

import fire

from .evaluator import evaluate as evaluate_fn
from .indexer import load_all_files
from .models import config


class CLI:
    def __init__(self) -> None:
        self._generator: Any = None
        self._retriever: Any = None

    @property
    def generator(self) -> Any:
        if self._generator is None:
            from .generator import Generator

            self._generator = Generator()
        return self._generator

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from .retrieval import Retriever

            self._retriever = Retriever()
        return self._retriever

    # index –max_chunk_size <int>
    # Ingest data/raw/ and build the index under data/processed/.
    def index(self, max_chunk_size: int | None = None) -> None:
        if max_chunk_size is not None:
            if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
                print(
                    f"max_chunk_size must be a positive integer, \
                        got {max_chunk_size}"
                )
                return
            config.max_chunk_size = max_chunk_size
        processed = Path(config.processed_dir)
        processed.mkdir(parents=True, exist_ok=True)
        load_all_files()

    # • search <query> –k <int>
    # Return the top-k sources for a single query.
    def search(self, query: str, k: int) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if not query or not str(query).strip():
            raise ValueError("Query is empty. Please provide a valid query.")
        result = self.retriever.search(query=query, k=k)
        for r in result:
            print(r.model_dump_json(indent=2))

    # • search_dataset –dataset_path <path> –k <int> –save_directory <dir>
    # Run search over a whole dataset and write a StudentSearchResults JSON.
    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int
    ) -> None:
        if not Path(dataset_path).is_file() and not dataset_path.endswith(
            ".json"
        ):
            raise FileNotFoundError(
                f"Dataset file not found or not valid '{dataset_path}'"
            )
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self.retriever.search_dataset(dataset_path, k, save_directory)

    # • answer <query> –k <int>
    # Answer a single query using the retrieved context.
    def answer(self, query: str, k: int) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if not query or not str(query).strip():
            raise ValueError("Query is empty. Please provide a valid query.")
        result = self.retriever.search(query=query, k=k)
        context = self.generator.bulid_context(result)
        ans = self.generator.generate_answer(context, query)
        print(ans)

    # answer_dataset –student_search_results_path <path> –save_directory <dir>
    # Generate answers for a dataset,producing a StudentSearchResultsAndAnswer.
    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        if not Path(
            student_search_results_path
        ).is_file() or not student_search_results_path.endswith(".json"):
            raise FileNotFoundError(
                f"Dataset file not found at '{student_search_results_path}'"
            )

        self.generator.answer_dataset(
            student_search_results_path, save_directory
        )

    # • evaluate –student_search_results_path <path> –dataset_path <path>
    # Report your own recall@k against a ground-truth dataset.
    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        if not Path(
            student_search_results_path
        ).is_file() or student_search_results_path.endswith(".json"):
            raise FileNotFoundError(
                f"Dataset file not found at '{student_search_results_path}'"
            )
        if not Path(dataset_path).is_file() or dataset_path.endswith(".json"):
            raise FileNotFoundError(
                f"Dataset file not found at '{dataset_path}'"
            )
        evaluate_fn(
            student_search_results_path,
            dataset_path,
        )


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
