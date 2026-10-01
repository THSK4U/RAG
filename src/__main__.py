# from .models import config

import json
from pathlib import Path

from .evaluator import evaluate
from .generator import Generator
from .indexer import load_all_files
from .retrieval import Retriever

if __name__ == "__main__":
    ## chunks
    try:
        with open("data/processed/bm25_index.pkl", "rb") as f:
            ...
        with open("data/processed/chunks.json") as f:
            ...
    except (FileNotFoundError, EOFError):
        processed = Path("data/processed")
        processed.mkdir(parents=True, exist_ok=True)
        chunk_text, chunk_code = load_all_files()

    try:
        with open("data/processed/query_cache.json", "r") as f:
            ...
    except (FileNotFoundError, EOFError):
        with open("data/processed/query_cache.json", "w") as f:
            json.dump({}, f)
    ## Normal
    # Query = "what is TCP/IP?"
    # search = Retriever().search
    # search_out = search(Query)
    ## Retriver
    generator = Generator()
    search_dataset = "data/datasets/UnansweredQuestions/dataset_code_public.json"
    save_res_path = "data/output/search_results/UnansweredQuestions"
    Retriever().search_dataset(search_dataset, 5, save_res_path)

    ## generator answeres
    stn_search_res_path = "data/output/search_results/UnansweredQuestions/dataset_code_public.json"
    save_dir = "data/output/search_results_and_answer/UnansweredQuestions"
    generator.answer_dataset(stn_search_res_path, save_dir)

    # context = generator.bulid_context(search_out)
    # result = generator.generate_answer(context, Query)

    # print(result)

    ## Recall@k
    r = Retriever()

    # # --- DOCS
    # r.search_dataset(
    #     dataset_path="data/datasets/UnansweredQuestions/dataset_docs_public.json",
    #     k=10,
    #     save_directory="data/output/search_results/UnansweredQuestions"
    # )

    # evaluate(
    # student_search_results_path="data/output/search_results/UnansweredQuestions/dataset_docs_public.json",
    # dataset_path="data/datasets/AnsweredQuestions/dataset_docs_public.json",
    # )

    # --- CODES
    r.search_dataset(
        dataset_path="data/datasets/UnansweredQuestions/dataset_code_public.json",
        k=5,
        save_directory="data/output/search_results/UnansweredQuestions"
    )

    evaluate(
    student_search_results_path="data/output/search_results/UnansweredQuestions/dataset_code_public.json",
    dataset_path="data/datasets/AnsweredQuestions/dataset_code_public.json",
    )
