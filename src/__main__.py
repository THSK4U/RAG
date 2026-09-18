# from .models import config

from .evaluator import evaluate
from .generator import Generator
from .indexer import load_all_files
from .retrieval import Retriever

if __name__ == "__main__":
    try:
        with open("data/processed/bm25_index.pkl", "rb") as f:
            ...
        with open("data/processed/chunks.json") as f:
            ...
    except (FileNotFoundError, EOFError):
        chunk_text, chunk_code = load_all_files()

    # Normal
    Query = "what is TCP/IP?"
    search = Retriever().search
    search_out = search(Query)
    generator = Generator()
    stn_search_res_path = "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
    save_dir = "data/output/search_results_and_answer/UnansweredQuestions"
    generator.answer_dataset(stn_search_res_path, save_dir)
    # context = generator.bulid_context(search_out)
    # result = generator.generate_answer(context, Query)

    # print(result)

    # Recall@k
    # r = Retriever()
    # r.search_dataset(
    #     dataset_path="data/datasets/UnansweredQuestions/dataset_docs_public.json",
    #     k=10,
    #     save_directory="data/output/search_results/UnansweredQuestions"
    # )

    # evaluate(
    # student_search_results_path="data/output/search_results/UnansweredQuestions/dataset_docs_public.json",
    # dataset_path="data/datasets/AnsweredQuestions/dataset_docs_public.json",
    # )

    # r.search_dataset(
    #     dataset_path="data/datasets/UnansweredQuestions/dataset_code_public.json",
    #     k=10,
    #     save_directory="data/output/search_results/UnansweredQuestions"
    # )

    # evaluate(
    # student_search_results_path="data/output/search_results/UnansweredQuestions/dataset_code_public.json",
    # dataset_path="data/datasets/AnsweredQuestions/dataset_code_public.json",
    # )
