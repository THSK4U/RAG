# from .models import config
from .indexer import load_all_files
from .retrieval import search
from .generator import generate_answer, bulid_context
if __name__ == "__main__":
    try :
        with open("data/processed/bm25_index.pkl", "rb") as f:
            ...
        with open("data/processed/chunks.json") as f:
            ...
    except (FileNotFoundError, EOFError):
        chunk_text, chunk_code = load_all_files()

    Query = "How install llm local"
    search_out = search(Query)
    context = bulid_context(search_out)
    generate_answer(context, Query)
