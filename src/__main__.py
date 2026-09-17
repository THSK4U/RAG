# from .models import config
from .indexer import load_all_files
from .retrieval import Retriever
from .generator import Generator

if __name__ == "__main__":
    try:
        with open("data/processed/bm25_index.pkl", "rb") as f:
            ...
        with open("data/processed/chunks.json") as f:
            ...
    except (FileNotFoundError, EOFError):
        chunk_text, chunk_code = load_all_files()

    Query = "what is TCP/IP?"

    search = Retriever().search
    search_out = search(Query)
    generator = Generator()
    context = generator.bulid_context(search_out)
    result = generator.generate_answer(context, Query)

    print(result)
