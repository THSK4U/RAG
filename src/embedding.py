import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
from .models import config
from src.models import StudentSearchResults

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
processed = Path(config.processed_dir)

def lexical(corpus):
    bm25 = BM25Plus(corpus)

    with open(processed / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)


def Semantic_embidding(student_search_results_path):

    with open(student_search_results_path) as f:
        dataset = StudentSearchResults.model_validate_json(f.read())

        answers = []

        for s in tqdm(dataset.search_results, desc="Answering"):
            embeddings = model.encode(s.)
            print(embeddings)

            print(embeddings.shape)
            answers.append(embeddings)



    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
