# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف — بشكل متوازي (Parallel)
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm_test import tqdm

from .chunks import code_strategies, text_strategies
from .models import FullSource, FunctionType, PythonMetadata, config


def _chunk_text(file_str: str):
    return text_strategies(file_str)


def _chunk_code(file_str: str):
    return code_strategies(file_str)


def load_all_files() -> tuple[list[FullSource], list[FullSource]]:
    try:
        codebase_database = Path(config.raw_dir)

        txt_files = [
            str(f)
            for f in codebase_database.rglob("*.txt")
            if not (f.name == "requirements.txt" and f.stat().st_size < 100)
            if "tests" not in f.parts
        ] + [str(f) for f in codebase_database.rglob("*.md")]
        code_files = [
            str(f)
            for f in codebase_database.rglob("*.py")
            if not (f.name == "__init__.py" and f.stat().st_size < 100)
            and "tests" not in f.parts
            and not f.name.startswith("test_")
        ]

        workers = min(4, 16)

        md: list = []
        py: list = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_chunk_text, f): f for f in txt_files}
            # futures = {executor.submit(_chunk_text, txt_files[2])}
            for future in tqdm(as_completed(futures), desc="Read Text Files"):
                result = future.result()
                if result:
                    md.extend(result)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_chunk_code, f): f for f in code_files}
            # futures = {executor.submit(_chunk_code, code_files[0])}
            for future in tqdm(as_completed(futures), desc="Read Code Files"):
                result = future.result()
                if result:
                    py.extend(result)

        # print(f"MD chunks: {len(md)} | PY chunks: {len(py)}")
        # print(md, py)
        # with open(config.processed_dir + "/chunks.json", "w") as f:
        #     # f.write(str(md))
        #     json.dump([x.model_dump(mode="json") for x in (md + py)], f, indent=4)

        build_index(md + py)

        return md, py

    except FileNotFoundError:
        raise FileNotFoundError("vllm-0.10.1 dataset files not found.\n")
    except Exception as e:
        raise Exception(f"Invalid dataset: {e}\n")


# class FullSource(BaseModel):
#     file_path: str
#     first_character_index: int
#     last_character_index: int
#     metadata: MarkdownMetadata | PythonMetadata
def build_from_text(chunk, content) -> str:
    first_char = chunk.first_character_index
    last_char = chunk.last_character_index
    meta = chunk.metadata

    parts = []
    if meta.title:
        parts.append(meta.title)
    if meta.header and meta.title != meta.header:
        parts.append(meta.header)

    parts.append(content[first_char:last_char])

    return "\n".join(parts)


def build_from_code(chunk, content):
    first_char = chunk.first_character_index
    last_char = chunk.last_character_index
    meta = chunk.metadata

    parts = []
    if meta.type:
        if meta.type == FunctionType.CLASS:
            parts.append(f"Class: {meta.name}")
        else:
            parts.append(f"function: {meta.name}")

    if meta.imports:
        parts.append(f"imports: {meta.imports}")
    if meta.globals:
        parts.append(f"globals: {meta.globals}")

    parts.append(content[first_char:last_char])

    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.lower().split()
    stopwords = {"the", "a", "an", "is", "in", "it", "of", "to", "and", "or"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def build_index(chunks):
    file_cash = {}
    corpus: list[list[str]] = []

    for chunk in chunks:
        file_path = chunk.file_path
        if file_path not in file_cash:
            with open(file_path, "r") as f:
                file_cash[file_path] = f.read()

        content = file_cash[file_path]
        metadata = chunk.metadata
        if isinstance(metadata, PythonMetadata):
            index_text = build_from_code(chunk, content)
        else:
            index_text = build_from_text(chunk, content)
        ##
        tokens = tokenize(index_text)
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True)
    with open(processed / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(processed / "chunks.json", "w") as f:
        json.dump([x.model_dump(mode="json") for x in chunks], f, indent=4)

    print(f"Index built: {len(chunks)} chunks")
