# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف — بشكل متوازي (Parallel)
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
import hashlib
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rank_bm25 import BM25Okapi, BM25Plus
from tqdm import tqdm

from .chunks import code_strategies, text_strategies
from .models import FullSource, FunctionType, MarkdownMetadata, PythonMetadata, config

try:
    with open("data/processed/file_hashes.json", "r") as f:
        hashes_cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    hashes_cache = {}


def calculate_file_hash(file_path):
    file_path = str(file_path)

    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if file_path in hashes_cache and hashes_cache[file_path] == file_hash:
        return False

    hashes_cache[file_path] = file_hash
    with open("data/processed/file_hashes.json", "w") as f:
        json.dump(hashes_cache, f, indent=2)

    return True


def load_all_files() -> tuple[list[FullSource], list[FullSource]]:
    full_rebuild = False
    processed = Path(config.processed_dir)

    chunks_path = processed / "chunks.json"
    if not chunks_path.exists() or chunks_path.stat().st_size <= 20:
        with open(processed / "chunks.json", "w") as f:
            json.dump({}, f)
            full_rebuild = True

    file_hashes_cache_path = processed / "file_hashes.json"
    if (
        not file_hashes_cache_path.exists()
        or file_hashes_cache_path.stat().st_size <= 20
    ):
        with open(processed / "file_hashes.json", "w") as f:
            json.dump({}, f)

    query_cache_path = processed / "query_cache.json"
    if not query_cache_path.exists() or query_cache_path.stat().st_size <= 20:
        with open(processed / "query_cache.json", "w") as f:
            json.dump({}, f)

    try:
        codebase_database = Path(config.raw_dir)

        txt_files = [
            str(f)
            for f in codebase_database.rglob("*.txt")
            if not (f.name == "requirements.txt" and f.stat().st_size < 100)
            if "tests" not in f.parts
            if calculate_file_hash(f) or full_rebuild
        ] + [
            str(f)
            for f in codebase_database.rglob("*.md")
            if calculate_file_hash(f) or full_rebuild
        ]
        code_files = [
            str(f)
            for f in codebase_database.rglob("*.py")
            if not (f.name == "__init__.py" and f.stat().st_size < 100)
            and "tests" not in f.parts
            and not f.name.startswith("test_")
            if calculate_file_hash(f) or full_rebuild
        ]

        print(txt_files)
        workers = min(4, 16)

        md: list = []
        py: list = []

        if txt_files:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(text_strategies, f): f for f in txt_files}
                for future in tqdm(as_completed(futures), desc="Read Text Files"):
                    result = future.result()
                    if result:
                        md.extend(result)
                build_index_cache(md)

        if code_files:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(code_strategies, f): f for f in code_files}
                for future in tqdm(as_completed(futures), desc="Read Code Files"):
                    result = future.result()
                    if result:
                        py.extend(result)
                build_index_cache(py)

        # print(f"MD chunks: {len(md)} | PY chunks: {len(py)}")
        # print(md, py)
        # with open(config.processed_dir + "/chunks.json", "w") as f:
        #     # f.write(str(md))
        #     json.dump([x.model_dump(mode="json") for x in (md + py)], f, indent=4)


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


def build_index_cache(chunks):
    processed = Path("data/processed")
    with open(processed / "chunks.json", "r") as f:
        old_chunks = json.load(f)

    new_chunks_dict = {}
    for chunk in chunks:
        key = (chunk.file_path, chunk.first_character_index, chunk.last_character_index)
        new_chunks_dict[key] = chunk

    updated_chunks = []
    for chunk in old_chunks:
        key = (chunk["file_path"], chunk["first_character_index"], chunk["last_character_index"])
        if key in new_chunks_dict:
            updated_chunks.append(new_chunks_dict[key])
            del new_chunks_dict[key]
        else:
            updated_chunks.append(FullSource.model_validate(chunk))

    updated_chunks.extend(new_chunks_dict.values())


    corpus: list[list[str]] = []
    file_cash = {}

    if not updated_chunks:
        return
    for chunk in updated_chunks:
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

    bm25 = BM25Plus(corpus)

    with open(processed / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(processed / "chunks.json", "w") as f:
        json.dump([x.model_dump(mode="json") for x in updated_chunks], f, indent=4)

    print(f"Index built: {len(new_chunks_dict)} chunks")
