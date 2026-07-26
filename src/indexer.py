# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف — بشكل متوازي (Parallel)
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
from .chunks import md_chunker, py_chunker
from pathlib import Path
from .models import config
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def _chunk_md(file_str: str):
    return md_chunker(file_str)

def _chunk_py(file_str: str):
    return py_chunker(file_str)

def load_all_files():
    try:
        md_database = Path(config.raw_dir)
        py_database = Path(config.raw_dir)

        md_files = [str(f) for f in md_database.rglob("*.md")]
        py_files = [
            str(f) for f in py_database.rglob("*.py")
            if not (f.name == "__init__.py" and f.stat().st_size < 100)
            and "tests" not in f.parts
            and not f.name.startswith("test_")
        ]

        workers = min(4, 16)

        md: list = []
        py: list = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # futures = {executor.submit(_chunk_md, f): f for f in md_files}
            futures = {executor.submit(_chunk_md, md_files[2])}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    md.extend(result)

        # with ProcessPoolExecutor(max_workers=workers) as executor:
        #     # futures = {executor.submit(_chunk_py, f): f for f in py_files}
        #     futures = {executor.submit(_chunk_py, py_files[0])}
        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result:
        #             py.extend(result)

        print(f"MD chunks: {len(md)} | PY chunks: {len(py)}")
        # print(md, py)
        with open("output.txt", "w") as f:
            # f.write(str(md))
            import json
            json.dump([x.model_dump(mode="json") for x in md], f, indent=4)

        return md, py

    except FileNotFoundError:
        raise FileNotFoundError("vllm-0.10.1 dataset files not found.\n")
    except Exception as e:
        raise Exception(f"Invalid dataset: {e}\n")
