# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف — بشكل متوازي (Parallel)
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
from .chunks import text_strategies, code_strategies
from pathlib import Path
from .models import config
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def _chunk_text(file_str: str):
    return text_strategies(file_str)

def _chunk_code(file_str: str):
    return code_strategies(file_str)

def load_all_files():
    try:
        codebase_database = Path(config.raw_dir)

        txt_files = [
            str(f) for f in codebase_database.rglob("*.txt")
            if not (f.name == "requirements.txt" and f.stat().st_size < 100)
            if "tests" not in f.parts
        ] + [str(f) for f in codebase_database.rglob("*.md")]
        code_files = [
            str(f) for f in codebase_database.rglob("*.py")
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
            for future in as_completed(futures):
                result = future.result()
                if result:
                    md.extend(result)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_chunk_code, f): f for f in code_files}
            # futures = {executor.submit(_chunk_code, code_files[0])}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    py.extend(result)

        print(f"MD chunks: {len(md)} | PY chunks: {len(py)}")
        # print(md, py)
        with open(config.processed_dir + "/chunks.json", "w") as f:
            # f.write(str(md))
            import json
            json.dump([x.model_dump(mode="json") for x in (md + py)], f, indent=4)

        return md, py

    except FileNotFoundError:
        raise FileNotFoundError("vllm-0.10.1 dataset files not found.\n")
    except Exception as e:
        raise Exception(f"Invalid dataset: {e}\n")
