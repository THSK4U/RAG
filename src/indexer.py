# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
from .chunks import md_chunker, py_chunker
from pathlib import Path
from .models import config

def load_all_files():
    try :
        # md_database = Path(config.raw_dir)
        # py_database = Path(config.raw_dir)


        # md_files = list(md_database.rglob("*.md"))
        # py_files = list(py_database.rglob("*.py"))

        # print(md_chunker(str(md_files[1]))[0])
        # print(md_chunker(str(md_files[1]))[1])

        md_database = Path("./")

        md_files = list(md_database.rglob("READ.md"))

        res = md_chunker(str(md_files[0]))
        print(res)

        # for file in md_files:
        #     md_chunker(str(file))

        # for file in py_files:
        #     py_chunker(str(file))


    except FileNotFoundError:
        raise FileNotFoundError("vllm-0.10.1 dataset files not found.\n")
    except Exception:
        raise Exception("Invalid dataset.\n")
