# للـ .md: يقطّع حسب العناوين ## و ###
# للـ .py: يقطّع حسب functions و classes (AST)
# يضمن أن كل chunk ≤ 2000 حرف
from .models import config 
from pathlib import Path
import re

def md_chunker(file: str):
    print(file)
    chunks = []

    with open(file, "r") as f:
        content = f.read()

    header_pattern = r"(^#{1,6}\s+.*$)"
    splits = list(re.finditer(header_pattern, content, re.MULTILINE))

    span = [0] + [m.start() for m in splits] + [len(content)]
    
    for i in range(len(span) - 1):
        start = span[i]
        end = span[i + 1]

        selected_text = content[start:end]

        if len(selected_text) <= config.max_chunk_size and len(selected_text.strip()) > 0:
            chunks.append({
                "file_path": file,
                "first_character_index": start,
                "last_character_index": end,
                "text": selected_text,
                })
        else:
            new_start = start
            while start < end:
                new_end = start + config.max_chunk_size
                if new_end < end:
                    split_text = selected_text[start:new_end]

                    split_pos = max(
                        split_text.rfind(".\n"),
                        split_text.rfind(". "),
                        split_text.rfind("\n\n"),
                        split_text.rfind("\n"),
                    )

    return chunks

def py_chunker(file: str) -> None:
    content = Path(file).read_text(encoding="utf-8")
    print(content)
    exit()

