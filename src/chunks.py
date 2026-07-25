# للـ .md: يقطّع حسب العناوين ## و ###
# للـ .py: يقطّع حسب functions و classes (AST)
# يضمن أن كل chunk ≤ 2000 حرف
from .models import config, MinimalSource, MarkdownMetadata, PythonMetadata, FunctionType
from pathlib import Path
import re

def md_chunker(file: str):
    print(file)
    chunks = []

    with open(file, "r") as f:
        content = f.read()

    header_pattern = r"^(#{1,6})\s+(.*)$"
    splits = list(re.finditer(header_pattern, content, re.MULTILINE))

    span = [0] + [m.start() for m in splits] + [len(content)]

    title = ""
    header = ""
    for i in range(len(span) - 1):
        start = span[i]
        end = span[i + 1]
        if i > 0:
            m = splits[i - 1]
            hashes = m.group(1)
            header_text = m.group(2)
            header = header_text
            if len(hashes) == 1:
                title = header_text

        selected_text = content[start:end]

        if selected_text.strip() == f"# {title}":
            continue

        if len(selected_text) <= config.max_chunk_size and len(selected_text.strip()) > 0:

            chunks.append(
                MinimalSource(
                    file_path=file,
                    metadata=MarkdownMetadata(
                        title=title,
                        header=header,
                    ),
                    first_character_index=start,
                    last_character_index=end,
                    # text=selected_text,
                )
            )
        else:
            new_start = start
            while new_start < end:
                new_end = new_start + config.max_chunk_size
                if new_end < end:
                    split_text = content[new_start:new_end]
                    split_pos = max(
                        split_text.rfind("\n### "),
                        split_text.rfind("\n#### "),
                        split_text.rfind("\n##### "),
                        split_text.rfind("\n###### "),
                        split_text.rfind("\n"),
                        split_text.rfind("\n\n"),
                        split_text.rfind(" "),
                        split_text.rfind("."),
                    )
                    if split_pos > 0:
                        end_chunk = new_start + split_pos
                    else:
                        end_chunk = new_end
                    chunks.append(
                        MinimalSource(
                            file_path=file,
                            metadata=MarkdownMetadata(
                                title=title,
                                header=header,
                            ),
                            first_character_index=new_start,
                            last_character_index=end_chunk,
                            # text=content[new_start:end_chunk],
                        )
                    )
                    segment = content[new_start:end_chunk]

                    last_newline = segment.rfind("\n")
                    last_space = segment.rfind(" ")

                    if last_newline > 0:
                        new_start += last_newline + 1
                    elif last_space > 0:
                        new_start += last_space + 1
                    else:
                        new_start = end_chunk
                else:
                    chunks.append(
                        MinimalSource(
                            file_path=file,
                            metadata=MarkdownMetadata(
                                title=title,
                                header=header,
                            ),
                            first_character_index=new_start,
                            last_character_index=end,
                            # text=content[new_start:end],
                        )
                    )
                    new_start = end

    return chunks

import ast
from itertools import accumulate

def py_chunker(file: str):
    print(file)

    chunks = []
    imports_list_total = {}
    global_list_total = []
    start_line = 0
    end_line = 0
    parts = []

    with open(file, "r") as f:
        content = f.read()

    line_lengths = [len(line) for line in content.splitlines(keepends=True)]
    cum_offsets = [0] + list(accumulate(line_lengths))

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    for node in tree.body:
        functiontype = ""
        obj_node = ""

        if isinstance(node, ast.Import):
            for body in node.names:
                if body.name not in imports_list_total:
                    imports_list_total[body.name] = f"import {body.name}"

        elif isinstance(node, ast.ImportFrom):
            for body in node.names:
                if body.asname:
                    key = body.asname
                    statement = f"from {node.module} import {body.name} as {body.asname}"
                else:
                    key = body.name
                    statement = f"from {node.module} import {body.name}"
                if key not in imports_list_total:
                    imports_list_total[key] = statement

        elif isinstance(node, ast.FunctionDef):
            if node.name:
                obj_node = node
                functiontype = FunctionType.FUNCTION
                start_line = node.lineno
                end_line = node.end_lineno

        elif isinstance(node, ast.ClassDef):
            if node.name:
                obj_node = node
                functiontype = FunctionType.CLASS
                start_line = node.lineno
                end_line = node.end_lineno
        else:
            global_list_total.append((node, node.lineno, node.end_lineno))

        if obj_node:
            parts.append((obj_node, functiontype, start_line, end_line))

    global_cache = []
    for node, lineno, end_lineno in global_list_total:
        segment = ast.get_source_segment(content, node)
        var_name = ""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    break
        global_cache.append((var_name, segment))

    for obj_node, functiontype, start_line, end_line in parts:
        first_char = cum_offsets[start_line - 1]
        last_char = cum_offsets[end_line]

        used_names = {
            node.id
            for node in ast.walk(obj_node)
            if isinstance(node, ast.Name)
        }

        imports_list = [
            import_
            for from_, import_ in imports_list_total.items()
            if from_ in used_names
        ]

        global_list = [
            segment
            for var_name, segment in global_cache
            if var_name and var_name in used_names
        ]

        chunks.append(
            MinimalSource(
                file_path=file,
                metadata=PythonMetadata(
                    type = functiontype,
                    name = obj_node.name,
                    imports = imports_list,
                    globals = global_list,
                ),
                first_character_index=first_char,
                last_character_index=last_char,
                # text=content[first_char:last_char],
            )
        )

    return chunks
