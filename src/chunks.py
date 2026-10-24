import ast
import re
from itertools import accumulate

from .models import (
    FullSource,
    MarkdownMetadata,
    config,
)


def text_strategies(file: str) -> list[FullSource]:
    """Split Markdown/text documents into chunks by headers or max size.

    Args:
        file: Path to the text or Markdown file.

    Returns:
        list[FullSource]: List of extracted document chunks.
    """
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

        if (
            len(selected_text) <= config.max_chunk_size
            and len(selected_text.strip()) > 0
        ):
            chunks.append(
                FullSource(
                    file_path=file,
                    metadata=MarkdownMetadata(
                        title=title,
                        header=header,
                    ),
                    first_character_index=start,
                    last_character_index=end,
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
                        FullSource(
                            file_path=file,
                            metadata=MarkdownMetadata(
                                title=title,
                                header=header,
                            ),
                            first_character_index=new_start,
                            last_character_index=end_chunk,
                        )
                    )

                    overlap = min(150, max(10, config.max_chunk_size // 5))
                    new_start = max(new_start + 1, end_chunk - overlap)


                else:
                    chunks.append(
                        FullSource(
                            file_path=file,
                            metadata=MarkdownMetadata(
                                title=title,
                                header=header,
                            ),
                            first_character_index=new_start,
                            last_character_index=end,
                        )
                    )
                    new_start = end

    return chunks


def code_strategies(file: str) -> list[FullSource]:
    """Parse Python source files into code chunks using AST.

    Args:
        file: Path to the Python source file.

    Returns:
        list[FullSource]: List of extracted Python code chunks.
    """
    chunks = []
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
        obj_node: ast.FunctionDef | ast.ClassDef | None = None

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            obj_node = node
            start_line = node.lineno
            end_line = node.end_lineno or node.lineno

        if obj_node is not None:
            parts.append((obj_node, start_line, end_line))

    for obj_node, start_line, end_line in parts:
        first_char = cum_offsets[start_line - 1]
        last_char = cum_offsets[end_line]

        if (last_char - first_char) <= config.max_chunk_size:
            chunks.append(
                FullSource(
                    file_path=file,
                    first_character_index=first_char,
                    last_character_index=last_char,
                )
            )
        else:
            new_start = first_char
            while new_start < last_char:
                new_end = new_start + config.max_chunk_size
                if new_end < last_char:
                    split_text = content[new_start:new_end]
                    split_pos = max(
                        split_text.rfind("\n"), split_text.rfind(" ")
                    )
                    if split_pos > 0:
                        end_chunk = new_start + split_pos
                    else:
                        end_chunk = new_end
                    chunks.append(
                        FullSource(
                            file_path=file,
                            first_character_index=new_start,
                            last_character_index=end_chunk,
                        )
                    )
                    new_start = end_chunk
                else:
                    chunks.append(
                        FullSource(
                            file_path=file,
                            first_character_index=new_start,
                            last_character_index=last_char,
                        )
                    )
                    new_start = last_char

    return chunks
