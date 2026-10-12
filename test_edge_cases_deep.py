import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    Config,
)
from src.indexer import tokenize, build_from_text, build_from_code
from src.chunks import text_strategies, code_strategies

def test_models_exact_schema():
    """Verify all 8 Pydantic models conform to the subject specification."""
    # 1. MinimalSource
    src = MinimalSource(file_path="data/raw/vllm/test.py", first_character_index=0, last_character_index=100)
    assert src.file_path == "data/raw/vllm/test.py"
    assert src.first_character_index == 0
    assert src.last_character_index == 100

    # 2. UnansweredQuestion
    uq = UnansweredQuestion(question="What is vLLM?")
    assert uq.question == "What is vLLM?"
    assert uq.question_id is not None

    # 3. AnsweredQuestion
    aq = AnsweredQuestion(question="What is vLLM?", sources=[src], answer="vLLM is an LLM engine.")
    assert aq.answer == "vLLM is an LLM engine."
    assert len(aq.sources) == 1

    # 4. RagDataset
    dataset = RagDataset(rag_questions=[uq, aq])
    assert len(dataset.rag_questions) == 2

    # 5. MinimalSearchResults
    msr = MinimalSearchResults(question_id=uq.question_id, question=uq.question, retrieved_sources=[src])
    assert len(msr.retrieved_sources) == 1

    # 6. MinimalAnswer
    ma = MinimalAnswer(question_id=uq.question_id, question=uq.question, retrieved_sources=[src], answer="test answer")
    assert ma.answer == "test answer"

    # 7. StudentSearchResults
    ssr = StudentSearchResults(search_results=[msr], k=10)
    assert ssr.k == 10
    assert len(ssr.search_results) == 1

    # 8. StudentSearchResultsAndAnswer
    ssra = StudentSearchResultsAndAnswer(search_results=[ma], k=10)
    assert ssra.k == 10
    assert len(ssra.search_results) == 1

def test_tokenizer_edge_cases():
    """Stress test the tokenizer on weird, degenerate, multilingual, or special code tokens."""
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize("the a an is in it of to and or") == []
    assert tokenize("!@#$%^&*()_+=-~`{}[]|\\:;\"'<>,.?/") == []
    # CamelCase splitting
    tokens = tokenize("OpenAICompatibleServer")
    assert "open" in tokens
    assert "compatible" in tokens
    assert "server" in tokens
    # Snake_case splitting
    tokens_snake = tokenize("vllm_engine_core_v2")
    assert "vllm" in tokens_snake
    assert "engine" in tokens_snake
    assert "core" in tokens_snake

def test_chunking_constraints():
    """Verify chunking respects max_chunk_size <= 2000 chars strictly."""
    # Test on an existing markdown file in raw
    md_files = list(Path("data/raw/vllm-0.10.1").rglob("*.md"))
    if md_files:
        sample_md = str(md_files[0])
        chunks = text_strategies(sample_md)
        for c in chunks:
            span_len = c.last_character_index - c.first_character_index
            assert span_len <= 2000, f"Markdown chunk exceeded 2000 chars: {span_len} in {sample_md}"

    py_files = list(Path("data/raw/vllm-0.10.1").rglob("*.py"))
    if py_files:
        sample_py = str(py_files[0])
        chunks = code_strategies(sample_py)
        for c in chunks:
            span_len = c.last_character_index - c.first_character_index
            assert span_len <= 2000, f"Code chunk exceeded 2000 chars: {span_len} in {sample_py}"

print("Unit edge-case tests structure defined successfully.")
