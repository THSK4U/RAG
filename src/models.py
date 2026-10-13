import uuid
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class FunctionType(str, Enum):
    """Enumeration of supported Python code construct types."""

    CLASS = "class"
    FUNCTION = "def"


class MarkdownMetadata(BaseModel):
    """Metadata schema for Markdown document sections."""

    title: str
    header: str


class PythonMetadata(BaseModel):
    """Metadata schema for Python code constructs."""

    type: FunctionType
    name: str
    imports: list[str] = []
    globals: list[str] = []


class FullSource(BaseModel):
    """Detailed source chunk schema including metadata."""

    file_path: str
    first_character_index: int
    last_character_index: int
    metadata: MarkdownMetadata | PythonMetadata


class MinimalSource(BaseModel):
    """Minimal source chunk representation without metadata."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question schema without answers or sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question schema containing ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset container holding RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Container for retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[FullSource | MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Container for a question, retrieved sources, and generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output schema for dataset search results at rank k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output schema for dataset search results and generated answers."""

    search_results: List[MinimalAnswer]
    k: int


class Config(BaseModel):
    """Global configuration settings for chunking and directories."""

    max_chunk_size: int = Field(default=2000)
    raw_dir: str = "data/raw/vllm-0.10.1"
    processed_dir: str = "data/processed"


config = Config()
