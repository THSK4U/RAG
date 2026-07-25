from enum import Enum
import uuid
from pydantic import BaseModel, Field
from typing import List

class FunctionType(str, Enum):
    CLASS = "class"
    FUNCTION = "def"

class MarkdownMetadata(BaseModel):
    title: str
    header: str

class PythonMetadata(BaseModel):
    type: FunctionType
    name: str
    imports: list[str] = []
    globals: list[str] = []
    calls: list[str] = []

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    metadata: MarkdownMetadata | PythonMetadata

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

class Config(BaseModel):
  max_chunk_size: int = 2000
  raw_dir: str = "data/raw/vllm-0.10.1"
  processed_dir: str = "data/processed"

config = Config()
