*This project has been created as part of the 42 curriculum by tsellak.*

# RAG Against the Machine

A lightweight and fast Retrieval-Augmented Generation (RAG) system built to index and answer questions about the **vLLM** project documentation and codebase.

---

## 1. System Architecture

The system consists of four main parts:

```
[Raw Documents / Code]
        │
        ▼
   1. Chunking  ───► (Markdown Headers & Python AST)
        │
        ▼
   2. Indexing  ───► (BM25 Index + Dense Vector Embeddings)
        │
        ▼
   3. Retrieval ───► (Hybrid Search: BM25 + Embeddings merged via RRF)
        │
        ▼
   4. Generator ───► (Qwen/Qwen3-0.6B LLM generates grounded answer)
```

1. **Ingestion & Chunking**: Reads markdown files and python files, splitting them into logical chunks up to 2000 characters.
2. **Indexing**: Creates a lexical index using BM25 and a semantic index using dense embeddings (`all-MiniLM-L6-v2`). Saves indices under `data/processed/`.
3. **Retrieval**: Searches the index for a user query using both BM25 and semantic similarity, merging results using Reciprocal Rank Fusion (RRF).
4. **Generation**: Feeds the top retrieved text chunks as context to `Qwen/Qwen3-0.6B` to generate concise, accurate answers.

---

## 2. Chunking Strategy

The system implements two specialized chunking strategies:

### A. Markdown / Text Strategy (`text_strategies`)
- Splits documents based on Markdown headers (`#`, `##`, `###`).
- Keeps related text together under its section header.
- If a section is larger than `--max_chunk_size` (default: 2000 characters), it safely splits on line breaks or spaces without cutting words in half.

### B. Python Code Strategy (`code_strategies`)
- Uses Python's Abstract Syntax Tree (`ast`) to parse code structurally.
- Extracts individual functions (`ast.FunctionDef`) and classes (`ast.ClassDef`).
- Automatically attaches relevant `import` statements and global variable definitions used by each function so the context remains standalone.

### Chunk Size Impact
- **Too small (e.g. < 200 characters)**: Chunks lose surrounding context, missing variables or explanation.
- **Too large (e.g. > 4000 characters)**: Dilutes relevant information with unrelated noise and exceeds the model's context window.
- **Configurable**: Set using `--max_chunk_size` (default is 2000 characters).

---

## 3. Retrieval Method

We use a **Hybrid Retrieval** method combining two complementary algorithms:

1. **Lexical Search (BM25Plus)**:
   - Matches exact keywords, function names, CLI flags, and error codes.
   - Uses BM25+ to prevent term-frequency saturation and length bias.

2. **Semantic Search (Dense Embeddings)**:
   - Uses `sentence-transformers/all-MiniLM-L6-v2` running on CPU.
   - Computes cosine similarity between question embedding and chunk embeddings to capture meaning even when wording differs.

3. **Ranking Mechanism (Reciprocal Rank Fusion - RRF)**:
   - Combines both rankings into a final score:
     $$\text{RRF Score} = \frac{1}{60 + \text{rank}_{\text{bm25}}} + \frac{1}{60 + \text{rank}_{\text{semantic}}}$$
   - Returns the top-$k$ highest-scoring chunks.

---

## 4. Performance Analysis

The system was evaluated against the official test datasets:

| Metric | Result | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Docs Recall@5** | **81.0%** | $\ge$ 80.0% | **PASSED** |
| **Code Recall@5** | **61.0%** | $\ge$ 50.0% | **PASSED** |
| **Indexing Time** | **7 seconds** | $\le$ 300 seconds | **PASSED** |
| **Retrieval Throughput** | **44 seconds** (200 questions) | $\le$ 90 seconds | **PASSED** |
| **Answer Quality** | **3/3 satisfactory** | $\ge$ 2/3 | **PASSED** |

---

## 5. Design Decisions & Trade-offs

- **AST Parsing vs Fixed Character Slicing**:
  - *Decision*: Parse Python with AST instead of fixed slices.
  - *Trade-off*: Slower initial parsing, but preserves code logic and prevents cutting functions in half, directly raising Code Recall@5 above 60%.
- **Hybrid Search vs Pure Lexical**:
  - *Decision*: Combine BM25 with `all-MiniLM-L6-v2`.
  - *Trade-off*: Requires CPU embedding computation, but catches conceptual questions that don't match exact words in docs.
- **Lightweight LLM (`Qwen/Qwen3-0.6B`)**:
  - *Decision*: Fast inference on CPU without requiring GPU hardware.
  - *Trade-off*: Limited complex reasoning, mitigated by strict prompt instructions to extract direct facts from retrieved context.

---

## 6. Example Usage

### Installation
```bash
# Sync dependencies with uv
uv sync
```

### 1. Indexing Corpus
```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Search a Single Query
```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

### 3. Search an Entire Dataset
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Answer a Single Query
```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

### 5. Answer an Entire Dataset
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 6. Evaluate Retrieval Results
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## 7. Bonus Features

The project implements **4 Advanced RAG bonuses**:

1. **Semantic Embeddings**: Uses CPU model `all-MiniLM-L6-v2` (`src/embedding.py`).
2. **Hybrid Retrieval**: Merges BM25 and vector rankings via Reciprocal Rank Fusion (`src/retrieval.py`).
3. **Incremental Indexing**: Uses SHA-256 hashes (`file_hashes.json`) to skip re-indexing unchanged files (`src/indexer.py`).
4. **Caching**: Caches chunks, file hashes, and query results in `data/processed/` for fast repeated retrieval.
