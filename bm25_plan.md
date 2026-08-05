# خطة بناء BM25 Index

## الوضع الحالي
- **11,643 chunk** جاهزة في `data/processed/chunks.json`
- كل chunk عندها: `file_path`, `first_character_index`, `last_character_index`, `metadata`

---

## المراحل

### المرحلة 1 — بناء نص الـ Index لكل chunk

قبل إعطاء الـ chunk للـ BM25، يجب **بناء نص غني** يدمج الـ metadata مع المحتوى.

#### للـ Python chunks:
```python
def build_py_index_text(chunk: dict, content: str) -> str:
    meta = chunk["metadata"]
    text = content[chunk["first_character_index"]:chunk["last_character_index"]]
    
    parts = []
    # اسم الدالة/الكلاس — مهم جداً لأن الأسئلة تذكره مباشرة
    if meta.get("name"):
        parts.append(f"function: {meta['name']}")
    # الـ imports — تساعد في مطابقة أسئلة عن مكتبات معينة
    if meta.get("imports"):
        parts.append("\n".join(meta["imports"]))
    # الـ globals — متغيرات مهمة
    if meta.get("globals"):
        parts.append("\n".join(meta["globals"]))
    # الكود الفعلي
    parts.append(text)
    
    return "\n".join(parts)
```

#### للـ Markdown/Text chunks:
```python
def build_text_index_text(chunk: dict, content: str) -> str:
    meta = chunk["metadata"]
    text = content[chunk["first_character_index"]:chunk["last_character_index"]]
    
    parts = []
    # العنوان — يساعد في البحث بالموضوع
    if meta.get("title"):
        parts.append(meta["title"])
    if meta.get("header") and meta["header"] != meta.get("title"):
        parts.append(meta["header"])
    parts.append(text)
    
    return "\n".join(parts)
```

---

### المرحلة 2 — Tokenization وتنظيف النص

BM25 يعتمد على كلمات — التنظيف مهم جداً:

```python
import re
from typing import list

def tokenize(text: str) -> list[str]:
    # حوّل CamelCase إلى كلمات منفصلة: "AsyncLLMEngine" → ["Async", "LLM", "Engine"]
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # حوّل snake_case إلى كلمات: "async_engine" → ["async", "engine"]
    text = text.replace('_', ' ')
    # أزل الرموز وأبق الحروف والأرقام
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # حوّل لأحرف صغيرة وقسّم
    tokens = text.lower().split()
    # أزل stopwords (اختياري لكن يحسّن الدقة)
    stopwords = {'the', 'a', 'an', 'is', 'in', 'it', 'of', 'to', 'and', 'or'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]
```

> **ملاحظة:** CamelCase splitting مهم جداً لكود Python لأن الأسئلة مثل *"What does LlamaAttention do?"* يجب أن تطابق `LlamaAttention` في الكود.

---

### المرحلة 3 — بناء الـ BM25 Index

```python
from rank_bm25 import BM25Okapi
import json
import pickle
from pathlib import Path

def build_index(chunks_path: str, raw_dir: str) -> None:
    with open(chunks_path) as f:
        chunks = json.load(f)
    
    # cache لمحتوى الملفات (لتجنب فتح نفس الملف أكثر من مرة)
    file_cache: dict[str, str] = {}
    
    corpus: list[list[str]] = []
    
    for chunk in chunks:
        # اقرأ الملف مرة واحدة
        fp = chunk["file_path"]
        if fp not in file_cache:
            with open(fp, "r", errors="ignore") as f:
                file_cache[fp] = f.read()
        
        content = file_cache[fp]
        meta = chunk["metadata"]
        
        # اختر طريقة البناء حسب نوع الملف
        if "name" in meta:  # Python chunk
            index_text = build_py_index_text(chunk, content)
        else:               # Markdown/Text chunk
            index_text = build_text_index_text(chunk, content)
        
        tokens = tokenize(index_text)
        corpus.append(tokens)
    
    # بناء الـ BM25
    bm25 = BM25Okapi(corpus)
    
    # حفظ الـ index
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True)
    
    with open(processed / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    
    # حفظ الـ chunks مرتبطة بالـ index (للـ retrieval)
    with open(processed / "chunks.json", "w") as f:
        json.dump(chunks, f)
    
    print(f"Index built: {len(chunks)} chunks")
```

---

### المرحلة 4 — الـ Retrieval

```python
def search(query: str, k: int = 5) -> list[dict]:
    # حمّل الـ index
    with open("data/processed/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    
    with open("data/processed/chunks.json") as f:
        chunks = json.load(f)
    
    # tokenize السؤال بنفس الطريقة
    query_tokens = tokenize(query)
    
    # احصل على scores
    scores = bm25.get_scores(query_tokens)
    
    # خذ أفضل k نتائج
    top_k_indices = scores.argsort()[::-1][:k]
    
    results = []
    for idx in top_k_indices:
        chunk = chunks[idx]
        results.append({
            "file_path": chunk["file_path"],
            "first_character_index": chunk["first_character_index"],
            "last_character_index": chunk["last_character_index"],
            "score": float(scores[idx])
        })
    
    return results
```

---

### المرحلة 5 — ضبط BM25 Parameters

BM25Okapi له معاملان مهمان:

| المعامل | القيمة الافتراضية | المعنى |
|---|---|---|
| `k1` | 1.5 | حساسية تكرار الكلمة (زيادة = أهمية أكبر للتكرار) |
| `b` | 0.75 | تطبيع طول الـ chunk (0 = لا تطبيع، 1 = تطبيع كامل) |

```python
# للكود (chunks أطول عادة) — قلّل b
bm25_code = BM25Okapi(code_corpus, k1=1.5, b=0.5)

# للـ docs — الافتراضي يعمل جيداً
bm25_docs = BM25Okapi(docs_corpus, k1=1.5, b=0.75)
```

> **نصيحة:** يمكن بناء **index منفصل** للكود وآخر للـ docs، ثم اختيار أيهما تُستخدم حسب نوع السؤال.

---

## ترتيب التنفيذ

- `[ ]` بناء دوال `build_py_index_text` و `build_text_index_text`
- `[ ]` بناء دالة `tokenize` مع CamelCase splitting
- `[ ]` بناء `build_index` وحفظ `bm25_index.pkl`
- `[ ]` بناء `search` وتجربتها على أسئلة من `AnsweredQuestions`
- `[ ]` قياس recall@5 وضبط الـ parameters

---

## المصادر — ابدأ بهذه

### فهم BM25
| المصدر | الرابط | النوع |
|---|---|---|
| ورقة BM25 الأصلية | [Robertson et al. 1994](https://www.staff.city.ac.uk/~sye/trec3.ps) | Paper |
| شرح مرئي ممتاز | [BM25 Explained - Towards Data Science](https://towardsdatascience.com/understanding-term-based-retrieval-methods-in-nlp-85cd86571c95) | Article |
| مقارنة BM25 vs TF-IDF | [Pinecone Blog](https://www.pinecone.io/learn/series/nlp/bm25/) | Article |

### rank_bm25 Library
| المصدر | الرابط |
|---|---|
| GitHub الرسمي | [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| API docs | في الـ README مباشرة |

### RAG + BM25 عملي
| المصدر | الرابط |
|---|---|
| LlamaIndex BM25 | [BM25 Retriever Docs](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/) |
| Haystack RAG Pipeline | [Haystack BM25](https://docs.haystack.deepset.ai/docs/bm25retriever) |
| RAG Survey Paper | [arxiv 2312.10997](https://arxiv.org/abs/2312.10997) |

### Code Retrieval خصيصاً
| المصدر | الرابط |
|---|---|
| Code Search Best Practices | [GitHub Code Search](https://github.blog/2023-02-06-the-technology-behind-githubs-new-code-search/) |
| CamelCase Tokenization | [Stack Overflow](https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case) |
