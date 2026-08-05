# مخطط معمارية مشروع RAG against the machine: الخطة المحدثة والمطابقة لكراس المشروع

هذا المستند يقدم المخطط المعماري المحدث والنهائي لمشروع **RAG against the machine** ليكون مطابقاً بنسبة 100% لكافة الاشتراطات والقيود التقنية الواردة في كراس المشروع الرسمي والمعايير التي تفحصها أداة التقييم الآلي (**Moulinette**) والجلسة الدفاعية (Defense).

---

## ⚠️ القيود والاشتراطات التقنية الصارمة

1. **إدارة الاعتماديات باستخدام `uv` حصراً:**
   - التقييم والـ Moulinette سينفذان فقط الأمر `uv sync`.
   - يجب وجود ملفي `pyproject.toml` و `uv.lock` في جذر المشروع.
2. **الواجهة البرمجية (CLI) باستخدام `Python Fire`:**
   - جميع الأوامر يتم استدعاؤها عبر الصيغة القياسية: `uv run python -m src <command> [options]`.
   - استخدام مكتبة `tqdm` لعرض أشرطة التقدم في العمليات طويلة التنفيذ.
3. **الالتزام الصارم بنماذج Pydantic:**
   - البيانات المتبادلة في ملفات JSON يجب أن تطابق تماماً هياكل Pydantic المحددة في الكراس (`MinimalSource`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`, إلخ).
4. **حد حجم الجزء (Max Chunk Size = 2000 حرف):**
   - الحجم الأقصى الافتراضي لأي جزء هو **2000 حرف (Character)**.
   - أي جزء يتجاوز 2000 حرف سيؤدي إلى تصفير نتيجة التقييم فوراً من الـ Moulinette.
5. **مطابقة المسارات المرجعية (Verbatim File Paths):**
   - حقل `file_path` يجب أن يطابق المسار النسبي الدقيق للأساس داخل مجلد المستودع (مثل `data/raw/vllm-0.10.1/docs/features/lora.md`).
6. **قيود الأداء والتوقيت:**
   - زمن الفهرسة كاملة: **أقل من 5 دقائق**.
   - زمن الاسترجاع لـ 200 سؤال: **أقل من 90 ثانية**.
   - دقة الاسترجاع المطلوبة: **Recall@5 ≥ 80%** للوثائق (`docs`) و **Recall@5 ≥ 50%** للكود (`code`).

---

## 📂 الهيكل المعتمد للمجلدات (Project Directory Layout)

```text
rag-against-the-machine/
├── pyproject.toml            <-- يُدار بواسطة uv
├── uv.lock                   <-- ملف قفل الاعتماديات
├── Makefile                  <-- يحتوي القواعد الإلزامية الخمس
├── README.md                 <-- وثيقة الشرح باللغة الإنجليزية
├── data/
│   ├── raw/                  <-- يحتوي مستودع vLLM الخام (مثل vllm-0.10.1)
│   ├── processed/            <-- يُحفظ فيه الفهرس المبني
│   ├── datasets/             <-- أسئلة التقييم (UnansweredQuestions / AnsweredQuestions)
│   └── output/               <-- المخرجات الموجهة عبر الـ CLI
│       ├── search_results/
│       └── search_results_and_answer/
└── src/
    ├── __init__.py
    ├── __main__.py           <-- نقطة تشغيل CLI باستخدام Fire
    ├── models.py             <-- نماذج Pydantic الإلزامية
    ├── indexer.py            <-- منطق AST/Markdown Chunking والفهرسة
    ├── retriever.py          <-- محرك البحث اللفظي (BM25 أو TF-IDF)
    ├── generator.py          <-- نموذج Qwen/Qwen3-0.6B وتوليد الإجابات
    └── evaluator.py          <-- حساب Recall@k الذاتي للاختبار المحلي
```

---

## 🧱 المخطط البرمجي للطبقات (Layer Architecture)

### 1️⃣ الطبقة الأولى: نماذج البيانات الصارمة (`src/models.py`)
تُبنى باستخدام **Pydantic** وحقولها مطابقة حرفياً للمواصفات لضمان قبول الـ Moulinette:

- **`MinimalSource`**:
  - `file_path: str`
  - `first_character_index: int`
  - `last_character_index: int`
- **`UnansweredQuestion`**:
  - `question_id: str` (افتراضياً UUID)
  - `question: str`
- **`AnsweredQuestion`**:
  - `sources: List[MinimalSource]`
  - `answer: str`
- **`StudentSearchResults`**:
  - `search_results: List[MinimalSearchResults]`
  - `k: int`
- **`StudentSearchResultsAndAnswer`**:
  - `search_results: List[MinimalAnswer]`
  - `k: int`

### 2️⃣ الطبقة الثانية: التقسيم والفهرسة الإلزامية (`src/indexer.py`)
تتحمل مسؤولية قراءة المستودع الخام وتقسيمه إلى أجزاء ذكية دون فصل المنطق البرمجي أو السياق النصي:

- **مقسم Python الكودي (`ast`):** استخدام شجرة Syntax Tree لتحديد حدود الدوال والفئات وحساب مؤشرات الحروف الأوليّة والنهائية بدقة، مع وجود دالة احتياطية (Fallback) عند تعذر تحليل بناء الجملة.
- **مقسم Markdown/Text:** تحليل الفقرات والعناوين وتجميع النصوص المرتبطة.
- **ضابط الحجم الصارم:** التقيد بالحد الأقصى `--max_chunk_size` (2000 حرف) وتقطيع الكتل الكبيرة بأمان.
- **حد الأداء:** إتمام عملية الفهرسة بالكامل وحفظ الناتج تحت `data/processed/` في زمن **أقل من 5 دقائق**.

### 3️⃣ الطبقة الثالثة: محرك الاسترجاع والبحث (`src/retriever.py`)
تمثل محرك البحث اللفظي (Lexical Retrieval):

- **الخوارزمية:** تطبيق **BM25** (باستخدام `rank_bm25` أو تطبيق مخصص) أو **TF-IDF** (باستخدام `scikit-learn`).
- **مطابقة المسارات:** إرجاع المسار الدقيق ونطاق الحروف لكل نتيجة.
- **حد الأداء:** معالجة **200 سؤال في أقل من 90 ثانية** مع تحقيق:
  - **Recall@5 ≥ 80%** لأسئلة التوثيق (`docs`).
  - **Recall@5 ≥ 50%** لأسئلة الكود (`code`).

### 4️⃣ الطبقة الرابعة: التوليد والتعزيز الموجه (`src/generator.py`)
دمج المصادر المسترجعة وتغذية النموذج المحلي:

- **النموذج:** **`Qwen/Qwen3-0.6B`** (محلياً عبر `transformers`).
- **إدارة نافذة السياق:** دمج المصادر داخل هندسة الأوامر (Prompt Template) دون تجاوز حد الرموز المسموح، وضمان الالتزام بالمصادر لمنع الهلوسة (Grounding).

### 5️⃣ الطبقة الخامسة: واجهة التحكم والأتمتة (`src/__main__.py`)
تُبنى باستخدام **`Python Fire`** وتستخدم **`tqdm`** لمتابعة التقدم. الأوامر الإلزامية:

1. `index --max_chunk_size <int>`
2. `search <query> --k <int>`
3. `search_dataset --dataset_path <path> --k <int> --save_directory <dir>`
4. `answer <query> --k <int>`
5. `answer_dataset --student_search_results_path <path> --save_directory <dir>`
6. `evaluate --student_search_results_path <path> --dataset_path <path>`

### 6️⃣ الطبقة السادسة: البنية التحتية والتحقق (`Makefile`)
يحتوي القواعد الإلزامية الخمس مع خيارات الفحص الصارمة:

```makefile
install:
	uv sync

run:
	uv run python -m src index

debug:
	uv run python -m pdb -m src index

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache data/processed/*

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src
```

---

## ⚡ خطة التنفيذ المتسلسلة (Actionable Roadmap)

1. **إعداد البيئة:** إنشاء بيئة العمل بـ `uv init` وتثبيت الاعتماديات الأساسية (`pydantic`, `fire`, `tqdm`, `scikit-learn` / `rank-bm25`, `transformers`, `torch`, `flake8`, `mypy`).
2. **إنشاء نماذج Pydantic (`src/models.py`):** كتابة وتأكيد صحة البيانات المتادلة وفق المواصفات.
3. **تطوير وحدات التقطيع والفهرسة (`src/indexer.py`):** دمج مقسم `ast` للبايثون ومقسم Markdown وتطبيق شرط الـ 2000 حرف.
4. **بناء محرك الاسترجاع (`src/retriever.py` & `src/evaluator.py`):** تطبيق BM25/TF-IDF واختبار Recall@k محلياً.
5. **ربط النموذج اللغوي (`src/generator.py`):** إعداد سياق التوليد باستخدام Qwen3-0.6B.
6. **بناء واجهة الأوامر CLI (`src/__main__.py`):** ربط الأوامر الستة بـ `Python Fire` ومعالجة المدخلات الاستثنائية والخطأ بلطف.
7. **كتابة وثيقة الشرح README.md باللغة الإنجليزية وتطبيق متطلبات الفصل VIII.**
