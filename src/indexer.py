# 1. يجد كل ملفات .md و .py في vllm-0.10.1
# 2. يستدعي chunker على كل ملف
# 3. يحفظ كل الـ chunks في data/processed/chunks.json
# 4. يبني الـ BM25 index ويحفظه
