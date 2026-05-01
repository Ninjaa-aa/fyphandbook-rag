# FYP Handbook RAG Assistant — Architectural Note

**Course:** Generative AI &middot; **Assignment:** #3 (RAG)
**Corpus:** *BS Final Year Project Handbook 2023*, FAST-NUCES.

## 1. Pipeline Overview

```
PDF -> heading-aware chunks -> BGE-M3 embeddings + BM25 -> hybrid retrieval (RRF)
    -> threshold gate (>= 0.25 cosine) -> cross-encoder rerank
    -> MMR (lambda=0.7) -> top-5 chunks -> Groq Llama 3.3 70B (temp 0.1)
```

The pipeline is built for **citation faithfulness**: every retrieved chunk
carries `page` / `page_end` / `section` metadata, and the system prompt forces
inline `(p. X)` citations on every factual sentence.

## 2. Chunking

- **Parser:** PyMuPDF (`get_text("dict")`) — preserves font-size and bold flags
  per span, enabling heading detection without OCR.
- **Strategy:** heading-aware sectioning. A span is a heading if any of:
  font-size > 1.15x median, bold + short, ALL-CAPS short line, or matches
  `^\d+(\.\d+)*\s+[A-Z]`.
- **Sub-chunking:** sections longer than 800 chars are split with
  `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`.
- **Merging:** sections shorter than 120 chars are merged into the next, so
  micro-headings (e.g. "Margins") inherit content from their successor.
- **Metadata stored:** `page`, `page_end` (for sections that span pages),
  `section` (the heading text), `chunk_id`.

## 3. Embedding Model

- **Model:** `BAAI/bge-m3` (1024 dims, multilingual, 8k context).
- **Reason:** state-of-the-art on MTEB retrieval at its size; runs locally via
  `sentence-transformers` (no API costs / rate limits).
- Embeddings are L2-normalized so ChromaDB cosine distance maps cleanly to
  `similarity = 1 - distance`.

## 4. Indexing

- **Vector DB:** ChromaDB persistent client at `data/chroma/` with
  `{"hnsw:space": "cosine"}`.
- **BM25:** `rank_bm25` BM25Okapi over a lightweight regex tokenizer
  (lowercase alpha-num + stopword filter), pickled to `data/bm25.pkl`.
- **Corpus snapshot:** `data/chunks.pkl` maps `chunk_id -> {text, metadata}` so
  the retriever can hydrate hits without round-tripping through Chroma.

## 5. Retrieval

| Stage | Setting | Why |
|---|---|---|
| Dense top-30 | BGE-M3 + Chroma cosine | Recall on paraphrase / semantic queries |
| BM25 top-30 | Pickled BM25Okapi | Recall on exact-token queries (e.g. "Ibid.", "op. cit.") |
| RRF fusion | k = 60, top-20 | Robust score-free fusion of dense + sparse |
| Threshold | max dense cosine >= 0.25 | Assignment-mandated refusal gate (no LLM call below threshold) |
| Reranker | `BAAI/bge-reranker-v2-m3` (top-10) | Cross-encoder precision boost over bi-encoder hits |
| Reranker guard | logit >= -2.0 | Drops weak chunks before LLM context window |
| MMR | lambda = 0.7, top-5 | Removes near-duplicate chunks to broaden coverage |

## 6. Generation

- **LLM:** `llama-3.3-70b-versatile` via Groq (free tier).
- **Temperature:** 0.1 — keeps answers faithful to retrieved text.
- **System prompt:**
  - Forces context-only answers,
  - Mandates `(p. X)` / `(p. X-Y)` inline citations on every factual claim,
  - Specifies the exact refusal string when context is insufficient.
- **Retries:** 2 retries with exponential backoff on 429 / transient errors.

## 7. Refusal & Guardrails

Two complementary guards keep the model bounded:

1. **Pre-LLM**: if max dense cosine < `0.25`, return the assignment-required
   refusal string *without* invoking the LLM.
2. **Post-rerank**: if the cross-encoder rejects all candidates (logit < -2),
   we refuse rather than send weak context.

This satisfies the spec's "below 0.25 -> 'I don't have that in the
handbook.'" requirement while also catching false-positive dense hits.

## 8. Validation Snapshot

The 6 required test queries plus 1 out-of-domain probe are run by
`scripts/evaluate.py`, which auto-populates `prompts.txt` (the deliverable
prompt log) and `outputs/eval.json` (structured results).

| # | Question | Expected Pages |
|---|---|---|
| 1 | Headings, fonts, sizes | 39 |
| 2 | Margins / spacing | 39 |
| 3 | Development FYP chapters | 41 |
| 4 | R&D FYP chapters | 42-43 |
| 5 | Ibid. / op. cit. usage | 38 |
| 6 | Executive Summary / Abstract | 36 |
| OOD | "Cafeteria menu at FAST?" | -> *refusal* |

## 9. Screenshots

*(Two screenshots inserted by `scripts/make_report_pdf.py` at render time, if
present in `outputs/screenshots/`.)*
