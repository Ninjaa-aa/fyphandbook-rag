# Documentation Note - Assignment #3: RAG FYP Handbook Assistant

**Course:** Generative AI  
**Corpus:** FAST-NUCES *BS Final Year Project Handbook 2023*  
**Interface:** Streamlit app (`app.py`) and optional CLI (`ask.py`)

## 1. System Overview

This project implements a text-only Retrieval-Augmented Generation pipeline that answers questions only from the uploaded FYP handbook. The one-time ingest flow reads the PDF, preserves handbook page numbers, creates page-cited chunks, embeds them, and persists both dense and sparse indexes locally.

```text
PDF -> PyMuPDF pages/spans -> heading-aware chunks + page fallback chunks
    -> BGE-M3 embeddings + ChromaDB cosine index
    -> BM25 index + chunk lookup
    -> dense/BM25 retrieval -> RRF -> refusal gate -> reranker -> MMR
    -> prompt with page-labelled context -> Groq Llama 3.3 70B answer
```

The main files are `ingest.py` for indexing, `app.py` for the Streamlit UI, `ask.py` for CLI questions, `src/fyp_rag/` for the reusable pipeline, `scripts/evaluate.py` for validation, `prompts.txt` for the prompt log, and `outputs/eval.json` for structured evaluation results.

## 2. Chunking, Models, and Indexes

- **PDF loading:** `src/fyp_rag/pdf_loader.py` uses PyMuPDF to extract page text plus font/span metadata. Printed page numbers are detected and stored for citations.
- **Chunking:** `src/fyp_rag/chunker.py` uses heading-aware sectioning, merges very small sections, splits long sections with a recursive text splitter, and adds page-level fallback chunks to preserve lists/tables.
- **Chunk settings:** `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`, `MIN_CHUNK_CHARS = 120`, and `MAX_SECTION_CHARS_BEFORE_SPLIT = 800`.
- **Embedding model:** `BAAI/bge-m3`, run locally through `sentence-transformers`.
- **Vector database:** ChromaDB persisted in `data/chroma` with cosine similarity.
- **Sparse index:** BM25 from `rank_bm25`, persisted as `data/bm25.pkl`.
- **Chunk lookup:** `data/chunks.pkl` stores `chunk_id -> {text, metadata}` for fast source hydration.

## 3. Retrieval and Guardrails

Retrieval combines semantic and lexical search so both paraphrased questions and exact terms like `Ibid.` or `op. cit.` can be found. Dense search returns `TOP_K_DENSE = 30`, BM25 returns `TOP_K_BM25 = 30`, Reciprocal Rank Fusion keeps `TOP_K_FUSED = 30`, and the default final context is `TOP_K_RERANK = 6` chunks. Chapter/report-format questions use adaptive settings that can keep up to 10 final chunks because those answers require longer lists.

The assignment-required refusal gate is implemented before generation: if the maximum dense cosine similarity is below `SIMILARITY_THRESHOLD = 0.25`, the system returns exactly `I don't have that in the handbook.` The reranker also filters weak candidates with `RERANKER_MIN_SCORE = -2.0`, and MMR uses `MMR_LAMBDA = 0.7` to reduce duplicate context.

## 4. Prompt and UI Alignment

The system prompt says the assistant must answer only from provided context, cite page numbers as `(p. X)` or `(p. X-Y)`, and use the exact refusal message when the answer is not present. The user prompt labels each context block with its page and section before asking the LLM to answer from that context only.

The Streamlit UI satisfies the minimal interface requirement with one question input, an **Ask** button, an answer panel, and a **Sources (page refs)** expander. The sources panel shows page labels, section names, similarity scores, reranker scores, RRF scores, and chunk previews. A debug expander also shows dense hits, BM25 hits, and RRF candidates.

## 5. Validation Evidence

`scripts/evaluate.py` runs the six required validation questions and one out-of-domain probe. It writes `prompts.txt` with the exact system/inference prompts and `outputs/eval.json` with answers, retrieved chunks, scores, pages, and refusal flags.

| Query | Evidence from current evaluation |
| --- | --- |
| Headings, fonts, and sizes | Answer cites page 39 for Times New Roman body text, title/sub-title sizes, and Arial heading sizes. |
| Margins and spacing | Answer cites pages 39-40 for top, bottom, left, right margins, line spacing, and paragraph spacing. |
| Development FYP report sections | Answer retrieves page 41 and lists development report sections/subsections. |
| R&D-based FYP report chapters | Answer retrieves pages 42-43 for R&D report format chapters, references, and appendices. |
| `Ibid.` and `op. cit.` endnotes | Answer cites page 38 for repeated-source endnote rules. |
| Executive Summary and Abstract | Answer cites page 36 for abstract length/content and executive summary purpose/length. |
| Out-of-domain probe | The cafeteria-menu question returns `I don't have that in the handbook.` |

## 6. Example Q&A Pairs and Screenshots

**Example 1:** For "What margins and spacing do we use?", the system answers with the required margins (`Top = 1.5"`, `Bottom = 1.0"`, `Left = 2.0"`, `Right = 1.0"`), line spacing `1.5`, and paragraph spacing `6 pts`, with page citations from pages 39-40.

**Example 2:** For "How should endnotes like 'Ibid.' and 'op. cit.' be used?", the system explains that `Ibid.` substitutes for an immediately repeated source, page numbers are added when the cited page changes, and `op. cit.` is used when the same source reappears after interrupting sources, citing page 38.

Screenshots for the final PDF should show the Streamlit answer panel and expanded **Sources (page refs)** for two validation questions. Place them in `outputs/screenshots/` before running `uv run python scripts/make_report_pdf.py`; the PDF builder appends available screenshots automatically.
