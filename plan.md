# RAG: FYP Handbook Assistant — Implementation Plan

## Overview

A Retrieval-Augmented Generation (RAG) pipeline with three phases: **Ingest → Retrieve → Answer**, plus a Streamlit UI. Every component in this plan is **100% free** — no paid API keys required.

---

## Tech Stack (All Free)

| Component | Choice | Why |
|---|---|---|
| PDF parser | `PyMuPDF (fitz)` | Best page-level text extraction with native page number metadata |
| Chunker | `RecursiveCharacterTextSplitter` (LangChain) | Respects sentence boundaries across page breaks |
| Chunk size | 500 chars / 100 overlap | Handbook has short dense sections; overlap prevents cross-page truncation |
| Embedding model | `BAAI/bge-m3` (local, sentence-transformers) | Free, runs locally, no rate limits, strong MTEB score |
| Vector store | `ChromaDB` (local embedded) | No server/Docker needed, native metadata support, persists to disk |
| Top-k | 5 | Enough context, not too noisy |
| Similarity threshold | 0.25 cosine similarity | Rejects clearly off-topic queries without invoking LLM |
| LLM | `Llama 3.3 70B Instruct` via **Groq API** (free tier) | 300+ tok/sec, 92% boundedness rate, best citation formatting, genuinely free |
| UI | Streamlit | Minimal setup, meets all assignment UI requirements |

> **Why Groq over Google AI Studio?** Gemini 2.5 Flash free tier caps at 10 RPM and 250 RPD — you'll hit 429 errors constantly during debugging. Groq's free tier is far more generous for iterative development.

---

## Phase 1: Project Structure

```
fyp-rag/
├── ingest.py          # PDF parsing, chunking, embedding, ChromaDB indexing
├── app.py             # Streamlit UI (retrieval + answer generation)
├── vectordb/          # Persisted ChromaDB index (auto-created on first run)
├── handbook.pdf       # BS FYP Handbook 2023
├── prompts.txt        # Prompt log (deliverable)
├── report.pdf         # 1-2 page architectural note (deliverable)
└── requirements.txt
```

---

## Phase 2: Dependencies

```txt
# requirements.txt
pymupdf                  # PDF parsing (import as fitz)
langchain
langchain-community
sentence-transformers    # BGE-M3 embedding model (runs locally)
chromadb                 # Local vector database
streamlit                # UI
groq                     # Groq API client (free LLM inference)
numpy
```

Install with:
```bash
pip install pymupdf langchain langchain-community sentence-transformers chromadb streamlit groq numpy
```

---

## Phase 3: API Key Setup

Only one API key needed — Groq (free):

1. Sign up at [console.groq.com](https://console.groq.com) — no credit card required
2. Create an API key under "API Keys"
3. Set as environment variable:
   ```bash
   export GROQ_API_KEY="your_key_here"
   ```
   Or create a `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

No embedding API key needed — BGE-M3 runs fully locally via sentence-transformers.

---

## Phase 4: ingest.py — Step by Step

### Step 4.1: Load & Extract Text Per Page

Use `PyMuPDF` to extract text **page by page**, preserving page numbers as metadata.

```python
import fitz  # PyMuPDF

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,  # 1-indexed to match handbook
                "text": text
            })
    return pages
```

> **Why PyMuPDF over pdfplumber?** PyMuPDF natively preserves bounding box coordinates and page indices, making it the most reliable option for maintaining the `page_number` metadata your citations depend on.

### Step 4.2: Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = []
    for page_data in pages:
        splits = splitter.split_text(page_data["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    "page": page_data["page"],
                    "chunk_id": f"p{page_data['page']}_c{i}"
                }
            })
    return chunks
```

**Settings rationale:**
- `chunk_size=500`: Captures 2-3 paragraphs — specific enough for citations, large enough for context
- `chunk_overlap=100`: Prevents sentence truncation at page/chunk boundaries (critical for pages 42-43 R&D format)
- Expected output: ~150-250 chunks from the handbook

### Step 4.3: Embed & Index with ChromaDB

```python
from sentence_transformers import SentenceTransformer
import chromadb

def build_index(chunks, db_path="vectordb"):
    # Load BGE-M3 locally — downloads once, cached afterward
    model = SentenceTransformer("BAAI/bge-m3")
    
    # Initialize ChromaDB with disk persistence
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="fyp_handbook",
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["metadata"]["chunk_id"] for c in chunks]
    
    # Embed all chunks
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    
    # Store in ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Indexed {len(chunks)} chunks into ChromaDB at '{db_path}/'")
```

### Full ingest.py Flow

```
handbook.pdf → PyMuPDF (page-by-page) → RecursiveCharacterTextSplitter (500/100)
→ BGE-M3 embeddings (local) → ChromaDB (persisted to vectordb/)
```

Run once before using the app:
```bash
python ingest.py
```

---

## Phase 5: app.py — Retrieval + Generation + UI

### Step 5.1: Load Index & Embedding Model

```python
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import streamlit as st

@st.cache_resource
def load_resources():
    model = SentenceTransformer("BAAI/bge-m3")
    client = chromadb.PersistentClient(path="vectordb")
    collection = client.get_collection("fyp_handbook")
    groq_client = Groq()  # Reads GROQ_API_KEY from environment
    return model, collection, groq_client
```

### Step 5.2: Retrieval with Threshold Check

```python
def retrieve(query, model, collection, k=5, threshold=0.25):
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    # ChromaDB cosine distance → similarity: similarity = 1 - distance
    distances = results["distances"][0]
    similarities = [1 - d for d in distances]
    
    # Threshold check
    if not similarities or max(similarities) < threshold:
        return None, []
    
    chunks = []
    for doc, meta, sim in zip(results["documents"][0], results["metadatas"][0], similarities):
        chunks.append({
            "text": doc,
            "page": meta["page"],
            "similarity": sim
        })
    
    return chunks, similarities
```

### Step 5.3: Prompt Construction

```python
SYSTEM_PROMPT = """You are a handbook assistant for FAST-NUCES students.
Answer ONLY from the context provided below.
Cite page numbers exactly like this: (p. X) or (p. X-Y) for ranges.
If the answer is not in the context, say exactly: "I don't have that in the handbook."
Do not use any external knowledge or make up information."""

def build_prompt(user_question, chunks):
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[p. {chunk['page']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    
    return f"Question: {user_question}\n\nContext:\n{context}"
```

### Step 5.4: LLM Generation via Groq

```python
def generate_answer(user_question, chunks, groq_client):
    user_prompt = build_prompt(user_question, chunks)
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free on Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,   # Low temperature for deterministic, factual output
        max_tokens=1024
    )
    
    return response.choices[0].message.content
```

> **Temperature 0.1:** Keeps the model close to the retrieved context, minimizing creative deviation from handbook text.

### Step 5.5: Streamlit UI

```python
def main():
    st.title("FYP Handbook Assistant")
    st.caption("FAST-NUCES BS FYP Handbook 2023 — ask any question about the FYP process.")
    
    model, collection, groq_client = load_resources()
    
    question = st.text_input("Ask a question about the FYP process...")
    
    if st.button("Ask") and question.strip():
        with st.spinner("Searching handbook..."):
            chunks, similarities = retrieve(question, model, collection)
        
        if chunks is None:
            st.warning("I don't have that in the handbook.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks, groq_client)
            
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("Sources (page refs)"):
                for chunk in chunks:
                    st.markdown(f"**Page {chunk['page']}** (similarity: {chunk['similarity']:.3f})")
                    st.caption(chunk["text"][:300] + "...")
                    st.divider()

if __name__ == "__main__":
    main()
```

### Full app.py Flow

```
User query → BGE-M3 embed → ChromaDB cosine search → threshold check
→ [p. X] context injection → Llama 3.3 70B (Groq) → answer + sources expander
```

---

## Phase 6: Prompt Log (prompts.txt)

Maintain this file during testing. It is a required deliverable.

**Template:**
```
=== SYSTEM PROMPT ===
You are a handbook assistant for FAST-NUCES students.
Answer ONLY from the context provided below.
Cite page numbers exactly like this: (p. X) or (p. X-Y) for ranges.
If the answer is not in the context, say exactly: "I don't have that in the handbook."
Do not use any external knowledge or make up information.

=== INFERENCE PROMPT (Example: Query 1) ===
Question: What headings, fonts, and sizes are required in the FYP report?

Context:
[p. 39]
Fonts, Type Styles Font Size = 11 (Normal Text) Font = Times New Roman
Title = 26 bold (Times New Roman) ...

=== RAW RESPONSE ===
According to the formatting guidelines, body text must use Times New Roman size 11...
(p. 39)

=== [Repeat for each test query] ===
```

---

## Phase 7: Validation Test Cases

Run all 6 queries and verify outputs before screenshotting for the report.

| # | Question | Expected Content | Expected Page |
|---|----------|-----------------|---------------|
| 1 | What headings, fonts, and sizes are required? | Times New Roman 11 body; Arial H1/H2/H3; Title 26pt | p. 39 |
| 2 | What margins and spacing do we use? | Top 1.5", Bottom 1.0", Left 2.0", Right 1.0"; 1.5 line spacing; 6pt paragraph | p. 39 |
| 3 | Required chapters of a Development FYP? | Intro, SRS, iteration plan, implementation, user manual, references, appendices | p. 41 |
| 4 | Required chapters of an R&D FYP? | Lit review, proposed approach, validation & testing, results & discussion, conclusions | p. 42-43 |
| 5 | How to use Ibid. and op. cit.? | Ibid. for immediate repeat; op. cit. after interrupting sources + page number | p. 38 |
| 6 | What goes in the Executive Summary and Abstract? | Abstract: 50-125 words, separate page; Exec summary: 1-2 pages, reference during presentations | p. 36 |

**Out-of-scope test (threshold check):**
- Query: "What is the cafeteria menu at FAST?"
- Expected: "I don't have that in the handbook." (similarity will be < 0.25)

---

## Phase 8: Report Note Content (1-2 pages PDF)

Document these decisions clearly:

**Chunking Settings:**
- Parser: PyMuPDF (fitz) — page-by-page extraction, metadata preserved
- Chunk size: 500 characters
- Overlap: 100 characters
- Rationale: handbook has short dense sections; overlap prevents content loss at page boundaries

**Embedding Model:**
- Model: `BAAI/bge-m3` via sentence-transformers
- Runs locally — no API key, no rate limits
- Output dimensions: 1024

**Retrieval Settings:**
- Vector DB: ChromaDB (local embedded, cosine similarity)
- k = 5 chunks retrieved per query
- Similarity threshold = 0.25 (below this → refuse, no LLM call)

**LLM:**
- Model: `llama-3.3-70b-versatile` via Groq API (free tier)
- Temperature: 0.1
- System prompt enforces context-only answers and (p. X) citations

**Screenshots required:** At least 2 of the 6 validation queries showing answer panel + page citations visible + sources expander open.

---

## Recommended Implementation Order

1. Set up project folder and install dependencies
2. Get Groq API key (5 minutes — no card needed)
3. Write and run `ingest.py` → verify chunks print correctly → confirm `vectordb/` folder appears
4. Write `app.py` → test CLI retrieval first before adding Streamlit UI
5. Run all 6 validation queries → check citations appear correctly
6. Screenshot 2+ results for the report
7. Fill in `prompts.txt` with actual system prompt + 2-3 full inference examples
8. Write `report.pdf` (1-2 pages documenting the above settings + screenshots)

---

## Common Issues & Fixes

| Issue | Fix |
|---|---|
| BGE-M3 slow on first run | Model downloads (~550MB) once, then cached at `~/.cache/huggingface/` |
| ChromaDB cosine vs dot product | Ensure collection created with `{"hnsw:space": "cosine"}` |
| Page numbers off by one | PyMuPDF is 0-indexed internally — always add `+1` when storing metadata |
| Groq rate limit (429) | Free tier allows 30 RPM and 14,400 RPD — more than enough for testing |
| Low similarity on valid questions | Try rephrasing query to match handbook terminology; check chunk size isn't too small |
| Chunks missing cross-page content | Increase overlap to 150 if R&D chapter content spans pages 42-43 incorrectly |