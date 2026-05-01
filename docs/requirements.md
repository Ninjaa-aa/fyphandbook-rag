# Generative AI - Assignment #3: RAG FYP Handbook Assistant

**Submission Date:** 2nd May, 2026  
**Group Size:** 2 Team members  

## Project Overview
Build a text-only Retrieval-Augmented Generation (RAG) pipeline that answers students' questions about the FAST-NUCES Final Year Project (FYP) process. The system uses the uploaded *BS Final Year Project Handbook 2023* as its exclusive knowledge base.

## System Architecture & Creation Process

### 1. Load & Chunk
- Extract text per page from the single PDF corpus (no images/OCR needed unless tables fail to parse cleanly).
- Preserve page numbers and store them as metadata.
- Select appropriate chunk size and overlap size.

### 2. Embed & Index
- Create embeddings for text chunks using a suitable LLM embedding model.
- Build a vector database index and persist it to disk locally.

### 3. Retrieve
- Embed the user query.
- Retrieve the most relevant chunks utilizing cosine similarity.
- Display retrieved chunks (and page references) in a collapsible or debug block.

### 4. Prompt Template
```text
You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like "(p. X)". If unsure, say you don't know.
Question: {user_question}
Context:
{top_chunks_text}
```

### 5. Precautions & Guardrails
- **Relevance Threshold:** If top-k similarity is below the defined threshold (e.g., `< 0.25`), the system must reply: *"I don't have that in the handbook."*
- Disallow questions that fall outside the handbook's scope.

### 6. Minimal UI
- **Framework:** CLI or a minimal Streamlit application.
- **Interface Components:**
  - One input box
  - One "Ask" button
  - An Answer panel
  - A collapsible "Sources (page refs)" list

---

## Validation Questions
Use the following test queries to validate your pipeline. The answers generated should successfully paraphrase the handbook content and accurately cite the correct pages.

1. **"What headings, fonts, and sizes are required in the FYP report?"**
   *(Expected points: Times New Roman 11 for body; Arial for headings; specific sizes for Title/H1/H2/H3. Include spacing rules if asked. Cite formatting specs.)*
2. **"What margins and spacing do we use?"**
   *(Expected points: Top 1.5", Bottom 1.0", Left 2.0", Right 1.0"; line spacing 1.5; paragraph spacing 6 pt. Cite formatting section.)*
3. **"What are the required chapters/sections of a Development FYP report?"**
   *(Expected points: Intro, research on existing products, vision, SRS, iterations, implementation details, user manual, references, appendices. Cite "Development FYP Report Format".)*
4. **"What are the required chapters of an R&D-based FYP report?"**
   *(Expected points: Intro, literature review, proposed approach, implementation, validation & testing, results & discussion, conclusions & future work, references, appendices. Cite R&D format pages.)*
5. **"How should endnotes like ‘Ibid.’ and ‘op. cit.’ be used?"**
   *(Expected points: Short guidance on endnotes, "Ibid." for immediate repeated citation, "op. cit." when source reappears after interruptions. Cite footnotes paragraph.)*
6. **"What goes into the Executive Summary and Abstract?"**
   *(Expected points: Abstract needs 50–125 words, separate page. Executive summary needs 1–2 pages overview. Cite report prelims section.)*

---

## Deliverables

1. **Code (`.ipynb` or `.py`)**
   - `ingest.py` (Parse & index script)
   - `ask.py` (Retrieve & answer script) OR `app.py` (Streamlit interface)
2. **Documentation Note (1–2 page PDF)**
   - Notes explaining your chunking settings, model used, `k` value, and similarity threshold.
   - Screenshots of two example Q&A pairs demonstrating page-cited answers.
3. **Prompt Log (`.txt`)**
   - A text file containing your exact system and inference prompts.