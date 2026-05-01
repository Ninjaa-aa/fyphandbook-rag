"""ingest.py — One-time index builder.

Pipeline:
    PDF -> heading-aware chunks -> BGE-M3 embeddings -> ChromaDB persist
                                                    \\-> BM25 (pickled)
                                                    \\-> chunks.pkl (id -> {text, metadata})

Run with:
    uv run python ingest.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Make `src/` importable when this file lives at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fyp_rag.bm25_store import build_index as build_bm25, save_index as save_bm25
from fyp_rag.chunker import chunk_pages
from fyp_rag.config import (
    BM25_PATH,
    CHROMA_DIR,
    CHUNKS_PATH,
    PDF_PATH,
    ensure_dirs,
)
from fyp_rag.embedder import embed_texts
from fyp_rag.logger import get_logger
from fyp_rag.pdf_loader import extract_pages
from fyp_rag.vector_store import add_chunks, reset_collection

log = get_logger(__name__)


def main(pdf: Path = PDF_PATH) -> None:
    ensure_dirs()

    log.info("Step 1/4: extract pages from %s", pdf)
    pages, body_size = extract_pages(pdf)

    log.info("Step 2/4: build heading-aware chunks")
    chunks = chunk_pages(pages, body_size)
    if not chunks:
        raise RuntimeError("No chunks produced. Check PDF content / chunker heuristics.")

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for cid in ids:
            seen[cid] = seen.get(cid, -1) + 1
            deduped.append(cid if seen[cid] == 0 else f"{cid}_{seen[cid]}")
        ids = deduped
        for c, new_id in zip(chunks, ids):
            c.chunk_id = new_id

    texts = [c.text for c in chunks]
    metadatas = [
        {
            "page": c.page,
            "page_end": c.page_end,
            "section": c.section,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]

    log.info("Step 3/4: embed %d chunks with BGE-M3 (this is the slow step)", len(chunks))
    embeddings = embed_texts(texts, normalize=True, batch_size=32)

    log.info("Step 4/4: persist Chroma + BM25 + chunk lookup")
    collection = reset_collection(path=CHROMA_DIR)
    add_chunks(
        collection,
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    bm25_index = build_bm25(ids, texts)
    save_bm25(bm25_index, BM25_PATH)

    chunk_lookup = {
        cid: {"text": text, "metadata": meta}
        for cid, text, meta in zip(ids, texts, metadatas)
    }
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("wb") as f:
        pickle.dump(chunk_lookup, f)
    log.info("Saved chunk lookup to %s (%d entries)", CHUNKS_PATH, len(chunk_lookup))

    log.info("Ingest complete.")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build the FYP handbook RAG index.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Path to the handbook PDF.")
    args = parser.parse_args()
    main(args.pdf)


if __name__ == "__main__":
    cli()
