"""End-to-end RAG pipeline: load resources once, answer queries on demand."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

from .bm25_store import BM25Index, load_index as load_bm25
from .config import (
    BM25_PATH,
    CHUNKS_PATH,
    REFUSAL_MESSAGE,
    SIMILARITY_THRESHOLD,
    TOP_K_RERANK,
)
from .llm import generate_answer
from .logger import get_logger
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .retriever import RetrievalResult, retrieve
from .vector_store import get_collection

log = get_logger(__name__)

_DEV_MARKER = "development fyp report format"
_RND_MARKER = "r&d-based fyp report format"
_CHAPTER_RE = re.compile(r"chapter\s+\d+\.\s*[a-z][a-z0-9/&(),\- ]{2,120}", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"\b\d+(?:\.\d+){0,2}\.?\s+[a-z][a-z0-9#&()/,\- ]{2,120}", re.IGNORECASE)


@dataclass
class PipelineAnswer:
    question: str
    answer: str
    refused: bool
    retrieval: RetrievalResult
    user_prompt: str
    system_prompt: str


@dataclass
class Resources:
    collection: object  # chromadb collection
    bm25: BM25Index
    chunk_lookup: dict[str, dict]


@lru_cache(maxsize=1)
def load_resources() -> Resources:
    """Load Chroma collection + BM25 + chunk-id -> {text, metadata} lookup."""
    log.info("Loading resources (Chroma + BM25 + chunk lookup) ...")
    collection = get_collection()
    bm25 = load_bm25(BM25_PATH)

    if not CHUNKS_PATH.exists():
        raise RuntimeError(
            f"Chunk lookup not found at {CHUNKS_PATH}. Run ingest.py first."
        )
    with CHUNKS_PATH.open("rb") as f:
        chunk_lookup: dict[str, dict] = pickle.load(f)

    log.info(
        "Resources loaded: %d chunks indexed, BM25 corpus size %d.",
        collection.count(),
        len(bm25.chunk_ids),
    )
    return Resources(collection=collection, bm25=bm25, chunk_lookup=chunk_lookup)


def _question_type(question: str) -> str | None:
    q = question.lower()
    if "required chapters" not in q and "report format" not in q:
        return None
    if "r&d" in q or "r&d-based" in q:
        return "rnd"
    if "development" in q:
        return "development"
    return None


def _normalize_item(item: str) -> str:
    item = re.sub(r"\s+", " ", item).strip()
    # Remove accidental duplicate numbering fragments from OCR/chunk stitching.
    item = re.sub(r"^\d+\.\s*\d+\.\s*", "", item)
    item = re.sub(r"^\d+\.\s*", "", item)
    item = re.sub(r"\s+\d+$", "", item)
    item = re.sub(r"(?<=\D)\s+\d+\s*(?=\()", " ", item)
    item = item.rstrip(".,;:")
    return item


def _collect_items_from_chunks(question_type: str, retrieval: RetrievalResult) -> list[tuple[str, int]]:
    """Extract list items directly from retrieved chunks for chapter-list queries."""
    if not retrieval.chunks:
        return []

    page_set: set[int] = set()
    for ch in retrieval.chunks:
        txt = ch.text.lower()
        if question_type == "development" and _DEV_MARKER in txt:
            page_set.add(ch.page)
        if question_type == "rnd" and _RND_MARKER in txt:
            page_set.add(ch.page)
            page_set.add(ch.page + 1)

    if not page_set:
        # Fallback: use pages present in top retrieval chunks.
        page_set = {ch.page for ch in retrieval.chunks[:3]}

    raw_items: list[tuple[str, int]] = []
    for ch in retrieval.chunks:
        if ch.page not in page_set:
            continue
        text = " ".join(ch.text.splitlines())
        if question_type == "development":
            for m in _NUMBERED_RE.finditer(text):
                item = _normalize_item(m.group(0))
                if item.lower().startswith(("41 ", "42 ", "43 ")):
                    continue
                raw_items.append((item, ch.page))
        else:
            for m in _CHAPTER_RE.finditer(text):
                raw_items.append((_normalize_item(m.group(0)), ch.page))
            # References/Appendices are required tail sections in R&D format.
            if "references" in text.lower():
                raw_items.append(("References", ch.page))
            if "appendices" in text.lower():
                raw_items.append(("Appendices", ch.page))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    ordered: list[tuple[str, int]] = []
    for item, page in raw_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append((item, page))
    return ordered


def _format_item_for_output(question_type: str, item: str) -> str:
    """Render clean item labels without inner numbering duplication."""
    normalized = _normalize_item(item)
    if question_type == "rnd":
        return normalized

    # Development list in handbook is mostly section names/subsection names.
    # Strip deep decimal prefixes (e.g., "3.1.") for cleaner output list.
    normalized = re.sub(r"^\d+(?:\.\d+){1,2}\.?\s*", "", normalized)
    # Keep single top-level chapter number if present (e.g., "8. Implementation Details").
    normalized = re.sub(r"^\d+\.\s+", "", normalized)
    return normalized.strip()


def _build_deterministic_chapter_answer(question_type: str, retrieval: RetrievalResult) -> str | None:
    items = _collect_items_from_chunks(question_type, retrieval)
    min_expected = 8 if question_type == "development" else 5
    if len(items) < min_expected:
        return None

    title = "Required Chapters of a Development FYP" if question_type == "development" else "Required Chapters of an R&D FYP"
    lines = [title]
    for idx, (item, page) in enumerate(items, start=1):
        clean_item = _format_item_for_output(question_type, item)
        if not clean_item:
            continue
        lines.append(f"{idx}. {clean_item} (p. {page})")
    return "\n".join(lines)


def answer(
    question: str,
    *,
    resources: Resources | None = None,
    top_k_final: int = TOP_K_RERANK,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    call_llm: bool = True,
) -> PipelineAnswer:
    """Answer a single question. If `call_llm=False`, returns retrieval only."""
    if resources is None:
        resources = load_resources()

    retrieval = retrieve(
        question,
        collection=resources.collection,
        bm25_index=resources.bm25,
        chunk_lookup=resources.chunk_lookup,
        top_k_final=top_k_final,
        similarity_threshold=similarity_threshold,
    )

    if retrieval.refused:
        return PipelineAnswer(
            question=question,
            answer=REFUSAL_MESSAGE,
            refused=True,
            retrieval=retrieval,
            user_prompt="",
            system_prompt=SYSTEM_PROMPT,
        )

    user_prompt = build_user_prompt(question, retrieval.chunks)
    qtype = _question_type(question)
    if qtype is not None:
        deterministic = _build_deterministic_chapter_answer(qtype, retrieval)
        if deterministic:
            return PipelineAnswer(
                question=question,
                answer=deterministic,
                refused=False,
                retrieval=retrieval,
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

    if not call_llm:
        return PipelineAnswer(
            question=question,
            answer="",
            refused=False,
            retrieval=retrieval,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

    raw = generate_answer(SYSTEM_PROMPT, user_prompt)
    final_refused = raw.strip() == REFUSAL_MESSAGE

    return PipelineAnswer(
        question=question,
        answer=raw,
        refused=final_refused,
        retrieval=retrieval,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
    )
