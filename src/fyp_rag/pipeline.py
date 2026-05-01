"""End-to-end RAG pipeline: load resources once, answer queries on demand."""

from __future__ import annotations

import pickle
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

    return PipelineAnswer(
        question=question,
        answer=raw,
        refused=False,
        retrieval=retrieval,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
    )
