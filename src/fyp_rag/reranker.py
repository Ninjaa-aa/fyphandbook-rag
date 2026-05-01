"""Cross-encoder reranker (BGE-reranker-v2-m3)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class RerankHit:
    chunk_id: str
    text: str
    metadata: dict
    rerank_score: float


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    log.info("Loading reranker %s ...", RERANKER_MODEL)
    model = CrossEncoder(RERANKER_MODEL, max_length=512)
    log.info("Reranker ready.")
    return model


def rerank(
    query: str,
    candidates: list[tuple[str, str, dict]],
    *,
    top_k: int,
    min_score: float | None = None,
) -> list[RerankHit]:
    """Score (query, candidate) pairs and return the top-k by logit score.

    Args:
        candidates: list of `(chunk_id, text, metadata)` tuples (already de-duped).
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c[1]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    enriched: list[RerankHit] = [
        RerankHit(chunk_id=cid, text=text, metadata=meta, rerank_score=float(s))
        for (cid, text, meta), s in zip(candidates, scores)
    ]

    if min_score is not None:
        enriched = [h for h in enriched if h.rerank_score >= min_score]

    enriched.sort(key=lambda h: h.rerank_score, reverse=True)
    return enriched[:top_k]
