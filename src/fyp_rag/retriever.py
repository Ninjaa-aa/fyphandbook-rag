"""Hybrid retrieval: dense + BM25 -> RRF fusion -> threshold gate -> rerank -> MMR.

Returns a structured `RetrievalResult` so the pipeline / UI can show debug info.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
from chromadb.api.models.Collection import Collection

from .bm25_store import BM25Hit, BM25Index, query_bm25
from .config import (
    MMR_LAMBDA,
    LIST_QUERY_ADJACENCY_WINDOW,
    LIST_QUERY_FORMAT_BOOST,
    LIST_QUERY_KEYWORDS,
    LIST_QUERY_MMR_ENABLED,
    LIST_QUERY_MMR_LAMBDA,
    LIST_QUERY_RERANK_MIN_SCORE,
    LIST_QUERY_TOP_K_FINAL,
    LIST_QUERY_TOP_K_FUSED,
    RERANKER_MIN_SCORE,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K_BM25,
    TOP_K_DENSE,
    TOP_K_FUSED,
    TOP_K_RERANK,
)
from .embedder import embed_query
from .logger import get_logger
from .reranker import RerankHit, rerank
from .vector_store import DenseHit, fetch_embeddings, query_dense

log = get_logger(__name__)


@dataclass
class FinalChunk:
    chunk_id: str
    text: str
    page: int
    page_end: int
    section: str
    similarity: float       # dense cosine
    rerank_score: float
    rrf_score: float


@dataclass
class RetrievalResult:
    chunks: list[FinalChunk]
    refused: bool
    max_similarity: float
    dense_hits: list[DenseHit] = field(default_factory=list)
    bm25_hits: list[BM25Hit] = field(default_factory=list)
    rrf_ranking: list[tuple[str, float]] = field(default_factory=list)
    rerank_hits: list[RerankHit] = field(default_factory=list)


_CHUNK_SUFFIX_RE = re.compile(r"^(?P<prefix>.+_c)(?P<idx>\d+)(?:_\d+)?$")


def _is_list_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in LIST_QUERY_KEYWORDS)


def _is_format_candidate(text: str, metadata: dict) -> bool:
    if bool(metadata.get("is_format_chunk", False)):
        return True
    section = str(metadata.get("section", "")).lower()
    joined = f"{section} {text.lower()}"
    return any(
        key in joined
        for key in (
            "report format",
            "chapter",
            "fyp report contents",
            "software requirement specifications",
            "literature review",
            "proposed approach",
            "validation and testing",
            "results and discussion",
            "conclusions and future work",
            "user manual",
            "iteration plan",
        )
    )


def _rrf_fuse(
    dense_hits: list[DenseHit],
    bm25_hits: list[BM25Hit],
    *,
    k: int = RRF_K,
    top_n: int = TOP_K_FUSED,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_idx, h in enumerate(dense_hits):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank_idx)
    for rank_idx, h in enumerate(bm25_hits):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank_idx)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n]


def _boost_format_chunks(
    fused: list[tuple[str, float]],
    chunk_lookup: dict[str, dict],
    boost: float,
) -> list[tuple[str, float]]:
    boosted: list[tuple[str, float]] = []
    for cid, score in fused:
        info = chunk_lookup.get(cid, {})
        meta = info.get("metadata", {}) if isinstance(info, dict) else {}
        if bool(meta.get("is_format_chunk", False)):
            score += boost
        boosted.append((cid, score))
    return sorted(boosted, key=lambda x: -x[1])


def _adjacent_chunk_ids(chunk_id: str, window: int) -> list[str]:
    """Return neighboring chunk IDs sharing the same prefix (e.g., ..._c0)."""
    m = _CHUNK_SUFFIX_RE.match(chunk_id)
    if not m:
        return []
    prefix = m.group("prefix")
    idx = int(m.group("idx"))
    out: list[str] = []
    for delta in range(-window, window + 1):
        if delta == 0:
            continue
        out.append(f"{prefix}{idx + delta}")
    return out


def _expand_adjacency(
    selected_ids: list[str],
    chunk_lookup: dict[str, dict],
    *,
    allowed_ids: set[str] | None,
    window: int,
    limit: int,
) -> list[str]:
    if window <= 0:
        return selected_ids[:limit]

    out = list(selected_ids)
    seen = set(out)
    for cid in list(selected_ids):
        for neighbor in _adjacent_chunk_ids(cid, window):
            if neighbor in seen:
                continue
            if allowed_ids is not None and neighbor not in allowed_ids:
                continue
            if neighbor in chunk_lookup:
                out.append(neighbor)
                seen.add(neighbor)
            if len(out) >= limit:
                return out[:limit]
    return out[:limit]


def _mmr(
    candidate_ids: list[str],
    candidate_embs: dict[str, np.ndarray],
    query_emb: np.ndarray,
    *,
    k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[str]:
    """Maximum Marginal Relevance — picks `k` ids balancing relevance & diversity."""
    if not candidate_ids:
        return []

    qn = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    embs = {cid: e / (np.linalg.norm(e) + 1e-9) for cid, e in candidate_embs.items() if e is not None}
    pool = [cid for cid in candidate_ids if cid in embs]
    if not pool:
        return candidate_ids[:k]

    selected: list[str] = []
    while pool and len(selected) < k:
        best_id, best_score = None, -1e9
        for cid in pool:
            relevance = float(np.dot(qn, embs[cid]))
            if selected:
                redundancy = max(float(np.dot(embs[cid], embs[s])) for s in selected)
            else:
                redundancy = 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score, best_id = score, cid
        if best_id is None:
            break
        selected.append(best_id)
        pool.remove(best_id)
    return selected


def retrieve(
    query: str,
    *,
    collection: Collection,
    bm25_index: BM25Index,
    chunk_lookup: dict[str, dict],
    top_k_final: int = TOP_K_RERANK,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> RetrievalResult:
    """End-to-end hybrid retrieval pipeline.

    Args:
        chunk_lookup: id -> {text, metadata} map produced at index time.
    """
    query_emb = embed_query(query)
    list_query = _is_list_query(query)

    dense_hits = query_dense(collection, query_emb, k=TOP_K_DENSE)
    bm25_hits = query_bm25(bm25_index, query, k=TOP_K_BM25)

    max_sim = max((h.similarity for h in dense_hits), default=0.0)

    if max_sim < similarity_threshold:
        log.info(
            "Refusal: max dense cosine %.3f < threshold %.2f",
            max_sim,
            similarity_threshold,
        )
        return RetrievalResult(
            chunks=[],
            refused=True,
            max_similarity=max_sim,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
        )

    fused = _rrf_fuse(
        dense_hits,
        bm25_hits,
        top_n=LIST_QUERY_TOP_K_FUSED if list_query else TOP_K_FUSED,
    )
    if list_query:
        fused = _boost_format_chunks(fused, chunk_lookup, LIST_QUERY_FORMAT_BOOST)
    fused_ids = [cid for cid, _ in fused]

    candidates: list[tuple[str, str, dict]] = []
    for cid in fused_ids:
        info = chunk_lookup.get(cid)
        if info is None:
            continue
        candidates.append((cid, info["text"], info["metadata"]))

    if list_query and candidates:
        format_candidates = [c for c in candidates if _is_format_candidate(c[1], c[2])]
        # For chapter/report-format questions, prioritize true format chunks.
        # If enough exist, drop noisy generic pages entirely.
        if len(format_candidates) >= max(4, top_k_final):
            candidates = format_candidates
        else:
            # Otherwise keep all but rank format chunks first.
            candidates = format_candidates + [c for c in candidates if c not in format_candidates]

    rerank_hits = rerank(
        query,
        candidates,
        top_k=max((LIST_QUERY_TOP_K_FINAL if list_query else top_k_final) * 3, top_k_final),
        min_score=LIST_QUERY_RERANK_MIN_SCORE if list_query else RERANKER_MIN_SCORE,
    )
    if not rerank_hits:
        log.info("Reranker filtered all candidates; refusing.")
        return RetrievalResult(
            chunks=[],
            refused=True,
            max_similarity=max_sim,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            rrf_ranking=fused,
        )

    target_k = LIST_QUERY_TOP_K_FINAL if list_query else top_k_final
    rerank_ids = [h.chunk_id for h in rerank_hits]
    cand_embs = fetch_embeddings(collection, rerank_ids)
    if list_query and not LIST_QUERY_MMR_ENABLED:
        seed_k = max(1, target_k - (2 * LIST_QUERY_ADJACENCY_WINDOW))
        selected_ids = rerank_ids[:seed_k]
    else:
        selected_ids = _mmr(
            rerank_ids,
            cand_embs,
            query_emb,
            k=target_k,
            lambda_=LIST_QUERY_MMR_LAMBDA if list_query else MMR_LAMBDA,
        )
    mmr_ids = _expand_adjacency(
        selected_ids,
        chunk_lookup,
        allowed_ids=set(rerank_ids),
        window=LIST_QUERY_ADJACENCY_WINDOW if list_query else 0,
        limit=target_k,
    )

    rerank_by_id = {h.chunk_id: h for h in rerank_hits}
    sim_by_id = {h.chunk_id: h.similarity for h in dense_hits}
    rrf_by_id = dict(fused)

    final: list[FinalChunk] = []
    for cid in mmr_ids:
        rh = rerank_by_id[cid]
        meta = rh.metadata or {}
        final.append(
            FinalChunk(
                chunk_id=cid,
                text=rh.text,
                page=int(meta.get("page", 0)),
                page_end=int(meta.get("page_end", meta.get("page", 0))),
                section=str(meta.get("section", "")),
                similarity=float(sim_by_id.get(cid, 0.0)),
                rerank_score=float(rh.rerank_score),
                rrf_score=float(rrf_by_id.get(cid, 0.0)),
            )
        )

    return RetrievalResult(
        chunks=final,
        refused=False,
        max_similarity=max_sim,
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        rrf_ranking=fused,
        rerank_hits=rerank_hits,
    )
