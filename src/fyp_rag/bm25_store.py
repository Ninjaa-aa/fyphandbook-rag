"""Persistent BM25 index over the chunked corpus.

Tokenization is intentionally lightweight (lowercase + alpha-num word match)
so we don't drag in NLTK punkt as a hard runtime dep. NLTK stopwords are
optional; we ship a small built-in list and try NLTK opportunistically.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import BM25_PATH
from .logger import get_logger

log = get_logger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_BUILTIN_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of in on at to for with from by as is are was were be
    been being has have had do does did this that these those it its it's they them their
    we you your i me my mine ours us he him his she her hers what which who whom whose
    so not no nor too very can will just should now also more most some such only own
    same than then once here there when where why how about against between into through
    during before after above below up down out off over under again further than
    """.split()
)


def _stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords as _sw  # type: ignore[import-not-found]

        return frozenset(_sw.words("english"))
    except Exception:
        return _BUILTIN_STOPWORDS


_STOPWORDS = _stopwords()


def tokenize(text: str) -> list[str]:
    """Lowercase, alpha-num token splitter with stopword filter."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]
    tokens: list[list[str]]


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


def build_index(chunk_ids: list[str], texts: list[str]) -> BM25Index:
    log.info("Tokenizing %d chunks for BM25 ...", len(texts))
    toks = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(toks)
    log.info("BM25 corpus built (vocab approx = %d).", len({tok for tl in toks for tok in tl}))
    return BM25Index(bm25=bm25, chunk_ids=chunk_ids, tokens=toks)


def save_index(index: BM25Index, path: Path = BM25_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)
    log.info("Saved BM25 index to %s", path)


def load_index(path: Path = BM25_PATH) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)


def query_bm25(index: BM25Index, query: str, k: int) -> list[BM25Hit]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = index.bm25.get_scores(q_tokens)
    # argsort desc
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    hits = [BM25Hit(chunk_id=index.chunk_ids[i], score=float(scores[i])) for i in order if scores[i] > 0.0]
    return hits
