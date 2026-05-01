"""BGE-M3 embedding wrapper.

Loads `BAAI/bge-m3` once per process and exposes batch encoding utilities.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL
from .logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Process-wide singleton for BGE-M3."""
    log.info("Loading embedding model %s ...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model ready (dim=%d).", model.get_sentence_embedding_dimension())
    return model


def embed_texts(texts: list[str], *, normalize: bool = True, batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts. Returns float32 ndarray of shape (n, d)."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 8,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns float32 vector of shape (d,)."""
    return embed_texts([text])[0]
