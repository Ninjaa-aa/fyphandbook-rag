"""ChromaDB persistent vector store wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from .config import CHROMA_COLLECTION, CHROMA_DIR
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class DenseHit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float       # cosine similarity in [-1, 1] (typically [0, 1] here)


def _client(path: Path | str = CHROMA_DIR) -> chromadb.api.ClientAPI:
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def reset_collection(path: Path | str = CHROMA_DIR, name: str = CHROMA_COLLECTION) -> Collection:
    """Drop and recreate the collection. Use during full reindex."""
    client = _client(path)
    try:
        client.delete_collection(name)
        log.info("Deleted existing Chroma collection '%s'.", name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("Created Chroma collection '%s' (cosine).", name)
    return collection


def get_collection(path: Path | str = CHROMA_DIR, name: str = CHROMA_COLLECTION) -> Collection:
    """Open existing collection (raises if missing)."""
    client = _client(path)
    return client.get_collection(name=name)


def add_chunks(
    collection: Collection,
    *,
    ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
    batch_size: int = 256,
) -> None:
    n = len(ids)
    log.info("Indexing %d chunks into Chroma in batches of %d ...", n, batch_size)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
    log.info("Indexed %d chunks (collection size now: %d).", n, collection.count())


def query_dense(
    collection: Collection,
    query_embedding: np.ndarray,
    k: int,
) -> list[DenseHit]:
    """Cosine-similarity dense query. Returns hits sorted by similarity desc."""
    res = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    if not res["ids"] or not res["ids"][0]:
        return []

    hits: list[DenseHit] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        # Chroma returns cosine *distance* in [0, 2]; convert to similarity.
        similarity = 1.0 - float(dist)
        hits.append(DenseHit(chunk_id=cid, text=doc, metadata=meta, similarity=similarity))
    return hits


def fetch_embeddings(collection: Collection, ids: list[str]) -> dict[str, np.ndarray]:
    """Fetch stored embeddings by id (used for MMR diversity)."""
    if not ids:
        return {}
    res = collection.get(ids=ids, include=["embeddings"])
    out: dict[str, np.ndarray] = {}
    for cid, emb in zip(res["ids"], res["embeddings"]):
        out[cid] = np.asarray(emb, dtype=np.float32)
    return out
