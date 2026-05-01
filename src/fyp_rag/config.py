"""Central configuration for the FYP RAG pipeline.

All paths, model identifiers, and tunable hyper-parameters live here so the
rest of the codebase stays free of magic constants.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DOCS_DIR: Path = ROOT_DIR / "docs"
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORT_DIR: Path = ROOT_DIR / "report"

PDF_PATH: Path = DOCS_DIR / "3. FYP-Handbook-2023.pdf"

CHROMA_DIR: Path = DATA_DIR / "chroma"
BM25_PATH: Path = DATA_DIR / "bm25.pkl"
CHUNKS_PATH: Path = DATA_DIR / "chunks.pkl"

PROMPTS_LOG: Path = ROOT_DIR / "prompts.txt"
EVAL_RESULTS: Path = OUTPUTS_DIR / "eval.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "BAAI/bge-m3"
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
LLM_MODEL: str = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MIN_CHUNK_CHARS: int = 120
MAX_SECTION_CHARS_BEFORE_SPLIT: int = 800

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K_DENSE: int = 30
TOP_K_BM25: int = 30
TOP_K_FUSED: int = 30
TOP_K_RERANK: int = 6

RRF_K: int = 60
MMR_LAMBDA: float = 0.7

# Assignment-mandated refusal gate on dense cosine similarity.
SIMILARITY_THRESHOLD: float = 0.25
# Secondary guard on cross-encoder logit (filters chunks before LLM call).
RERANKER_MIN_SCORE: float = -2.0

# Adaptive settings for chapter/list style questions.
LIST_QUERY_KEYWORDS: tuple[str, ...] = (
    "required chapters",
    "chapters of",
    "chapter-wise",
    "report format",
    "development fyp",
    "r&d fyp",
    "r&d-based fyp",
)
LIST_QUERY_TOP_K_FUSED: int = 45
LIST_QUERY_TOP_K_FINAL: int = 10
LIST_QUERY_MMR_ENABLED: bool = False
LIST_QUERY_MMR_LAMBDA: float = 0.9
LIST_QUERY_RERANK_MIN_SCORE: float = -3.0
LIST_QUERY_ADJACENCY_WINDOW: int = 1
LIST_QUERY_FORMAT_BOOST: float = 0.003

# Long list chunks are often truncated at 512 tokens by default.
RERANKER_MAX_LENGTH: int = 1024

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

CHROMA_COLLECTION: str = "fyp_handbook"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 1024
LLM_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Refusal text (must match assignment spec exactly).
# ---------------------------------------------------------------------------

REFUSAL_MESSAGE: str = "I don't have that in the handbook."


def ensure_dirs() -> None:
    """Create runtime directories if they don't exist."""
    for d in (DATA_DIR, CHROMA_DIR, OUTPUTS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
