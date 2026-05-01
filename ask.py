"""ask.py — Minimal CLI entry point for one-off queries.

Usage:
    uv run python ask.py "What margins and spacing do we use?"
    uv run python ask.py --debug "Required chapters of an R&D FYP?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fyp_rag.pipeline import answer  # noqa: E402


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ask the FYP Handbook RAG.")
    parser.add_argument("question", nargs="+", help="Your question (quote it).")
    parser.add_argument("--debug", action="store_true", help="Print retrieval debug info.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    result = answer(question)

    print("\n=== Question ===")
    print(question)
    print("\n=== Answer ===")
    print(result.answer)

    if not result.refused:
        print("\n=== Sources ===")
        for ch in result.retrieval.chunks:
            page_label = (
                f"p. {ch.page}-{ch.page_end}" if ch.page_end != ch.page else f"p. {ch.page}"
            )
            print(
                f"- {page_label} | {ch.section[:60]}"
                f" | sim={ch.similarity:.3f} rerank={ch.rerank_score:+.3f}"
            )

    if args.debug:
        r = result.retrieval
        print("\n=== Debug ===")
        print(f"max_dense_similarity = {r.max_similarity:.3f}")
        print(f"dense_hits           = {len(r.dense_hits)}")
        print(f"bm25_hits            = {len(r.bm25_hits)}")
        print(f"rrf_top              = {[cid for cid, _ in r.rrf_ranking[:10]]}")


if __name__ == "__main__":
    cli()
