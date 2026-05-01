"""Run the 6 required validation queries + 1 out-of-domain probe.

Outputs:
    - `prompts.txt`     : full system prompt + per-query user prompt + raw response
    - `outputs/eval.json` : structured results (chunks, scores, refusal flags)

Run with:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fyp_rag.config import EVAL_RESULTS, PROMPTS_LOG, ensure_dirs  # noqa: E402
from fyp_rag.logger import get_logger  # noqa: E402
from fyp_rag.pipeline import PipelineAnswer, answer, load_resources  # noqa: E402

log = get_logger(__name__)


VALIDATION_QUERIES: list[str] = [
    "What headings, fonts, and sizes are required in the FYP report?",
    "What margins and spacing do we use?",
    "What are the required chapters/sections of a Development FYP report?",
    "What are the required chapters of an R&D-based FYP report?",
    "How should endnotes like 'Ibid.' and 'op. cit.' be used?",
    "What goes into the Executive Summary and Abstract?",
]

OUT_OF_DOMAIN_QUERY: str = "What is the cafeteria menu at FAST?"


def _serialize(result: PipelineAnswer) -> dict:
    chunks = [
        {
            "page": c.page,
            "page_end": c.page_end,
            "section": c.section,
            "similarity": round(c.similarity, 4),
            "rerank_score": round(c.rerank_score, 4),
            "rrf_score": round(c.rrf_score, 4),
            "text_preview": c.text[:300],
        }
        for c in result.retrieval.chunks
    ]
    return {
        "question": result.question,
        "refused": result.refused,
        "answer": result.answer,
        "max_similarity": round(result.retrieval.max_similarity, 4),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


def _write_prompts_log(records: list[PipelineAnswer]) -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("FYP HANDBOOK RAG — PROMPT LOG")
    lines.append("=" * 78)
    lines.append("")
    lines.append("=== SYSTEM PROMPT ===")
    lines.append(records[0].system_prompt if records else "")
    lines.append("")

    for i, rec in enumerate(records, start=1):
        lines.append("-" * 78)
        lines.append(f"=== INFERENCE PROMPT (Query {i}) ===")
        lines.append(f"Question: {rec.question}")
        lines.append("")
        if rec.refused:
            lines.append("[REFUSED — max similarity below threshold]")
            lines.append(f"max_similarity = {rec.retrieval.max_similarity:.3f}")
        else:
            lines.append("=== USER PROMPT ===")
            lines.append(rec.user_prompt)
        lines.append("")
        lines.append("=== RAW RESPONSE ===")
        lines.append(rec.answer)
        lines.append("")

    PROMPTS_LOG.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote prompt log to %s", PROMPTS_LOG)


def main() -> None:
    ensure_dirs()
    resources = load_resources()

    queries = VALIDATION_QUERIES + [OUT_OF_DOMAIN_QUERY]
    records: list[PipelineAnswer] = []
    json_results: list[dict] = []

    for i, q in enumerate(queries, start=1):
        kind = "OOD" if q == OUT_OF_DOMAIN_QUERY else "VALID"
        log.info("[%d/%d %s] %s", i, len(queries), kind, q)

        result = answer(q, resources=resources)
        records.append(result)
        json_results.append(_serialize(result))

        if result.refused:
            log.info("  -> refused (max_sim=%.3f)", result.retrieval.max_similarity)
        else:
            log.info("  -> answered with %d chunks", len(result.retrieval.chunks))

    EVAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS.write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote eval results JSON to %s", EVAL_RESULTS)

    _write_prompts_log(records)

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    for rec in records:
        status = "REFUSED" if rec.refused else "ANSWERED"
        print(f"[{status}] {rec.question}")
        if not rec.refused:
            cited_pages = sorted({c.page for c in rec.retrieval.chunks})
            print(f"  cited pages: {cited_pages}")
    print("=" * 60)


if __name__ == "__main__":
    main()
