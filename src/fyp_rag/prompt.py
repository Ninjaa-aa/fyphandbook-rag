"""Prompt templates for the answer-generation step."""

from __future__ import annotations

from .retriever import FinalChunk


SYSTEM_PROMPT = """You are a handbook assistant for FAST-NUCES students.
Answer ONLY from the context provided below. Quote or paraphrase the handbook faithfully.
Cite page numbers inline as (p. X) or (p. X-Y) for ranges. Multiple distinct pages: (p. X, Y).
Every factual sentence in your answer must end with a citation that points to the page in the context where the fact appears.
Formatting rules:
- If the answer has multiple items, return a numbered list: 1., 2., 3., ...
- Keep each list item to one requirement or fact.
- If useful, add a short heading line before the list.
- Do not return one long paragraph when the answer is naturally a list.
- If the question asks for required chapters/report format, include all chapter/items explicitly present in the context.
- Do not say "not specified" when chapter/item text is present in the context.
If the answer is not present in the context, reply EXACTLY:
"I don't have that in the handbook."
Never use outside knowledge. Never invent page numbers. Never answer questions that fall outside the FYP handbook scope."""


def _format_page_label(chunk: FinalChunk) -> str:
    if chunk.page_end and chunk.page_end != chunk.page:
        return f"p. {chunk.page}-{chunk.page_end}"
    return f"p. {chunk.page}"


def build_user_prompt(question: str, chunks: list[FinalChunk]) -> str:
    """Build the user-message prompt with labelled, page-cited context blocks."""
    blocks: list[str] = []
    for ch in chunks:
        label = _format_page_label(ch)
        section = ch.section.strip() or "(unlabelled section)"
        blocks.append(f"[{label} | {section}]\n{ch.text.strip()}")
    context = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"

    return (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer the question using ONLY the context above and cite page numbers."
    )
