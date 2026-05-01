"""Streamlit UI for the FYP Handbook RAG.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fyp_rag.config import REFUSAL_MESSAGE, SIMILARITY_THRESHOLD, TOP_K_RERANK  # noqa: E402
from fyp_rag.pipeline import Resources, answer, load_resources  # noqa: E402


st.set_page_config(
    page_title="FYP Handbook Assistant",
    page_icon=":books:",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading models and indexes (one-time)...")
def _resources() -> Resources:
    return load_resources()


def _format_page_label(page: int, page_end: int) -> str:
    if page_end and page_end != page:
        return f"p. {page}-{page_end}"
    return f"p. {page}"


def main() -> None:
    st.title("FYP Handbook Assistant")
    st.caption(
        "FAST-NUCES BS Final Year Project Handbook 2023 — ask any question about the FYP process."
    )

    try:
        resources = _resources()
    except Exception as e:
        st.error(
            f"Failed to load resources: {e}\n\n"
            "Run `uv run python ingest.py` first to build the index."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### Settings")
        top_k = st.slider("Final chunks (k)", 1, 10, TOP_K_RERANK)
        threshold = st.slider(
            "Similarity threshold", 0.0, 1.0, float(SIMILARITY_THRESHOLD), 0.05,
            help="Refuse to answer if max dense cosine < threshold.",
        )
        st.markdown("---")
        st.markdown("**Stack**")
        st.markdown(
            "- BGE-M3 dense + BM25 (RRF)\n"
            "- BGE-reranker-v2-m3 cross-encoder\n"
            "- MMR diversity\n"
            "- Groq Llama 3.3 70B"
        )

    question = st.text_input(
        "Ask a question about the FYP process...",
        placeholder="e.g. What margins and spacing do we use?",
    )
    ask_clicked = st.button("Ask", type="primary")

    if not (ask_clicked and question.strip()):
        return

    with st.spinner("Searching handbook and generating answer..."):
        result = answer(
            question.strip(),
            resources=resources,
            top_k_final=top_k,
            similarity_threshold=threshold,
        )

    st.subheader("Answer")
    if result.refused:
        st.warning(REFUSAL_MESSAGE)
    else:
        st.markdown(result.answer)

    if not result.refused:
        with st.expander("Sources (page refs)", expanded=True):
            if not result.retrieval.chunks:
                st.caption(
                    f"No chunks passed the similarity threshold "
                    f"(max cosine = {result.retrieval.max_similarity:.3f})."
                )
            else:
                for ch in result.retrieval.chunks:
                    label = _format_page_label(ch.page, ch.page_end)
                    st.markdown(
                        f"**{label}** &middot; *{ch.section}*"
                        f"  \n`sim={ch.similarity:.3f}  rerank={ch.rerank_score:+.3f}  rrf={ch.rrf_score:.3f}`"
                    )
                    snippet = ch.text[:400] + ("..." if len(ch.text) > 400 else "")
                    st.caption(snippet)
                    st.divider()

    with st.expander("Debug: retrieval pipeline"):
        r = result.retrieval
        st.markdown(f"**Max dense similarity:** `{r.max_similarity:.3f}`")
        st.markdown(f"**Dense hits:** {len(r.dense_hits)} &nbsp; "
                    f"**BM25 hits:** {len(r.bm25_hits)} &nbsp; "
                    f"**RRF candidates:** {len(r.rrf_ranking)}")
        if r.dense_hits:
            st.markdown("**Top dense hits**")
            st.table(
                [
                    {
                        "chunk_id": h.chunk_id,
                        "page": h.metadata.get("page"),
                        "similarity": round(h.similarity, 3),
                    }
                    for h in r.dense_hits[:10]
                ]
            )
        if r.bm25_hits:
            st.markdown("**Top BM25 hits**")
            st.table(
                [
                    {"chunk_id": h.chunk_id, "bm25": round(h.score, 3)}
                    for h in r.bm25_hits[:10]
                ]
            )

    with st.expander("Prompt sent to LLM"):
        st.markdown("**System prompt**")
        st.code(result.system_prompt, language="text")
        if result.user_prompt:
            st.markdown("**User prompt**")
            st.code(result.user_prompt, language="text")


if __name__ == "__main__":
    main()
