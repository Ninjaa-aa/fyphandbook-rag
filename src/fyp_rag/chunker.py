"""Heading-aware chunker.

Pipeline:
    1. Walk every span in document order.
    2. Detect heading spans via font-size + bold + ALL-CAPS heuristics.
    3. Group consecutive non-heading spans into sections under each heading.
    4. Sub-split overly long section bodies with `RecursiveCharacterTextSplitter`.
    5. Merge tiny sections into the next one so we never produce a single-line chunk.

Each emitted chunk carries `{page, page_end, section, chunk_id}` metadata so
the LLM can cite pages exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_SECTION_CHARS_BEFORE_SPLIT,
    MIN_CHUNK_CHARS,
)
from .logger import get_logger
from .pdf_loader import Page, Span

log = get_logger(__name__)


_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
_ALL_CAPS_LINE = re.compile(r"^[A-Z0-9 ,.&'()\-/:]+$")


@dataclass
class Section:
    title: str
    page_start: int
    page_end: int
    body: str = ""
    pages_seen: set[int] = field(default_factory=set)

    def add(self, text: str, page: int) -> None:
        self.body = (self.body + " " + text).strip() if self.body else text
        self.pages_seen.add(page)
        self.page_end = max(self.page_end, page)
        self.page_start = min(self.page_start, page)


@dataclass
class Chunk:
    text: str
    page: int
    page_end: int
    section: str
    chunk_id: str
    is_format_chunk: bool = False


def _is_heading(span: Span, body_size: float) -> bool:
    """Heuristic heading detector tuned for the FYP handbook layout."""
    text = span.text.strip()
    if len(text) < 3 or len(text) > 120:
        return False

    if span.size > body_size * 1.15:
        return True

    if span.is_bold and len(text) <= 80 and not text.endswith(("." , ",", ";", ":")):
        # Bold-only headings usually omit terminal punctuation.
        return True

    if _NUMBERED_HEADING.match(text):
        return True

    if len(text) <= 60 and _ALL_CAPS_LINE.match(text) and any(c.isalpha() for c in text):
        # Short ALL-CAPS lines are nearly always section labels.
        return True

    return False


def _build_sections(pages: Iterable[Page], body_size: float) -> list[Section]:
    sections: list[Section] = []
    current = Section(title="(Preamble)", page_start=1, page_end=1)

    for page in pages:
        if not current.pages_seen:
            current.page_start = page.page
            current.page_end = page.page

        for span in page.spans:
            if _is_heading(span, body_size):
                if current.body.strip():
                    sections.append(current)
                current = Section(
                    title=span.text.strip(),
                    page_start=page.page,
                    page_end=page.page,
                )
            else:
                current.add(span.text, page.page)

    if current.body.strip():
        sections.append(current)

    return sections


def _merge_tiny(sections: list[Section]) -> list[Section]:
    """Merge sections shorter than `MIN_CHUNK_CHARS` into the next one."""
    merged: list[Section] = []
    buffer: Section | None = None

    for sec in sections:
        if buffer is not None:
            sec.body = (buffer.title + ". " + buffer.body + " " + sec.body).strip()
            sec.page_start = min(buffer.page_start, sec.page_start)
            sec.page_end = max(buffer.page_end, sec.page_end)
            buffer = None

        if len(sec.body) < MIN_CHUNK_CHARS:
            buffer = sec
            continue

        merged.append(sec)

    if buffer is not None:
        if merged:
            tail = merged[-1]
            tail.body = tail.body + " " + buffer.title + ". " + buffer.body
            tail.page_end = max(tail.page_end, buffer.page_end)
        else:
            merged.append(buffer)

    return merged


def _split_long_section(sec: Section, splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    parts = splitter.split_text(sec.body)
    chunks: list[Chunk] = []
    for i, part in enumerate(parts):
        chunks.append(
            Chunk(
                text=part.strip(),
                page=sec.page_start,
                page_end=sec.page_end,
                section=sec.title,
                chunk_id=f"p{sec.page_start}_s{abs(hash(sec.title)) % 10**6}_c{i}",
                is_format_chunk=_looks_like_report_format(sec.title, part),
            )
        )
    return chunks


def _looks_like_report_format(section: str, text: str) -> bool:
    joined = f"{section} {text}".lower()
    return any(
        marker in joined
        for marker in (
            "report format",
            "fyp report contents",
            "chapter",
            "software requirement specifications",
            "iteration plan",
            "user manual",
            "literature review",
            "proposed approach",
            "validation and testing",
            "results and discussion",
            "conclusions and future work",
        )
    )


def _build_page_fallback_chunks(
    pages: list[Page],
    splitter: RecursiveCharacterTextSplitter,
) -> list[Chunk]:
    """Add page-level fallback chunks to preserve list/table-style text blocks.

    Span-based sectioning can occasionally flatten or skip structured lines.
    These page chunks ensure exact report-format lists remain retrievable.
    """
    page_chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            page_chunks.append(
                Chunk(
                    text=part.strip(),
                    page=page.page,
                    page_end=page.page,
                    section=f"Page {page.page} full text",
                    chunk_id=f"p{page.page}_page_c{i}",
                    is_format_chunk=_looks_like_report_format(f"Page {page.page}", part),
                )
            )
    return page_chunks


def chunk_pages(pages: list[Page], body_size: float) -> list[Chunk]:
    """End-to-end: pages -> heading-aware sections -> sized chunks."""
    sections = _build_sections(pages, body_size)
    sections = _merge_tiny(sections)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for sec in sections:
        body = sec.body.strip()
        if not body:
            continue

        if len(body) <= MAX_SECTION_CHARS_BEFORE_SPLIT:
            chunks.append(
                Chunk(
                    text=body,
                    page=sec.page_start,
                    page_end=sec.page_end,
                    section=sec.title,
                    chunk_id=f"p{sec.page_start}_s{abs(hash(sec.title)) % 10**6}_c0",
                    is_format_chunk=_looks_like_report_format(sec.title, body),
                )
            )
        else:
            chunks.extend(_split_long_section(sec, splitter))

    # Fallback representation: preserve raw page text for better recall on
    # numbered chapter lists (e.g., Development FYP report format).
    chunks.extend(_build_page_fallback_chunks(pages, splitter))

    log.info(
        "Built %d chunks from %d sections (avg chunk len %.0f chars)",
        len(chunks),
        len(sections),
        sum(len(c.text) for c in chunks) / max(1, len(chunks)),
    )
    return chunks
