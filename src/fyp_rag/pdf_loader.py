"""PDF loading using PyMuPDF.

Extracts both flat text and a structured span list (with font sizes / flags)
per page. The structured form lets the chunker detect headings without OCR.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class Span:
    """A single styled text fragment from PyMuPDF."""

    text: str
    size: float
    flags: int           # PyMuPDF flags bitmask (bold = 16, italic = 2, ...)
    font: str
    page: int            # 1-indexed

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 16) or "Bold" in self.font or "bold" in self.font


@dataclass
class Page:
    page: int            # 1-indexed
    text: str
    spans: list[Span]


def extract_pages(pdf_path: str | Path) -> tuple[list[Page], float]:
    """Extract every page's text + styled spans, plus the median body font size.

    Returns:
        pages: list of `Page` objects (page numbers are 1-indexed).
        body_size: median span font size across the whole document, used by
                   the chunker as the heuristic threshold for headings.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: list[Page] = []
    sizes: list[float] = []

    for idx in range(len(doc)):
        page = doc[idx]
        page_no = idx + 1

        flat_text = page.get_text("text").strip()
        if not flat_text:
            continue

        spans: list[Span] = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 == text block
                continue
            for line in block.get("lines", []):
                for raw_span in line.get("spans", []):
                    text = raw_span.get("text", "").strip()
                    if not text:
                        continue
                    size = float(raw_span.get("size", 0.0))
                    spans.append(
                        Span(
                            text=text,
                            size=size,
                            flags=int(raw_span.get("flags", 0)),
                            font=str(raw_span.get("font", "")),
                            page=page_no,
                        )
                    )
                    sizes.append(size)

        pages.append(Page(page=page_no, text=flat_text, spans=spans))

    body_size = statistics.median(sizes) if sizes else 11.0
    log.info(
        "Extracted %d pages from %s (median body font size = %.2f)",
        len(pages),
        pdf_path.name,
        body_size,
    )
    return pages, body_size
