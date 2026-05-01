"""PDF loading using PyMuPDF.

Extracts both flat text and a structured span list (with font sizes / flags)
per page. The structured form lets the chunker detect headings without OCR.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .logger import get_logger

log = get_logger(__name__)
_PAGE_NUM_RE = re.compile(r"^\d{1,3}$")


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
    page: int            # handbook printed page number (preferred for citations)
    text: str
    spans: list[Span]


def _extract_printed_page_number(page_text: str, fallback_pdf_page: int) -> int:
    """Best-effort parse of handbook printed page number from page footer.

    Heuristic:
    - walk lines bottom-up
    - pick the first standalone integer token (1-3 digits), e.g. `39`
    - fallback to PDF physical page index when not found
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _PAGE_NUM_RE.match(line):
            page_num = int(line)
            if 1 <= page_num <= 999:
                return page_num
    return fallback_pdf_page


def extract_pages(pdf_path: str | Path) -> tuple[list[Page], float]:
    """Extract every page's text + styled spans, plus the median body font size.

    Returns:
        pages: list of `Page` objects (using printed handbook page numbers when
               available; otherwise PDF physical index).
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
        pdf_page_no = idx + 1

        flat_text = page.get_text("text").strip()
        if not flat_text:
            continue
        handbook_page_no = _extract_printed_page_number(flat_text, pdf_page_no)

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
                            page=handbook_page_no,
                        )
                    )
                    sizes.append(size)

        pages.append(Page(page=handbook_page_no, text=flat_text, spans=spans))

    body_size = statistics.median(sizes) if sizes else 11.0
    log.info(
        "Extracted %d pages from %s (median body font size = %.2f)",
        len(pages),
        pdf_path.name,
        body_size,
    )
    return pages, body_size
