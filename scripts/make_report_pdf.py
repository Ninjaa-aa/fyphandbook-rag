"""Render `report/report.md` to a PDF deliverable using ReportLab.

The renderer is intentionally minimal: it parses the markdown line-by-line for
headings, bullets, code fences, and pipe tables — enough for our own report
file, not a general-purpose markdown engine. A cover page, architecture image,
and ordered screenshots are appended from `outputs/screenshots/`.

Run with:
    uv run python scripts/make_report_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from fyp_rag.config import REPORT_DIR, ROOT_DIR  # noqa: E402

REPORT_MD = REPORT_DIR / "report.md"
REPORT_PDF = REPORT_DIR / "report.pdf"
SCREENSHOT_DIR = ROOT_DIR / "outputs" / "screenshots"
COVER_LOGO = SCREENSHOT_DIR / "logo.png"
ORDERED_SCREENSHOTS: tuple[tuple[str, str], ...] = (
    ("architecture.png", "System Architecture"),
    ("q1.png", "Validation Query 1 - Headings, Fonts, and Sizes"),
    ("q2.png", "Validation Query 2 - Margins and Spacing"),
    ("q3.png", "Validation Query 3 - Development FYP Report Sections"),
    ("q4.png", "Validation Query 4 - R&D-Based FYP Report Chapters"),
    ("q5.png", "Validation Query 5 - Ibid. and op. cit. Endnotes"),
    ("q6.png", "Validation Query 6 - Executive Summary and Abstract"),
    ("out-of-scope.png", "Out-of-Scope Refusal Test"),
)


def _styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#111827"),
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=16,
            textColor=colors.HexColor("#1f2937"),
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            parent=base["BodyText"],
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
            textColor=colors.HexColor("#111827"),
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#1f2937"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=12, spaceAfter=6, textColor=colors.HexColor("#111827"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9.5, leading=12.5,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontSize=8.5, leading=11,
            backColor=colors.HexColor("#f3f4f6"),
            textColor=colors.HexColor("#111827"),
            borderPadding=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=9.5, leading=12.5, leftIndent=14, bulletIndent=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
        ),
    }


_INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_INLINE_ITAL = re.compile(r"\*(.+?)\*")
_INLINE_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _INLINE_BOLD.sub(r"<b>\1</b>", text)
    text = _INLINE_ITAL.sub(r"<i>\1</i>", text)
    text = _INLINE_CODE.sub(r"<font face='Courier'>\1</font>", text)
    return text


def _parse_table(table_lines: list[str]) -> Table:
    rows: list[list[str]] = []
    for ln in table_lines:
        if re.match(r"^\s*\|?\s*-{2,}", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return Table([[""]])

    styles = _styles()
    body_style = ParagraphStyle("td", parent=styles["body"], fontSize=8.5, leading=10.5)
    head_style = ParagraphStyle("th", parent=body_style, textColor=colors.white)

    data = []
    for i, row in enumerate(rows):
        style = head_style if i == 0 else body_style
        data.append([Paragraph(_inline(c), style) for c in row])

    table = Table(data, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _md_to_flowables(md: str):
    styles = _styles()
    flow = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            j = i + 1
            buf: list[str] = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            code = "\n".join(buf).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            flow.append(Paragraph(f"<pre>{code}</pre>", styles["code"]))
            flow.append(Spacer(1, 4))
            i = j + 1
            continue

        if line.lstrip().startswith("|") and "|" in line:
            tbl_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            flow.append(_parse_table(tbl_lines))
            flow.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            flow.append(Paragraph(_inline(line[2:].strip()), styles["h1"]))
        elif line.startswith("## "):
            flow.append(Paragraph(_inline(line[3:].strip()), styles["h2"]))
        elif line.startswith("### "):
            flow.append(Paragraph(_inline(line[4:].strip()), styles["h2"]))
        elif line.lstrip().startswith(("- ", "* ")):
            text = line.lstrip()[2:]
            flow.append(Paragraph(f"&bull; {_inline(text)}", styles["bullet"]))
        elif not line.strip():
            flow.append(Spacer(1, 4))
        else:
            flow.append(Paragraph(_inline(line), styles["body"]))
        i += 1

    return flow


def _scaled_image(path: Path, *, max_width: float, max_height: float) -> Image:
    """Create a ReportLab image that preserves aspect ratio within bounds."""
    width_px, height_px = ImageReader(str(path)).getSize()
    scale = min(max_width / width_px, max_height / height_px)
    img = Image(
        str(path),
        width=width_px * scale,
        height=height_px * scale,
        kind="proportional",
    )
    img.hAlign = "CENTER"
    return img


def _cover_page():
    styles = _styles()
    flow = []

    if COVER_LOGO.exists():
        flow.append(_scaled_image(COVER_LOGO, max_width=2.0 * inch, max_height=2.0 * inch))
        flow.append(Spacer(1, 0.35 * inch))

    flow.append(Paragraph("FYP Handbook RAG Assistant", styles["cover_title"]))
    flow.append(Paragraph("Assignment 03", styles["cover_subtitle"]))
    flow.append(Spacer(1, 0.2 * inch))
    flow.append(Paragraph("<b>Submitted By</b>", styles["cover_meta"]))
    flow.append(Paragraph("Hammad Zahid", styles["cover_meta"]))
    flow.append(Paragraph("SE-D", styles["cover_meta"]))
    flow.append(Paragraph("22i-2433", styles["cover_meta"]))
    flow.append(Spacer(1, 0.25 * inch))
    flow.append(Paragraph("<b>Submitted To</b>", styles["cover_meta"]))
    flow.append(Paragraph("Dr. Shahela Saif", styles["cover_meta"]))
    flow.append(PageBreak())
    return flow


def _append_ordered_screenshots(flow):
    styles = _styles()
    flow.append(PageBreak())
    flow.append(Paragraph("Architecture and Screenshots", styles["h1"]))
    flow.append(Spacer(1, 6))

    for idx, (filename, title) in enumerate(ORDERED_SCREENSHOTS):
        img_path = SCREENSHOT_DIR / filename
        if idx:
            flow.append(PageBreak())

        flow.append(Paragraph(title, styles["h2"]))
        if not img_path.exists():
            flow.append(Paragraph(f"Missing image: <font face='Courier'>{img_path}</font>", styles["body"]))
            continue

        flow.append(_scaled_image(img_path, max_width=7.0 * inch, max_height=5.6 * inch))
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(f"<i>{filename}</i>", styles["caption"]))


def main() -> None:
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Missing {REPORT_MD}")

    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
    md = REPORT_MD.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="FYP Handbook RAG — Architectural Note",
        author="FYP RAG Assistant",
    )

    flow = _cover_page()
    flow.extend(_md_to_flowables(md))
    _append_ordered_screenshots(flow)

    doc.build(flow)
    print(f"Wrote {REPORT_PDF}")


if __name__ == "__main__":
    main()
