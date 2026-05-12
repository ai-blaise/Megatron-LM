#!/usr/bin/env python3
"""Render the Corsaire technical report Markdown source to a clean PDF.

This is intentionally self-contained: it uses ReportLab from the temporary
authoring environment and does not modify the Megatron-LM Python environment.
It implements only the Markdown subset used by the report source.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_X = 0.72 * inch
MARGIN_Y = 0.68 * inch
FRAME_TOP_PAD = 0.26 * inch
FRAME_BOTTOM_PAD = 0.22 * inch


@dataclass(frozen=True)
class ReportMetadata:
    title: str
    author: str
    date: str


class ReportDocTemplate(BaseDocTemplate):
    """DocTemplate that records headings for the table of contents."""

    def __init__(self, filename: str, title: str) -> None:
        frame = Frame(
            MARGIN_X,
            MARGIN_Y + FRAME_BOTTOM_PAD,
            PAGE_WIDTH - 2 * MARGIN_X,
            PAGE_HEIGHT - 2 * MARGIN_Y - FRAME_TOP_PAD - FRAME_BOTTOM_PAD,
            id="body",
        )
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_Y,
            bottomMargin=MARGIN_Y,
            title=title,
            author="BlaiseAI Research",
        )
        self.report_title = title
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self.draw_page)])

    def afterFlowable(self, flowable) -> None:  # noqa: ANN001 - ReportLab callback
        if not isinstance(flowable, Paragraph):
            return
        level_by_style = {
            "ReportHeading1": 0,
            "ReportHeading2": 1,
            "ReportHeading3": 2,
        }
        level = level_by_style.get(flowable.style.name)
        if level is None:
            return
        text = flowable.getPlainText()
        self.notify("TOCEntry", (level, text, self.page))

    def draw_page(self, canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D0D5DD"))
            canvas.setLineWidth(0.4)
            y = PAGE_HEIGHT - MARGIN_Y + 0.05 * inch
            canvas.line(MARGIN_X, y, PAGE_WIDTH - MARGIN_X, y)
            canvas.setFillColor(colors.HexColor("#475467"))
            canvas.setFont("Helvetica", 8)
            canvas.drawString(MARGIN_X, y + 0.08 * inch, self.report_title)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 0.42 * inch, str(doc.page))
        canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=31,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#101828"),
        spaceAfter=18,
    )
    styles["Subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475467"),
        spaceAfter=7,
    )
    styles["Body"] = ParagraphStyle(
        "ReportBody",
        parent=base["BodyText"],
        fontName="Times-Roman",
        fontSize=9.4,
        leading=12.4,
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceAfter=6.5,
        textColor=colors.HexColor("#111827"),
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["Abstract"] = ParagraphStyle(
        "ReportAbstract",
        parent=styles["Body"],
        leftIndent=0.35 * inch,
        rightIndent=0.35 * inch,
        fontSize=9.2,
        leading=12.2,
        textColor=colors.HexColor("#1F2937"),
    )
    styles["Heading1"] = ParagraphStyle(
        "ReportHeading1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=20,
        textColor=colors.HexColor("#101828"),
        spaceBefore=15,
        spaceAfter=7,
        keepWithNext=True,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["Heading2"] = ParagraphStyle(
        "ReportHeading2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.3,
        leading=15.3,
        textColor=colors.HexColor("#1D2939"),
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["Heading3"] = ParagraphStyle(
        "ReportHeading3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.6,
        leading=13,
        textColor=colors.HexColor("#344054"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["Bullet"] = ParagraphStyle(
        "ReportBullet",
        parent=styles["Body"],
        leftIndent=0.25 * inch,
        firstLineIndent=-0.16 * inch,
        bulletIndent=0.08 * inch,
        spaceAfter=4.7,
    )
    styles["Reference"] = ParagraphStyle(
        "ReportReference",
        parent=styles["Body"],
        fontSize=8.2,
        leading=10.5,
        leftIndent=0.15 * inch,
        firstLineIndent=-0.15 * inch,
        spaceAfter=4.0,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["Code"] = ParagraphStyle(
        "ReportCode",
        parent=base["Code"],
        fontName="Courier",
        fontSize=7.8,
        leading=9.5,
        leftIndent=0.10 * inch,
        rightIndent=0.10 * inch,
        spaceBefore=4,
        spaceAfter=7,
        backColor=colors.HexColor("#F2F4F7"),
        borderColor=colors.HexColor("#D0D5DD"),
        borderWidth=0.4,
        borderPadding=5,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["TableCell"] = ParagraphStyle(
        "ReportTableCell",
        parent=styles["Body"],
        fontName="Times-Roman",
        fontSize=7.8,
        leading=9.3,
        spaceAfter=0,
        splitLongWords=1,
        wordWrap="CJK",
    )
    styles["TableHead"] = ParagraphStyle(
        "ReportTableHead",
        parent=styles["TableCell"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
    )
    styles["Caption"] = ParagraphStyle(
        "ReportCaption",
        parent=styles["Body"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10,
        textColor=colors.HexColor("#475467"),
        spaceAfter=5,
    )
    styles["TOCTitle"] = ParagraphStyle(
        "ReportTOCTitle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=16,
    )
    return styles


def parse_metadata(lines: list[str]) -> tuple[ReportMetadata, int]:
    title = "Corsaire-1 Technical Report"
    author = "BlaiseAI Research"
    draft_date = "May 11, 2026"
    start_idx = 0

    for idx, line in enumerate(lines[:12]):
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            start_idx = idx + 1
        elif stripped.startswith("**Draft date:**"):
            draft_date = stripped.replace("**Draft date:**", "").strip()
            start_idx = idx + 1
        elif stripped.startswith("**") and stripped.endswith("**"):
            author = stripped.strip("*").strip()
            start_idx = idx + 1
        elif stripped.startswith("## "):
            break

    while start_idx < len(lines) and not lines[start_idx].strip().startswith("## "):
        start_idx += 1
    return ReportMetadata(title=title, author=author, date=draft_date), start_idx


def inline_markup(text: str) -> str:
    """Escape text and apply a small Markdown inline subset for ReportLab."""

    def convert_bold(segment: str) -> str:
        escaped = html.escape(segment)
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    parts = text.split("`")
    rendered: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2:
            rendered.append(f'<font name="Courier">{html.escape(part)}</font>')
        else:
            rendered.append(convert_bold(part))
    return "".join(rendered)


def normalize_paragraph(lines: Iterable[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip())


def make_paragraph(text: str, styles: dict[str, ParagraphStyle], name: str = "Body") -> Paragraph:
    return Paragraph(inline_markup(text), styles[name])


def table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_table_separator(line: str) -> bool:
    cells = table_cells(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def column_widths(rows: list[list[str]], total_width: float) -> list[float]:
    columns = max(len(row) for row in rows)
    weights: list[float] = []
    for col in range(columns):
        values = [row[col] if col < len(row) else "" for row in rows]
        longest = max((len(value) for value in values), default=1)
        weights.append(max(6, min(longest, 42)))
    total_weight = sum(weights) or columns
    widths = [total_width * weight / total_weight for weight in weights]

    min_width = 0.55 * inch
    if any(width < min_width for width in widths) and columns * min_width < total_width:
        deficit = sum(max(0, min_width - width) for width in widths)
        widths = [max(min_width, width) for width in widths]
        adjustable = [idx for idx, width in enumerate(widths) if width > min_width]
        adjustable_total = sum(widths[idx] - min_width for idx in adjustable) or 1
        for idx in adjustable:
            widths[idx] -= deficit * ((widths[idx] - min_width) / adjustable_total)
    return widths


def render_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [table_cells(line) for line in lines if not is_table_separator(line)]
    if not rows:
        rows = [[""]]
    columns = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (columns - len(row)))

    data = []
    for ridx, row in enumerate(rows):
        style = styles["TableHead"] if ridx == 0 else styles["TableCell"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = Table(
        data,
        colWidths=column_widths(rows, PAGE_WIDTH - 2 * MARGIN_X),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#344054")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#101828")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D5DD")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ]
        )
    )
    return table


def parse_list_item(lines: list[str], start: int) -> tuple[str, str, int]:
    first = lines[start]
    match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", first)
    if not match:
        raise ValueError("not a list item")
    indent, marker, body = match.groups()
    consumed = start + 1
    collected = [body.strip()]
    base_indent = len(indent)

    while consumed < len(lines):
        current = lines[consumed]
        stripped = current.strip()
        if not stripped:
            break
        if re.match(r"^\s*([-*]|\d+\.)\s+", current):
            break
        if current.startswith(" " * (base_indent + 2)):
            collected.append(stripped)
            consumed += 1
            continue
        if stripped.startswith("|") or stripped.startswith("#") or stripped.startswith("```"):
            break
        break
    return marker, normalize_paragraph(collected), consumed


def build_title_page(meta: ReportMetadata, styles: dict[str, ParagraphStyle]) -> list:
    return [
        Spacer(1, 1.45 * inch),
        Paragraph(inline_markup(meta.title), styles["Title"]),
        Paragraph(inline_markup(meta.author), styles["Subtitle"]),
        Paragraph(inline_markup(f"Draft date: {meta.date}"), styles["Subtitle"]),
        Spacer(1, 0.4 * inch),
        Table(
            [["Compressed SFT for a REAP-pruned DeepSeek-V3.2 MoE under NVFP4 W4A4/KV4 training constraints"]],
            colWidths=[PAGE_WIDTH - 2.2 * inch],
            hAlign="CENTER",
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("LEADING", (0, 0), (-1, -1), 16),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#344054")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#98A2B3")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#98A2B3")),
                    ("TOPPADDING", (0, 0), (-1, -1), 13),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
                ]
            ),
        ),
        Spacer(1, 0.5 * inch),
        Paragraph(
            inline_markup(
                "This draft emphasizes BlaiseAI's training-system contributions: "
                "low-bit optimizer state, transient-master updates, activation "
                "residual correction, trainable cache quantizers, long-context "
                "replay, and B200 kernel integration."
            ),
            styles["Abstract"],
        ),
        PageBreak(),
    ]


def build_toc(styles: dict[str, ParagraphStyle]) -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel0",
            fontName="Helvetica-Bold",
            fontSize=9.6,
            leading=12,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
            splitLongWords=1,
            wordWrap="CJK",
        ),
        ParagraphStyle(
            "TOCLevel1",
            fontName="Helvetica",
            fontSize=8.7,
            leading=11,
            leftIndent=0.22 * inch,
            firstLineIndent=0,
            splitLongWords=1,
            wordWrap="CJK",
        ),
        ParagraphStyle(
            "TOCLevel2",
            fontName="Helvetica",
            fontSize=8.1,
            leading=10.2,
            leftIndent=0.44 * inch,
            firstLineIndent=0,
            textColor=colors.HexColor("#475467"),
            splitLongWords=1,
            wordWrap="CJK",
        ),
    ]
    return [
        Paragraph("Contents", styles["TOCTitle"]),
        toc,
        PageBreak(),
    ]


def markdown_to_flowables(lines: list[str], styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    idx = 0
    paragraph: list[str] = []
    in_references = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = normalize_paragraph(paragraph)
        style_name = "Reference" if in_references and re.match(r"^\[\d+\]", text) else "Body"
        story.append(make_paragraph(text, styles, style_name))
        paragraph = []

    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            story.append(Spacer(1, 2.5))
            idx += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            fence_lang = stripped.strip("`").strip()
            idx += 1
            code_lines: list[str] = []
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx].rstrip("\n"))
                idx += 1
            if idx < len(lines):
                idx += 1
            code = "\n".join(code_lines)
            story.append(Preformatted(code, styles["Code"], maxLineLength=92))
            if fence_lang:
                story.append(Paragraph(inline_markup(f"Code block: {fence_lang}"), styles["Caption"]))
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph()
            table_lines: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx].rstrip("\n"))
                idx += 1
            story.append(KeepTogether([render_table(table_lines, styles), Spacer(1, 6)]))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            hashes, text = heading_match.groups()
            level = len(hashes)
            if text == "References":
                in_references = True
            if level <= 2:
                style = "Heading1"
            elif level == 3:
                style = "Heading2"
            else:
                style = "Heading3"
            story.append(Paragraph(inline_markup(text), styles[style]))
            idx += 1
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            flush_paragraph()
            marker, body, next_idx = parse_list_item(lines, idx)
            bullet = "\u2022" if marker in {"-", "*"} else marker
            story.append(Paragraph(inline_markup(body), styles["Bullet"], bulletText=bullet))
            idx = next_idx
            continue

        paragraph.append(line)
        idx += 1

    flush_paragraph()
    return story


def render(markdown_path: Path, pdf_path: Path) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    meta, content_start = parse_metadata(lines)
    styles = build_styles()

    story = []
    story.extend(build_title_page(meta, styles))
    story.extend(build_toc(styles))
    story.extend(markdown_to_flowables(lines[content_start:], styles))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(str(pdf_path), title=meta.title)
    doc.multiBuild(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, help="Path to the report Markdown source")
    parser.add_argument("pdf", type=Path, help="Path to write the rendered PDF")
    args = parser.parse_args()
    render(args.markdown, args.pdf)


if __name__ == "__main__":
    main()
