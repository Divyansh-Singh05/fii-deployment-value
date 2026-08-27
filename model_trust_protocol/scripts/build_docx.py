"""Render the manuscript to MS Word under the Indian Journal of Finance rules.

Formatting is mandated rather than chosen: Times New Roman 12 black throughout,
double spacing, one-inch margins, main headings in upper case and bold,
sub-headings in title case, bold and italic, table titles italic above the
table with the source below it. Every main section starts on a new page.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

from scripts.equations import EQUATIONS

SRC = Path("submission/ijf/manuscript.md")
OUT = Path("submission/ijf/IJF_manuscript_blinded.docx")
COVER = Path("submission/ijf/IJF_cover_page.docx")

FONT, SIZE = "Times New Roman", Pt(12)
BLACK = RGBColor(0, 0, 0)


def _style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = SIZE
    normal.font.color.rgb = BLACK
    pf = normal.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)


def _run(par, text: str, *, bold=False, italic=False) -> None:
    r = par.add_run(text)
    r.font.name = FONT
    r.font.size = SIZE
    r.font.color.rgb = BLACK
    r.bold = bold
    r.italic = italic


def _para(doc, text: str, *, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    _emit_inline(p, text, bold=bold, italic=italic)
    return p


def _emit_inline(par, text: str, *, bold=False, italic=False) -> None:
    """Render **bold** and *italic* spans, dropping the markers."""
    for token in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            _run(par, token[2:-2], bold=True, italic=italic)
        elif token.startswith("*") and token.endswith("*"):
            _run(par, token[1:-1], bold=bold, italic=True)
        else:
            _run(par, token, bold=bold, italic=italic)


def _equation(doc, equation) -> None:
    """One displayed equation, centred, with its number at the right margin."""
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    width = doc.sections[0].page_width - doc.sections[0].left_margin \
        - doc.sections[0].right_margin
    pf.tab_stops.add_tab_stop(int(width * 0.5), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    par.add_run("\t")
    for text, kind in equation.word:
        r = par.add_run(text)
        r.font.name = FONT
        r.font.size = SIZE
        r.font.color.rgb = BLACK
        r.italic = kind in ("i", "si")
        r.font.subscript = kind in ("s", "si")
        r.font.superscript = kind == "u"
    par.add_run("\t")
    n = par.add_run(f"({equation.number})")
    n.font.name = FONT
    n.font.size = SIZE
    n.font.color.rgb = BLACK


def _table(doc, rows: list[list[str]]) -> None:
    header, *body = rows
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for cell, txt in zip(t.rows[0].cells, header, strict=True):
        cell.text = ""
        _run(cell.paragraphs[0], txt, bold=True)
        cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    for row in body:
        cells = t.add_row().cells
        for cell, txt in zip(cells, row, strict=True):
            cell.text = ""
            _emit_inline(cell.paragraphs[0], txt)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0


def build() -> None:
    lines = SRC.read_text().splitlines()
    doc = Document()
    _style(doc)

    i = 0
    pending_table: list[list[str]] = []

    def flush_table() -> None:
        nonlocal pending_table
        if pending_table:
            _table(doc, pending_table)
            pending_table = []

    while i < len(lines):
        line = lines[i].rstrip()
        i += 1

        eq = re.match(r"\[\[EQ:(\d+)\]\]", line.strip())
        if eq:
            _equation(doc, EQUATIONS[int(eq.group(1))])
            continue

        m = re.match(r"\[\[FIGURE:([^\]]+)\]\]", line.strip())
        if m:
            img = SRC.parent / "figures" / m.group(1)
            if not img.is_file():
                raise FileNotFoundError(f"figure not found: {img}")
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.line_spacing = 1.0
            par.add_run().add_picture(str(img), width=Inches(6.0))
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            pending_table.append(cells)
            continue
        flush_table()

        if not line.strip():
            continue

        if line.startswith("# "):
            p = _para(doc, line[2:].strip(), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_after = Pt(12)
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            par = _para(doc, heading.upper(), bold=True)
            par.paragraph_format.space_before = Pt(12)
            continue

        if line.startswith("### "):
            _para(doc, line[4:].strip(), bold=True, italic=True)
            continue

        # Table titles and sources are already italicised in the source.
        _para(doc, line)

    flush_table()
    doc.save(OUT)
    print(f"written {OUT}")


def cover() -> None:
    doc = Document()
    _style(doc)
    _para(doc, "COVER PAGE", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(
        doc,
        "What Is an Alternative Dataset Worth? Mapping the Friction Boundary of "
        "Institutional Flow Data",
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    doc.add_paragraph()
    for label, value in [
        ("Name", "Divyansh Singh"),
        ("Designation", "[TO COMPLETE]"),
        ("Institute (name and full postal address with pincode)", "[TO COMPLETE]"),
        ("Email id", "divyanshsinghpost@gmail.com"),
        ("ORCID id", "[TO COMPLETE]"),
    ]:
        p = doc.add_paragraph()
        _run(p, f"{label}: ", bold=True)
        _run(p, value)
    doc.add_paragraph()
    _para(doc, "Declaration", bold=True)
    for claim in [
        "The paper is my original contribution and has not been plagiarized from "
        "any source/individual. It does not infringe on any copyright, trademark, "
        "patent, statutory right, or propriety right of others and the paper does "
        "not contain any libelous or unlawful statements. All the references are "
        "duly acknowledged at the appropriate places and I sign for and accept the "
        "responsibility for releasing this material on behalf of my co-authors.",
        "The work has been submitted only to Indian Journal of Finance, New Delhi "
        "and it has not been previously published or submitted elsewhere for "
        "publication in a refereed or copyrighted publication.",
        "It is agreed that the sole and exclusive rights in the whole copyright of "
        "the said paper to the contribution identified above is transferred to "
        "Indian Journal of Finance, New Delhi.",
        "I have permission from copyright owner(s) to reproduce/adapt any content "
        "that I have reproduced or adapted in this paper.",
        "I agree to indemnify Indian Journal of Finance, New Delhi against any "
        "claim or action alleging facts which, if true, constitute a breach of any "
        "of the foregoing warranties.",
    ]:
        _para(doc, claim)
    doc.save(COVER)
    print(f"written {COVER}")


if __name__ == "__main__":
    build()
    cover()
