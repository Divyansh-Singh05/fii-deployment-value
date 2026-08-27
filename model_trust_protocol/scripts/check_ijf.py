"""Check the rendered manuscript against the Indian Journal of Finance rules.

Each check corresponds to a stated requirement. A table *title* is a line
beginning "Table N."; a mention of "Table N" elsewhere is a cross-reference,
which the guidelines require rather than forbid, so the two are counted apart.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt

DOC = Path("submission/ijf/IJF_manuscript_blinded.docx")
WORD_LIMIT = 6000
ABSTRACT_LIMIT = 250


def main() -> int:
    d = Document(DOC)
    issues: list[str] = []
    notes: list[str] = []

    normal = d.styles["Normal"]
    notes.append(
        f"font {normal.font.name} {normal.font.size.pt:.0f}pt, "
        f"line spacing {normal.paragraph_format.line_spacing}"
    )
    if normal.font.name != "Times New Roman" or normal.font.size != Pt(12):
        issues.append("font must be Times New Roman 12")
    if normal.paragraph_format.line_spacing != 2.0:
        issues.append("text must be double spaced")

    sec = d.sections[0]
    margins = {
        round(sec.top_margin.inches, 2),
        round(sec.bottom_margin.inches, 2),
        round(sec.left_margin.inches, 2),
        round(sec.right_margin.inches, 2),
    }
    notes.append(f"margins {margins}")
    if margins != {1.0}:
        issues.append("margins must be one inch")

    paras = [p.text for p in d.paragraphs]
    text = "\n".join(paras)
    tables_text = "\n".join(
        c.text for t in d.tables for r in t.rows for c in r.cells
    )
    full = text + "\n" + tables_text

    body = full.split("REFERENCES")[0]
    words = len(re.findall(r"[A-Za-z0-9'\-]+", body))
    notes.append(f"body words excluding references: {words}")
    if words > WORD_LIMIT:
        issues.append(f"word limit exceeded: {words} > {WORD_LIMIT}")

    abstract = text.split("Purpose.", 1)[-1].split("Keywords")[0]
    a_words = len(re.findall(r"[A-Za-z0-9'\-]+", abstract))
    notes.append(f"abstract words: {a_words}")
    if a_words > ABSTRACT_LIMIT:
        issues.append(f"abstract exceeds {ABSTRACT_LIMIT} words: {a_words}")
    for component in ("Purpose", "Design/Methodology/Approach", "Findings",
                      "Practical Implications", "Originality/Value"):
        if component not in text:
            issues.append(f"abstract missing component: {component}")

    titles = re.findall(r"(?m)^\*?Table (\d+)\.", text)
    notes.append(f"table titles: {titles}; table objects: {len(d.tables)}")
    if titles != [str(i + 1) for i in range(len(titles))]:
        issues.append(f"table titles not consecutive: {titles}")
    if len(titles) != len(d.tables):
        issues.append(f"{len(titles)} titles for {len(d.tables)} tables")
    for t in titles:
        mentions = len(re.findall(rf"Table {t}\b", text))
        if mentions < 2:
            issues.append(f"Table {t} is never cross-referenced in the text")
    fig_titles = re.findall(r"(?m)^\*?Figure (\d+)\.", text)
    n_images = len(d.inline_shapes)
    notes.append(f"figure titles: {fig_titles}; embedded images: {n_images}")
    if fig_titles != [str(i + 1) for i in range(len(fig_titles))]:
        issues.append(f"figure titles not consecutive in order of appearance: {fig_titles}")
    if len(fig_titles) != n_images:
        issues.append(f"{len(fig_titles)} figure titles for {n_images} embedded images")
    for f in fig_titles:
        if len(re.findall(rf"Figure {f}\b", text)) < 2:
            issues.append(f"Figure {f} is never cross-referenced in the text")

    if re.search(r"(Table|Figure) \d+\.\d", text):
        issues.append("exhibits must not be sub-numbered")

    refs = [
        line for line in text.split("REFERENCES", 1)[-1].splitlines() if line.strip()
    ]
    dois = re.findall(r"https://doi\.org/\S+", text)
    notes.append(f"references: {len(refs)}, DOIs in https form: {len(dois)}")
    if len(dois) < len(refs):
        issues.append(f"{len(refs) - len(dois)} reference(s) lack a DOI")

    for token in ("Divyansh", "divyanshsinghpost", "ORCID"):
        if token.lower() in full.lower():
            issues.append(f"author identifier '{token}' appears in the blinded body")

    for required in ("AUTHOR'S CONTRIBUTION", "CONFLICT OF INTEREST",
                     "FUNDING ACKNOWLEDGEMENT", "JEL", "Keywords"):
        if required.upper() not in full.upper():
            issues.append(f"missing required element: {required}")

    print("CHECKS")
    for n in notes:
        print(f"  - {n}")
    if issues:
        print("\nISSUES")
        for i in issues:
            print(f"  FAIL {i}")
        return 1
    print("\nAll journal requirements checked pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
