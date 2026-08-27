"""Rewrite the IEEE Access source into plain LaTeX that pandoc can read.

The class defines macros pandoc has no handler for. Each is either dropped
(page furniture that has no Word equivalent) or rewritten into a construct
pandoc understands. Citations are left as \\cite{} so citeproc can resolve
them against references.bib.

Run via make_docx.sh; writes _docx_source.tex.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("IEEE_Access_manuscript.tex")
OUT = Path("_docx_source.tex")


def drop_cmd(s: str, name: str) -> str:
    """Remove \\name{...}, honouring nested braces."""
    key = "\\" + name + "{"
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len(key), 1
        while k < len(s) and depth:
            depth += (s[k] == "{") - (s[k] == "}")
            k += 1
        i = k


def main() -> None:
    s = SRC.read_text()

    # page furniture with no Word equivalent
    for c in ("history", "doi", "markboth", "corresp", "tfootnote",
              "title", "author"):
        s = drop_cmd(s, c)
    s = re.sub(r"\\address\[1\]\{.*?\n", "", s, flags=re.S)
    s = re.sub(r"\\titlepgskip=[^\n]*\n", "", s)
    s = s.replace(r"\EOD", "").replace(r"\maketitle", "")
    s = s.replace(r"\documentclass{ieeeaccess}", r"\documentclass[11pt]{article}")

    # drop-cap opener -> ordinary word
    s = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}",
               lambda m: m.group(1) + m.group(2).lower(), s)

    # environments pandoc does not know
    s = s.replace(r"\begin{keywords}", "\\section*{Index Terms}\n")
    s = s.replace(r"\end{keywords}", "")
    s = s.replace(r"\appendices", r"\section*{Appendix}")

    # biographies -> headed paragraphs. These sit AFTER the bibliography in the
    # source; they must survive the bibliography swap below.
    s = re.sub(r"\\begin\{IEEEbiographynophoto\}\{(.+?)\}",
               r"\\subsection*{\1}", s)
    s = s.replace(r"\end{IEEEbiographynophoto}", "")

    # hand the reference list to citeproc, WITHOUT taking the biographies with it
    s = s.replace("\\bibliographystyle{IEEEtran}\n\\bibliography{references}",
                  "\\section*{References}")

    OUT.write_text(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
