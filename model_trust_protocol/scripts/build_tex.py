"""Render the manuscript to LaTeX and compile it to PDF.

The LaTeX version is a reading copy: same text, same numbering, same exhibits,
typeset properly. The Word file remains the submission artifact, since the
journal requires it. Equations come from ``scripts.equations`` so the two
renderings cannot diverge.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from scripts.equations import EQUATIONS

SRC = Path("submission/ijf/manuscript.md")
TEX = Path("submission/ijf/IJF_manuscript.tex")
FIGDIR = Path("submission/ijf/figures")

PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{array}
\usepackage{tabularx}
\usepackage{ragged2e}
\usepackage[hidelinks]{hyperref}

\captionsetup[table]{labelsep=period,justification=raggedright,
  singlelinecheck=false,font={it},labelfont={it}}
\captionsetup[figure]{labelsep=period,justification=raggedright,
  singlelinecheck=false,font={it},labelfont={it}}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\doublespacing

\newcommand{\mainhead}[1]{%
  \par\vspace{10pt}\noindent\textbf{\MakeUppercase{#1}}\par\vspace{2pt}}
\newcommand{\subhead}[1]{%
  \par\vspace{6pt}\noindent\textbf{\textit{#1}}\par\vspace{1pt}}

\begin{document}
"""


def _esc(text: str) -> str:
    """Escape LaTeX specials, then restore the inline emphasis markers."""
    out = text
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        out = out.replace(a, b)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", out)
    return out


def _table(rows: list[list[str]], caption: str) -> str:
    """Render as a tabularx pinned to \\textwidth.

    Long row labels and long column headers overflowed a rigid l/r tabular and
    were clipped at the right margin. Every column is an X column so the text
    wraps; the label column is given 45% of the width and the value columns
    share the rest, with the \\hsize factors summing to the column count as
    tabularx requires.
    """
    header, *body = rows
    n_col = len(header)
    f_lab = 0.45 * n_col
    f_val = (0.55 * n_col) / (n_col - 1) if n_col > 1 else 1.0
    cols = (rf">{{\hsize={f_lab:.4f}\hsize\RaggedRight\arraybackslash}}X"
            + rf">{{\hsize={f_val:.4f}\hsize\RaggedLeft\arraybackslash}}X"
            * (n_col - 1))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
        rf"\caption{{{_esc(caption)}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{cols}}}", r"\toprule",
        " & ".join(rf"\textbf{{{_esc(h)}}}" for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(_esc(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def build() -> Path:
    lines = SRC.read_text().splitlines()
    out: list[str] = [PREAMBLE]
    pending: list[list[str]] = []
    caption = ""
    pending_fig_caption = ""
    i = 0

    def flush_table() -> None:
        nonlocal pending, caption
        if pending:
            out.append(_table(pending, caption))
            pending, caption = [], ""

    while i < len(lines):
        raw = lines[i].rstrip()
        i += 1

        if raw.startswith("|"):
            cells = [c.strip() for c in raw.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            pending.append(cells)
            continue

        eq = re.match(r"\[\[EQ:(\d+)\]\]", raw.strip())
        if eq:
            flush_table()
            e = EQUATIONS[int(eq.group(1))]
            out.append(
                "\\begin{equation}\n" + e.latex + "\n\\end{equation}"
            )
            continue

        fig = re.match(r"\[\[FIGURE:([^\]]+)\]\]", raw.strip())
        if fig:
            flush_table()
            out.append("\n".join([
                r"\begin{figure}[htbp]", r"\centering",
                rf"\caption{{{_esc(pending_fig_caption)}}}",
                rf"\includegraphics[width=\textwidth]{{figures/{fig.group(1)}}}",
                r"\end{figure}",
            ]))
            pending_fig_caption = ""
            continue

        # A table or figure title line becomes the caption of what follows.
        title = re.match(r"\*(Table|Figure) (\d+)\.\s*(.*?)\*$", raw.strip())
        if title:
            kind, _, body = title.groups()
            if kind == "Table":
                caption = body
            else:
                pending_fig_caption = body
            continue

        if raw.strip().startswith("*Source."):
            flush_table()
            out.append(r"{\footnotesize\textit{" + _esc(raw.strip("*")) + "}}")
            continue

        flush_table()
        if not raw.strip():
            continue
        if raw.startswith("# "):
            out.append(
                r"\begin{center}\textbf{\large " + _esc(raw[2:]) + r"}\end{center}"
            )
            continue
        if raw.startswith("## "):
            out.append(rf"\mainhead{{{_esc(raw[3:])}}}")
            continue
        if raw.startswith("### "):
            out.append(rf"\subhead{{{_esc(raw[4:])}}}")
            continue
        out.append(_esc(raw))

    flush_table()
    out.append(r"\end{document}")
    TEX.write_text("\n\n".join(out) + "\n")
    return TEX


def compile_pdf(tex: Path) -> Path | None:
    engine = shutil.which("tectonic")
    if engine is None:
        print("no LaTeX engine found; .tex written but not compiled")
        return None
    res = subprocess.run(
        [engine, "--keep-logs", "--outdir", str(tex.parent), str(tex)],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        print(res.stderr[-3000:])
        return None
    return tex.with_suffix(".pdf")


if __name__ == "__main__":
    t = build()
    print("written", t)
    pdf = compile_pdf(t)
    if pdf:
        print("written", pdf)
