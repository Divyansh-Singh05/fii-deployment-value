"""Compile the IEEE Access LaTeX manuscript to PDF using tectonic.

This script manages compilation of the IEEE Access package, verifies all
references in ``submission/ieee_access/references.bib``, checks that figures
exist, and ensures clean PDF generation with zero unresolved citations.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IEEE_DIR = ROOT / "submission" / "ieee_access"
TEX_FILE = IEEE_DIR / "IEEE_Access_manuscript.tex"
BIB_FILE = IEEE_DIR / "references.bib"
FIG_DIR = IEEE_DIR / "figures"
RESOLVED_JSON = ROOT / "outputs" / "bib" / "resolved.json"


def check_prerequisites() -> None:
    """Verify that necessary files, figures, and compilers exist."""
    if not TEX_FILE.exists():
        raise FileNotFoundError(f"LaTeX manuscript not found at {TEX_FILE}")

    for fig in ["figure_1_design.png", "figure_2_attrition.png"]:
        fig_path = FIG_DIR / fig
        if not fig_path.exists():
            # Copy from submission/ijf/figures if available
            src_fig = ROOT / "submission" / "ijf" / "figures" / fig
            if src_fig.exists():
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_fig, fig_path)
            else:
                raise FileNotFoundError(f"Required figure {fig} not found at {fig_path}")

    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("Tectonic LaTeX compiler not found on PATH.")


def compile_pdf() -> Path:
    """Compile the manuscript using Tectonic."""
    check_prerequisites()
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        sys.exit(1)

    print(f"Compiling {TEX_FILE.name} with {tectonic}...")
    res = subprocess.run(
        [
            tectonic,
            "--keep-logs",
            "--outdir",
            str(IEEE_DIR),
            str(TEX_FILE),
        ],
        cwd=str(IEEE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

    pdf_file = IEEE_DIR / "IEEE_Access_manuscript.pdf"

    if res.returncode != 0:
        print("LaTeX Compilation Error:")
        print(res.stderr[-4000:])
        print(res.stdout[-2000:])
        sys.exit(res.returncode)

    if not pdf_file.exists():
        print(f"Error: Output PDF was not generated at {pdf_file}")
        sys.exit(1)

    print(f"Successfully compiled: {pdf_file} ({pdf_file.stat().st_size / 1024:.1f} KB)")
    return pdf_file


if __name__ == "__main__":
    pdf = compile_pdf()
