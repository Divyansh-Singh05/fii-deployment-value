#!/usr/bin/env bash
# Build the print PDF + DOCX from the canonical thesis markdown.
# Usage: ./scripts/build_thesis_pdf.sh ["rev label"]
# The markdown (docs/paper/FII_thesis.md) is the source of truth; this
# applies print-only substitutions (glyphs Times New Roman lacks, the
# manual ToC pandoc replaces) without touching the source.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REV="${1:-$(date +%Y-%m-%d)}"

python3 - "$REPO" "$REV" <<'EOF'
import re, sys
from pathlib import Path
repo, rev = sys.argv[1], sys.argv[2]
t = Path(repo, "docs/paper/FII_thesis.md").read_text()
t = t.replace("⇒", "→").replace("₹", "Rs.")
t = re.sub(r"^# Foreign Institutional Flow Regimes.*\n## A complete account.*\n", "", t, count=1)
t = re.sub(r"## Table of contents.*?\n---\n---\n", "", t, flags=re.S, count=1)
t = t.replace("β₁₂₀", "$\\beta_{120}$").replace("vol₂₀", "$vol_{20}$")
t = t.replace("threshold₊ᵇ", "threshold(+, buy)")
t = t.replace("threshold₊", "threshold(+)").replace("threshold₋", "threshold(−)")
t = t.replace("∧", "and").replace("≫", ">>").replace("q₍29₎", "q(0.29)")
t = t.replace("β_SD", "$\\beta_{SD}$").replace("β_HO", "$\\beta_{HO}$")
t = t.replace("\\sum_{S \\subseteq F\\setminus\\{j\\}}",
              "\\sum_{S \\subseteq F,\\ j\\notin S}")
for a, b in zip("₀₁₂₃₄₊₋", "01234+-"):
    t = t.replace(a, b)
meta = ('---\ntitle: "Foreign Institutional Flow Regimes in Indian Equities"\n'
        'subtitle: "A complete account: motivation, construction, mathematics,'
        f' validation, and economic meaning"\ndate: "{rev}"\n---\n\n')
Path("/tmp/FII_thesis_print.md").write_text(meta + t)
EOF

cd "$REPO/docs/paper"
pandoc /tmp/FII_thesis_print.md -o FII_thesis.pdf --pdf-engine=tectonic \
  --toc --toc-depth=1 -V geometry:margin=2.4cm -V fontsize=11pt \
  -V mainfont="Times New Roman" -V monofont="Menlo" -V linkcolor=black \
  2>&1 | { grep -i "could not represent" || true; }
pandoc /tmp/FII_thesis_print.md -o FII_thesis.docx --toc
echo "built: docs/paper/FII_thesis.pdf + .docx  (rev: $REV)"
