#!/bin/bash
# Build a Word version of the manuscript.
#
# The IEEE Access class uses macros pandoc cannot interpret (\authorrefmark,
# \address, \markboth, \corresp, \tfootnote, \IEEEbiographynophoto,
# \IEEEPARstart, \appendices). This script rewrites them into plain LaTeX,
# resolves citations through citeproc against references.bib, and embeds all
# figures. Run from submission/ieee_access/.
set -euo pipefail
python3 tools_docx_prep.py
pandoc _docx_source.tex -o IEEE_Access_manuscript.docx \
  --citeproc --bibliography=references.bib --csl=ieee-numeric.csl \
  --extract-media=media \
  --metadata title="From Statistical Evidence to Decision Value: Six Deployment Constraints on an Institutional Flow Dataset" \
  --metadata author="Divyansh Singh; Aryan Malviya; Rishabh Masuriya; Pravin Patil" \
  --metadata institute="Department of Artificial Intelligence and Data Science, K. J. Somaiya Institute of Technology, Mumbai 400077, India"
rm -f _docx_source.tex
echo "wrote IEEE_Access_manuscript.docx"
