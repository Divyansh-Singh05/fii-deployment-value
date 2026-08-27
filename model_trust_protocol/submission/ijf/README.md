# Indian Journal of Finance — submission package

| File | Role |
|---|---|
| `IJF_cover_page.docx` | Page 1: author details and the five copyright declarations. Three fields marked `[TO COMPLETE]` |
| `IJF_manuscript_blinded.docx` | **The submission artifact.** No author identifiers anywhere |
| `IJF_manuscript.pdf` | Reading copy, 22 pages, typeset from the same source |
| `IJF_manuscript.tex` | LaTeX source of the reading copy |
| `manuscript.md` | Single source. Every other file is generated from it |
| `figures/` | Greyscale PNGs at 600 dpi |

## Rebuilding

```bash
PYTHONPATH=. uv run python scripts/build_figures.py
PYTHONPATH=. uv run python scripts/build_docx.py
PYTHONPATH=. uv run python scripts/build_tex.py     # also compiles the PDF
PYTHONPATH=. uv run python scripts/check_ijf.py
```

Equations live in `scripts/equations.py` and are read by both builders, so the
Word and LaTeX renderings cannot drift apart. In Word they are centred with
italic variables, real subscripts and superscripts, and the number set at the
right margin; in LaTeX they are set in math mode.

## Compliance state

```
font Times New Roman 12pt, line spacing 2.0
margins 1.0 inch on all sides
body flows continuously; no forced page breaks
body words excluding references  5872   (limit 6000)
abstract words                    250   (limit 250)
tables 1-4 and figures 1-2, each cross-referenced in the text
figures greyscale, 600 dpi, numbered in order of appearance
equations (1)-(4), each cross-referenced in the text
references 27, DOIs in https form 27
```

`scripts/check_ijf.py` verifies each of these and fails on any breach. It also
confirms that no author identifier appears in the blinded file.

## Before submitting

1. **Complete the cover page.** Designation, institute name and full postal
   address with pincode, and ORCID identifier.
2. **Check three references by hand.** Every DOI resolved through Crossref
   under a title-similarity floor with preprints filtered out, but automated
   resolution is not a substitute for looking at the record. Priority:
   Aguilar-Loyo (2025), Waghmare and Ziegel (2026), Maheshwari and Naik (2026).
3. **Confirm the two exceptions to the recency window.** Hou et al. (2020) and
   Pénasse (2022) fall outside a strict three-year window. Hou is retained
   because the paper's framing rests on it.
4. **Read the manuscript once end to end.** Numerals, citations, headings,
   exhibits and formatting are checked mechanically. The argument is not.

## Numbers

Every numeral is bound to an artifact, a locator within it, and a recorded
digest. `convgap verify` reports 8/8 applications traced to a primary source,
`convgap facts` reports 32/32 data-section claims, and the manuscript scanner
confirms that no numeral in the prose is unaccounted for.
