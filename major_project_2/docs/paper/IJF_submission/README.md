# IJF Submission Package

Two LaTeX files, written to the Indian Journal of Finance author guidelines.

| File | Purpose |
|---|---|
| `IJF_cover_page.tex` | Page 1 only: author details + copyright declaration. Submitted separately. |
| `IJF_manuscript.tex` | The blinded paper. Contains **no** author name or affiliation. |

## Compliance status

| Requirement | Status |
|---|---|
| Times New Roman, 12pt, black | `newtxtext` (Times metric) at `12pt` |
| Double spaced | `\doublespacing` (setspace) |
| 1-inch margins | `geometry, margin=1in` |
| Word limit 6,000 | **4,540** words (4,115 excluding references) |
| Abstract ≤ 250 words | **223** words, past tense, all five required components |
| Keywords + ≥3 JEL codes | 6 keywords; JEL C58, G15, G17 |
| Author details on cover page only | Body verified free of name/affiliation |
| Tables numbered consecutively | Tables 1–5, titles above, italicized, source noted below |
| Tables as real tables | Real `tabular` cells; column counts verified |
| Table numbers cited in text | All five cited by number; no "above/below table" phrasing |
| Primary headings UPPERCASE bold | Yes |
| Sub-headings Title Case bold italic | Yes |
| Equations numbered | (1), (2), (3) |
| Ethical disclosures before references | Author's Contribution, Conflict of Interest, Funding |
| References APA 7th, alphabetical, all cited | 18 entries, verified all cited in text |
| DOIs in https format | **18/18** — every DOI verified against Crossref |
| No figures | None used (guidelines ask for minimum exhibits) |

## Placeholders to fill before submission

In `IJF_cover_page.tex`: designation, institute name and full address with
pincode, ORCID iD.
In `IJF_manuscript.tex`: the three date fields near the abstract, and the name
in the Author's Contribution statement.

## Build outputs (already generated)

| File | Notes |
|---|---|
| `IJF_manuscript.pdf` | 17 pages, compiled clean, **zero overfull boxes** |
| `IJF_cover_page.pdf` | 2 pages (details p.1, declaration p.2) |
| `IJF_manuscript.docx` | **submit this one** — IJF requires MS-Word |
| `IJF_cover_page.docx` | submitted separately |
| `ijf_reference.docx` | pandoc style template; not submitted |

### Rebuilding after an edit

```bash
tectonic IJF_manuscript.tex                       # -> PDF
pandoc IJF_manuscript.tex --reference-doc=ijf_reference.docx \
       -o IJF_manuscript.docx                     # -> DOCX
python3 fix_docx_margins.py                       # re-inject 1in margins
```

The margin step is required because pandoc writes its own section properties
and ignores the reference document's page margins.

### Verified in the generated .docx

Font uniform Times New Roman; size uniform 12pt; double spacing (480 twips);
1-inch margins on all four sides; no theme colours and every run black;
5 tables preserved; 4,573 words; no author identifier present.

## Provenance of every number in the paper

| Paper location | Source |
|---|---|
| Table 1, Table 2 | `outputs/phase3/C7_RISK_REPORT.md` |
| Table 3 | stage `institutional_share` (`module19_institutional_share.py`) |
| Table 4 | stage `market_flow_engine` (`module21_market_flow_engine.py`) |
| Table 5 | stages `market_flow_engine_v2/v3`, `weekly_engine` (modules 22–24) |
| Pre-registrations | `docs/PREREG_AGGREGATE_FLOW.md`, `PREREG_MARKET_ENGINE.md`, `PREREG_MARKET_ENGINE_V2.md`, `PREREG_MARKET_ENGINE_V3.md`, `PREREG_WEEKLY_ENGINE.md` |
| Internal audit paragraph | `docs/AUDIT_RETRACTIONS.md` R6 |
| Cost/backtest sentence | `outputs/tables/T6_backtest_metrics.csv` (regenerated 2026-08-23) |

## Scope note

This paper deliberately excludes the archetype/concentration-axis economic
claims. Those were retracted in `docs/AUDIT_RETRACTIONS.md` R1 and no
replacement headline is asserted, so nothing from that line of work appears
here. The paper is built only on results that survived the audit.
