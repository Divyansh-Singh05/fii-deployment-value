# Manuscript

Working title: **What Is an Alternative Dataset Worth? Mapping the friction
boundary of institutional flow data**

| File | Status |
|---|---|
| `00_abstract.md` | drafted — all numerals traced |
| `01_introduction.md` | drafted — all numerals traced |
| `02_data.md` | drafted — all numerals traced |
| `03_methodology.md` | drafted |
| `related_work.md` | drafted — verified citations only, with a quarantine list |
| `related_work_candidates.md` | 35 candidates from a 2026-08-25 search; **none read**, status flagged per entry |
| `04_results.md` | drafted — all numerals traced |
| `05_discussion.md` | drafted |
| `06_limitations.md` | drafted — includes the framework applied to this paper |
| `07_conclusion.md` | drafted |

## Number discipline

Every numeral in the manuscript must clear four gates before it is admitted:

1. **fresh** — the artifact is not older than the code that writes it;
2. **primary** — the source is the computation, not prose describing it and not
   a copy of its output;
3. **traced** — the artifact digest matches `outputs/evidence.lock.json`;
4. **extracted** — the artifact still holds the declared value.

A fourth check applies the same standard to the prose: `convgap manuscript`
extracts every numeral from the manuscript and reports any that is neither
verified evidence nor exempt with a stated reason. Exemptions are enumerated in
code, not assumed.

Current state, from the repository root:

```
convgap verify      8/8  applications publishable
convgap facts      32/32 data-section claims traced
convgap manuscript  every numeral verified or explicitly exempt
```

Everything regenerates from here. `convgap reproduce` re-runs the nine
producing computations across both source trees (~22 min); `convgap replicate`
regenerates the three quantities this repository derives itself — the daily
aggregate-flow screen, the weekly risk screen, and the descriptives table.
