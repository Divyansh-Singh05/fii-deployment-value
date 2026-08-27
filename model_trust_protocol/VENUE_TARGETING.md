# Venue targeting and repository selection

A reusable brief for starting a new paper from this body of work. Part A is
verified fact about the three repositories. Part B is the redundancy discipline.
Part C is the prompt to run at the start of a new paper.

Verified 2026-08-27 by direct file comparison, not by recollection.

---

## Disclaimer — read before touching any code

**These trees contain redundant, superseded and near-duplicate code. Some of it
looks current and is not.** Three specific traps:

1. **The same stage exists in two trees under two names.** The stock-day engine
   is `s01`–`s13` in `fii_risk_engine` and `module_c1`–`module_c9` in
   `Major Project 2/src/fii/phase3`. They are *not* copies — two have diverged
   scientifically.
2. **Modification time lies.** Several `fii_risk_engine` files are newer by
   timestamp but older by content; the later edits were import-path rewrites.
3. **Numbered variants are not ordered by recency.** `module21`, `module22`,
   `module23`, `module24` are four *different* engines (EWMA base, GJR base,
   globals, weekly), not four revisions of one. Likewise `_v1/_v2/_v3` suffixes
   on data files.

**Before writing code against any module, or quoting any number produced by
one, confirm with the user which module is the intended source.** Ask once,
early, naming the specific candidates. Do not infer it from filenames, from
version suffixes, or from modification dates. A wrong choice here does not
fail loudly — it produces a plausible number that is quietly wrong, which is
the exact failure mode this research programme documents in its own source
material.

---

## Part A0 — Repository anatomy

Read this before searching for code, so you look in one place rather than three.

### `Major Project 2` — the research tree (8.8 GB, 208 modules)

The authoritative source for every empirical result. Layout:

```
src/fii/
  data_prep/    price panel, corporate-action factors, ISIN point-in-time map
  features/     the stock-day feature store
  models/       HMM variants (gap_aware, factorial, regime), LightGBM, registry
  validation/   module5b4, module6..module24 — the numbered validation studies
  phase2/       causal filtering, calibration, hazard, decision layer
  phase3/       module_c1..c9 — the stock-day density stack (NEWEST version)
  backtest/     engine, engine_gates, module12b..12e strategy books
  reporting/    collect_outputs, make_exhibits
data/
  ISIN_MAPPING/     1.6 GB — identity map, corporate actions, state objects
  VALIDATION_DATA/  1.0 GB — returns panels, hazard preds, macro, deals
outputs/        metrics, predictions, phase3, tables, validation, logs, figures
docs/           00..05 numbered guides, AUDIT_RETRACTIONS.md, data dictionary
legacy/         DO NOT USE — superseded Colab modules
```

**Where things actually are.** Equation (1)'s panel: `validation/module19_institutional_share.py`.
Market-level flow engines: `module21` (EWMA base), `module22` (GJR base),
`module23` (globals), `module24` (weekly control) — four distinct engines.
Strategy books: `backtest/module12b_strategies_base.py`. Breakeven and turnover:
`backtest/engine.py`. Read `docs/AUDIT_RETRACTIONS.md` before trusting any
figure you find in an older document.

### `fii_risk_engine` — the standalone engine (428 MB, 19 modules)

Flat, self-contained, runnable: `s01_feature_store` → `s13_lookahead_audit`,
plus `gap_aware_hmm`, `isin_pit_map`, `lineage`, `paths`, `trading_calendar`,
`panel_inference`. Its own `outputs/`. Ships `DATA.md` and `LIMITATIONS.md`.

Its README states it is self-contained: every module it imports lives in that
directory, everything it writes goes to its own `outputs/`, and the only
external dependency is the read-only market and flow data.

Use it when you want the HMM/density pipeline to run without the research tree.
**Do not use it to claim a current result** — it predates the EVT tail benchmark
and the institutional-share volatility correction.

### `model_trust_protocol` — the publication tree (757 MB)

```
src/convgap/    the provenance tooling: evidence, extract, lockfile,
                provenance, manuscript, facts, friction, registry, cli
paper/          the research-version manuscript sections (00..07)
paper_v2/       round-4 re-derivations and the SOURCES map
submission/
  ieee_access/  LaTeX source, figures, bibliography, docx build scripts
  ijf/          the finance-journal version of the same study
case_studies/fii   the applications registry
outputs/        evidence.lock.json, exhibits, bib
templates/      venue templates
```

`convgap` is the verification layer: it binds every reported numeral to a
producing artifact, a SHA-256 digest, and an extractor that reads the value
back out. `convgap verify`, `convgap facts`, `convgap manuscript`.

The separate reproduction package `FIHMM_Deployment_Value` (self-contained,
24 MB of data, no external paths) is what a reviewer runs.

---

## Part A — Which repository to use

### The finding that matters most

**`Major Project 2` holds the newest engine code, not `fii_risk_engine`.**
This is the opposite of the natural assumption, and it was checked stage by
stage rather than by timestamp alone, because timestamps mislead here.

All nine overlapping stages of the stock-day risk engine exist in both trees:

| Stage | `fii_risk_engine` | `Major Project 2` | Real divergence? |
|---|---|---|---|
| refit harness | `s05_refit_harness` | `module_c1_refit_harness` | paths only |
| daily filter | `s06_daily_filter` | `module_c2_daily_filter` | paths only |
| archetype probs | `s07_archetype_probs` | `module_c3_archetype_probs` | **identical** |
| outcome densities | `s08_outcome_densities` | `module_c4_outcome_densities` | **YES — 188 lines** |
| predictive mixture | `s09_predictive_mixture` | `module_c5_predictive_mixture` | **YES — 20 lines** |
| validation | `s10_validation` | `module_c6_validation` | paths only |
| risk profiles | `s11_stock_risk_profiles` | `module_c7_stock_risk_profiles` | paths only |
| risk engine | `s12_risk_engine` | `module_c8_risk_engine` | paths only |
| look-ahead audit | `s13_lookahead_audit` | `module_c9_lookahead_audit` | paths only |

Most differences are layout: `fii_risk_engine` is the flat standalone version
(`from paths import ...`), `Major Project 2` is the packaged version
(`from fii.paths import ...`). Two are **substantive science that exists only
in Major Project 2**:

- `module_c4_outcome_densities` adds an **institutional-share volatility
  correction** and an **EVT tail benchmark** (188 lines absent from `s08`).
- `module_c5_predictive_mixture` adds **`evt`** to the benchmark set — a pooled
  empirical body with a GPD tail spliced below the 5th percentile, fitted per
  vintage on matured past outcomes. Against `clim_roll`, which reads its tail
  off the histogram, this isolates whether *extrapolating* the tail beats
  *counting* it.

A few `fii_risk_engine` files carry later modification times. Those edits are
import-path rewrites, not science. **Modification time is not a version signal
in these trees; diff the content.**

### Selection rules

| Repository | Use it for | Do not use it for |
|---|---|---|
| **`Major Project 2`** | Any new empirical result. All validation modules (`module7`–`module24`), the phase-3 density stack, backtests, the 25.16m-trade record, all derived panels. The full research history. | Anything where you need a clean minimal runnable artifact — it is 8.8 GB with heavy redundancy. |
| **`fii_risk_engine`** | A standalone, flat-layout, runnable copy of the stock-day stack. Useful when you want the HMM/density pipeline without the whole research tree. | Claiming a current result. It predates the EVT benchmark and the share-volatility correction. |
| **`model_trust_protocol`** | Writing, submission, provenance. Manuscripts, LaTeX/Word builds, figure generation, the `convgap` verification tooling, the reproduction package. | Computing new empirical results. It consumes artifacts; it does not produce the science. |

**Default:** compute in `Major Project 2`, write in `model_trust_protocol`,
reach for `fii_risk_engine` only when a self-contained engine copy is the point.

---

## Part B — Redundancy discipline

### Risk ranking, by count of redundancy-flagged artifacts

| Repository | Flagged artifacts | Audit docs | Risk |
|---|---:|---:|---|
| **`Major Project 2`** | **66** | 25 | **HIGH** |
| `fii_risk_engine` | 10 | 13 | LOW |
| `model_trust_protocol` | 3 | 1 | LOW, but see below |

`Major Project 2` carries 11 `PREAUDIT`, 3 `BASELINE`, 11 `_smoke`, 10 `_trunc`,
1 `_unit`, 28 `_v1/_v2/_v3`, a `legacy/` tree, and files ending `(1)` and `(2)`.

### The rule

Never read a path containing `PREAUDIT`, `BASELINE`, `_smoke`, `_trunc`,
`_unit`, or living under `legacy/`. Current artifacts are the same path with
the infix removed. Where several `_v1/_v2/_v3` exist, confirm which the
producing stage actually writes — do not assume the highest number.

This is not hygiene theatre. The programme's own provenance protocol found
**three instances of stale material reaching a reported number**: two
pre-registered gating regressions with no producing stage, a metrics table
quoted from a copy six weeks behind its origin, and a stage whose reported
effect size was a third larger than its current object supported because it was
never re-executed after an audit. None changed a verdict. All changed a number
that had already been written down.

A digest match on a stale copy still passes. That is why *fresh* is a separate
gate from *traced*.

### The trap specific to `model_trust_protocol`

Low artifact count, but it has carried **two divergent manuscript versions of
the same paper**. `paper/*.md` once sat a full correction round behind
`submission/`, still containing numbers that re-derivation had disproved
(`-2.35`/`-0.95` for a pair that was never controlled; `-0.32` for a lag row
that is `-0.43`). Before quoting any number from a manuscript file, confirm it
against the most recently built submission, or against the reproduction
package's `verify_results.py`.

### Standing check before any new paper

1. `python reproduction/verify_results.py` in `FIHMM_Deployment_Value` — every
   headline number recomputes from `data/` and is compared to the manuscript.
2. Diff, do not date. If two trees hold the same stage, diff them and separate
   import-path noise from science.
3. Treat any number you cannot trace to a producing stage as not yet real.

---

## Part C — The venue-targeting prompt

Paste this at the start of a new paper effort.

### Step 1 — Characterise the venue before writing a word

Answer all eight. If any answer is a guess, go and check.

1. **Discipline and reader.** Engineering, finance, management, statistics, or
   data science? Who is the modal reviewer, and what will they not tolerate?
2. **Indexing and tier.** Scopus / SCIE / ABDC / ABS / UGC-CARE? Impact factor
   or SJR quartile? This sets how much novelty is required versus how much
   rigour.
3. **Hard limits.** Word or page ceiling, figure and table ceiling, reference
   count, colour policy, supplementary policy.
4. **Mandated structure.** Structured abstract? Named sections
   (Research Gap, Objectives, Managerial Implications)? Ethics and
   contribution statements?
5. **Citation style.** Numeric bare `[1]`, numeric with names, or author–date?
   Are DOIs printed?
6. **What the journal actually publishes.** Read three recent issues. What
   methods recur? What claim shape is rewarded?
7. **Tolerance for negative results.** Does the journal publish papers whose
   headline is that something failed? If not, the survivor must lead.
8. **Review model and timeline.** Double-blind? Submission windows? APC?

### Step 2 — Choose the angle the venue rewards

The same research supports several papers. Pick the one that fits.

| Venue type | Foreground | Background | Title verb |
|---|---|---|---|
| Engineering / CS (IEEE Access, IEEE TKDE) | The **architecture** and the validation protocol as reusable engineering | The market-specific economics | "constraints", "framework", "validation" |
| Finance (JEMF, JAM, PBFJ, FRL) | The **economic** finding and identification | The pipeline engineering | "evidence", "does X predict Y" |
| Management / practitioner (IJF, IIMB) | The **decision implication** for a named actor | Machinery, into an appendix | "implications for", "what X is worth" |
| Data science / meta-science | The **provenance protocol** and the reproducibility result | The asset class | "reproducibility", "audit" |
| Statistics / econometrics | The **inference problem** (nested-model gate, multiplicity) | The application | "testing", "inference under" |

### Step 3 — Restructure, but never re-number

**Change freely:** section names and order; framing of the contribution; which
result leads; depth of methodological exposition; what moves to an appendix;
citation style; abstract format; title.

**Never change:** any reported number, any verdict, any stated limitation.
If a venue's framing requires suppressing a failure, that is the wrong venue.

### Worked example — the same study at two venues

| | *Indian Journal of Finance* | *IEEE Access* |
|---|---|---|
| Words | 6,815 | 6,403 |
| Literature section | "REVIEW OF LITERATURE" with explicit **Research Gap** and **Objectives of the Study** | "Related Work", thematic, no gap/objective headings |
| Framework section | folded into Methodology | standalone **"The Six Deployment Constraints"**, formalised with numbered equations |
| Implications | "MANAGERIAL AND PRACTICAL IMPLICATIONS" | "Managerial and **Engineering** Implications" |
| Limitations | "LIMITATIONS OF THE STUDY AND SCOPE FOR FURTHER RESEARCH" (mandated heading) | "Limitations and Future Research" |
| Front matter | structured abstract, JEL codes | unstructured abstract, IEEE index terms |
| Citations | author–date in text | bare numeric `[1]` |
| Figures | few, greyscale, simple (journal norm) | 11 figures incl. a four-figure appendix |
| Extra sections | Author's Contribution, Conflict of Interest | Acknowledgment, biographies |

Identical science. The framework was **promoted to its own formalised section
with equations** for the engineering venue and **demoted into Methodology** for
the finance venue; "Practical" became "Engineering"; the figure budget tripled.
Not one number moved.

### Step 4 — Fit check before submitting

- Can you name three papers in the last two years of this journal that a
  reviewer would see as neighbours? If not, the fit is wrong.
- Does the contribution statement use the vocabulary the journal uses?
- Is every hard limit met — words, figures, references, structure?
- Does the lead result match what this journal rewards?
- Would a reviewer from this field have the machinery to referee the methods?
  If not, either simplify or change venue. Do not submit and hope.
