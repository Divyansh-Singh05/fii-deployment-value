# Verification log

Dated record of what changed as each application was promoted from a project
document to the artifact that produced its numbers. Append-only.

## Gates

A value may enter an exhibit only when all of the following hold:

1. **fresh** — the artifact is not older than the code that writes it, and no
   stage upstream of it has been re-run since;
2. **primary** — the source is the computation, not prose describing it, and
   not a *copy* of the computation's output;
3. **traced** — the artifact digest matches the one in
   `outputs/evidence.lock.json`;
4. **extracted** — where an extractor is defined, the artifact still holds the
   declared value, or the status is `MISMATCH`.

Gate 1 was added after V1 and is the reason V1 had to be reversed.

---

## V1 · A5 (portfolio) — 2026-08-24 — **REVERSED, see V1a**

Promoted `docs/paper/FII_final_results_one_pager.md` §6 →
`outputs/metrics/backtest_metrics.csv`, and recorded a discrepancy: the
artifact held 1.44 / −1.54 where the document claimed 1.51 / −1.53. The
conclusion drawn was that the document was a stale transcription.

**That conclusion was wrong.**

## V1a · A5 — correction — 2026-08-24

`outputs/metrics/backtest_metrics.csv` is not a primary artifact. It is a
**copy**, written by `src/fii/reporting/collect_outputs.py:46` from
`outputs/tables/T6_backtest_metrics.csv`. The two had diverged:

| Artifact | Written | S3_PROXY TEST gross | net @ 15 bp |
|---|---|---:|---:|
| `outputs/tables/T6_backtest_metrics.csv` | 2026-08-23 17:15 | **1.51** | **−1.53** |
| `outputs/metrics/backtest_metrics.csv` | 2026-07-12 14:59 | 1.44 | −1.54 |

The exhibit table was regenerated on the post-audit object; the copy in
`metrics/` was left behind from a July run. The one-pager agrees with the
current exhibit table. `docs/05_backtesting.md`, which quotes 1.46/1.44, is
also pre-audit.

**Three lessons, all now enforced:**

- A digest and an extractor are not sufficient. Both passed on a file that was
  six weeks stale. Freshness relative to the producing code is an independent
  gate (gate 1 above).
- "Primary" must exclude *copies*. A file written by a reporting step from
  another file is one remove from the computation and can drift from it.
- The direction of a discrepancy cannot be assumed. The document was right.

**Action taken.** The backtest phase was re-executed
(`pipeline.py --phase backtest`, 2026-08-24 21:53, all five stages PASS),
which rewrote `bt12_baselines/hmm/style.parquet`. `T6_backtest_metrics.csv` is
therefore now itself stale relative to that run, and A5's numbers are **not
settled**. The exhibits phase must be re-run before A5 can be promoted.

**Status: superseded by V1b.**

---

## V1b · A5 — settled — 2026-08-24

`pipeline.py --phase exhibits` re-run (paper_exhibits 108.0s PASS,
collect_outputs 0.6s PASS). `T6_backtest_metrics.csv` regenerated from the
backtest re-run; the copy in `outputs/metrics/` is now byte-identical to it.

A5 re-pointed at **`outputs/tables/T6_backtest_metrics.csv`** — the table the
reporting stage writes directly — with `producer=src/fii/reporting/make_exhibits.py`
so the freshness gate applies. Six values, all confirmed by extraction:

| | TRAIN | TEST |
|---|---:|---:|
| gross Sharpe (cost 0) | 1.36 | **1.51** |
| breakeven one-way cost (`margin_bps`) | 7.33 bp | **7.41 bp** |
| net Sharpe @ 15 bp | −1.42 | **−1.53** |
| n | 2,530 | 920 |

**Both objections raised in V1 were wrong.** The project document's figures
(1.51 / −1.53) were correct throughout. And S3_PROXY is the highest gross
Sharpe of all eight books in the test era, not merely the best no-model book,
so the selection claim was correct too. Both errors traced to the same stale
copy.

**Status: A5 `verified` / `primary` — publishable. 1 of 8.**

---

## Gate added as a result: freshness

`VerificationStatus.OUTDATED`. A `Source` may now name the code that writes it;
if that code is newer than the artifact, verification fails before the digest
or the value is examined. This is the only gate that would have caught V1: the
stale copy passed both the digest check and the extractor.

Two rules follow and are enforced in the code:

- **Never cite a copy.** `outputs/metrics/backtest_metrics.csv` is written by a
  reporting step from `outputs/tables/T6_backtest_metrics.csv`. One remove from
  the computation is one remove too many.
- **Re-run before verifying.** An artifact is a claim about a moment. Where a
  stage can be re-executed, it is, and the digest is recorded against the fresh
  output.

---

## V2 · Framing change — 2026-08-24

The object of study was fixed as **the dataset**, not any model built on it.
Models are stress-test environments imposing ascending levels of institutional
friction. The package was restructured accordingly:

- `Channel` → `Application`; each declares the friction level at which its
  value is lost.
- Added `FrictionLevel`: existence, replication, availability, competition,
  detectability, execution.
- Added `Role`: a non-flow predictor (A3, the S&P → Nifty weekly spillover)
  is a **control**, not an application. A control dying at the same level
  indicates the friction belongs to the evaluation transition rather than to
  this dataset.
- Table 1 is ordered as a descent and reports attrition per level.

**First reading of the ladder, on unpromoted numbers:**

| Level | Applications lost |
|---|---:|
| 0 existence | 0 |
| 1 replication | 0 |
| 2 availability | 1 |
| 3 competition | 2 |
| 4 detectability | 2 |
| 5 execution | 1 |
| survives all | 1 of 7 |

Nothing died at levels 0 or 1. Every application of the record was
statistically real and replicated out of sample. All attrition occurred under
deployment friction. Subject to promotion of the remaining seven sources.

---

## V2 · A8 and A2 (market-level density) — 2026-08-24

Both stages re-executed before verification rather than trusting stored output:

| Stage | Log | Wall | Verdict |
|---|---|---:|---|
| `market_flow_engine` (module 21) | `20260824_223938_...log` | 62.9 s | GATE **PASS** |
| `market_flow_engine_v2` (module 22) | `20260824_224048_...log` | 135.1 s | GATE **FAIL** |

The stored parquets were dated 2026-08-23 16:32 with the newest matching run log
at 14:37 — a run had written the artifact without leaving a log in place. Both
were re-run to obtain a consistent artifact/log pair.

**Value halves verified, all eight figures confirmed by extraction:**

| | A8 (EWMA base) | A2 (GJR base) |
|---|---:|---:|
| leg (a) DM t, full | **−2.35** (bar ≤ −2.0) PASS | **−0.95** FAIL |
| leg (b) CRPS ×100 TRAIN | −0.0019 | −0.0008 |
| leg (b) CRPS ×100 TEST | −0.0004 | **+0.0004** — sign flip, FAIL |
| leg (c) Kupiec p5 / p1 | 0.225 / 0.161 PASS | — |
| scored days | 2,739 | 2,235 |

The two differ only in the baseline density the same flow tilt is asked to
improve. That is the controlled test of the competition level, and the
difference in scored days (2,739 vs 2,235) is itself part of the mechanism: the
GJR base requires a longer burn-in.

**Transcription error caught.** An earlier draft recorded the two Kupiec
p-values as TRAIN and TEST. They are the **5% and 1% VaR levels on the full
sample**. Corrected.

## V2a · BLOCKER — the significance half has no producing artifact

The C1–C4 screen regression is the pre-registered finding that motivates the
entire market-engine family, and supplies the significance half of both A8 and
A2. **No pipeline stage computes it.**

Searched: `src/fii/stages/registry.py` has no stage for it; `grep` across
`src/` finds the figures only in the *docstring* of
`module21_market_flow_engine.py:6-8`, which cites
`docs/PREREG_AGGREGATE_FLOW.md`. The prereg document states the specification
and the bar; it does not contain the result. No log, table or parquet produces
t = −4.78 / −2.49 / −2.51.

The regression was evidently run outside the pipeline and never brought into
it. Its figures have propagated into the module docstring, the results summary
and the thesis, none of which is a primary source.

**This blocks A8 and A2 at the `primary` gate**, and A8 is the survivor — the
row the paper's boundary claim rests on.

**Recommended remedy.** The pre-registration specifies the construction
completely: NF(t) as buy minus sell VALUE_INR over `TR_TYPE ∈ {1,4}`,
`RATE > 0`, `REG_DL_INSTR_EQ`, scaled by the trailing 250-day mean of daily
gross flow ending t−1; target |nifty_ret(t+1)|; controls five lags of the
target plus india_vix(t); NEG(t) = min(NF(t), 0); Newey–West 10 lags; the
frozen era split. Re-implementing C1–C4 inside this repository, reading the
same source data, would produce the figure as a `DERIVED` value computed by
code a referee can inspect — strictly better provenance than the current
inherited number, and it keeps the source trees read-only.

**Status: A8 and A2 blocked pending a decision on the C4 re-implementation.**

---

## V2b · A8 and A2 unblocked — the screen re-derived — 2026-08-24

The C1–C4 screen was re-implemented in this repository from the pre-registered
specification (`src/convgap/replication/aggregate_flow.py`), reading the same
raw depository records and market series. Source trees remain read-only.

**Result — C4 passes, replicating the inherited conclusion.**

| cell | description | FULL t | TRAIN t | TEST t | verdict |
|---|---|---:|---:|---:|---|
| C1 | direction, same-day | +0.35 | +0.92 | −0.10 | FAIL |
| C2 | risk, same-day | −2.12 | +0.49 | −0.42 | FAIL |
| C3 | direction, deployable | −0.29 | −1.72 | +1.98 | FAIL |
| **C4** | **risk, deployable** | **−4.83** | **−2.53** | **−2.51** | **PASS** |

Against the inherited −4.78 / −2.49 / −2.51. Same sign, same verdict, both legs
of the bar cleared. One of four cells passes, as reported.

**An ambiguity in the inherited specification, found by re-implementing it.**
The pre-registration does not say what becomes of days on which no flow was
observed — the two months masked at source and the three carrying a null
direction flag. Two readings are admissible:

| reading | FULL t | TRAIN t | TEST t | n | verdict |
|---|---:|---:|---:|---:|---|
| exclude (canonical) | −4.83 | −2.53 | −2.51 | 3,178 | PASS |
| retain as NF = 0 | −4.90 | −2.84 | −2.60 | 3,485 | PASS |

The choice moves the training-era statistic by 0.31 t-units and changes no
verdict. We take exclusion as canonical — a day with no observation has no
measurement rather than a measurement of zero — and report both. The engine
module's own construction uses the second reading, which is why it is worth
stating rather than resolving silently.

**Method note.** The screen is the only figure in the exhibit computed by this
repository rather than read from the source programme. Its tests plant a known
effect in synthetic data at the specified lag and confirm the estimator
recovers it, confirm a pure-noise regressor does not clear the bar, and confirm
that a sign flip between eras fails regardless of t.

**Status: A2, A5, A8 `verified` / `primary` — 3 of 8 publishable, including
the survivor.**

---

## V3 · A1 · institutional share — 2026-08-24

Stage `institutional_share` re-run (71.8 s, PASS). All six figures confirmed by
extraction against the declared values: stock+date FE t = **−5.53 / −4.42 /
−3.25**, and at t−2 **−0.32 / −0.30 / −0.13**. No discrepancy.

## V4 · A3 · the control — 2026-08-24

Stage `weekly_engine` re-run (67.5 s, PASS). Engine gate confirmed: DM
t = **+1.58**, TRAIN +0.0020, TEST +0.0000, GATE FAIL.

**The screen has no producing stage either** — the same gap as C4, and the same
shape: the engine has a pipeline stage, the pre-registered regression that gates
it does not. Its result lives in a section appended to the pre-registration and
in the consuming module's docstring.

Re-derived in `src/convgap/replication/weekly_screen.py`. **All three cell
verdicts replicate, including the reasons the two failures fail:**

| cell | predictor | FULL | TRAIN | TEST | verdict | inherited verdict |
|---|---|---:|---:|---:|---|---|
| W1 | flow persistence | −0.96 | −3.85 | +1.14 | FAIL — sign flips | FAIL, sign flips |
| W2 | USDINR 5-day | +3.75 | +3.73 | +0.75 | FAIL — TRAIN only | FAIL, TRAIN-only |
| **W3** | **S&P 500 5-day** | **−3.86** | **−2.91** | **−4.46** | **PASS** | PASS, −3.50/−2.72/−4.39 |

**A bug in our own replication, caught by a guard.** The first run returned NaN
for FULL and TEST. Cause: in polars, NaN is a *value*, not a null, so
`drop_nulls` leaves the forward-window NaNs at the end of the target series, and
one NaN makes the whole regression return NaN. This is the same library trap the
source programme records in its own failure log. An assertion now rejects a
non-finite target that survives `drop_nulls`, rather than letting it propagate.

## V5 · A6 · composition block — 2026-08-24

Stage `incremental_value` re-run (111.4 s, PASS). Confirmed exactly: COMP-only
Q5−Q1 **+74.8 bp** at non-overlap **t = +2.86**; ΔIC **+0.0012**, paired daily
**t = +0.67** against bars of 0.005 and 2. Stage verdict: NOT ESTABLISHED.

## V6 · A4 · hazard model — **materially different on the rebuilt object**

Stages `phase2_hazard` (61.4 s) and `phase2_decision` (96.5 s) re-run.

| | project documents | fresh run |
|---|---:|---:|
| k=1 AUC | 0.797 | **0.647** |
| KM baseline AUC | 0.569 | 0.569 |
| TRAIN / TEST AUC | 0.786 / 0.809 | **0.637 / 0.657** |
| paired t vs KM | +41.6 | **+14.42** |
| 16D TEST gain | +23 bp [+7, +40] | **−18 bp [−52, +19]** |
| 16D paired vs age rule | −15 bp, t = −2.96 | **−31 bp, t = −1.85** |

The 0.797 figure comes from a run dated **2026-07-13** — before the audit
rebuilt the state object on the corrected concentration feature. **Phase II was
never re-executed afterwards**, and the pre-audit numbers propagated into the
results summary, which is otherwise post-audit.

Both verdicts are unchanged — 16C still passes, 16D still fails — but the
magnitudes are much smaller, and the decision-layer result is now *worse* for
the model: its test-era anticipation gain is negative rather than positive. The
memorisation check survives: out-of-sample AUC still exceeds in-sample.

## V7 · A7 · mechanism ablation — 2026-08-24

Significance half re-run (`model_descriptives`): episode clustering **2.58×
TRAIN / 2.64× TEST**, both p = 0.005, against the thesis's pre-audit 2.48 /
2.38. Slightly stronger on the rebuilt object.

Value half read from `c6_results.parquet` in the engine tree.

**A sign-convention error caught before it entered the exhibit.** The C6 module
computes `diff = loss(b) − loss(a)`, so a *positive* statistic means the first
system — the conditioned one — is better. A first reading had it inverted. On
the corrected reading:

- pre-registered **primary** stratum (archetype identity uncertain, where the
  mechanism should bind hardest): t = **−0.82**, p = 0.41 — indistinguishable;
- **full panel**: t = **+2.90**, p = 0.0037 — statistically favours conditioning,
  by an effect of **5 × 10⁻⁵** CRPS against a score whose level is about 0.55.

That is the sharper statement, and it is the detectability mechanism exactly:
significant and vacant.

---

## Reproducibility

Every application now declares the commands that regenerate its artifacts.
`convgap reproduce` executes them — nine distinct recipes across both source
trees, about 22 minutes — and `convgap replicate` regenerates the two screens
this repository re-derives. After both, no figure in the exhibit is inherited.

**Status: 8 of 8 applications `verified` / `primary`. Every declared value
confirmed by extraction from an artifact produced by a run executed on
2026-08-24.**

---

## V8 · The data section — 2026-08-24

The protocol was extended to the descriptive numerals. Results are traced by
the application registry; the data section makes claims too, and in this
programme several of them were stale — quoted from logs written before the
audit rebuilt the object they describe.

`src/convgap/replication/descriptives.py` computes every figure the data
section reports, in one pass over the source artifacts, into
`outputs/replication/descriptives.csv`. `convgap facts` verifies the paper's
assertions against it: **20/20 traced.**

**Corrections to the draft.**

| Claim as drafted | Verified | |
|---|---|---|
| "approximately 24 million trades" | **25,155,785** | understated |
| "approximately 946 instruments" | **1,028** | the 946 is from a pre-audit provenance log |
| "939 issuers" | — | **untraceable to any artifact; the claim is withdrawn** |
| 3.0% of trades in null-direction months | 3.004% | confirmed |
| TRAIN 618 / 508,563; TEST 851 / 294,243 | confirmed exactly | |

**A new figure worth reporting.** Only **441 instruments appear in both eras** —
71% of the training universe, 52% of the test universe. That is a far sharper
statement of "not a repeat sample" than the era counts alone, and it was not
previously computed.

## V8a · The identifier audit, re-derived

The re-minting claim was the last untraced assertion in the paper. It is the
constraint that bounds every entity statistic in the programme, so it was
computed rather than cited.

The test is the one the source programme specifies: distinct months per
identifier, run on a control population whose real-world persistence is known.

| population | distinct identifiers | median months | mean months |
|---|---:|---:|---:|
| masked participant | 363,663 | 1.0 | 1.21 |
| masked sub-account | 394,398 | 1.0 | 1.21 |
| **broker (control)** | **22,262** | **1.0** | **1.40** |

Over a 171-month span, a real population of a few hundred brokers presents as
22,262 identifiers with a median lifetime of one month. The control cannot have
that turnover, so the identifiers are re-minted; the masked populations behave
identically and must be treated the same way.

The result confirms the source programme's claim by its own method, and is
now computed from the raw records rather than inherited.

---

## State

```
convgap verify     8/8 applications  traced + primary
convgap facts     20/20 data-section claims traced
convgap reproduce  9 recipes across both trees, ~22 min
convgap replicate  3 quantities derived here: daily screen, weekly screen,
                   descriptives
```

No numeral in the abstract, introduction or data section is inherited. Every
one descends from a computation executed on 2026-08-24 and is bound to a
digest recorded in `outputs/evidence.lock.json`.

---

## V9 · The manuscript — 2026-08-24

The protocol was extended a third time, to the prose. `convgap manuscript`
extracts every numeral from the manuscript and reports any that is neither
matched to verified evidence nor listed as exempt with a stated reason. It is a
report rather than a gate: a paper legitimately contains numerals that are not
evidence. What it establishes is that none is *silently* unaccounted for.

**What the scan found, and what was done about it.**

| Finding | Resolution |
|---|---|
| Sample dimensions quoted in prose (972 instruments, 2,081 dates) with no declared source | Removed from the prose. The traced instrument-day count carries the claim |
| Run lengths behind the permutation ratios (3.61 / 1.40 / 4.05 / 1.53 d) reported but not declared | Declared as evidence on A7 with extractors |
| p-values (0.41, 0.0037) reported in tables but not declared | Added to the corresponding evidence items |
| Paired t = −1.85 reported in a table but not declared | Declared as evidence on A4 |
| "5,945,022 rows" in the price panel | **Wrong. The artifact holds 5,945,010.** Corrected in the paper and in the fact |
| "within 2,000 rows", "71% of the training universe" | Approximations with no artifact. Reworded to state the relation without the invented precision |
| Superseded pre-audit figures (0.797, +41.6, +23 bp) | Exempt, with the reason that §4.3.2 quotes them deliberately in order to withdraw them |
| Publication years, pre-registered bars, cited literature figures, date components | Exempt, each with a stated reason |

The scanner also required narrowing: "S&P 500" and "SHA-256" contain digits
that are not quantities, and section cross-references are not measurements.
Those are handled by explicit exclusion rather than by loosening the match.

**One error found in the paper's own arithmetic.** The price panel row count was
overstated by twelve rows — a transcription slip in a figure nobody would have
checked, in a sentence nobody would have queried. It is corrected. That it was
found at all is the argument for the check.

---

## Final state

```
convgap verify      8/8  applications  traced + primary
convgap facts      32/32 data-section claims traced
convgap manuscript  every numeral verified or explicitly exempt
convgap reproduce   9 recipes across both trees, ~22 min
convgap replicate   3 quantities derived here
```

117 tests. Lint and strict type checking clean. Draft complete through the
conclusion.
