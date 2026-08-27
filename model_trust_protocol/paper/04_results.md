# 4 · Results

<!-- provenance -->
*Every numeral in this section is traced. `convgap verify` reports 8/8
applications publishable — each bound to a primary artifact, its digest
recorded, and its value read back out of that artifact by an extractor.*
<!-- /provenance -->

---

## 4.1 The ladder

Seven applications of the record, plus one external control, placed at the
lowest level of friction each fails to survive.

| level | question | applications lost | still standing |
|---|---|---:|---:|
| L0 existence | is the effect present under conventional inference? | **0** | 7 |
| L1 replication | does it hold on an era the design never touched? | **0** | 7 |
| L2 availability | is it knowable when the decision must be made? | 1 | 6 |
| L3 competition | does it beat the baseline it would be deployed against? | 2 | 4 |
| L4 detectability | is it large enough for the metric the decision consumes? | 2 | 2 |
| L5 execution | does the edge exceed the cost of capturing it? | 1 | **1** |

**Nothing is lost at the first two levels.** Every application of the record
produced an effect that was statistically present and replicated on a frozen
out-of-sample era covering a substantially different cross-section: only 441
instruments are common to the two eras, a minority of each. On the standard by which the replication literature judges
published work — clearing a conventional significance hurdle and holding up on
re-testing — the entire programme passes.

The attrition begins where institutional constraint does, and it is distributed
across four different levels. No single friction explains the pattern.

## 4.2 Availability — the effect is not knowable when it is needed

**A1 · Institutional share of turnover → next-day volatility.** 577,245
instrument-days. The dependent variable is the
squared standardised return: volatility the model's own EWMA has already failed
to explain.

| specification | FULL | TRAIN | TEST |
|---|---:|---:|---:|
| instrument + date FE, log-turnover controlled | **−5.53** | **−4.42** | **−3.25** |
| identical, signal known only at t−2 | **−0.32** | **−0.30** | **−0.13** |

The upper row is a strong, era-stable result. The lower row is the same
regression on the same panel with one change: the predictor is taken at the lag
a decision could actually use. Nothing else differs — same fixed effects, same
controls, same clustering, and a sample differing only by the rows the
additional lag removes.

The effect does not weaken; it disappears. That locates the finding
unambiguously as a statement about contemporaneous market structure rather than
a forecast, and the source module says so itself: *"the t−2 row is the honest
limit."*

## 4.3 Competition — the baseline already has it

Three applications fail here, and their three baselines are of different kinds:
a better-specified model of the same quantity, a trivial rule, and — for the
control — a model that absorbs the predictor through another channel.

### 4.3.1 A controlled test: the same predictor against two baselines

**A8 and A2** share a significance half. The C4 screen — net inflow at t−2 on
next-day index absolute return, beyond VIX, Newey–West 10 lags — gives

| | FULL | TRAIN | TEST |
|---|---:|---:|---:|
| C4 flow coefficient, t | **−4.83** | **−2.53** | **−2.51** |

clearing both legs of its pre-registered bar (|t| ≥ 2.50 full; same sign with
|t| ≥ 1.5 in each era). One of four pre-registered cells passes, as designed.

The two applications then differ in exactly one respect: the baseline density
the flow tilt is asked to improve.

| | A8 · EWMA base | A2 · GJR-GARCH base |
|---|---:|---:|
| CRPS Diebold–Mariano, full (bar ≤ −2.0) | **−2.35** PASS | **−0.95** FAIL |
| mean CRPS advantage ×100, TRAIN | −0.0019 | −0.0008 |
| mean CRPS advantage ×100, TEST | −0.0004 | **+0.0004** |
| Kupiec p, 5% / 1% VaR | 0.225 / 0.161 PASS | — |
| scored days | 2,739 | 2,235 |

Against the simpler base the tilt clears every leg. Against the
better-specified base it fails two of three, and the test-era sign flips — a
failure under the pre-registration regardless of magnitude.

**Incremental value is a property of the pair, not of the predictor.** The
information is real in both cases; whether it is *incremental* depends entirely
on what the baseline already contains. The difference in scored days is part of
the same mechanism rather than an inconvenience: the richer base needs a longer
burn-in, so it both explains more and is asked to explain fewer days.

### 4.3.2 The trivial rule

**A4 · Episode-end hazard model.** A walk-forward discrete-time hazard with
yearly refits, right-censoring handled, forecasting whether a flow episode ends
today.

| | model | age-only Kaplan–Meier |
|---|---:|---:|
| AUC, k=1 pooled | **0.647** | 0.569 |
| AUC, TRAIN | 0.637 | — |
| AUC, TEST | **0.657** | — |
| paired t on out-of-window log loss | **+14.42** | — |

The skill is real: it beats the duration-only baseline decisively, and it
discriminates *better out of sample than in*, which rules out memorisation.
The pre-registered gate passes.

The decision layer does not.

| test era, bp per episode | |
|---|---:|
| model anticipation gain | **−18** (95% CI [−52, +19]) |
| age-only rule gain | **+16** |
| paired difference on 402 common episodes | **−31** (t = −1.85) |

A three-line rule — act once an episode is a few days old — outperforms the
fitted model on the episodes both act upon. Note what a weaker design would
have reported: each rule compared against zero separately, with the model
declared the better of two positives. Only the paired comparison on shared
events reveals the ordering.

**A finding about the programme, not only about the data.** Every figure above
comes from a re-execution on the current state object. Project documents report
AUC 0.797 with paired t = +41.6 and a test-era anticipation gain of +23 bp.
Those come from a run dated 2026-07-13, before an audit rebuilt the states on a
corrected feature; the Phase II stages were never re-executed afterwards, and
the pre-audit figures propagated into a results summary that is otherwise
post-audit. Both verdicts survive — the hazard still passes, the decision layer
still fails — but the skill is a third smaller than reported and the decision
result is *worse*: the model's test-era gain is negative where it was recorded
as positive. We report this because a validation protocol that cannot catch a
stale stage in its own source material is not a protocol.

### 4.3.3 The control

**A3 · S&P 500 → Nifty weekly volatility.** Not an application of the record.
A predictor from outside the dataset entirely, run through the same screen, the
same bar and the same engine construction.

| W3 screen, next-week realised volatility | FULL | TRAIN | TEST |
|---|---:|---:|---:|
| 5-day S&P 500 return, t | **−3.86** | **−2.91** | **−4.46** |

Among the most era-stable coefficients in the programme: same sign, comparable
magnitude, and stronger out of sample than in. The other two cells fail for
reasons the re-derivation reproduces exactly — flow persistence flips sign
between eras, USDINR is significant in training only.

The engine built on it fails.

| module 24 weekly gate | |
|---|---:|
| CRPS Diebold–Mariano, full (bar ≤ −2.0) | **+1.58** FAIL |
| CRPS advantage ×100, TRAIN / TEST | +0.0020 / +0.0000 FAIL |

Positive: the tilted engine scores *worse* than its base. A GJR-GARCH base
absorbs the spillover through the index's own lagged returns.

**This is the paper's control result.** A predictor drawn from outside the flow
record, more era-stable than anything the record produced, dies at the same
level and by the same mechanism. The friction is a property of the transition
from statistical evidence to institutional use — not a peculiarity of this
dataset.

## 4.4 Detectability — the effect is below the resolution of the metric

Two applications survive competition and fail because the quantity a user would
consult cannot see them.

**A6 · Participant-composition feature block.**

| | value | bar |
|---|---:|---:|
| composition-only Q5−Q1 forward-return spread | **+74.8 bp** (non-overlap t = **+2.86**) | — |
| incremental IC over conventional flow features | **+0.0012** | 0.005 |
| paired daily t on that increment | **+0.67** | 2 |

Both are computed on the same block, the same panel, the same run. The
extremes carry a spread that is significant and stable on non-overlapping
episodes; the average carries nothing. A single summary statistic reports
whichever half it weights, and the incremental coefficient — the quantity a
user of the data would consult before adding the block to an existing model —
weights the half that is empty. The stage's own verdict: **NOT ESTABLISHED**.

**A7 · Flow-regime conditioning in the stock-day density.** The significance
half comes from a test that never sees a price. Episode labels are compared
against a within-instrument shuffled null that preserves each instrument's
label count and destroys only temporal adjacency:

| | observed mean run | shuffled null | ratio | p |
|---|---:|---:|---:|---:|
| TRAIN | 3.61 d | 1.40 d | **2.58×** | 0.005 |
| TEST | 4.05 d | 1.53 d | **2.64×** | 0.005 |

The labels are real, they cluster into episodes, and this replicates out of
sample. The value half is the ablation: the identical density machinery re-run
with the conditioning removed and everything else fixed.

| stratum, h=1 | DM statistic | p | effect (CRPS) |
|---|---:|---:|---:|
| pre-registered **primary** (archetype identity uncertain) | **−0.82** | 0.41 | −4×10⁻⁵ |
| full panel | **+2.90** | 0.0037 | **+5.0×10⁻⁵** |

In the stratum the pre-registration nominated — the days on which archetype
identity is genuinely uncertain, where the mechanism should bind hardest — the
conditioning is indistinguishable from its own removal. On the full panel it is
statistically favoured, by five parts in a hundred thousand of a score whose
level is around 0.55: roughly one part in eleven thousand.

**Significant and vacant.** Comparison against an external benchmark
establishes that a system is better. Only ablation establishes *why*, and the
answer here is the empirical shape of the standardised distribution rather than
the flow record that the system is named for.

## 4.5 Execution — the edge and its cost are the same order

**A5 · Cross-sectional concentration book.** A mechanical long/short proxy for
the concentration measure, carrying no fitted model at all, so what is priced
is the data rather than an architecture.

| | TRAIN | TEST |
|---|---:|---:|
| gross Sharpe | **1.36** | **1.51** |
| breakeven one-way cost | **7.33 bp** | **7.41 bp** |
| net Sharpe at 15 bp one-way | −1.42 | **−1.53** |

The signal is real, replicates across the split, and is the highest gross
Sharpe of the eight books in the test era. Its breakeven — the one-way cost at
which net profit is zero — is roughly half a realistic institutional cost.

We report the breakeven rather than the net figure because the breakeven is a
property of the strategy while a net Sharpe is a property of the cost
assumption. A reader whose execution costs differ from ours can use the first
and cannot use the second.

## 4.6 What survives

**A8 · Aggregate flow → market-level density.** One application clears every
level: the record aggregated to market level, entering a Student-t density at a
two-day reporting lag, against an EWMA baseline that does not already contain
it. All three legs of its pre-registered gate pass (§4.3.1).

Two things make the pass credible rather than fortunate.

First, it is the *only* engine in its family unchanged under two later
estimation fixes — an out-of-sample maturity floor and an optimiser
path-independence correction — that widened its three siblings from narrow
misses into clear failures. A result that does not move when its neighbours
move under a correction is a result that was not resting on the defect.

Second, its conditions are legible and restrictive, and they are precisely the
conditions the six failures violate: aggregation to a level where the
measurement is stable, availability at a deployable lag, a magnitude the
consuming metric can resolve, and a baseline that has not already absorbed it.

The ladder is therefore a boundary rather than a verdict. The record is worth
something, in one identifiable use, and the map of where that stops is the
result.
