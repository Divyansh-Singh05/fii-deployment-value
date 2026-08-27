# 00 · Framing

## Scope

**Object of study:** a model that issues a forward-looking distributional or
risk statement — VaR, expected shortfall, a predictive density — for an asset
over a horizon, estimated walk-forward and scored out of sample against a
benchmark.

**In scope:** the design of the backtest and the validation battery around such
a model. What has to be true before its scorecard is evidence.

**Out of scope, deliberately:**
- Data preparation, cleaning, identifier resolution, corporate-action handling.
  These are a separate literature and a separate paper.
- Implementation defects. The paper is not a catalogue of mistakes and does not
  discuss how errors escape human attention. Every failure mode below is a
  **property of the evaluation design**, reproducible by a competent
  practitioner following standard practice, on correct code and clean data.

The scoping assumption is stated once and honestly: we assume the data is
right. That is a strong assumption. The paper's claim is that **even when it
holds, the standard risk-model backtest can return PASS on a model that has not
been validated** — and that is what makes the assumption safe to make.

## Thesis

> A risk model that passes the standard backtest battery has not been
> validated. Kupiec, Christoffersen, PIT, CRPS and Diebold–Mariano are
> necessary and radically insufficient. Each can be computed correctly and
> still return PASS on a model whose forecast is misaligned with the outcome it
> is scored against, whose tail quantiles rest on a fraction of the evidence its
> support floors claim, whose evaluation strata condition on the future, and
> whose distinguishing mechanism contributes nothing measurable.

The contribution is a **sequence** — nine questions in a fixed order, each with
a stated *void scope* (what a failure at that step invalidates downstream) —
and, for each, a measured quantity showing what the question is worth.

## The demonstration

A flow-conditioned non-parametric risk engine, scored walk-forward on **578,481
stock-days** across 562 names, 2016–2025, against a closed-form EWMA-Normal
benchmark under an identical volatility model and identical evaluation dates.

**What the standard scorecard says.**

| Metric, h = 1 | Engine | EWMA-Normal | Target |
|---|---:|---:|---:|
| Breach rate at 1% | **1.04%** | 1.62% | 1.00% |
| Kupiec p at 1% | **0.673** | 0.000 | > 0.05 |
| ES(1%) coverage | **1.027** | 1.246 | 1.000 |
| Per-name Kupiec pass rate | **90.4%** | 60.4% | — |
| PIT χ² | **0.7** | 303.3 | small |
| Mean CRPS | **0.5579** | 0.5610 | lower |

Read as published, this is a decisive win at every level, on a sample large
enough that nobody asks about power.

**What the protocol returns.**

| Question | What it found | Magnitude |
|---|---|---|
| Q5 · can the test reject? | The PIT verdict came from a bootstrap whose null was constructed from the observed statistic. It cannot reject a PIT with **all mass in one bin** | Zero power. Every "not rejected" verdict void, on both engines |
| Q5 · is the reference valid? | The textbook χ²(9) critical value is 16.9; under panel dependence the correct value is **584** | Wrong by ~36× |
| Q4 · how much evidence? | Support floors counted overlapping h-day windows as independent observations | h = 5 tails rested on ≈ 1/5 the claimed evidence; correcting it removes the 1% forecast from **33%** of 5-day rows and makes h = 20 unservable from first principles |
| Q3 · is the *evaluation* causal? | A calibration gradient of 0.30 across volatility strata, reported as a model defect | Against a causal stratum, the same forecasts run flat at **−0.04**. The gradient was in the diagnostic |
| Q1 · are paired quantities aligned? | Tail-fatness computed as mean predicted ES over mean realised VaR — different row populations | Sign inverted: reported 1.131 (thinner than Gaussian), actual **1.428** (fatter) |
| Q6 · does the thesis contribute? | The conditioning mechanism the model exists for, ablated | **Statistically indistinguishable from zero at every horizon and in every stratum**, powered to 0.03% of CRPS |

**What survives.** The engine really does beat the Gaussian in the deep tail,
and by more than first reported — once the benchmark is held to the same
corrected null, it rejects at every horizon where the broken test had it
passing. But the win is attributable to **distributional shape**, not to the
conditioning variable. The correct statement of the result is *"the Gaussian is
wrong for this market"*, not *"our variable is right"* — and the difference
between those two sentences is the entire output of Q6.

That is the paper's arc: **the same evidence, correctly evaluated, supports a
smaller and different claim than the one the scorecard appeared to support.**

## Contribution claims

| # | Claim | Defensibility |
|---|---|---|
| C1 | A nine-question sequence with explicit void scopes for validating walk-forward risk models | Ordering and void-scoping are the novel part; the individual statistics are standard and are cited as such |
| C2 | **Backtest power is not free.** A size-and-power audit of one's own test statistics, on synthetic samples with known ground truth, must precede any PASS verdict | Demonstrable, not assertible: a null-centred construction rejects all four labelled cases at 5.0% size where the resampled-observed construction rejects none |
| C3 | **Evaluation-set look-ahead.** A causally clean model can be made to display a spurious conditional-calibration defect purely through the choice of stratification variable | Measured on identical forecasts: 0.30 vs −0.04 |
| C4 | **Effective evidence, not row count**, must gate every support floor, abstention rule and tail quantile in an overlapping-window backtest | Measured: overlap costs efficiency, not quantile accuracy — and the distinction determines whether the fix is to widen tails or to withhold them |
| C5 | **Mandatory pre-registered mechanism ablation**: a risk model must be scored against itself with its distinguishing mechanism removed, or its win cannot be attributed | The strongest single practice in the paper, and the one the case study fails |
| C6 | **Proof of absence of look-ahead** by whole-pipeline truncation with availability equality, not by assertion of causal gates | Reproducible; exact zeros across 12 risk and 24 scoring columns at three truncation dates |

**Honest calibration, stated in the paper.** No statistic here is new. The
contribution is sequencing, void-scoping, and quantifying what each question is
worth on a large real engine. Pitched as new statistics this work would be
rightly rejected.
