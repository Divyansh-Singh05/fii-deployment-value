# What the engine produced — an interpretation

Generated 2026-08-23 from a clean run of this standalone tree
(`./run_all.sh`, then two `s12_risk_engine` queries). Every number below is
from `outputs/` in **this** directory, not copied from the research project.
The chain reproduces the parent project's figures exactly, which is the first
result worth stating: the port changed no arithmetic.

```
s06_daily_filter        PASS    18s      s10_validation          PASS    64s
s07_archetype_probs     PASS     3s      s11_stock_risk_profiles PASS    57s
s08_outcome_densities   PASS    45s      s13_lookahead_audit     PASS    91s
s09_predictive_mixture  PASS   145s
```

---

## 1 · The headline: it beats the Gaussian where it matters, and admits where it doesn't

| Engine | h | Breach 5% | Breach 1% | ES coverage | ES/VaR | PIT χ² | Mean CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Flowsense** | 1 | 0.0512 | **0.0104** | **1.023** | 1.382 | **0.6** | 0.55652 |
| EWMA-Normal | 1 | 0.0438 | 0.0162 | 1.237 | 1.146 | 312.7 | 0.55964 |
| **Flowsense** | 5 | 0.0518 | **0.0091** | **1.015** | 1.384 | **1.3** | 0.58737 |
| EWMA-Normal | 5 | 0.0510 | 0.0198 | 1.237 | 1.146 | 87.4 | 0.58878 |
| Flowsense | 20 | 0.1153 | *not served* | — | — | 155.1 | 0.64106 |
| EWMA-Normal | 20 | 0.0504 | 0.0215 | 1.340 | 1.146 | 66.4 | **0.60489** |

*Regenerated after the L3/L4 fixes — see [LIMITATIONS.md](LIMITATIONS.md).
h=5 improved on every calibration measure once its tail floor was made to
count independent rather than overlapping observations; h=20 now has no
servable 1% forecast at all.*

*Targets: 0.0500 / 0.0100 breach, 1.000 ES coverage.*

**Read it this way.** At h=1 and h=5 Flowsense hits the 1% level almost exactly
(1.04%, 1.06%) while the Gaussian breaches 1.62% and 1.98% — **62% to 98% too
often** at the level that actually matters for capital. Expected shortfall
coverage tells the same story: 1.02 against 1.24. The Gaussian is not merely
imprecise in the tail, it is *systematically* wrong there.

Note what the Gaussian gets *right*: at the 5% level it is fine, even
conservative at h=1 (4.37% against a 5% target). This is not a paradox.
Standardised equity returns are fat-tailed **and** peaked, so a fat-tailed law
places its 5% quantile *nearer* zero than a Gaussian does. The Gaussian only
becomes dangerous once you go far enough out — which is exactly where a risk
engine is asked to work.

**The margin that isn't close is distributional shape.** PIT χ² of 0.6 against
314.4 at h=1. Scored against a date-block bootstrap null rather than a
textbook χ²(9) — which is invalid under panel dependence and wrong by ~36× here
— Flowsense's raw statistic is 38.0 against a 95th-percentile critical value of
**618.0**: not rejected. The Gaussian's is 19,579. The engine is not just
better calibrated in the tail; it has approximately the *right shape*.

Formally, on daily-aggregated CRPS with Newey-West errors, Flowsense beats the
Gaussian at h=1 with **t = 9.81** against a Bonferroni α of 0.00556.

---

## 2 · The honest failure: h=20

Flowsense breaches **11.53%** at the 5% level at h=20 and serves **no 1%
forecast at all** — under a tail floor that counts independent observations,
no h=20 cell has the support to carry a 1% tail. The Gaussian's CRPS is also
lower (0.60489 vs 0.64106). The engine loses at 20 days, twice over.

The cause is structural, not fixable by tuning: consecutive 20-day windows
overlap by 19 days, so the independent information is roughly a twentieth of
what the row count implies. `s12_risk_engine` therefore **retires h=20 from
the public interface** — `--horizon` accepts 1 and 5, and 20 is reachable only
with `allow_internal=True` in code. That the engine refuses to serve its own
worst horizon is a design decision, not an oversight.

---

## 3 · Tails are real, and the Gaussian cannot represent them

| threshold | observed | Gaussian expects | ratio |
|---|---:|---:|---:|
| z < −4 | 1,523 | 18.32 | 83× |
| z < −5 | 691 | 0.17 | 4,167× |
| z < −6 | 355 | 0.00 | 622,019× |

These are *volatility-standardised* returns — already divided by an EWMA σ.
The fat tail is what survives after the obvious volatility scaling is removed.

Flowsense's ES/VaR ratio runs **1.38–1.43** against the Gaussian's structural
**1.1457**, and it *moves*: 1.4282 at h=20 versus 1.3832 at h=1. Tails get
fatter with horizon; they do not Gaussianise through aggregation.

---

## 4 · Calibration is conditional — the most important caveat

The pooled 5% breach rate is 1.02× nominal, which looks like a clean pass.
It hides a gradient that should not exist:

| stratum | n | breach rate | 95% CI | × nominal |
|---|---:|---:|---|---:|
| ALL (pooled) | 558,709 | 0.0514 | [0.0446, 0.0592] | 1.03 |
| EWMA vol Q1 (calmest) | 111,820 | 0.0540 | [0.0484, 0.0606] | 1.08 |
| EWMA vol Q2 | 111,217 | 0.0553 | [0.0492, 0.0625] | 1.11 |
| EWMA vol Q3 | 111,342 | 0.0516 | [0.0450, 0.0588] | 1.03 |
| EWMA vol Q4 | 111,649 | 0.0508 | [0.0432, 0.0603] | 1.02 |
| EWMA vol Q5 (wildest) | 112,681 | 0.0452 | [0.0344, 0.0584] | 0.90 |

**No stratum is flagged miscalibrated.** Before the L6 fix this ran
1.23 / 1.15 / 1.02 / 0.96 / **0.76** with three strata flagged.

The residual spread above is **not** a model defect. `EWMA vol Q1` is cut on
the whole sample, so it means *in the calmest fifth of this stock's history
including its future* — information the engine did not have. `s11` now also
reports a causal cut using only each stock's own past, and against that the
engine is flat:

| causal own-past vol quintile, h=1 @5% | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| × nominal | 1.02 | 1.07 | 1.02 | 1.00 | 1.03 |

Same engine, same rows. See [LIMITATIONS.md](LIMITATIONS.md) §L6.

The COVID window breaches at 3.19× nominal — but its CI is [0.0479, 0.2619],
which *covers* nominal. With 11,456 highly dependent rows the honest interval
is enormous, and the crash cannot be called a calibration failure on this
evidence. Contrast that with the volatility gradient, whose intervals are tight
and clearly exclude nominal. **The gradient is the real finding; COVID is not.**

---

## 5 · Two worked queries

### RELIANCE, h=5, COVID crash (Feb–May 2020) — the engine under stress

```
window 2020-02-03 to 2020-05-29  |  77 FII-active scored days
vintage 51 (asof 2020-04-30), fitted on 152,127 stock-days / 502 stocks

engine       breach5%  breach1%  ES1% pred  ES1% real  ES cover     CRPS
flowsense      0.1429    0.0779     -3.861     -4.680     1.212  0.87400
normal         0.1429    0.1169     -2.665     -3.976     1.492  0.87074
```

Both engines break: 14.29% breaches against a 5% target. Flowsense is *less*
broken at the 1% level (7.79% vs 11.69%) and its ES coverage is closer
(1.21 vs 1.49), but neither is calibrated through a systemic shock. Watch the
VaR itself move — from −9.2% in early February to **−26.7%** by late May, as
the past-only volatility estimate absorbs the crash. The engine adapts, but it
adapts *behind* the event, which is the fundamental limit of a backward-looking
volatility model.

Note the archetype column reads `ROBOT` with `pa = 1.00` throughout. RELIANCE
in this window is not a flow-concentration story at all; the risk numbers are
being driven by the volatility model and the fitted tail shape, not by the
flow regime. That is worth knowing before attributing crash performance to the
flow conditioning.

### RELIANCE, h=1, calm year (2023) — the engine in its comfort zone

```
174 scored days
engine       breach5%  breach1%  Kupiec p(1%)  ES1% pred  ES1% real  ES cover     CRPS
flowsense      0.0402    0.0057        0.5401     -3.939     -6.667     1.693  0.54872
normal         0.0287    0.0057        0.5401     -2.665     -6.667     2.502  0.55301
```

Both close to nominal; Flowsense wins CRPS by +0.775%. Note ES coverage 1.69
and 2.50 — with one breach in 174 days, "realised ES" is a single observation.
The engine prints the warning itself (*"only 174 observations carry both an
outcome and a 1% forecast (1.7 breaches expected) — the Kupiec p-values are
descriptive only"*). **Single-stock, single-window validation is descriptive.
The panel numbers in §1 are the evidence.**

---

## 6 · No look-ahead — verified, not asserted

`s13_lookahead_audit` truncates every input at 2021-06-30, re-runs s06→s09 in
a world that ends there, and compares:

```
rows on or before 2021-06-30:  full-sample 291,003 | truncated 291,003
[L1] filtered posteriors        max|diff| 0.000e+00   rows differing 0   PASS
[L2] archetype probabilities    max|diff| 0.000e+00   rows differing 0   PASS
[L3] VaR and ES (12 cols)       max|diff| 0.000e+00   rows differing 0   PASS
[L4] CRPS and PIT (24 cols)     max|diff| 0.000e+00   rows differing 0   PASS
[L5] coverage — rows present in one run only: full-only 0, truncated-only 0  PASS
```

**Exact zeros, not tolerances.** The gate also requires equal *availability*:
a value that flips finite→NaN between the two runs fails rather than being
quietly skipped, which is the loophole most look-ahead audits leave open.

`s12_risk_engine` additionally re-derives its answers from the raw densities
on startup and checks them against the stored pipeline output — max VaR error
1.14e-07. The engine cannot silently drift from the artifact it was validated
on.

---

## 7 · What I would not claim from this output

- **It forecasts distributions, not direction.** Nothing here is a trading
  signal, and no return-predictability claim is supported by these numbers.
- **The flow conditioning's contribution is not isolated here.** Flowsense
  beats the Gaussian, but `soft_roll` vs `hard_roll` — soft archetype
  weighting against a forced hard call — differs by only 1.5e-5 CRPS on the
  full panel. Statistically significant, economically negligible except in the
  5.5% of rows where archetype identity is genuinely uncertain.
- **The economic meaning of the archetypes is a separate and currently
  unsettled question**, under revision in the parent research project. The
  risk engine's calibration does not depend on it, and nothing above should be
  read as evidence for any interpretation of *why* concentration matters.
- **The volatility-quintile gradient (§4) is unexplained.** Until it is, treat
  per-stock risk numbers in very calm or very volatile names as biased in the
  direction that table shows.
