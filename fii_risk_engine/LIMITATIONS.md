# Structural and statistical limitations

The mechanical look-ahead position is strong: forward-only filtering, vintages
with strictly prior `asof`, an `s + h <= asof` embargo, and a truncation audit
that passes with exact zeros. None of that is in question below.

What follows are the limitations that survive a clean causal audit — the ones
that live in the mathematics rather than in the data flow. Each was measured on
the 2026-08-23 run rather than reasoned about.

---

## L1 · Strictly end-of-day. Not just causally, but structurally.

`s01_feature_store.py:280-281` standardises every feature by cross-sectional
rank within the trading day:

```python
n_valid = pl.col("_m").is_not_null().sum().over("TR_DATE")
rnk     = pl.col("_m").rank(method="average").over("TR_DATE")
```

A stock's score depends on **every other eligible stock's flow that same day**.
This is causally safe at EOD and structurally invalid before it: the true
cross-section does not exist mid-session, so any intraday query is answering a
question the feature cannot represent.

There is a second consequence that follows from the same line and is easier to
miss: the rank is taken over the **eligible universe**, so changing universe
membership changes every surviving stock's score. Feature values are not
properties of a stock alone. Two runs with different liquidity floors are not
comparable, and a backfilled universe silently reprices history.

**Status: accepted by design, not mitigated.** The engine must be queried at or
after the close.

---

## L2 · Abstention is a coverage limit, not a survivorship bias

`s09` withholds a VaR when the mixture tail cannot support the level
(`MIN_TAIL_EFF=50`, `MIN_TAIL_RAW=200`). The concern is that Kupiec and ES
coverage then score only the rows the engine chose to answer, flattering it.

Measured on realised `z_h1` (h=1, 578,481 rows with an outcome):

| | n | sd | p1 | p0.1 | min | P(z<−4) | P(z<−6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1% level** — forecast | 541,916 | 1.063 | −2.71 | −5.29 | −32.6 | 0.266% | 0.062% |
| **1% level** — abstained | 36,565 (6.32%) | 1.026 | −2.49 | −4.95 | −11.3 | 0.227% | 0.055% |
| **5% level** — forecast | 558,709 | 1.064 | −2.72 | −5.30 | −32.6 | 0.268% | — |
| **5% level** — abstained | 19,772 (3.42%) | 0.983 | −2.22 | −4.44 | −11.3 | 0.137% | — |

The abstained rows are **calmer, not riskier**, on every measure. The floor
binds where the archetype mixture is thin — early vintages, rare archetype
states — not where the stock is dangerous; volatile names are typically liquid
and well populated. If anything the reported metrics are mildly conservative.

**Status: the selection is real and non-random, but its sign is the opposite of
the concern.** What remains is a genuine *coverage* limit: 6.3% of rows get no
1% number at all, and that is worst exactly where history is shortest.

---

## L3 · Overlapping windows cost effective sample size, not tail width

`s08` builds h-day forward returns by sliding a one-day window, so one crash
enters the h=5 density in five overlapping windows. The violation is real. The
predicted consequence — artificially narrow tails, overconfidence — is not.

Overlapping windows are still valid draws from the *marginal* h-day
distribution; the empirical CDF stays consistent, it just loses efficiency.
Testing it directly, overlapping against five disjoint non-overlapping series
(every 5th day per stock):

| quantile | overlapping | decimated (mean of 5) | spread across the 5 disjoint offsets |
|---|---:|---:|---|
| p5 | −1.658 | −1.659 | [−1.666, −1.649] |
| p1 | −2.883 | −2.880 | [−2.909, −2.859] |
| p0.5 | −3.538 | −3.537 | [−3.563, −3.488] |
| p0.1 | −5.497 | −5.485 | [−5.596, −5.376] |

The overlap-vs-decimated gap at p0.1 is **0.012**; the spread across the
disjoint offsets is **0.22** — roughly twenty times larger. No detectable
narrowing.

**Where the point does bite, hard:** the ESS floor of 1,000 in `s08` counts
overlapping observations. At h=5 that is ~200 genuinely independent windows;
at h=20, ~50. The floor overstates the information behind every density by
about a factor of h, and any test assuming independence is wrong by the same
order. That is why `s10` scores Kupiec and PIT against date-block bootstrap
nulls rather than textbook references, and it is the mechanism behind h=20's
failure — the same mechanism, one horizon earlier, at h=5.

**Status: FIXED (2026-08-23).** `s08` now deflates the Kish ESS by `h` at
source, and `s09` deflates its direct tail ESS and raises the raw-count bar by
`h`. Every floor — `MIN_ESS`, `MIN_TAIL_EFF`, `MIN_TAIL_RAW` — now counts
*independent* observations, so a 5-day number must clear the same evidential
bar a 1-day number does. h=1 is unchanged by construction. Consequences in
§Effect of the fixes.

---

## L4 · EWMA initialisation shock — real, direction correct, blast radius small

`s08:144` seeds the variance state with the first observed squared return:

```python
v_run = np.where(ok, np.where(seen, lam * v_run + (1 - lam) * r2, r2), v_run)
```

With `HL=20`, `lambda = 0.965936`, so the first day's weight after k steps is
`lambda^k`:

| k | 20 | 59 | 100 | 200 |
|---|---:|---:|---:|---:|
| weight | 50.00% | **12.94%** | 3.13% | 0.10% |

At obs 60 — the moment `MIN_VOL_OBS` lets sigma be used — **12.94% of the
variance is still a single day's squared return.** And first days are not
ordinary: across 3,022 stocks, first-day `r²` runs a median 1.53× the stock's
own median, p90 18×, p99 292×, max 74,388×. 1,652 stocks enter after the panel
start, so they have a genuine first-trading-day rather than an arbitrary
window edge.

Against a 20-day burn-in seed, the current initialisation inflates sigma by
**>10% for 14.4% of stocks at obs 60** (p99 3.02×, max 32.75×), decaying to
0.66% of stocks by obs 200. Inflated sigma suppresses `z` and understates risk,
exactly as expected.

The saving grace is exposure. Scored rows sit far from the contaminated region:

| stock age at scoring | < 60 d | < 100 d | < 140 d | < 260 d |
|---|---:|---:|---:|---:|
| scored rows | 122 (0.02%) | 5,251 (0.91%) | 10,504 (1.82%) | 25,324 (4.38%) |

**Status: FIXED (2026-08-23).** `s08` now seeds the variance state with an
equal-weighted running mean over the first `VOL_BURN = HL = 20` observations
before handing it to the EWMA. Still strictly causal. The first day's weight at
observation 60 falls from **12.94% to 1.25%**, a tenfold reduction.

---

## L5 · Delisting is retained, but truncated at the last traded price

The fear is that distressed names vanish from the vendor panel before they go
to zero, deleting the worst left tail. Measured:

- 3,581 stocks in the price panel; **1,198 (33.5%) stop more than 60 trading
  days before the panel ends** — dead names are *kept*, not dropped.
- **130** of those are in the scored universe. Their final price against their
  own trailing one-year high: median −12.4%, p10 −68.7%, min −98.9%, with
  **22 names worse than −50%**.

DHFL (`INE202B01012`) is the canonical case and survives into the output with
`z = −32.58`, the worst standardised loss in the panel.

**What is genuinely understated:** a name suspended at −60% that ultimately
paid nothing is recorded at −60%, not −100%. The left tail is *truncated at the
suspension price*, not fabricated away. That is a bounded, one-directional
understatement, and it is the residual survivorship the engine carries.

**What cannot be checked from inside the data:** whether the vendor
retroactively backdated corporate-action effective dates. The point-in-time
ISIN map (`isin_pit_map.py`) fixes identity resolution given the CA dates it is
handed; it cannot detect a date that was silently rewritten before delivery.
Falsifying that needs vendor snapshots taken at different times, which this
project does not have. **Stated as an unfalsifiable assumption, not as a
cleared risk.**


---

## Effect of the two fixes (2026-08-23)

Both changes were applied and the whole chain re-run. h=1 barely moves; h=5
moves a lot; h=20 collapses, correctly.

### Calibration

| Engine | h | Breach 5% | Breach 1% | ES coverage | PIT χ² | before → after |
|---|---:|---:|---:|---:|---:|---|
| Flowsense | 1 | 0.0512 | 0.0104 | 1.023 | 0.6 | essentially unchanged |
| Flowsense | 5 | **0.0518** | **0.0091** | **1.015** | **1.3** | was 0.0535 / 0.0106 / 1.025 / 2.4 |
| Flowsense | 20 | 0.1153 | *no served rows* | — | 155.1 | was 0.0567 / 0.0133 / 1.132 |

**h=5 calibration improved on every measure**: the 5% breach rate moved 0.0535
→ 0.0518, the 1% rate 0.0106 → 0.0091, ES coverage 1.025 → 1.015, PIT χ² 2.4 →
1.3. The engine had been serving 5-day tails off support that could not carry
them; once the floor counts independent observations, the numbers it does serve
are better.

### What it costs

| | h=1 | h=5 |
|---|---:|---:|
| warm-up frontier | vintage 6 (2016-07-29) | vintage 6 → **20** (2017-09-29) |
| rows below frontier | 3.4% | 1.0% → **14.1%** |
| served 1% VaR | 93.6% of outcomes | **66.9%** |
| served 5% VaR | 96.6% of outcomes | **85.8%** |

A third of 5-day rows no longer get a 1% number. That is the honest price of
the fix, and the direction is right: those rows were previously being served
off tails with roughly a fifth of the independent support the floor assumed.

### h=20 is now unservable, from first principles

Under an independent-observation floor no h=20 block clears `MIN_ESS` until
vintage 62–98, and the per-row tail floor withholds **every** 1% forecast. The
horizon was already retired from the public interface on measured
over-breaching; it now fails the data-sufficiency test directly. The frontier
gate in `s08` is therefore binding only for `PUBLIC_H = (1, 5)` — a horizon the
engine does not serve is allowed to be unservable.

### The volatility gradient — see L6, fixed separately

### Four latent assumptions this exposed

Scaling the floor falsified an assumption baked into four places that had never
been reachable before: *every cell has at least one served row*. `s09`'s J4 gate
and J5 CRPS check, `s11`'s breach-rate loop, and `s08`'s frontier reporter all
crashed on empty slices rather than reporting "nothing served". All four now
distinguish *no data* from *bad data*. None of them was wrong before — none
could be reached before.


---

## L6 · The conditional volatility gradient — FIXED (2026-08-23)

**The defect.** Breach rates ran 1.23× nominal in the calmest EWMA-volatility
quintile and 0.76× in the wildest at h=1 (1.34× / 0.73× at h=5). Outcomes are
already divided by that same volatility, so the gradient should not exist.

**Diagnosis.** Decomposing which kind of "volatility" drives it:

| quintile basis | sd(z) Q1 → Q5 | breach × nominal |
|---|---|---|
| stock **type** (the stock's average σ) | 1.056 → 1.085 | 1.02 → 0.96 — **flat** |
| **within-stock** over time | 1.218 → 0.935 | 1.36 → 0.72 — **the defect** |

It is not that volatile stocks are mishandled. A stock that is calm *relative
to its own history* has an over-dispersed outcome, and one that is agitated has
an under-dispersed one. That is **volatility mean reversion**: EWMA(20)
estimates volatility *now*, while the outcome depends on volatility over
(t, t+h]. When today is unusually quiet for this stock, tomorrow reverts upward
and |z| lands larger than the pooled density implies.

**The fix.** With `dev = log(σ_fast / σ_slow)` against a slow EWMA
(`HL_SLOW = 250`), the scale error is log-linear in `dev`, so shrinking the
deviation removes it:

```
σ* = σ_slow · (σ_fast / σ_slow)^(1+k)  =  σ_fast · exp(k · dev)
```

A constant multiplicative error would be harmless — the empirical density
absorbs any fixed scale. Only the `dev`-dependent part matters, and that is all
that is corrected. `k` is estimated **per vintage from matured observations
only** (`day + h ≤ asof`), by cumulative-by-day sufficient statistics, so the
estimate at vintage *v* sees nothing after its own asof. Median `k`: −0.238
(h=1), −0.365 (h=5), −0.573 (h=20).

### Result

| within-stock σ quintile, h=1 @5% | Q1 | Q2 | Q3 | Q4 | Q5 | spread |
|---|---:|---:|---:|---:|---:|---:|
| before | 1.36 | 1.14 | 1.01 | 0.89 | 0.72 | 0.64 |
| **after** | **1.17** | **1.08** | **1.03** | **0.97** | **0.88** | **0.29** |

Pooled-quintile strata, which is what `s11` reports: h=1 @5% now runs
1.08 / 1.11 / 1.03 / 1.02 / 0.90 and h=5 @5% runs 1.15 / 1.13 / 1.07 / 1.01 /
0.94. **No stratum is flagged MISCALIBRATED at either horizon** — three were
before. Overall dispersion also improved (sd(z) h=1 1.0614 → 1.0359, h=5
1.0986 → 1.0567) and CRPS fell at both served horizons (h=1 0.55652 → 0.54331,
h=5 0.58737 → 0.56839).

### Two things this cost, and one it caught

**The audit caught a real leak.** The first implementation estimated `k` by
summing over the whole price matrix. That matrix is indexed by `stocks`, which
is drawn from the archetype panel — so in a truncated world it silently
contains a *different stock set*, and summing over it leaks the identity of
names that only enter the universe later. `s13` failed immediately (L3/L4, over
a million rows differing). The estimator now runs over **scored rows only**,
which gate L5 certifies are identical across the two worlds. All six gates are
back to exact zeros.

**The first estimator targeted the wrong moment.** OLS on `log z²` estimates
`E[log z² | dev]`; breach rates depend on `E[z² | dev]`. With heavy tails those
slopes differ by roughly a factor of two, and the first pass under-shrank by
exactly that much, leaving half the gradient standing (Q1 1.23 rather than
1.17). Diagnosed by checking `p5 of z/sd`, which stayed flat across quintiles —
proof the residual was still *scale*, not *shape*, and therefore still fixable
by shrinkage. The estimator now takes the second-moment slope directly, with
`z` clipped to the density grid so a single −32σ day cannot set the correction
for the panel.

### What remains — and why most of it was never the model's fault

The power law left a residual spread of 0.29 (from 0.64). Closing it took two
more steps, and the second one changed what the number means.

**Step 1 — g became non-parametric.** `exp(k·dev)` was replaced by a
piecewise-linear `g(dev)` with knots on a fixed grid
(`DEV_EDGES = arange(-1.2, 0.91, 0.15)`, 16 bins, 1.19%/0.09% in the catch-all
tails), estimated per vintage as the relative second moment per bin and shrunk
toward the power law with a pseudo-count so sparse bins degrade to `exp(k·dev)`
rather than to noise. Bins are fixed and data-independent, which is what keeps
the per-bin cumulative sums causal.

This did exactly what it promises. Measured on `dev` itself, the gradient is
**gone**: sd(z) by dev quintile now runs 1.066 / 1.073 / 1.064 / 1.056 / 1.071,
a Q1/Q5 ratio of 0.995.

**But the reported gradient barely moved** (0.29 → 0.29). The functional form
was never the binding constraint.

**Step 2 — the state variable, and then the stratum.** `dev` correlates only
**+0.456** with the within-stock volatility rank the diagnostic actually cuts
on. An expanding z-score of log σ correlates +0.786 — but conditioning on that
instead, *fitted in sample* (the best possible case), moved the spread only
0.29 → 0.25.

That ruled out the model. The remaining explanation is the diagnostic itself:

| within-stock quintile, h=1 @5% | full-sample stratum | causal expanding stratum |
|---|---:|---:|
| Q1 (calmest) | 1.16 | **1.02** |
| Q2 | 1.09 | 1.05 |
| Q3 | 1.04 | 1.03 |
| Q4 | 0.99 | 0.99 |
| Q5 (wildest) | 0.85 | **1.06** |
| **spread** | **0.30** | **−0.04** |

Same engine, same rows, same forecasts. `quintiles()` cuts on the whole sample,
so "EWMA vol Q1" means *in the calmest fifth of this stock's history including
its future*. That is forward-looking information, and no causal model can price
it — knowing a day sits in the bottom fifth of a window extending past it
genuinely predicts volatility will rise. Conditioning on it manufactures a
gradient in a model that has none.

`panel_inference.expanding_quintiles` now provides the causal cut and `s11`
reports both. On the engine's own pooled report the causal strata run
**1.02 / 1.07 / 1.02 / 1.00 / 1.03** — flat, nothing flagged.

### Final status of L6

- **The real, causally-addressable defect was volatility mean reversion.** It
  was genuine, and it is fixed: the full-sample-cut spread fell from 0.64 to
  0.30 and no stratum is flagged MISCALIBRATED at either served horizon.
- **The residual 0.30 is an artifact of a forward-looking stratum**, not a
  model failure. Against a causal stratum the engine is flat to −0.04.
- The full-sample cut is still reported. It is a legitimate ex-post question —
  "how did it do in what turned out to be quiet regimes?" — it is just not a
  statement about calibration, and is now labelled as such.

**One caveat worth keeping.** The causal stratum is flat *on average*; it does
not prove the engine is well calibrated in every volatility regime, only that
the apparent gradient does not survive an honest conditioning set. The Q5
interval is wide ([0.0377, 0.0701]) because volatile regimes are clustered in
time and the block bootstrap prices that correctly.

---

# Code review round 2 (2026-08-23) — six findings, all confirmed

Raised against the code rather than the comments. All six reproduced; four
were fixed, one was a scope overclaim, one a naming error.

## R2-1 · The PIT "bootstrap null" had zero power — FIXED

`pit_chi2_bootstrap` block-resampled the **observed** PIT counts and used the
95th percentile as the critical value. That centres the reference distribution
on the alternative, so the critical value rises in lockstep with the statistic.
Tested on 600,000 synthetic draws over 2,000 dates:

| PIT sample | χ² | old crit95 | rejected? |
|---|---:|---:|---|
| uniform | 16.8 | 41.0 | no |
| u² (mass piled at 0) | 343,485 | 345,970 | **no** |
| all mass in the bottom half | 600,006 | 600,029 | **no** |
| **all mass in ONE bin** | **5,400,000** | **5,400,000** | **no** |

It could not reject a PIT concentrated entirely in a single bin. **Every "PIT
not rejected" verdict produced with it was void.**

Replaced with a null-centred block test: bin counts are split into the
expectation under uniformity and a residual, the residuals are recentred across
dates (imposing H₀ by construction), and it is the recentred residuals that are
block-resampled. The null keeps the panel's real dependence but no longer
tracks the statistic. Verified: rejects every non-uniform case above, and size
is 5.0% on truly-uniform date-clustered data (conservative, 0%, on independent
draws).

**What changed in the verdicts.** Flowsense is still not rejected at h=1
(41.8 vs 584.1) or h=5 (119.2 vs 2253.6) — that conclusion survives, though the
evidence previously offered for it did not. What changes is the benchmark: the
Gaussian now **REJECTS at every horizon** (h=1: 18,961 vs 641), where the
broken test had it passing. h=20 rejects for every system.

The textbook χ²(9) critical value of 16.9 is still far too small — the correct
block critical value at h=1 is 584 — so that half of the earlier finding holds.

## R2-2 · A failed optimiser could become a production vintage — FIXED

`_m_step_A` returned `r.x` without checking `r.success`; the EM loop recorded
`monotone_ok_` and no consumer read it; and `s05` did
`max(good or cands, ...)`, so when **no** candidate passed the structural test
the best structurally *invalid* one was published with only a counter to show
for it.

- `_m_step_A` now accepts a step only if the optimiser converged **and** the
  objective improved; otherwise it keeps the incoming `A`, which leaves Q
  unchanged and so preserves EM monotonicity.
- `fit()` exposes `ok_ = monotone_ok_ and converged_ and no M-step failures`.
- `s05` requires both gates and **raises** rather than publishing a fallback,
  naming the vintage and the per-candidate reasons. New diagnostic columns:
  `n_unsound`, `converged`, `monotone_ok`.

## R2-3 · The raw tail-support gate was wrong for hard densities — FIXED

`cnt` was stored once per vintage, `(NV, NB)`, on the argument — in the code's
own comment — that "under soft weighting the same past day contributes to all
seven components". True for soft, false for hard: a hard density is built from
one archetype's observations, so its tail was being certified by a count pooled
over all seven. Thin hard-archetype tails passed the gate on other archetypes'
data.

Counts are now per archetype, `(NV, NA, NB)`, incremented wherever an archetype
carries weight, and `s09` combines them with the *same weights the density
uses* — one-hot under hard picks that archetype's own count, soft reproduces
the pooled total. The effect is large and in the expected direction:
`hard_roll` h=5 at the 1% level now suppresses **63.5%** of rows, and
`hard_roll` h=5 1% is newly flagged MISCALIBRATED on the 3,811 rows that
survive. Those numbers were previously being served off support that was not
there.

## R2-4 · The C1 look-ahead gate crashed — FIXED

`window_rows` has returned five values since the gap-aware refit; `--with-c1`
unpacked four, so the gate raised `ValueError` and **the C1 audit had never
run**. Fixed, and inverted to `--skip-c1`: C1 is now audited by default. `[L0]`
now executes (6 vintages probed of 64 available at T).

## R2-5 · The verdict was scoped too broadly — FIXED (wording + default)

The audit rebuilds C2–C5 from stored states and vintages. It does not rebuild
the feature store, the HMM backbone, the threshold calibration, or the price
panel. The verdict now says so explicitly, names C1 in scope, and adds the
limitation that it truncates on `TR_DATE` and therefore **cannot see a
publication lag** — a row using flow data not yet reported at the time would
pass every gate.

## R2-6 · `_ret` fields were log returns — FIXED

C4 builds forward **log** returns, so `z · σ · √h` is a log return. The fields
were named `_ret` and the query layer applied `expm1` for display, so the
stored field and the served number disagreed in units. Renamed `_logret`.
Nothing else consumed them.
