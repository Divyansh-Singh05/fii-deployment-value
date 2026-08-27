# FII Flow-Regime Project — Final Results, One Page

*If someone asks "what did this research actually find," this is the answer.*

Rewritten **2026-08-23** on the post-audit research object. Every number below
comes from a stage in `pipeline.py` (62 stages) run on the rebuilt data, not
from a previous draft. The pre-audit version of this document is preserved at
`FII_final_results_one_pager.PREAUDIT.md` and is **not paper-eligible**.

---

## The one-sentence result

The project set out to show that the **composition** of foreign institutional
flow carries risk information. At the stock-day distributional level it does
not — and the machinery built to test that question turned out to beat the
industry-standard risk model anyway, for a reason that has nothing to do with
FII flow: **Indian equity returns are far too fat-tailed for the Gaussian
assumption that standard VaR engines are built on.**

Two genuine flow effects survived, both pre-registered and both out-of-sample:
institutional **share of turnover** predicts next-day stock volatility, and
**aggregate** FII net flow improves market-level risk forecasts.

---

## 1 · The largest result: the distributional engine (Phase 3)

Walk-forward, `s + h <= asof` embargoed, **578,481 stock-days**, 562 stocks,
2016-02-02 → 2025-03-28. Both engines share the same past-only EWMA volatility
and the same evaluation dates; the only difference is the assumed **shape** of
the standardised distribution.

| Metric (h=1) | Flowsense (empirical) | EWMA-Normal (standard) |
|---|---:|---:|
| 1% breach rate (target 1.00%) | **1.04%** | 1.62% |
| Kupiec p-value at 1% | **0.673** | 0.000 |
| PIT χ² (date-block bootstrap null) | **0.7** | 303.3 |
| Per-stock Kupiec pass rate | **90.4%** | 60.4% |
| ES(1%) coverage (target 1.000) | **1.027** | 1.246 |
| Mean CRPS | **0.55785** | 0.56095 |

Holds at h=5 (breach 0.77% vs 1.99%; Kupiec 0.027 vs 0.000). **Fails at h=20**
(12.4% breach) and is not presented as a 20-day engine.

**Why the Gaussian fails — the single clearest number in the project.**
Volatility-standardised losses, observed vs. what a Gaussian expects:

| threshold | observed | Gaussian expects | ratio |
|---|---:|---:|---:|
| z < −4 | 1,562 | 18.3 | 85× |
| z < −5 | 742 | 0.17 | 4,475× |
| z < −6 | 398 | 0.00 | ~700,000× |
| z < −8 | 133 | 0.00 | — |

A Gaussian VaR engine on this market is not "slightly conservative." It is
wrong by orders of magnitude exactly where capital is lost. The empirical
engine holds ES/VaR at **1.39–1.41** against the Gaussian's structurally fixed
**1.146**.

**The honest half:** C6 found the archetype-conditional mixture statistically
**indistinguishable** from the same machinery with flow conditioning removed —
at every horizon, in every stratum, powered to detect ~0.03% of CRPS. The win
is distribution **shape**, not FII regime labels.

---

## 2 · Institutional share of turnover → next-day volatility (module 19)

577,245 stock-days, 972 stocks, 2,081 dates. Dependent variable z², i.e.
volatility the engine's own EWMA σ has already failed to explain.

| specification | FULL | TRAIN | TEST |
|---|---:|---:|---:|
| date FE | −0.0283 (t=−5.25) | −0.0370 (−4.43) | −0.0198 (−2.85) |
| **stock + date FE** | **−0.0300 (t=−5.53)** | **−0.0365 (−4.42)** | **−0.0232 (−3.25)** |
| + own lagged z² | −0.0255 (−4.59) | −0.0332 (−3.87) | −0.0180 (−2.50) |
| **share known only at t−2** | **−0.0015 (t=−0.32)** | −0.0020 (−0.30) | −0.0008 (−0.13) |

`log(turnover)` enters at **t = +24** throughout, so the volume control has real
teeth and the share effect is independent of it. Monotone across within-day
quintiles (implied σ 1.0911 → 1.0357; lowest-institutional-share days are
**+5.3% more volatile**). Robust to every winsorisation (t = −6.51 to −4.26),
stronger on `|z|` (t = −7.32), stronger in high-VIX regimes (−4.19 vs −3.13).

**The t−2 row is the finding's own limit, stated by the module itself:** the
effect dies at deployable lag. It is economics, not a tradeable signal.

---

## 3 · Aggregate FII flow → market risk (C4 pre-registration + module 21)

**The regression** (`docs/PREREG_AGGREGATE_FLOW.md`, cell C4, bar fixed in
advance at |t| ≥ 2.50 full and same-sign |t| ≥ 1.5 in both eras):
net inflow at **t−2** predicts lower next-day Nifty |return| **beyond VIX** —
**t = −4.78** full, −2.49 TRAIN, −2.51 TEST. **PASSED.**

It is not a leverage effect in disguise: adding signed-return controls
min(r,0) at t…t−2 leaves **t = −4.23** (−2.28 / −2.05 by era), and past signed
returns explain only **2.8%** of the flow variable.

**The engine** (`docs/PREREG_MARKET_ENGINE.md`, module 21). Student-t density,
EWMA base, flow tilt, all parameters walk-forward on past data only, 2,739
scored days. Pre-registered gate, all three legs:

| leg | requirement | result |
|---|---|---|
| (a) | flow-beyond-VIX CRPS DM t ≤ −2.0 | **t = −2.35** |
| (b) | better in TRAIN and TEST separately | −0.0019 / −0.0004 |
| (c) | VaR calibration intact | Kupiec 0.225 / 0.161 |

**PASSED.** Economically: ~**2%** less capital at matched 99% coverage, and
average loss beyond VaR99 falls **0.770% → 0.613%** (−20%). The gain is
monotone in the signal — −0.0003 on quiet days, −0.0119 on strong-inflow days —
and concentrated in the 1% tail (pinball t = −2.02, negative in both eras).

---

## 4 · Where the signal stops: the engine boundary

Three further pre-registered attempts, all disclosed, all reported:

| attempt | question | verdict |
|---|---|---|
| v2 (module 22) | flow on a **GJR-GARCH** base | **FAIL** (t = −0.95) |
| v3 (module 23) | USDINR / S&P500 / **DXY** / Fed–RBI rates | **FAIL** (t = −0.67) |
| v4 (module 24) | **weekly** horizon | screen PASS, engine **FAIL** (+1.58) |

Three findings fall out of these failures:

- **DXY and policy-rate differentials add nothing beyond USDINR** (+1.81). The
  rupee already prices the dollar force for India.
- **The S&P→Nifty weekly spillover is real** (screen t = −3.50 / −2.72 / −4.39,
  the most era-stable coefficient in the project) **and still worthless to a
  density engine**, because a GARCH base already absorbs it through Nifty's own
  lagged returns.
- The recurring pattern, now measured four independent times: **conditional-mean
  volatility predictors with regression t-stats of 3–5 systematically fail to
  improve a well-specified GARCH-t density forecast**, and succeed only against
  weak baselines. A ~1–2% modulation of σ is real, survives every control, and
  sits below density-score detectability.

---

## 5 · Phase 2: state estimation works, decision value does not

- **16C — episode ends are genuinely forecastable.** k=1 log-loss .4303 vs an
  age-only Kaplan-Meier .5308 (paired t **+41.6**), **AUC .797 vs .569**; and
  *stronger in TEST than TRAIN* (.809 vs .786), which rules out memorisation.
- **16D — that skill does not convert.** Acting early earns **+23bp/episode**
  CI [+7,+40] — but a trivial age rule earns +18bp [+2,+33], and head-to-head
  on 2,764 common episodes the model is **worse** (d = −15bp, t = −2.96).

*"Don't wait for confirmation" is worth ~20bp and needs no model.*

---

## 6 · What does not work (established, not assumed)

- **No tradeable strategy.** Best TEST gross Sharpe is 1.51 (S3_PROXY); at a
  realistic 15bp it is **−1.53**. Every one of eight strategies is negative net
  of costs, in both eras. Costs bind — that is the finding.
- **No tail effect.** Module 14: TEST excess CAR20 **+36bp**, CI [−51,+128],
  p = 0.374, against a **98bp** friction bar. The reversal does not generalise
  to the illiquid tail.
- **No flow conditioning value in the stock-day density** (C6, above).

---

## 7 · What was withdrawn (`docs/AUDIT_RETRACTIONS.md`)

The August 2026 audit retracted the project's original headline and five
further claims. Current status of the concentration axis, on the rebuilt object
(`outputs/tables/T1`, regenerated 2026-08-23):

| archetype | TRAIN | TEST |
|---|---:|---:|
| **HOSTAGE** (dispersed selling) | **+56.7bp (t=4.65)** | **+42.8bp (t=2.76)** |
| SHARK_DIST (concentrated selling) | +16.4 (t=1.50, ns) | +17.9 (t=1.24, ns) |

The published claim was the **opposite** — a concentrated-selling reversal — and
it was an artifact of a feature that computed each entity's *own-book*
Herfindahl instead of within-stock-day participation shares (the two correlate
**r = −0.2460**). **No replacement headline is asserted**: the surviving effect
sits on the archetype the project's own pre-registration started from, the
re-pointed skeptic gates are exploratory rather than pre-registered, and two of
them still fail in TEST.

Also withdrawn: the "PIN loads ~3× on dispersed selling" corroboration (the
contrast is insignificant in TRAIN, p=0.41, and PIN does **not** moderate the
reversal); the "+28% incremental IC" claim (dIC .0060 → .0012, t=0.67).

**R6 (found 2026-08-23, in a review of this session's own engines):** two
estimation defects — an out-of-sample maturity handicap and optimizer
path-dependence — were fixed. **Module 21's PASS is unchanged to every printed
digit** under both fixes; the v2/v3 near-misses widened into clear failures; and
one transient "recovered positive" (t = −2.79) was traced to the optimizer path
and **rejected**. This is on the record because it is the clearest evidence in
the project that small-edge engine claims can be manufactured by estimation
artifacts.

---

## 8 · What this research achieved

1. **A validated non-parametric risk engine** that beats the standard Gaussian
   benchmark decisively in the deep tail on 578k out-of-sample stock-days —
   90.4% vs 60.4% per-stock Kupiec pass, PIT 0.7 vs 303.3.
2. **Two real, pre-registered flow effects**: institutional share of turnover
   cross-sectionally (t = −5.53, both eras), and aggregate flow at market level
   (t = −4.78), the latter converting into a **passing** out-of-sample risk
   engine worth ~2% of capital.
3. **A mapped boundary** — four pre-registered gates, one pass, three disclosed
   failures — quantifying where volatility-signal research stops paying.
4. **An audit apparatus that caught six of its own errors**, including one on
   the day of writing that would otherwise have placed a fabricated positive in
   the paper.

**The unifying thesis:** this project measures the distance between statistical
significance and forecasting value — and finds it large, systematic, and mostly
unreported in the literature it responds to.

---

## Scope of the data

NSDL FII transaction data, **January 2011 – March 2025**, ~24M trades. Frozen
split at **≤2021-04-30 / ≥2021-07-01** (May–June 2021 masked: a real structural
break and the embargo buffer). On the rebuilt object TEST covers **851 stocks**
(294,243 stock-days) against TRAIN's **618** (508,563) — a different universe,
not a repeat sample. *(The pre-audit draft stated 830 vs 572 and "April 2011";
both were stale and are corrected here.)* Five months are absent from the flow
feed (2021-05/06 masked at source; 2023-06/09/11 carry a 100%-null direction
flag, 3.0% of trades, entirely in TEST); they are excluded, never imputed.
