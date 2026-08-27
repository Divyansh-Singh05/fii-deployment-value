# Audit retractions

*Consolidated record of published claims withdrawn or corrected by the
2026-08 engineering audit. Every number below is from a re-run of the
production pipeline on the rebuilt research object, not from re-reading code.*

Branch: `audit-rebuild`. Pre-audit artifacts are preserved alongside the new
ones with a `.PREAUDIT` suffix (`vintages_PREAUDIT.parquet`,
`C7_RISK_REPORT.PREAUDIT.md`, `stock_risk_profiles.PREAUDIT.csv`, …).

---

## R1 · The concentration axis measured the wrong construct (audit item 1)

**Status: headline finding WITHDRAWN. No replacement asserted.**

`entity_hhi_raw` computed a participation-weighted mean of each entity's *own
book* Herfindahl across stocks — how focused a seller is — not the Herfindahl
of participation shares *within the stock-day*. `docs/02_feature_engineering.md`
documented the second formula; `docs/paper/FII_thesis.md` §Axis 3 documented
the first; the code implemented the first. Every economic claim interpreted it
as the second. The two correlate **r = −0.2460** on the rebuilt panel.

Re-running the full Phase I battery on the corrected feature:

| Test (published spec) | Published | Corrected |
|---|---|---|
| Panel R2 TRAIN | SHARK_DIST +54.0 (t=3.5) | SHARK_DIST +21.1 (t=1.37, ns) |
| Panel R2 TEST | SHARK_DIST +47.5 (t=2.9) | SHARK_DIST +18.2 (t=1.26, ns) |
| Panel R2 TRAIN | HOSTAGE ≈ 0 (p≥0.62) | **HOSTAGE +78.6 (t=4.47)** |
| Panel R2 TEST | HOSTAGE ≈ 0 (p≥0.62) | **HOSTAGE +42.4 (t=2.73)** |
| Event study, END anchor | SHARK_DIST +68 / +33 bp | SHARK_DIST −1 (p=0.94) / +13 (p=0.55) |
| Event study, END anchor | HOSTAGE null | HOSTAGE +48 (p=0.002) / +23 (p=0.21) |
| Sell-side block-deal rate | SHARK_DIST 0.73% < HOSTAGE 0.96% | **SHARK_DIST 1.08% > HOSTAGE 0.68%** |
| Day-0 relative volume | SHARK_DIST 1.12× / 1.13× | SHARK_DIST 1.01× / **0.93×** (gate: >1) |
| PIN loading | dispersed ~3× | `sh_host` 0.1334/0.1369 vs `sh_sd` 0.1007/0.0394 ns |

The reversal survives ~11 alternative specifications — but it has moved to
**dispersed** selling, which is the reading the project's own pre-registration
started from and claimed to have overturned.

**Why no replacement claim is stated.** Two unresolved problems:

1. **Internal conflict.** The panel regression says dispersed selling *reverts*
   (transitory). The PIN estimator loads on dispersed selling, which reads as
   *informed* (permanent). Both cannot be true.
2. **The pre-registered skeptic gates now fail.** All three passed pre-audit:
   - T1 bounce → **FAIL**, "the reversal is materially bid-ask bounce"
   - T2 SD–HOSTAGE contrast → **FAIL**, "not distinguishable"
   - T3b public volume dummy → **FAIL**, "a free volume signal absorbs it"

One thing the correction *fixed*: block-deal corroboration previously refuted
the block-impact mechanism (SHARK_DIST enriched *less* than HOSTAGE, z=−4.2).
On the corrected measure concentrated selling does coincide with real block
and bulk sell deals, as theory predicts. That is evidence the new feature
behaves like genuine concentration.

## R1a · The adversarial gates were hardcoded to the wrong archetype

**Resolved 2026-08-23.** The "all skeptic gates now fail" reading in R1 was
itself partly an artifact. Every verdict in `module15_skeptic_tests.py` and
`module13b_incremental_value.py` is hardcoded to `D_SHARK_DIST`, because that
is where the effect sat when they were written. On the corrected feature
SHARK_DIST is not distinguishable from zero, so T1 divides a bounce-free
coefficient by a null, and T3b asks whether a null survives a control. Those
are not meaningful questions and "FAIL" is not the right reading of either.

Both modules now also report a **post-hoc re-pointed** block: the same gates
applied to whichever archetype carries the reversal, selected by a mechanical
rule (largest TRAIN t-statistic). This is an exploratory re-test on data
already seen and **cannot restore the evidential status of a
pre-registration** — it is labelled as such in the code and the output.

| Gate | As printed (SHARK_DIST) | Re-pointed (HOSTAGE) |
|---|---|---|
| T1 bid-ask bounce | FAIL "materially bounce" | **FAIL, narrowly** — TRAIN 68% retained p=0.0010 ✓; TEST 56% p=0.0778 ✗ |
| T2 SD–HO contrast | FAIL "not distinguishable" | **FAIL, genuinely** — symmetric; TRAIN p=0.0153 ✓, TEST p=0.3029 ✗ |
| T3b public volume dummy | FAIL "volume absorbs it" | **PASS** — HOSTAGE t=+4.61/+2.75 vs `D_VCR` t=+1.73/+0.73 |
| T3a GBT ladder | PARTIAL/FAIL | **FAIL** — archetype-independent; dIC +0.0014 (t=1.07), was +0.0060 (t=3.09) |
| 13B T1 flow controls | "DEGRADED" | **PASS** — +56.7→+41.6 (p=0.0003) / +42.8→+40.4 (p=0.0085) |
| 13B T2 incremental IC | below bar | **FAIL** — archetype-independent; dIC +0.0012 (t=0.67), was +0.0060 (t=3.09) |
| 13B COMP-only spread | — | **SURVIVES** — Q5−Q1 +74.8bp (t=2.86); pre-audit +73.2bp (t=2.84) |

**What survives.** The reversal is not explained by conventional flow
magnitude or imbalance (13B T1, both eras), is not absorbed by a free public
volume proxy (T3b, both eras), and the composition block's *extremes* still
carry the same spread they did pre-audit (+74.8 vs +73.2 bp).

**What genuinely fails, on the corrected feature and independent of
archetype.** The composition block adds no *average* incremental information:
dIC collapses from +0.0060 (t=3.09, passing both pre-registered bars) to
+0.0012–0.0014 (t=0.67–1.07), and turns *negative* once conventional flow is
already in the model. The pre-audit claim of "+28% relative IC" is withdrawn.

**The consistent weak point is out-of-sample replication.** TRAIN is strong
everywhere; TEST fails the bounce-free window (p=0.078), the SD–HOSTAGE
contrast (p=0.303), and the incremental-IC bars.

## R1b · The PIN conflict — resolved, and the corroboration claim withdrawn

**Resolved 2026-08-23** by `pin_conflict` (`module16_pin_conflict.py`), with
its reading pre-committed in the module header before running.

R1 flagged an apparent contradiction: the panel says dispersed selling
*reverts* (transitory, liquidity), while the FII-PIN regression loads on
dispersed selling (*informed*, permanent). Three tests:

**A · The PIN loading is not specific to dispersed selling.** All three
archetype shares load positively at similar magnitude against the omitted
ROBOT/untagged base — including SHARK_ACC, *concentrated buying*, which no
information story about dispersed selling predicts:

| | `sh_host` | `sh_sd` | `sh_sa` | contrast `sh_host − sh_sd` |
|---|---:|---:|---:|---|
| TRAIN | +0.1370 (t=5.24) | +0.1039 (t=3.64) | +0.1249 (t=5.57) | +0.0331, **p=0.4055 ns** |
| TEST | +0.1161 (t=4.73) | +0.0297 (t=0.90) | +0.0578 (t=2.18) | +0.0864, p=0.0426 |

**The "~3× informed-trading loading on dispersed selling" claim is
withdrawn.** The contrast against concentrated selling is insignificant in
TRAIN and only marginal in TEST. What the pooled coefficients mostly capture
is a common FII-activity component, not an information signal.

**B · The association is real and within-stock.** With stock fixed effects
`sh_host` survives (+0.0992 t=3.56 TRAIN; +0.1108 t=3.43 TEST) while `sh_sd`
dies (+0.0017 / +0.0142, both ns). So it is not merely a stock-type
confound: within a stock, years with more dispersed-selling days do carry
higher PIN.

**C · Decisive — PIN does not moderate the reversal.** Interacting the
archetype dummy with a pre-frozen era-median `hiPIN` split, inside the same
panel specification the reversal is measured in:

| | `D_HOSTAGE` | `× hiPIN` | reversal lo-PIN | hi-PIN |
|---|---:|---:|---:|---:|
| TRAIN | +41.4 (t=2.23) | +26.7, **t=0.99 ns** | +41.4 bp | +68.1 bp |
| TEST | +47.8 (t=1.75) | −24.6, **t=−0.89 ns** | +47.8 bp | +23.1 bp |

The interaction is insignificant in both eras **and flips sign between
them**. Per the pre-committed reading, that is the "interaction ≈ 0" branch.

**The resolution.** The two results are not in conflict, because they are
statements about different objects. PIN is associated with *how often a stock
sits in the dispersed-selling state* — a stock-year frequency property. It
has **no episode-level bite**: the price path after a dispersed-selling
episode is statistically the same whether the stock is high- or low-PIN. If
dispersed selling were informed *as an episode*, its impact should be more
permanent where information asymmetry is higher. It is not.

So PIN cannot arbitrate the transitory/permanent question at all, in either
direction. It was never independent corroboration of the price result; it was
a separate fact about state frequency that was read as though it spoke to
price impact.

**Honest limitation.** Test C is not sharply powered: the interaction standard
errors are ≈27 bp, so the 95% intervals are roughly [−26, +80] (TRAIN) and
[−79, +30] (TEST). A moderate PIN moderation cannot be excluded — only a large
one. The claim here is that PIN provides no evidence of moderation, not that
moderation is proven absent.

## R2 · Tail fatness at h = 20 had the sign of its excess inverted (item 9)

`tailfat` divided mean *predicted* ES by mean *realised* VaR — different row
populations. Published **1.1306** (thesis table: 1.121), *below* the Gaussian
1.1457, i.e. a thinner-than-normal tail. Correctly paired: **1.4282**. The
tail gets fatter with horizon, not thinner.

| h | Flowsense ES/VaR | Gaussian | excess |
|---|---:|---:|---:|
| 1 | 1.3832 | 1.1457 | +0.2375 |
| 5 | 1.3812 | 1.1457 | +0.2355 |
| 20 | **1.4282** | 1.1457 | **+0.2825** |

## R3 · The PIT rejection was an artifact of the wrong null (item 10)

Scored against a textbook χ²(9) critical value of 16.9. Under panel dependence
that reference is wrong by ~36×: the statistic's own date-block bootstrap null
has a 95th percentile of 618.0 at h = 1. Flowsense's PIT is **not rejected at
any horizon** (h=1 38.0 vs 618.0; h=5 175.1 vs 2193.7; h=20 1050.3 vs 10194.6).
The claim that it was "rejected outright" is withdrawn.

## R4 · Thread B (TVTP) is void on its premise (item 3)

Pre-registered to fix a regime-persistence deficit that was an artifact of
estimating `A` per observation and deploying it per trading day — a 77% error
in mixing half-life. Corrected half-life **33.8 d** against a pre-audit 8.7 d
and a k=1 benchmark of 20.4 d. The chain is *more* persistent than the
benchmark; the deficit does not exist.

## R5 · Threshold values are not comparable across the audit

Vintage-median cut-points were re-derived on the corrected feature: hostage
−0.1546, shark_dist +0.8124, dispersed_acc −0.1233, shark_acc +0.8540
(pre-audit −0.513 / +0.877 / −0.593 / +0.795). The two measures are not
commensurable, so the old and new numbers cannot be compared.

## R6 · Market-engine tilt estimation: maturity and optimizer-path defects
(found 2026-08-23 in a code review of modules 21-24; both fixed in code)

Two defects in the walk-forward tilt estimation of the market-level
engines (modules 21/22/23/24):

1. **Estimation maturity.** In the GJR-based engines (v2/v3/v4) the base
   sigma only exists from day 750, so the first tilt vintages were fit on
   0-500 pairs and those parameters were nevertheless SCORED. A 2-parameter
   base barely suffers; 5-8 parameter feature models suffer badly, so the
   defect asymmetrically handicapped exactly the models the gates tested,
   over ~18% of the scored window. Fix: MIN_PAIRS = 500 — no parameter is
   applied out-of-sample on fewer training pairs; immature blocks are left
   unscored, symmetrically for every model. v1 (EWMA base, ~690 pairs at
   its first vintage) is untouched by construction.
2. **Optimizer path-dependence.** Warm-started BFGS with default
   tolerances stopped at path-dependent points on the flat small-edge
   likelihood: identical training data reached different parameters
   depending on the (defective) early-fit trajectory, flipping one
   diagnostic comparison between t = -2.79 and t = -0.90. Fix: tight-gtol
   BFGS plus a Nelder-Mead polish, making fits warm-start-independent.

Consequences after the corrected re-runs (single re-run each, same gates):
- **v1 (module 21) PASS is unchanged to every printed digit** under both
  fixes — it was never touched by either defect. It remains the engine
  headline.
- v2/v3/v4 gates still FAIL, now by wider margins (G2 vs G1 -0.95;
  H1 vs H0 -0.67; weekly +1.58). The previous near-misses were partly
  defect noise, not suppressed signal.
- RETRACTED from the v2/v3 result appendices (contaminated-run numbers):
  "the VIX tilt breaks 99% calibration on the GJR base" (corrected G1
  Kupiec p = 0.248); "H1 repairs H0's calibration 0.023 -> 0.622" (the
  properly-estimated H0 calibrates fine: hit1 1.25%, p = 0.248); the
  TEST-era DM values -3.04/-2.88 (corrected: -1.88/-1.63, direction only);
  G2a-vs-G1 t = +3.09 (corrected +0.28).
- The transient mature-window finding G3 vs G0 t = -2.79 (flow-only tilt
  on GJR) was investigated as a possible hidden positive and REJECTED:
  it does not survive the optimizer fix (-0.90). It was an artifact of
  the defective warm-start path.

---

## Not affected

The engineering audit's other findings stand and the risk engine is unchanged
by R1: C9 reproduces every pre-truncation quantity exactly (291,003 rows,
max|diff| 0.000e+00), and items 2–8 and 10–12 are implemented and gated. The
research logs under `docs/research_log/` are **not** rewritten — they are a
dated record of what was believed at the time, and are left intact
deliberately.
