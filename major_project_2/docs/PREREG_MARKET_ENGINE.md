# Pre-registration — market-level flow risk engine (module 21)

Date: 2026-08-23. Written BEFORE the engine was run. The aggregate flow
signal itself was validated in PREREG_AGGREGATE_FLOW.md (cell C4 PASSED:
NF(t−2) → |nifty_ret(t+1)| beyond VIX, t=−4.78 full, −2.49/−2.51 by era).
This document fixes how that regression finding is converted into a
forecasting engine and what counts as success. Nothing outside this file
will be claimed.

## Design (fixed)
Target: next-day Nifty 50 log return r(t+1). Info set at close of day t:
returns ≤ t, India VIX ≤ t, aggregate FII flow ≤ t−2 (reporting lag,
same alignment as cell C4).

Base: sigma_EWMA(t), RiskMetrics lambda = 0.94, 60-day burn-in seed.
Density: r(t+1) = s · m(t) · sigma_EWMA(t) · T_nu, Student-t.
Tilt: m(t) = exp(theta · x(t)), clamped to [0.5, 2.0] at forecast time.

Models (feature sets, and no others):
  M0 ewma_t    x = []                                (s, nu only)
  M1 vix       x = [ln(VIX(t)/20)]
  M2 vix_flow  x = [ln(VIX(t)/20), NF+(t−2), NF−(t−2)]
  M3 flow      x = [NF+(t−2), NF−(t−2)]              (diagnostic only)

NF(t) = (Σbuy − Σsell VALUE_INR) / trailing 250-day mean of daily gross
flow ending t−1 (exactly the C4 construction; TR_TYPE ∈ {1,4}, RATE>0,
REG_DL_INSTR_EQ). NF+ = max(NF,0), NF− = min(NF,0). Days with no flow
observation (the masked 2021-05/06 gap and the three TR_TYPE-null months
2023-06/09/11) enter as NF = 0: the engine sees "no information", never
imputed direction.

Walk-forward: all parameters (s, nu, theta) fit by Student-t MLE on the
expanding past only; first fit after 750 scored days; refit every 21
trading days; parameters applied out-of-sample to the next block. No
parameter ever sees its own scoring block.

## Evaluation (fixed)
Primary metric: CRPS (quantile-decomposition, tau = 0.01…0.99).
Secondary: pinball loss at 5% and 1%; VaR(95/99) hit rates with Kupiec
and Christoffersen tests. Eras: TRAIN ≤ 2021-04-30, TEST ≥ 2021-07-01,
plus full sample. DM tests: Newey-West 5 lags on daily loss differences.

## Decisive gate (fixed)
The engine claim "aggregate FII flow improves market risk forecasts"
PASSES only if BOTH hold for M2 vs M1 (flow beyond VIX):
  (a) full-sample CRPS DM t ≤ −2.0 (M2 better);
  (b) M2's mean CRPS is lower than M1's in TRAIN and in TEST separately.
Supporting (not gating): M1 vs M0 documents the VIX layer; M3 vs M0
documents flow alone; VaR hit rates must not be rejected by Kupiec at 1%
for M2 (an engine that wins CRPS by breaking calibration fails).

## Interpretation (fixed in advance)
- Gate passes: the C4 regression finding survives conversion to a real
  out-of-sample forecasting engine — the paper's headline engine result.
- (a) fails but (b) holds: directionally consistent, underpowered; will
  be reported as such, not upgraded by metric shopping.
- M2 worse than M1: the finding does not convert; reported as a negative
  result exactly like the stock-day sigma correction.
No other metrics, horizons, feature sets, clamps, or refit schedules
will be tried after seeing results.
