# Pre-registration — weekly (h=5) risk information screen (v4)

Date: 2026-08-23. Written BEFORE any weekly test was run. Fourth step
in the disclosed sequence (v1 PASS / v2 FAIL / v3 FAIL at -1.79; daily
Nifty CRPS closed). Question: at the WEEKLY horizon, do FII flow or
global conditions predict Nifty risk beyond its own history and VIX?
Two-stage design copied from PREREG_FLOW_SIGMA.md: a regression screen
gates the engine build. If the screen fails, no weekly engine is built
and the market-level program ends entirely.

## Stage 0 — screen (fixed)
Target: RV5(t) = sqrt( sum_{k=1..5} r(t+k)^2 ), next-week realized
volatility of Nifty. Controls in every cell: RV5(t-5) (own trailing
week, non-overlapping), sqrt-EWMA sigma(t) (lambda .94), india_vix(t).
Newey-West 10 lags (overlap). Eras as always (TRAIN <= 2021-04-30,
TEST >= 2021-07-01).

Three pre-declared cells — one predictor each, and no others:
  W1  NF5(t-2)  = sum of scaled NF over t-6..t-2   (flow persistence)
  W2  inr5(t)   = 5-day USDINR log return ending t
  W3  spx5      = 5-day S&P500 log return ending last US close < t

Pass bar per cell (same family as C4): full-sample |NW t| >= 2.50 AND
same sign with |t| >= 1.5 in both eras. Bonferroni intent: three cells,
bar kept at 2.50.

## Stage 1 — engine (built ONLY if >= 1 cell passes)
Weekly analogue of the v2/v3 machinery: GJR base iterated 5 steps,
Student-t tilt on the passing predictors only, FIRST_FIT 750, refit 21,
non-overlapping scoring offsets reported alongside NW-corrected DM.
Gate: tilted model vs GJR-5 base, CRPS DM t <= -2.0 full (NW 10),
better in both eras, 95/99 VaR calibration intact.

## Interpretation (fixed)
- Stage 0 all-fail: weekly risk information is also null; the whole
  market-level engine program terminates with the boundary fully
  mapped (daily: GJR frontier; weekly: no signal). Reported as such.
- Stage 0 pass but Stage 1 fail: same power-boundary reading as v2.
- Both pass: the weekly engine is the headline engine result.
No other targets, horizons, predictors or bars after seeing results.

## Result (2026-08-23, appended after the single Stage-0 and Stage-1 runs)
STAGE 0: PASS on W3 only. spx5 -> next-week Nifty RV: t = -3.50 full,
-2.72 TRAIN, -4.39 TEST (same sign). W1 flow persistence FAIL (sign
flips between eras); W2 USDINR FAIL (TRAIN-only).
STAGE 1: GATE FAIL. B1 (spx5 tilt) vs B0 (GJR-5): FULL DM t = +1.64,
not better in either era; calibration fine for both (B0 hit1 = 1.10%).
Reading, per the fixed interpretation: the same power boundary as v2 —
the S&P->Nifty vol spillover is real in conditional-mean regressions
but adds no density-forecast value because the GJR base already
transmits the shock through Nifty's own lagged returns.
The market-level engine program is now COMPLETE at both horizons:
daily (v1 PASS on simple base; v2/v3 FAIL on GJR) and weekly (screen
pass, conversion fail). No further gates will be attempted.

## Correction (2026-08-23, see AUDIT_RETRACTIONS.md R6)
Corrected re-run under the R6 fixes (2,210 scored days): STAGE 1 GATE
still FAIL, B1 vs B0 t = +1.58 (was +1.64). Stage 0 (pure regression,
no engine machinery) is unaffected. Conclusions unchanged.
