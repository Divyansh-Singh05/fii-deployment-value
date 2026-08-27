# Pre-registration 2 — flow-augmented volatility (written before any result)

Date: 2026-08-23. Follows the C4 pass in PREREG_AGGREGATE_FLOW.md: aggregate
FII net flow at t−2 predicts next-day market |return| beyond VIX, both eras,
inflow side only. This prereg fixes how that market-level signal is tested
INSIDE the engine before anything is looked at.

## Stage 0 — transmission gate (run first; stop here if it fails)
Question: does the engine's OWN volatility standardisation miss the flow
signal? The engine's z is already divided by EWMA·g(dev) σ. If σ were
complete, daily mean z² would be unpredictable from NF(t−2).

- Series: daily equal-weight mean of z_h1² across scored stocks (current
  outcome_z.parquet, all fixes in force); NF exactly as in prereg 1.
- Regression: ⟨z²⟩(t) = a + b·NF(t−2) + c·NEG(t−2) + 5 lags of ⟨z²⟩. NW-10.
  (z_h1 dated t covers t→t+1, matching the market cell's alignment.)
- Bar (same style as prereg 1): |t(b)| ≥ 2.50 full sample AND same sign with
  |t| ≥ 1.5 in TRAIN and TEST separately. Eras as before.
- Predicted sign from C4: b > 0 is FAIL regardless (inflows must LOWER z²).

## Stage 1 — engine integration (only if Stage 0 passes)
- Multiplier m(t)² = clip(1 + b̂·NF(t−2) + ĉ·NEG(t−2), 0.25, 4.0), with b̂, ĉ
  re-estimated per vintage from matured data only (dates ≤ asof), expanding.
- New system `soft_fs`: identical to soft_roll with σ* = σ·m. C4→C6 re-run.
- Primary bar: DM on CRPS, soft_fs vs soft_roll, h=1: NW t ≥ 2.0 full sample
  in soft_fs's favour AND same sign in both eras.
- Secondary (reported, not claimed): NF-quintile breach-rate spread at h=1/5%
  shrinks vs baseline.
- No other σ forms, lags, clips or horizons will be tried. A Stage-1 fail is
  reported as: the signal exists but does not improve the engine.
