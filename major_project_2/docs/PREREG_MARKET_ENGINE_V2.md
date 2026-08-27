# Pre-registration — market flow engine v2 (module 22)

Date: 2026-08-23. Written AFTER seeing v1 results (module 21: gate
passed, DM −2.35 flow-beyond-VIX, tail-concentrated, ~2% capital
efficiency) and BEFORE v2 was run. Sequential research, disclosed as
such. Motivations fixed here: (i) v1's EWMA base is below the GARCH
standard a referee will demand; (ii) v1 uses one aggregate number per
day — net flow — ignoring other level dimensions the project has never
tested (persistence, breadth, intensity).

## Changes relative to v1 — and no others
1. Base volatility: GJR-GARCH(1,1) with leverage, Gaussian QMLE, refit
   every 21 days on the expanding past, variance filtered with the
   vintage's parameters. Replaces EWMA. Everything else (Student-t tilt
   MLE, clamp [0.5,2], FIRST_FIT 750, refit 21, eras) unchanged.
2. Flow feature set, fixed at exactly five (all at deployable lag):
     nf+  = max(NF(t−2),0), nf− = min(NF(t−2),0)   (v1's pair)
     nf5  = Σ NF over t−6..t−3   (persistence beyond freshest day;
             window chosen to not overlap nf±)
     br   = breadth(t−2) − trailing 250d mean ending t−3, where
             breadth(d) = (#stocks net bought − #net sold)/#traded
     gi   = ln(gross(t−2) / trailing 250d mean of gross ending t−3)
3. Models: G0 = GJR-t base; G1 = G0 + [vix]; G2 = G0 + [vix, all 5
   flow features]; diagnostics (not gating): G2a = G0 + [vix, nf+, nf−]
   (v1 features on the new base, to attribute any gain), G3 = G0 +
   [5 flow features, no vix].
4. Declared economic metrics (reported, not gating): mean |VaR99| at
   matched 1% coverage; mean excess loss beyond VaR99.

## Gate (fixed; same bar as v1)
"Flow improves market risk forecasts on a competitive base" PASSES iff
for G2 vs G1: (a) full-sample CRPS DM t ≤ −2.0; (b) lower mean CRPS in
TRAIN and TEST separately; (c) G2 Kupiec p > 0.01 at 95% and 99%.

## Interpretation (fixed in advance)
- Gate passes on GJR base: the claim survives a competitive benchmark —
  supersedes v1 as the headline.
- Gate fails but v1 stands: the flow signal is real but redundant once
  the base model is strong; BOTH results are reported, v1 is NOT
  promoted as the headline in place of v2.
- G2a vs G2 attributes any gain between old and new features.
No further feature sets, bases, clamps, horizons or metrics will be
tried after seeing v2 results. A failed v2 will not be re-run with
tweaks under this or any successor pre-registration without a new
dated section disclosing the failure first.

## Result (2026-08-23, appended after the single run)
GATE FAIL. G2 vs G1 FULL DM t = +0.61; G2 not better in either era.
Reported per the fixed interpretation above; v1 is not promoted.
Diagnostics from the same run (declared models/metrics only):
- G2a (v1 features on GJR base) is WORSE than VIX-only (t = +3.09):
  the v1 nf± tilt does not transfer to a strong base.
- G3 (GJR + flow, no VIX) is the best model pointwise: CRPS 0.5121 vs
  G0 0.5128, better in both eras, best 99% VaR calibration (hit 1.17%,
  Kupiec p = 0.389), lowest matched-coverage capital (2.451%). Not
  significant (DM t = -1.36) and NOT claimed as a pass.
- The VIX tilt breaks 99% calibration on the GJR base (G1 hit 1.61%,
  Kupiec p = 0.003).
Exploratory follow-up (labelled, not pre-registered): the C4 regression
coefficient on NF(t-2) SURVIVES adding signed-return leverage controls
min(r,0) at t..t-2: t = -4.23 full, -2.28 TRAIN, -2.05 TEST; R2 of
NF(t-2) on r(t-2..t-7) is 0.028. The v2 failure is therefore a POWER
boundary (a ~1-2% sigma modulation is below CRPS detectability on a
GJR base over 2,739 days), not evidence the signal is spurious or a
leverage proxy.

## Correction (2026-08-23, see AUDIT_RETRACTIONS.md R6)
The result section above was produced by runs with two estimation
defects (tilt maturity; optimizer path-dependence), both fixed in code.
Corrected single re-run (2,235 scored days): GATE still FAIL, G2 vs G1
t = -0.95. Superseded claims from the contaminated run: G2a-vs-G1
t = +3.09 (now +0.28); "VIX breaks 99% calibration" (corrected G1
Kupiec p = 0.248); G3 pointwise superiority is now within noise
(t = -0.90) though it retains the lowest matched-coverage capital
(2.402%). The transient mature-window G3 t = -2.79 was investigated
and rejected as an optimizer-path artifact.
