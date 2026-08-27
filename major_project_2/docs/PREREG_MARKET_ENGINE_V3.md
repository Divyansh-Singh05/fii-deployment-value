# Pre-registration — market flow engine v3: global forces (module 23)

Date: 2026-08-23. Written BEFORE v3 was run. Disclosure: this is the
THIRD engine attempt in a disclosed sequence — v1 (EWMA base) PASSED its
gate; v2 (GJR base) FAILED its gate; the exploratory follow-up showed
the flow signal survives leverage controls but modulates sigma by only
~1-2%, below CRPS detectability on a strong base. v3 tests a different
hypothesis with MORE prior power, not a re-try of the same one: global
dollar/rate/US-equity conditions — the documented DRIVERS of FII flows —
may carry the information at larger magnitude than the flow echo does.
The paper will report all three attempts whatever v3 shows.

## Data (fixed; snapshots in data/VALIDATION_DATA/external_macro/)
DTWEXBGS (broad dollar, daily), DFF (fed funds, daily),
IRSTCI01INM156N (India call money rate, monthly), plus local
usdinr.parquet and sp500.parquet.

## Features (fixed, at deployable availability)
At India close of day t, forecasting r(t+1):
  inr5  = 5-day log return of USDINR ending t          (INR pressure)
  spx5  = 5-day log return of S&P500 ending the last US close
          strictly before India day t (join_asof backward on t-1;
          US close 21:00 UTC precedes India close 10:30 UTC next day)
  spxn1 = min(1-day S&P500 log return at that same US close, 0)
  dxy5  = 5-day log return of DTWEXBGS, same availability rule as spx
  ddif  = change over 63 trading days of (India call rate − fed funds);
          the monthly India rate for month m enters from the 15th of
          month m+1 (conservative publication lag), DFF from t−1
Flow: nf± = max/min(NF(t−2),0), the v1 pair, C4 construction.

## Models — and no others
  H0 = GJR(1,1)-t walk-forward base (identical machinery to v2's G0)
  H1 = H0 + [inr5, spx5, spxn1]              (parsimonious global tilt)
  H2 = H0 + [inr5, spx5, spxn1, dxy5, ddif]  (full global tilt)
  H3 = H2 + [nf+, nf−]        (diagnostic: does flow survive globals?)
Tilt estimation, clamp [0.5, 2.0], FIRST_FIT 750, refit 21, Student-t
MLE, eras, metrics: all identical to v1/v2. Parsimony is deliberate:
v2 showed estimation noise on a small signal actively hurts.

## Gate (fixed; same bar as v1/v2)
Primary claim "global conditions improve Nifty risk forecasts beyond a
competitive GARCH base" PASSES iff for H1 vs H0:
  (a) full-sample CRPS DM t <= -2.0;
  (b) lower mean CRPS in TRAIN and TEST separately;
  (c) H1 Kupiec p > 0.01 at 95% and 99%.
Secondary (declared, reported, not gating): H2 vs H0 same three checks;
H3 vs H2 (flow beyond globals); economic metrics (matched-coverage
capital, ES gap) for all models.

## Interpretation (fixed in advance)
- H1 passes: the global-transmission engine is the headline; FII flow's
  role is reinterpreted via H3 (channel vs. independent signal).
- H1 fails, H2 passes: reported as pass of the declared secondary only,
  with the multiplicity caveat stated plainly.
- Both fail: the GJR base stands as the frontier for this market at
  daily horizon; the engine program ends and the paper reports the
  completed boundary. No v4 will be attempted on daily Nifty CRPS.

## Result (2026-08-23, appended after the single run)
PRIMARY GATE FAIL on leg (a) only: H1 vs H0 full-sample DM t = -1.79
(bar -2.0); legs (b) both-era improvement and (c) calibration PASSED.
SECONDARY (H2) FAIL. Reported per the fixed interpretation; no pass is
claimed. Declared-metric observations from the same run:
- H1 repairs H0's marginal 99% VaR calibration: hit 1.46% -> 1.10%,
  Kupiec p 0.023 -> 0.622, with era-consistent CRPS gains.
- TEST era alone: H1 vs H0 t = -3.04, H2 vs H0 t = -2.88 — global
  spillovers into Nifty risk are strong post-2021 and weak before;
  noted as heterogeneity, not claimed (post-hoc era emphasis).
- DXY and Fed-RBI rate differential add NOTHING beyond USDINR + S&P500
  (H2 vs H1 t = +2.25, i.e. actively worse): the India-relevant dollar
  information is already in the rupee itself.
- Flow adds nothing beyond globals (H3 vs H2 t = +1.56).
Per the interpretation clause, no further attempt will be made on
DAILY Nifty CRPS. The weekly (h=5) question is distinct and remains
open; any attempt on it requires its own pre-registration citing this
sequence.

## Correction (2026-08-23, see AUDIT_RETRACTIONS.md R6)
The result section above came from runs with the R6 estimation defects.
Corrected single re-run (2,235 scored days): PRIMARY still FAIL, now
H1 vs H0 t = -0.67 (was -1.79 — the near-miss was partly defect noise);
SECONDARY still FAIL (+0.12). Superseded claims: "H1 repairs H0's 99%
calibration" (properly-estimated H0 is fine: hit1 1.25%, Kupiec 0.248);
TEST-era t = -3.04/-2.88 (corrected -1.88/-1.63, direction only).
Directionally retained: DXY/rate-differential add nothing beyond
USDINR+S&P (H2 vs H1 +1.81); flow adds nothing beyond globals (+0.69).
