# Pre-registration — aggregate rupee FII flow (written before any result was seen)

Date: 2026-08-23. Every choice below was fixed before the flow series was
joined to the market series. Nothing outside this file will be claimed.

## Hypothesis (external, not mined here)
The India literature reports that aggregate FII flows co-move with, and are
sometimes said to predict, Nifty returns and volatility. Our feature pipeline
rank-normalises within day, so this LEVEL information has never entered any
test in this project.

## Construction (fixed)
- NF(t) = [Σ buy VALUE_INR − Σ sell VALUE_INR] on day t, over rows with
  TR_TYPE ∈ {1,4}, RATE>0, RFDE_INSTR_TYPE = REG_DL_INSTR_EQ, all stocks.
- Scaled: NF(t) / trailing 250-day mean of daily gross flow (ending t−1).
  No other scaling will be tried.
- Market: nifty50_ret from returns_panel_v3 (one value per date);
  india_vix likewise.
- Eras: TRAIN ≤ 2021-04-30, TEST ≥ 2021-07-01 (the masked-gap split).

## The four pre-declared cells — and no others
Controls are fixed: 5 lags of the target; the volatility target additionally
controls india_vix(t). Newey-West 10 lags. Linear only; the single allowed
asymmetry term is NEG(t) = min(NF(t), 0), included in the vol cells.

  C1  nifty_ret(t+1)  ~ NF(t)      direction, same-day alignment
  C2  |nifty_ret(t+1)| ~ NF(t), NEG(t)   risk, same-day alignment
  C3  nifty_ret(t+1)  ~ NF(t−2)    direction, deployable (reporting lag)
  C4  |nifty_ret(t+1)| ~ NF(t−2), NEG(t−2)  risk, deployable

## Pass bar (fixed)
A cell PASSES only if BOTH hold:
  (a) full-sample |NW t| ≥ 2.50 on the flow coefficient (Bonferroni, 4 cells);
  (b) the same coefficient has the same sign with |t| ≥ 1.5 in TRAIN and in
      TEST separately.
Anything else is a FAIL for that cell. No quintiles, no thresholds, no
functional-form changes, no post-hoc sub-periods, no alternative vol proxies.
A sign that flips between eras is a fail regardless of t.

## Interpretation (fixed in advance)
- C2 or C4 passing = "aggregate FII flow carries risk information."
- C1 or C3 passing alone = return predictability, a different claim.
- All four failing = the level information is also null; reported as such.
