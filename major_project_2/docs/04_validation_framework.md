# Validation Framework

Phase: `validation` (stages `event_study` … `pin_model`) + `audit`.
The unabridged evidence trail, including every failure and correction, is
`research_log/FII_Module5_validation_log.md` (§3a–§3o).

## Philosophy

Three rules, enforced across all stages:

1. **Pre-registration.** The verdict rule (what PASS/FAIL means) is written
   into the stage header before the numbers exist.
2. **One claim per stage.** Data repairs are never bundled with model-facing
   analysis; diagnostics are read-only and precede fixes.
3. **Baselines, not zero.** Every effect is tested against the relevant
   counterfactual (era baseline, regime baseline, mechanical twin) — the +52
   bp/20d TEST-era drift makes "different from zero" meaningless.

## 1. Event study (`event_study`, `liquidity_shock`)

Unit: the archetype *episode* (maximal run of one archetype). Anchors: START
(drift while flow is active) and END (reversal — flow has stopped). For
stock $i$ with market-adjusted, CA-adjusted, clipped returns $ar_{it}$:

$$CAR_i(h) = \sum_{t=\tau_i+1}^{\tau_i+h} ar_{it}, \qquad
\text{excess } CAR = CAR^{arch}(h) - CAR^{baseline}(h),$$

with date-clustered bootstrap intervals and delisting truncation (events
kept, windows truncated at last trade — dropping them is survivorship bias).
The mechanism stage adds the volume arc: relative volume = volume / trailing
20-day own average.

**Why difference-in-differences**: the first inference design (test vs zero)
made everything "significant" including the placebo — caught, corrected,
logged (§3f).

## 2. Panel regression (`panel_regression`, `robustness`)

$$y_{it} = \alpha_i + \delta_t + \sum_{a} \beta_a D^{a}_{it}
 + \boldsymbol{\gamma}'\mathbf{z}_{it} + \varepsilon_{it}$$

- $y_{it}$: forward 20-day abnormal return (bp) from stock-day $t$.
- $\alpha_i, \delta_t$: stock and **date** fixed effects (date FE absorb all
  market-wide drift, including VIX levels).
- $D^a_{it}$: archetype-END event dummies; omitted category UNTAGGED.
- $\mathbf{z}$: rolling $\beta_{120}$, momentum $t{-}126..{-}21$, Amihud
  illiquidity, log turnover, 20-day vol, relative volume, log price, log
  episode length; spec R2 adds the pre-episode 20-day return as a
  deliberate *bad control* (conservative lower bound — absorbs generic
  loser-reversal).
- Standard errors: two-way clustered (stock × calendar month) via
  `linearmodels.PanelOLS.fit(cov_type="clustered", ...)`.

Estimated regressors (archetype labels) attenuate toward zero — the
estimates are conservative. Language discipline: *predicts conditional on
controls*; never causal.

**Headline (R2, both eras)**: SHARK_DIST +65.4***/+48.6***, SHARK_ACC
−87.9***/−47.6***, HOSTAGE ≈ 0. Robustness: non-overlap episodes
*strengthen* (+76/+108), horizons 10–60 build & persist, dose-response
right-signed, ROBOT placebo shown structurally invalid at the END anchor
(its end *is* a directional transition) — HOSTAGE is the valid null.

## 3. Flow-surprise control — NET_INNOV (`flow_innovation`)

Per stock, an AR(5) on scaled net flow $n_{it} = NET_{it}/\overline{GROSS}_i$:

$$n_{it} = c_i + \sum_{k=1}^{5} \phi_{ik}\, n_{i,t-k} + u_{it}, \qquad
INNOV_{it} = \hat u_{it}.$$

**Deliberate look-ahead** (full-sample fit), with the rationale on record:
INNOV is a *yardstick*, not a claim; hindsight makes it the best-case
competitor, so SHARK_DIST surviving the control (+65.4, unchanged to the
decimal) is the stronger result. Its own forward IC (−0.023/−0.017, t
−3.2/−2.6 both eras) is a **decomposition** finding, not a tradable signal.

## 4. FII-PIN — Easley–O'Hara (`pin_model`)

Daily buy/sell counts $(B_t, S_t)$ are modeled as a mixture over news
states: no news (prob $1-\alpha$), bad news ($\alpha\delta$), good news
($\alpha(1-\delta)$), with Poisson arrival rates — uninformed
$\varepsilon_b, \varepsilon_s$, informed $\mu$ added to the event side:

$$\mathcal{L} = \prod_t \Big[(1-\alpha)\, P_{\varepsilon_b}(B_t) P_{\varepsilon_s}(S_t)
 + \alpha\delta\, P_{\varepsilon_b}(B_t) P_{\varepsilon_s+\mu}(S_t)
 + \alpha(1-\delta)\, P_{\varepsilon_b+\mu}(B_t) P_{\varepsilon_s}(S_t)\Big],$$

maximized per stock-year (L-BFGS-B, Lin–Ke logsumexp stabilization, logit/log
transforms). The probability of informed trading:

$$PIN = \frac{\alpha\mu}{\alpha\mu + \varepsilon_b + \varepsilon_s}.$$

Result: PIN loads on HOSTAGE share (+0.231***/+0.149***) ≈ 3× its SHARK_DIST
loading — an **independent endorsement** (inputs share nothing with the
return tests) of dispersed = information, concentrated = liquidity.
Limitation: FII-slice, not all-market.

## 5. Model-choice gate (`gbt_challenger`, `demeaning_check`)

LightGBM with all 10 features vs a frozen regime baseline. Decomposition by
within-stock demeaning (TRAIN-era means) separates *which-stock* from
*which-day* signal. Pre-registered bar for justifying sequence models:
demeaned quintile-spread t > 2. Observed: t = 1.48 → **not met**; the HMM is
not the bottleneck. (A single-feature IC artifact — missing per-day dropna —
was caught and owned here; the GBT itself was unaffected.)

## 6. State dependence (`vix_lambda`) — reported, not claimed

VIX enters only via interaction (levels absorbed by date FE). VIX×SHARK_DIST
significant in TEST only (non-replicating → suggestive). FII-flow Kyle λ
(trailing $\mathrm{cov}(ar, flow)/\mathrm{var}(flow)$) interaction null; the
raw high-λ amplification is a characteristic effect the FE machinery
correctly prevented from becoming a false claim.

## Interpreting the battery as a whole

The chain closes every alternative we could construct: generic loser
reversal (pre20 control), drift/size (date FE + baseline), characteristics
in disguise (demeaning), flow-surprise magnitude (INNOV), overlap
(non-overlap stronger), horizon artifacts (builds & persists), informed
reversal (PIN split). What remains is the claim itself: **participation
concentration separates transitory from permanent FII price impact.**
