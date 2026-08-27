> [!CAUTION]
> **STALE — DO NOT USE (flagged 2026-08-23).**
>
> This document predates the August 2026 engineering audit and states the
> retracted concentration-axis headline (`entity_hhi_raw` measured each
> entity's own-book Herfindahl, not within-stock-day participation shares;
> the two correlate r = −0.2460). Its economic claims about SHARK_DIST are
> **withdrawn**, and its exhibit numbers were generated from pre-audit
> tables now preserved under `outputs/tables.PREAUDIT_20260712/`.
>
> Current results: `docs/paper/FII_final_results_one_pager.md`.
> Full ledger: `docs/AUDIT_RETRACTIONS.md` (R1–R6).

# Results Section — Slide-Ready Content

*Organized as a sequence of slides. Every number is sourced from a specific
log in this repo (cited under each table) — nothing here is paraphrased from
memory. Use as many or as few slides as your time allows; Slide 1 alone works
as a single "Results" slide if you need just one.*

---

## Slide 1 — Results Overview (the one-slide version)

**One line:** Concentrated FII flow predicts a reversal; dispersed FII
selling does not — confirmed across ten independent tests, out-of-sample,
but not itself tradeable after realistic costs.

| Test | What it checks | Verdict |
|---|---|---|
| Event study (CAR) | Does price reverse after the episode ends? | SHARK_DIST/ACC reverse; HOSTAGE does not |
| Panel regression (Table 1) | Effect size with stock+date FE, clustered SEs | SHARK_DIST +48 to +65bp, SHARK_ACC −48 to −88bp, both eras significant |
| Robustness | Non-overlap, horizons, bounce-correction | Survives all three |
| Mechanism (block deals + internal arc) | Is it really a liquidity shock? | Confirmed: pressure + volume climax + reversal |
| Flow-surprise control (INNOV) | Is it just flow-surprise size? | No — survives the control |
| ML challenger (GBT) | Does a flexible model beat the regime baseline? | Yes, but... |
| Characteristics check (demeaning) | ...is that edge real dynamics? | No — falls short of the pre-registered bar |
| Independent check (PIN) | Does an unrelated estimator agree? | Yes — 2.3–3.0× higher informed-trading loading on dispersed selling |
| Referee test (rule vs. HMM) | Is the HMM even necessary? | No — a simple rule reproduces Table 1 |
| Tail extension | Does it hold in illiquid names? | Direction holds; doesn't clear friction |
| Backtest | Is it tradeable? | Breakevens 0–8bp vs. ~15bp realistic cost — no |
| Phase II (causal, hazard) | Does it survive going fully real-time? | Yes — causal labels reproduce Table 1; episode-end is forecastable |
| Phase II (decision) | Is the forecast monetizable? | Not yet — doesn't beat a naive age-only timing rule |

---

## Slide 2 — The Headline Result: Panel Regression ("Table 1")

**Specification:** post-episode 20-day market-adjusted abnormal return,
regressed on archetype dummies, with stock and calendar-date fixed effects,
two-way clustered standard errors (stock × month).

| Archetype | TRAIN coefficient | TRAIN t-stat | TEST coefficient | TEST t-stat |
|---|---|---|---|---|
| SHARK_DIST (concentrated sell → reversal) | **+65.4bp** | +5.35 *** | **+48.5bp** | +2.95 *** |
| SHARK_ACC (concentrated buy → give-back) | **−87.9bp** | −6.23 *** | **−47.6bp** | −2.84 *** |
| HOSTAGE (dispersed sell → no reversal) | −3.6bp | −0.38 (ns) | −8.1bp | −0.48 (ns) |
| ROBOT (placebo) | −7.7bp | −0.65 (ns) | −38.4bp | −3.44 *** (see note) |

*Source: `outputs/validation/backbone_ablation.log` (identical Table-1 spec
on HMM labels; also reproduced in `panel_regression.log`). Note on ROBOT-TEST:
investigated separately (Module 7B-A) — traced to what regime a "Robot"
episode transitions into next, not a leak.*

**Read:** concentrated selling reverses; concentrated buying gives back;
dispersed selling does neither — consistent with concentrated flow being
temporary/liquidity-driven and dispersed flow being permanent/information-driven.

---

## Slide 3 — Event Study: Where the Result First Surfaced (and Inverted the Hypothesis)

| Archetype | Original prediction | TRAIN excess CAR20 | TEST excess CAR20 |
|---|---|---|---|
| HOSTAGE | **Reversal** (the expected flagship) | −15bp, p=0.23 | +10bp, p=0.60 |
| SHARK_DIST | Continued decline | **+68bp, p=0.000** | **+33bp, p=0.042** |
| SHARK_ACC | Continued rise | **−50bp, p=0.000** | −13bp (ns) |
| ROBOT (placebo) | ≈0 | ≈0 | ≈0 |

*Source: `outputs/validation/event_study.log`; thesis §10.3.*

**Read:** the placebo behaving as expected proved the method was sound — which
is what made the flipped hypothesis rows credible rather than dismissible.
This single table is what reframed the entire project: concentration marks
temporary pressure, not information.

---

## Slide 4 — Robustness and Self-Audit

| Check | Method | Result |
|---|---|---|
| Bid-ask bounce | Delay measurement window 2 extra days (t+3..t+22) | SHARK_DIST TRAIN +58.0bp (t=4.75), TEST +36.7bp (t=2.35) — survives, bounce is only 11-24% of the effect |
| Formal SD-vs-Hostage contrast | Direct statistical test of the difference, not eyeballing | Significant in both eras (TRAIN p=0.000, TEST p=0.016) |
| Non-overlapping episodes | De-overlap post-windows, rerun regression | Same sign, holds |
| Horizons | CAR10/20/30/60 | Reversal builds 10→20, persists through 60 |
| Public-factor control | Add a public volume-conditioned-reversal dummy | Composition beats the public proxy head-to-head |

*Source: `outputs/validation/skeptic_tests.log` (Module 15).*

---

## Slide 5 — Independent Corroboration: the PIN Model

An entirely separate estimator (Probability of Informed Trading, Easley-O'Hara,
maximum likelihood per stock-year) — built purely from FII **order counts**,
never sees price, return, or archetype label.

| Regressor (share of stock-year days) | TRAIN coefficient | TRAIN t | TEST coefficient | TEST t |
|---|---|---|---|---|
| HOSTAGE share (dispersed sell) | **0.2306** | 7.91 *** | **0.1488** | 4.47 *** |
| SHARK_DIST share (concentrated sell) | 0.0766 | 2.87 *** | 0.0645 | 2.54 ** |
| **Ratio (Hostage : Shark-Dist loading)** | **3.01×** | | **2.31×** | |

*Source: `outputs/validation/pin_model.log`.*

**Read:** dispersed selling loads 2.3–3.0× more heavily on independently-estimated
informed-trading probability — a completely different model, on the same raw
counts, landing on the same permanent/transitory story.

---

## Slide 6 — The Machine-Learning Challenger: Did the HMM Leave Signal on the Table?

| Model | TEST-era daily IC | % positive days | Non-overlap t-stat | Quintile spread (Q5−Q1, 20d) |
|---|---|---|---|---|
| Regime baseline (archetype mean) | +0.0117 | 59% | +1.68 | — |
| LightGBM, all 10 features | **+0.0277** | 65% | +1.97 | **+68.9bp, t=+2.02** |
| LightGBM, TRAIN (overfit gauge) | +0.2187 | 99% | +29.94 | — (memorization, expected) |
| LightGBM, de-meaned (dynamics only) | +0.0161 | — | 1.48 (< 2 bar) | +53.4bp (bar not met) |

*Source: `outputs/validation/gbt_challenger.log`, `demeaning_check.log`.*

**Read:** the flexible model does beat the regime baseline — but once static
stock characteristics are removed, the pure-dynamics edge falls short of its
own pre-registered bar. Documented, deliberate decision **not** to escalate
to a sequence model (LSTM).

---

## Slide 7 — Referee-Anticipation Tests

| Question a reviewer would ask | Test | Result |
|---|---|---|
| Is the HMM even necessary? | Census-matched rule backbone vs. HMM, identical regression | RULE TRAIN SD +71.6(t=6.21) / SA −78.0(t=−5.54); TEST SD +47.4(t=3.15) / SA −52.3(t=−3.08) — **all within band of the HMM numbers** |
| Is composition just flow-magnitude in disguise? | Add flow-magnitude/imbalance controls | SHARK_DIST moves only −11 to −13%, stays significant |
| Is this passive index-rebalancing? | Exclude episodes near index-review dates | Result stands |

*Source: `outputs/validation/backbone_ablation.log`;
`docs/research_log/FII_Module5_validation_log.md` §13B, §13C.*

**Verdict printed by the code itself:** *"V1 — HMM NOT NECESSARY (contribution
= the composition measure; reframe per plan)."*

---

## Slide 8 — Extending to the Illiquid Tail

| | Liquid universe (main result) | Illiquid tail (1,618 names) |
|---|---|---|
| SHARK_DIST TEST excess CAR20 | +33-49bp (event study / regression) | +36bp, CI [−51,+128], p=0.374 |
| Friction bar (round-trip cost) | ~15bp assumed | **98bp median Roll spread** |
| Verdict | Real, survives costs at model scale | **NO TAIL EFFECT** — direction holds, doesn't clear friction |

*Source: `outputs/validation/tail_economics.log` (Module 14C).*

**Read:** same directional story further down the liquidity spectrum, but
transaction friction there is roughly 6-7× higher than the effect itself —
real but currently unharvestable.

---

## Slide 9 — Economic Significance: Does It Trade?

| Book | Era | Base gross Sharpe | HMM gross Sharpe | Breakeven cost (one-way) | Net Sharpe @ 15bp |
|---|---|---|---|---|---|
| S1H (reversal-gated) | TRAIN | +0.11 | +0.41 | 3.5bp | −1.34 |
| S1H | TEST | −0.74 | +0.00 | 0.0bp | −2.28 |
| S2H (flow-filtered) | TRAIN | −0.59 | −0.46 | −2.2bp | −3.58 |
| S2H | TEST | +1.15 | +1.43 | 4.3bp | −3.56 |
| S3H (regime book) | TRAIN | +1.46 | +0.69 | 2.6bp | −3.22 |
| S3H | TEST | +1.44 | +0.90 | 3.0bp | −3.62 |

*Source: `outputs/validation/bt_gross_diagnosis.log` (Module 12D).*

**Read:** breakevens cluster around **0–8bp**, well under a realistic **15bp**
one-way trading cost. Signal is real in places (S2H TEST: "SIGNAL REAL, COSTS
BIND") but no book survives net of realistic costs as a standalone strategy.

---

## Slide 10 — Phase II: From Retrospective Labels to a Real-Time Predictor

| Stage | Question | Result |
|---|---|---|
| 16A Causal filtering | Does the economics survive fully causal (no-future-information) labels? | **PASS** — Table 1 reproduced on causal labels |
| 16B Calibration | Are the filtered posteriors real probabilities? | Beats prior and causal persistence baselines (amended, transparently, after an initial baseline design flaw was caught) |
| 16C Hazard model | Can we forecast *when* an episode ends? | **PASS** — beats age-only KM baseline: k=1 paired t=+41.6, k=3 paired t=+34.4, AUC 0.80 vs. KM's 0.57 |
| 16D Decision | Does acting early monetize the forecast? | TEST gain +23bp CI[+7,+40] (real, CI>0) but loses to a naive KM-anticipator, paired t=−2.96 — **forecastable, not yet monetizable** |

*Source: `outputs/logs/20260713_015357_phase2_hazard.log`,
`..._015735_phase2_decision.log`.*

---

## Slide 11 — Final Summary Metrics (one table for the appendix/backup slide)

| Metric | Value |
|---|---|
| Data span | Apr 2011 – Mar 2025 |
| Train / test split | ≤2021-04-30 (509,185 rows, 572 stocks) / ≥2021-07-01 (295,773 rows, 830 stocks) |
| Headline effect (SHARK_DIST reversal) | +48.5 to +65.4bp, both eras significant, t≥2.95 |
| Independent corroboration (PIN ratio) | 2.3–3.0× |
| GBT challenger edge (TEST IC gap) | +0.016 (gap), falls to non-significant once de-meaned |
| Backtest breakeven vs. realistic cost | 0–8bp vs. ~15bp |
| Value on a ₹100cr order timed around the reversal | ~₹37–49 lakh |
| Hazard model AUC vs. age-only baseline | 0.80 vs. 0.57 |
| Tail-universe friction bar vs. effect | 98bp vs. +36bp — doesn't clear |

---

*All figures cross-checked against: `event_study.log`, `panel_regression.log`,
`backbone_ablation.log`, `skeptic_tests.log`, `pin_model.log`,
`gbt_challenger.log`, `demeaning_check.log`, `tail_economics.log`,
`bt_gross_diagnosis.log`, `phase2_hazard.log`, `phase2_decision.log` — all in
`outputs/validation/` or `outputs/logs/`.*
