# Phase II Charter — Probabilistic Market-State Prediction
### Pre-registered design, bars, and baselines (written before any Phase II result exists)

**Standing on Phase I.** Phase I (Modules 1–15) validated the latent
execution regimes as genuine market structure: statistically stable OOS,
economically meaningful (transitory/permanent decomposition), model-class
independent (13A), incremental over public data and conventional flow
(15-T3), with honest boundaries (bounce component 11–24%, no tail effect,
liquid-universe scope). Phase II treats those states as **validated state
variables** and poses a new problem: *using only information through time
t, estimate the current latent regime and forecast its evolution, then
convert forecasts into decisions — under walk-forward discipline.*

**What Phase I already tells Phase II (constraints, not hopes):**
- The dynamic increment beyond regimes failed its bar once (8B, t=1.48).
- Nothing cleared 15 bps at daily rebalance (12); the tail is dead (14).
- Therefore **standalone net P&L is NOT a Phase II success bar** — it is
  reported, but the pre-registered value criterion is execution-overlay
  economics (bp saved per episode for an agent already trading).
- Phase I labels are smoothed (full-sequence Viterbi ⇒ the state at t sees
  data after t). 13A de-risks this — a fully causal rule labeler
  reproduces Table 1 — but Phase II must run on **filtered** posteriors.
- The persistence baseline is brutal (self-transition ≈ 0.95): "predict no
  change" is the mandatory comparator for every forecast.

---

## Stages (Module 16 series; one verifiable step each)

### 16A — Causal filtering (`phase2_filtering`)
Forward-filtered posteriors P(S_t | x_{1:t}) from the **frozen** Phase-I
backbone parameters (`outputs/trained_models/hmm_backbone_params.json`)
+ frozen overlay thresholds. No refitting.
- Diagnostics: filtered-vs-smoothed backbone agreement, flip rate, onset
  lag at regime starts.
- **Pre-registered gate A2 (economic invariance):** Table-1 R2 on
  fully-filtered archetype labels keeps SHARK_DIST and SHARK_ACC at
  |t| ≥ 2 both eras within ±50% of Phase-I coefficients →
  "CAUSAL FOUNDATION CONFIRMED"; else Phase II stops and reports.
- Output: `phase2_filtered_states.parquet` (posteriors + filtered labels).

### 16B — Nowcast calibration (`phase2_calibration`)
Are the filtered posteriors *probabilities* or just scores?
- Reliability curves, Brier score and log-loss for the 3-state nowcast,
  vs two baselines: census prior and persistence.
- **Bar:** filtered nowcast beats BOTH baselines on Brier/log-loss in the
  TEST era. (If it can't nowcast, it can't forecast.)

### 16C — The hazard model (`phase2_hazard`) — the centerpiece
The economically valuable forecast is the **episode END** (the reversal
starts when concentrated flow stops). Discrete-time hazard:
P(episode ends within k ∈ {1,3,5} days | episode age, within-episode
feature trajectory, volume, posterior margin).
- Estimator: gradient-boosted or logistic discrete-time hazard.
- **Walk-forward protocol:** expanding window, yearly refits; ALL
  parameters/thresholds learned in-window, frozen, then applied to the
  next year. May–Jun 2021 embargo preserved.
- **Bar:** beats the age-only baseline hazard (Kaplan–Meier by episode
  age) on out-of-window log-loss with paired t ≥ 2. Persistence-style
  "never ends tomorrow" also reported.

### 16D — Decision layer + backtest (`phase2_decision`)
Convert the hazard into entries: anticipate the end (enter when hazard >
threshold learned in-window) vs Phase I's confirmed-end entry (+1 day).
- Engine: the certified Module-12 backtester unchanged (gates re-run).
- **Primary metric:** bp/episode gained by anticipation over
  confirmation, gross and at the 0/5/10/15/30 cost grid; plus the
  execution-overlay accounting (cost saved for an obligated trader).
- Net Sharpe reported; **not a bar** (see constraints above).

## Protocol rules (inherited from Phase I, non-negotiable)
Pre-registered verdicts in every script header before results; one claim
per stage; persistence baseline everywhere; deterministic seeds; no
post-hoc threshold tuning (bugs fixed openly, cuts never); 2024–25
coverage confound flagged on every TEST metric; failures go in the
ledger; the thesis (`docs/paper/FII_thesis.md`) is the canonical record.

## What Phase II is NOT
Not a search for a trading strategy (Phase I priced that); not an LSTM
revival (8B's bar stands until new data); not a re-litigation of Phase I
claims. It is the engineering completion: validated states → causal
estimation → calibrated forecasting → decision value, measured honestly.

---

## Outcome (charter completed 2026-07-13 — full record: research_log/FII_Phase2_log.md)

16A PASS (causal foundation; Table 1 intact on filtered labels).
16B original gate FAIL recorded (oracle-contaminated baseline), amended
causal-baseline gate PASS; mid-range overconfidence measured.
16C PASS decisively (end-forecast AUC .797/.809 OOS; t +42/+34).
16D FAIL on its bar: anticipation adds +23bp/episode (CI>0, cost-neutral)
but does not beat an age-only rule (paired t = −2.96).

**Phase II verdict: the states are causally estimable, calibrated, and
their transitions forecastable — and the decision value of the forecast
is capped at what a three-line age heuristic already captures. Reported
as designed: the bars decided, not the narrative.**
