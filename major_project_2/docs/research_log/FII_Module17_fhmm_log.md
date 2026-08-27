# Module 17 — Factorial HMM Challenger (validation log)
### Does a learned concentration channel replace the frozen overlay thresholds?
*Run date: 2026-07-15. Code: `src/fii/models/fhmm_stages/` (17A/17B/17C) +
`src/fii/models/factorial_hmm.py`. Fully separate from the naive HMM chain
(`models/hmm_stages/` untouched). Logs: `outputs/logs/20260715_*_fhmm_*.log`.*

---

## 0. Charter (pre-registered before any result existed)

Module 2 established that a **flat** Gaussian HMM at any k never forms
concentration states: the persistent direction axis captures the likelihood
mass, and the hybrid design (3-state backbone + frozen TRAIN-quantile
overlay thresholds) was adopted as a consequence. Module 17 tests the
structural fix: a **factorial HMM** (Ghahramani & Jordan, 1997) with two
independent chains — direction (k=3) and concentration (k=3) — emitting
additively, estimated by exact EM on the 9-state product space via a
constrained `hmmlearn.GaussianHMM` (only the M-step is constrained;
independent pure-numpy forward pass as implementation cross-check).

Protocol identical to 3A: same four features (same 5d-smoothed re-ranked
entity features), frozen split (train ≤ 2021-04-30, May–Jun 2021 masked,
test ≥ 2021-07-01), same MIN_SEQ/FIT_CAP random-block subsample, same seed,
frozen decode of both eras. Archetypes decoded **end-to-end — no
thresholds anywhere**.

Pre-registered gates: G1 EM monotone; G2 exactness (rel < 1e-8);
G3 concentration channel earns states (census ≥ 5% each; F_entity_s
loading spread ≥ 0.20); G4 OOS signature drift (chain D ≤ 0.15 on
F_persist; chain C ≤ 0.25 on F_entity_s). 17B: B1 join ≥ 99%; B2 backbone
kappa ≥ 0.60 both eras; B3 mean archetype runs within 2× of naive.
17C verdicts: V1 = FHMM SHARK_DIST **and** SHARK_ACC within band
(|t| ≥ 2 and coef within ±50% of naive) in both eras **and** HOSTAGE null
holds → thresholds replaced; V2 = SD within band in ≥ 1 era; V3 = SD out
of band in both eras → thresholds stand.

## 1. 17A — training and structure (all gates PASS)

Fit: 177,601 rows (random blocks ≤ 400/stock from 509,185 TRAIN
stock-days, 572 stocks). **All 5 seeded inits converged to the same
optimum** (loglik −819,327) — the solution is not an initialization
artifact. G1 PASS (0 dips); G2 PASS (hmmlearn vs independent numpy
forward: rel 5.9e-16).

**G3 — the headline structural result: the concentration channel FORMED.**
The Module-2 starvation finding does **not** extend to factorial form.
Given its own transition matrix and additive emission contribution, the
concentration axis earned three well-populated states (TRAIN census
31–36% each, spread 1.49 probit units ≫ 0.20 bar):

| chain C state | F_persist | F_block | F_entity_s | F_entity_buy_s | dwell |
|---|---|---|---|---|---|
| DISPERSED | +0.21 | −0.08 | **−0.80** | **−0.67** | 14.6d |
| CONC_SELL | −0.41 | +0.02 | **+0.69** | −0.35 | 12.3d |
| CONC_BUY | +0.16 | +0.06 | +0.19 | **+1.02** | 13.7d |

Chain D reproduces the familiar backbone (SELL/NEUTRAL/BUY by F_persist,
dwell 15–16d). G4 PASS: frozen-decode drift TRAIN→TEST 0.063 (chain D)
and 0.080 (chain C) — both chains replicate out of sample.

**Labeling correction (recorded honestly).** The first 17A run named
chain-C states low/MID/high on the *mean* entity loading. The fitted
means show the chain is **side-specific** (one state concentrates
selling, another buying), which that rule mislabels — it called the
sell-concentrated state "MID" and mapped SHARK_DIST to the
buy-concentrated state. Corrected to side-aware naming (DISPERSED = min
mean loading; CONC_SELL = max F_entity_s; CONC_BUY = remaining, verified
max F_entity_buy_s) **after seeing the 17A signatures but before any 17C
economics were run**. The fit itself is unchanged (identical loglik);
only names and the archetype map moved. Both logs retained.

*Unplanned structural finding:* the concentration factor is
**two-sided** — dispersal is one state across both sides, but
concentration splits by side. The 2-D "direction × concentration" regime
space is really closer to 2.5-D.

## 2. 17B — descriptives and agreement (B1 PASS, B2 PASS, B3 FAIL)

- B1 join integrity: 100% (same universe as calibrated states). PASS.
- B2 backbone agreement: kappa 0.755 (TRAIN) / 0.760 (TEST) vs the naive
  backbone — the two models identify the same direction axis. PASS.
- Archetype agreement (reported, not gated): overall 70.2%, kappa 0.557.
  Overlap with naive labels: SHARK_DIST 40%, SHARK_ACC 67%, HOSTAGE 71%,
  ROBOT 89%. Expected direction of disagreement: thresholds tag
  **tail days**, latent states tag **persistent spells**.
- **B3 FAIL (pre-registered band 0.5–2.0×):** FHMM archetype runs are
  2.1–2.6× longer (SHARK_DIST 10.1d vs 4.5d; SHARK_ACC 9.7 vs 4.5;
  HOSTAGE 9.3 vs 3.6). Consequence: ~40% fewer episode-end anchors in
  17C (TRAIN n 33,296 vs 63,259) — less power at the END anchor.

## 3. 17C — Table-1 economics (C0 PASS; verdict V3)

C0 decode consistency states_v3 vs 17A: PASS on every cell (shares match
to 0.01pp; direct-join agreement 100.0%). The naive-HMM columns reproduce
the published Table 1 exactly — the machinery is 13A's, verbatim.

Post-episode 20d abnormal return (bp), R2 spec, PanelOLS stock+date FE,
SE clustered stock × month:

| cell | naive HMM | FHMM (end-to-end) | band check |
|---|---|---|---|
| TRAIN SHARK_DIST | +65.4 (t +5.35) | **+115.7 (t +5.08)** | OUT (coef **larger** by 77%) |
| TEST SHARK_DIST | +48.5 (t +2.95) | +37.9 (t +1.30) | OUT (underpowered) |
| TRAIN SHARK_ACC | −87.9 (t −6.23) | −101.7 (t −4.85) | WITHIN |
| TEST SHARK_ACC | −47.6 (t −2.84) | **−109.7 (t −3.78)** | OUT (coef larger) |
| TRAIN HOSTAGE | −3.6 (t −0.38) | **+47.9 (t +2.31)** | null VIOLATED |
| TEST HOSTAGE | −8.1 (t −0.48) | +3.0 (t +0.13) | null holds |

**Pre-registered verdict: V3 — THRESHOLDS STAND.** SHARK_DIST is out of
band in both eras, so by the rule written before the numbers existed, the
factorial channel does not replace the frozen thresholds.

## 4. Honest interpretation (what V3 does and does not mean)

1. **The representation problem is solved; the identification problem is
   not.** The factorial structure makes concentration *learnable* (G3/G4
   pass decisively, refining Module 2: starvation is a property of the
   flat parameterization, not of the data). But its states carve
   ~30%-census spells, while the published effect lives in the
   **concentration tail** — exactly where the quantile thresholds sit
   and exactly what the Module-7B dose-response already showed ("weaker
   away from the tail").
2. **The buy side survives end-to-end.** FHMM SHARK_ACC is significant
   and right-signed in both eras (−101.7***/−109.7***) with no
   threshold anywhere — the strongest pro-factorial cell.
3. **The sell side is right-signed but underpowered OOS** (+37.9,
   t = 1.30; coefficient within 22% of naive's +48.5). Longer episodes →
   ~40% fewer END anchors (B3) is a plausible driver; a power-matched
   comparison is future work, and was NOT pre-registered, so it is not
   claimed.
4. **TRAIN-era HOSTAGE null violation (+47.9**)** is the clearest
   substantive failure: the FHMM's broader HOSTAGE (9.9% census vs the
   naive tail) sweeps in reversal-carrying days in-sample. The
   permanent-impact null is a tail property too.
5. **Paper placement:** Module 17 strengthens the paper as a structural
   robustness exhibit — "the concentration axis is a genuine latent
   factor (factorial HMM recovers it end-to-end, OOS-stable), but the
   transitory/permanent economics are a property of its tail, which the
   calibrated thresholds isolate and the latent states do not." This is
   the exact shape of 13A's conclusion (contribution = the composition
   measure, model class secondary), now established from the generative
   side as well.

## 5. Phase-II battery on the FHMM (17D–17G, run 2026-07-15;
## mirrors 16A–16D stage-for-stage)

**17D — causal filtering (16A protocol): A2F PASS.** Forward-filtered
product posteriors from frozen params, chain marginals, side-aware
archetypes, no thresholds, no look-ahead. Fidelity: chain D agreement
with smoothed 89.3%, chain C 84.0%, archetype 85.1%; onset lag median
1 day (p90 4). Filtered Table-1 vs the smoothed-FHMM references:
SHARK_ACC −84.5*** / −59.4*** (both eras within band), SHARK_DIST
TRAIN +68.5*** (within band). Notably, causal filtering *repairs* the
two smoothed-FHMM defects: TRAIN HOSTAGE returns to null (+14.7,
t 0.95, vs +47.9** smoothed) and TRAIN SHARK_DIST (+68.5) lands next
to the naive +65.4 instead of the smoothed +115.7 — the Viterbi
oversmoothing, not the factorial structure, drove both anomalies.

**17E — nowcast calibration (16B protocol, amended causal baseline
inherited): BOTH gates PASS.** TEST era, truth = smoothed state.
Chain D: Brier 0.160 vs 0.271 (causal persistence) / 0.625 (prior),
accuracy 0.892, ECE 0.052. Chain C: Brier 0.224 vs 0.355 / 0.664,
accuracy 0.842, ECE 0.056. The chain-C result is the novel object:
a **calibrated, causal, daily probabilistic concentration nowcast**,
which the naive HMM cannot produce at all (its concentration axis is
a threshold, not a probability).

**17F — episode-END hazard (16C protocol + p_csell feature, declared):
PASS, stronger than the naive 16C.** 56,577 FHMM SHARK_DIST
episode-days, 9,225 runs. Pooled out-of-window: k=1 AUC 0.872 (TEST
0.869), k=3 AUC 0.787 (TEST 0.791), k=5 AUC 0.735; paired t vs
age-only KM +46.2 / +48.6 / +37.2. Naive 16C reference: AUC
.797/.809, t +42/+34. The richer causal state (concentration
posterior) plus longer episodes make FHMM episode ends MORE
forecastable.

**17G — decision layer (16D protocol): FAIL on the pre-registered
bar.** Walk-forward theta always selected 0.15. TEST era: model
anticipation gain +1 bp/episode, CI [−21, +21]; KM anticipator +18
[−9, +44]; paired model-vs-KM on 2,158 common episodes d = −17 bp,
t = −2.00. Same verdict shape as 16D 
(naive: +23 bp CI>0 but lost to KM at t −2.96), and quantitatively
WEAKER: the FHMM's long latent episodes make early entry expensive —
anticipating at low theta enters days of remaining decline that the
naive model's short threshold episodes did not carry.

**Phase-II verdict for the FHMM, in the charter's language:** the
factorial states are causally estimable (17D), calibrated on both
chains (17E), and their transitions are MORE forecastable than the
naive model's (17F, AUC .87 vs .80) — and the decision value of that
better forecast is still capped by the age heuristic (17G). The 16D
conclusion — "forecastable but not monetizable over confirmation" —
is robust to the model class, which strengthens it from an anecdote
about one model into a statement about the phenomenon: the
episode-end edge, under any latent representation tried so far, is
duration information plus a margin too thin to trade ahead of
confirmation.

## 6. Artifacts

- `data/ISIN_MAPPING/stockday_states_fhmm.parquet` (804,958 × 10;
  also copied to `outputs/predictions/`)
- `outputs/trained_models/fhmm_params.json` (frozen chain parameters +
  gate results), `outputs/trained_models/factorial_hmm.txt`
- `data/VALIDATION_DATA/fhmm_filtered_states.parquet` (chain
  posteriors + filtered/smoothed labels),
  `data/VALIDATION_DATA/fhmm_hazard_preds.parquet`
- Stage logs: `20260715_002127`/`20260715_004151` (17A, first run and
  side-aware rerun), `20260715_011743` (17B), `20260715_011814` (17C),
  `20260715_023604` (17D), `20260715_030354` (17E),
  `20260715_033327` (17F), `20260715_033518` (17G)
- Run: `python pipeline.py --model factorial_hmm` (17A→17B→17C) and
  stages `fhmm_filtering` → `fhmm_calibration` → `fhmm_hazard` →
  `fhmm_decision` for the Phase-II battery; excluded from `--all`
  by design.
