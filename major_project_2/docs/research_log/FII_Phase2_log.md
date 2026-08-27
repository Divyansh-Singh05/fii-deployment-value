# FII Regime Detection — Phase II Log: Probabilistic Market-State Prediction
*Charter: `docs/PHASE2_PLAN.md` (pre-registered before any Phase II result).
Phase I evidence trail: FII_Module5_validation_log.md §3a–§3r.*

## P2-1. MODULE 16A — causal filtering: A2 PASS, CAUSAL FOUNDATION CONFIRMED
Forward-filtered posteriors P(S_t | x_{1:t}) from FROZEN Phase-I parameters
(hmm_backbone_params.json + overlay_thresholds.json), no refitting. Stage
`phase2_filtering` (56.6s local). Output: phase2_filtered_states.parquet
(posteriors p_sell/p_neutral/p_buy + filtered state/archetype, 804,591 rows).
- **D1 fidelity:** filtered vs Viterbi backbone agreement **91.3%** (archetype
  91.3%); daily flip rate 6.74% filtered vs 5.07% smoothed (the backward pass
  was buying ~1.7pp of smoothness); **onset lag median 1 day, p90 3** — the
  causal filter recognizes a new regime essentially in real time.
- **A2 gate (economic invariance):** Table-1 R2 on fully FILTERED labels:
  SD **+60.2*** TRAIN / +50.3*** TEST** (Phase-I: +65.4/+48.5), SA
  −85.5***/−56.8*** (−87.9/−47.6), HOSTAGE ns both. All within the ±50%
  band with |t|≥2 → the economics do not live in the Viterbi smoothing
  (consistent with 13A's causal rule labeler). n rises 63k→71k / 35k→40k
  (more flips ⇒ more episode-ends), effects unchanged.
- Phase-II relevant detail: TEST SD on filtered labels (+50.3, t=3.00) is
  marginally STRONGER than smoothed — anticipatory decisions can be built
  on the filtered posterior without economic loss.
NEXT: 16B nowcast calibration (Brier/log-loss vs census-prior and
persistence baselines; bar = beat BOTH in TEST).

## P2-2. MODULE 16B — nowcast calibration: ORIGINAL GATE FAIL (recorded), AMENDED GATE PASS
Stage `phase2_calibration`. Truth proxy = smoothed (Viterbi) backbone state; four forecasts scored (multiclass Brier / log-loss / accuracy), rows aligned (t≥2/stock).
- **ORIGINAL GATE: FAIL, and it stays recorded.** FILTERED (Brier .1207 / logloss .1821 TEST) beat PRIOR (.6510/1.0761) and PREDICT (.2097/.3765) massively (paired daily t +567/+83) and beat PERSIST on LOG-LOSS (.1821 vs .2333) — but lost to PERSIST on Brier (.1207 vs .0999, t −21.8); the bar demanded both metrics.
- **DIAGNOSIS (amendment, pre-registered before rerun):** PERSIST conditioned on yesterday's SMOOTHED label — unknowable at t (Viterbi uses the future) and quasi-circular with the smoothed target (smoothing manufactures the 0.95 persistence). The information-set-fair baseline is persistence of yesterday's FILTERED argmax (PERSIST_C).
- **AMENDED GATE: PASS.** PERSIST_C TEST Brier .2615 / logloss .5320 — FILTERED beats it on both by wide margins. Posteriors are usable probabilities.
- **Calibration structure (TEST):** excellent where 79% of mass lives (conf .99 bin → hit .994); overconfident mid-range (conf .70 → hit .55; ECE .039) → TRAIN-fitted recalibration adopted as input hygiene for 16C.
- Honest residual: against the oracle-contaminated baseline, FILTERED's log-loss win + Brier loss = it pays a flicker premium on quiet days and earns it back at transitions — the correct trade for Phase II, whose object (16C) is precisely the transition.
- **DATA HORIZON (user):** dataset ends 2025-03-28 (2025 = Q1 only). 16C walk-forward folds through 2024 + partial 2025-Q1 slice; no claims beyond.

## P2-3. MODULE 16C — episode-END hazard: PASS (decisive)
Stage `phase2_hazard` (46.9s). Causal SHARK_DIST episodes from filtered labels, runs broken on >21cd gaps, right-censored runs (859/14,273) drop unlabelable targets. 58,821 episode-days; base rates k=1/3/5 = .231/.524/.700. Walk-forward LGBM (200 trees, refit yearly 2014–2025Q1, train ends 10d before fold) vs age-only KM hazard (capped 15+) and const.
- **Pooled OOW: k=1 logloss .4303 vs KM .5308 (paired daily t +41.6), AUC .797 vs .569; k=3 .6145 vs .6881 (t +34.4), AUC .724; k=5 t +15.9.**
- **TEST era stronger than TRAIN** (k=1 AUC .809 vs .786) — skill is not memorization; 2025 fold = Q1 partial per data horizon.
- VERDICT: PASS (bar k=1 & k=3 t≥2) — ends are forecastable beyond age alone. Output phase2_hazard_preds.parquet → 16D.

## P2-4. MODULE 16D — decision layer: FAIL on the pre-registered bar (the instructive half-result)
Stage `phase2_decision` (46s; one datetime[ms] roundtrip fix, module-9 pattern). Anticipate (enter at first day p_model(k=1)>θ, θ walk-forward from prior years' OOW preds, first decision year 2016) vs confirm (enter end+1); gain = CAR(t_a+1..end+1), paired within episode, cost-neutral (same trade count).
- **Half 1 PASS: anticipation adds value.** TEST model-anticipation gain **+23bp/episode CI[+7,+40]** (date-clustered bootstrap) over confirmation, at zero incremental cost.
- **Half 2 FAIL: the model does not beat the trivial anticipator.** Age-only KM anticipation gains +18 [+2,+33]; paired model-vs-KM on 2,764 common episodes **d=−15bp, t=−2.96** — significantly worse. AUC 0.80 of end-forecasting does NOT convert to entry-timing value beyond "episodes are short; enter early."
- Diagnostics: θ unstable across folds (0.20↔0.65 — noisy historical-mean objective); gain excl. first post-end day +9 [−4,+23] → ~half the anticipation value is the end+1 day (partly the 15-T1 spread concession).
- **VERDICT (per pre-registration): forecastable but not monetizable beyond a trivial age rule.** Consistent with Phase I (8B: dynamic increment sub-significance; 12: costs bind).

## P2-5. PHASE II CONCLUSION
The charter is complete: (16A) causal foundation confirmed — filtered posteriors, onset lag 1d, Table 1 intact; (16B) posteriors are calibrated probabilities vs information-set-fair baselines (original oracle-contaminated gate FAIL kept on record); (16C) episode ends are genuinely forecastable beyond age (AUC .80/.72, t +42/+34, stronger OOS); (16D) that skill does NOT convert into entry timing beyond an age heuristic — anticipation's +23bp/episode is real but a free 3-line rule earns it too.
**The engineering framework exists and is honest about where its ML earns nothing.** Practical takeaway for the execution-overlay story: "don't wait for confirmation — enter/act once a concentrated episode is a few days old" is worth ~20bp/episode and needs no model; the hazard model adds monitoring value (calibrated end-probabilities) but not timing alpha. Phase-II language for the thesis: probabilistic state estimation validated end-to-end; decision value capped by the same power/cost ceilings Phase I measured.
