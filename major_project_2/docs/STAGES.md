# Stage Manifest

Every pipeline stage: what it does, what it reads, what it writes, and the
gate that decides PASS. Stages run in this order under
`python pipeline.py --all` (audit phase excluded; run with
`--phase audit`). Full narrative: `research_log/`.

Data roots: `V/` = `data/VALIDATION_DATA`, `I/` = `data/ISIN_MAPPING`.

## data_prep

| Stage | Purpose | Inputs | Outputs | Gate |
|---|---|---|---|---|
| `price_panel` | repair & assemble the daily price panel | `V/bhavcopy_parquets/*`, macro parquets | `V/returns_panel.parquet` | G1 rows/coverage, G2 model-join |
| `ca_factors` | parse + tape-verify split/bonus factors | `V/nse_corporate_actions.csv`, `V/ca_raw.parquet`, panel | `V/ca_adjustment_factors.parquet` | ≥95% tape-confirmed (obs. ratio band ×/÷1.3) |
| `apply_adjustment` | CA-adjust returns with application guard | panel + factors | `V/returns_panel_v2.parquet` | Gate A: ex-day median \|ret\| < 0.05 (got 0.038); Gate B spot-checks |

## features

| Stage | Purpose | Inputs | Outputs | Gate |
|---|---|---|---|---|
| `feature_store` | 10 probit-ranked flow features per stock-day | `I/2011..2025.parquet` | `I/stockday_features_v2.parquet` | complete-case & coverage accounting |

## model

*Note: `module2_v4` (overlay design exploration) is preserved in
`models/hmm_stages/` but is not a runnable stage — it continued the
Module-2 notebook session. The production chain is 3a → 3b → 3c.*

| Stage | Purpose | Inputs | Outputs | Gate |
|---|---|---|---|---|
| `hmm_train_oos` | frozen-split fit + both-era decode | feature store | `I/stockday_states_split.parquet` | OOS signature/census/transition replication |
| `threshold_calibration` | TRAIN-quantile overlay thresholds | states + features | `I/stockday_states_calibrated.parquet` | GMM stability falsification; three-way convergence |
| `model_descriptives` | signatures, census, transitions | calibrated states | printed tables | effect-size drift ≤ 0.15 train→test |

## canonical

| Stage | Purpose | Inputs | Outputs | Gate |
|---|---|---|---|---|
| `canonical_panel` | issuer-bounded ISIN closure, dual-side | v2 panel, states, `I/isin_lookup.parquet`, active/inactive tables | `V/returns_panel_v3.parquet`, `V/states_v3.parquet` | Gate A re-run on v3 (0.038); coverage 98.49%, uniform across archetypes |

## validation

| Stage | Purpose | Inputs | Outputs | Verdict |
|---|---|---|---|---|
| `event_study` | excess-CAR diff-in-diff, START/END anchors | v3 panel + states | printed tables | signs + replication vs baseline |
| `deal_corroboration` | block/bulk/short-deal coincidence | v3 + `V/(block|bulk|short) deals/` | rates + z | **negative** (kept: visibility threshold) |
| `liquidity_shock` | pressure/volume/reversal arc | v3 panel + states | arc tables | pre-committed 3-part bar — met both eras |
| `panel_regression` | PanelOLS FE + 2-way cluster, R0/R1/R2 | v3 panel + states | coefficient tables | SHARK_DIST survives R2 both eras |
| `robustness` | non-overlap, horizons, dose, placebo | v3 panel + states | tables | each sub-test pre-registered |
| `gbt_challenger` | LightGBM vs regime baseline + SHAP | feature store + v3 | IC/spread/SHAP | gap > 0.01 & spread t > 2 |
| `demeaning_check` | dynamics vs characteristics | feature store + v3 | ICs raw vs demeaned | demeaned spread t > 2 (**not met** → no LSTM) |
| `flow_innovation` | AR(5) INNOV yardstick | features + v3 | T1–T4 tables | SHARK_DIST survives INNOV control |
| `vix_lambda` | state dependence (VIX, Kyle λ) | v3 + `V/india_vix.parquet` | interaction tables | replication across eras (**mixed/null**) |
| `pin_model` | EHO PIN MLE per stock-year | features (counts) + v3 | `V/fii_pin_stockyear.parquet` | sh_host > sh_sd, both eras |

### validation — referee tests (Module 13)

| Stage | Purpose | Verdict |
|---|---|---|
| `backbone_ablation` | census-matched rule backbone vs HMM, Table 1 side-by-side | **V1 — HMM not necessary** (κ=0.89; contribution = the measure) |
| `incremental_value` | flow-magnitude controls + GBT ± composition block | episode effect robust; ΔIC +0.0060 (t=3.09, +28% rel.), tail-concentrated |
| `recon_exclusion` | ±7cd exclusion around 150 index-review dates | **passive mechanics excluded** (+61.5***/+50.9** clean) |

### validation — skeptic self-audit (Module 15)

| Stage | Purpose | Verdict |
|---|---|---|
| `skeptic_tests` | bounce-free windows; direct SD−HO contrast; public VCR head-to-head; PUBLIC→FLOW→COMP GBT ladder | all four pre-registered PASS; **claims repriced** — bounce = 11%/24% of headline (bounce-free SD +58.0***/+36.7**); SD−HO contrast +69.0***/+56.6** standard, bounce-free TEST t=1.31 (underpowered, stated); composition adds beyond public data (ΔIC +0.0066, t=5.2) and conventional flow (+0.0041, t=3.2) |

## backtest

| Stage | Purpose | Inputs | Outputs | Verdict |
|---|---|---|---|---|
| `engine_gates` | engine correctness G1/G2/G3 | none (synthetic) | printed gates | all must PASS |
| `bt_baselines` | S1/S2/S3 no-model books (frozen) | v3, features, states | `V/bt12_baselines.parquet` | sanity gates only |
| `bt_hmm_twins` | HMM twins + paired ΔSharpe | v3, features, states, baselines | `V/bt12_hmm.parquet` | TEST net ΔSharpe block-bootstrap CI |
| `bt_gross_diagnosis` | signal vs implementation | the two bt12 parquets | printed diagnosis | gross ΔSharpe CI rules |
| `bt_style_switch` | S4 style-switch pair | v3, features, states | `V/bt12_style.parquet` | same CI rule + gross diagnostics |

## exhibits

| Stage | Purpose | Outputs |
|---|---|---|
| `paper_exhibits` | every paper table + figure as files | `outputs/tables/` (T1–T6, CSV+LaTeX), `outputs/figures/` (F1–F4 PNG), `outputs/regression_outputs/` (full PanelOLS coefficient tables) |
| `collect_outputs` | populate the remaining outputs/ buckets | `predictions/` (states parquets), `trained_models/` (backbone params + thresholds JSON), `validation/` (stable-named latest log per evidence stage), `descriptive_statistics/`, `metrics/` (backtest metrics + run_summary.json) |

## audit (read-only, run any time)

### tail-probe extension study (Module 14; run a → b → c)

| Stage | Purpose | Verdict |
|---|---|---|
| `tail_census` | feasibility census of the attributable illiquid tail (1,618 names) | viable (gates G1–G3 pass) |
| `tail_labels` | method-frozen rule labels on the tail | gates pass; clustering 2.25×/2.86× |
| `tail_economics` | event study + Roll friction bar | **NO TAIL EFFECT** — dislocation transfers, reversal does not |

| Stage | Question it answers |
|---|---|
| `data_audit` | are the raw collected files sane (dates, nulls, tz)? |
| `adjustment_diagnose` | what exactly is stored in the v2 returns? (found the paste-corruption) |
| `car_start_legacy` | START-anchor CARs (superseded inference, kept for the record) |
| `data_lineage` | table heads along the full derivation chain |
| `attrition` | where do model→tape join losses come from? |
| `universe_audit` | is the 946-name universe internally consistent? |
| `isin_provenance` | 5,960 → 3,812 → 946 funnel accounting |
| `cisin_validation` | are canonical ISINs valid? |
| `tape_canonicalization` | how much coverage does closure recover? (+7.8%) |
| `isin_accounting` | does active/inactive closure account close? |
| `noisin_probe` | do degenerate keys (NOISIN/null) touch the model? (no) |
| `universe_integrity` | is 946 the right company count? (939 issuers) |

## phase2 (charter: docs/PHASE2_PLAN.md; run explicitly, not in --all)

| Stage | Purpose | Verdict |
|---|---|---|
| `phase2_filtering` | 16A: forward-filtered posteriors from frozen params | **A2 PASS** — Table 1 intact on causal labels (SD +60.2***/+50.3***); onset lag 1d |
| `phase2_calibration` | 16B: nowcast Brier/log-loss/ECE vs baselines | original gate **FAIL (recorded)** — baseline was oracle-contaminated; amended causal-baseline gate **PASS** |
| `phase2_hazard` | 16C: episode-END hazard, walk-forward 2014–2025Q1 | **PASS** — AUC .797 vs age-only .569 (TEST .809), paired t +42/+34 |
| `phase2_decision` | 16D: anticipation vs confirmation, θ walk-forward | **FAIL on bar** — +23bp/episode CI[+7,+40] but age-only anticipator wins paired (−15bp, t=−2.96); forecastable ≠ monetizable |
