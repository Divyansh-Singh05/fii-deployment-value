# Data files

This project **reads** market and flow data from an external directory and
**writes** everything to `outputs/` inside this tree.

- **Input data** — resolved by `paths.py` in this order: `$FII_DATA_ROOT`,
  then `./data`, then `~/Desktop/Major Project 2/data`. It must contain
  `VALIDATION_DATA/` and `ISIN_MAPPING/`. Import fails loudly otherwise.
- **Outputs** — always `./outputs/` here (override with `$FII_OUTPUTS`).
  The `outputs/phase3/` tables described below are produced by this project,
  not read from anywhere else.

`outputs/phase3/vintages.parquet` **ships with the repository**: it is the
fitted model and is the one artifact you do not need to rebuild.

| mark | meaning |
|---|---|
| **★ FINAL** | read directly by the risk engine (`s05`–`s13`) |
| **▲ FEEDS FINAL** | intermediate the engine's inputs are built from |
| **● UPSTREAM** | raw or plumbing input, needed only to rebuild from scratch |
| **○ SUPERSEDED** | older version, nothing reads it any more |
| **◌ OTHER THREAD** | a different research question, not the engine |
| **✗ ORPHAN** | no reference found anywhere in the source tree |

---

## `outputs/phase3/` — everything the engine produces

| mark | file | description |
|---|---|---|
| ★ | `vintages.parquet` | **The core artifact.** 106 immutable monthly parameter snapshots, 69 columns — HMM means, covariances, transition matrix, and archetype thresholds with stock-clustered bootstrap SEs, 95% intervals and bootstrap normality diagnostics. 42 of the columns are the threshold family: the production values are posterior-weighted quantiles, and `*_argmax`, `*_hard` and `*_viterbi` are stored beside them as unused controls so the estimator change can be decomposed. Written by `s05`. |
| ★ | `daily_posteriors.parquet` | 587k stock-days of causal filtered state probabilities `P(S_t \| x_1..t)`, plus vintage id, parameter age and gap steps. Written by `s06`. |
| ★ | `archetype_probs.parquet` | The same rows as seven archetype probabilities with soft threshold bands. Written by `s07`. |
| ★ | `outcome_densities.npz` | 12 blocks of past-only outcome CDFs (3 horizons × soft/hard weighting × expanding/rolling window), each 106 vintages × 7 archetypes × 1001 grid points, with the per-cell effective sample sizes used to enforce the ESS floor. Written by `s08`. |
| ★ | `outcome_z.parquet` | Per-row volatility-standardised forward returns at h = 1, 5, 20, with the EWMA sigma used. Written by `s08`. |
| ★ | `predictive.parquet` | **What the engine serves.** VaR, ES, CRPS and PIT for five competing systems across three horizons — 85 columns, 136 MB. Carries `stock_out_of_window` so rows scored outside their vintage's fitting universe stay identifiable downstream. VaR and ES are **null** where the warm-up frontier or the tail-support floor withholds them — 32,689 rows at the 1% level for `soft_roll` h=1, which is correct behaviour, not missing data. Written by `s09`, read by `s12`. |
| ★ | `c6_results.parquet` | Pre-registered test statistics: Diebold-Mariano, Kupiec, Christoffersen, PIT, by stratum and horizon. Written by `s10`. |
| ★ | `stock_risk_profiles.csv` | Per-stock comparative metrics, 546 stocks × 68 columns, for downstream plotting. Written by `s11`. |
| ★ | `C7_RISK_REPORT.md` | Head-to-head narrative report against the parametric benchmark. Written by `s11`. |
| ○ | `vintages_original.parquet` | Output of the original estimation harness, which was lost before it could be version-controlled. Its thresholds are **Viterbi-decoded** — a fact only established when the current run stored all four variants side by side. Kept so the reconstruction can be compared against it. Not read by anything. |
| ○ | `vintages_iidboot.parquet` | The reconstruction before the threshold fixes. It differs from the current file in two ways: standard errors came from IID rather than stock-clustered resampling, understating them ~4.5×; and thresholds were hard-argmax rather than posterior-weighted. The HMM fits themselves are identical (log-likelihood matches to the digit), so the two files isolate the estimator change exactly. Kept to make that attributable. Not read by anything. |

---

## `data/VALIDATION_DATA/` — model inputs

| mark | file | description |
|---|---|---|
| ★ | `states_v3.parquet` | **Primary engine input.** 795k stock-days: the four flow features, canonical ISIN, era tag, plus the Phase-I state and archetype labels. Read by `s05`, `s06`. |
| ★ | `returns_panel_v3.parquet` | **Primary engine input.** Corporate-action-adjusted price panel with market, USD/INR, S&P 500 and VIX series merged in. Supplies the trading calendar, forward returns and EWMA volatility. Read by `s06`, `s08`, `s11`, `s12`. |
| ▲ | `returns_panel_v2.parquet` | Predecessor of v3, before the ISIN-closure remap. Needed only to rebuild v3. |
| ▲ | `returns_panel.parquet` | Predecessor of v2, before corporate-action adjustment. |
| ▲ | `ca_adjustment_factors.parquet` | Split and bonus adjustment factors applied to build the adjusted return series. |
| ▲ | `nse_corporate_actions.csv` | Raw NSE corporate-action feed behind those factors. |
| ● | `nifty50.parquet` | Market index series, merged into `returns_panel_v3` as `nifty50_ret`. |
| ● | `sp500.parquet` | US index, merged in lagged one day as `sp500_ret_lag`. |
| ● | `usdinr.parquet` | Currency series, merged in as `usdinr_ret`. |
| ● | `india_vix.parquet` | Volatility index, merged in as `india_vix`. |
| ◌ | `phase2_filtered_states.parquet` | Output of `s04` — filtered posteriors under frozen Phase-I parameters. Consumed by the Phase II calibration and hazard follow-ons, not by the walk-forward engine. |
| ◌ | `phase2_hazard_preds.parquet` | Phase II hazard-model predictions. |
| ◌ | `fhmm_filtered_states.parquet` | Factorial-HMM experiment — an alternative regime specification that was tried and not adopted. |
| ◌ | `fhmm_hazard_preds.parquet` | Hazard predictions from that same factorial variant. |
| ◌ | `fii_pin_stockyear.parquet` | Independent probability-of-informed-trading estimates, used to corroborate the concentration story in Phase I. |
| ◌ | `bt12_baselines.parquet` | Backtest baselines from the trading-strategy thread. |
| ◌ | `bt12_hmm.parquet` | Backtest results for the HMM-conditioned strategies. |
| ◌ | `bt12_style.parquet` | Style-switching backtest variant. |
| ◌ | `tail_states_v1.parquet` | Tail-event census labels from the extreme-move analysis. |
| ◌ | `index_recon_dates.csv` | Index reconstitution dates, used to test whether results survive excluding them. |
| ◌ | `bhavcopy_parquets/` | Yearly raw NSE price files behind `returns_panel`. |
| ◌ | `bulk deals/`, `block deals/`, `short deals/` | Raw NSE disclosure CSVs, used for independent corroboration of the flow measures. |
| ✗ | `crisis_windows.csv` | Crisis date windows. No reference anywhere in the source tree. |
| ✗ | `ca_raw.parquet` | Unprocessed corporate actions. Superseded by the ISIN_MAPPING copy; unreferenced here. |
| ✗ | `bse_corporate_actions.csv` | BSE corporate actions. The NSE feed is used instead; this is unreferenced. |

---

## `data/ISIN_MAPPING/` — raw institutional data and feature build

| mark | file | description |
|---|---|---|
| ★ | `2011.parquet` … `2025.parquet` | **The raw institutional transaction record**, one file per year, ~500 MB total. Custodian-reported FII trades: entity id, sub-account, broker, ISIN, trade date, buy/sell flag, rate, quantity, value. Read by `s01` via a glob, not by literal filename. Note: `TR_TYPE` is 100% null in 2023-06, 2023-09 and 2023-11 — a parse failure that costs 3.0% of the sample. |
| ★ | `active_isins.csv` | Currently listed ISINs, used to build the canonical-ISIN map in `s01`. |
| ★ | `inactive_isins.csv` | Delisted or renamed ISINs with their successors, for the same map. |
| ▲ | `stockday_features_v2.parquet` | The full ten-feature stock-day panel produced by `s01`; four of its columns become the model features. |
| ▲ | `stockday_states_split.parquet` | Feature panel with HMM states attached and the train/test era tag, from `s02`. |
| ▲ | `stockday_states_calibrated.parquet` | The same with statistically calibrated archetype thresholds applied, from `s03`. This is what `states_v3` is derived from. |
| ● | `isin_lookup.parquet` | Old-to-canonical ISIN mapping built from corporate actions. |
| ● | `isin_master_clean.parquet` | Cleaned ISIN master reference table. |
| ● | `ca_raw.parquet` | Raw corporate actions feeding the ISIN closure logic. |
| ○ | `stockday_features.parquet` | First feature-store build, superseded by `_v2` (which fixed a null-ISIN phantom-stock bug and added seven features). Nothing reads it. |
| ○ | `stockday_states.parquet` | First state assignment, superseded by the split/calibrated chain. Nothing reads it. |
| ○ | `stockday_states_final.parquet` | Hybrid overlay variant from an earlier design, superseded by the calibrated thresholds. Read only by the retired overlay module. |
| ◌ | `stockday_states_fhmm.parquet` | States under the factorial-HMM alternative. |
| ● | `isin_*.csv`, `isin_*.xlsx`, `non_active_isins.*`, `nsdl_isin_status.csv`, `restructure_*.csv` | Manual-review spreadsheets and diagnostics from the ISIN reconciliation. Audit trail, not pipeline inputs. |

---

## Minimum set to run the engine

Two external input files, plus the shipped model:

```
<FII_DATA_ROOT>/VALIDATION_DATA/states_v3.parquet         19 MB   external
<FII_DATA_ROOT>/VALIDATION_DATA/returns_panel_v3.parquet 205 MB   external
outputs/phase3/vintages.parquet                          0.1 MB   ships here
```

`./run_all.sh` turns those into everything else in about seven minutes.
`s11` also reads `VALIDATION_DATA/crisis_windows.csv` for the conditional
calibration strata.

`s12_risk_engine` then needs only files this project produced:
`predictive.parquet`, `archetype_probs.parquet`, `vintages.parquet` and
`outcome_densities.npz`. Everything else listed above is required only to
rebuild the chain from the raw transaction record (`s01`-`s05`).

---

## Freshness

`vintages.parquet` was refitted on **2026-08-22** with the gap-aware
estimator: the transition matrix is now expressed per trading day, so
observations `k` days apart are linked by `A^k` rather than by one transition
per observed row. Regime half-life moves from 8.7 to **33.8 days** as a
result. The features underneath it were rebuilt at the same time on a
point-in-time identity map and the corrected participation-concentration
measure.

Everything downstream was regenerated in this tree on **2026-08-23** and
reproduces the research project's figures exactly. The look-ahead audit passed
every gate with exact zeros at `--asof 2021-06-30`.

Two files this project reads are *inputs* built upstream and are not
regenerated here: `states_v3.parquet` and `returns_panel_v3.parquet`.
