# Demonstration engine

Read-only pointers. Nothing here duplicates data or code.

## The engine

`~/Desktop/fii_risk_engine` — `./run_all.sh` (~7 min from the sealed vintages).

| What | Where |
|---|---|
| Structural limitations L1–L6 and code review R2-1…R2-6 — **the register's primary source** | `LIMITATIONS.md` |
| Head-to-head scorecard vs EWMA-Normal | `outputs/phase3/C7_RISK_REPORT.md` |
| Truncation audit (D3) | `s13_lookahead_audit.py` |
| Inference primitives: block bootstrap, null-centred PIT, ICC, Kupiec, Christoffersen, DM | `panel_inference.py`, `s10_validation.py` |
| Evidence floors, ESS deflation, burn-in seed | `s08_outcome_densities.py` |
| Per-component support counts, abstention | `s09_predictive_mixture.py` |
| Conditional calibration, causal strata (D2) | `s11_stock_risk_profiles.py`, `panel_inference.expanding_quintiles` |
| Vintage refit, convergence gate, no-fallback rule | `s05_refit_harness.py` |
| Serve-vs-artifact re-derivation, horizon retirement | `s12_risk_engine.py` |

## The research object

`~/Desktop/Major Project 2` — `python pipeline.py --all` (~8 min).

| What | Where |
|---|---|
| Mechanism ablation (C6), market-engine estimation results (R6) | `docs/AUDIT_RETRACTIONS.md` |
| Pre-registrations with binding bars and failure branches | `docs/PHASE3_PREREG.md`, `docs/PREREG_*.md` |
| Post-audit results summary | `docs/paper/FII_final_results_one_pager.md` |

## Not yet done

- Every magnitude in `docs/03_FAILURE_MODES.md` must be traced to a current
  output file. The engine was re-run after the evidence floors and the
  volatility correction changed; some figures in older documents predate that.
