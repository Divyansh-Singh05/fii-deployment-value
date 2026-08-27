# Expected results

Transcribed from the accepted manuscript's tables, not recomputed here. These
are the regression fixtures: `reproduction/verify_results.py` recomputes each
quantity from the source trees and compares against these values within the
stated per-row tolerance, emitting PASS / FAIL / MISMATCH.

| File | Manuscript object |
|---|---|
| `expected_table_1.csv` | Table 1 - availability lag, application A1 |
| `expected_table_1_robustness.csv` | Table 1 note - five variance estimators |
| `expected_table_2.csv` | Table 2 - one predictor, two baselines (A8 vs A2) |
| `expected_table_3.csv` | Table 3 - hazard skill vs decision value (A4) |
| `expected_table_4.csv` | Table 4 - ablation of state conditioning (A7) |
| `expected_survivor_boundary.csv` | Five boundary conditions on the survivor |
| `expected_level_5_execution.csv` | Level 5 - breakeven and net Sharpe (A5) |
| `expected_reporting_lag.csv` | Depository reporting-lag curve |

A tolerance of 0 means the value must match exactly (counts, day totals).

Two rows deliberately record FAIL. The survivor's Clark-West adjustment and its
common-window multiplicity p-value do not clear their bars, and the manuscript
reports both. A reproduction that turned them into passes would be wrong.
