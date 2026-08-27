# Re-derivations for manuscript v2

Five scripts, each answering one referee charge with a number computed from a
primary artifact rather than asserted. `RESULTS.log` is their combined output.

Run from the research tree root (`~/Desktop/Major Project 2`) with its `.venv`;
`exp3` additionally needs `convgap` on `PYTHONPATH`.

| script | charge answered | result |
|---|---|---|
| `exp1_controlled_pair.py` | The Table 2 pair varied the baseline **and** the flow feature set (2 vs 5) and was scored on different day counts. | On identical days with an identical tilt: EWMA −2.22 (pass), GJR **+0.28** (fail). Claim survives, strengthened. Also: the tilt vs the *bare* EWMA base is −1.29 and misses its own bar. |
| `exp2_multiplicity.py` | "Four engine variants examined without a correction." | Survivor one-sided p = 0.0095 → Bonferroni/Holm ×4 = **0.038**, survives at 5%; **0.052** on the common window, marginal. |
| `exp3_lag_sensitivity.py` | The two-day lag is an assumption, so the availability result may be knife-edge. | C4 coefficient negative and full-sample significant at every lag t−1…t−5; the both-era leg passes only at t−2 and t−4. Not knife-edge in sign; era legs are noisy. |
| `exp4b_reporting_lag_conservative.py` | The reporting latency is never documented. | Measured from `RFDE_RPT_DT` vs `TR_DATE`, value-weighted, conservative denominator: **0.13%** by t+0, **94.30%** by t+1, **97.51%** by t+2. t−2 is justified and conservative. |
| `exp5_table1_likeforlike.py` | — (found here, not raised by either reviewer) | Published Table 1 used stock+date FE on row 1 and date FE only on row 2. Like-for-like: −5.53 → **−0.43** / −0.09 / −0.44. Conclusion unchanged. |
| `exp6_table2_v2.py` | — | Exact Table 2 v2 contents on the common 2,235 days. |

## Corrections these forced in the manuscript

1. Table 2 rebuilt on the true controlled pair (`G2a`, not `G2`).
2. Table 1 rebuilt like-for-like; equation (1) clustering description corrected
   from "instrument and calendar month" to **date**, which is what runs.
3. Measured reporting lag added; the two-day lag is no longer an assumption.
4. Multiplicity priced rather than argued around.
5. Survivor's failure against the bare baseline (−1.29) now reported.
6. "Highest of eight books" dropped — a max-selection statistic.
7. "Replication" narrowed to era-stability on a temporal holdout.

## Untraced figure found and not used

`module19_institutional_share.py` states in its docstring that custodian
reporting is "1.5% of value same-day, 67% by T+1". Both figures are wrong —
measured values are 0.13% and 94.30% — and neither has a producing stage. This
is a fourth instance of the failure mode the verification protocol already
reports three of. The docstring figure is not cited anywhere in the manuscript.
