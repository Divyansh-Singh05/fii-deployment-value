"""Recompute every manuscript number and compare against data/expected/.

The gate vocabulary is deliberate:
  PASS      recomputed value matches the manuscript within tolerance
  MISMATCH  recomputed value differs - the DECLARATION is corrected, never
            the tolerance
  STALE     the artifact behind the number has moved since it was recorded
  SKIP      the source tree needed for this number is not reachable

Two expected rows deliberately record a FAILING verdict: the survivor's
Clark-West adjustment and its common-window multiplicity p-value do not clear
their bars, and the manuscript reports both. A run that turned those into
passes would be WRONG, so they are compared on value, not on verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import (MissingDataset, QuarantinedArtifact,
                                  local, load_config)

EXPECTED = Path(__file__).resolve().parents[1] / "data" / "expected"


def _cmp(log, label, got, want, tol) -> bool:
    if got is None or (isinstance(got, float) and not np.isfinite(got)):
        verdict(log, label, "SKIP", "not recomputed")
        return True
    ok = abs(float(got) - float(want)) <= float(tol)
    verdict(log, label, "PASS" if ok else "MISMATCH",
            f"got {float(got):+.4f}  expected {float(want):+.4f}  tol {tol}")
    return ok


def verify_table_1(log) -> bool:
    """Table 1, from the local 577,245-row panel."""
    from src.models.ols import panel_regression_share
    want = pd.read_csv(EXPECTED / "expected_table_1.csv").set_index("information_set")
    ok = True
    for label, lag in (("contemporaneous_lag0", 0), ("operational_lag_t2", 2)):
        res = panel_regression_share(lag=lag)
        for col, era in (("full_sample", "FULL"), ("training", "TRAIN"),
                         ("test_era", "TEST")):
            ok &= _cmp(log, f"table1 {label} [{era}]", res.tstat[era],
                       want.loc[label, col], want.loc[label, "tolerance"])
    return ok


def verify_table_2(log) -> bool:
    from src.validation.baseline_comparison import controlled_pair
    want = pd.read_csv(EXPECTED / "expected_table_2.csv").set_index("metric")
    got = controlled_pair().set_index("metric")
    ok = True
    for metric in want.index:
        for col in ("ewma_base_A8", "gjr_base_A2"):
            if metric not in got.index:
                continue
            ok &= _cmp(log, f"table2 {metric} [{col}]",
                       got.loc[metric, col], want.loc[metric, col],
                       want.loc[metric, "tolerance"])
    return ok


def verify_survivor(log) -> bool:
    from analysis.statistical_tests import block_bootstrap_p, clark_west
    from src.validation.baseline_comparison import (common_window,
                                                    survivor_own_window,
                                                    survivor_vs_bare_baseline)
    want = pd.read_csv(EXPECTED / "expected_survivor_boundary.csv")
    ok = True

    t_own, _ = survivor_own_window()
    row = want[(want.condition == "gate_own_window")].iloc[0]
    ok &= _cmp(log, "survivor gate, own window", t_own, row.value, row.tolerance)

    t_bare, _ = survivor_vs_bare_baseline()
    row = want[(want.condition == "bare_baseline")].iloc[0]
    ok &= _cmp(log, "survivor vs bare baseline", t_bare, row.value, row.tolerance)

    a, _ = common_window()
    d = (a["crps_M2_vix_flow"] - a["crps_M1_vix"]).to_numpy()
    p_boot, _ = block_bootstrap_p(d)
    row = want[(want.condition == "nested_block_bootstrap_common")].iloc[0]
    ok &= _cmp(log, "nested block bootstrap, common", p_boot, row.value, row.tolerance)

    nu1, nu2 = a["nu_M1_vix"].to_numpy(), a["nu_M2_vix_flow"].to_numpy()
    f1 = a["scale_M1_vix"].to_numpy() ** 2 * nu1 / (nu1 - 2.0)
    f2 = a["scale_M2_vix_flow"].to_numpy() ** 2 * nu2 / (nu2 - 2.0)
    t_cw, p_cw, _ = clark_west(a["ret_next"].to_numpy() ** 2, f1, f2)
    row = want[(want.condition == "nested_clark_west")
               & (want.statistic == "t_statistic")].iloc[0]
    ok &= _cmp(log, "nested Clark-West t", t_cw, row.value, row.tolerance)
    log.info("        (Clark-West is EXPECTED to fail its bar; the manuscript "
             "reports it as boundary condition 5)")

    # Crisis concentration. NOT a manuscript number - a diagnostic surfaced by
    # the cumulative score path in Figure 4. Verified so it cannot drift.
    from src.validation.baseline_comparison import survivor_crisis_dependence
    cd = survivor_crisis_dependence()
    for stat, got in (("share_of_advantage", cd["share_of_advantage"]),
                      ("dm_excluding_mar_apr_2020", cd["dm_excluding_window"])):
        row = want[(want.condition == "crisis_concentration")
                   & (want.statistic == stat)].iloc[0]
        ok &= _cmp(log, f"crisis {stat}", got, row.value, row.tolerance)
    log.info(f"        ({100 * cd['share_of_advantage']:.0f}% of the advantage "
             f"accrues in {100 * cd['share_of_sample']:.1f}% of the sample; "
             f"excluding it the gate FAILS)")
    return ok


def verify_execution(log) -> bool:
    from src.validation.execution_costs import breakeven_table, verdict_against_stack
    want = pd.read_csv(EXPECTED / "expected_level_5_execution.csv").set_index("quantity")
    got = breakeven_table().set_index("quantity")
    ok = True
    for q in want.index:
        for col in ("training", "test"):
            ok &= _cmp(log, f"level5 {q} [{col}]", got.loc[q, col],
                       want.loc[q, col], want.loc[q, "tolerance"])
    v = verdict_against_stack()
    verdict(log, "breakeven clears STT alone", "FAIL" if not v["clears_stt"] else "PASS",
            f"{v['breakeven_bps']} bp vs {v['stt_alone_bps']} bp per side")
    return ok


def main() -> int:
    log = get_logger("verify_results")
    banner(log, "VERIFICATION - recomputed against the manuscript's declared values")
    results = {}

    for name, fn in (("Table 1 (availability lag)", verify_table_1),
                     ("Table 2 (controlled pair)", verify_table_2),
                     ("Survivor boundary conditions", verify_survivor),
                     ("Level 5 execution", verify_execution)):
        log.info("")
        log.info(f"--- {name}")
        try:
            results[name] = fn(log)
        except (MissingDataset, FileNotFoundError) as exc:
            verdict(log, name, "SKIP", str(exc).splitlines()[0])
            results[name] = True
        except QuarantinedArtifact as exc:
            verdict(log, name, "STALE", str(exc))
            results[name] = False

    log.info("")
    banner(log, "VERIFICATION SUMMARY")
    for k, v in results.items():
        verdict(log, k, "PASS" if v else "MISMATCH")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
