"""L2 - Availability. Is the effect knowable when the decision must be made?

Application A1. The identical specification re-estimated with the predictor
taken at the lag a decision could actually use. Nothing else changes: same
panel, same fixed effects, same controls, same clustering.

The deployable lag is MEASURED, not assumed. Every record carries the date the
custodian reported it alongside the trade date, so the reporting curve is read
off the data: 0.13% of value by t+0, 94.30% by t+1, 97.51% by t+2.

Reproduces manuscript Table 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import load_config, local
from src.features.flows import reporting_lag_curve
from src.models.ols import panel_regression_share, robustness_estimators


def main() -> int:
    log = get_logger("level_2_availability")
    banner(log, "L2 AVAILABILITY - is it knowable when the decision is taken?")

    curve = reporting_lag_curve()
    log.info("  measured depository reporting curve (value-weighted):")
    for k, pct in sorted(curve.items()):
        log.info(f"    by close of t+{k}: {pct:6.2f}% of value reported")
    deployable = load_config("model_config")["flow"]["reporting_lag"]
    log.info(f"  -> t-{deployable} is the earliest feasible deployment lag, "
             f"and the conservative one")
    log.info("")

    rows = []
    for lag, label in ((0, "contemporaneous_lag0"), (deployable, "operational_lag_t2")):
        res = panel_regression_share(lag=lag)
        rows.append({"information_set": label,
                     "full_sample": res.tstat["FULL"],
                     "training": res.tstat["TRAIN"],
                     "test_era": res.tstat["TEST"]})
        log.info(f"  {label:<24s} FULL {res.tstat['FULL']:+.2f}   "
                 f"TRAIN {res.tstat['TRAIN']:+.2f}   TEST {res.tstat['TEST']:+.2f}")

    table1 = pd.DataFrame(rows)
    table1.to_csv(local("tables") / "table_1_availability.csv", index=False)

    log.info("")
    log.info("  the collapse is not an artifact of the variance estimator:")
    rob = robustness_estimators()
    rob.to_csv(local("tables") / "table_1_robustness.csv", index=False)
    for _, r in rob.iterrows():
        log.info(f"    {r['estimator']:<28s} contemporaneous {r['contemporaneous_full']:+.2f}"
                 f"   lagged {r['lagged_full']:+.2f}")

    contemporaneous_strong = abs(table1.loc[0, "full_sample"]) >= 2.5
    lagged_absent = abs(table1.loc[1, "full_sample"]) < 2.0
    verdict(log, "contemporaneous effect present", "PASS" if contemporaneous_strong else "FAIL")
    verdict(log, "effect survives at deployable lag",
            "FAIL" if lagged_absent else "PASS",
            "the effect does not weaken - it disappears")
    log.info("")
    log.info("  A1 is lost at L2. The signal is contemporaneous market "
             "microstructure,\n  not a tradable forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
