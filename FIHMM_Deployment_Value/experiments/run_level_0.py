"""L0 - Existence. Is the effect present under conventional inference?

Panel regression (manuscript eq. 1) of squared standardised return on the
institutional share of turnover, with instrument and date fixed effects and
log turnover controlled, SEs clustered on date.

L0 is a NECESSARY condition, never a result. With 577,245 instrument-days,
significance is close to automatic; the paper's finding is that all seven
applications clear this bar and six still fail to reach a use.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import load_config
from src.validation.frozen_split import era_masks
from src.models.ols import panel_regression_share


def main() -> int:
    log = get_logger("level_0_existence")
    banner(log, "L0 EXISTENCE - is the effect present under conventional inference?")

    cfg = load_config("model_config")["gates"]
    res = panel_regression_share(lag=0)
    log.info(f"  n = {res.nobs:,} instrument-days")

    ok = True
    for era in ("FULL", "TRAIN", "TEST"):
        t = res.tstat[era]
        passed = abs(t) >= cfg["screen_abs_t"] if era == "FULL" else abs(t) >= cfg["screen_era_abs_t"]
        ok &= passed
        verdict(log, f"share coefficient t, {era}", "PASS" if passed else "FAIL",
                f"t = {t:+.2f}")

    log.info("")
    log.info("  L0 is necessary, not sufficient. With samples this size a "
             "conventional\n  threshold is close to automatic and is treated "
             "as a filter, not a finding.")
    verdict(log, "L0 overall", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
