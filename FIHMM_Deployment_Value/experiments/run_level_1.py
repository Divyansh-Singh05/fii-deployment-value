"""L1 - Persistence. Does the effect hold on an era the design never touched?

NOT replication in the credibility literature's sense, and the manuscript is
explicit about that: this is out-of-sample persistence under a SINGLE research
design. Only 441 of the two eras' 1,028 instruments are common, so a pass here
is persistence across a shifted cross-section.

A sign flip between eras is failure regardless of significance in either.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import load_config
from src.validation.frozen_split import describe_split
from src.models.ols import panel_regression_share


def main() -> int:
    log = get_logger("level_1_persistence")
    banner(log, "L1 PERSISTENCE - does it hold on a frozen, never-touched era?")

    split = describe_split()
    log.info(f"  train ends {split['train_end']}   embargo {split['embargo']}   "
             f"test opens {split['test_start']}")
    log.info(f"  common instruments: {split['common_instruments']} of "
             f"{split['total_instruments']} - a minority of each era")
    log.info("")

    res = panel_regression_share(lag=0)
    t_tr, t_te = res.tstat["TRAIN"], res.tstat["TEST"]

    same_sign = (t_tr < 0) == (t_te < 0)
    verdict(log, "sign stable across eras", "PASS" if same_sign else "FAIL",
            f"TRAIN {t_tr:+.2f}  TEST {t_te:+.2f}")

    bar = load_config("model_config")["gates"]["screen_era_abs_t"]
    strong = abs(t_tr) >= bar and abs(t_te) >= bar
    verdict(log, f"both eras |t| >= {bar}", "PASS" if strong else "FAIL")

    ok = same_sign and strong
    log.info("")
    log.info("  No application in the programme is lost at L0 or L1. All "
             "attrition\n  occurs under the operational constraints L2-L5.")
    verdict(log, "L1 overall", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
