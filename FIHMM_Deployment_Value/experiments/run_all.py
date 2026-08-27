"""Run the full ladder, L0 through L5, and emit the attrition summary.

The ladder is ORDERED: a finding must clear each level to reach the next, so
the level at which an application stops is a complete statement of what the
data was worth in that use. Attrition is cumulative and interpretable - an
application that dies at L5 has, by construction, survived four filters that
killed others.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger
from analysis.summary_results import write_attrition_summary

LEVELS = [
    ("run_level_0", "L0 existence"),
    ("run_level_1", "L1 persistence"),
    ("run_level_2", "L2 availability"),
    ("run_level_3", "L3 competition"),
    ("run_level_4", "L4 detectability"),
    ("run_level_5", "L5 execution"),
]


def main() -> int:
    log = get_logger("run_all")
    banner(log, "THE DEPLOYMENT LADDER - six ascending constraints, in order")
    t0 = time.time()
    failures: list[str] = []

    for module_name, label in LEVELS:
        log.info("")
        log.info(f"----- {label} " + "-" * (60 - len(label)))
        mod = __import__(f"experiments.{module_name}", fromlist=["main"])
        try:
            rc = mod.main()
        except Exception as exc:                      # noqa: BLE001
            log.info(f"  ERROR in {label}: {exc}")
            failures.append(label)
            continue
        if rc != 0:
            failures.append(label)

    log.info("")
    summary = write_attrition_summary()
    banner(log, "ATTRITION SUMMARY")
    for line in summary.splitlines():
        log.info("  " + line)

    log.info("")
    log.info(f"  elapsed {time.time() - t0:.1f}s")
    if failures:
        log.info(f"  stages reporting a non-zero status: {', '.join(failures)}")
        log.info("  NOTE: a non-zero status is not necessarily an error. Six of")
        log.info("  seven applications are SUPPOSED to fail; that is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
