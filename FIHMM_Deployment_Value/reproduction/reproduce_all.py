"""One command: run the ladder, regenerate tables and figures, then verify."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger


def main() -> int:
    log = get_logger("reproduce_all")
    banner(log, "FULL REPRODUCTION")

    from experiments import run_all
    from reproduction import reproduce_figures, reproduce_tables, verify_results

    run_all.main()
    reproduce_tables.main()
    reproduce_figures.main()
    rc = verify_results.main()

    log.info("")
    log.info("  outputs/tables, outputs/figures, outputs/machine_readable and")
    log.info("  outputs/logs now hold the regenerated package.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
