"""There is no download.

The depository settlement record analysed in this paper is proprietary and is
not a commercial data product. It cannot be redistributed with this package and
there is no endpoint this script could call. Pretending otherwise - shipping a
downloader that fails at runtime - would be worse than saying so here.

What this module does instead: state the access basis, and tell a reader
exactly which results they can and cannot regenerate without the raw record.
See data/source/README_NSDL.md for the full statement.
"""
from __future__ import annotations

import sys

from src.common.logging import get_logger

ACCESS_BASIS = (
    "Depository settlement records obtained for academic research under an "
    "institutional data-access arrangement, with participant identifiers "
    "masked at source by the provider before release. The records are not "
    "redistributable and no endpoint serves them."
)

# What a reader WITHOUT the raw record can still do.
REGENERABLE_WITHOUT_RAW = (
    "Tables 2 and 4 and the survivor's boundary conditions, from the scored "
    "density artifacts; the cost grid and every Level 5 figure, from the "
    "backtest metrics; all statistical machinery in analysis/, against its "
    "own unit fixtures.",
)

REQUIRES_RAW = (
    "Table 1 and the Level 0-2 regressions, which need the 577,245-row panel; "
    "the reporting-lag curve, which needs the per-record report dates; the "
    "identifier-persistence test, which needs the masked id columns.",
)


def main() -> int:
    log = get_logger("download_nsdl")
    log.info("ACCESS BASIS")
    log.info("  " + ACCESS_BASIS)
    log.info("")
    log.info("REGENERABLE FROM THE DERIVED ARTIFACTS SHIPPED/REFERENCED HERE")
    for s in REGENERABLE_WITHOUT_RAW:
        log.info("  " + s)
    log.info("")
    log.info("REQUIRES THE RAW RECORD")
    for s in REQUIRES_RAW:
        log.info("  " + s)
    log.info("")
    log.info("This script intentionally downloads nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
