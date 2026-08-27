"""L5 - Execution. Does the edge exceed the cost of capturing it?

Application A5: a mechanical long/short proxy for the concentration measure,
carrying no fitted model at all, so what is priced is the DATA rather than an
architecture.

We report the BREAKEVEN one-way cost rather than a net figure at an assumed
cost. The breakeven is a property of the strategy; a net Sharpe is a property
of the assumption, and a reader whose costs differ can use the first and
cannot use the second.

The instrument is declared in config/execution_config.yaml. It has to be: the
producing engine models cost as a bare bp rate on turnover, and in the Indian
market the instrument decides the verdict. For cash-delivery equity, STT alone
is 10 bp per side - which already exceeds this strategy's 7.4 bp breakeven.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import load_config, local
from src.validation.execution_costs import (breakeven_table, cost_grid,
                                            declared_cost_stack)


def main() -> int:
    log = get_logger("level_5_execution")
    banner(log, "L5 EXECUTION - does the edge exceed the cost of capturing it?")

    cfg = load_config("execution_config")
    log.info(f"  instrument: {cfg['instrument']}  ({cfg['segment']}, "
             f"{cfg['rebalance']} rebalance)")
    log.info("")

    stack = declared_cost_stack()
    log.info("  declared one-way cost stack (basis points):")
    for k, v in stack["components"].items():
        log.info(f"    {k:<32s} {v:>6.2f}")
    log.info(f"    {'TOTAL (ex impact)':<32s} {stack['total_ex_impact']:>6.2f}")
    log.info(f"    {'STT alone':<32s} {stack['stt']:>6.2f}")
    log.info("")

    be = breakeven_table()
    be.to_csv(local("tables") / "table_5_execution.csv", index=False)
    for _, r in be.iterrows():
        log.info(f"    {r['quantity']:<28s} TRAIN {r['training']:>8}"
                 f"   TEST {r['test']:>8}")
    log.info("")

    grid = cost_grid()
    grid.to_csv(local("tables") / "table_5_cost_grid.csv", index=False)
    log.info("  net Sharpe across the cost grid (not a single assumption):")
    for _, r in grid.iterrows():
        log.info(f"    {r['cost_bps']:>6.1f} bp   TRAIN {r['net_sharpe_train']:+.2f}"
                 f"   TEST {r['net_sharpe_test']:+.2f}")
    log.info("")

    be_test = float(be.set_index("quantity").loc["breakeven_one_way_bps", "test"])
    verdict(log, "breakeven exceeds STT alone",
            "FAIL" if be_test < stack["stt"] else "PASS",
            f"breakeven {be_test:.2f} bp vs STT {stack['stt']:.2f} bp per side")
    verdict(log, "breakeven exceeds full declared stack",
            "FAIL" if be_test < stack["total_ex_impact"] else "PASS")
    log.info("")
    log.info("  A5 is lost at L5, and by a wider margin than the headline 15 bp")
    log.info("  comparison implies. The signal is economically real and")
    log.info("  unharvestable in the instrument it would have to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
