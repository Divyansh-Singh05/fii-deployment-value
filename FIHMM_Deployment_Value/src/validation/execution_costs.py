"""L5 - the breakeven cost, and the stack it must be compared against.

We report the BREAKEVEN rather than a net figure at an assumed cost:

    c* = E[gross pnl] / E[turnover]        (manuscript eq. 4)

The breakeven is a property of the strategy; a net Sharpe is a property of the
assumption. A reader whose costs differ can use the first and cannot use the
second.

The producing engine models cost as a bare one-way bp rate on turnover and does
NOT model an instrument. Since A5's breakeven is 7.4 bp and Indian cash-delivery
STT alone is 10 bp per side, the instrument decides the verdict - so it is
declared explicitly in config/execution_config.yaml rather than left implicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utilities import dataset, load_config

# A5 in the manuscript = the S3_PROXY book in the metrics table:
# a mechanical concentration proxy carrying no fitted model, so what
# Level 5 prices is the DATA rather than an architecture.
STRATEGY = "S3_PROXY"


def declared_cost_stack() -> dict:
    """The one-way cash-delivery stack, itemised."""
    cfg = load_config("execution_config")
    comp = dict(cfg["cost_stack_bps"])
    impact = comp.pop("market_impact_reference", 0.0)
    return {
        "instrument": cfg["instrument"],
        "components": comp,
        "total_ex_impact": float(sum(comp.values())),
        "impact_reference": float(impact),
        "stt": float(comp["securities_transaction_tax"]),
    }


def _metrics() -> pd.DataFrame:
    return pd.read_csv(dataset("backtest_metrics"))


def breakeven_table() -> pd.DataFrame:
    """Level 5 figures, computed from the strategy's own scored metrics.

    `margin_bps` in the metrics table IS the breakeven of equation (4): the
    ratio of expected gross profit to expected turnover.
    """
    m = _metrics()
    a5 = m[(m["strategy"] == STRATEGY) & (m["cost_bps"] == 0)]
    at15 = m[(m["strategy"] == STRATEGY) & (m["cost_bps"] == 15)]

    def pick(frame, col, era):
        return float(frame[frame["era"] == era][col].iloc[0])

    return pd.DataFrame([
        {"quantity": "gross_sharpe",
         "training": round(pick(a5, "sharpe", "TRAIN"), 2),
         "test": round(pick(a5, "sharpe", "TEST"), 2)},
        {"quantity": "breakeven_one_way_bps",
         "training": round(pick(a5, "margin_bps", "TRAIN"), 2),
         "test": round(pick(a5, "margin_bps", "TEST"), 2)},
        {"quantity": "net_sharpe_at_15bp",
         "training": round(pick(at15, "sharpe", "TRAIN"), 2),
         "test": round(pick(at15, "sharpe", "TEST"), 2)},
    ])


def cost_grid() -> pd.DataFrame:
    """Net Sharpe across the declared cost grid.

    The engine scores only two cost points (0 and 15 bp), but the grid is exact
    at every point rather than approximate. Cost enters as
    `net = gross - c * turnover`, and turnover is a property of the book that
    cost does not change, so annualised return is linear in c and annualised
    volatility is INVARIANT to it. Sharpe is therefore linear in c, and the
    two scored anchors determine the whole line - including beyond 15 bp.

    `np.interp` would be wrong here: it clamps outside the scored range and
    would report the 15 bp figure for 20 and 30 bp.
    """
    m = _metrics()
    a5 = m[m["strategy"] == STRATEGY]
    rows = []
    for c in load_config("execution_config")["cost_grid_bps"]:
        row = {"cost_bps": float(c)}
        for era, col in (("TRAIN", "net_sharpe_train"), ("TEST", "net_sharpe_test")):
            e = a5[a5["era"] == era].sort_values("cost_bps")
            x = e["cost_bps"].to_numpy(float)
            y = e["sharpe"].to_numpy(float)
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            row[col] = round(float(y[0] + slope * (c - x[0])), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_against_stack() -> dict:
    """Does the edge clear the cost it must actually pay?"""
    stack = declared_cost_stack()
    be = float(breakeven_table().set_index("quantity")
               .loc["breakeven_one_way_bps", "test"])
    return {
        "breakeven_bps": be,
        "stt_alone_bps": stack["stt"],
        "full_stack_bps": stack["total_ex_impact"],
        "clears_stt": be >= stack["stt"],
        "clears_full_stack": be >= stack["total_ex_impact"],
        "reading": ("STT alone exceeds the breakeven, so the execution "
                    "constraint binds harder than the headline 15 bp "
                    "comparison implies"),
    }


def strategy_book_sharpes() -> pd.DataFrame:
    """All eight books at zero cost - the family the deflated Sharpe prices."""
    m = _metrics()
    return (m[m["cost_bps"] == 0][["strategy", "era", "sharpe", "margin_bps", "n"]]
            .sort_values(["era", "sharpe"], ascending=[True, False])
            .reset_index(drop=True))
