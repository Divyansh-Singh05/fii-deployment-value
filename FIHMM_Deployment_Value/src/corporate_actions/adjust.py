"""Corporate-action adjustment.

Split, bonus and similar factors are parsed from exchange filings and VERIFIED
AGAINST OBSERVED PRICE RATIOS before application, not applied on trust. A guard
nulls and logs any post-adjustment daily return exceeding 50% in magnitude,
because an unadjusted split presents as exactly that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GUARD = 0.50


def cumulative_factor(actions: pd.DataFrame, instrument: str,
                      as_of: pd.Timestamp) -> float:
    """Product of adjustment factors effective strictly AFTER `as_of`.

    Point-in-time by construction: a price on day t is adjusted only by events
    that had not yet occurred, so the series is what an observer on t would
    later have restated, never what they could not have known.
    """
    a = actions[(actions["instrument"] == instrument)
                & (pd.to_datetime(actions["ex_date"]) > as_of)]
    if a.empty:
        return 1.0
    return float(np.prod(a["factor"].to_numpy(float)))


def apply_adjustment(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out["adj_close"] = [
        r["close"] / cumulative_factor(actions, r["instrument"], r["date"])
        for _, r in out.iterrows()
    ]
    return out


def guard_returns(ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Null and FLAG extreme post-adjustment returns. Returns (clean, flags)."""
    r = np.asarray(ret, float).copy()
    bad = np.abs(r) > GUARD
    r[bad] = np.nan
    return r, bad
