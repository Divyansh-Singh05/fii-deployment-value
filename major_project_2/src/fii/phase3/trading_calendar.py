"""
SHARED TRADING CALENDAR  —  audit item 3

One definition of "a trading day", used by BOTH the estimator (C1) and the
filter (C2), so the two cannot drift apart again.

The calendar is defined by PRICES, not by the FII feed. The custodian feed
carries rows stamped on non-trading days -- 8,256 on weekends plus 910 on
market holidays, 97% of them in 2016-17. A trade reported on a Saturday has no
day-t price, so it can carry no forward return, and admitting such a date
poisons every multi-day outcome window that spans it.

This module exists because item 3's fix only holds if the elapsed-time unit is
identical on both sides. C2 previously built this inline; C1 had no notion of a
calendar at all, which is precisely how the per-observation / per-trading-day
mismatch arose.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from fii.paths import VALIDATION_DATA

MIN_PRICED = 50          # a date counts once this many modelled stocks price


def trading_days(cut: dt.date | None = None) -> list[dt.date]:
    """Ordered trading dates. `cut` truncates the world, for the C9 audit."""
    rp = pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet",
                         columns=["date", "ret_adj"])
    if cut is not None:
        rp = rp.filter(pl.col("date") <= cut)
    d = (rp.group_by("date").agg(pl.col("ret_adj").is_finite().sum().alias("n"))
           .filter(pl.col("n") >= MIN_PRICED)
           .sort("date")["date"])
    return d.to_list()


def day_index(dates, calendar: list[dt.date]) -> np.ndarray:
    """Position of each date on the calendar; -1 if it is not a trading day."""
    pos = {d: i for i, d in enumerate(calendar)}
    return np.array([pos.get(d, -1) for d in dates], dtype=np.int64)


def gaps_from_frame(w: pl.DataFrame, calendar: list[dt.date],
                    group: str = "cisin", datecol: str = "TR_DATE"
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Elapsed trading-day gaps for a frame already sorted by (group, date).

    Returns (gaps, lengths). gaps[i] is 0 at each sequence start, otherwise the
    number of trading days since that stock's previous observation -- the k in
    A^k.
    """
    from fii.models.gap_aware_hmm import trading_day_gaps
    di = day_index(w[datecol].to_list(), calendar)
    if (di < 0).any():
        raise ValueError(
            f"{int((di < 0).sum())} rows sit on non-trading dates; filter to "
            "the calendar before computing gaps")
    lens = w.group_by(group, maintain_order=True).len()["len"].to_numpy()
    return trading_day_gaps(di, list(lens)), lens
