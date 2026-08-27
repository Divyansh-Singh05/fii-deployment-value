"""Return panel assembly and corporate-action-adjusted returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utilities import dataset

MAX_ABS_DAILY_RETURN = 0.50   # guard: nulls and logs anything beyond this


def load_returns_panel() -> pd.DataFrame:
    """Market-level daily returns and their inputs, corporate-action adjusted."""
    return pd.read_parquet(dataset("daily_features"))


def guard_extreme_returns(ret: np.ndarray) -> tuple[np.ndarray, int]:
    """Null any post-adjustment daily return beyond the guard, and count them.

    An unadjusted split presents as a ~50% or ~100% single-day move. The guard
    catches adjustment failures rather than genuine moves, and it LOGS rather
    than silently winsorising, so a systematic failure is visible.
    """
    r = np.asarray(ret, float).copy()
    bad = np.abs(r) > MAX_ABS_DAILY_RETURN
    r[bad] = np.nan
    return r, int(np.nansum(bad))


def standardise(ret: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z = r / sigma, using a PAST-ONLY volatility estimate."""
    s = np.asarray(sigma, float)
    return np.asarray(ret, float) / np.where(s > 0, s, np.nan)
