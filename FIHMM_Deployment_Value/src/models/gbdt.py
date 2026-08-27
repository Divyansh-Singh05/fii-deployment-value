"""A6 - the participant-composition feature block, scored by gradient boosting.

WHAT THIS BLOCK CAN AND CANNOT MEASURE. Masked participant identifiers are
re-minted at monthly boundaries, so every composition feature is computed
WITHIN-DAY: how concentrated or dispersed the participation behind a day's
activity is. The block does NOT track a participant across days and no result
here depends on doing so.
"""
from __future__ import annotations

import numpy as np


def herfindahl(values: np.ndarray) -> float:
    """Concentration of one day's participation in one instrument."""
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    share = v / v.sum()
    return float((share ** 2).sum())


def composition_features(day_values: np.ndarray) -> dict[str, float]:
    """Within-day composition descriptors for a single instrument-day."""
    v = np.abs(np.asarray(day_values, float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return {"hhi": np.nan, "n_participants": 0, "top1_share": np.nan,
                "entropy": np.nan}
    share = v / v.sum()
    return {
        "hhi": float((share ** 2).sum()),
        "n_participants": int(v.size),
        "top1_share": float(share.max()),
        "entropy": float(-(share * np.log(share)).sum()),
    }


def information_coefficient(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Rank IC between a signal and a forward return."""
    from scipy import stats
    s = np.asarray(signal, float)
    r = np.asarray(forward_return, float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 3:
        return np.nan
    return float(stats.spearmanr(s[m], r[m]).statistic)


def quintile_spread(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Top-minus-bottom quintile forward-return spread, in basis points."""
    s = np.asarray(signal, float)
    r = np.asarray(forward_return, float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 10:
        return np.nan
    s, r = s[m], r[m]
    lo, hi = np.quantile(s, [0.2, 0.8])
    return float((r[s >= hi].mean() - r[s <= lo].mean()) * 1e4)
