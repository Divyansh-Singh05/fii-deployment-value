"""A4 - discrete-time hazard model of flow-episode termination.

Walk-forward with yearly refits, right-censoring handled, forecasting whether
an episode ends today. The model has GENUINE skill: it beats the duration-only
baseline decisively and discriminates better out of sample than in, which rules
out memorisation.

The decision layer is where it dies. A three-line rule - act once an episode is
a few days old - outperforms it on the episodes both act upon. Note what a
weaker design reports: each rule compared against zero separately, with the
model declared the better of two positives. Only the PAIRED comparison on
shared events recovers the ordering.

Reads data/derived/hazard_predictions.parquet only.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.common.utilities import dataset


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """Columns: cisin, TR_DATE, era, k, year, y, p_model, p_km, p_const."""
    return pd.read_parquet(dataset("hazard"))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-corrected."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hazard_summary() -> pd.DataFrame:
    """Reproduce manuscript Table 3.

    Discrimination is recomputed from the predictions. The decision-layer rows
    are episode-level: they come from the producing stage's episode book, which
    prices the anticipation gain of acting on each rule.
    """
    d = load_predictions()
    k1 = d[d["k"] == 1]
    rows = [
        {"metric": "auc_pooled",
         "gbdt_model": round(auc(k1["y"], k1["p_model"]), 3),
         "age_heuristic": round(auc(k1["y"], k1["p_km"]), 3)},
        {"metric": "auc_training",
         "gbdt_model": round(auc(k1[k1.era == "TRAIN"]["y"],
                                 k1[k1.era == "TRAIN"]["p_model"]), 3),
         "age_heuristic": None},
        {"metric": "auc_test",
         "gbdt_model": round(auc(k1[k1.era == "TEST"]["y"],
                                 k1[k1.era == "TEST"]["p_model"]), 3),
         "age_heuristic": None},
        {"metric": "paired_logloss_stat", "gbdt_model": 14.42, "age_heuristic": None},
        {"metric": "test_anticipation_gain_bp", "gbdt_model": -18, "age_heuristic": 16},
        {"metric": "paired_difference_bp", "gbdt_model": -31, "age_heuristic": None},
        {"metric": "paired_difference_t", "gbdt_model": -1.85, "age_heuristic": None},
        {"metric": "common_episodes", "gbdt_model": 402, "age_heuristic": None},
    ]
    return pd.DataFrame(rows)


def calibration(n_bins: int = 10, era: str | None = None) -> pd.DataFrame:
    """Reliability table and Brier decomposition.

    The manuscript states plainly that no calibration statistics were reported
    for A4, and that discrimination alone is not evidence the latent mechanism
    is well specified. This supplies the missing diagnostic. It is NOT a
    manuscript number.
    """
    d = load_predictions()
    d = d[d["k"] == 1]
    if era:
        d = d[d["era"] == era]
    p = d["p_model"].to_numpy(float)
    y = d["y"].to_numpy(float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])

    rows, base, rel, res = [], y.mean(), 0.0, 0.0
    for b in range(n_bins):
        s = idx == b
        if not s.any():
            continue
        nb, pb, ob = int(s.sum()), float(p[s].mean()), float(y[s].mean())
        rel += nb * (pb - ob) ** 2
        res += nb * (ob - base) ** 2
        rows.append({"bin": b, "n": nb, "mean_forecast": round(pb, 4),
                     "observed_rate": round(ob, 4),
                     "gap": round(pb - ob, 4)})

    out = pd.DataFrame(rows)
    n = len(y)
    out.attrs.update(brier=float(np.mean((p - y) ** 2)),
                     reliability=rel / n, resolution=res / n,
                     uncertainty=float(base * (1 - base)), n=n)
    return out
