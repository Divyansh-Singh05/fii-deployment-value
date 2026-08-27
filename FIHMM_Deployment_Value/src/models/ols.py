"""Equation (1): the existence and availability regression.

    z_{i,t+1}^2 = a_i + b_t + g * share_{i,t-L} + controls + e

Instrument and date fixed effects, log turnover controlled. Date fixed effects
absorb all market-wide variation by construction, so a market-level variable
can enter only as an interaction - a restriction imposed rather than worked
around.

Standard errors are clustered on DATE. Two-way and bootstrap alternatives are
reported alongside by `robustness_estimators`, because date clustering protects
the cross-section but not within-instrument serial dependence, and this panel
has both.

Reads data/derived/panel_analysis.parquet only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common.utilities import dataset


@dataclass
class RegressionResult:
    coef: float
    tstat: dict[str, float] = field(default_factory=dict)
    nobs: int = 0
    lag: int = 0


@lru_cache(maxsize=1)
def _panel() -> tuple[pd.DataFrame, float]:
    """The 577,245-row analysis panel and its winsorisation constant."""
    d = pd.read_parquet(dataset("panel"))
    meta = json.loads(dataset("panel_meta").read_text())
    return d, float(meta["clip"])


def _demean2(x: np.ndarray, day: np.ndarray, stk: np.ndarray,
             iters: int = 20, tol: float = 1e-10) -> np.ndarray:
    """Alternating projections: sweep out instrument AND date means.

    Equivalent to including both fixed-effect sets as dummies, without
    materialising a 1,028 + 3,514 column design matrix.
    """
    v = np.asarray(x, float).copy()
    for _ in range(iters):
        before = v.copy()
        v -= np.bincount(day, v, minlength=day.max() + 1)[day] / np.bincount(
            day, minlength=day.max() + 1)[day]
        v -= np.bincount(stk, v, minlength=stk.max() + 1)[stk] / np.bincount(
            stk, minlength=stk.max() + 1)[stk]
        if np.max(np.abs(v - before)) < tol:
            break
    return v


@lru_cache(maxsize=32)
def _design_cached(lag: int, era: str | None = None):
    d, clip = _panel()
    mask = None if era is None else (d["era"].to_numpy() == era)
    y_raw = np.clip(d["z_h1"].to_numpy(float), -clip, clip) ** 2
    # `vol` is ALREADY log(turnover / trailing-60d mean); do not log it again.
    xs = list(_lagged(lag))

    ok = np.ones(len(d), bool) if mask is None else mask.copy()
    ok &= np.isfinite(y_raw) & np.all([np.isfinite(c) for c in xs], axis=0)

    _, day = np.unique(d["TR_DATE"].to_numpy()[ok], return_inverse=True)
    _, stk = np.unique(d["cisin"].to_numpy()[ok], return_inverse=True)
    y = _demean2(y_raw[ok], day, stk)
    X = np.column_stack([np.ones(int(ok.sum()))]
                        + [_demean2(c[ok], day, stk) for c in xs])
    return y, X, day, stk, int(ok.sum())


def _design(lag: int, era: str | None = None):
    return _design_cached(lag, era)


def _meat(X: np.ndarray, e: np.ndarray, g: np.ndarray) -> np.ndarray:
    M = np.zeros((X.shape[1],) * 2)
    for k in np.unique(g):
        m = g == k
        z = X[m].T @ e[m]
        M += np.outer(z, z)
    return M


def _fit(lag: int, era: str | None = None) -> tuple[float, dict, int]:
    y, X, day, stk, n = _design_cached(lag, era)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)

    V_d = XtXi @ _meat(X, e, day) @ XtXi
    V_s = XtXi @ _meat(X, e, stk) @ XtXi
    V_w = XtXi @ ((X * (e ** 2)[:, None]).T @ X) @ XtXi
    # Cameron-Gelbach-Miller: each (instrument, date) cell holds exactly one
    # observation, so the intersection clustering IS White.
    V_2 = V_d + V_s - V_w

    out = {}
    for lab, V in (("date", V_d), ("instrument", V_s),
                   ("two-way", V_2), ("white", V_w)):
        se = np.sqrt(max(V[1, 1], 0.0))
        out[lab] = float(b[1] / se) if se > 0 else np.nan
    return float(b[1]), out, n


@lru_cache(maxsize=16)
def _lagged(lag: int) -> tuple[np.ndarray, np.ndarray]:
    """share and vol at `lag` trading days, shifted WITHIN instrument.

    Materialised lags (0, 2) are read straight from the panel. Any other lag is
    built here by shifting within `cisin`, so a lag never reaches across a gap
    in one instrument's history into another's.
    """
    d, _ = _panel()
    if lag == 0:
        return d["share"].to_numpy(float), d["vol"].to_numpy(float)
    if lag == 2:
        return d["share_L2"].to_numpy(float), d["vol_L2"].to_numpy(float)
    g = d.groupby("cisin", sort=False)
    return (g["share"].shift(lag).to_numpy(float),
            g["vol"].shift(lag).to_numpy(float))


def panel_regression_share(lag: int = 0) -> RegressionResult:
    """Estimate equation (1) with the share taken at `lag` trading days."""
    out = RegressionResult(coef=np.nan, lag=lag)
    for name in ("FULL", "TRAIN", "TEST"):
        coef, ts, n = _fit(lag, None if name == "FULL" else name)
        out.tstat[name] = ts["date"]
        if name == "FULL":
            out.coef, out.nobs = coef, n
    return out


def variance_estimators(lag: int) -> dict:
    """All four analytic estimators for one lag, full sample."""
    _, ts, _ = _fit(lag, None)
    return ts


def lag_profile(max_lag: int = 5) -> pd.DataFrame:
    """Coefficient and t-statistic across information lags 0..max_lag.

    This is the availability argument in one object: the SAME specification,
    varying only the lag at which the predictor becomes knowable.
    """
    rows = []
    for lag in range(max_lag + 1):
        coef, ts, n = _fit(lag, None)
        rows.append({"lag": lag, "coef": coef, "t_date": ts["date"],
                     "t_two_way": ts["two-way"], "n": n})
    return pd.DataFrame(rows)


def robustness_estimators() -> pd.DataFrame:
    from analysis.robustness import variance_estimator_grid
    return variance_estimator_grid()
