"""A5 - the mechanical long/short concentration book.

Carries NO fitted model, so what L5 prices is the DATA rather than an
architecture. Timing is specified as an equation rather than described: a
signal formed at the close of day t is traded at the close of t+L and earns the
close-to-close return of t+L+1, with L = 1.
"""
from __future__ import annotations

import numpy as np

LAG = 1


def align_signal_to_return(signal: np.ndarray, ret: np.ndarray,
                           lag: int = LAG) -> tuple[np.ndarray, np.ndarray]:
    """Shift so that signal[t] earns ret[t + lag + 1]. No look-ahead survives."""
    s = np.asarray(signal, float)
    r = np.asarray(ret, float)
    n = len(s)
    shift = lag + 1
    return s[:n - shift], r[shift:]


def gross_pnl(weights: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return np.nansum(np.asarray(weights, float) * np.asarray(ret, float), axis=-1)


def turnover(weights: np.ndarray) -> np.ndarray:
    """One-way turnover per rebalance."""
    w = np.asarray(weights, float)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    return np.nansum(np.abs(w - prev), axis=1) / 2.0


def breakeven_cost_bps(gross: np.ndarray, turn: np.ndarray) -> float:
    """c* = E[gross pnl] / E[turnover], in basis points (manuscript eq. 4).

    Reported INSTEAD of a net figure at an assumed cost. The breakeven is a
    property of the strategy; a net Sharpe is a property of the assumption.
    """
    g = np.asarray(gross, float)
    t = np.asarray(turn, float)
    m = np.isfinite(g) & np.isfinite(t)
    if not m.any() or np.mean(t[m]) <= 0:
        return np.nan
    return float(np.mean(g[m]) / np.mean(t[m]) * 1e4)


def net_sharpe(gross: np.ndarray, turn: np.ndarray, cost_bps: float,
               periods: int = 252) -> float:
    """net = gross - c * turnover, exactly. The engine's cost gate enforces it."""
    g = np.asarray(gross, float) - cost_bps * 1e-4 * np.asarray(turn, float)
    g = g[np.isfinite(g)]
    if g.size < 2 or g.std(ddof=1) == 0:
        return np.nan
    return float(g.mean() / g.std(ddof=1) * np.sqrt(periods))
