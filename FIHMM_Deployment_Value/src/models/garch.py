"""The GJR-GARCH baseline.

A BASELINE, not a contribution. It exists so the flow tilt can be scored
against a well-specified conditional-variance model rather than a weak one.
Because it already absorbs asymmetric volatility persistence, it is the
baseline against which the flow signal provides no incremental value - which
is the paper's L3 finding, not a defect of the baseline.

Frozen BEFORE its model counterpart so the comparison cannot be tuned.
"""
from __future__ import annotations

import numpy as np


def gjr_recursion(ret: np.ndarray, omega: float, alpha: float,
                  gamma: float, beta: float, burn: int = 60) -> np.ndarray:
    """GJR(1,1,1) conditional variance, strictly past-only.

        h_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * h_{t-1}

    The leverage term `gamma` is what makes this baseline hard to beat with a
    flow tilt: bad news already raises the forecast variance.
    """
    r = np.asarray(ret, float)
    n = len(r)
    h = np.full(n, np.nan)
    if n <= burn:
        return h
    h[burn - 1] = float(np.var(r[:burn]))
    for t in range(burn, n):
        prev = r[t - 1]
        shock = (alpha + (gamma if prev < 0 else 0.0)) * prev ** 2
        h[t] = omega + shock + beta * h[t - 1]
    return h


def sigma(ret: np.ndarray, **kw) -> np.ndarray:
    return np.sqrt(gjr_recursion(ret, **kw))
