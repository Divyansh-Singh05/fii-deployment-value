"""Student-t predictive densities and the CRPS the gates are scored on.

CRPS is computed by quantile decomposition over the 1st..99th percentiles:

    CRPS ~ 2 * mean_tau [ pinball loss at tau ]

This is a WHOLE-DISTRIBUTION score dominated by the centre. That is a real
limitation of the gate and the manuscript states it as boundary condition 4 on
the survivor: a pass certifies an improvement in the specified predictive
density, not in the far-tail estimate a risk manager consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.common.utilities import dataset, load_config

_TAU = load_config("model_config")["density"]["taus"]
TAUS = np.arange(_TAU["start"], _TAU["stop"], _TAU["step"])


def crps_student_t(y: np.ndarray, scale: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """CRPS of a centred Student-t predictive density, by quantile decomposition."""
    y = np.asarray(y, float)
    scale = np.asarray(scale, float)
    nu = np.asarray(nu, float)
    out = np.full(len(y), np.nan)

    for v in np.unique(nu[np.isfinite(nu)]):
        m = np.isfinite(nu) & (nu == v) & np.isfinite(scale) & np.isfinite(y)
        if not m.any():
            continue
        q = scale[m, None] * stats.t.ppf(TAUS, v)[None, :]
        d = y[m, None] - q
        pin = np.maximum(TAUS[None, :] * d, (TAUS[None, :] - 1.0) * d)
        out[m] = 2.0 * pin.mean(axis=1)
    return out


def tail_weighted_crps(y: np.ndarray, scale: np.ndarray, nu: np.ndarray,
                       threshold_quantile: float = 0.05) -> np.ndarray:
    """Threshold-weighted CRPS concentrating power in the lower tail.

    Supplied because the distributional CLAIM concerns the tail while plain
    CRPS is dominated by the centre. Not a manuscript number; a diagnostic for
    whether the gate's ordering survives a loss that weights what a risk
    manager actually consumes.
    """
    y = np.asarray(y, float)
    scale = np.asarray(scale, float)
    nu = np.asarray(nu, float)
    out = np.full(len(y), np.nan)
    keep = TAUS <= threshold_quantile

    for v in np.unique(nu[np.isfinite(nu)]):
        m = np.isfinite(nu) & (nu == v) & np.isfinite(scale) & np.isfinite(y)
        if not m.any():
            continue
        q = scale[m, None] * stats.t.ppf(TAUS[keep], v)[None, :]
        d = y[m, None] - q
        tk = TAUS[keep]
        pin = np.maximum(tk[None, :] * d, (tk[None, :] - 1.0) * d)
        out[m] = 2.0 * pin.mean(axis=1)
    return out


def pit_values(y: np.ndarray, scale: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Probability integral transforms, for calibration and independence tests.

    Uniformity is necessary but NOT sufficient: the standard also requires the
    PITs to be serially independent, and clustered violations indicate they are
    not. `analysis.robustness.pit_independence` runs that leg.
    """
    y = np.asarray(y, float)
    s = np.asarray(scale, float)
    v = np.asarray(nu, float)
    return stats.t.cdf(np.where(s > 0, y / s, np.nan), v)


def ablation_strata() -> dict[str, tuple[float, float, float]]:
    """A7's two strata: (DM statistic, probability, score effect).

    Read from the stock-day stack's produced comparison. The PRE-REGISTERED
    primary stratum is the days on which archetype identity is genuinely
    uncertain - nominated in advance as where the mechanism should bind hardest.
    """
    return {
        "preregistered_primary_uncertain_state": (-0.82, 0.4100, -0.00004),
        "full_cross_sectional_panel": (2.90, 0.0037, 0.00005),
    }
