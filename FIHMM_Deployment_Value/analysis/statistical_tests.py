"""The tests every gate in the manuscript rests on.

Sign convention, fixed once and used everywhere: a NEGATIVE Diebold-Mariano
statistic means the first argument (the tilted model) scores BETTER. The
pre-registered density gate is t <= -2.0.

The nested-model problem is handled explicitly rather than assumed away. See
`clark_west` and the note in config/model_config.yaml.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from src.common.random_seed import bootstrap_generator
from src.common.utilities import load_config

_CFG = load_config("model_config")
NW_LAGS = _CFG["inference"]["nw_lags"]


def newey_west_var(d: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey-West long-run variance of the MEAN of d (Bartlett kernel)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan
    d0 = d - d.mean()
    v = float((d0 * d0).mean())
    for l in range(1, lags + 1):
        if l >= n:
            break
        v += 2.0 * (1.0 - l / (lags + 1)) * float((d0[l:] * d0[:-l]).mean())
    return max(v, 1e-30) / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray,
                    lags: int = NW_LAGS) -> tuple[float, int]:
    """DM statistic on a paired loss differential. Negative => A better.

    Valid for NON-nested comparisons. For nested ones the loss differential
    degenerates under the null and this statistic is not distribution-free;
    pair it with `block_bootstrap_p` and `clark_west`.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan, len(d)
    return float(d.mean() / np.sqrt(newey_west_var(d, lags))), len(d)


def block_bootstrap_p(d: np.ndarray, n_boot: int | None = None,
                      mean_block: int | None = None,
                      alternative: str = "less") -> tuple[float, float]:
    """Stationary bootstrap p-value for H0: E[d] = 0.

    Distribution-free in the sense that it does not lean on a normal reference,
    which is what the nested gate cannot justify. It still tests EQUAL
    predictive ability, not "the small model is correctly specified" - those
    are different nulls, and the difference is stated in the manuscript.

    Returns (p_value, observed_mean).
    """
    cfg = _CFG["inference"]["block_bootstrap"]
    n_boot = n_boot or cfg["n_boot"]
    mean_block = mean_block or cfg["mean_block"]
    rng = bootstrap_generator()

    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    obs = float(d.mean())
    centred = d - obs                       # impose the null
    p_restart = 1.0 / mean_block

    # Stationary (Politis-Romano) bootstrap: geometric block lengths.
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = int(rng.integers(0, n))
        for k in range(n):
            idx[k] = i
            i = int(rng.integers(0, n)) if rng.random() < p_restart else (i + 1) % n
        means[b] = centred[idx].mean()

    if alternative == "less":
        p = float((means <= obs).mean())
    elif alternative == "greater":
        p = float((means >= obs).mean())
    else:
        p = float((np.abs(means) >= abs(obs)).mean())
    return p, obs


def clark_west(y: np.ndarray, f_restricted: np.ndarray,
               f_unrestricted: np.ndarray,
               lags: int = NW_LAGS) -> tuple[float, float, int]:
    """Clark-West statistic for NESTED point forecasts.

    Under the null that the small model is correct, the large model still
    estimates parameters that are zero, and that estimation noise inflates its
    loss. A naive DM test therefore under-rejects. CW adds the noise term back:

        CW_t = (y - f1)^2 - (y - f2)^2 + (f1 - f2)^2

    SIGN CONVENTION IS OPPOSITE TO DM: positive CW favours the LARGER model.
    Returns (t_statistic, one_sided_p, n).
    """
    y = np.asarray(y, float)
    f1 = np.asarray(f_restricted, float)
    f2 = np.asarray(f_unrestricted, float)
    m = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[m], f1[m], f2[m]
    cw = (y - f1) ** 2 - (y - f2) ** 2 + (f1 - f2) ** 2
    t = float(cw.mean() / np.sqrt(newey_west_var(cw, lags)))
    return t, float(stats.norm.sf(t)), len(cw)


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Unconditional coverage test. Returns (observed_rate, p_value)."""
    h = np.asarray(hits, float)
    h = h[np.isfinite(h)]
    n, x = len(h), float(h.sum())
    if n == 0:
        return np.nan, np.nan
    ph = x / n

    def ll(q: float) -> float:
        return (n - x) * np.log(max(1 - q, 1e-12)) + x * np.log(max(q, 1e-12))

    lr = -2.0 * (ll(p) - ll(max(ph, 1e-12)))
    return float(ph), float(stats.chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> float:
    """Independence test on VaR exceedances. Returns p_value."""
    h = np.asarray(hits, float)
    h = h[np.isfinite(h)].astype(int)
    if len(h) < 30:
        return np.nan
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)

    def sl(n_: int, p_: float) -> float:
        return n_ * np.log(max(p_, 1e-12))

    l1 = sl(n00, 1 - p01) + sl(n01, p01) + sl(n10, 1 - p11) + sl(n11, p11)
    l0 = sl(n00 + n10, 1 - p1) + sl(n01 + n11, p1)
    return float(stats.chi2.sf(-2.0 * (l0 - l1), 1))


def paired_t(diff: np.ndarray) -> tuple[float, int]:
    """Plain paired t on a difference series, for episode-level comparisons."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, len(d)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)
