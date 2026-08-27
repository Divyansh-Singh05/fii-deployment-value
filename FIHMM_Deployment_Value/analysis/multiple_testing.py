"""Pricing the search.

The pre-registration limits the damage - bars were fixed in advance and misses
were reported rather than re-specified - but a pre-registration is not a
multiplicity correction. This module supplies the correction.

One honesty note carried through to the manuscript: the four engine gates were
a SEQUENTIAL search, not a fixed family. Each threshold was fixed before its
own run, but the second was written after the first had passed. Treating them
as a family of four therefore UNDERSTATES the search cost. `holm` and
`bonferroni` report the family-of-four number; `sequential_search_note`
returns the caveat so no caller can drop it silently.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bonferroni(p: float, m: int) -> float:
    """Bonferroni-adjusted p for one hypothesis in a family of m."""
    return float(min(1.0, p * m))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed as supplied."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # enforce monotonicity
        out[k] = running
    return out


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          sharpe_variance: float | None = None) -> tuple[float, float]:
    """Bailey & Lopez de Prado deflated Sharpe ratio.

    Deflates an observed Sharpe by the highest Sharpe one would expect from
    `n_trials` independent backtests of the same length under a null of no
    skill, then tests the observed value against that benchmark.

    `sharpe` and the return are in the SAME periodicity as `n_obs`.
    Returns (expected_max_sharpe_under_null, dsr_probability).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    var = 1.0 / n_obs if sharpe_variance is None else sharpe_variance
    sd = np.sqrt(var)

    # Expected maximum of n_trials standard normals (Euler-Mascheroni form).
    gamma = 0.5772156649015329
    if n_trials == 1:
        e_max_z = 0.0
    else:
        e_max_z = ((1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
                   + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr_star = float(e_max_z * sd)

    # Test the observed Sharpe against sr_star under non-normal returns.
    num = (sharpe - sr_star) * np.sqrt(n_obs - 1)
    den = np.sqrt(1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2)
    if den <= 0 or not np.isfinite(den):
        return sr_star, np.nan
    return sr_star, float(stats.norm.cdf(num / den))


def probability_of_backtest_overfitting(is_ranks: np.ndarray,
                                        n_trials: int) -> float:
    """PBO: share of splits where the in-sample best is below-median OOS.

    `is_ranks` is the OOS rank (0 = worst) achieved by the in-sample winner in
    each combinatorial split.
    """
    r = np.asarray(is_ranks, float)
    r = r[np.isfinite(r)]
    if len(r) == 0 or n_trials < 2:
        return np.nan
    return float((r / (n_trials - 1) < 0.5).mean())


def sequential_search_note() -> str:
    """The caveat that must travel with any correction reported here."""
    return (
        "The four engine gates were a sequential search, not a fixed family: "
        "each threshold was fixed before its own run, but the second was "
        "written after the first had passed. A family-of-four adjustment "
        "therefore understates the search cost. Eight strategy books were also "
        "examined and are not priced; none passed, so no positive claim rests "
        "on them."
    )
