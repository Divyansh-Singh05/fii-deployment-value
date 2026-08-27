"""Past-only volatility estimators: EWMA and the GJR-GARCH baseline.

These are BASELINES, not contributions. They exist so the flow tilt can be
scored against something a desk would actually be running. Both are
constructed and frozen BEFORE their model counterparts so the comparison
cannot be tuned.
"""
from __future__ import annotations

import numpy as np

from src.common.utilities import load_config

_CFG = load_config("model_config")["ewma"]


def ewma_variance(ret: np.ndarray, lam: float | None = None,
                  burn: int | None = None) -> np.ndarray:
    """RiskMetrics-style EWMA variance, seeded on a burn-in window.

    Strictly past-only: var[t] uses ret[t] and var[t-1], and the forecast for
    t+1 is var[t]. Nothing at t+1 enters.
    """
    lam = _CFG["lam"] if lam is None else lam
    burn = _CFG["burn"] if burn is None else burn
    r = np.asarray(ret, float)
    n = len(r)
    var = np.full(n, np.nan)
    if n <= burn:
        return var
    var[burn - 1] = float(np.var(r[:burn]))
    for i in range(burn, n):
        var[i] = lam * var[i - 1] + (1.0 - lam) * r[i] ** 2
    return var


def ewma_sigma(ret: np.ndarray, **kw) -> np.ndarray:
    return np.sqrt(ewma_variance(ret, **kw))


def gjr_scored_losses() -> "pd.DataFrame":
    """The frozen GJR-baseline CRPS columns, as scored by the producing stage.

    Read rather than refitted: refitting would silently change the baseline the
    controlled pair holds fixed, and the whole point of that pair is that ONLY
    the baseline differs between its two columns.
    """
    import pandas as pd
    from src.common.utilities import dataset
    d = pd.read_parquet(dataset("analysis_dataset"))
    return d[["date", "crps_G1_vix", "crps_G2a_vix_nf"]]
