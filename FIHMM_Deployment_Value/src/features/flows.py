"""Flow features, and the measured reporting-lag curve.

The market-level flow variable (manuscript eq. 2) is

    NF_t = (buy value - sell value)_t / mean(gross value)_{t-250..t-1}

scaled by a TRAILING mean ending the previous day, so the scaling itself
carries no look-ahead. Days with no flow observation are excluded rather than
entered as zero.

Reads data/derived only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utilities import dataset, load_config


def market_net_flow() -> pd.DataFrame:
    """Daily market-level flow frame: net, gross, trailing mean, and NF."""
    return pd.read_parquet(dataset("daily_features"))


def reporting_lag_curve() -> dict[int, float]:
    """Cumulative % of trade VALUE reported by the close of t+k.

    Measured from the reporting date each record carries alongside its trade
    date - not assumed. Value-weighted, with a CONSERVATIVE denominator: value
    whose report date falls off the exchange calendar is counted as never
    reported by t+k rather than being dropped.

    Large trades arrive LATER, so the value-weighted curve is the binding one
    and a count-weighted curve would flatter the contemporaneous specification.
    """
    d = pd.read_csv(dataset("reporting_lag"))
    return {int(r.lag_days): float(r.cumulative_pct_value_reported)
            for r in d.itertuples()}


def deployable_lag() -> int:
    return int(load_config("model_config")["flow"]["reporting_lag"])


def flow_tilt_features(frame: pd.DataFrame | None = None) -> np.ndarray:
    """X_t = [ln(VIX_t / 20), NF+(t-2), NF-(t-2)].

    The negative part is the single permitted asymmetry, fixed before the
    engine ran. NF enters at the deployable lag, never contemporaneously.
    """
    if frame is None:
        frame = market_net_flow()
    vix = frame["india_vix"].to_numpy(float)
    nf = frame["nf"].to_numpy(float)
    lag = deployable_lag()
    nf_lagged = np.r_[np.zeros(lag), nf[:-lag]]
    return np.column_stack([np.log(vix / 20.0),
                            np.maximum(nf_lagged, 0.0),
                            np.minimum(nf_lagged, 0.0)])
