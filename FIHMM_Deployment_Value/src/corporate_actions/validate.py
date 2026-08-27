"""Verify parsed adjustment factors against observed price ratios."""
from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCE = 0.02          # a parsed factor must explain the observed jump


def verify_factor(observed_ratio: float, parsed_factor: float) -> bool:
    if not np.isfinite(observed_ratio) or not np.isfinite(parsed_factor):
        return False
    return abs(observed_ratio / parsed_factor - 1.0) <= TOLERANCE


def validation_report(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per action, with a PASS/FAIL on the price-ratio check."""
    out = actions.copy()
    out["verified"] = [
        verify_factor(r.get("observed_ratio", np.nan), r.get("factor", np.nan))
        for _, r in out.iterrows()
    ]
    return out
