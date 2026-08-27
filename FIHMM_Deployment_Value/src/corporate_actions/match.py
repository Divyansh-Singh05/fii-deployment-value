"""Match exchange corporate-action filings to instruments and dates."""
from __future__ import annotations

import pandas as pd


def match_actions(filings: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Join filings onto the canonical instrument identity.

    Unmatched filings are RETAINED with a null instrument rather than dropped,
    so `validate.py` can report the match rate. Silently dropping them would
    hide exactly the adjustment failures the guard exists to catch.
    """
    return filings.merge(universe[["instrument", "symbol"]], on="symbol",
                         how="left", validate="many_to_one")


def match_rate(matched: pd.DataFrame) -> float:
    if matched.empty:
        return float("nan")
    return float(matched["instrument"].notna().mean())
