"""Checks on the point-in-time instrument map."""
from __future__ import annotations

import pandas as pd


def overlapping_intervals(mapping: pd.DataFrame) -> pd.DataFrame:
    """Rows where one symbol has two live identities at once - always a bug."""
    bad = []
    for sym, g in mapping.groupby("symbol"):
        g = g.sort_values("valid_from")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if pd.isna(a["valid_to"]) or pd.to_datetime(a["valid_to"]) > pd.to_datetime(b["valid_from"]):
                bad.append({"symbol": sym, "first": a["cisin"], "second": b["cisin"]})
    return pd.DataFrame(bad)


def coverage(mapping: pd.DataFrame, panel: pd.DataFrame) -> float:
    """Share of panel instrument-days resolving to a canonical identity."""
    if panel.empty:
        return float("nan")
    return float(panel["cisin"].isin(mapping["cisin"]).mean())
