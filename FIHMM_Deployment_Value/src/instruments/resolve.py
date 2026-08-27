"""Point-in-time instrument identity with issuer-bounded closure.

Identifier changes strand history at exactly the events that matter most - the
corporate events a volatility study most wants to see. A point-in-time map
resolves each instrument to a canonical entity AS OF each date, and closure is
bounded by issuer so that two unrelated instruments sharing a recycled symbol
are never merged.
"""
from __future__ import annotations

import pandas as pd


def resolve_as_of(mapping: pd.DataFrame, symbol: str,
                  as_of: pd.Timestamp) -> str | None:
    """Canonical identity of `symbol` as it stood on `as_of`."""
    m = mapping[(mapping["symbol"] == symbol)
                & (pd.to_datetime(mapping["valid_from"]) <= as_of)
                & ((mapping["valid_to"].isna())
                   | (pd.to_datetime(mapping["valid_to"]) > as_of))]
    if m.empty:
        return None
    return str(m.sort_values("valid_from").iloc[-1]["cisin"])


def closure(mapping: pd.DataFrame) -> pd.DataFrame:
    """Transitive closure of identifier changes, BOUNDED BY ISSUER.

    Without the issuer bound, closure over a recycled symbol chains two
    unrelated companies into one synthetic instrument with a spurious
    fourteen-year history.
    """
    return (mapping.sort_values(["issuer", "valid_from"])
                   .groupby("issuer", group_keys=False)
                   .apply(lambda g: g.assign(cisin=g["cisin"].ffill())))
