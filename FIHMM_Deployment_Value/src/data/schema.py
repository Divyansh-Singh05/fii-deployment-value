"""The source record's schema, and what each field is allowed to support.

Declared here so `validate_source` can check an extract before any computation
touches it, and so a reader can see exactly which fields the programme uses.
"""
from __future__ import annotations

from dataclasses import dataclass

TRANSACTION_FIELDS = {
    "TR_DATE":            "date the trade occurred",
    "RFDE_RPT_DT":        "date the custodian REPORTED it - the availability clock",
    "cisin":              "canonical instrument identity, post closure resolution",
    "RFDE_INSTR_TYPE":    "instrument type; the programme keeps REG_DL_INSTR_EQ",
    "TR_TYPE":            "side; 1 = buy",
    "VALUE_INR":          "traded value",
    "RATE":               "trade price; must be > 0",
    "participant_masked": "masked participant id - RE-MINTED MONTHLY",
    "subaccount_masked":  "masked sub-account id - RE-MINTED MONTHLY",
    "broker_id":          "broker; the control population for the persistence test",
}

# Fields that must never be used to link observations across a monthly boundary.
NON_PERSISTENT_IDENTIFIERS = ("participant_masked", "subaccount_masked", "broker_id")

# Fields whose absence makes a row unusable rather than merely incomplete.
REQUIRED = ("TR_DATE", "cisin", "TR_TYPE", "VALUE_INR")


@dataclass(frozen=True)
class SchemaViolation:
    field: str
    problem: str
    n_rows: int

    def __str__(self) -> str:
        return f"{self.field}: {self.problem} ({self.n_rows:,} rows)"


def max_aggregation_window(field: str) -> str:
    """The widest window over which a field may be aggregated."""
    return "within_day" if field in NON_PERSISTENT_IDENTIFIERS else "unbounded"
