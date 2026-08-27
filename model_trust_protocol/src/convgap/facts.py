"""Descriptive claims made by the paper's data section.

Results are traced by :mod:`convgap.application`. The data section makes claims
too - sample sizes, spans, coverage - and in this programme several of them were
stale, quoted from logs written before an audit rebuilt the object they
describe. They are held to the same standard here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from convgap.evidence import Evidence, Statistic
from convgap.extract import csv_cell
from convgap.provenance import Source, SourceTree, VerificationStatus

_TABLE = "outputs/replication/descriptives.csv"
_PRODUCER = "src/convgap/replication/descriptives.py"


def _fact(key: str, value: float, claim: str, *, rtol: float = 1e-6) -> Evidence:
    return Evidence(
        value,
        Statistic.RATIO,
        Source(
            tree="local",
            artifact=_TABLE,
            locator=f"row key={key}, column value",
            producer=_PRODUCER,
        ),
        note=claim,
        extractor=csv_cell({"key": key}, "value"),
        rtol=rtol,
    )


@dataclass(frozen=True, slots=True)
class DataClaim:
    """One numeral the data section asserts."""

    text: str
    evidence: Evidence

    def verify(
        self, trees: dict[str, SourceTree], locks: dict[str, str] | None = None
    ) -> DataClaim:
        return replace(self, evidence=self.evidence.verify(trees, locks))

    @property
    def ok(self) -> bool:
        return self.evidence.status.is_publishable


CLAIMS: tuple[DataClaim, ...] = (
    DataClaim(
        "The record covers approximately 25.2 million trades.",
        _fact("raw_trades", 25_155_785, "raw transaction count"),
    ),
    DataClaim(
        "The record spans 5,960 distinct instrument identifiers.",
        _fact("raw_distinct_isins", 5960, "distinct instrument identifiers"),
    ),
    DataClaim(
        "It spans January 2011 to March 2025.",
        _fact("raw_first_year", 2011, "first year of the record"),
    ),
    DataClaim(
        "It spans January 2011 to March 2025.",
        _fact("raw_last_year", 2025, "last year of the record"),
    ),
    DataClaim(
        "Three months carry a wholly null direction flag, comprising 3.0% of trades.",
        _fact("null_direction_trade_share_pct", 3.0043, "share of affected trades", rtol=1e-4),
    ),
    DataClaim(
        "The broker control population fragments: 22,262 distinct identifiers "
        "with a median lifetime of one month over a 171-month span, for a "
        "real-world population numbering in the hundreds. The identifiers are "
        "re-minted, and no cross-period entity statistic is admissible.",
        _fact("brker_distinct_ids", 22_262, "distinct broker identifiers"),
    ),
    DataClaim(
        "The broker control population fragments.",
        _fact("brker_median_months", 1.0, "broker: median months per identifier"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_median_months", 1.0, "participant: median months per identifier"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_distinct_ids", 363_663, "distinct participant identifiers"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_mean_months", 1.2093, "participant: mean months per identifier", rtol=1e-3),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_distinct_ids", 394_398, "distinct sub-account identifiers"),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_median_months", 1.0, "sub-account: median months"),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_mean_months", 1.2101, "sub-account: mean months", rtol=1e-3),
    ),
    DataClaim(
        "The broker control population's mean identifier lifetime is 1.40 months.",
        _fact("brker_mean_months", 1.3968, "broker: mean months", rtol=1e-3),
    ),
    DataClaim(
        "The modelling universe is 1,028 instruments over 802,806 instrument-days.",
        _fact("model_instruments", 1028, "instruments in the modelling universe"),
    ),
    DataClaim(
        "The broker control population fragments: 22,262 distinct identifiers "
        "with a median lifetime of one month over a 171-month span, for a "
        "real-world population numbering in the hundreds. The identifiers are "
        "re-minted, and no cross-period entity statistic is admissible.",
        _fact("brker_distinct_ids", 22_262, "distinct broker identifiers"),
    ),
    DataClaim(
        "The broker control population fragments.",
        _fact("brker_median_months", 1.0, "broker: median months per identifier"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_median_months", 1.0, "participant: median months per identifier"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_distinct_ids", 363_663, "distinct participant identifiers"),
    ),
    DataClaim(
        "Masked participant identifiers fragment identically.",
        _fact("fii_mean_months", 1.2093, "participant: mean months per identifier", rtol=1e-3),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_distinct_ids", 394_398, "distinct sub-account identifiers"),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_median_months", 1.0, "sub-account: median months"),
    ),
    DataClaim(
        "Masked sub-account identifiers fragment identically.",
        _fact("sub_acc_mean_months", 1.2101, "sub-account: mean months", rtol=1e-3),
    ),
    DataClaim(
        "The broker control population's mean identifier lifetime is 1.40 months.",
        _fact("brker_mean_months", 1.3968, "broker: mean months", rtol=1e-3),
    ),
    DataClaim(
        "The modelling universe is 1,028 instruments over 802,806 instrument-days.",
        _fact("model_stock_days", 802_806, "instrument-days in the modelling universe"),
    ),
    DataClaim(
        "The training era covers 618 instruments across 508,563 instrument-days.",
        _fact("train_instruments", 618, "training-era instruments"),
    ),
    DataClaim(
        "The training era covers 618 instruments across 508,563 instrument-days.",
        _fact("train_stock_days", 508_563, "training-era instrument-days"),
    ),
    DataClaim(
        "The test era covers 851 instruments across 294,243 instrument-days.",
        _fact("test_instruments", 851, "test-era instruments"),
    ),
    DataClaim(
        "The test era covers 851 instruments across 294,243 instrument-days.",
        _fact("test_stock_days", 294_243, "test-era instrument-days"),
    ),
    DataClaim(
        "Only 441 instruments appear in both eras, so replication is on a "
        "substantially different cross-section rather than a repeat sample.",
        _fact("era_instrument_overlap", 441, "instruments present in both eras"),
    ),
    DataClaim(
        "The price panel carries 5,945,010 rows over 3,514 trading days.",
        _fact("panel_trading_days", 3514, "trading days in the price panel"),
    ),
    DataClaim(
        "The price panel carries 5,945,010 rows over 3,514 trading days.",
        _fact("panel_rows", 5_945_010, "rows in the price panel"),
    ),
)


def verify_all(
    trees: dict[str, SourceTree], locks: dict[str, str] | None = None
) -> list[DataClaim]:
    return [c.verify(trees, locks) for c in CLAIMS]


def summarise(claims: list[DataClaim]) -> str:
    ok = sum(1 for c in claims if c.ok)
    lines = [f"{ok}/{len(claims)} data-section claims traced", ""]
    for c in claims:
        flag = "PASS" if c.ok else "FAIL"
        detail = c.evidence.extraction_note or str(c.evidence.status)
        lines.append(f"  [{flag}] {c.evidence.value:>12,.4g}  {c.evidence.note:<42}{detail}")
    unresolved = [c for c in claims if c.evidence.status is VerificationStatus.MISMATCH]
    if unresolved:
        lines.append("")
        lines.append(
            "A MISMATCH means the paper asserts a figure the artifact does "
            "not hold. Correct the paper."
        )
    return "\n".join(lines)
