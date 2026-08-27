"""Descriptive facts about the record, computed rather than inherited.

A paper's data section carries numerals - sample sizes, spans, coverage - that
are rarely traced with the care applied to its results. They are the figures a
reader uses to judge whether the study is worth reading at all, and in this
programme several of them turned out to be stale: quoted from logs written
before an audit rebuilt the object they describe.

Every figure this module emits is computed from a source artifact in one pass
and written to a single table, so the data section can be held to the same
standard as the results.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

NULL_DIRECTION_MONTHS: Final = ("2023-06", "2023-09", "2023-11")
"""Months present in the feed but carrying a wholly null direction flag."""

MASKED_MONTHS: Final = ("2021-05", "2021-06")
"""Months absent from the feed at source, coincident with the frozen split."""


def _year(value: object) -> tuple[float, str]:
    """Year and ISO string of a date pulled from a polars aggregate."""
    if not isinstance(value, dt.date):
        raise TypeError(f"expected a date, got {type(value).__name__}: {value!r}")
    return float(value.year), value.isoformat()


@dataclass(frozen=True, slots=True)
class Fact:
    """One descriptive quantity, with the artifact it was computed from."""

    key: str
    value: float
    unit: str
    source: str
    note: str = ""


def _raw_record(isin_mapping: Path) -> list[Fact]:
    files = sorted(str(p) for p in isin_mapping.glob("20[0-9][0-9].parquet"))
    lf = pl.scan_parquet(files)
    agg = (
        lf.select(
            n_trades=pl.len(),
            first=pl.col("TR_DATE").min(),
            last=pl.col("TR_DATE").max(),
            n_isin=pl.col("ISIN").n_unique(),
            n_dates=pl.col("TR_DATE").n_unique(),
        )
        .collect()
        .to_dicts()[0]
    )

    src = "ISIN_MAPPING/20*.parquet"
    raw_lo, raw_lo_iso = _year(agg["first"])
    raw_hi, raw_hi_iso = _year(agg["last"])
    facts = [
        Fact("raw_trades", float(agg["n_trades"]), "trades", src),
        Fact("raw_distinct_isins", float(agg["n_isin"]), "instruments", src),
        Fact("raw_distinct_dates", float(agg["n_dates"]), "dates", src),
        Fact("raw_first_year", raw_lo, "year", src, note=raw_lo_iso),
        Fact("raw_last_year", raw_hi, "year", src, note=raw_hi_iso),
    ]

    # Share of trades falling in the months whose direction flag is wholly null.
    month = pl.col("TR_DATE").dt.strftime("%Y-%m")
    null_share = (
        lf.select(
            total=pl.len(),
            in_null_months=month.is_in(list(NULL_DIRECTION_MONTHS)).sum(),
        )
        .collect()
        .to_dicts()[0]
    )
    pct = 100.0 * null_share["in_null_months"] / null_share["total"]
    facts.append(
        Fact(
            "null_direction_trade_share_pct",
            pct,
            "%",
            src,
            note="Trades in " + ", ".join(NULL_DIRECTION_MONTHS),
        )
    )
    return facts


def _identifier_persistence(isin_mapping: Path) -> list[Fact]:
    """The control-population test for identifier re-minting.

    Masked participant identifiers look persistent and need not be. Month-to-
    month reappearance cannot settle it, because it conflates "did not trade"
    with "was re-masked". The decisive test runs the persistence statistic on a
    population whose real-world persistence is known independently - brokers and
    clearing members, a near-fixed set numbering in the hundreds. If that control
    fragments, the identifiers are re-minted and every cross-period entity
    statistic computed on them is fiction.
    """
    files = sorted(str(p) for p in isin_mapping.glob("20[0-9][0-9].parquet"))
    lf = pl.scan_parquet(files).with_columns(month=pl.col("TR_DATE").dt.strftime("%Y-%m"))
    src = "ISIN_MAPPING/20*.parquet"
    facts: list[Fact] = []
    for col, label in (
        ("FII", "masked participant"),
        ("SUB_ACC", "masked sub-account"),
        ("BRKER", "broker (control population)"),
    ):
        d = (
            lf.select(pl.col(col), pl.col("month"))
            .drop_nulls()
            .group_by(col)
            .agg(m=pl.col("month").n_unique())
            .select(
                ids=pl.len(),
                median_months=pl.col("m").median(),
                mean_months=pl.col("m").mean(),
            )
            .collect()
            .to_dicts()[0]
        )
        key = col.lower()
        facts += [
            Fact(f"{key}_distinct_ids", float(d["ids"]), "identifiers", src, note=label),
            Fact(
                f"{key}_median_months",
                float(d["median_months"]),
                "months",
                src,
                note=f"{label}: median distinct months per identifier",
            ),
            Fact(
                f"{key}_mean_months",
                float(d["mean_months"]),
                "months",
                src,
                note=f"{label}: mean distinct months per identifier",
            ),
        ]
    return facts


def _modelling_universe(validation_data: Path) -> list[Fact]:
    states = pl.read_parquet(validation_data / "states_v3.parquet")
    src = "VALIDATION_DATA/states_v3.parquet"
    lo_year, lo_iso = _year(states["TR_DATE"].min())
    hi_year, hi_iso = _year(states["TR_DATE"].max())
    facts = [
        Fact("model_stock_days", float(states.height), "stock-days", src),
        Fact(
            "model_instruments",
            float(states["cisin"].n_unique()),
            "instruments",
            src,
            note="Union across eras; the two eras do not cover the same set.",
        ),
        Fact("model_first_year", lo_year, "year", src, note=lo_iso),
        Fact("model_last_year", hi_year, "year", src, note=hi_iso),
    ]
    for era in ("TRAIN", "TEST"):
        sub = states.filter(pl.col("era") == era)
        facts += [
            Fact(f"{era.lower()}_stock_days", float(sub.height), "stock-days", src),
            Fact(f"{era.lower()}_instruments", float(sub["cisin"].n_unique()), "instruments", src),
        ]
    train = set(states.filter(pl.col("era") == "TRAIN")["cisin"].unique().to_list())
    test = set(states.filter(pl.col("era") == "TEST")["cisin"].unique().to_list())
    facts.append(
        Fact(
            "era_instrument_overlap",
            float(len(train & test)),
            "instruments",
            src,
            note="Instruments present in both eras. The test era is not a "
            "repeat sample; replication is on a partly different cross-section.",
        )
    )
    return facts


def _price_panel(validation_data: Path) -> list[Fact]:
    panel = (
        pl.scan_parquet(validation_data / "returns_panel_v3.parquet")
        .select(
            n_rows=pl.len(),
            n_dates=pl.col("date").n_unique(),
            first=pl.col("date").min(),
            last=pl.col("date").max(),
        )
        .collect()
        .to_dicts()[0]
    )
    src = "VALIDATION_DATA/returns_panel_v3.parquet"
    p_lo, p_lo_iso = _year(panel["first"])
    p_hi, p_hi_iso = _year(panel["last"])
    return [
        Fact("panel_rows", float(panel["n_rows"]), "rows", src),
        Fact("panel_trading_days", float(panel["n_dates"]), "days", src),
        Fact("panel_first_year", p_lo, "year", src, note=p_lo_iso),
        Fact("panel_last_year", p_hi, "year", src, note=p_hi_iso),
    ]


def run(validation_data: Path, isin_mapping: Path) -> list[Fact]:
    """Compute every descriptive the data section reports."""
    return [
        *_raw_record(isin_mapping),
        *_identifier_persistence(isin_mapping),
        *_modelling_universe(validation_data),
        *_price_panel(validation_data),
    ]


def render(facts: list[Fact]) -> str:
    lines = [
        "descriptives, computed from source artifacts",
        f"{'key':<34}{'value':>16}  {'unit':<14}source",
    ]
    for f in facts:
        lines.append(f"{f.key:<34}{f.value:>16,.4g}  {f.unit:<14}{f.source}")
    return "\n".join(lines)


def to_csv(facts: list[Fact], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["key", "value", "unit", "source", "note"])
        for f in facts:
            w.writerow([f.key, f"{f.value:.6f}", f.unit, f.source, f.note])
    return path
