"""Extractors: read a declared value back out of the artifact that produced it.

A digest match proves an artifact is unchanged. It does not prove the number
transcribed from it was ever the right one. An extractor closes that gap by
locating the value inside the artifact and comparing it against what the channel
declares, so a transcription error is reported as a MISMATCH rather than
travelling into an exhibit.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Mapping
from pathlib import Path

type Extractor = Callable[[Path], float]


class ExtractionError(RuntimeError):
    """Raised when an extractor cannot locate its value in the artifact."""


def _matches(row: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    for key, want in filters.items():
        if key not in row:
            raise ExtractionError(f"column {key!r} not present; have {sorted(row)}")
        got = row[key]
        if got == want:
            continue
        # numeric-tolerant comparison so "15.0" matches "15"
        try:
            if float(got) == float(want):
                continue
        except ValueError:
            pass
        return False
    return True


def csv_cell(filters: Mapping[str, str], column: str) -> Extractor:
    """Select the unique row matching ``filters`` and return ``column`` as a float.

    Uniqueness is enforced: an ambiguous selection means the locator does not
    identify a single number, which is exactly the condition a referee would
    fail to resolve by hand.
    """

    def _extract(path: Path) -> float:
        with path.open(newline="") as fh:
            hits = [r for r in csv.DictReader(fh) if _matches(r, filters)]
        if not hits:
            raise ExtractionError(f"no row in {path.name} matches {dict(filters)}")
        if len(hits) > 1:
            raise ExtractionError(
                f"{len(hits)} rows in {path.name} match {dict(filters)}; "
                "a locator must identify exactly one number"
            )
        if column not in hits[0]:
            raise ExtractionError(f"column {column!r} not present in {path.name}")
        return float(hits[0][column])

    return _extract


def parquet_cell(filters: Mapping[str, object], column: str) -> Extractor:
    """As :func:`csv_cell`, for a parquet table."""

    def _extract(path: Path) -> float:
        import pandas as pd

        frame = pd.read_parquet(path)
        mask = pd.Series(True, index=frame.index)
        for key, want in filters.items():
            if key not in frame.columns:
                raise ExtractionError(
                    f"column {key!r} not in {path.name}; have {list(frame.columns)}"
                )
            mask &= frame[key] == want
        hits = frame.loc[mask]
        if hits.empty:
            raise ExtractionError(f"no row in {path.name} matches {dict(filters)}")
        if len(hits) > 1:
            raise ExtractionError(
                f"{len(hits)} rows in {path.name} match {dict(filters)}; "
                "a locator must identify exactly one number"
            )
        if column not in hits.columns:
            raise ExtractionError(f"column {column!r} not in {path.name}")
        return float(hits.iloc[0][column])

    return _extract


def log_regex(pattern: str, group: int = 1, *, dotall: bool = False) -> Extractor:
    """Capture a float from a run log by regular expression.

    The pattern must match exactly once. A log line is weaker evidence than a
    table cell and should be used only where no table carries the quantity.

    Parameters
    ----------
    dotall
        Let ``.`` span newlines, so a value can be anchored to a section header
        several lines above it. Needed where a log repeats the same row label
        under different headings and the label alone is ambiguous.
    """
    compiled = re.compile(pattern, re.DOTALL if dotall else 0)

    def _extract(path: Path) -> float:
        found = compiled.findall(path.read_text(errors="replace"))
        if not found:
            raise ExtractionError(f"pattern {pattern!r} not found in {path.name}")
        if len(found) > 1:
            raise ExtractionError(
                f"pattern {pattern!r} matched {len(found)} times in {path.name}; "
                "a locator must identify exactly one number"
            )
        hit = found[0]
        raw = hit[group - 1] if isinstance(hit, tuple) else hit
        return float(raw)

    return _extract
