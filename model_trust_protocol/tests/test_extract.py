from __future__ import annotations

from pathlib import Path

import pytest

from convgap.extract import ExtractionError, csv_cell, log_regex


@pytest.fixture
def metrics(tmp_path: Path) -> Path:
    p = tmp_path / "m.csv"
    p.write_text(
        "strategy,era,cost_bps,sharpe\nA,TEST,0.0,1.44\nA,TEST,15.0,-1.54\nB,TEST,0.0,0.90\n"
    )
    return p


def test_csv_cell_selects_unique_row(metrics: Path) -> None:
    get = csv_cell({"strategy": "A", "era": "TEST", "cost_bps": "15.0"}, "sharpe")
    assert get(metrics) == pytest.approx(-1.54)


def test_csv_cell_tolerates_numeric_formatting(metrics: Path) -> None:
    assert csv_cell({"strategy": "A", "cost_bps": "0"}, "sharpe")(metrics) == pytest.approx(1.44)


def test_ambiguous_selection_is_an_error(metrics: Path) -> None:
    with pytest.raises(ExtractionError, match="exactly one number"):
        csv_cell({"era": "TEST"}, "sharpe")(metrics)


def test_no_match_is_an_error(metrics: Path) -> None:
    with pytest.raises(ExtractionError, match="no row"):
        csv_cell({"strategy": "Z"}, "sharpe")(metrics)


def test_unknown_column_is_an_error(metrics: Path) -> None:
    with pytest.raises(ExtractionError, match="not present"):
        csv_cell({"strategy": "A", "cost_bps": "0.0"}, "nope")(metrics)


def test_log_regex_requires_a_unique_match(tmp_path: Path) -> None:
    p = tmp_path / "run.log"
    p.write_text("AUC = 0.797\nAUC = 0.569\n")
    with pytest.raises(ExtractionError, match="matched 2 times"):
        log_regex(r"AUC = ([\d.]+)")(p)
    p.write_text("final AUC = 0.797\n")
    assert log_regex(r"final AUC = ([\d.]+)")(p) == pytest.approx(0.797)
