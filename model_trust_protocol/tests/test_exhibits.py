from __future__ import annotations

from pathlib import Path

from convgap import exhibits
from convgap.provenance import load_source_trees
from convgap.registry import all_applications


def _table() -> exhibits.Table1:
    trees = load_source_trees()
    return exhibits.build([a.resolve(trees) for a in all_applications()])


def test_markdown_rows_have_uniform_column_count() -> None:
    lines = [ln for ln in _table().to_markdown().splitlines() if ln.startswith("|")]
    widths = {ln.count("|") - ln.count("\\|") for ln in lines}
    assert len(widths) == 1, "a literal pipe inside a cell has broken the table"


def test_csv_round_trips(tmp_path: Path) -> None:
    import csv

    out = tmp_path / "t1.csv"
    table = _table()
    table.to_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(table.rows)
    assert {r["key"] for r in rows} == {r.key for r in table.rows}


def test_audit_note_reports_both_kinds_of_work() -> None:
    note = _table().audit_note()
    assert "publishable" in note
    assert "untraced" in note or "STALE" in note or "MISSING" in note


def test_ladder_accounts_for_every_application() -> None:
    table = _table()
    lost = sum(table.attrition.values())
    assert lost + len(table.survivors) == len(table.applications)


def test_rows_are_ordered_by_levels_cleared() -> None:
    clears = [r.clears for r in _table().rows]
    assert clears == sorted(clears), "the exhibit is read as a descent"
