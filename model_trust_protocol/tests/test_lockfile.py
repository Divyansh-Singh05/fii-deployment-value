from __future__ import annotations

from pathlib import Path

import pytest

from convgap import lockfile


def test_missing_lock_is_empty_not_an_error(tmp_path: Path) -> None:
    assert lockfile.load(tmp_path / "absent.json") == {}


def test_round_trip_and_sorted_output(tmp_path: Path) -> None:
    p = tmp_path / "lock.json"
    lockfile.save({"b:2": "yy", "a:1": "xx"}, p)
    assert lockfile.load(p) == {"a:1": "xx", "b:2": "yy"}
    assert p.read_text().index('"a:1"') < p.read_text().index('"b:2"')


def test_unknown_schema_rejected(tmp_path: Path) -> None:
    p = tmp_path / "lock.json"
    p.write_text('{"schema": 99, "artifacts": {}}')
    with pytest.raises(ValueError, match="unsupported lock schema"):
        lockfile.load(p)
