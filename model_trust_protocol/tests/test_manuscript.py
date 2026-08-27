from __future__ import annotations

from pathlib import Path

from convgap import manuscript
from convgap.lockfile import load
from convgap.provenance import load_source_trees

_PAPER = Path(__file__).resolve().parents[1] / "paper"


def test_no_unaccounted_numeral_in_the_manuscript() -> None:
    """Every numeral is verified evidence or exempt with a stated reason.

    This is the paper's own standard applied to the paper. A numeral that is
    neither is a figure asserted without a source.
    """
    verified = manuscript.verified_values(load_source_trees(), load())
    unmatched = manuscript.scan(_PAPER, verified)
    assert not unmatched, manuscript.report(unmatched)


def test_every_exemption_states_a_reason() -> None:
    for numeral, reason in manuscript.EXEMPT.items():
        assert len(reason.split()) >= 2, (
            f"{numeral}: an exemption without a reason is an unexplained numeral"
        )


def test_scanner_detects_an_unsourced_numeral(tmp_path: Path) -> None:
    (tmp_path / "04_x.md").write_text("The effect is 41.732 basis points.\n")
    assert manuscript.scan(tmp_path, verified={"1.0"})


def test_scanner_accepts_a_sourced_numeral(tmp_path: Path) -> None:
    (tmp_path / "04_x.md").write_text("The effect is 41.732 basis points.\n")
    assert not manuscript.scan(tmp_path, verified={"41.732"})
