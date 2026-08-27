from __future__ import annotations

from pathlib import Path

import pytest

from convgap.provenance import (
    Source,
    SourceTree,
    SourceTreeError,
    VerificationStatus,
    check,
    digest,
    load_source_trees,
)


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, SourceTree]:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "m.csv").write_text("a,b\n1,2\n")
    return {"research": SourceTree("research", tmp_path, "fixture")}


def test_missing_artifact_reports_missing(tree: dict[str, SourceTree]) -> None:
    status, _ = check(Source("research", "outputs/absent.csv", "col a"), tree)
    assert status is VerificationStatus.MISSING


def test_first_sighting_records_digest_but_stays_unverified(
    tree: dict[str, SourceTree],
) -> None:
    status, stamped = check(Source("research", "outputs/m.csv", "col a"), tree)
    assert status is VerificationStatus.UNVERIFIED
    assert stamped.sha256 is not None
    assert stamped.verified_on is not None


def test_recorded_digest_verifies(tree: dict[str, SourceTree]) -> None:
    src = Source("research", "outputs/m.csv", "col a")
    _, stamped = check(src, tree)
    locks = {"research:outputs/m.csv": stamped.sha256 or ""}
    status, _ = check(src, tree, locks)
    assert status is VerificationStatus.VERIFIED


def test_changed_artifact_is_stale_not_silently_accepted(
    tmp_path: Path, tree: dict[str, SourceTree]
) -> None:
    src = Source("research", "outputs/m.csv", "col a")
    _, stamped = check(src, tree)
    locks = {"research:outputs/m.csv": stamped.sha256 or ""}
    (tmp_path / "outputs" / "m.csv").write_text("a,b\n9,9\n")
    status, restamped = check(src, tree, locks)
    assert status is VerificationStatus.STALE
    assert restamped.sha256 != stamped.sha256


def test_unknown_tree_raises(tree: dict[str, SourceTree]) -> None:
    with pytest.raises(SourceTreeError):
        Source("nowhere", "x", "y").path(tree)


def test_digest_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    assert digest(p) == digest(p)


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONVGAP_RESEARCH_ROOT", str(tmp_path))
    trees = load_source_trees()
    # roots are resolved, so compare against the resolved form
    assert trees["research"].root == tmp_path.resolve()
    assert trees["research"].available


def test_publishable_statuses() -> None:
    assert VerificationStatus.VERIFIED.is_publishable
    assert VerificationStatus.DERIVED.is_publishable
    assert not VerificationStatus.UNVERIFIED.is_publishable
    assert not VerificationStatus.STALE.is_publishable
    assert not VerificationStatus.MISSING.is_publishable
