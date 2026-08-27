from __future__ import annotations

import pytest

from convgap.application import Application
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.friction import CLEARS_ALL, FrictionLevel
from convgap.provenance import Source, VerificationStatus
from convgap.registry import all_applications, controls, coverage_gaps, survivors
from convgap.role import Role


def test_every_instrument_is_used() -> None:
    assert coverage_gaps() == [], (
        "the dataset is claimed to have been stressed across the modelling stack; "
        "an unused instrument is a hole in that claim"
    )


def test_a_survivor_exists() -> None:
    assert survivors(), (
        "without an application that clears the ladder the exhibit reports the "
        "dataset is worthless rather than mapping where its value stops"
    )


def test_an_external_control_exists() -> None:
    assert controls(), (
        "without a non-dataset control, a failure cannot be attributed to the "
        "data rather than to the evaluation transition"
    )


def test_keys_are_unique_and_sorted() -> None:
    keys = [a.key for a in all_applications()]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("app", all_applications(), ids=lambda a: a.key)
def test_declares_a_mechanism(app: Application) -> None:
    assert len(app.mechanism.split()) >= 15, (
        f"{app.key}: why the value is lost is the contribution; a label is not enough"
    )


@pytest.mark.parametrize("app", all_applications(), ids=lambda a: a.key)
def test_friction_level_is_valid(app: Application) -> None:
    assert app.fails_at is None or isinstance(app.fails_at, FrictionLevel)
    assert app.role in (Role.APPLICATION, Role.CONTROL)


@pytest.mark.parametrize("app", all_applications(), ids=lambda a: a.key)
def test_both_halves_carry_sourced_evidence(app: Application) -> None:
    for half in (app.significance(), app.value()):
        assert isinstance(half, EvidenceSet)
        assert half.items
        for item in half.items:
            assert isinstance(item.source, Source)
            assert item.source.locator.strip(), "a locator must be actionable"
            assert item.source.artifact.strip()


@pytest.mark.parametrize("app", all_applications(), ids=lambda a: a.key)
def test_untraced_numbers_are_not_publishable(app: Application) -> None:
    for item in app.significance().items:
        assert item.status is VerificationStatus.UNVERIFIED
        assert not item.status.is_publishable


def test_clears_counts_levels_survived() -> None:
    for app in all_applications():
        result_clears = CLEARS_ALL if app.fails_at is None else int(app.fails_at)
        assert 0 <= result_clears <= CLEARS_ALL


def test_evidence_set_status_is_the_weakest_member() -> None:
    src = Source("research", "a.csv", "col")
    weak = Evidence(1.0, Statistic.T_STAT, src, status=VerificationStatus.MISSING)
    strong = Evidence(2.0, Statistic.T_STAT, src, status=VerificationStatus.VERIFIED)
    assert EvidenceSet("s", (strong, weak)).status is VerificationStatus.MISSING


def test_replication_detects_sign_flip() -> None:
    src = Source("research", "a.csv", "col")
    same = EvidenceSet(
        "s",
        (
            Evidence(-2.0, Statistic.T_STAT, src, era=Era.TRAIN),
            Evidence(-3.0, Statistic.T_STAT, src, era=Era.TEST),
        ),
    )
    flipped = EvidenceSet(
        "s",
        (
            Evidence(-2.0, Statistic.T_STAT, src, era=Era.TRAIN),
            Evidence(+3.0, Statistic.T_STAT, src, era=Era.TEST),
        ),
    )
    single = EvidenceSet("s", (Evidence(-2.0, Statistic.T_STAT, src, era=Era.FULL),))
    assert same.replicates is True
    assert flipped.replicates is False
    assert single.replicates is None
