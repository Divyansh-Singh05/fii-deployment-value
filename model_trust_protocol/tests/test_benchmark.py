from __future__ import annotations

import math

import pytest

from convgap.benchmark import (
    Detectability,
    crps_gain_from_sigma_modulation,
    sharpe_gain_from_ic,
)


def test_conversion_ratio_and_verdicts() -> None:
    assert Detectability(1.0, 0.9, "CRPS", "m").verdict == "converts"
    assert Detectability(1.0, 0.3, "CRPS", "m").verdict == "partial conversion"
    assert Detectability(1.0, 0.02, "CRPS", "m").verdict == "no conversion"
    assert Detectability(1.0, -0.5, "CRPS", "m").verdict == "inverted"
    assert Detectability(2.0, 1.0, "CRPS", "m").conversion_ratio == pytest.approx(0.5)


def test_method_is_mandatory() -> None:
    with pytest.raises(ValueError, match="method"):
        Detectability(1.0, 0.5, "CRPS", "   ")


def test_implied_must_be_positive_magnitude() -> None:
    with pytest.raises(ValueError, match="positive magnitude"):
        Detectability(0.0, 0.5, "CRPS", "m")


def test_crps_gain_is_linear_in_sigma_modulation() -> None:
    a = crps_gain_from_sigma_modulation(0.01, baseline_crps=0.56)
    b = crps_gain_from_sigma_modulation(0.02, baseline_crps=0.56)
    assert b == pytest.approx(2 * a)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_crps_gain_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        crps_gain_from_sigma_modulation(bad, baseline_crps=0.56)


def test_sharpe_gain_follows_fundamental_law() -> None:
    assert sharpe_gain_from_ic(0.02, breadth=2500) == pytest.approx(0.02 * math.sqrt(2500))
