"""The detectability benchmark.

A channel reports that a predictor is significant and that it delivers little
value. On its own that is two numbers and an assertion; a referee may
reasonably answer that a small effect naturally yields a small improvement.

The benchmark closes that gap. For each channel it asks: *given the effect
size the significance test itself measured, how much should the value metric
have improved if the effect were real and fully exploitable?* The ratio of
observed to implied improvement is the channel's conversion rate, and it is
comparable across channels in a way the raw statistics are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detectability:
    """Implied versus observed improvement in a value metric.

    Parameters
    ----------
    implied
        Improvement the value metric should show if the measured effect were
        real and exploited without friction. Always a positive magnitude.
    observed
        Improvement actually measured. May be negative.
    unit
        Unit both figures are expressed in, e.g. ``"CRPS"`` or ``"bp"``.
    method
        How ``implied`` was derived. Required: the benchmark is only as
        credible as the derivation, and it must be legible to a referee.
    """

    implied: float
    observed: float
    unit: str
    method: str

    def __post_init__(self) -> None:
        if self.implied <= 0:
            raise ValueError("implied improvement must be a positive magnitude")
        if not self.method.strip():
            raise ValueError("method must state how the implied improvement was derived")

    @property
    def conversion_ratio(self) -> float:
        """Observed improvement as a fraction of the implied maximum.

        Near 1 means the effect converted essentially fully. Near 0 means it
        did not convert. Negative means acting on the effect made the forecast
        worse than ignoring it.
        """
        return self.observed / self.implied

    @property
    def verdict(self) -> str:
        r = self.conversion_ratio
        if r < 0:
            return "inverted"
        if r < 0.10:
            return "no conversion"
        if r < 0.50:
            return "partial conversion"
        return "converts"

    def __str__(self) -> str:
        return (
            f"{self.observed:+.4g}/{self.implied:.4g} {self.unit} "
            f"= {self.conversion_ratio:.0%} ({self.verdict})"
        )


def crps_gain_from_sigma_modulation(
    sigma_ratio: float,
    *,
    baseline_crps: float,
) -> float:
    """Implied CRPS improvement from a proportional change in forecast scale.

    For a location-scale predictive density, CRPS scales linearly in the scale
    parameter. A predictor that modulates the conditional standard deviation by
    a factor ``sigma_ratio`` around its baseline therefore implies a CRPS
    improvement bounded above by the scale reduction it can deliver on the
    fraction of observations where it points the right way.

    This is a deliberately *generous* upper bound: it credits the predictor with
    perfect sign accuracy and no estimation error. A channel that fails to
    convert against this bound has failed against the most favourable
    accounting available to it.

    Parameters
    ----------
    sigma_ratio
        Multiplicative modulation of sigma attributable to the predictor,
        e.g. ``0.02`` for a 2% modulation.
    baseline_crps
        Mean CRPS of the baseline density forecast.

    Returns
    -------
    float
        Implied reduction in mean CRPS, as a positive magnitude.
    """
    if not 0 < sigma_ratio < 1:
        raise ValueError("sigma_ratio must lie strictly between 0 and 1")
    if baseline_crps <= 0:
        raise ValueError("baseline_crps must be positive")
    # E|X| for a mean-zero scale family is proportional to sigma, so a
    # sigma reduction of r maps to a CRPS reduction of r to first order.
    return baseline_crps * sigma_ratio


def sharpe_gain_from_ic(ic: float, *, breadth: int) -> float:
    """Implied annualised Sharpe from an information coefficient.

    The fundamental law of active management, ``IR = IC * sqrt(breadth)``,
    used here only as an upper bound: it assumes an unconstrained,
    frictionless portfolio that harvests the full cross-sectional signal.

    Parameters
    ----------
    ic
        Information coefficient (cross-sectional rank correlation).
    breadth
        Number of independent bets per year.
    """
    if breadth <= 0:
        raise ValueError("breadth must be positive")
    return ic * math.sqrt(breadth)
