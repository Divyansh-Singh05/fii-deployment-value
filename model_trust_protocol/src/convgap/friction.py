"""The friction ladder.

The dataset is the object of study. Every model applied to it is a stress-test
environment, not a subject in its own right. This module defines the ascending
levels of real-world friction those environments impose, and it is the ordering
principle of the paper's central exhibit.

The levels are ordered by *institutional reality*, not by statistical severity.
A result must clear each level to reach the next, so the level at which an
application dies is a complete statement of what the data was worth in that
use — and, read across applications, the ladder maps the boundary of the
dataset's usefulness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FrictionLevel(IntEnum):
    """Ascending constraints a finding must survive to be worth acting on."""

    EXISTENCE = 0
    """The effect is present in sample under conventional inference."""

    REPLICATION = 1
    """It survives a frozen out-of-sample split on a disjoint era."""

    AVAILABILITY = 2
    """It is knowable at the moment a decision must be taken, not only
    contemporaneously with the outcome."""

    COMPETITION = 3
    """It survives against the baseline the application would actually be
    deployed against - a well-specified conditional-variance model, a trivial
    rule, a freely available public proxy."""

    DETECTABILITY = 4
    """Its magnitude is large enough to register in the metric the decision
    consumes, which is generally not the metric the significance test used."""

    EXECUTION = 5
    """The edge exceeds the cost of capturing it, measured as a breakeven
    rather than as a net figure at an assumed cost."""

    @property
    def label(self) -> str:
        return _DESCRIPTIONS[self].label

    @property
    def question(self) -> str:
        return _DESCRIPTIONS[self].question

    @property
    def is_statistical(self) -> bool:
        """Levels 0-1 are questions about the sample; 2-5 are about deployment."""
        return self <= FrictionLevel.REPLICATION


@dataclass(frozen=True, slots=True)
class _Description:
    label: str
    question: str


_DESCRIPTIONS: dict[FrictionLevel, _Description] = {
    FrictionLevel.EXISTENCE: _Description(
        "existence",
        "Is the effect present under conventional inference?",
    ),
    FrictionLevel.REPLICATION: _Description(
        "replication",
        "Does it hold on an era the design never touched?",
    ),
    FrictionLevel.AVAILABILITY: _Description(
        "availability",
        "Is it knowable when the decision must be made?",
    ),
    FrictionLevel.COMPETITION: _Description(
        "competition",
        "Does it beat the baseline it would be deployed against?",
    ),
    FrictionLevel.DETECTABILITY: _Description(
        "detectability",
        "Is it large enough to register in the metric the decision consumes?",
    ),
    FrictionLevel.EXECUTION: _Description(
        "execution",
        "Does the edge exceed the cost of capturing it?",
    ),
}

CLEARS_ALL = FrictionLevel.EXECUTION + 1
"""Sentinel for an application that survives every level."""
