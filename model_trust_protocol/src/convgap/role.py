"""What an entry in the exhibit is for."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Whether an entry applies the dataset under study, or controls for it."""

    APPLICATION = "application"
    """An application of the institutional flow record itself."""

    CONTROL = "control"
    """A predictor drawn from outside the dataset, run through identical
    machinery. A control that fails at the same level indicates the friction is
    a property of the evaluation transition rather than of this dataset; a
    control that clears it indicates the opposite."""
