"""The frozen split, and era masks derived from it.

All parameters, thresholds and specifications are fixed on data through
2021-04-30. The test era opens 2021-07-01. The intervening two months are
masked at source - absent from the flow feed - so the embargo and the data
outage coincide rather than the embargo being a chosen buffer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utilities import load_config


def _cfg() -> dict:
    return load_config("frozen_split")


def describe_split() -> dict:
    c = _cfg()
    return {
        "train_end": c["train_end"],
        "test_start": c["test_start"],
        "embargo": f"{c['embargo']['start']} .. {c['embargo']['end']}",
        "embargo_reason": c["embargo"]["reason"],
        "common_instruments": c["universe"]["common_instruments"],
        "total_instruments": c["universe"]["total_instruments"],
    }


def era_masks(dates) -> dict[str, np.ndarray]:
    """FULL / TRAIN / TEST boolean masks keyed on the OUTCOME date.

    Keying on the outcome rather than the signal date matters: a forecast made
    on the last training day resolves inside the embargo, and that observation
    belongs to neither era.
    """
    c = _cfg()
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    train_end = pd.Timestamp(c["train_end"])
    test_start = pd.Timestamp(c["test_start"])
    return {
        "FULL": np.ones(len(d), bool),
        "TRAIN": (d <= train_end).to_numpy(),
        "TEST": (d >= test_start).to_numpy(),
    }


def in_embargo(dates) -> np.ndarray:
    c = _cfg()
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    return ((d >= pd.Timestamp(c["embargo"]["start"]))
            & (d <= pd.Timestamp(c["embargo"]["end"]))).to_numpy()


def excluded_months() -> list[str]:
    """Months excluded and NEVER imputed (null direction flag at source)."""
    return list(_cfg()["null_direction_months"])
