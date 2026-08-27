"""One seed, set in one place.

Bootstrap replicates and any stochastic null in this package draw from a
generator seeded here. Stages take a `numpy.random.Generator` explicitly
rather than touching global state, so a stage's result does not depend on
what ran before it.
"""
from __future__ import annotations

import numpy as np

from .utilities import load_config

MASTER_SEED = 20260826


def generator(stream: str = "") -> np.random.Generator:
    """Return a Generator for a named stream.

    Distinct `stream` names give independent, reproducible sequences, so
    adding a bootstrap in one stage cannot shift the draws in another.
    """
    if not stream:
        return np.random.default_rng(MASTER_SEED)
    offset = int.from_bytes(stream.encode()[:8].ljust(8, b"\0"), "little")
    return np.random.default_rng((MASTER_SEED + offset) % (2**63))


def bootstrap_generator() -> np.random.Generator:
    """The generator configured for the density-gate block bootstrap."""
    cfg = load_config("model_config")["inference"]["block_bootstrap"]
    return np.random.default_rng(cfg["seed"])
