"""The flow-regime state model - an INSTRUMENT, not an object of study.

The methodological commitment that governs the whole programme: the dataset is
the object of study and every model applied to it is an instrument. When the
ablation reports that this model's conditioning contributes nothing to a risk
density, that is a statement about the FLOW DATA, not about hidden Markov
models. Under a model-centred reading it would invite a better architecture;
under a data-centred reading it is a measurement.

A known weakness, stated rather than defended: the geometric dwell-time
implication of a plain HMM is why episode ENDS needed a separate hazard layer
at all. A hidden semi-Markov formulation parameterises dwell time natively and
would have unified the two models into one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utilities import dataset


def load_episode_predictions() -> pd.DataFrame:
    """Post-audit episode panel. Pre-audit objects are quarantined by path."""
    return pd.read_parquet(dataset("hazard"))


def episode_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as (label, start_index, length)."""
    lab = np.asarray(labels)
    if len(lab) == 0:
        return []
    change = np.r_[True, lab[1:] != lab[:-1]]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.r_[starts, len(lab)])
    return [(lab[s], int(s), int(n)) for s, n in zip(starts, lengths)]


def shuffled_run_null(labels: np.ndarray, rng: np.random.Generator,
                      n_draws: int = 200) -> np.ndarray:
    """Mean run length under a within-instrument label shuffle.

    Preserves each instrument's LABEL COUNT and destroys only temporal
    adjacency, so the null isolates clustering rather than label frequency.
    """
    lab = np.asarray(labels)
    out = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(lab)
        out[i] = float(np.mean([n for _, _, n in episode_runs(perm)]))
    return out
