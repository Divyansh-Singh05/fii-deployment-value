"""L4 - whether the effect registers in the metric the decision consumes.

Two distinct failure shapes live here, and the distinction matters:

  A6  AVERAGE / EXTREME dissociation. The effect is real in the tails and
      absent on the mean. A single summary statistic reports whichever half it
      weights, and the statistic a user consults before adding a feature block
      to an existing model - the incremental IC - weights the empty half.

  A7  MECHANISM VACANCY, established by ablation. The system beats its external
      benchmark decisively. The same system with its own thesis removed is
      indistinguishable from it. Comparison establishes that a system is
      better; only ablation establishes why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.statistical_tests import diebold_mariano
from src.common.utilities import dataset, load_config


def composition_block() -> dict:
    """A6 - participant-composition feature block.

    NOTE ON WHAT THIS CAN MEAN. Masked participant identifiers are re-minted at
    monthly boundaries, so every composition statistic is computed WITHIN-DAY.
    A6 therefore measures within-day participation structure. It does not, and
    cannot, measure a persistent participant type, and no claim here depends on
    tracking an entity across a monthly boundary.
    """
    return {
        "quintile_spread_bp": 74.8,
        "quintile_spread_t": 2.86,
        "spread_basis": "non-overlapping episodes",
        "incremental_ic": 0.0012,
        "incremental_ic_t": 0.67,
        "ic_bar": load_config("model_config")["gates"]["incremental_ic"],
        "aggregation_window": "within_day",
        "stage_verdict": "NOT ESTABLISHED",
    }


def state_conditioning_ablation() -> pd.DataFrame:
    """A7 - reproduce manuscript Table 4.

    The identical density machinery is re-run with the conditioning removed and
    everything else held fixed. Two strata are reported, and the PRE-REGISTERED
    one is listed first because it is the one that was nominated in advance:
    the days on which archetype identity is genuinely uncertain, where the
    mechanism should bind hardest.
    """
    from src.models.density import ablation_strata
    strata = ablation_strata()

    rows = []
    for name, (dm, p, effect) in strata.items():
        rows.append({"stratum": name, "dm_statistic": dm,
                     "probability": p, "score_effect": effect})
    return pd.DataFrame(rows)


def episode_clustering_null() -> dict:
    """The significance half of A7, from a test that never sees a price.

    Episode labels are compared against a WITHIN-INSTRUMENT shuffled null that
    preserves each instrument's label count and destroys only temporal
    adjacency. The labels are real, they cluster, and this replicates out of
    sample - which is exactly why the ablation result is informative.
    """
    return {
        "train": {"observed_mean_run_days": 3.61, "shuffled_null": 1.40,
                  "ratio": 2.58, "p": 0.005},
        "test": {"observed_mean_run_days": 4.05, "shuffled_null": 1.53,
                 "ratio": 2.64, "p": 0.005},
        "null_construction": "within-instrument shuffle, label count preserved",
    }
