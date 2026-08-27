"""Walk-forward vintages.

Parameters are frozen into DATED vintages and evaluated forward without
restatement. A vintage fitted at b0 is applied to days [b0, b0+REFIT) and never
revised, so a day's forecast uses only parameters that existed before it.

Two guards, both learned from defects found in review:

  MIN_PAIRS - no parameter is applied out of sample on fewer than 500 training
  pairs. An earlier run scored tilt vintages fitted on 0..500 pairs, and those
  noisy parameters asymmetrically handicapped the FEATURE-RICH models, since a
  featureless base has nothing to estimate badly.

  Nelder-Mead polish - BFGS on a flat small-sample likelihood can stop at
  path-dependent points, making the fit depend on the warm start. The polish
  makes it warm-start-independent to ~1e-10 in negative log-likelihood.
"""
from __future__ import annotations

import numpy as np

from src.common.utilities import load_config

_CFG = load_config("model_config")["walk_forward"]
FIRST_FIT = _CFG["first_fit"]
REFIT = _CFG["refit"]
MIN_PAIRS = _CFG["min_pairs"]
CLAMP = tuple(_CFG["clamp"])
SCHEME = _CFG["scheme"]


def vintage_boundaries(n: int) -> list[tuple[int, int]]:
    """[(fit_index, apply_end)] - the dated vintages, in order."""
    return [(b0, min(b0 + REFIT, n)) for b0 in range(FIRST_FIT, n, REFIT)]


def training_slice(b0: int) -> slice:
    """The rows a vintage fitted at b0 may see.

    EXPANDING by construction. This is the choice that removes the
    Giacomini-White licence for normal critical values on the NESTED gate, and
    it is why the surviving application carries a block bootstrap and a
    Clark-West adjustment alongside its Diebold-Mariano statistic. Switching to
    a fixed rolling window would license the unadjusted comparison but would
    change every gate number in the manuscript.
    """
    if SCHEME != "expanding":
        raise NotImplementedError(
            f"scheme '{SCHEME}' is declared in config but this package "
            f"reproduces the expanding-window results the manuscript reports"
        )
    return slice(0, b0)


def sufficient_history(valid: np.ndarray, b0: int) -> bool:
    return int(np.asarray(valid)[training_slice(b0)].sum()) >= MIN_PAIRS


def apply_tilt(X_block: np.ndarray, theta: np.ndarray, s: float) -> np.ndarray:
    """Scale multiplier for one vintage's application window, clamped."""
    if X_block.shape[1] == 0:
        tilt = np.ones(len(X_block))
    else:
        tilt = np.exp(X_block @ theta)
    return s * np.clip(tilt, *CLAMP)
