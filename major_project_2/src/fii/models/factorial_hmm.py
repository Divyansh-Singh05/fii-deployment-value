"""Factorial HMM challenger (stage-backed) — Module 17.

Two independent latent chains — direction (k=3) and concentration
(k=3) — emitting additively (Ghahramani & Jordan, 1997), estimated by
exact EM on the 9-state product space via a constrained
hmmlearn.GaussianHMM (fhmm_stages/fhmm_core.py).

Motivation: the naive backbone needed threshold OVERLAYS for the
concentration archetypes because a flat HMM's likelihood is captured
by the persistent direction axis (Module 2).  The factorial structure
gives concentration its own latent channel; archetypes are decoded
END-TO-END with no thresholds.  Whether that channel forms, and
whether its archetypes reproduce Table 1, are pre-registered gates
(17A G3, 17C V1-V3).

Code isolation: everything lives in fhmm_stages/ + this file.  The
naive HMM chain (models/hmm_stages/, hmm_regime.py) is untouched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fii.models.base import BaseModel
from fii.paths import ISIN_MAPPING, PREDICTIONS, TRAINED_MODELS


class FactorialHMMModel(BaseModel):
    name = "factorial_hmm"

    states_file = ISIN_MAPPING / "stockday_states_fhmm.parquet"

    def train(self) -> None:
        # fit on the frozen split + decode both eras + gates G1-G4
        self._run_stages("fhmm_train_oos")

    def predict(self) -> Path:
        if not self.states_file.exists():
            raise FileNotFoundError(
                f"{self.states_file} missing — run train() first")
        return self.states_file

    def evaluate(self) -> None:
        # same battery shape as the naive HMM: descriptives + the
        # Table-1 economics on FHMM labels (13A pattern)
        self._run_stages("fhmm_descriptives", "fhmm_table1")

    def save_outputs(self) -> None:
        dest = PREDICTIONS / self.states_file.name
        shutil.copy2(self.predict(), dest)
        marker = TRAINED_MODELS / "factorial_hmm.txt"
        marker.write_text(
            "stage-backed model; chain parameters in "
            "trained_models/fhmm_params.json\n"
            f"states: {dest}\n")
