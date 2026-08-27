"""The main model: hybrid HMM regime detector (stage-backed).

Architecture (Modules 2-3, docs/03_models.md):
  * k=3 diagonal-Gaussian HMM on directional flow features gives a
    persistent SELL / NEUTRAL / BUY backbone (dwell ~13-17 days);
  * concentration archetypes (HOSTAGE, SHARK_DIST, SHARK_ACC) are
    overlay rules on frozen TRAIN-quantile thresholds, because the HMM
    cannot natively form concentration states (persistence dominates
    the likelihood — Module 2 finding).

Everything is frozen at the temporal split (train <= 2021-04-30);
predict() decodes both eras with the frozen parameters.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fii.models.base import BaseModel
from fii.paths import ISIN_MAPPING, PREDICTIONS, TRAINED_MODELS


class HMMRegimeModel(BaseModel):
    name = "hmm_regime"

    #: artifact produced by the calibrated pipeline
    states_file = ISIN_MAPPING / "stockday_states_calibrated.parquet"

    def train(self) -> None:
        # backbone fit on the frozen split + overlay threshold
        # calibration (TRAIN era only; falsification protocol inside)
        self._run_stages("hmm_train_oos", "threshold_calibration")

    def predict(self) -> Path:
        # decoding happens inside the training stages (both eras with
        # frozen parameters); predict() exposes the artifact.
        if not self.states_file.exists():
            raise FileNotFoundError(
                f"{self.states_file} missing — run train() first")
        return self.states_file

    def evaluate(self) -> None:
        # signatures / census / transitions / OOS replication
        self._run_stages("model_descriptives")

    def save_outputs(self) -> None:
        dest = PREDICTIONS / self.states_file.name
        shutil.copy2(self.predict(), dest)
        marker = TRAINED_MODELS / "hmm_regime.txt"
        marker.write_text(
            "stage-backed model; parameters live inside the states "
            f"artifact provenance.\nstates: {dest}\n")
