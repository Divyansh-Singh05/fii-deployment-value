"""LightGBM challenger model (stage-backed).

Role in the research: NOT a replacement for the HMM but the
pre-registered "is the regime model the bottleneck?" test (Module 8).
train()/evaluate() run the challenger against the frozen regime
baseline; the demeaning check (Module 8B) separates dynamic signal
from static characteristics and holds the verdict to the
pre-registered bar (non-overlap spread t > 2 — not met).
"""
from __future__ import annotations

from pathlib import Path

from fii.models.base import BaseModel
from fii.paths import ISIN_MAPPING


class LightGBMChallenger(BaseModel):
    name = "lightgbm_gbt"

    def train(self) -> None:
        self._run_stages("gbt_challenger")

    def predict(self) -> Path:
        # the challenger's predictions are produced inside the stage;
        # the feature store is its immutable input.
        return ISIN_MAPPING / "stockday_features_v2.parquet"

    def evaluate(self) -> None:
        self._run_stages("demeaning_check")

    def save_outputs(self) -> None:
        # stage prints + logs are the artifact (documented negative
        # result: dynamic increment below the pre-registered bar).
        pass
