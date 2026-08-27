"""Template for adding a NEW model — copy this file, nothing else.

Steps:
  1. cp _template.py my_model.py
  2. rename the class, set a unique ``name``
  3. implement the four verbs (native code or stage-backed)
  4. done — `python pipeline.py --model my_model` works immediately
     (registry auto-discovers BaseModel subclasses in this folder)

Contract every model MUST honour (the validation battery assumes it):
  * respect the frozen temporal split: fit on train <= 2021-04-30
    only; 2021-05/06 masked; test >= 2021-07-01 untouched
  * consume features ONLY from the feature store
    (ISIN_MAPPING/stockday_features_v2.parquet) — never rebuild them
  * emit stock-day predictions/states keyed by (cisin, TR_DATE) so the
    existing validation stages can evaluate them unchanged
  * pre-register evaluation bars BEFORE looking at test-era output
    (see docs/04_validation_framework.md)
"""
from __future__ import annotations

from pathlib import Path

from fii.models.base import BaseModel
from fii.paths import ISIN_MAPPING, PREDICTIONS


class MyNewModel(BaseModel):
    name = ""          # <- set a unique key, e.g. "xgboost_regime"

    FEATURES = ISIN_MAPPING / "stockday_features_v2.parquet"

    def train(self) -> None:
        raise NotImplementedError

    def predict(self) -> Path:
        out = PREDICTIONS / f"{self.name}_stockday.parquet"
        raise NotImplementedError

    def evaluate(self) -> None:
        # reuse the battery: point the validation stages at your
        # predictions artifact, or call self._run_stages(...) for the
        # generic ones (event_study, panel_regression, ...).
        raise NotImplementedError

    def save_outputs(self) -> None:
        raise NotImplementedError
