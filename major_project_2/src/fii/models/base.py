"""Common model interface.

Every model in fii/models exposes the same four verbs so the pipeline
(and any future experiment runner) can drive it without knowing what it
is.  Adding a new model = adding ONE file in fii/models/ that defines a
BaseModel subclass with a unique ``name`` — the registry discovers it
automatically (see registry.py and _template.py).

Two implementation styles are supported and both are legitimate here:

* native models implement train/predict directly (future work);
* stage-backed models delegate to the preserved, gate-verified research
  scripts (the HMM and LightGBM below).  This is a deliberate choice:
  the scripts ARE the certified research artifact, and wrapping them
  keeps the published numbers byte-reproducible.
"""
from __future__ import annotations

import abc
from pathlib import Path

from fii.runner import run_stage
from fii.stages.registry import by_name


class BaseModel(abc.ABC):
    """Interface: train() -> predict() -> evaluate() -> save_outputs()."""

    #: unique registry key, e.g. "hmm_regime"
    name: str = ""

    @abc.abstractmethod
    def train(self) -> None:
        """Fit the model (must respect the frozen temporal split)."""

    @abc.abstractmethod
    def predict(self) -> Path:
        """Produce/refresh predictions; return the artifact path."""

    @abc.abstractmethod
    def evaluate(self) -> None:
        """Run the model's evaluation battery."""

    @abc.abstractmethod
    def save_outputs(self) -> None:
        """Copy/export artifacts into outputs/trained_models etc."""

    # ---- helper for stage-backed models -------------------------------
    @staticmethod
    def _run_stages(*stage_names: str) -> None:
        for n in stage_names:
            s = by_name(n)
            r = run_stage(s.name, s.script)
            if not r.ok:
                raise RuntimeError(
                    f"stage '{n}' failed — see {r.log_file}")
