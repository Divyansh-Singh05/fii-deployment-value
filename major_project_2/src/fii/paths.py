"""Central path resolution for the FII research pipeline.

Every stage imports its data locations from here instead of hardcoding
them.  The repository root is discovered relative to this file, so the
pipeline runs from any working directory.  ``FII_DATA_ROOT`` overrides
the data location without editing config.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DATA_ROOT = Path(os.environ.get("FII_DATA_ROOT", REPO_ROOT / "data"))

VALIDATION_DATA: Path = _DATA_ROOT / "VALIDATION_DATA"
ISIN_MAPPING: Path = _DATA_ROOT / "ISIN_MAPPING"

OUTPUTS: Path = REPO_ROOT / "outputs"
LOGS: Path = OUTPUTS / "logs"
FIGURES: Path = OUTPUTS / "figures"
TABLES: Path = OUTPUTS / "tables"
METRICS: Path = OUTPUTS / "metrics"
TRAINED_MODELS: Path = OUTPUTS / "trained_models"
PREDICTIONS: Path = OUTPUTS / "predictions"
VALIDATION_OUT: Path = OUTPUTS / "validation"
DIAGNOSTICS: Path = OUTPUTS / "diagnostics"
REGRESSIONS: Path = OUTPUTS / "regression_outputs"
DESCRIPTIVES: Path = OUTPUTS / "descriptive_statistics"


def ensure_output_tree() -> None:
    """Create the full outputs/ hierarchy (idempotent)."""
    for p in (OUTPUTS, LOGS, FIGURES, TABLES, METRICS, TRAINED_MODELS,
              PREDICTIONS, VALIDATION_OUT, DIAGNOSTICS, REGRESSIONS,
              DESCRIPTIVES):
        p.mkdir(parents=True, exist_ok=True)


def assert_data_present() -> None:
    """Fail fast with a helpful message if the data tree is missing."""
    missing = [str(p) for p in (VALIDATION_DATA, ISIN_MAPPING)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Data not found: " + ", ".join(missing)
            + "\nRun scripts/setup_data.sh (or set FII_DATA_ROOT) first."
        )
