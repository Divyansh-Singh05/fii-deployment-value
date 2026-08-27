"""Path resolution for the standalone FII risk engine.

This project is self-contained: every module it imports lives in this
directory, and everything it writes goes to `outputs/` here. The ONE external
dependency is the read-only market/flow data, which is large enough that it is
not duplicated into the tree.

Data location, in priority order:
  1. $FII_DATA_ROOT                      explicit override
  2. <this directory>/data               a local copy or symlink
  3. ~/Desktop/Major Project 2/data      the original research tree

See DATA.md for what each file is and which ones the engine actually reads.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

_CANDIDATES = [
    Path(os.environ["FII_DATA_ROOT"]) if os.environ.get("FII_DATA_ROOT") else None,
    HERE / "data",
    Path.home() / "Desktop" / "Major Project 2" / "data",
]
DATA_ROOT: Path | None = next(
    (p for p in _CANDIDATES if p is not None and (p / "VALIDATION_DATA").is_dir()),
    None)

if DATA_ROOT is None:
    tried = "\n  ".join(str(p) for p in _CANDIDATES if p is not None)
    raise SystemExit(
        "FII risk engine: could not locate the data directory.\n"
        f"Looked for a VALIDATION_DATA/ subfolder under:\n  {tried}\n\n"
        "Point FII_DATA_ROOT at the directory that contains VALIDATION_DATA/ "
        "and ISIN_MAPPING/, e.g.\n"
        "  export FII_DATA_ROOT='/Users/you/Desktop/Major Project 2/data'\n"
        "or place a copy (or symlink) at ./data .")

VALIDATION_DATA: Path = DATA_ROOT / "VALIDATION_DATA"
ISIN_MAPPING:    Path = DATA_ROOT / "ISIN_MAPPING"

# Everything the engine WRITES stays inside this project.
OUTPUTS:         Path = Path(os.environ.get("FII_OUTPUTS", HERE / "outputs"))
LOGS:            Path = OUTPUTS / "logs"
FIGURES:         Path = OUTPUTS / "figures"
TABLES:          Path = OUTPUTS / "tables"
METRICS:         Path = OUTPUTS / "metrics"
TRAINED_MODELS:  Path = OUTPUTS / "trained_models"
PREDICTIONS:     Path = OUTPUTS / "predictions"
VALIDATION_OUT:  Path = OUTPUTS / "validation"
DIAGNOSTICS:     Path = OUTPUTS / "diagnostics"
PHASE3:          Path = OUTPUTS / "phase3"


def ensure_output_tree() -> None:
    for p in (LOGS, FIGURES, TABLES, METRICS, TRAINED_MODELS, PREDICTIONS,
              VALIDATION_OUT, DIAGNOSTICS, PHASE3):
        p.mkdir(parents=True, exist_ok=True)


ensure_output_tree()
