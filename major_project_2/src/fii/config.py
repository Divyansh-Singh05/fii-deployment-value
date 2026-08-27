"""Typed access to config/config.yaml.

Usage::

    from fii.config import CFG
    CFG["eras"]["train_end"]
    CFG.seed          # convenience attributes for common values
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from fii.paths import REPO_ROOT

_CONFIG_FILE = REPO_ROOT / "config" / "config.yaml"


class _Config(dict):
    """Dict with a few convenience attributes; single source of truth."""

    @property
    def seed(self) -> int:
        return int(self["reproducibility"]["seed"])

    @property
    def train_end(self) -> str:
        return str(self["eras"]["train_end"])

    @property
    def test_start(self) -> str:
        return str(self["eras"]["test_start"])

    @property
    def tcost_bps(self) -> float:
        return float(self["backtest"]["tcost_bps_oneway"])


@lru_cache(maxsize=1)
def load() -> _Config:
    with open(_CONFIG_FILE) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return _Config(raw)


CFG = load()
