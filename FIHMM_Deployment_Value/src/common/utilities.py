"""Config loading, path resolution, and artifact digesting.

Every stage resolves its inputs through `dataset()`, which resolves ONLY
package-relative paths under data/ and refuses quarantined ones. Nothing at
runtime reads outside this directory, so a pre-audit artifact from an upstream
tree cannot enter a computation by accident - the failure mode this programme
documents three instances of in its own source material.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PKG_ROOT / "config"


class QuarantinedArtifact(RuntimeError):
    """Raised when a stage asks for a path that predates the audit."""


class MissingDataset(FileNotFoundError):
    """Raised when a declared dataset file is absent from data/."""


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load one config file by stem, e.g. load_config('model_config')."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def _paths() -> dict[str, Any]:
    return load_config("paths")


def dataset(key: str) -> Path:
    """Resolve a declared dataset file. PACKAGE-RELATIVE ONLY.

    There is no escape hatch and no environment override: every input the
    reproduction path consumes lives under data/. If a file is missing, the
    fix is to re-run `python -m src.data.ingest`, not to point the package
    somewhere else.
    """
    paths = _paths()
    rel = paths["dataset"].get(key) or paths["reference"].get(key)
    if rel is None:
        known = sorted([*paths["dataset"], *paths["reference"]])
        raise KeyError(f"unknown dataset key {key!r}; declared keys are {known}")
    guard_quarantine(rel)
    path = PKG_ROOT / rel
    if not path.exists():
        raise MissingDataset(
            f"{rel} is absent. Run `python -m src.data.ingest` to rebuild the "
            f"data layer, or see data/source/README_NSDL.md."
        )
    return path


def guard_quarantine(path: str | Path) -> None:
    """Refuse any path carrying a quarantined infix."""
    s = str(path)
    for infix in _paths()["quarantined_infixes"]:
        if infix in s:
            raise QuarantinedArtifact(
                f"refusing {s}: carries quarantined infix '{infix}'. "
                f"These artifacts predate the audit that rebuilt the state "
                f"object and must never enter a reported number."
            )


def local(kind: str) -> Path:
    """Resolve a package-local directory, creating it if needed."""
    p = PKG_ROOT / _paths()["local"][kind]
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, for the provenance record."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def digest_line(path: str | Path) -> str:
    """One-line provenance stamp: digest, size, path."""
    p = Path(path)
    return f"{sha256(p)[:16]}  {p.stat().st_size:>12,}  {p}"
