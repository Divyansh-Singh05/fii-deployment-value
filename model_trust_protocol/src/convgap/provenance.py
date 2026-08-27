"""Source tracing for every number that reaches an exhibit.

The paper's own thesis is that a result is only as good as the audit around
it. This module enforces that on the paper: an :class:`Evidence` value cannot
enter a table without naming the artifact it came from, and the artifact's
digest is recorded so that a later change is reported rather than absorbed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

_CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
_CHUNK: Final = 1 << 20


class VerificationStatus(StrEnum):
    """Lifecycle of a single traced number."""

    UNVERIFIED = "unverified"
    """Transcribed from a project document; not yet traced to a live artifact."""

    VERIFIED = "verified"
    """Traced to an artifact whose digest matches the recorded one."""

    STALE = "stale"
    """Artifact exists but its digest differs from the recorded one."""

    MISSING = "missing"
    """Artifact could not be located under the configured source tree."""

    OUTDATED = "outdated"
    """Artifact is older than the code that writes it, so it may not reflect
    what that code now produces."""

    MISMATCH = "mismatch"
    """Artifact was located, but it does not contain the declared value."""

    DERIVED = "derived"
    """Computed by this package from inputs that are themselves verified."""

    @property
    def is_publishable(self) -> bool:
        """Whether a number in this state may appear in a submitted exhibit."""
        return self in (VerificationStatus.VERIFIED, VerificationStatus.DERIVED)


class SourceTier(StrEnum):
    """How close a number is to the computation that produced it.

    A digest match on a project document proves only that the document has not
    changed - not that the number in it is right. Promoting every exhibit row
    from ``SECONDARY`` to ``PRIMARY`` is therefore a distinct piece of work from
    verification, and the two are tracked separately so neither can stand in for
    the other.
    """

    PRIMARY = "primary"
    """A data artifact written by a pipeline stage: parquet, csv, npz, or a run log."""

    SECONDARY = "secondary"
    """A prose document transcribing a number computed elsewhere."""

    @classmethod
    def infer(cls, artifact: str) -> SourceTier:
        primary_suffixes = (".parquet", ".csv", ".npz", ".json", ".log", ".tsv", ".tex")
        return cls.PRIMARY if artifact.endswith(primary_suffixes) else cls.SECONDARY


class SourceTreeError(RuntimeError):
    """Raised when a source tree is referenced but cannot be resolved on disk."""


@dataclass(frozen=True, slots=True)
class SourceTree:
    """A read-only project directory that supplies artifacts."""

    key: str
    root: Path
    description: str

    def resolve(self, artifact: str) -> Path:
        return self.root / artifact

    @property
    def available(self) -> bool:
        return self.root.is_dir()


def load_source_trees(config_path: Path | None = None) -> dict[str, SourceTree]:
    """Read ``config/sources.yaml``, applying environment-variable overrides."""
    path = config_path or _CONFIG_PATH
    spec = yaml.safe_load(path.read_text())
    trees: dict[str, SourceTree] = {}
    for key, entry in spec["trees"].items():
        override = os.environ.get(entry["env_var"])
        raw = override if override else entry["default"]
        trees[key] = SourceTree(
            key=key,
            root=Path(raw).expanduser().resolve(),
            description=entry.get("description", "").strip(),
        )
    return trees


@dataclass(frozen=True, slots=True)
class Source:
    """Where a number came from, precisely enough for a third party to check.

    Parameters
    ----------
    tree
        Key into ``config/sources.yaml`` (``"research"`` or ``"engine"``).
    artifact
        Path relative to that tree's root.
    locator
        Human-readable instruction for finding the number inside the artifact —
        a column and row, a log-line pattern, a section heading. This is what a
        referee follows; it must be specific enough to be actionable.
    sha256
        Digest of the artifact recorded at verification time. ``None`` until
        the evidence has been verified at least once.
    verified_on
        Date of the last successful verification.
    """

    tree: str
    artifact: str
    locator: str
    producer: str | None = None
    """Path, relative to the same tree, of the code that writes this artifact.

    Supplying it enables the freshness gate. An artifact older than its producer
    may not reflect what that producer now emits, and no digest or value check
    can detect that: both pass happily on a stale file. Copies are the common
    case - a reporting step duplicating a table to a second location, then
    falling behind its origin.
    """
    sha256: str | None = None
    verified_on: date | None = None

    @property
    def tier(self) -> SourceTier:
        """Whether this points at a computed artifact or at prose describing one."""
        return SourceTier.infer(self.artifact)

    def path(self, trees: dict[str, SourceTree]) -> Path:
        if self.tree not in trees:
            raise SourceTreeError(f"unknown source tree {self.tree!r}")
        return trees[self.tree].resolve(self.artifact)

    def __str__(self) -> str:
        return f"{self.tree}:{self.artifact} [{self.locator}]"

    def describe(self) -> str:
        return f"({self.tier[:4]}) {self}"


def digest(path: Path) -> str:
    """SHA-256 of a file, streamed so large parquet artifacts are cheap."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check(
    source: Source,
    trees: dict[str, SourceTree],
    locks: dict[str, str] | None = None,
) -> tuple[VerificationStatus, Source]:
    """Compare a source against the filesystem and the recorded digest.

    Returns the resulting status and a copy of the source carrying the digest
    observed on disk. Recording the digest is the point: it converts "I read
    this number once" into "this number came from this exact file".
    """
    path = source.path(trees)
    if not path.is_file():
        return VerificationStatus.MISSING, source

    if source.producer is not None:
        producer = trees[source.tree].resolve(source.producer)
        if producer.is_file() and producer.stat().st_mtime > path.stat().st_mtime:
            return VerificationStatus.OUTDATED, source

    observed = digest(path)
    stamped = replace(source, sha256=observed, verified_on=date.today())
    recorded = (locks or {}).get(f"{source.tree}:{source.artifact}")
    if recorded is None:
        return VerificationStatus.UNVERIFIED, stamped
    if recorded != observed:
        return VerificationStatus.STALE, stamped
    return VerificationStatus.VERIFIED, stamped
