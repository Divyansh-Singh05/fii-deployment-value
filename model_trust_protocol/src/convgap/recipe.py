"""How each artifact is regenerated.

An exhibit whose numbers can only be checked, never rebuilt, is reproducible in
a weak sense: a reader can confirm we transcribed a file correctly, but not that
the file is what the code produces. This module closes that gap by making every
application declare the command that regenerates the artifact it depends on.

``convgap reproduce`` executes those commands in order. After it, every figure
in the exhibit descends from a computation run on this machine, from this
repository, in a recorded session - regardless of whether the computation lives
here or in a source tree.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from convgap.provenance import SourceTree


class RecipeKind(StrEnum):
    """Where the regenerating computation lives."""

    PIPELINE_STAGE = "pipeline stage"
    """A stage of the source programme's pipeline, invoked in its own tree."""

    LOCAL = "local"
    """A module in this repository, re-deriving a figure the source programme
    reports but does not produce."""


@dataclass(frozen=True, slots=True)
class Recipe:
    """A reproducible command and the artifacts it writes.

    Parameters
    ----------
    kind
        Whether the computation lives in a source tree or here.
    tree
        Key of the tree the command runs in, or ``"local"``.
    argv
        Command, as a list. The first element may be ``"{python}"``, which is
        substituted with the interpreter of the tree's own virtual environment
        so that the source programme runs under the dependencies it was
        developed against, not ours.
    produces
        Artifacts written, relative to ``tree``. Used to report what changed.
    est_seconds
        Rough wall time, so a caller can plan. Not enforced.
    """

    kind: RecipeKind
    tree: str
    argv: Sequence[str]
    produces: Sequence[str]
    est_seconds: int = 60
    note: str = ""

    def resolve_argv(self, trees: dict[str, SourceTree], repo_root: Path) -> list[str]:
        root = repo_root if self.tree == "local" else trees[self.tree].root
        out: list[str] = []
        for token in self.argv:
            if token == "{python}":
                venv = root / ".venv" / "bin" / "python"
                out.append(str(venv) if venv.is_file() else "python3")
            else:
                out.append(token)
        return out

    def cwd(self, trees: dict[str, SourceTree], repo_root: Path) -> Path:
        return repo_root if self.tree == "local" else trees[self.tree].root


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Result of executing one recipe."""

    key: str
    ok: bool
    seconds: float
    returncode: int
    tail: str

    def __str__(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        return f"[{flag}] {self.key:<4} {self.seconds:6.1f}s  rc={self.returncode}"


def execute(
    key: str,
    recipe: Recipe,
    trees: dict[str, SourceTree],
    repo_root: Path,
    *,
    timeout: int = 3600,
) -> RunOutcome:
    """Run one recipe, capturing enough output to diagnose a failure."""
    argv = recipe.resolve_argv(trees, repo_root)
    cwd = recipe.cwd(trees, repo_root)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RunOutcome(key, False, time.monotonic() - started, -1, "timed out")
    elapsed = time.monotonic() - started
    combined = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(combined.strip().splitlines()[-12:])
    return RunOutcome(key, proc.returncode == 0, elapsed, proc.returncode, tail)
