"""Stage executor.

The research was developed as self-contained, sequentially-verified
Colab scripts, each with pre-registered PASS/FAIL gates in its printed
output.  That byte-level provenance is an asset (the validation log
cites those exact scripts), so stages are preserved as scripts and
executed here rather than rewritten into functions.

The runner provides what Colab provided, but reproducibly:
  * config-driven paths (fii.paths replaces /content/drive)
  * deterministic seeding before every stage
  * stdout/stderr captured to outputs/logs/<timestamp>_<stage>.log
  * hard stop on stage failure (a failed gate raises downstream)
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import runpy
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fii.config import CFG
from fii.paths import LOGS, ensure_output_tree


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        for st in self._streams:
            try:
                st.write(s)
            except ValueError:  # stream already closed at teardown
                pass
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except ValueError:
                pass


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    log_file: Path
    error: str | None = None


def run_stage(name: str, script: Path) -> StageResult:
    """Execute one stage script with seeding, logging and isolation."""
    ensure_output_tree()
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"{ts}_{name}.log"
    np.random.seed(CFG.seed)

    t0 = time.time()
    ok, err = True, None
    with open(log_file, "w") as fh:
        tee_out = _Tee(sys.stdout, fh)
        tee_err = _Tee(sys.stderr, fh)
        header = (f"=== stage {name} | {script} | seed={CFG.seed} "
                  f"| {ts} ===\n")
        tee_out.write(header)
        with contextlib.redirect_stdout(tee_out), \
                contextlib.redirect_stderr(tee_err):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as e:  # scripts may sys.exit on gate fail
                ok = (e.code in (None, 0))
                err = None if ok else f"SystemExit({e.code})"
            except Exception:
                ok = False
                err = traceback.format_exc()
                tee_err.write(err)
    return StageResult(name, ok, time.time() - t0, log_file, err)
