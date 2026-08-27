"""Run logging with a verdict vocabulary.

Every stage that tests something emits PASS / FAIL / STALE / MISMATCH through
`verdict()`. `outputs/logs/` is the durable record; the console mirror is for
the operator.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

from .utilities import local

VERDICTS = {"PASS", "FAIL", "STALE", "MISMATCH", "SKIP", "NOT ESTABLISHED"}


def get_logger(stage: str) -> logging.Logger:
    """Logger writing to outputs/logs/<timestamp>_<stage>.log and stdout."""
    log = logging.getLogger(stage)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(local("logs") / f"{stamp}_{stage}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def verdict(log: logging.Logger, label: str, value: str, detail: str = "") -> None:
    """Emit one verdict line. Unknown verdicts are a programming error."""
    if value not in VERDICTS:
        raise ValueError(f"unknown verdict {value!r}; expected one of {VERDICTS}")
    log.info(f"  {label:<52s} {value:<15s} {detail}")


def banner(log: logging.Logger, title: str) -> None:
    log.info("=" * 78)
    log.info(title)
    log.info("=" * 78)
