"""Regenerate the manuscript figures. Every figure, or any one on demand.

    python reproduction/reproduce_figures.py                 # all figures
    python reproduction/reproduce_figures.py 3               # just Figure 3
    python reproduction/reproduce_figures.py 1 2 6           # a subset
    python reproduction/reproduce_figures.py --list          # what exists

Each figure is a single function in analysis/figures.py, reads only from
data/, and writes a 400 dpi PNG into outputs/figures/. Nothing here reaches
outside the package.

LABELLING CONSTRAINT: level 1 is "Persistence", never "Replication". The test
asks whether an effect estimated on the training era holds on a frozen later
era drawn from a shifted cross-section (441 of 1,028 instruments common). That
is out-of-sample persistence under a single research design - strictly weaker
than replication in the credibility literature's sense.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import digest_line

# number -> (function name, one-line description, placement)
REGISTRY: dict[str, tuple[str, str, str]] = {
    "1": ("figure_1_framework",
          "The deployment-validation architecture and the six constraints", "main"),
    "2d": ("figure_2_design",
           "Frozen split, embargo, and walk-forward vintage design", "main"),
    "2": ("figure_2_attrition",
          "Application attrition: 7 -> 7 -> 6 -> 4 -> 2 -> 1", "main"),
    "3": ("figure_3_availability",
          "A1: coefficient and t across information lags", "main"),
    "4": ("figure_4_competition",
          "A8 vs A2: the same tilt against two baselines", "main"),
    "5": ("figure_5_discrimination_vs_decision",
          "A4: ROC against realised decision value", "main"),
    "6": ("figure_6_execution_frontier",
          "A5: the transaction-cost frontier", "main"),
    "7": ("figure_7_survivor_rolling",
          "A8: rolling out-of-sample advantage", "supplementary"),
    "8": ("figure_8_survivor_fragility",
          "A8: how its inferential status changes under stricter accounting",
          "supplementary"),
    "9": ("figure_9_data_pipeline",
          "Data construction and its validation gates", "supplementary"),
    "10": ("figure_10_external_control",
           "Cext: a non-flow predictor failing identically", "supplementary"),
}


def build(keys: list[str]) -> int:
    log = get_logger("reproduce_figures")
    banner(log, "REGENERATING FIGURES")
    import analysis.figures as F

    failed = 0
    for k in keys:
        fname, desc, where = REGISTRY[k]
        fn = getattr(F, fname, None)
        if fn is None:
            verdict(log, f"Figure {k}", "SKIP", f"{fname} not implemented")
            continue
        try:
            path = fn()
        except Exception as exc:                       # noqa: BLE001
            verdict(log, f"Figure {k}", "FAIL", f"{type(exc).__name__}: {exc}")
            failed += 1
            continue
        verdict(log, f"Figure {k}  ({where})", "PASS", digest_line(path))
        log.info(f"        {desc}")

    log.info("")
    log.info("  Level 1 is labelled 'Persistence' — never 'Replication'.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--list" in argv:
        print(f"{'fig':>4}  {'placement':<14} function")
        for k, (fname, desc, where) in REGISTRY.items():
            print(f"{k:>4}  {where:<14} {fname}")
            print(f"      {'':<14} {desc}")
        return 0

    keys = [a for a in argv if not a.startswith("-")]
    if not keys:
        keys = list(REGISTRY)
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}. "
              f"Known: {', '.join(REGISTRY)}", file=sys.stderr)
        return 2
    return build(keys)


if __name__ == "__main__":
    raise SystemExit(main())
