#!/usr/bin/env python3
"""Unified research pipeline for the FII flow-regimes project.

One command reproduces the entire research chain:

    python pipeline.py --all                 # everything, in order
    python pipeline.py --phase validation    # one phase
    python pipeline.py --stage panel_regression
    python pipeline.py --from-stage canonical_panel   # resume mid-chain
    python pipeline.py --model hmm_regime    # train/evaluate one model
    python pipeline.py --list                # show the stage manifest

Design (see README): stages are the original, gate-verified research
scripts, preserved byte-for-byte apart from two mechanical
substitutions (paths, engine import).  The pipeline supplies config,
seeding, logging and ordering.  A stage whose pre-registered gate
fails raises downstream, halting the chain — exactly the discipline
the research was built under.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fii.paths import assert_data_present, ensure_output_tree  # noqa: E402
from fii.runner import run_stage  # noqa: E402
from fii.stages.registry import (  # noqa: E402
    PHASE_ORDER, STAGES, by_name, phase_stages)

RUN_PHASES_DEFAULT = [p for p in PHASE_ORDER
                      if p not in ("audit", "phase2", "fhmm")]


def _print_list() -> None:
    for phase in PHASE_ORDER:
        tag = "" if phase != "audit" else "   (optional, read-only)"
        print(f"\n[{phase}]{tag}")
        for s in phase_stages(phase):
            print(f"  {s.name:24s} {s.desc}")


def _run(stages) -> int:
    ensure_output_tree()
    results = []
    for s in stages:
        print(f"\n>>> {s.phase}/{s.name}")
        r = run_stage(s.name, s.script)
        results.append(r)
        if not r.ok:
            print(f"\nSTAGE FAILED: {s.name} ({r.error and r.error.splitlines()[-1]})")
            print(f"log: {r.log_file}")
            break
    print("\n" + "=" * 66)
    for r in results:
        print(f"  {'PASS' if r.ok else 'FAIL'}  {r.name:24s} "
              f"{r.seconds:7.1f}s  {r.log_file.name}")
    print("=" * 66)
    return 0 if all(r.ok for r in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="run every phase except audit")
    g.add_argument("--phase", choices=PHASE_ORDER)
    g.add_argument("--stage", metavar="NAME")
    g.add_argument("--from-stage", metavar="NAME",
                   help="run --all order, starting at NAME")
    g.add_argument("--model", metavar="MODEL",
                   help="train+evaluate one registered model "
                        "(see fii/models)")
    g.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        _print_list()
        return 0

    assert_data_present()

    if args.model:
        from fii.models.registry import get_model
        m = get_model(args.model)
        m.train()
        m.predict()
        m.evaluate()
        m.save_outputs()
        return 0

    if args.stage:
        return _run([by_name(args.stage)])
    if args.phase:
        return _run(phase_stages(args.phase))
    ordered = [s for p in RUN_PHASES_DEFAULT for s in phase_stages(p)]
    if args.from_stage:
        names = [s.name for s in ordered]
        ordered = ordered[names.index(args.from_stage):]
    return _run(ordered)


if __name__ == "__main__":
    raise SystemExit(main())
