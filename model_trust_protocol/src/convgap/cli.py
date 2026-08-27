"""Command-line entry point.

convgap applications   list registered applications and ladder coverage
convgap verify         trace every number to the artifact that produced it
convgap table1         build the friction-ladder exhibit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convgap import exhibits, lockfile
from convgap.friction import FrictionLevel
from convgap.provenance import (
    SourceTier,
    SourceTree,
    VerificationStatus,
    load_source_trees,
)
from convgap.registry import all_applications, by_friction, controls, coverage_gaps
from convgap.role import Role

_DEFAULT_OUT = Path("outputs/exhibits")


def _trees(args: argparse.Namespace) -> dict[str, SourceTree]:
    trees = load_source_trees(Path(args.config) if args.config else None)
    for tree in trees.values():
        marker = "ok " if tree.available else "MISSING"
        print(f"  [{marker}] {tree.key:<9} {tree.root}", file=sys.stderr)
    return trees


def cmd_applications(_: argparse.Namespace) -> int:
    apps = all_applications()
    print(f"{len(apps)} applications registered\n")
    print("the dataset under stress, ordered by where its value is lost:\n")
    for level, group in by_friction():
        header = "survives every level" if level is None else f"L{int(level)} {level.label}"
        question = "" if level is None else f"  -- {level.question}"
        print(f"  {header}{question}")
        for app in group:
            tag = " (control)" if app.role is Role.CONTROL else ""
            print(f"      {app.key}  {app.title}{tag}")
            print(f"            instrument: {app.layer}")
        print()

    n_ctrl = len(controls())
    n_apps = len(apps) - n_ctrl
    survivors = [a for a in apps if a.fails_at is None and a.role is Role.APPLICATION]
    print(f"applications of the dataset : {n_apps}")
    print(f"external controls           : {n_ctrl}")
    print(f"surviving every level       : {len(survivors)}")

    if not survivors:
        print(
            "\nWARNING: no application survives the ladder. Without one the exhibit "
            "reports that the dataset is worthless rather than mapping where its "
            "value stops, which is a weaker and less falsifiable claim.",
            file=sys.stderr,
        )
    if not controls():
        print(
            "\nWARNING: no external control. Without one, a failure cannot be "
            "attributed to the dataset rather than to the evaluation transition.",
            file=sys.stderr,
        )
    gaps = coverage_gaps()
    if gaps:
        print("uncovered instruments:", ", ".join(str(g) for g in gaps), file=sys.stderr)
    unused = [lv for lv in FrictionLevel if not any(a.fails_at is lv for a in apps)]
    if unused:
        print(
            "friction levels with no casualty:",
            ", ".join(lv.label for lv in unused),
            file=sys.stderr,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    print("source trees:", file=sys.stderr)
    trees = _trees(args)
    lock_path = Path(args.lock) if args.lock else lockfile.DEFAULT_PATH
    locks = lockfile.load(lock_path)
    print(f"  lock: {lock_path} ({len(locks)} artifact(s) recorded)", file=sys.stderr)
    print()

    untraced, secondary, mismatched = 0, 0, 0
    observed: dict[str, str] = {}
    for app in all_applications():
        result = app.resolve(trees, locks)
        for evset in (result.significance, result.value):
            for item in evset.items:
                if item.source.sha256:
                    observed[lockfile.key(item.source.tree, item.source.artifact)] = (
                        item.source.sha256
                    )
                if item.status is VerificationStatus.MISMATCH:
                    mismatched += 1
        if not result.status.is_publishable:
            untraced += 1
        if result.tier is SourceTier.SECONDARY:
            secondary += 1
        flag = "PASS" if result.publishable else "FAIL"
        print(f"[{flag}] {result.key}  {result.status:<11} {result.tier:<9} {result.title}")
        for label, evset in (("sig", result.significance), ("val", result.value)):
            for item in evset.items:
                extra = f"  <- {item.extraction_note}" if item.extraction_note else ""
                print(f"         {label}  {item.status:<11} {item.source.describe()}{extra}")

    from convgap import facts as _facts

    for claim in _facts.verify_all(trees, locks):
        src = claim.evidence.source
        if src.sha256:
            observed[lockfile.key(src.tree, src.artifact)] = src.sha256

    total = len(all_applications())
    print()
    print(f"traced to a live artifact : {total - untraced}/{total}")
    print(f"sourced from computation  : {total - secondary}/{total}")
    if mismatched:
        print(
            f"\n{mismatched} value(s) MISMATCH: the artifact does not hold the number "
            f"this application declares. Fix the declaration, never the tolerance.",
            file=sys.stderr,
        )
    if args.record:
        written = lockfile.save(observed, lock_path)
        print(
            f"\nrecorded {len(observed)} artifact digest(s) -> {written}",
            file=sys.stderr,
        )
    return 0 if not (untraced or secondary or mismatched) else 1


def cmd_table1(args: argparse.Namespace) -> int:
    trees = _trees(args)
    locks = lockfile.load(Path(args.lock) if args.lock else None)
    table = exhibits.build([a.resolve(trees, locks) for a in all_applications()])
    print()
    print(table.ladder())
    print()
    print(table.to_markdown())
    print()
    print(table.audit_note())
    out = Path(args.out or _DEFAULT_OUT) / "table1_friction_ladder.csv"
    table.to_csv(out)
    print(f"\nwritten: {out}", file=sys.stderr)
    return 0 if not table.blocking else 1


def cmd_replicate(args: argparse.Namespace) -> int:
    """Re-derive the aggregate-flow screen from its pre-registered specification."""
    from convgap.replication import aggregate_flow as af
    from convgap.replication import weekly_screen as ws

    trees = _trees(args)
    root = trees["research"].root
    vdata, imap = root / "data" / "VALIDATION_DATA", root / "data" / "ISIN_MAPPING"
    out_dir = Path(args.out or "outputs/replication")

    daily = af.run_screen(vdata, imap)
    print()
    print(af.render(daily))
    print()
    print(af.sensitivity(vdata, imap))
    af.to_csv(daily, out_dir / "aggregate_flow_screen.csv")

    weekly = ws.run_screen(vdata, imap)
    print()
    print(ws.render(weekly))
    af.to_csv(weekly, out_dir / "weekly_screen.csv")

    from convgap.replication import descriptives as dsc

    facts = dsc.run(vdata, imap)
    print()
    print(dsc.render(facts))
    dsc.to_csv(facts, out_dir / "descriptives.csv")

    print(f"\nwritten: {out_dir}/aggregate_flow_screen.csv", file=sys.stderr)
    print(f"written: {out_dir}/weekly_screen.csv", file=sys.stderr)
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    from convgap import facts, lockfile

    trees = _trees(args)
    locks = lockfile.load(Path(args.lock) if args.lock else None)
    claims = facts.verify_all(trees, locks)
    print()
    print(facts.summarise(claims))
    return 0 if all(c.ok for c in claims) else 1


def cmd_manuscript(args: argparse.Namespace) -> int:
    from convgap import lockfile, manuscript

    trees = _trees(args)
    locks = lockfile.load(Path(args.lock) if args.lock else None)
    verified = manuscript.verified_values(trees, locks)
    paper = Path(args.paper or "paper")
    unmatched = manuscript.scan(paper, verified)
    print()
    print(f"{len(verified)} verified value forms; scanning {paper}/")
    print()
    print(manuscript.report(unmatched))
    return 0 if not unmatched else 1


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Re-run every computation the exhibit depends on.

    After this, no figure in the exhibit is inherited: each descends from a
    process started here, in a recorded session.
    """
    from convgap.recipe import Recipe, execute

    trees = _trees(args)
    repo = Path(__file__).resolve().parents[2]
    apps = all_applications()
    if args.only:
        wanted = {k.strip().upper() for k in args.only.split(",")}
        apps = [a for a in apps if a.key in wanted]
        if not apps:
            print(f"no application matches {args.only!r}", file=sys.stderr)
            return 2

    # Distinct recipes only: several applications share a producing stage.
    seen: dict[tuple[str, ...], str] = {}
    plan: list[tuple[str, Recipe]] = []
    for app in apps:
        for rec in app.recipes():
            sig = tuple(rec.argv)
            if sig in seen:
                print(f"  {app.key}: shares a recipe already queued for {seen[sig]}")
                continue
            seen[sig] = app.key
            plan.append((app.key, rec))

    total = sum(r.est_seconds for _, r in plan)
    print(f"\n{len(plan)} distinct recipe(s), estimated {total // 60}m {total % 60}s\n")
    if args.dry_run:
        for key, rec in plan:
            print(f"  {key}  [{rec.tree}]  {' '.join(rec.resolve_argv(trees, repo))}")
        return 0

    outcomes = []
    for key, rec in plan:
        print(f"running {key} ...", flush=True)
        out = execute(key, rec, trees, repo, timeout=args.timeout)
        print(f"  {out}")
        if not out.ok:
            print(out.tail, file=sys.stderr)
        outcomes.append(out)

    failed = [o for o in outcomes if not o.ok]
    print()
    for o in outcomes:
        print(f"  {o}")
    print(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} recipes succeeded")
    if failed:
        return 1
    print("\nNow run: convgap verify --record")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convgap",
        description="What is an alternative dataset worth under ascending "
        "levels of real-world friction?",
    )
    parser.add_argument("--config", help="path to sources.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "applications", help="list registered applications and ladder coverage"
    ).set_defaults(func=cmd_applications)

    p_verify = sub.add_parser("verify", help="trace every number to its source")
    p_verify.add_argument("--lock", help="path to the digest lock file")
    p_verify.add_argument(
        "--record",
        action="store_true",
        help="write observed digests to the lock file, marking the current state "
        "of the source artifacts as the verified one",
    )
    p_verify.set_defaults(func=cmd_verify)

    p_table = sub.add_parser("table1", help="build the friction-ladder exhibit")
    p_table.add_argument("--out", help="output directory")
    p_table.add_argument("--lock", help="path to the digest lock file")
    p_table.set_defaults(func=cmd_table1)

    p_rep = sub.add_parser("replicate", help="re-derive figures that have no producing artifact")
    p_rep.add_argument("--out", help="output directory")
    p_rep.set_defaults(func=cmd_replicate)

    p_repro = sub.add_parser("reproduce", help="re-run every producing computation, end to end")
    p_repro.add_argument("--only", help="comma-separated application keys")
    p_repro.add_argument(
        "--dry-run", action="store_true", help="print the commands without running"
    )
    p_repro.add_argument("--timeout", type=int, default=3600)
    p_repro.set_defaults(func=cmd_reproduce)

    p_facts = sub.add_parser("facts", help="verify the data section's claims")
    p_facts.add_argument("--lock", help="path to the digest lock file")
    p_facts.set_defaults(func=cmd_facts)

    p_ms = sub.add_parser("manuscript", help="check the paper's numerals")
    p_ms.add_argument("--paper", help="manuscript directory")
    p_ms.add_argument("--lock", help="path to the digest lock file")
    p_ms.set_defaults(func=cmd_manuscript)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
