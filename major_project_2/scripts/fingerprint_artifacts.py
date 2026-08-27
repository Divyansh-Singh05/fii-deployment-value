"""Fingerprint the key research artifacts (before/after a full re-run).

Records shape, key-column hashes and archetype censuses so a full
pipeline re-run can be compared against the frozen (Colab-certified)
artifacts the paper numbers derive from.

Usage:
    python scripts/fingerprint_artifacts.py before
    python scripts/fingerprint_artifacts.py after   # prints a diff too
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fii.paths import DIAGNOSTICS, ISIN_MAPPING, VALIDATION_DATA  # noqa: E402

TARGETS = {
    "returns_panel": VALIDATION_DATA / "returns_panel.parquet",
    "returns_panel_v2": VALIDATION_DATA / "returns_panel_v2.parquet",
    "returns_panel_v3": VALIDATION_DATA / "returns_panel_v3.parquet",
    "states_v3": VALIDATION_DATA / "states_v3.parquet",
    "ca_factors": VALIDATION_DATA / "ca_adjustment_factors.parquet",
    "features_v2": ISIN_MAPPING / "stockday_features_v2.parquet",
    "states_calibrated": ISIN_MAPPING / "stockday_states_calibrated.parquet",
}


def fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"missing": True}
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    fp: dict = {"rows": lf.select(pl.len()).collect().item(),
                "cols": len(schema.names())}
    if "archetype" in schema.names():
        cen = (lf.group_by("archetype").len().collect()
                 .sort("archetype"))
        fp["census"] = dict(zip(cen["archetype"].to_list(),
                                cen["len"].to_list()))
    for c in ("cisin", "isin"):
        if c in schema.names():
            fp[f"n_{c}"] = (lf.select(pl.col(c).n_unique())
                            .collect().item())
    return fp


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "before"
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    out = {n: fingerprint(p) for n, p in TARGETS.items()}
    f = DIAGNOSTICS / f"fingerprints_{tag}.json"
    f.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {f}")
    for n, v in out.items():
        print(f"  {n:20s} {v}")
    if tag == "after":
        before = json.loads(
            (DIAGNOSTICS / "fingerprints_before.json").read_text())
        print("\n=== DIFF vs frozen ===")
        same = True
        for n in TARGETS:
            if before.get(n) != out.get(n):
                same = False
                print(f"  CHANGED {n}:\n    before {before.get(n)}"
                      f"\n    after  {out.get(n)}")
        print("  identical to frozen artifacts" if same else
              "  ^ regenerated artifacts differ — paper numbers came "
              "from the FROZEN versions (restore: scripts/setup_data.sh)")


if __name__ == "__main__":
    main()
