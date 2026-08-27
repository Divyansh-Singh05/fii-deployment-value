# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 4C · VALIDATION DATA AUDIT — formats, dtypes, heads, consistency
#
# Recursively scans VALIDATION DATA/ (top-level files + subfolders like
# "block deals/", "short deals/", "bulk deals/"), reports schema + preview
# for every file, and cross-checks schema consistency within any group of
# multiple files (yearly bhavcopy parquets, multi-file deal folders, BSE vs
# NSE corporate actions) — the exact class of bug that bit us twice already.
# ============================================================================
from pathlib import Path
import polars as pl
import re
from collections import defaultdict

DRIVE = VALIDATION_DATA

def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

files = sorted([f for f in DRIVE.rglob("*") if f.is_file()])
print(f"=== {len(files)} files under {DRIVE} (recursive) ===\n")
for f in files:
    rel = f.relative_to(DRIVE)
    print(f"  {str(rel):<60} {fmt_bytes(f.stat().st_size):>10}  .{f.suffix.lstrip('.')}")

# ── per-file inspection ─────────────────────────────────────────────────
def inspect_parquet(fp):
    lf = pl.scan_parquet(fp)
    schema = lf.collect_schema()
    try:
        n = lf.select(pl.len()).collect().item()
    except Exception:
        n = None
    head = lf.head(5).collect()
    print(f"    rows: {n if n is not None else '?'}   cols: {len(schema)}")
    print(f"    schema:")
    for name, dtype in zip(schema.names(), schema.dtypes()):
        print(f"      {name:<30} {dtype}")
    print(f"    head:\n{head}")

def inspect_csv(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = [next(fh) for _ in range(3)]
        print(f"    raw first lines:")
        for l in raw_lines: print(f"      {l.rstrip()[:150]}")
    except Exception as e:
        print(f"    could not peek raw lines: {e}")

    try:
        raw = pl.read_csv(fp, infer_schema_length=0, n_rows=2000,
                          ignore_errors=True, truncate_ragged_lines=True)
        print(f"    columns ({len(raw.columns)}): {raw.columns}")
    except Exception as e:
        print(f"    FAILED raw string read: {e}")
        return
    try:
        inferred = pl.read_csv(fp, n_rows=2000, ignore_errors=True, truncate_ragged_lines=True)
        print(f"    inferred dtypes:")
        for name, dtype in zip(inferred.columns, inferred.dtypes):
            print(f"      {name:<30} {dtype}")
        try:
            n = pl.scan_csv(fp, ignore_errors=True, truncate_ragged_lines=True).select(pl.len()).collect().item()
        except Exception:
            n = "?"
        print(f"    rows (full file): {n}")
        print(f"    head:\n{inferred.head(5)}")
    except Exception as e:
        print(f"    inferred-dtype read FAILED (likely mixed/ragged rows): {e}")

for f in files:
    print(f"\n{'─'*90}\n{f.relative_to(DRIVE)}\n{'─'*90}")
    try:
        if f.suffix == ".parquet":
            inspect_parquet(f)
        elif f.suffix in (".csv", ".txt"):
            inspect_csv(f)
        elif f.suffix in (".xls", ".xlsx"):
            df = pl.read_excel(f)
            print(f"    rows: {df.height}   cols: {df.columns}")
            print(f"    head:\n{df.head(5)}")
        else:
            print(f"    (skipped — unrecognized extension)")
    except Exception as e:
        print(f"    AUDIT FAILED for this file: {e}")

# ── cross-file consistency checks, grouped by parent subfolder / name pattern
print(f"\n\n{'█'*30}  CROSS-FILE CONSISTENCY CHECKS  {'█'*30}")

def csv_schema(fp):
    cols = pl.read_csv(fp, infer_schema_length=0, n_rows=5).columns
    return dict(zip(cols, ["str"] * len(cols)))

def parquet_schema(fp):
    s = pl.scan_parquet(fp).collect_schema()
    return dict(zip(s.names(), [str(d) for d in s.dtypes()]))

def check_group(group_name, filelist):
    if len(filelist) < 2:
        if len(filelist) == 1:
            print(f"\n--- {group_name}: only 1 file, nothing to compare ---")
        return
    print(f"\n--- {group_name}: {len(filelist)} files ---")
    schemas = {}
    for fp in filelist:
        try:
            loader = parquet_schema if fp.suffix == ".parquet" else csv_schema
            schemas[str(fp.relative_to(DRIVE))] = loader(fp)
        except Exception as e:
            schemas[str(fp.relative_to(DRIVE))] = {"__ERROR__": str(e)}
    allcols = sorted({c for s in schemas.values() for c in s})
    drift_found = False
    for c in allcols:
        seen = {fname: str(s.get(c, "—MISSING—")) for fname, s in schemas.items()}
        if len(set(seen.values())) > 1:
            drift_found = True
            print(f"  DRIFT  {c}: {seen}")
    if not drift_found:
        print(f"  OK — schema identical across all {len(filelist)} files")

# group 1: yearly bhavcopy parquets (top-level, by naming pattern)
check_group("bhavcopy yearly price parquets", sorted(DRIVE.glob("prices_*.parquet")))

# group 2: corporate-action CSVs (BSE vs NSE, top-level, by naming pattern)
ca_files = [f for f in files if f.parent == DRIVE and f.suffix == ".csv"
           and re.search(r"corp.*action|ca[_-]", f.name, re.I)]
check_group("corporate-action CSVs (BSE/NSE)", ca_files)

# group 3+: each subfolder (block deals/, bulk deals/, short deals/, etc.)
# treated as its own consistency group — multi-file folders often mean
# one file per year/period, exactly where schema drift tends to hide.
subfolders = sorted({f.parent for f in files if f.parent != DRIVE})
for sf in subfolders:
    sf_files = sorted([f for f in files if f.parent == sf and f.suffix in (".csv", ".parquet")])
    check_group(f"folder: {sf.relative_to(DRIVE)}/", sf_files)

print(f"\n\n{'█'*30}  DONE  {'█'*30}")
print("Read: any DRIFT lines above need a normalization step before those "
      "files can be safely concatenated or joined in the Stage-1 build.")
