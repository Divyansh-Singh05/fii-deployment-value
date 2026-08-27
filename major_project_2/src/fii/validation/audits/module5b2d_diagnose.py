# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5B-STEP2d · DIAGNOSE THE GATE-A FAILURE (read-only, writes nothing)
#
# Gate A failed with internally contradictory numbers (e.g. TITAN ex-day
# ret_cc printed +0.053 while step-1 measured the drop at 18.75x on the
# same row; 0.0% of ex-days adjusted small). This step pins the mechanism:
#   D1  duplicate-key audit on returns_panel_v2
#   D2  does a FRESH close-to-close recompute match the stored ret_cc?
#   D3  row-level context around known failing ex-days (eyeball the tape)
#   D4  on confirmed ex-days: stored ret_cc vs prev_close return vs fresh
#       recompute — which one lost the crash?
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA

v2 = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
fac = pl.read_parquet(DRIVE / "ca_adjustment_factors.parquet")
print("v2 panel:", v2.shape)
print("v2 columns:", v2.columns)

# ---- D1: duplicate keys ------------------------------------------------------
d_isin = v2.height - v2.unique(subset=["isin", "date"]).height
d_sym = v2.height - v2.unique(subset=["symbol", "date"]).height
print("")
print("D1 duplicates: (isin,date) =", d_isin, " (symbol,date) =", d_sym)
print("   ((symbol,date) dups can be legit: two ISINs one symbol;")
print("    (isin,date) dups are NOT legit)")

# ---- D2: fresh recompute vs stored ret_cc -----------------------------------
v2 = v2.sort(["isin", "date"])
v2 = v2.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("isin") - 1)
    .alias("ret_cc_fresh")
)
both = v2.filter(pl.col("ret_cc").is_not_null()
                 & pl.col("ret_cc_fresh").is_not_null())
mism = both.filter((pl.col("ret_cc") - pl.col("ret_cc_fresh")).abs() > 1e-9)
print("")
print("D2 stored ret_cc vs fresh recompute: mismatches =", mism.height,
      "of", both.height)
if mism.height > 0:
    print("   NONZERO -> the stored ret_cc was computed on a different row")
    print("   order or a different frame (executed cell != script).")
    print(mism.select("symbol", "isin", "date", "close",
                      "ret_cc", "ret_cc_fresh").head(10))

# ---- D3: row context around known failing ex-days ---------------------------
CASES = [
    ("TITAN", "2011-06-16", "2011-06-30"),
    ("UNOMINDA", "2022-07-29", "2022-08-12"),
    ("VAKRANGEE", "2013-11-20", "2013-12-04"),
    ("SUNILHITEC", "2016-11-24", "2016-12-08"),
]
print("")
print("D3 row-level context (raw tape around the ex-day):")
show = ["date", "symbol", "isin", "series", "close", "prev_close",
        "ret_cc", "ret_cc_fresh", "adj_factor", "ret_adj"]
show = [c for c in show if c in v2.columns or c == "ret_cc_fresh"]
for sym, d0, d1 in CASES:
    w = v2.filter(
        (pl.col("symbol") == sym)
        & (pl.col("date") >= pl.lit(d0).str.to_date())
        & (pl.col("date") <= pl.lit(d1).str.to_date())
    ).sort("date")
    print("")
    print("----", sym, d0, "->", d1, "----")
    print(w.select([c for c in show if c in w.columns]))

# ---- D4: on confirmed ex-days, which return series holds the crash? --------
conf = fac.filter(pl.col("confirmed") == True)  # noqa: E712
g = conf.join(v2, left_on=["symbol", "ex_date"],
              right_on=["symbol", "date"], how="inner")
g = g.with_columns(
    (pl.col("close") / pl.col("prev_close") - 1).alias("ret_prevclose")
)
print("")
print("D4 confirmed ex-days, medians of |return| by series:")
for c in ["ret_cc", "ret_cc_fresh", "ret_prevclose"]:
    if c in g.columns:
        m = g[c].abs().median()
        print("   median |", c, "| =",
              round(float(m), 4) if m is not None else None)
print("expected: all three ~0.49 (the crash). Whichever is SMALL is the")
print("series that lost the crash -> that computation path is the bug.")
print("")
print("also: expected-vs-actual adjusted return using the FRESH series:")
g = g.with_columns(
    ((1 + pl.col("ret_cc_fresh")) * pl.col("factor") - 1).alias("adj_fresh")
)
m = g["adj_fresh"].abs().median()
ok = float((g["adj_fresh"].abs() < 0.20).mean())
print("   median |adj_fresh| =", round(float(m), 4),
      "| share < 20% =", round(100 * ok, 1), "%")
print("   if this is small/high while stored ret_adj failed Gate A, the")
print("   factor table is fine and ONLY the executed 5B-2 cell was corrupt:")
print("   re-run module5b2 from the .py file via exec(open(...).read()).")
