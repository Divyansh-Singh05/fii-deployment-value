# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5B-STEP3 · FORWARD CARs — EPISODE-START ANCHOR (drift test only)
#
# First model-facing result on CERTIFIED prices (Gate A passed, attrition
# diagnosed as genuine non-trading in 5D). Pre-registered START-anchor
# predictions:
#   SHARK_ACC   ABOVE baseline  (informed buying -> forward drift up)
#   SHARK_DIST  BELOW baseline  (informed selling -> forward drift down)
#   ROBOT       ~ baseline      (passive placebo)
#   HOSTAGE     no strong START call (selling still ongoing; its real test
#               is the END anchor, 5B-4) -- expect <= baseline
#   ALL_LABELED baseline row: read every archetype RELATIVE to this, not
#               vs zero (small-cap tilt vs NIFTY shifts the whole level).
#
# Method: abnormal ret = ret_adj_mktadj (already NIFTY-adjusted, beta=1),
# clipped +/-50pct, nulls->0 in accumulation. CARs from t+1 (day 0 is
# mechanical, reported separately). Delisting/ISIN-change: CAR truncated at
# last traded price, event kept; trunc% reported (survivorship watch).
# Inference: date-clustered bootstrap (1000 reps) on car20.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
HORIZONS = [1, 5, 10, 20]
ABS_CAP = 0.50
N_BOOT = 1000
rng = np.random.default_rng(42)

px = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
px = px.select("isin", "date", "ret_adj_mktadj").sort(["isin", "date"])
px = px.with_columns(
    pl.col("ret_adj_mktadj").clip(-ABS_CAP, ABS_CAP).fill_null(0.0).alias("ar"))
px = px.with_columns(pl.col("ar").cum_sum().over("isin").alias("_cum"))
for k in HORIZONS:
    px = px.with_columns(
        (pl.coalesce(pl.col("_cum").shift(-k).over("isin"),
                     pl.col("_cum").last().over("isin"))
         - pl.col("_cum")).alias("car" + str(k)),
        pl.col("_cum").shift(-k).over("isin").is_null().alias("trunc" + str(k)))
keep = (["isin", "date", pl.col("ar").alias("day0_ar")]
        + ["car" + str(k) for k in HORIZONS]
        + ["trunc" + str(k) for k in HORIZONS])
anchors = px.select(keep)

states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
states = states.select("cisin", "TR_DATE", "era", "archetype")
states = states.sort(["cisin", "TR_DATE"])
runs = states.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_run"))
runs = runs.group_by("cisin", "_run").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").first().alias("start_date"), pl.len().alias("ep_len"))
print("episodes:", runs.height)

ev = runs.join(anchors, left_on=["cisin", "start_date"],
               right_on=["isin", "date"], how="inner")
print("episode starts with a price row:", ev.height,
      "(", round(100 * ev.height / runs.height, 1), "%)")
base = states.join(anchors, left_on=["cisin", "TR_DATE"],
                   right_on=["isin", "date"], how="inner")

ARCHS = ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT",
         "UNTAGGED_DIRECTIONAL"]

def row_stats(a, date_col):
    day0 = float(a["day0_ar"].mean()) * 1e4
    cars = [float(a["car" + str(k)].mean()) * 1e4 for k in HORIZONS]
    med20 = float(a["car20"].median()) * 1e4
    pos20 = 100 * float((a["car20"] > 0).mean())
    tr20 = 100 * float(a["trunc20"].mean())
    per = a.group_by(date_col).agg(
        pl.col("car20").sum().alias("s"), pl.len().alias("c"))
    s = per["s"].to_numpy(); c = per["c"].to_numpy().astype(float)
    nd = len(s)
    idx = rng.integers(0, nd, size=(N_BOOT, nd))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1) * 1e4
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return day0, cars, med20, pos20, tr20, lo, hi, p

def fmt(name, n, r):
    day0, cars, med20, pos20, tr20, lo, hi, p = r
    return (name.ljust(22) + str(n).rjust(7) + ("%.0f" % day0).rjust(7)
            + "".join(("%.0f" % v).rjust(7) for v in cars)
            + ("%.0f" % med20).rjust(7) + ("%.0f" % pos20).rjust(5)
            + ("%.0f" % tr20).rjust(5)
            + ("[%.0f,%.0f]" % (lo, hi)).rjust(15) + ("%.3f" % p).rjust(7))

print("")
print("ANCHOR = EPISODE START | CARs from t+1, market-adjusted, basis points")
head = ("archetype".ljust(22) + "n".rjust(7) + "day0".rjust(7)
        + "".join(("car" + str(k)).rjust(7) for k in HORIZONS)
        + "med20".rjust(7) + "pos".rjust(5) + "tr%".rjust(5)
        + "boot95%car20".rjust(15) + "p".rjust(7))
for era in ("TRAIN", "TEST"):
    e = ev.filter(pl.col("era") == era)
    b = base.filter(pl.col("era") == era)
    print("")
    print("--- " + era + " ---")
    print(head)
    for arch in ARCHS:
        a = e.filter(pl.col("archetype") == arch)
        if a.height < 30:
            print(arch.ljust(22) + str(a.height).rjust(7) + "  too few")
            continue
        print(fmt(arch, a.height, row_stats(a, "start_date")))
    print(fmt("ALL_LABELED (base)", b.height, row_stats(b, "TR_DATE")))

print("""
READ (this step tests DRIFT only):
 1. Every archetype RELATIVE to ALL_LABELED of its era (a shared level
    shift is beta/size vs NIFTY, not signal).
 2. SHARK_ACC minus baseline > 0; SHARK_DIST minus baseline < 0;
    ROBOT ~ baseline. HOSTAGE may be <= baseline at START.
 3. TRAIN direction must replicate in TEST.
 4. tr% per archetype: differential truncation = survivorship watch
    (esp. HOSTAGE, per the 5D distress finding).
NEXT (5B-4): END-anchor reversal test (HOSTAGE headline).
""")
