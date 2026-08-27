# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5B-STEP4 · FORWARD CARs, DIFFERENCE-IN-DIFFERENCES (both anchors)
#
# Supersedes 5B-3's inference. 5B-3's bootstrap p tested car20 vs ZERO; in
# TEST the whole labeled universe drifted +52bp vs NIFTY (beta=1 vs a
# large-cap index in a midcap bull market), so every row was "significant"
# including ROBOT the placebo. The ONLY valid signal is archetype MINUS the
# universe baseline. This script reports that EXCESS CAR directly, with a
# date-clustered bootstrap on the DIFFERENCE, for BOTH anchors:
#   START anchor  -> drift test (SHARK_ACC excess>0, SHARK_DIST excess<0)
#   END anchor    -> reversal test (HOSTAGE excess>0 after selling stops)
# ROBOT excess ~ 0 is the placebo; if ROBOT separates, the method leaks.
#
# baseline = mean forward CAR over ALL labeled stock-days of the era
# (anchor-independent generic drift; n=280k-445k so its sampling error is
# negligible vs the 4-10k archetype samples -> treated as a fixed offset,
# which is why only the archetype arm is resampled).
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

px = pl.read_parquet(DRIVE / "returns_panel_v3.parquet")   # canonical-keyed
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
carcols = ["car" + str(k) for k in HORIZONS]
anchors = px.select(["isin", "date"] + carcols + ["trunc20"])

states = pl.read_parquet(DRIVE / "states_v3.parquet")   # fragments merged
states = states.select("cisin", "TR_DATE", "era", "archetype")
states = states.sort(["cisin", "TR_DATE"])
runs = states.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_run"))
runs = runs.group_by("cisin", "_run").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").first().alias("start_date"),
    pl.col("TR_DATE").last().alias("end_date"))

base = states.join(anchors, left_on=["cisin", "TR_DATE"],
                   right_on=["isin", "date"], how="inner")

ARCHS = ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT",
         "UNTAGGED_DIRECTIONAL"]

def excess(a, base_means, date_col):
    # point excess at each horizon
    exc = [float(a["car" + str(k)].mean()) * 1e4 - base_means[k]
           for k in HORIZONS]
    tr20 = 100 * float(a["trunc20"].mean())
    # date-clustered bootstrap on car20 excess (baseline fixed)
    per = a.group_by(date_col).agg(
        pl.col("car20").sum().alias("s"), pl.len().alias("c"))
    s = per["s"].to_numpy(); c = per["c"].to_numpy().astype(float)
    nd = len(s)
    idx = rng.integers(0, nd, size=(N_BOOT, nd))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1) * 1e4 - base_means[20]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return exc, tr20, lo, hi, p

for era in ("TRAIN", "TEST"):
    b = base.filter(pl.col("era") == era)
    bm = {k: float(b["car" + str(k)].mean()) * 1e4 for k in HORIZONS}
    print("")
    print("=" * 74)
    print("ERA:", era, "| baseline car5/10/20 (bp):",
          round(bm[5]), "/", round(bm[10]), "/", round(bm[20]),
          "| baseline n =", b.height)
    print("=" * 74)
    for anchor_col, tag in [("start_date", "START (drift)"),
                            ("end_date", "END (reversal)")]:
        ev = runs.filter(pl.col("era") == era).join(
            anchors, left_on=["cisin", anchor_col],
            right_on=["isin", "date"], how="inner")
        print("")
        print("-- ANCHOR:", tag, "| EXCESS CAR vs baseline (bp) --")
        print("archetype".ljust(22) + "n".rjust(7)
              + "exc5".rjust(7) + "exc10".rjust(7) + "exc20".rjust(7)
              + "tr%".rjust(5) + "boot95%_exc20".rjust(16) + "p".rjust(7))
        for arch in ARCHS:
            a = ev.filter(pl.col("archetype") == arch)
            if a.height < 30:
                print(arch.ljust(22) + str(a.height).rjust(7) + "  too few")
                continue
            exc, tr20, lo, hi, p = excess(a, bm, anchor_col)
            print(arch.ljust(22) + str(a.height).rjust(7)
                  + "".join(("%.0f" % v).rjust(7) for v in
                           [exc[1], exc[2], exc[3]])
                  + ("%.0f" % tr20).rjust(5)
                  + ("[%.0f,%.0f]" % (lo, hi)).rjust(16)
                  + ("%.3f" % p).rjust(7))

print("""
READ (excess = archetype minus era baseline; p is on excess20, so it
tests differ-from-UNIVERSE not differ-from-zero):
 DRIFT (START):    SHARK_ACC exc>0 ? SHARK_DIST exc<0 ? ROBOT ~0 ?
 REVERSAL (END):   HOSTAGE exc>0 ?  (the flagship prediction)
 Replication:      same signs in TRAIN and TEST ?
 Placebo:          ROBOT exc ~ 0 with p>>0.05 in both ?
 Survivorship:     HOSTAGE tr% vs others (halt-to-zero mid-window).
If END-anchor HOSTAGE excess is also flat/negative, the next lever is the
market model itself (beta=1 -> size-decile or per-stock beta), NOT more
label tweaking -- the labels are frozen from Module 3.
""")
