# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 6B · LIQUIDITY-SHOCK PROFILE — the full event arc (read-only)
#
# FINAL mechanism test, pre-committed stopping rule. Uses ONLY the certified
# internal panel (returns_panel_v3 + states_v3): no third-party deal data.
#
# A liquidity shock has an undeniable price/volume signature:
#   pressure phase: negative abnormal returns + ELEVATED VOLUME while the
#                   concentrated seller demands liquidity
#   reversion:      price drifts back once flow stops (PROVEN: +68/+33bp)
# This profiles the whole arc per archetype:
#   pre20   = CAR over the 20 days BEFORE episode start (backdrop decline)
#   day0    = abnormal return on the episode's first day
#   epCAR   = cumulative abnormal return start..end inclusive (the shock)
#   post20  = CAR t+1..t+20 after episode END (the reversal, reprinted)
#   rvol0   = day-0 volume / stock's own trailing-20d avg volume (past-only)
#
# PREDICTIONS (liquidity-shock reading of the concentration axis):
#   SHARK_DIST: pre20<0, epCAR<0, rvol0 ELEVATED vs baseline, post20>baseline
#   SHARK_ACC : mirror (epCAR>0, rvol0 elevated, post20<baseline)
#   HOSTAGE   : pressure allowed, but NO reversal after (post20~baseline)
#   ROBOT     : flat arc, rvol0 ~ baseline (placebo)
# WIN = SHARK_DIST shows pressure (epCAR<0 AND elevated rvol0) + known
#       reversal. FAIL = no pressure -> report reversal as unexplained
#       regularity (Option 3) and STOP mechanism-hunting.
# NOTE: day0/epCAR are mechanically correlated with flow (same-day impact);
# that is fine here — this characterizes the mechanism, not prediction.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA

p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "volume", "ret_adj_mktadj")
       .sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-0.5, 0.5).fill_null(0.0).alias("ar"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
p = p.with_columns(
    pl.col("volume").rolling_mean(window_size=20).over("isin").alias("_vma"))
p = p.with_columns(
    (pl.col("volume") / pl.col("_vma").shift(1).over("isin")).alias("rvol"))
p = p.with_columns(
    (pl.col("cum").shift(1).over("isin")
     - pl.col("cum").shift(21).over("isin")).alias("pre20"),
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                 pl.col("cum").last().over("isin"))
     - pl.col("cum")).alias("post20"))
anchors = p.select("isin", "date", "ar", "cum", "pre20", "post20", "rvol")

states = (pl.read_parquet(DRIVE / "states_v3.parquet")
            .select("cisin", "TR_DATE", "era", "archetype")
            .sort(["cisin", "TR_DATE"]))
runs = states.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").first().alias("sd"),
    pl.col("TR_DATE").last().alias("ed"),
    pl.len().alias("eplen"))

ev = (runs.join(anchors, left_on=["cisin", "sd"],
                right_on=["isin", "date"], how="inner")
          .rename({"ar": "d0", "cum": "cum_s", "pre20": "pre",
                   "rvol": "rvol0"})
          .drop("post20"))
ev = (ev.join(anchors.select("isin", "date",
                             pl.col("cum").alias("cum_e"),
                             pl.col("post20").alias("post")),
              left_on=["cisin", "ed"], right_on=["isin", "date"],
              how="inner"))
ev = ev.with_columns(
    (pl.col("cum_e") - pl.col("cum_s") + pl.col("d0")).alias("epcar"))

base = states.join(anchors, left_on=["cisin", "TR_DATE"],
                   right_on=["isin", "date"], how="inner")

ARCHS = ["SHARK_DIST", "SHARK_ACC", "HOSTAGE", "ROBOT",
         "UNTAGGED_DIRECTIONAL"]
HDR = ("archetype".ljust(22) + "n_ep".rjust(7) + "mlen".rjust(6)
       + "pre20".rjust(8) + "day0".rjust(7) + "d0med".rjust(7)
       + "epCAR".rjust(8) + "epMed".rjust(7) + "post20".rjust(8)
       + "rvol0".rjust(7))

def bp(x):
    return "%.0f" % (1e4 * x) if x is not None else "  -"

for era in ("TRAIN", "TEST"):
    e = ev.filter(pl.col("era") == era)
    b = base.filter(pl.col("era") == era)
    print("")
    print("=" * 79)
    print("ERA:", era, "| all figures bp except rvol0 (x own 20d avg vol,",
          "median) and mlen")
    print("=" * 79)
    print(HDR)
    for a in ARCHS:
        x = e.filter(pl.col("archetype") == a)
        if x.height < 30:
            print(a.ljust(22) + str(x.height).rjust(7) + "  too few")
            continue
        print(a.ljust(22) + str(x.height).rjust(7)
              + ("%.0f" % float(x["eplen"].median())).rjust(6)
              + bp(float(x["pre"].mean())).rjust(8)
              + bp(float(x["d0"].mean())).rjust(7)
              + bp(float(x["d0"].median())).rjust(7)
              + bp(float(x["epcar"].mean())).rjust(8)
              + bp(float(x["epcar"].median())).rjust(7)
              + bp(float(x["post"].mean())).rjust(8)
              + ("%.2f" % float(x["rvol0"].drop_nulls().median())).rjust(7))
    # baseline: day-level stats (no episodes -> epCAR = single-day ar)
    print("ALL_LABELED (days)".ljust(22) + str(b.height).rjust(7)
          + "-".rjust(6)
          + bp(float(b["pre20"].mean())).rjust(8)
          + bp(float(b["ar"].mean())).rjust(7)
          + bp(float(b["ar"].median())).rjust(7)
          + "-".rjust(8) + "-".rjust(7)
          + bp(float(b["post20"].mean())).rjust(8)
          + ("%.2f" % float(b["rvol"].drop_nulls().median())).rjust(7))

    # the two decision numbers, spelled out
    sdx = e.filter(pl.col("archetype") == "SHARK_DIST")
    rb = float(b["rvol"].drop_nulls().median())
    rs = float(sdx["rvol0"].drop_nulls().median())
    print("")
    print("  DECISION -- SHARK_DIST", era + ":")
    print("   pressure: epCAR mean", bp(float(sdx["epcar"].mean())),
          "bp (need < 0), median", bp(float(sdx["epcar"].median())))
    print("   volume  : day0 relvol", round(rs, 2), "vs baseline",
          round(rb, 2), "->", round(rs / rb, 2), "x  (need > 1)")

print("""
READ:
 1. SHARK_DIST arc should be a V: pre20<0 (regime decline), epCAR<0
    (the shock), post20 above the ALL_LABELED post20 (the reversal).
 2. rvol0 vs the ALL_LABELED rvol row = the liquidity-demand footprint
    (compare ratios, not vs 1.0 -- median day is below its 20d mean).
 3. SHARK_ACC should mirror with signs flipped. HOSTAGE: any pressure but
    post20 ~ baseline. ROBOT: flat, rvol ~ baseline (placebo).
 4. Replication TRAIN -> TEST as always.
PRE-COMMITTED STOP: if SHARK_DIST shows NO pressure (epCAR >= 0 or rvol0
ratio <= 1), the liquidity-shock mechanism is dead -> report the post-END
reversal as an unexplained empirical regularity (Option 3) and stop
mechanism-hunting. No further instruments after this one.
""")
