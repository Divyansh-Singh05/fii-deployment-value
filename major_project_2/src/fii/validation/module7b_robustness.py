# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 7B · ROBUSTNESS BATCH (the paper's robustness table)
#
#  A  ROBOT anomaly: transition matrix (what follows each archetype's END,
#     by era) + ROBOT post20 SPLIT by next-regime direction. If TEST-era
#     NEUTRAL exits skew to SELL and the negative post20 loads on those,
#     the placebo "failure" is mechanical and fully explained.
#  B  Beta-null fix: zero-fill sparse nifty50_ret gaps (~0.6%) BEFORE the
#     120d rolling beta (zero-fill, NOT interpolation — interpolating
#     levels smears returns). Re-run R2 on the RESTORED full sample.
#  C  Non-overlap subsample: within stock, greedily keep episodes whose
#     post-window doesn't overlap the previous kept one (>=28 cal days).
#  D  Horizons: CAR10 / CAR30 / CAR60 (R2 spec; dummies shown).
#  E  Dose-response (continuous, sidesteps estimated-label attenuation):
#     sell-regime episodes: post20 ~ mean F_entity_s (+controls+FE),
#       prediction: coef > 0 (more concentrated -> more reversal);
#     buy-regime episodes: post20 ~ mean F_entity_buy_s, prediction < 0.
# All regressions: linearmodels.PanelOLS, stock+date FE, SE clustered
# stock x month. Omitted category = UNTAGGED (dummy specs).
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA

# ---- panel + characteristics (WITH the beta-null fix) ------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret")
       .sort(["isin", "date"]))
n_gap = p.filter(pl.col("nifty50_ret").is_null()).height
p = p.with_columns(pl.col("nifty50_ret").fill_null(0.0))   # B: the fix
print("index-return gaps zero-filled:", n_gap, "rows",
      "(", round(100 * n_gap / p.height, 2), "% )")
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-0.5, 0.5).fill_null(0.0).alias("ar"),
    pl.col("ret_adj").clip(-0.5, 0.5).fill_null(0.0).alias("r"),
    (pl.col("close") * pl.col("volume")).alias("to"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
W = 120
p = p.with_columns(
    (pl.col("r") * pl.col("nifty50_ret")).alias("_xy"),
    (pl.col("nifty50_ret") ** 2).alias("_y2"))
p = p.with_columns(
    pl.col("_xy").rolling_mean(window_size=W).over("isin").alias("_mxy"),
    pl.col("r").rolling_mean(window_size=W).over("isin").alias("_mx"),
    pl.col("nifty50_ret").rolling_mean(window_size=W).over("isin")
      .alias("_my"),
    pl.col("_y2").rolling_mean(window_size=W).over("isin").alias("_my2"))
p = p.with_columns(
    ((pl.col("_mxy") - pl.col("_mx") * pl.col("_my"))
     / (pl.col("_my2") - pl.col("_my") ** 2))
    .shift(1).over("isin").alias("beta120"))
p = p.with_columns(
    (pl.col("cum").shift(21).over("isin")
     - pl.col("cum").shift(127).over("isin")).alias("mom"),
    (pl.col("cum").shift(1).over("isin")
     - pl.col("cum").shift(21).over("isin")).alias("pre20"),
    pl.col("ar").rolling_std(window_size=20).over("isin")
      .shift(1).alias("vol20"),
    (pl.col("ar").abs() / (pl.col("to") + 1.0)).alias("_ilq"),
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
for k in (10, 20, 30, 60):
    p = p.with_columns(
        (pl.coalesce(pl.col("cum").shift(-k).over("isin"),
                     pl.col("cum").last().over("isin"))
         - pl.col("cum")).alias("post" + str(k)))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1) * 1e9
     + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anchors = p.select("isin", "date", "pre20", "post10", "post20", "post30",
                   "post60", "vol20", "relvol", "logclose", "beta120",
                   "mom", "amihud", "logto")

# ---- episodes (with state + episode-mean concentration) ----------------------
states = (pl.read_parquet(DRIVE / "states_v3.parquet")
            .sort(["cisin", "TR_DATE"]))
runs = states.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("state").first().alias("state"),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"),
    pl.col("F_entity_s").mean().alias("fent"),
    pl.col("F_entity_buy_s").mean().alias("fentb")).sort(["cisin", "_r"])
runs = runs.with_columns(
    pl.col("state").shift(-1).over("cisin").alias("next_state"))

ev = runs.join(anchors, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for k in (10, 20, 30, 60):
    ev = ev.with_columns((1e4 * pl.col("post" + str(k)))
                         .alias("y" + str(k)))
for a in ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"]:
    ev = ev.with_columns((pl.col("archetype") == a).cast(pl.Float64)
                         .alias("D_" + a))

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL2 = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
        "logclose", "logeplen", "pre20bp"]

def run_spec(sub, xcols, ycol, tag, show=None):
    need = [ycol, "cisin", "ed", "month"] + xcols
    d = sub.drop_nulls(need).select(need).to_pandas()
    d = d.set_index(["cisin", "ed"])
    res = PanelOLS(d[ycol], d[xcols], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, clusters=d[["month"]])
    print("\n " + tag + f"  (n={res.nobs})")
    for v in (show or xcols):
        pv = float(res.pvalues[v])
        star = "***" if pv < .01 else "**" if pv < .05 else \
               "*" if pv < .10 else ""
        print("   " + v.ljust(14)
              + ("%.2f" % float(res.params[v])).rjust(10)
              + "  t=" + ("%.2f" % float(res.tstats[v])).rjust(6)
              + "  p=" + ("%.3f" % pv) + " " + star)

# ============================ A: ROBOT anomaly ================================
print("\n" + "=" * 70)
print("A · WHAT FOLLOWS A ROBOT (NEUTRAL) EPISODE — transition + post20")
print("=" * 70)
rb = ev.filter(pl.col("archetype") == "ROBOT")
print("next-regime after ROBOT ends, by era (%):")
print(rb.group_by("era", "next_state").agg(pl.len().alias("n"))
        .with_columns((100 * pl.col("n") / pl.col("n").sum().over("era"))
                      .round(1).alias("pct")).sort(["era", "next_state"]))
print("\nROBOT post20 (bp, mean) split by next regime:")
print(rb.group_by("era", "next_state").agg(
    pl.len().alias("n"),
    (1e4 * pl.col("post20").mean()).round(0).alias("post20bp"))
    .sort(["era", "next_state"]))
print("read: if TEST exits skew to SELL and the negative post20 sits on")
print("SELL-exits, the placebo anomaly is mechanical (measuring the first")
print("days of the next sell-off), not a leak.")

# ============================ B: restored-sample R2 ===========================
print("\n" + "=" * 70)
print("B · R2 ON THE RESTORED SAMPLE (beta-null fix)")
print("=" * 70)
for era in ("TRAIN", "TEST"):
    run_spec(ev.filter(pl.col("era") == era), DUM + CTL2, "y20",
             era + "  R2 full controls (expect n ~= R0 now)", show=DUM)

# ============================ C: non-overlap subsample ========================
print("\n" + "=" * 70)
print("C · NON-OVERLAPPING EPISODES (>=28 cal days between kept ENDs)")
print("=" * 70)
evs = ev.sort(["cisin", "ed"])
keep = np.zeros(evs.height, dtype=bool)
cis = evs["cisin"].to_numpy()
eds = evs["ed"].to_numpy()
last_c, last_d = None, None
for i in range(evs.height):
    if cis[i] != last_c or (eds[i] - last_d).astype("timedelta64[D]") \
            >= np.timedelta64(28, "D"):
        keep[i] = True
        last_c, last_d = cis[i], eds[i]
evno = evs.with_columns(pl.Series("_keep", keep)).filter(pl.col("_keep"))
print("kept", evno.height, "of", ev.height, "episodes")
for era in ("TRAIN", "TEST"):
    run_spec(evno.filter(pl.col("era") == era), DUM + CTL2, "y20",
             era + "  R2, non-overlap subsample", show=DUM)

# ============================ D: horizons =====================================
print("\n" + "=" * 70)
print("D · HORIZON ROBUSTNESS (R2 spec; dummies only)")
print("=" * 70)
for era in ("TRAIN", "TEST"):
    for k in (10, 30, 60):
        run_spec(ev.filter(pl.col("era") == era), DUM + CTL2,
                 "y" + str(k), era + "  CAR" + str(k), show=DUM)

# ============================ E: dose-response ================================
print("\n" + "=" * 70)
print("E · CONTINUOUS DOSE-RESPONSE (no labels, no discretization)")
print("=" * 70)
sell = ev.filter(pl.col("state") == "SELL_REGIME")
buy = ev.filter(pl.col("state") == "BUY_REGIME")
for era in ("TRAIN", "TEST"):
    run_spec(sell.filter(pl.col("era") == era), ["fent"] + CTL2, "y20",
             era + "  SELL episodes: post20 ~ mean F_entity_s "
             "(predict coef > 0)", show=["fent"])
    run_spec(buy.filter(pl.col("era") == era), ["fentb"] + CTL2, "y20",
             era + "  BUY episodes: post20 ~ mean F_entity_buy_s "
             "(predict coef < 0)", show=["fentb"])

print("""
READ:
 A  explains (or not) the TEST ROBOT placebo anomaly mechanically.
 B  the paper's main-table spec: full sample, no attrition artifact.
 C  overlap objection: SHARK_DIST should hold sign/significance.
 D  reversal should build 10->30 and persist (not flip) at 60.
 E  the elegant one: monotone concentration effect WITHIN sell (and buy)
    regimes, no thresholds involved. fent>0 & fentb<0 in both eras =
    the concentration mechanism confirmed as a continuum.
""")
