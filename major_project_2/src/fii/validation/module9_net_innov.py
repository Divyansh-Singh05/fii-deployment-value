# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 9 · NET_INNOV — flow-surprise validation object (original plan item)
#
# NET_INNOV = the SURPRISE component of daily net FII flow. Built the
# pre-agreed way: per-stock AR(5) on scaled net flow, FULL-SAMPLE fit
# (look-ahead ALLOWED by design — this is a validation object, never a
# feature/signal). Scale = stock's full-sample mean GROSS.
#
# What it tests (the last alternative explanation for the headline):
#  T1  sanity: INNOV vs SAME-DAY abnormal return (flow surprise should
#      move prices contemporaneously — if not, INNOV is broken).
#  T2  INNOV vs FORWARD 20d return (does surprise magnitude itself
#      predict/revert?).
#  T3  THE KEY TEST: module-7 panel regression with episode-mean INNOV
#      added. If D_SHARK_DIST survives, concentration != flow-surprise
#      magnitude -> the composition axis is independent information.
#  T4  two-way sort: within INNOV terciles, SHARK_DIST vs HOSTAGE post20
#      (does the concentration gap hold at every surprise level?).
# Universe: calibrated states (original cisins) for clean feature joins;
# targets from the certified v3 panel.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from scipy.stats import spearmanr

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING

# ---- build NET_INNOV ---------------------------------------------------------
f = (pl.read_parquet(MODELD / "stockday_features_v2.parquet")
       .select("cisin", "TR_DATE", "NET", "GROSS")
       .sort(["cisin", "TR_DATE"]))
s = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
       .select("cisin", "TR_DATE", "era", "archetype"))
d = s.join(f, on=["cisin", "TR_DATE"], how="inner")
gm = d.group_by("cisin").agg(pl.col("GROSS").mean().alias("gmean"))
d = d.join(gm, on="cisin", how="left")
d = d.with_columns((pl.col("NET") / pl.col("gmean")).alias("nets"))
lo, hi = d["nets"].quantile(0.005), d["nets"].quantile(0.995)
d = d.with_columns(pl.col("nets").clip(lo, hi))
for k in range(1, 6):
    d = d.with_columns(pl.col("nets").shift(k).over("cisin")
                       .alias("l" + str(k)))
print("stock-days with flow data:", d.height)

# per-stock AR(5), full-sample OLS (look-ahead allowed by design)
d = d.sort(["cisin", "TR_DATE"])
parts = []
for cis, g in d.to_pandas().groupby("cisin", sort=False):
    gg = g.dropna(subset=["nets", "l1", "l2", "l3", "l4", "l5"])
    if len(gg) < 120:
        continue
    X = np.column_stack([np.ones(len(gg))]
                        + [gg["l" + str(k)].to_numpy()
                           for k in range(1, 6)])
    yv = gg["nets"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    gg = gg.assign(innov=yv - X @ beta)
    parts.append(gg[["cisin", "TR_DATE", "innov", "era", "archetype"]])
import pandas as pd
innov = pl.from_pandas(pd.concat(parts, ignore_index=True))
# pandas round-trip turns Date into datetime[ms]; cast back for joins
innov = innov.with_columns(pl.col("TR_DATE").cast(pl.Date))
print("INNOV built for", innov["cisin"].n_unique(), "stocks,",
      innov.height, "stock-days")
print("INNOV mean/std:", round(float(innov["innov"].mean()), 4), "/",
      round(float(innov["innov"].std()), 4),
      "(mean ~0 by construction)")

# ---- targets + characteristics (same machinery as module 7b) ----------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret").sort(["isin", "date"]))
p = p.with_columns(pl.col("nifty50_ret").fill_null(0.0))
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
    ((pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                  pl.col("cum").last().over("isin"))
      - pl.col("cum")) * 1e4).alias("y20"),
    pl.col("ar").rolling_std(window_size=20).over("isin")
      .shift(1).alias("vol20"),
    (pl.col("ar").abs() / (pl.col("to") + 1.0)).alias("_ilq"),
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1) * 1e9
     + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anchors = p.select("isin", "date", pl.col("ar").alias("day0"),
                   "pre20", "y20", "vol20", "relvol", "logclose",
                   "beta120", "mom", "amihud", "logto")

dd = innov.join(anchors, left_on=["cisin", "TR_DATE"],
                right_on=["isin", "date"], how="inner")
te = dd.filter(pl.col("era") == "TEST")
trn = dd.filter(pl.col("era") == "TRAIN")

def daily_ic(df, pred, ycol):
    out = []
    for _, g in df.select("TR_DATE", pred, ycol).to_pandas() \
                  .groupby("TR_DATE"):
        g = g.dropna()
        if len(g) >= 30 and g[pred].nunique() > 1:
            ic = spearmanr(g[pred], g[ycol]).statistic
            if not np.isnan(ic):
                out.append(ic)
    return np.array(out)

print("\n=== T1 · sanity: INNOV vs SAME-DAY abnormal return ===")
for nm, df in (("TRAIN", trn), ("TEST", te)):
    ics = daily_ic(df, "innov", "day0")
    print(f"  {nm}: mean IC {ics.mean():+.4f} | %pos "
          f"{100*(ics>0).mean():.0f}%  (expect clearly positive)")

print("\n=== T2 · INNOV vs FORWARD 20d return ===")
for nm, df in (("TRAIN", trn), ("TEST", te)):
    ics = daily_ic(df, "innov", "y20")
    no = ics[::20]
    t = no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))
    print(f"  {nm}: mean IC {ics.mean():+.4f} | non-overlap t {t:+.2f}")

# ---- episodes with mean INNOV -------------------------------------------------
st = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
        .select("cisin", "TR_DATE", "era", "archetype")
        .sort(["cisin", "TR_DATE"]))
st = st.join(innov.select("cisin", "TR_DATE", "innov"),
             on=["cisin", "TR_DATE"], how="left")
runs = st.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"),
    pl.col("innov").mean().alias("minnov"))
ev = runs.join(anchors, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"),
    pl.col("y20").alias("y"))
for a in ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"]:
    ev = ev.with_columns((pl.col("archetype") == a).cast(pl.Float64)
                         .alias("D_" + a))
DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]

def run_spec(sub, xcols, tag, show=None):
    need = ["y", "cisin", "ed", "month"] + xcols
    dd2 = sub.drop_nulls(need).select(need).to_pandas()
    dd2 = dd2.set_index(["cisin", "ed"])
    res = PanelOLS(dd2["y"], dd2[xcols], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=dd2[["month"]])
    print("\n " + tag + f"  (n={res.nobs})")
    for v in (show or xcols):
        pv = float(res.pvalues[v])
        star = "***" if pv < .01 else "**" if pv < .05 else \
               "*" if pv < .10 else ""
        print("   " + v.ljust(14)
              + ("%.2f" % float(res.params[v])).rjust(10)
              + "  t=" + ("%.2f" % float(res.tstats[v])).rjust(6)
              + "  p=" + ("%.3f" % pv) + " " + star)

print("\n=== T3 · KEY TEST: does SHARK_DIST survive INNOV control? ===")
for era in ("TRAIN", "TEST"):
    sub = ev.filter(pl.col("era") == era)
    run_spec(sub, DUM + CTL + ["minnov"],
             era + "  R2 + episode-mean INNOV",
             show=DUM + ["minnov"])

print("\n=== T4 · two-way sort: INNOV tercile x concentration (post20 bp) ===")
sell = ev.filter(pl.col("archetype").is_in(["SHARK_DIST", "HOSTAGE"])
                 & pl.col("minnov").is_not_null())
sell = sell.with_columns(
    pl.col("minnov").qcut(3, labels=["surprise_lo", "surprise_mid",
                                     "surprise_hi"]).alias("iq"))
print(sell.group_by("era", "iq", "archetype")
          .agg(pl.len().alias("n"), pl.col("y").mean().round(0)
               .alias("post20bp"))
          .sort(["era", "iq", "archetype"]))
print("read: if SHARK_DIST > HOSTAGE within EVERY tercile, concentration")
print("is not flow-surprise magnitude in disguise.")

print("""
VERDICT:
 T1 positive = INNOV is a working flow-surprise measure (sanity).
 T3 is the claim-defining cell: D_SHARK_DIST surviving episode-mean
 INNOV (both eras) = the concentration/composition axis carries
 information INDEPENDENT of flow-surprise magnitude — the last
 alternative explanation for the reversal is closed.
 T4 shows the same non-parametrically.
""")
