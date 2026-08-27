# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 10 · STATE-DEPENDENCE OF THE REVERSAL — INDIA VIX + KYLE LAMBDA
#
# Two conditioning tests of the liquidity mechanism (pre-registered):
#  P1 VIX: reversal STRONGER in high-VIX periods (scarce liquidity
#     provision -> bigger overshoot -> bigger reversion).
#     Econometrics: VIX is a pure time-series -> date FE absorbs its level;
#     only the INTERACTION D_arch x highVIX is identified. That is the spec.
#  P2 KYLE LAMBDA (FII-flow version): reversal STRONGER in high-impact
#     stocks. lambda_i,t = trailing 120-flow-day cov(ar, scaled FII net
#     flow)/var(flow), PAST-ONLY (shift 1). Stated limitation: FII-flow
#     impact, not total-order-flow Kyle lambda.
#  Expected: D_SHARK_DIST x hi > 0 and D_SHARK_ACC x hi < 0 for both
#  conditioners; HOSTAGE interactions ~ 0 (nothing to amplify).
#  Non-parametric tercile tables shown alongside the regressions.
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
MODELD = ISIN_MAPPING

# ---- panel + characteristics (7b machinery, beta-null fix included) ---------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret", "india_vix")
       .sort(["isin", "date"]))
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

# ---- FII-flow Kyle lambda (trailing, past-only, on flow days) ----------------
f = (pl.read_parquet(MODELD / "stockday_features_v2.parquet")
       .select("cisin", "TR_DATE", "NET", "GROSS"))
gm = f.group_by("cisin").agg(pl.col("GROSS").mean().alias("gmean"))
f = f.join(gm, on="cisin", how="left")
f = f.with_columns((pl.col("NET") / pl.col("gmean")).alias("nets"))
lo, hi = f["nets"].quantile(0.005), f["nets"].quantile(0.995)
f = f.with_columns(pl.col("nets").clip(lo, hi))
fl = (f.select("cisin", "TR_DATE", "nets")
        .join(p.select("isin", "date", "ar"),
              left_on=["cisin", "TR_DATE"], right_on=["isin", "date"],
              how="inner").sort(["cisin", "TR_DATE"]))
LW = 120
fl = fl.with_columns((pl.col("ar") * pl.col("nets")).alias("_axy"),
                     (pl.col("nets") ** 2).alias("_an2"))
fl = fl.with_columns(
    pl.col("_axy").rolling_mean(window_size=LW).over("cisin").alias("_m1"),
    pl.col("ar").rolling_mean(window_size=LW).over("cisin").alias("_m2"),
    pl.col("nets").rolling_mean(window_size=LW).over("cisin").alias("_m3"),
    pl.col("_an2").rolling_mean(window_size=LW).over("cisin").alias("_m4"))
fl = fl.with_columns(
    ((pl.col("_m1") - pl.col("_m2") * pl.col("_m3"))
     / (pl.col("_m4") - pl.col("_m3") ** 2))
    .shift(1).over("cisin").alias("lam"))
lam = fl.select("cisin", "TR_DATE", "lam")
print("lambda computed:", lam.drop_nulls().height, "stock-days |",
      "median lam:", round(float(lam["lam"].drop_nulls().median()), 4),
      "(positive = flow moves price, sanity)")

# ---- episodes ----------------------------------------------------------------
st = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
        .select("cisin", "TR_DATE", "era", "archetype")
        .sort(["cisin", "TR_DATE"]))
runs = st.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
anch = p.select("isin", "date", "y20", "pre20", "vol20", "relvol",
                "logclose", "beta120", "mom", "amihud", "logto",
                "india_vix")
ev = runs.join(anch, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.join(lam, left_on=["cisin", "ed"],
             right_on=["cisin", "TR_DATE"], how="left")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    pl.col("y20").alias("y"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a in ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"]:
    ev = ev.with_columns((pl.col("archetype") == a).cast(pl.Float64)
                         .alias("D_" + a))

# conditioners: era-median splits (hi = 1/0)
ev = ev.with_columns(
    (pl.col("india_vix")
     > pl.col("india_vix").median().over("era")).cast(pl.Float64)
    .alias("hiVIX"),
    (pl.col("lam")
     > pl.col("lam").median().over("era")).cast(pl.Float64)
    .alias("hiLAM"))
for a in ["SHARK_DIST", "SHARK_ACC", "HOSTAGE"]:
    ev = ev.with_columns(
        (pl.col("D_" + a) * pl.col("hiVIX")).alias("D_" + a + "_xV"),
        (pl.col("D_" + a) * pl.col("hiLAM")).alias("D_" + a + "_xL"))

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]
IXV = ["D_SHARK_DIST_xV", "D_SHARK_ACC_xV", "D_HOSTAGE_xV"]
IXL = ["D_SHARK_DIST_xL", "D_SHARK_ACC_xL", "D_HOSTAGE_xL"]

def run_spec(sub, xcols, tag, show=None):
    need = ["y", "cisin", "ed", "month"] + xcols
    d = sub.drop_nulls(need).select(need).to_pandas()
    d = d.set_index(["cisin", "ed"])
    res = PanelOLS(d["y"], d[xcols], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, clusters=d[["month"]])
    print("\n " + tag + f"  (n={res.nobs})")
    for v in (show or xcols):
        pv = float(res.pvalues[v])
        star = "***" if pv < .01 else "**" if pv < .05 else \
               "*" if pv < .10 else ""
        print("   " + v.ljust(17)
              + ("%.2f" % float(res.params[v])).rjust(10)
              + "  t=" + ("%.2f" % float(res.tstats[v])).rjust(6)
              + "  p=" + ("%.3f" % pv) + " " + star)

print("\n" + "=" * 70)
print("SPEC V · VIX interaction (main VIX level absorbed by date FE)")
print("interaction coef = EXTRA reversal in high-VIX halves")
print("=" * 70)
for era in ("TRAIN", "TEST"):
    run_spec(ev.filter(pl.col("era") == era), DUM + IXV + CTL,
             era, show=["D_SHARK_DIST", "D_SHARK_DIST_xV",
                        "D_SHARK_ACC", "D_SHARK_ACC_xV",
                        "D_HOSTAGE", "D_HOSTAGE_xV"])

print("\n" + "=" * 70)
print("SPEC L · KYLE-LAMBDA interaction (hiLAM = stock-level, main incl.)")
print("=" * 70)
for era in ("TRAIN", "TEST"):
    run_spec(ev.filter(pl.col("era") == era),
             DUM + ["hiLAM"] + IXL + CTL,
             era, show=["D_SHARK_DIST", "D_SHARK_DIST_xL",
                        "D_SHARK_ACC", "D_SHARK_ACC_xL",
                        "D_HOSTAGE", "D_HOSTAGE_xL", "hiLAM"])

print("\n" + "=" * 70)
print("NON-PARAMETRIC · SHARK_DIST post20 (bp) by conditioner tercile")
print("=" * 70)
sd = ev.filter(pl.col("archetype") == "SHARK_DIST")
for cond, nm in (("india_vix", "VIX"), ("lam", "lambda")):
    s2 = sd.filter(pl.col(cond).is_not_null())
    s2 = s2.with_columns(pl.col(cond).qcut(3, labels=["lo", "mid", "hi"])
                         .over("era").alias("tc"))
    print("\n-- by " + nm + " tercile --")
    print(s2.group_by("era", "tc").agg(
        pl.len().alias("n"), pl.col("y").mean().round(0).alias("post20bp"))
        .sort(["era", "tc"]))

print("""
READ (pre-registered):
 P1: D_SHARK_DIST_xV > 0 (reversal bigger under high VIX) and
     D_SHARK_ACC_xV < 0. HOSTAGE interactions ~ 0.
 P2: D_SHARK_DIST_xL > 0 (bigger in high-impact stocks), mirror for ACC.
 Terciles should rise monotonically for SHARK_DIST if the mechanism is
 liquidity: more stress / more impact -> more overshoot -> more reversion.
 Nulls here do NOT threaten the main result (it's unconditional); they
 would just mean the reversal is state-INdependent at this power.
""")
