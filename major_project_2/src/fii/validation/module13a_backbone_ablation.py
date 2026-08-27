# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 13A · ABLATION — is the HMM backbone NECESSARY, or is the
# contribution the composition measure alone? (referee objection #1)
#
# Competitor: the SIMPLEST possible backbone. No HMM, no EM, no Viterbi:
#   rule_state = SELL if F_persist < c_lo | BUY if > c_hi | else NEUTRAL
# c_lo/c_hi are TRAIN-era quantiles chosen to MATCH the HMM backbone's
# TRAIN census (so the two backbones tag the same *fraction* of days and
# differ only in *which* days). Frozen from TRAIN, applied to TEST.
# The SAME frozen overlay thresholds are then applied:
#   HOSTAGE < -0.513 | SHARK_DIST > +0.877 | SHARK_ACC > +0.795.
# Then the paper's Table-1 regression (R2 spec, PanelOLS stock+date FE,
# SE clustered stock x month) is run on BOTH label sets side by side.
#
# PRE-REGISTERED VERDICTS (written before results):
#  V1 "HMM NOT NECESSARY": rule-based D_SHARK_DIST and D_SHARK_ACC keep
#     |t| >= 2 in BOTH eras AND coefficients within +/-50% of the HMM
#     versions. -> paper reframes: contribution = composition measure;
#     HMM = taxonomy convenience. (This outcome STRENGTHENS the paper.)
#  V2 "HMM ADDS IDENTIFICATION VALUE": rule version loses significance
#     in either era or |coef| shrinks >50%. -> quantify the gap.
#  Either way: report label agreement (Cohen's kappa) + episode-run
#  structure so the mechanism of any difference is visible.
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
THR_H, THR_SD, THR_SA = -0.513, 0.877, 0.795   # frozen (Module 3B)

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
print("states_v3:", st.shape, "| cols ok:",
      all(c in st.columns for c in
          ("F_persist", "F_entity_s", "F_entity_buy_s", "state")))

# ---- rule backbone: census-matched TRAIN quantiles ---------------------------
tr = st.filter(pl.col("era") == "TRAIN")
p_sell = (tr["state"] == "SELL_REGIME").mean()
p_buy = (tr["state"] == "BUY_REGIME").mean()
c_lo = float(tr["F_persist"].quantile(p_sell))
c_hi = float(tr["F_persist"].quantile(1 - p_buy))
print(f"HMM TRAIN census: sell {p_sell:.3f} buy {p_buy:.3f}")
print(f"frozen rule cuts: F_persist < {c_lo:+.3f} = SELL | "
      f"> {c_hi:+.3f} = BUY")

st = st.with_columns(
    pl.when(pl.col("F_persist") < c_lo).then(pl.lit("SELL_REGIME"))
     .when(pl.col("F_persist") > c_hi).then(pl.lit("BUY_REGIME"))
     .otherwise(pl.lit("NEUTRAL")).alias("rstate"))
st = st.with_columns(
    pl.when((pl.col("rstate") == "SELL_REGIME")
            & (pl.col("F_entity_s") < THR_H)).then(pl.lit("HOSTAGE"))
     .when((pl.col("rstate") == "SELL_REGIME")
           & (pl.col("F_entity_s") > THR_SD)).then(pl.lit("SHARK_DIST"))
     .when((pl.col("rstate") == "BUY_REGIME")
           & (pl.col("F_entity_buy_s") > THR_SA)).then(pl.lit("SHARK_ACC"))
     .when(pl.col("rstate") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("rarch"))

# ---- diagnostics: agreement + episode structure ------------------------------
print("\n=== D1 · label agreement (rule vs HMM) ===")
agree = (st["archetype"] == st["rarch"]).mean()
po = float(agree)
cats = ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT",
        "UNTAGGED_DIRECTIONAL"]
pe = sum(float((st["archetype"] == c).mean())
         * float((st["rarch"] == c).mean()) for c in cats)
kappa = (po - pe) / (1 - pe)
print(f"overall agreement {100*po:.1f}%  Cohen's kappa {kappa:.3f}")
for c in ("SHARK_DIST", "SHARK_ACC", "HOSTAGE"):
    h = st.filter(pl.col("archetype") == c)
    r = st.filter(pl.col("rarch") == c)
    both = st.filter((pl.col("archetype") == c)
                     & (pl.col("rarch") == c)).height
    print(f"  {c:11s} HMM n={h.height:6d} rule n={r.height:6d} "
          f"overlap {both:6d} ({100*both/max(h.height,1):.0f}% of HMM)")

def mean_run(df, col, val):
    d = df.sort(["cisin", "TR_DATE"]).with_columns(
        ((pl.col(col) != pl.col(col).shift(1)).fill_null(True))
        .cum_sum().over("cisin").alias("_r"))
    runs = (d.filter(pl.col(col) == val)
              .group_by("cisin", "_r").len())
    return float(runs["len"].mean())

print("\n=== D2 · episode structure (mean run, days) ===")
for c in ("SHARK_DIST", "SHARK_ACC", "HOSTAGE"):
    print(f"  {c:11s} HMM {mean_run(st,'archetype',c):.2f} "
          f"| rule {mean_run(st,'rarch',c):.2f}")

# ---- Table-1 regression on both label sets -----------------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret").sort(["isin", "date"]))
p = p.with_columns(pl.col("nifty50_ret").fill_null(0.0))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"),
    pl.col("ret_adj").clip(-.5, .5).fill_null(0.0).alias("r"),
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
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"),
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                 pl.col("cum").last().over("isin"))
     - pl.col("cum")).alias("post20"))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
     * 1e9 + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anch = p.select("isin", "date", "pre20", "post20", "vol20", "relvol",
                "logclose", "beta120", "mom", "amihud", "logto")

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]

def episodes(label_col):
    runs = st.with_columns(
        ((pl.col(label_col) != pl.col(label_col).shift(1))
         .fill_null(True)).cum_sum().over("cisin").alias("_r"))
    runs = runs.group_by("cisin", "_r").agg(
        pl.col(label_col).first().alias("arch"),
        pl.col("era").first(),
        pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
    ev = runs.join(anch, left_on=["cisin", "ed"],
                   right_on=["isin", "date"], how="inner")
    ev = ev.with_columns(
        pl.col("ed").dt.strftime("%Y-%m").alias("month"),
        (1e4 * pl.col("pre20")).alias("pre20bp"),
        (1e4 * pl.col("mom")).alias("mombp"),
        (1e4 * pl.col("post20")).alias("y20"),
        pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
    for a in ("HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"):
        ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                             .alias("D_" + a))
    return ev

def run_r2(ev, era, tag):
    need = ["y20", "cisin", "ed", "month"] + DUM + CTL
    d = (ev.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    res = PanelOLS(d["y20"], d[DUM + CTL], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=d[["month"]])
    out = {}
    print(f"\n {tag} {era} (n={res.nobs})")
    for v in DUM:
        pv = float(res.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        c, t = float(res.params[v]), float(res.tstats[v])
        out[v] = (c, t)
        print(f"   {v:14s}{c:+9.1f}  t={t:+6.2f} {s}")
    return out

print("\n=== T1 · Table-1 regression: HMM labels vs RULE labels ===")
res = {}
for tag, col in (("HMM ", "archetype"), ("RULE", "rarch")):
    ev = episodes(col)
    for era in ("TRAIN", "TEST"):
        res[(tag, era)] = run_r2(ev, era, tag)

print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICT")
ok = True
for era in ("TRAIN", "TEST"):
    for v in ("D_SHARK_DIST", "D_SHARK_ACC"):
        ch, th = res[("HMM ", era)][v]
        cr, tr_ = res[("RULE", era)][v]
        keep = abs(tr_) >= 2 and abs(cr - ch) <= 0.5 * abs(ch)
        ok = ok and keep
        print(f"  {era:5s} {v:14s} HMM {ch:+7.1f}(t{th:+5.2f}) "
              f"RULE {cr:+7.1f}(t{tr_:+5.2f}) "
              f"{'WITHIN BAND' if keep else 'DEGRADED'}")
print("\nVERDICT:", "V1 — HMM NOT NECESSARY (contribution = the"
      " composition measure; reframe per plan)" if ok else
      "V2 — HMM ADDS IDENTIFICATION VALUE (quantified above)")
print("=" * 70)
