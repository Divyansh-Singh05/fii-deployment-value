# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 13B · INCREMENTAL VALUE — does participant COMPOSITION carry
# information beyond standard predictors? (referee objection #2, the
# central demand: "models materially improve only when the composition
# signal is added")
#
# Composition block  = {F_entity, F_entity_buy, F_breadth}   (who trades)
# Standard block     = {F_persist, F_block, F_imbal, F_streak,
#                       F_activity, F_flowbeta, F_sizedisp}  (how much/how)
# Price-based standard predictors (momentum, reversal, vol, turnover,
# Amihud, relvol, price) are already in the R2 regression controls.
#
# T1 · EPISODE LEVEL: add episode-mean flow MAGNITUDE and IMBALANCE
#     (the most obvious existing measures) to the R2 spec:
#       mimb   = mean(NET/GROSS) over the episode
#       mlgr   = mean(log(1+GROSS))            (size of flow)
#       mnet_s = mean(NET)/stock's full-sample mean GROSS  (scaled net;
#                stock-scaling is full-sample -> mild look-ahead in a
#                CONTROL, conservative direction, INNOV precedent)
#     PRE-REGISTERED: composition ROBUST if D_SHARK_DIST stays p<0.05 in
#     BOTH eras with coefficient within 50% of the same-run baseline R2.
#
# T2 · DAILY CROSS-SECTION: LightGBM with vs without the composition
#     block, same M8 machinery (fwd 20d abnormal CAR target, frozen
#     split, TEST-era metrics only).
#     PRE-REGISTERED: "BROAD incremental information" if TEST dIC >=
#     0.005 with paired-daily-IC t >= 2 AND Q5-Q1 spread higher.
#     If T2 fails but T1 passes: "composition information is EPISODE/
#     TAIL-CONCENTRATED" (consistent with the dose-response finding) —
#     report as the honest scope of the claim.
#
# NOT TESTABLE HERE (stated): ESG/mandate exits producing permanent
# non-informational declines — a boundary-of-interpretation limitation
# handled in the paper's language (transitory/permanent, not
# informed/uninformed).
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
import lightgbm as lgb

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
COMP = ["F_entity_conc", "F_entity_conc_buy", "F_breadth"]
STD = ["F_persist", "F_block", "F_imbal", "F_streak", "F_activity",
       "F_flowbeta", "F_sizedisp"]

# ---- shared: price panel characteristics (module7b construction) -------------
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

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
f = (pl.read_parquet(MODELD / "stockday_features_v2.parquet")
       .select(["cisin", "TR_DATE", "NET", "GROSS"] + COMP + STD))

# ============================ T1 · episode level ==============================
print("=" * 70)
print("T1 · R2 + flow-magnitude/imbalance controls (episode level)")
print("=" * 70)
sf = st.join(f.select("cisin", "TR_DATE", "NET", "GROSS"),
             on=["cisin", "TR_DATE"], how="left")
cov = sf["NET"].is_not_null().mean()
print(f"flow-join coverage: {100*float(cov):.2f}%  "
      f"(gate >= 95%: {'PASS' if cov >= .95 else 'FAIL'})")
gbar = (sf.group_by("cisin")
          .agg(pl.col("GROSS").mean().alias("gbar")))
sf = sf.join(gbar, on="cisin")

runs = sf.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1))
     .fill_null(True)).cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first().alias("arch"), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"),
    (pl.col("NET") / (pl.col("GROSS") + 1e-9)).mean().alias("mimb"),
    (pl.col("GROSS") + 1.0).log().mean().alias("mlgr"),
    (pl.col("NET").mean() / (pl.col("gbar").first() + 1e-9))
    .alias("mnet_s"))
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
DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]
FLOW = ["mimb", "mlgr", "mnet_s"]

def run_spec(era, xtra, tag):
    need = ["y20", "cisin", "ed", "month"] + DUM + CTL + xtra
    d = (ev.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    res = PanelOLS(d["y20"], d[DUM + CTL + xtra], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=d[["month"]])
    print(f"\n {tag} {era} (n={res.nobs})")
    for v in DUM + xtra:
        pv = float(res.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        print(f"   {v:14s}{float(res.params[v]):+9.1f}"
              f"  t={float(res.tstats[v]):+6.2f} {s}")
    RES.setdefault(_TAG[0], {})[_TAG[1]] = res
    return (float(res.params["D_SHARK_DIST"]),
            float(res.tstats["D_SHARK_DIST"]),
            float(res.pvalues["D_SHARK_DIST"]))

t1 = {}
RES, _TAG = {}, [None, None]
for era in ("TRAIN", "TEST"):
    _TAG[:] = [era, "base"]
    base = run_spec(era, [], "R2 baseline   ")
    _TAG[:] = [era, "full"]
    full = run_spec(era, FLOW, "R2 + FLOW ctrls")
    t1[era] = (base, full)

# ============================ T2 · daily GBT ==================================
print("\n" + "=" * 70)
print("T2 · LightGBM with vs without the composition block (TEST era)")
print("=" * 70)
d = (f.join(st.select("cisin", "TR_DATE", "era"),
            on=["cisin", "TR_DATE"], how="inner")
       .join(anch.select(pl.col("isin").alias("cisin"),
                         pl.col("date").alias("TR_DATE"),
                         (1e4 * pl.col("post20")).alias("y20")),
             on=["cisin", "TR_DATE"], how="inner")
       .drop_nulls(["y20"]))
print("GBT sample:", d.height, "stock-days")
dtr = d.filter(pl.col("era") == "TRAIN").to_pandas()
dte = d.filter(pl.col("era") == "TEST").to_pandas()

def fit_ic(feats, tag):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05,
                          num_leaves=63, random_state=7, n_jobs=-1,
                          verbose=-1)
    m.fit(dtr[feats], dtr["y20"])
    dte["_p"] = m.predict(dte[feats])
    ics, days = [], sorted(dte["TR_DATE"].unique())
    for day in days:
        g = dte[dte["TR_DATE"] == day]
        if len(g) >= 30:
            ics.append(spearmanr(g["_p"], g["y20"]).statistic)
    ics = np.array([x for x in ics if np.isfinite(x)])
    # quintile spread on non-overlapping 20d anchor days
    sub = dte[dte["TR_DATE"].isin(days[::20])].copy()
    sub["q"] = sub.groupby("TR_DATE")["_p"].transform(
        lambda x: np.ceil(x.rank(pct=True) * 5))
    sp = (sub[sub.q == 5].groupby("TR_DATE")["y20"].mean()
          - sub[sub.q == 1].groupby("TR_DATE")["y20"].mean()).dropna()
    tsp = sp.mean() / sp.std(ddof=1) * np.sqrt(len(sp))
    print(f" {tag}: IC {ics.mean():+.4f} ({len(ics)}d) | "
          f"Q5-Q1 {sp.mean():+.1f}bp (non-overlap t={tsp:+.2f})")
    return ics, float(sp.mean())

ic_full, sp_full = fit_ic(STD + COMP, "FULL (std+comp)")
ic_std, sp_std = fit_ic(STD, "STD only       ")
ic_comp, sp_comp = fit_ic(COMP, "COMP only      ")
n = min(len(ic_full), len(ic_std))
dif = ic_full[:n] - ic_std[:n]
tdif = dif.mean() / dif.std(ddof=1) * np.sqrt(n)
dic = float(ic_full.mean() - ic_std.mean())
print(f"\n dIC (full - std) = {dic:+.4f}, paired daily t = {tdif:+.2f}"
      f" | dSpread = {sp_full - sp_std:+.1f}bp")

# ============================ VERDICTS ========================================
print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICTS")
ok1 = True
for era in ("TRAIN", "TEST"):
    (cb, tb, pb), (cf, tf, pf) = t1[era]
    keep = pf < .05 and abs(cf - cb) <= .5 * abs(cb)
    ok1 = ok1 and keep
    print(f"  T1 {era}: D_SHARK_DIST {cb:+.1f} -> {cf:+.1f} w/ flow "
          f"ctrls (p={pf:.4f}) {'ROBUST' if keep else 'DEGRADED'}")
ok2 = dic >= 0.005 and tdif >= 2 and sp_full > sp_std
print(f"  T2: dIC {dic:+.4f} (bar .005), t {tdif:+.2f} (bar 2) -> "
      f"{'BROAD incremental info' if ok2 else 'below bar'}")
if ok1 and ok2:
    v = "COMPOSITION ADDS INFORMATION, broadly AND in episodes"
elif ok1:
    v = ("COMPOSITION INFORMATION IS EPISODE/TAIL-CONCENTRATED — "
         "robust where claimed (Table 1), not a broad daily alpha")
elif ok2:
    v = "broad daily info but episode claim NOT robust to flow ctrls"
else:
    v = "NOT ESTABLISHED — composition adds no incremental info"
print("\nVERDICT:", v)
print("=" * 70)

# ---------------------------------------------------------------------------
# POST-HOC RE-POINTING (added 2026-08-23, audit item 1). NOT PRE-REGISTERED.
# T1 above is hardcoded to D_SHARK_DIST, which on the corrected concentration
# feature is not distinguishable from zero -- so "DEGRADED" is a statement
# about a null, not about the reversal. T2 is archetype-independent and its
# verdict stands as printed.
# ---------------------------------------------------------------------------
CAND = ["D_HOSTAGE", "D_SHARK_DIST", "D_SHARK_ACC"]
tgt = max(CAND, key=lambda v: float(RES["TRAIN"]["base"].tstats[v]))
print("\nPOST-HOC RE-POINTING — NOT PRE-REGISTERED")
print(f" carrying archetype by TRAIN t-stat: {tgt}")
print(" T1 — does the reversal survive flow magnitude/imbalance controls?")
ok1p = True
for era in ("TRAIN", "TEST"):
    cb = float(RES[era]["base"].params[tgt])
    cf = float(RES[era]["full"].params[tgt])
    pf = float(RES[era]["full"].pvalues[tgt])
    keep = pf < .05 and abs(cf - cb) <= .5 * abs(cb)
    ok1p &= keep
    print(f"   {era:5s} {tgt} {cb:+7.1f} -> {cf:+7.1f} w/ flow ctrls "
          f"({100*cf/cb:.0f}% retained, p={pf:.4f})  "
          f"{'ROBUST' if keep else 'DEGRADED'}")
print(f"   T1(post-hoc): {'PASS' if ok1p else 'FAIL'} "
      f"(needs p<0.05 and <=50% movement in BOTH eras)")
print(" T2 is archetype-independent (a feature-block IC test) — unchanged.")
print("=" * 70)
