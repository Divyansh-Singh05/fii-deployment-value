# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 8B · STATIC-CHARACTERISTIC vs DYNAMIC-FLOW DECOMPOSITION
#
# Module 8 found GBT signal beyond the regimes, but SHAP flagged F_breadth
# (standalone IC -0.135) as a likely quasi-static size/liquidity proxy.
# A raw cross-sectional IC has no stock fixed effect: a feature that is
# CONSTANT per stock can "predict" returns by just ranking stocks. This
# decomposes the Module-8 edge:
#
#  A  STATICNESS per feature: between-stock variance share + per-stock-mean
#     rank stability TRAIN->TEST (a true characteristic has high both).
#  B  Raw vs WITHIN-STOCK-DEMEANED single-feature ICs (demeaning uses
#     TRAIN-era per-stock means only -> no look-ahead; test-only stocks
#     get 0 = the cross-sectional center of probit features).
#  C  DYNAMIC GBT: retrain LightGBM on the demeaned features; TEST IC +
#     quintile spread = the extractable signal from flow DYNAMICS alone.
#
# DECISION (pre-registered): if the dynamic GBT keeps a monotone quintile
# spread with non-overlap t > 2, the sequence-model (LSTM) path is
# justified on dynamics. If the spread collapses, Module 8's extra edge
# was characteristics-in-disguise -> regimes remain the right dynamic
# description; enrich with characteristics, do NOT build an LSTM.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lightgbm"])
    import lightgbm as lgb

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
FEATS = ["F_persist", "F_block", "F_entity_conc", "F_entity_conc_buy", "F_breadth",
         "F_sizedisp", "F_flowbeta", "F_activity", "F_streak", "F_imbal"]

# ---- same data assembly as module 8 -----------------------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "ret_adj_mktadj").sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-0.5, 0.5).fill_null(0.0).alias("ar"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
p = p.with_columns(
    ((pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                  pl.col("cum").last().over("isin"))
      - pl.col("cum")) * 1e4).alias("y"))
tgt = p.select("isin", "date", "y")

f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
# AUDIT — a missing feature must HALT, not silently vanish. The audit rebuild
# renamed F_entity -> F_entity_conc, and this pattern quietly ran the GBT
# challenger on 8 of 10 features with the whole composition block absent, then
# reported PASS. A stage that cannot see its own inputs must say so.
_missing = [c for c in FEATS if c not in f.columns]
if _missing:
    raise SystemExit(f"FEATURE STORE MISSING {_missing}; have {f.columns}")
f = f.select(["cisin", "TR_DATE"] + [c for c in FEATS if c in f.columns])
have = [c for c in FEATS if c in f.columns]
s = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
       .select("cisin", "TR_DATE", "era"))
d = s.join(f, on=["cisin", "TR_DATE"], how="inner")
d = d.join(tgt, left_on=["cisin", "TR_DATE"],
           right_on=["isin", "date"], how="inner")
tr = d.filter(pl.col("era") == "TRAIN")
te = d.filter(pl.col("era") == "TEST")
print("train:", tr.height, "| test:", te.height)

def daily_ic(df, pred):
    out = []
    for _, g in df.select("TR_DATE", pred, "y").to_pandas() \
                  .groupby("TR_DATE"):
        g = g.dropna()
        if len(g) >= 30 and g[pred].nunique() > 1:
            ic = spearmanr(g[pred], g["y"]).statistic
            if not np.isnan(ic):
                out.append(ic)
    return np.array(out)

# ---- A: staticness ------------------------------------------------------------
print("\n=== A · STATICNESS (is the feature a stock characteristic?) ===")
print("feature".ljust(15) + "between-stock var%".rjust(19)
      + "  rank-stab TRAIN->TEST")
trm = tr.group_by("cisin").agg(
    [pl.col(c).mean().alias(c) for c in have])
tem = te.group_by("cisin").agg(
    [pl.col(c).mean().alias(c) for c in have])
mm = trm.join(tem, on="cisin", how="inner", suffix="_te")
for c in have:
    tot = d[c].drop_nulls().var()
    sm = d.group_by("cisin").agg(pl.col(c).mean().alias("m"),
                                 pl.len().alias("n"))
    grand = d[c].drop_nulls().mean()
    btw = float((sm["n"] * (sm["m"] - grand) ** 2).sum()
                / max(d[c].drop_nulls().len() - 1, 1))
    share = 100 * btw / tot if tot and tot > 0 else float("nan")
    a = mm[c].to_numpy(); b = mm[c + "_te"].to_numpy()
    msk = ~(np.isnan(a) | np.isnan(b))
    stab = spearmanr(a[msk], b[msk]).statistic if msk.sum() > 30 \
        else float("nan")
    print(c.ljust(15) + ("%.1f%%" % share).rjust(19)
          + ("%.2f" % stab).rjust(12))
print("(high var% + high stability = characteristic, not flow signal)")

# ---- B: raw vs demeaned single-feature ICs ------------------------------------
print("\n=== B · single-feature TEST ICs: RAW vs WITHIN-STOCK DEMEANED ===")
means = trm  # TRAIN-era per-stock means (no look-ahead)
ted = te.join(means, on="cisin", how="left", suffix="_m")
for c in have:
    ted = ted.with_columns(
        (pl.col(c) - pl.col(c + "_m").fill_null(0.0)).alias(c + "_d"))
print("feature".ljust(15) + "raw IC".rjust(9) + "demeaned IC".rjust(13))
for c in have:
    r = daily_ic(te.select("TR_DATE", pl.col(c).alias("p1"), "y"), "p1")
    dm = daily_ic(ted.select("TR_DATE", pl.col(c + "_d").alias("p1"),
                             "y"), "p1")
    rv = r.mean() if len(r) else float("nan")
    dv = dm.mean() if len(dm) else float("nan")
    print(c.ljust(15) + ("%+.4f" % rv).rjust(9)
          + ("%+.4f" % dv).rjust(13))
print("(a raw IC that collapses when demeaned was a characteristic)")

# ---- C: dynamic GBT (trained on demeaned features) ----------------------------
print("\n=== C · DYNAMIC GBT (demeaned features only) ===")
trd = tr.join(means, on="cisin", how="left", suffix="_m")
for c in have:
    trd = trd.with_columns(
        (pl.col(c) - pl.col(c + "_m").fill_null(0.0)).alias(c + "_d"))
dcols = [c + "_d" for c in have]
model = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=42, verbose=-1)
model.fit(trd.select(dcols).to_numpy(), trd["y"].to_numpy())
ted = ted.with_columns(
    pl.Series("pg", model.predict(ted.select(dcols).to_numpy())))
ics = daily_ic(ted, "pg")
no = ics[::20]
t_ic = no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))
print(f"dynamic GBT TEST IC: {ics.mean():+.4f} | %pos "
      f"{100*(ics>0).mean():.0f}% | non-overlap t {t_ic:+.2f}")
tep = ted.select("TR_DATE", "pg", "y").to_pandas()
tep["q"] = tep.groupby("TR_DATE")["pg"].transform(
    lambda x: np.ceil(x.rank(pct=True) * 5).clip(1, 5))
qm = tep.groupby("q")["y"].mean()
for q in sorted(qm.index):
    print(f"  Q{int(q)}  {qm[q]:+7.1f}")
spread = tep[tep.q == 5].groupby("TR_DATE")["y"].mean() \
    - tep[tep.q == 1].groupby("TR_DATE")["y"].mean()
sp = spread.dropna().to_numpy()
no = sp[::20]
t_sp = no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))
print(f"  Q5-Q1: {sp.mean():+.1f} bp | non-overlap t {t_sp:+.2f}")

# SHAP of the dynamic model — what carries the dynamic signal
idx = np.random.default_rng(42).choice(
    ted.height, size=min(60000, ted.height), replace=False)
Xs = ted.select(dcols).to_numpy()[idx]
contrib = model.predict(Xs, pred_contrib=True)
sh = np.abs(contrib[:, :-1]).mean(axis=0)
order = np.argsort(sh)[::-1]
print("\ndynamic-GBT SHAP ranking:")
for j in order:
    print("  " + dcols[j].ljust(18) + "%.2f" % sh[j])

print("""
DECISION (pre-registered):
 - dynamic GBT keeps monotone quintiles with spread t > 2  ->  genuine
   flow-DYNAMICS signal beyond the regimes; sequence-model (LSTM) path
   justified; its SHAP ranking names the carrying features.
 - spread collapses (t < 2 / non-monotone) -> Module 8's edge was mostly
   static characteristics; regimes stay the right dynamic description;
   enrich with characteristics instead of building an LSTM.
Either way: any successor model gets the FULL Module 5-7 validation
battery (temporal split, gates, placebo, event-study, panel regression).
""")
