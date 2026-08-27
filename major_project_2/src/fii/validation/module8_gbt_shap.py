# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 8 · GBT + SHAP — IS THE HMM THE BOTTLENECK? (strategic fork)
#
# Question: do the flow features contain forward-return signal that the
# HMM-regime pipeline leaves on the table? Same information set (the 10
# probit flow features; HMM used 4), same universe & temporal split, but a
# supervised learner (LightGBM) predicting forward 20d abnormal returns.
#
# CALIBRATION (pre-registered): the VALIDATED regime effect (~+50bp/20d on
# a ~7% tail) is worth daily cross-sectional IC ~0.005-0.01. So GBT vs 0
# is the wrong test; the test is GBT vs a REGIME BASELINE (each day scored
# by its archetype's TRAIN-mean forward return).
# VERDICT RULES:
#   GBT IC ~= baseline IC (gap < ~0.005)  -> features are the ceiling;
#       HMM is NOT the bottleneck; do NOT build LSTM for prediction.
#   GBT IC >> baseline (gap > ~0.01, decile spread monotone, replicated
#       on non-overlapping days) -> extractable signal beyond regimes ->
#       richer models justified.
# SHAP then says WHAT drives any GBT edge (concentration tail again, or
# features the HMM never used).
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

# ---- target: forward 20d abnormal CAR (bp) from the certified panel --------
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

# ---- features + era/archetype on the model universe -------------------------
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
print("features used:", have)
s = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
       .select("cisin", "TR_DATE", "era", "archetype"))
d = s.join(f, on=["cisin", "TR_DATE"], how="inner")
d = d.join(tgt, left_on=["cisin", "TR_DATE"],
           right_on=["isin", "date"], how="inner")
print("model stock-days with features+target:", d.height, "/ 804,958")
tr = d.filter(pl.col("era") == "TRAIN")
te = d.filter(pl.col("era") == "TEST")
print("train:", tr.height, "| test:", te.height)

# ---- regime BASELINE: archetype -> TRAIN-mean forward return ----------------
bmap = dict(tr.group_by("archetype").agg(pl.col("y").mean())
              .iter_rows())
print("\nbaseline scores (TRAIN mean y by archetype, bp):",
      {k: round(v, 1) for k, v in bmap.items()})
te = te.with_columns(
    pl.col("archetype").replace_strict(bmap, default=0.0).alias("pb"))

# ---- LightGBM on TRAIN only --------------------------------------------------
Xtr = tr.select(have).to_numpy()
ytr = tr["y"].to_numpy()
Xte = te.select(have).to_numpy()
model = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=42, verbose=-1)
model.fit(Xtr, ytr)
te = te.with_columns(pl.Series("pg", model.predict(Xte)))
tr = tr.with_columns(pl.Series("pg", model.predict(Xtr)))

# ---- daily cross-sectional Spearman IC ---------------------------------------
def daily_ic(df, pred):
    out = []
    for _, g in df.select("TR_DATE", pred, "y").to_pandas() \
                  .groupby("TR_DATE"):
        if len(g) >= 30 and g[pred].nunique() > 1:
            ic = spearmanr(g[pred], g["y"]).statistic
            if not np.isnan(ic):
                out.append(ic)
    return np.array(out)

def report(name, ics):
    no = ics[::20]           # ~non-overlapping targets
    t = no.mean() / (no.std(ddof=1) / np.sqrt(len(no))) if len(no) > 2 \
        else float("nan")
    print(f"  {name:<26} mean IC {ics.mean():+.4f} | %pos "
          f"{100*(ics>0).mean():.0f}% | non-overlap t {t:+.2f} "
          f"(n_days={len(ics)})")

print("\n=== TEST-era daily cross-sectional IC (the verdict numbers) ===")
ic_g = daily_ic(te, "pg")
ic_b = daily_ic(te, "pb")
report("GBT (10 features)", ic_g)
report("regime baseline", ic_b)
print("  GAP (GBT - baseline):", round(ic_g.mean() - ic_b.mean(), 4))
ic_tr = daily_ic(tr, "pg")
report("GBT on TRAIN (overfit gauge)", ic_tr)

print("\nsingle-feature TEST ICs (context — is any raw feature enough?):")
for c in have:
    sub = te.select("TR_DATE", pl.col(c).alias("p1"), "y")
    ics = daily_ic(sub, "p1")
    print(f"  {c:<16} {ics.mean():+.4f}")

# ---- quintile spread ----------------------------------------------------------
print("\n=== TEST quintile spread on GBT prediction (bp over 20d) ===")
tep = te.select("TR_DATE", "pg", "y").to_pandas()
tep["q"] = tep.groupby("TR_DATE")["pg"].transform(
    lambda x: np.ceil(x.rank(pct=True) * 5).clip(1, 5))
qm = tep.groupby("q")["y"].mean()
for q in sorted(qm.index):
    print(f"  Q{int(q)}  {qm[q]:+7.1f}")
spread = tep[tep.q == 5].groupby("TR_DATE")["y"].mean() \
    - tep[tep.q == 1].groupby("TR_DATE")["y"].mean()
sp = spread.dropna().to_numpy()
no = sp[::20]
t = no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))
print(f"  Q5-Q1 spread: {sp.mean():+.1f} bp | non-overlap t {t:+.2f}")

# ---- SHAP (LightGBM native pred_contrib = exact TreeSHAP) --------------------
print("\n=== SHAP on TEST (what drives the GBT signal) ===")
idx = np.random.default_rng(42).choice(
    len(Xte), size=min(60000, len(Xte)), replace=False)
contrib = model.predict(Xte[idx], pred_contrib=True)
sh = contrib[:, :-1]
imp = np.abs(sh).mean(axis=0)
order = np.argsort(imp)[::-1]
print("feature".ljust(16) + "mean|SHAP|".rjust(11) + "  direction")
for j in order:
    x = Xte[idx][:, j]
    m = ~np.isnan(x)
    rho = np.corrcoef(sh[m, j], x[m])[0, 1] if m.sum() > 100 else np.nan
    print(have[j].ljust(16) + ("%.2f" % imp[j]).rjust(11)
          + f"   corr(shap,x) {rho:+.2f}")

print("""
VERDICT RULES (pre-registered above):
 - GAP < ~0.005 and quintile spread weak -> the 10 features are the
   ceiling; HMM is NOT the bottleneck; LSTM/deeper models NOT justified
   for return prediction. Regimes stand as the right description.
 - GAP > ~0.01 with monotone quintiles and non-overlap t > 2 -> real
   signal beyond the regimes; richer supervised models justified;
   SHAP ranking says where to look (if F_entity/F_entity_buy dominate,
   it is the same concentration tail, just used more efficiently).
 - TRAIN-vs-TEST IC ratio is the overfit gauge (huge train IC with tiny
   test IC = memorization, expected for financial noise).
""")
