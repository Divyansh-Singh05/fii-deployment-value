# [Phase II — charter: docs/PHASE2_PLAN.md]
from fii.paths import VALIDATION_DATA  # noqa: E402
# ============================================================================
# MODULE 16C · EPISODE-END HAZARD MODEL — the Phase-II centerpiece
#
# The reversal starts when concentrated selling STOPS, so the valuable
# forecast is the episode END. Standing at close of day t inside a
# (causal, filtered-label) SHARK_DIST episode, predict:
#     target_k = 1[ episode's last day occurs within the next k days ]
#                (k = 1: today was the last day), k in {1, 3, 5}.
#
# Features (info through t only): episode age, the four state features,
# p_sell posterior, within-episode concentration drift (vs episode
# start; vs 3 days back), relative volume, log turnover.
#
# PROTOCOL (walk-forward, charter rules):
#  - eval years 2014..2024 + partial 2025 (data ends 2025-03-28; final
#    fold reported as partial); train = all episode-days dated
#    >= 10 days before the eval year starts (expanding window); model
#    refit per fold; nothing tuned on eval folds.
#  - runs broken on calendar gaps > 21 days (mask/suspensions cannot
#    weld two episodes); right-censored runs (end unobserved at the
#    stock's last labeled day) contribute no targets they cannot label.
#
# BASELINES: KM  = age-only discrete hazard (train-window Kaplan-Meier,
#                  ages capped at 15+), composed to k-day probability;
#            CONST = train base rate of target_k.
#
# PRE-REGISTERED BAR: model beats KM on pooled out-of-window log-loss
# with paired daily t >= 2 for BOTH k=1 and k=3 (k=5 reported; its base
# rate is high since median episode is ~3 days). AUC and era split
# reported. Output: phase2_hazard_preds.parquet (for 16D).
# ============================================================================
import numpy as np
import polars as pl
import lightgbm as lgb

DRIVE = VALIDATION_DATA
KS = (1, 3, 5)
AGE_CAP = 15

# ---- data: filtered labels + features + volume context -----------------------
fs = (pl.read_parquet(DRIVE / "phase2_filtered_states.parquet")
        .select("cisin", "TR_DATE", "era", "p_sell", "farch"))
sv = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .select("cisin", "TR_DATE", "F_persist", "F_block",
                "F_entity_s", "F_entity_buy_s"))
d = fs.join(sv, on=["cisin", "TR_DATE"], how="inner")
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume").sort(["isin", "date"]))
p = p.with_columns((pl.col("close") * pl.col("volume")).alias("to"))
p = p.with_columns(
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_tma"),
    pl.col("volume").rolling_mean(window_size=20).over("isin")
      .alias("_vma"))
p = p.with_columns(
    (pl.col("volume") / pl.col("_vma").shift(1).over("isin"))
    .alias("relvol"),
    (pl.col("_tma").shift(1).over("isin") + 1.0).log().alias("logto"))
d = d.join(p.select("isin", "date", "relvol", "logto"),
           left_on=["cisin", "TR_DATE"], right_on=["isin", "date"],
           how="left").sort(["cisin", "TR_DATE"])

# ---- SHARK_DIST runs with gap-breaking ----------------------------------------
d = d.with_columns(
    ((pl.col("farch") != pl.col("farch").shift(1)).fill_null(True)
     | ((pl.col("TR_DATE") - pl.col("TR_DATE").shift(1))
        .dt.total_days() > 21))
    .cum_sum().over("cisin").alias("_r"))
sd = d.filter(pl.col("farch") == "SHARK_DIST")
# per-run indices
sd = sd.with_columns(
    pl.int_range(pl.len()).over("cisin", "_r").alias("age0"),
    pl.len().over("cisin", "_r").alias("runlen"),
    pl.col("F_entity_s").first().over("cisin", "_r").alias("ent0"),
    pl.col("F_entity_s").shift(3).over("cisin", "_r").alias("ent_l3"),
    pl.col("TR_DATE").last().over("cisin").alias("last_day"),
    pl.col("TR_DATE").last().over("cisin", "_r").alias("run_end"))
sd = sd.with_columns(
    (pl.col("age0") + 1).alias("age"),
    (pl.col("runlen") - 1 - pl.col("age0")).alias("R"),
    (pl.col("run_end") == pl.col("last_day")).alias("censored"),
    (pl.col("F_entity_s") - pl.col("ent0")).alias("d_ent"),
    (pl.col("F_entity_s") - pl.col("ent_l3")).fill_null(0.0)
    .alias("d3_ent"))
print("SHARK_DIST episode-days:", sd.height,
      "| runs:", sd.select("cisin", "_r").unique().height,
      "| censored runs:",
      sd.filter(pl.col("censored")).select("cisin", "_r")
        .unique().height)

FEATS = ["age", "F_persist", "F_block", "F_entity_s", "F_entity_buy_s",
         "p_sell", "d_ent", "d3_ent", "relvol", "logto"]
for k in KS:
    sd = sd.with_columns(
        pl.when(pl.col("censored") & (pl.col("R") < k)).then(None)
          .otherwise((pl.col("R") <= k - 1).cast(pl.Int8))
        .alias(f"y{k}"))
pdf = sd.select(["cisin", "TR_DATE", "era", "R", "age"] + FEATS[1:]
                + [f"y{k}" for k in KS]).to_pandas()
pdf["year"] = pdf["TR_DATE"].dt.year
print("targets base rate (uncensored):",
      {f"k={k}": round(float(pdf[f"y{k}"].mean()), 3) for k in KS})

# ---- KM baseline helpers -------------------------------------------------------
def km_hazard(train):
    """h(a) = P(R=0 | age=a) from train rows; ages capped."""
    a = np.minimum(train["age"].to_numpy(), AGE_CAP)
    end = (train["R"].to_numpy() == 0).astype(float)
    h = np.zeros(AGE_CAP + 1)
    for age in range(1, AGE_CAP + 1):
        sel = a == age
        h[age] = end[sel].mean() if sel.sum() >= 50 else end.mean()
    return h

def km_prob(h, ages, k):
    """P(end within k days | alive at age a) = 1 - prod(1-h(a..a+k-1))."""
    a = np.minimum(ages, AGE_CAP)
    out = np.zeros(len(a))
    for i, age in enumerate(a):
        s = 1.0
        for j in range(k):
            s *= 1 - h[min(age + j, AGE_CAP)]
        out[i] = 1 - s
    return out

# ---- walk-forward ---------------------------------------------------------------
EPS = 1e-6
preds = []
years = list(range(2014, 2026))
for k in KS:
    dk = pdf.dropna(subset=[f"y{k}"]).copy()
    for y in years:
        tr = dk[dk["TR_DATE"] < np.datetime64(f"{y}-01-01") -
                np.timedelta64(10, "D")]
        ev = dk[dk["year"] == y]
        if len(ev) == 0 or len(tr) < 2000:
            continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05,
                               num_leaves=31, min_child_samples=50,
                               random_state=7, n_jobs=-1, verbose=-1)
        m.fit(tr[FEATS], tr[f"y{k}"])
        pm = np.clip(m.predict_proba(ev[FEATS])[:, 1], EPS, 1 - EPS)
        h = km_hazard(tr)
        pk = np.clip(km_prob(h, ev["age"].to_numpy(), k), EPS, 1 - EPS)
        pc = np.clip(np.full(len(ev), tr[f"y{k}"].mean()), EPS, 1 - EPS)
        preds.append(pl.DataFrame({
            "cisin": ev["cisin"].to_numpy(),
            "TR_DATE": ev["TR_DATE"].to_numpy(),
            "era": ev["era"].to_numpy(), "k": k, "year": y,
            "y": ev[f"y{k}"].to_numpy().astype(np.int8),
            "p_model": pm, "p_km": pk, "p_const": pc}))
P = pl.concat(preds)
P.write_parquet(DRIVE / "phase2_hazard_preds.parquet")
print("\nOOW predictions:", P.height, "rows across",
      P["year"].n_unique(), "folds  (2025 fold = Q1 partial)")

# ---- evaluation -----------------------------------------------------------------
def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def auc(y, p):
    r = p.argsort().argsort().astype(float) + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

print("\n=== pooled out-of-window results ===")
ver = {}
for k in KS:
    q = P.filter(pl.col("k") == k)
    y = q["y"].to_numpy().astype(float)
    lm = ll(y, q["p_model"].to_numpy())
    lk = ll(y, q["p_km"].to_numpy())
    lc = ll(y, q["p_const"].to_numpy())
    # paired daily t on log-loss difference (KM - model)
    dts = q["TR_DATE"].to_numpy()
    uniq, inv = np.unique(dts, return_inverse=True)
    dd = np.bincount(inv, weights=lk - lm) / np.bincount(inv)
    t = dd.mean() / dd.std(ddof=1) * np.sqrt(len(dd))
    ver[k] = t
    print(f" k={k}: logloss model {lm.mean():.4f} | KM {lk.mean():.4f}"
          f" | const {lc.mean():.4f} | paired t (vs KM) {t:+.2f}"
          f" | AUC {auc(y, q['p_model'].to_numpy()):.3f}"
          f" (KM {auc(y, q['p_km'].to_numpy()):.3f})")
    for e in ("TRAIN", "TEST"):
        s = q.filter(pl.col("era") == e)
        if s.height:
            ys = s["y"].to_numpy().astype(float)
            print(f"    {e:5s} n={s.height:6d}  model "
                  f"{ll(ys, s['p_model'].to_numpy()).mean():.4f}  KM "
                  f"{ll(ys, s['p_km'].to_numpy()).mean():.4f}  AUC "
                  f"{auc(ys, s['p_model'].to_numpy()):.3f}")

print("\n" + "=" * 70)
ok = ver[1] >= 2 and ver[3] >= 2
print("VERDICT:", "16C PASS — the hazard model beats the age-only "
      "baseline (k=1 & k=3, paired t>=2); episode ends are "
      "forecastable beyond duration alone. Proceed to 16D." if ok else
      "16C FAIL — episode ends are not forecastable beyond age; the "
      "decision layer would have no edge over waiting for "
      "confirmation. Report and stop.")
print("=" * 70)
