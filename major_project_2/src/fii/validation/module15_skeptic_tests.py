# [written directly in-repo; Colab copy in legacy/ and ~/Desktop/temp]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 15 · SKEPTIC TESTS — three first-order holes found by a fresh
# hostile read of the whole project (log §3r). Pre-registered verdicts.
#
# T1 · BID-ASK BOUNCE: episode ends after concentrated selling close at
#     the bid; mechanical mid-reversion fakes a "+reversal". Test: R2
#     spec with the forward window DELAYED to t+3..t+22 (skips the
#     bounce horizon). PASS (bounce not the driver) if delayed-start
#     D_SHARK_DIST >= 60% of the standard coefficient AND p<0.05, both
#     eras. Report the implied bounce component (standard - delayed).
#
# T2 · THE DIRECT CONTRAST: the paper's claim is SD reverts, HOSTAGE
#     does not — but "significant vs not significant" is not a test.
#     Compute the linear-combination test b_SD - b_HO = 0 from the
#     fitted covariance. PASS if p<0.05 both eras.
#
# T3 · PUBLIC-DATA BENCHMARK (the volume-conditioned-reversal prior
#     art, Conrad-Hameed-Niden 1994 etc.):
#  a) daily GBT ladder: PUBLIC (price/volume only) -> +FLOW
#     (conventional private flow features) -> +COMPOSITION. Composition
#     must add beyond BOTH: paired daily-IC t >= 2 for each increment.
#  b) episode head-to-head: add a public volume-conditioned-reversal
#     dummy D_VCR (era-bottom-quintile pre20 AND relvol>1.1) to R2.
#     PASS if D_SHARK_DIST keeps |t|>=2 both eras alongside it.
# ============================================================================
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from linearmodels.panel import PanelOLS
import lightgbm as lgb

DRIVE, MODELD = VALIDATION_DATA, ISIN_MAPPING
COMP = ["F_entity_conc", "F_entity_conc_buy", "F_breadth"]
FLOW = ["F_persist", "F_block", "F_imbal", "F_streak", "F_activity",
        "F_flowbeta", "F_sizedisp"]
PUB = ["pre20bp", "relvol", "vol20", "mombp", "amihud", "logto",
       "logclose"]

# ---- panel characteristics (module7b construction, + delayed window) ---------
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
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
last = pl.col("cum").last().over("isin")
p = p.with_columns(
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"), last)
     - pl.col("cum")).alias("post20"),
    (pl.coalesce(pl.col("cum").shift(-22).over("isin"), last)
     - pl.coalesce(pl.col("cum").shift(-2).over("isin"), last))
    .alias("post20d"))                       # t+3 .. t+22 (bounce-free)
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
     * 1e9 + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anch = p.select("isin", "date", "pre20", "post20", "post20d", "vol20",
                "relvol", "logclose", "beta120", "mom", "amihud",
                "logto")

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
runs = st.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1))
     .fill_null(True)).cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first().alias("arch"), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
ev = runs.join(anch, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    (1e4 * pl.col("post20")).alias("y20"),
    (1e4 * pl.col("post20d")).alias("y20d"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a in ("HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"):
    ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                         .alias("D_" + a))
# public volume-conditioned-reversal dummy (era-frozen 20th pct of pre20)
q20 = {e: float(ev.filter(pl.col("era") == e)["pre20bp"].quantile(0.2))
       for e in ("TRAIN", "TEST")}
ev = ev.with_columns(
    pl.when(((pl.col("era") == "TRAIN")
             & (pl.col("pre20bp") < q20["TRAIN"])
             | (pl.col("era") == "TEST")
             & (pl.col("pre20bp") < q20["TEST"]))
            & (pl.col("relvol") > 1.1)).then(1.0).otherwise(0.0)
    .alias("D_VCR"))

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]


def fit(era, ycol, xtra=None):
    x = DUM + CTL + (xtra or [])
    need = [ycol, "cisin", "ed", "month"] + x
    d = (ev.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    return PanelOLS(d[ycol], d[x], entity_effects=True,
                    time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=d[["month"]])


def line(res, v):
    pv = float(res.pvalues[v])
    s = "***" if pv < .01 else "**" if pv < .05 else \
        "*" if pv < .10 else ""
    return (f"   {v:14s}{float(res.params[v]):+9.1f}  "
            f"t={float(res.tstats[v]):+6.2f} {s}")


def contrast(res, a="D_SHARK_DIST", b="D_HOSTAGE"):
    d = float(res.params[a] - res.params[b])
    V = res.cov
    se = float(np.sqrt(V.loc[a, a] + V.loc[b, b] - 2 * V.loc[a, b]))
    t = d / se
    from scipy.stats import norm
    return d, t, 2 * (1 - norm.cdf(abs(t)))

print("=" * 70)
print("T1/T2 · standard vs bounce-free window + the direct contrast")
print("=" * 70)
res_std, res_del = {}, {}
for era in ("TRAIN", "TEST"):
    r1 = fit(era, "y20")
    r2 = fit(era, "y20d")
    res_std[era], res_del[era] = r1, r2
    print(f"\n {era} standard (t+1..t+20, n={r1.nobs}):")
    for v in ("D_SHARK_DIST", "D_HOSTAGE", "D_SHARK_ACC"):
        print(line(r1, v))
    d, t, pv = contrast(r1)
    print(f"   SD-HO contrast {d:+9.1f}  t={t:+6.2f}  p={pv:.4f}")
    print(f" {era} bounce-free (t+3..t+22, n={r2.nobs}):")
    for v in ("D_SHARK_DIST", "D_HOSTAGE", "D_SHARK_ACC"):
        print(line(r2, v))
    d2, t2, pv2 = contrast(r2)
    print(f"   SD-HO contrast {d2:+9.1f}  t={t2:+6.2f}  p={pv2:.4f}")
    sd1 = float(r1.params["D_SHARK_DIST"])
    sd2 = float(r2.params["D_SHARK_DIST"])
    print(f"   implied bounce component: {sd1 - sd2:+.1f}bp "
          f"({100*(sd1-sd2)/sd1:.0f}% of standard)")

print("\n" + "=" * 70)
print("T3b · episode head-to-head vs public VCR dummy")
print("=" * 70)
res_vcr = {}
for era in ("TRAIN", "TEST"):
    r = fit(era, "y20", ["D_VCR"])
    res_vcr[era] = r
    print(f"\n {era} (n={r.nobs}):")
    for v in ("D_SHARK_DIST", "D_HOSTAGE", "D_VCR"):
        print(line(r, v))

print("\n" + "=" * 70)
print("T3a · daily GBT ladder: PUBLIC -> +FLOW -> +COMPOSITION")
print("=" * 70)
f = (pl.read_parquet(MODELD / "stockday_features_v2.parquet")
       .select(["cisin", "TR_DATE"] + FLOW + COMP))
d = (f.join(st.select("cisin", "TR_DATE", "era"),
            on=["cisin", "TR_DATE"], how="inner")
       .join(anch.select(pl.col("isin").alias("cisin"),
                         pl.col("date").alias("TR_DATE"),
                         (1e4 * pl.col("post20")).alias("y20"),
                         (1e4 * pl.col("pre20")).alias("pre20bp"),
                         "relvol", "vol20",
                         (1e4 * pl.col("mom")).alias("mombp"),
                         "amihud", "logto", "logclose"),
             on=["cisin", "TR_DATE"], how="inner")
       .drop_nulls(["y20"]))
print("sample:", d.height, "stock-days")
dtr = d.filter(pl.col("era") == "TRAIN").to_pandas()
dte = d.filter(pl.col("era") == "TEST").to_pandas()


def fit_ic(feats, tag):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05,
                          num_leaves=63, random_state=7, n_jobs=-1,
                          verbose=-1)
    m.fit(dtr[feats], dtr["y20"])
    dte["_p"] = m.predict(dte[feats])
    ics = []
    for day, g in dte.groupby("TR_DATE"):
        if len(g) >= 30:
            v = spearmanr(g["_p"], g["y20"]).statistic
            if np.isfinite(v):
                ics.append(v)
    ics = np.array(ics)
    print(f" {tag:18s} TEST IC {ics.mean():+.4f} ({len(ics)}d)")
    return ics


ic_p = fit_ic(PUB, "PUBLIC")
ic_pc = fit_ic(PUB + COMP, "PUBLIC+COMP")
ic_pf = fit_ic(PUB + FLOW, "PUBLIC+FLOW")
ic_all = fit_ic(PUB + FLOW + COMP, "PUBLIC+FLOW+COMP")


def paired(a, b, tag):
    n = min(len(a), len(b))
    dif = a[:n] - b[:n]
    t = dif.mean() / dif.std(ddof=1) * np.sqrt(n)
    print(f" d({tag}) = {dif.mean():+.4f}, paired t = {t:+.2f}")
    return float(dif.mean()), float(t)

d1, t1v = paired(ic_pc, ic_p, "PUB+COMP - PUB")
d2, t2v = paired(ic_all, ic_pf, "PUB+FLOW+COMP - PUB+FLOW")

print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICTS")
ok1 = all(
    float(res_del[e].params["D_SHARK_DIST"])
    >= 0.6 * float(res_std[e].params["D_SHARK_DIST"])
    and float(res_del[e].pvalues["D_SHARK_DIST"]) < .05
    for e in ("TRAIN", "TEST"))
print(f" T1 bounce: {'PASS — bounce is not the driver' if ok1 else 'FAIL — the reversal is materially bid-ask bounce'}")
ok2 = all(contrast(res_std[e])[2] < .05 for e in ("TRAIN", "TEST"))
print(f" T2 SD-HO contrast: {'PASS — the headline contrast is itself significant' if ok2 else 'FAIL — SD and HOSTAGE are not distinguishable'}")
ok3b = all(abs(float(res_vcr[e].tstats["D_SHARK_DIST"])) >= 2
           for e in ("TRAIN", "TEST"))
print(f" T3b vs public VCR dummy: {'PASS — concentration beats the public proxy head-to-head' if ok3b else 'FAIL — a free volume signal absorbs it'}")
ok3a = t1v >= 2 and t2v >= 2
print(f" T3a GBT ladder: {'PASS — composition adds beyond public AND conventional flow' if ok3a else 'PARTIAL/FAIL — see deltas above'}")
print("=" * 70)

# ---------------------------------------------------------------------------
# POST-HOC RE-POINTING  (added 2026-08-23, audit item 1)
#
# NOT A PRE-REGISTERED RESULT. The verdicts above are hardcoded to
# D_SHARK_DIST because that is where the effect sat when they were written.
# On the corrected concentration feature SHARK_DIST is a null coefficient, so
# T1 divides a bounce-free coefficient by a coefficient that is not
# distinguishable from zero, and T3b asks whether a null survives a control.
# Neither question is meaningful, and "FAIL" is not the right reading of
# either.
#
# Below, the SAME three gates are applied to whichever archetype actually
# carries the reversal, chosen by a mechanical rule stated before looking:
# the largest t-statistic in the TRAIN standard-window regression. This is an
# exploratory re-test on data already seen, and must be reported as such --
# it cannot restore the evidential status of a pre-registration.
# ---------------------------------------------------------------------------
CAND = ["D_HOSTAGE", "D_SHARK_DIST", "D_SHARK_ACC"]
target = max(CAND, key=lambda v: float(res_std["TRAIN"].tstats[v]))
other = "D_HOSTAGE" if target != "D_HOSTAGE" else "D_SHARK_DIST"

print("\nPOST-HOC RE-POINTING — NOT PRE-REGISTERED")
print(f" carrying archetype by TRAIN t-stat: {target} "
      f"(t={float(res_std['TRAIN'].tstats[target]):+.2f})")

print(f"\n T1 bounce — does {target} survive a bounce-free window?")
t1_ok = True
for e in ("TRAIN", "TEST"):
    a = float(res_std[e].params[target])
    b = float(res_del[e].params[target])
    pv = float(res_del[e].pvalues[target])
    keep = b / a if a else float("nan")
    ok = keep >= 0.6 and pv < .05
    t1_ok &= ok
    print(f"   {e:5s} standard {a:+7.1f} -> bounce-free {b:+7.1f} "
          f"({100*keep:4.0f}% retained, p={pv:.4f})  "
          f"bounce {a-b:+.1f}bp ({100*(a-b)/a:.0f}%)  {'ok' if ok else 'MISS'}")
print(f"   T1(post-hoc): {'PASS' if t1_ok else 'FAIL'} "
      f"(needs >=60% retained AND p<0.05 in BOTH eras)")

print(f"\n T2 contrast — are {target} and {other} distinguishable?")
print("   (symmetric: re-pointing cannot change this p-value)")
t2_ok = True
for e in ("TRAIN", "TEST"):
    d, t, pv = contrast(res_std[e], target, other)
    t2_ok &= pv < .05
    print(f"   {e:5s} {target}-{other} {d:+7.1f}  t={t:+6.2f}  p={pv:.4f}"
          f"  {'ok' if pv < .05 else 'MISS'}")
print(f"   T2(post-hoc): {'PASS' if t2_ok else 'FAIL'}")

print(f"\n T3b public VCR dummy — is {target} absorbed by free volume info?")
t3b_ok = True
for e in ("TRAIN", "TEST"):
    t = float(res_vcr[e].tstats[target])
    tv = float(res_vcr[e].tstats["D_VCR"])
    t3b_ok &= abs(t) >= 2
    print(f"   {e:5s} {target} t={t:+6.2f}   D_VCR t={tv:+6.2f}"
          f"  {'ok' if abs(t) >= 2 else 'MISS'}")
print(f"   T3b(post-hoc): {'PASS' if t3b_ok else 'FAIL'} "
      f"(needs |t|>=2 in BOTH eras alongside D_VCR)")

print("\n T3a GBT ladder is archetype-independent — it tests the composition"
      "\n feature block, so re-pointing does not apply. Verdict unchanged.")
print("=" * 70)
