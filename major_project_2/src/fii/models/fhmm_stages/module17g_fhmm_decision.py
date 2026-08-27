from fii.paths import VALIDATION_DATA  # noqa: E402
# ============================================================================
# MODULE 17G · FACTORIAL HMM — DECISION LAYER (16D protocol)
#
# Anticipation vs confirmation on the FHMM's causal SHARK_DIST
# episodes, paired within episode, cost-neutral by construction:
#     gain = CAR(t_a+1 .. end+1),
# where t_a = first in-episode day with p_model(k=1) > theta.  Theta is
# learned in-window (expanding, episodes ending before the eval year,
# grid 0.15..0.70 step .05, >=200 history episodes), frozen, applied
# out of window; first decision year 2016.  The same rule builds the
# KM-based anticipator — the "age alone could do it" control that
# decided 16D against the naive model (16D: model gain +23bp/episode,
# CI>0, but paired model-vs-KM t = -2.96).
#
# PRE-REGISTERED VERDICT (16D, unchanged; TEST era pooled):
#   PASS = model-anticipation mean gain > 0 with date-clustered
#   bootstrap 95% CI > 0, AND model beats KM-anticipation on common
#   episodes with paired t >= 2.
#   Else: reported as "forecastable but not monetizable over
#   confirmation" — the 16D outcome; the interesting question is
#   whether the FHMM's longer, latent-state episodes change it.
# Bounce caveat inherited: gain_x1 (excluding the first post-end day)
# reported alongside.
# ============================================================================
import numpy as np
import polars as pl

DRIVE = VALIDATION_DATA
rng = np.random.default_rng(16)
GRID = np.round(np.arange(0.15, 0.71, 0.05), 2)

# ---- hazard preds (k=1) ---------------------------------------------------------
H = (pl.read_parquet(DRIVE / "fhmm_hazard_preds.parquet")
       .filter(pl.col("k") == 1)
       .with_columns(pl.col("TR_DATE").cast(pl.Date))
       .select("cisin", "TR_DATE", "era", "year", "p_model", "p_km"))

# ---- rebuild the same causal SD runs (as 17F) -----------------------------------
fs = (pl.read_parquet(DRIVE / "fhmm_filtered_states.parquet")
        .select("cisin", "TR_DATE", "farch").sort(["cisin", "TR_DATE"]))
fs = fs.with_columns(
    ((pl.col("farch") != pl.col("farch").shift(1)).fill_null(True)
     | ((pl.col("TR_DATE") - pl.col("TR_DATE").shift(1))
        .dt.total_days() > 21))
    .cum_sum().over("cisin").alias("_r"))
sd = fs.filter(pl.col("farch") == "SHARK_DIST")
sd = sd.with_columns(
    pl.col("TR_DATE").last().over("cisin", "_r").alias("run_end"),
    pl.col("TR_DATE").last().over("cisin").alias("last_day"))
sd = sd.filter(pl.col("run_end") != pl.col("last_day"))  # uncensored
sd = sd.join(H, on=["cisin", "TR_DATE"], how="inner")

# ---- market-adjusted cumulative returns -----------------------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "ret_adj_mktadj").sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
p = p.with_columns(
    pl.col("cum").shift(-1).over("isin").alias("cum_n1"),
    pl.col("cum").shift(-2).over("isin").alias("cum_n2"))

def episode_table(theta_col, theta):
    t = (sd.filter(pl.col(theta_col) > theta)
           .group_by("cisin", "_r")
           .agg(pl.col("TR_DATE").min().alias("ta"),
                pl.col("run_end").first().alias("ed"),
                pl.col("era").first(), pl.col("year").first()))
    t = t.join(p.select("isin", "date", pl.col("cum").alias("cum_a")),
               left_on=["cisin", "ta"], right_on=["isin", "date"],
               how="inner")
    t = t.join(p.select("isin", "date",
                        pl.col("cum_n1").alias("cum_e1"),
                        pl.col("cum_n2").alias("cum_e2"),
                        pl.col("cum").alias("cum_e0")),
               left_on=["cisin", "ed"], right_on=["isin", "date"],
               how="inner")
    t = t.with_columns(
        ((pl.coalesce(pl.col("cum_e1"), pl.col("cum_e0"))
          - pl.col("cum_a")) * 1e4).alias("gain"),
        ((pl.col("cum_e0") - pl.col("cum_a")) * 1e4).alias("gain_x1"))
    return t.drop_nulls("gain")

def boot_ci(t, col="gain", n=1000):
    dts = t["ed"].to_numpy()
    v = t[col].to_numpy()
    uniq, inv = np.unique(dts, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv).astype(float)
    k = len(uniq)
    means = np.empty(n)
    for i in range(n):
        take = rng.integers(0, k, k)
        means[i] = sums[take].sum() / cnts[take].sum()
    return float(v.mean()), *np.percentile(means, [2.5, 97.5])

# ---- walk-forward theta selection --------------------------------------------------
print("=" * 70)
print("theta selection (expanding, uses only prior years' OOW preds)")
print("=" * 70)
rows_m, rows_k = [], []
for y in range(2016, 2026):
    for tag, col, store in (("model", "p_model", rows_m),
                            ("km", "p_km", rows_k)):
        best_th, best_g = None, -1e9
        for th in GRID:
            hist = episode_table(col, th).filter(
                pl.col("ed") < pl.date(y, 1, 1))
            if hist.height < 200:
                continue
            g = float(hist["gain"].mean())
            if g > best_g:
                best_g, best_th = g, th
        if best_th is None:
            continue
        ev = episode_table(col, best_th).filter(
            (pl.col("ed") >= pl.date(y, 1, 1))
            & (pl.col("ed") < pl.date(y + 1, 1, 1)))
        if ev.height:
            store.append(ev.with_columns(pl.lit(best_th).alias("theta")))
        if tag == "model":
            print(f"  {y}: theta_model={best_th:.2f} "
                  f"(hist gain {best_g:+.0f}bp) -> {ev.height} episodes")
M = pl.concat(rows_m)
K = pl.concat(rows_k)

print("\n=== anticipation gain (bp/episode, OOW 2016+) ===")
res = {}
for e in ("TRAIN", "TEST"):
    m = M.filter(pl.col("era") == e)
    kk = K.filter(pl.col("era") == e)
    gm, lo, hi = boot_ci(m)
    gx, lox, hix = boot_ci(m, "gain_x1")
    gk, lok, hik = boot_ci(kk)
    res[e] = (gm, lo, hi)
    print(f" {e}: MODEL n={m.height}  gain {gm:+.0f} [{lo:+.0f},{hi:+.0f}]"
          f"  | excl. first post-end day {gx:+.0f} [{lox:+.0f},{hix:+.0f}]")
    print(f"        KM    n={kk.height}  gain {gk:+.0f} "
          f"[{lok:+.0f},{hik:+.0f}]")
    j = m.select("cisin", "_r", pl.col("gain").alias("g_m")).join(
        kk.select("cisin", "_r", pl.col("gain").alias("g_k")),
        on=["cisin", "_r"], how="inner")
    if j.height > 30:
        d = (j["g_m"] - j["g_k"]).to_numpy()
        t = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
        res[e + "_t"] = t
        print(f"        paired model-KM on {j.height} common episodes: "
              f"d={d.mean():+.0f}bp, t={t:+.2f}")

print("\n" + "=" * 70)
gm, lo, hi = res["TEST"]
tmk = res.get("TEST_t", 0.0)
ok = lo > 0 and tmk >= 2
print("VERDICT:", "17G PASS — FHMM anticipation converts hazard skill "
      "into bp/episode over confirmation, cost-neutral, and beats the "
      "age-only anticipator (which 16D's naive version could not)."
      if ok else
      "17G: FHMM anticipation does NOT clear the pre-registered bar "
      f"(TEST gain {gm:+.0f} CI[{lo:+.0f},{hi:+.0f}], model-vs-KM "
      f"t={tmk:+.2f}) — same shape as 16D unless stated otherwise; "
      "reported as such.")
print("=" * 70)
