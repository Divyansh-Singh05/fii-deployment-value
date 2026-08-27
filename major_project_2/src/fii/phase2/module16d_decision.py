# [Phase II — charter: docs/PHASE2_PLAN.md]
from fii.paths import VALIDATION_DATA  # noqa: E402
# ============================================================================
# MODULE 16D · DECISION LAYER — does hazard skill convert to basis
# points? Anticipation vs confirmation, paired within episode.
#
# Phase-I entry rule (backtests, 12/13): CONFIRM — the end is knowable
# at close of day end+1 (the first day the label changes); entry then,
# earning from end+2. The hazard model permits ANTICIPATE: enter at
# close of the first in-episode day t_a where p_model(k=1) > theta,
# earning from t_a+1. Both books hold through the same horizon, so the
# anticipation GAIN is exactly the segment the early entry adds:
#     gain = CAR(t_a+1 .. end+1)   (paired within episode)
# — which nets the risk of remaining episode decline against capturing
# the first reversal days. Trade count is identical (one entry either
# way) -> the comparison is COST-NEUTRAL by construction.
#
# THETA (charter: learned in-window, frozen): for eval year Y, theta_Y
# = argmax over grid {0.15..0.70 step .05} of mean gain across episodes
# ENDING before Y, using only their own out-of-window hazard preds
# (expanding, causal, no refit). First decision year: 2016 (needs the
# 2014-15 preds as selection history). Same rule builds a KM-based
# anticipator (theta on p_km) — the "age alone could do it" control.
#
# PRE-REGISTERED VERDICT (TEST era 2021-07..2025-03 pooled):
#   PASS = model-anticipation mean gain > 0 with date-clustered
#   bootstrap 95% CI > 0, AND model gain exceeds KM-anticipation gain
#   (paired on common episodes, t >= 2).
#   Else: anticipation adds nothing over confirmation -> Phase II
#   concludes "forecastable but not monetizable"; report as such.
# Bounce caveat inherited: day end+1 includes the spread-concession
# component (15-T1); gain also reported excluding the first post-end
# day (gain_x1) so the reader sees both.
# ============================================================================
import numpy as np
import polars as pl

DRIVE = VALIDATION_DATA
rng = np.random.default_rng(16)
GRID = np.round(np.arange(0.15, 0.71, 0.05), 2)

# ---- hazard preds (k=1) --------------------------------------------------------
H = (pl.read_parquet(DRIVE / "phase2_hazard_preds.parquet")
       .filter(pl.col("k") == 1)
       .with_columns(pl.col("TR_DATE").cast(pl.Date))  # pandas roundtrip
       .select("cisin", "TR_DATE", "era", "year", "p_model", "p_km"))

# ---- rebuild the same causal SD runs (as 16C) ---------------------------------
fs = (pl.read_parquet(DRIVE / "phase2_filtered_states.parquet")
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

# ---- market-adjusted cumulative returns ---------------------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "ret_adj_mktadj").sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
p = p.with_columns(
    pl.col("cum").shift(-1).over("isin").alias("cum_n1"),
    pl.col("cum").shift(-2).over("isin").alias("cum_n2"))

def episode_table(theta_col, theta):
    """first trigger day per run at given theta; gain segments."""
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
    # gain = CAR(ta+1 .. end+1); gain_x1 excludes the first post-end day
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

# ---- walk-forward theta selection ----------------------------------------------
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
    # paired model-vs-KM on common episodes
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
print("VERDICT:", "16D PASS — anticipation converts hazard skill into "
      "bp/episode over confirmation, cost-neutral, and beats the "
      "age-only anticipator." if ok else
      "16D: anticipation does NOT clear the pre-registered bar "
      f"(TEST gain {gm:+.0f} CI[{lo:+.0f},{hi:+.0f}], model-vs-KM "
      f"t={tmk:+.2f}) — forecastable but not (yet) monetizable over "
      "confirmation; reported as such.")
print("=" * 70)
