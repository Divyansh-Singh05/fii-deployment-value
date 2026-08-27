# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 14B · TAIL LABELS — rule backbone + quantile overlays on the
# measurable illiquid tail (1,618 names; 14A gates PASSED)
#
# METHOD FREEZE (not number freeze): the 13A-validated procedure is
# re-derived on TAIL TRAIN only, frozen, applied to TAIL TEST:
#  - within-TAIL-day probit ranks (days with >=30 ranked names)
#  - backbone: persistence-rank cuts census-matched to the MODEL
#    backbone's TRAIN shares (sell 29.1% / buy 28.0%)
#  - overlays: q25/q75 of smoothed entity-HHI rank within rule-SELL
#    days (HOSTAGE / SHARK_DIST); q75 buy-side within rule-BUY
#    (SHARK_ACC — expected under-powered: buy attribution 24%)
#  - HHI smoothing: 5d trailing mean of daily snapshots (audit-legal),
#    re-ranked; May-Jun 2021 masked
#
# PRE-REGISTERED GATES (all must PASS for 14C):
#  B1 census sanity: SHARK_DIST and HOSTAGE each 3-15% of labeled
#     tail days in BOTH eras (a degenerate cut fails this)
#  B2 episode clustering: mean run >= 1.5x within-stock-shuffle null,
#     p < 0.05, for SHARK_DIST, in BOTH eras (200 shuffles)
#  B3 power: >= 300 SHARK_DIST episode-ENDs with a price row on the
#     end date, per era
# Output: tail_states_v1.parquet (for 14C).
# SCOPE (inherited from 14A, stated once): results describe the
# ATTRIBUTABLE tail (days with >=50%-covered entity attribution);
# TEST-era dominance + 2024-25 ID-missingness confound carried over.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from scipy.special import ndtri

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
MIN_XS = 30            # min ranked names per day
SH_SELL, SH_BUY = 0.291, 0.280   # model backbone TRAIN census (frozen)

f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
st = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
tail = sorted(set(f["cisin"].unique()) - set(st["cisin"].unique()))
t = (f.filter(pl.col("cisin").is_in(tail) & (pl.col("N") >= 2))
      .select("cisin", "TR_DATE", "N", "persistence_raw",
              "entity_conc_raw", "entity_conc_buy_raw")
      .sort(["cisin", "TR_DATE"]))
t = t.filter(~pl.col("TR_DATE").is_between(pl.date(2021, 5, 1),
                                           pl.date(2021, 6, 30)))
t = t.with_columns(
    pl.when(pl.col("TR_DATE") <= pl.date(2021, 4, 30))
      .then(pl.lit("TRAIN")).otherwise(pl.lit("TEST")).alias("era"))
t = t.filter(pl.col("persistence_raw").is_not_null()
             & pl.col("entity_conc_raw").is_not_null())
print("tail core:", t.height, "days,", t["cisin"].n_unique(), "names")

# ---- 5d trailing smoothing of HHI snapshots (per stock, min 2 obs) -----------
t = t.with_columns(
    pl.col("entity_conc_raw").rolling_mean(window_size=5, min_samples=2)
      .over("cisin").alias("hhi_s"),
    pl.col("entity_conc_buy_raw").rolling_mean(window_size=5,
                                              min_samples=2)
      .over("cisin").alias("hhib_s"))

# ---- within-tail-day probit ranks --------------------------------------------
def probit_rank(df, col, out):
    return df.with_columns(
        pl.when(pl.col(col).is_not_null()
                & (pl.col(col).count().over("TR_DATE") >= MIN_XS))
          .then(((pl.col(col).rank().over("TR_DATE"))
                 / (pl.col(col).count().over("TR_DATE") + 1)))
          .otherwise(None).alias("_p"),
    ).with_columns(
        pl.col("_p").map_batches(
            lambda s: pl.Series(ndtri(s.to_numpy())), is_elementwise=True)
        .fill_nan(None)   # NaN != null in polars; NaN poisons quantiles
        .alias(out)).drop("_p")

t = probit_rank(t, "persistence_raw", "Fp")
t = probit_rank(t, "hhi_s", "Fe")
t = probit_rank(t, "hhib_s", "Feb")
t = t.filter(pl.col("Fp").is_not_null() & pl.col("Fe").is_not_null())
print("ranked core:", t.height, "days")

# ---- TRAIN-frozen cuts ---------------------------------------------------------
tr = t.filter(pl.col("era") == "TRAIN")
c_lo = float(tr["Fp"].quantile(SH_SELL))
c_hi = float(tr["Fp"].quantile(1 - SH_BUY))
# sanity guard: probit quantiles at 29%/72% must be finite and straddle 0
assert np.isfinite(c_lo) and np.isfinite(c_hi) and c_lo < 0 < c_hi, \
    f"cut sanity failed (c_lo={c_lo}, c_hi={c_hi}) — NaN/rank bug"
t = t.with_columns(
    pl.when(pl.col("Fp") < c_lo).then(pl.lit("SELL"))
     .when(pl.col("Fp") > c_hi).then(pl.lit("BUY"))
     .otherwise(pl.lit("NEUTRAL")).alias("bb"))
sell_tr = t.filter((pl.col("era") == "TRAIN") & (pl.col("bb") == "SELL"))
buy_tr = t.filter((pl.col("era") == "TRAIN") & (pl.col("bb") == "BUY"))
thr_h = float(sell_tr["Fe"].quantile(0.25))
thr_sd = float(sell_tr["Fe"].quantile(0.75))
feb_tr = buy_tr["Feb"].drop_nulls()
# buy-side attribution is thin in the tail (24%, 14A-C2); derive the
# SHARK_ACC cut only if enough TRAIN material exists, else DISABLE the
# buy overlay in the tail (pre-flagged degradation, not a failure)
thr_sa = float(feb_tr.quantile(0.75)) if feb_tr.len() >= 500 else None
print(f"frozen cuts: backbone {c_lo:+.3f}/{c_hi:+.3f} | HOSTAGE "
      f"<{thr_h:+.3f} SHARK_DIST >{thr_sd:+.3f} | SHARK_ACC "
      + (f">{thr_sa:+.3f}" if thr_sa is not None else
         f"DISABLED (TRAIN buy-days w/ Feb: {feb_tr.len()})"))

lab = (pl.when((pl.col("bb") == "SELL") & (pl.col("Fe") < thr_h))
         .then(pl.lit("HOSTAGE"))
        .when((pl.col("bb") == "SELL") & (pl.col("Fe") > thr_sd))
         .then(pl.lit("SHARK_DIST")))
if thr_sa is not None:
    lab = lab.when((pl.col("bb") == "BUY") & (pl.col("Feb") > thr_sa)) \
             .then(pl.lit("SHARK_ACC"))
lab = (lab.when(pl.col("bb") == "NEUTRAL").then(pl.lit("ROBOT"))
          .otherwise(pl.lit("UNTAGGED")).alias("archetype"))
t = t.with_columns(lab)

print("\n=== B1 · census (% of labeled tail days) ===")
cen = (t.group_by("era", "archetype").len()
        .with_columns((100 * pl.col("len")
                       / pl.col("len").sum().over("era")).round(2)
                      .alias("pct")).sort(["era", "archetype"]))
print(cen)
b1 = True
for era in ("TRAIN", "TEST"):
    for a in ("SHARK_DIST", "HOSTAGE"):
        row = cen.filter((pl.col("era") == era)
                         & (pl.col("archetype") == a))
        pc = float(row["pct"][0]) if row.height else 0.0
        if not (3.0 <= pc <= 15.0):
            b1 = False
            print(f"  B1 breach: {era} {a} = {pc}%")

# ---- B2 · episode-clustering permutation test --------------------------------
def clustering(era, arch, nshuf=200, seed=14):
    rng = np.random.default_rng(seed)
    e = t.filter(pl.col("era") == era).sort(["cisin", "TR_DATE"])
    runs_obs, runs_null = [], np.zeros(nshuf)
    seqs = []
    for (_,), g in e.group_by(["cisin"], maintain_order=True):
        v = (g["archetype"] == arch).to_numpy()
        if v.sum() == 0:
            continue
        seqs.append(v)
        d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
        runs_obs += list(np.flatnonzero(d == -1)
                         - np.flatnonzero(d == 1))
    obs = float(np.mean(runs_obs))
    for k in range(nshuf):
        tot_len, tot_runs = 0, 0
        for v in seqs:
            w = rng.permutation(v)
            d = np.diff(np.concatenate([[0], w.astype(int), [0]]))
            tot_len += w.sum()
            tot_runs += (d == 1).sum()
        runs_null[k] = tot_len / max(tot_runs, 1)
    p = (1 + (runs_null >= obs).sum()) / (1 + nshuf)
    return obs, float(runs_null.mean()), p

b2 = True
print("\n=== B2 · SHARK_DIST episode clustering (200 shuffles) ===")
for era in ("TRAIN", "TEST"):
    obs, nul, p = clustering(era, "SHARK_DIST")
    ratio = obs / nul
    ok = ratio >= 1.5 and p < 0.05
    b2 = b2 and ok
    print(f"  {era}: mean run {obs:.2f}d vs null {nul:.2f}d = "
          f"{ratio:.2f}x, p={p:.3f} {'PASS' if ok else 'FAIL'}")

# ---- B3 · powered episode-ends with price coverage ---------------------------
p3 = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
        .select("isin", "date").with_columns(pl.lit(1).alias("hasp")))
runs = t.sort(["cisin", "TR_DATE"]).with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1))
     .fill_null(True)).cum_sum().over("cisin").alias("_r"))
ep = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"))
ep = ep.join(p3, left_on=["cisin", "ed"], right_on=["isin", "date"],
             how="left")
b3 = True
print("\n=== B3 · SHARK_DIST episode-ENDs with a price row ===")
for era in ("TRAIN", "TEST"):
    n = ep.filter((pl.col("era") == era)
                  & (pl.col("archetype") == "SHARK_DIST")
                  & pl.col("hasp").is_not_null()).height
    ok = n >= 300
    b3 = b3 and ok
    print(f"  {era}: {n} {'PASS' if ok else 'FAIL'} (bar 300)")

t.select("cisin", "TR_DATE", "era", "archetype", "Fp", "Fe",
         "Feb").write_parquet(DRIVE / "tail_states_v1.parquet")
print("\nwrote tail_states_v1.parquet", t.shape)
print("\n" + "=" * 70)
print("VERDICT:", "ALL GATES PASS — proceed to 14C (economics + "
      "friction bar)" if b1 and b2 and b3 else
      "GATE FAILURE — the tail's label structure is not credible; "
      "stop and report (do not tune cuts post hoc)")
print("=" * 70)
