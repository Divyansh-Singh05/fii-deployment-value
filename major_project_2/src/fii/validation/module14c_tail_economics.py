# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 14C · TAIL ECONOMICS — the verdict: is the tail reversal real,
# and does it clear the tail's friction?
#
# Inputs: tail_states_v1.parquet (14B, gates passed) + returns_panel_v3.
#
# T1 · EVENT STUDY (END anchor, excess vs tail ALL-LABELED baseline):
#     CAR5/CAR20 from t+1, +/-50% daily clip, delisting truncation kept
#     and reported, date-clustered bootstrap (N=1000) on the DIFFERENCE.
#     Arc context: pre20, episode CAR, day-0 relative volume (mechanism:
#     concentrated episodes should be volume-marked in the tail too).
#
# T2 · FRICTION BAR (no HIGH/LOW available -> pre-registered fallback):
#     per (stock, era) Roll spread s = 2*sqrt(-cov(r_t, r_{t-1})) on
#     CA-adjusted returns where cov<0; plus zero-volume share and
#     median |ret| for context. Round-trip cost bar = median Roll s
#     across SHARK_DIST episode names (per era). Model-universe Roll
#     printed alongside for comparison.
#
# PRE-REGISTERED VERDICT (TEST era decides):
#   TAIL HARVESTABLE      excess CAR20 bootstrap 95% CI > 0 AND point
#                         estimate > friction bar
#   REAL BUT UNHARVESTABLE  CI > 0 but point <= bar
#   NO TAIL EFFECT        CI includes 0
# Also reported: tail vs liquid magnitude (the "500bp not 50bp" claim).
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
rng = np.random.default_rng(14)

st = (pl.read_parquet(DRIVE / "tail_states_v1.parquet")
        .sort(["cisin", "TR_DATE"]))
tailnames = st["cisin"].unique().to_list()

p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "ret_adj", "ret_adj_mktadj", "volume")
       .filter(pl.col("isin").is_in(tailnames))
       .sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"))
p = p.with_columns(
    pl.col("ar").cum_sum().over("isin").alias("cum"),
    pl.col("volume").rolling_mean(window_size=20).over("isin")
      .alias("_vma"))
p = p.with_columns(
    (pl.col("volume") / pl.col("_vma").shift(1).over("isin"))
    .alias("rvol"),
    (pl.col("cum").shift(1).over("isin")
     - pl.col("cum").shift(21).over("isin")).alias("pre20"),
    pl.col("date").shift(-20).over("isin").alias("_d20"))
for k in (5, 20):
    p = p.with_columns(
        (pl.coalesce(pl.col("cum").shift(-k).over("isin"),
                     pl.col("cum").last().over("isin"))
         - pl.col("cum")).alias(f"post{k}"))
anch = p.select("isin", "date", "ar", "cum", "pre20", "post5",
                "post20", "rvol", "_d20")

# ---- episodes -----------------------------------------------------------------
runs = st.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1))
     .fill_null(True)).cum_sum().over("cisin").alias("_r"))
ep = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first().alias("arch"), pl.col("era").first(),
    pl.col("TR_DATE").first().alias("sd"),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
ev = ep.join(anch, left_on=["cisin", "ed"],
             right_on=["isin", "date"], how="inner")
ev = ev.join(anch.select("isin", "date",
                         pl.col("cum").alias("cum_s"),
                         pl.col("pre20").alias("pre_s"),
                         pl.col("rvol").alias("rvol0")),
             left_on=["cisin", "sd"], right_on=["isin", "date"],
             how="left")
ev = ev.with_columns(
    ((pl.col("cum") - pl.col("cum_s")) * 1e4).alias("epcar_bp"),
    pl.col("_d20").is_null().alias("trunc"))

# tail baseline: ALL labeled tail stock-days, per era
base = st.select("cisin", "TR_DATE", "era").join(
    anch.select("isin", "date", "post5", "post20"),
    left_on=["cisin", "TR_DATE"], right_on=["isin", "date"],
    how="inner")

def boot_diff(sub, col, bvals, n=1000):
    """date-clustered bootstrap of mean(sub[col]) - mean(bvals).
    Resamples anchor DAYS with replacement (a day drawn twice counts
    twice); per-day sums/counts make each draw O(n_days)."""
    dts = sub["ed"].to_numpy()
    vals = sub[col].to_numpy() * 1e4
    bm = float(np.mean(bvals) * 1e4)
    uniq, inv = np.unique(dts, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnts = np.bincount(inv).astype(float)
    k = len(uniq)
    means = np.empty(n)
    for i in range(n):
        take = rng.integers(0, k, k)
        means[i] = sums[take].sum() / cnts[take].sum() - bm
    lo, hi = np.percentile(means, [2.5, 97.5])
    pt = vals.mean() - bm
    pv = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return pt, lo, hi, pv

print("=" * 70)
print("T1 · tail event study (END anchor, excess vs tail baseline, bp)")
print("=" * 70)
res = {}
for era in ("TRAIN", "TEST"):
    b5 = base.filter(pl.col("era") == era)["post5"].drop_nulls()
    b20 = base.filter(pl.col("era") == era)["post20"].drop_nulls()
    print(f"\n--- {era} (baseline post20 "
          f"{1e4*float(b20.mean()):+.0f}bp) ---")
    for a in ("SHARK_DIST", "HOSTAGE", "ROBOT"):
        s = ev.filter((pl.col("era") == era) & (pl.col("arch") == a)
                      & pl.col("post20").is_not_null())
        pt5, lo5, hi5, _ = boot_diff(s, "post5", b5.to_numpy())
        pt, lo, hi, pv = boot_diff(s, "post20", b20.to_numpy())
        tr = 100 * float(s["trunc"].mean())
        pre = float(s["pre_s"].drop_nulls().mean() * 1e4)
        epc = float(s["epcar_bp"].mean())
        rv = float(s["rvol0"].drop_nulls().mean())
        print(f" {a:11s} n={s.height:5d}  exc5 {pt5:+6.0f}  "
              f"exc20 {pt:+6.0f} [{lo:+.0f},{hi:+.0f}] p={pv:.3f}")
        print(f"             arc: pre20 {pre:+6.0f} ep {epc:+6.0f} "
              f"day0-rvol {rv:.2f}x  trunc {tr:.1f}%")
        if a == "SHARK_DIST":
            res[era] = (pt, lo, hi, pv)

# ---- T2 · friction bar ----------------------------------------------------------
print("\n" + "=" * 70)
print("T2 · friction (Roll spread, zero-volume, |ret|)")
print("=" * 70)
r = p.select("isin", "date", "ret_adj", "volume").drop_nulls("ret_adj")
r = r.with_columns(
    pl.col("ret_adj").shift(1).over("isin").alias("_l1"),
    pl.when(pl.col("date") <= pl.date(2021, 4, 30))
      .then(pl.lit("TRAIN"))
      .when(pl.col("date") >= pl.date(2021, 7, 1))
      .then(pl.lit("TEST")).otherwise(pl.lit("MASK")).alias("era"))
roll = (r.filter(pl.col("era") != "MASK").drop_nulls("_l1")
          .group_by("isin", "era").agg(
              ((pl.col("ret_adj") - pl.col("ret_adj").mean())
               * (pl.col("_l1") - pl.col("_l1").mean())).mean()
              .alias("acov"),
              (pl.col("ret_adj") == 0).mean().alias("zret"),
              pl.col("ret_adj").abs().median().alias("mabs"),
              pl.len().alias("nd")))
roll = roll.filter(pl.col("nd") >= 60).with_columns(
    pl.when(pl.col("acov") < 0)
      .then(2 * (-pl.col("acov")).sqrt()).otherwise(None)
      .alias("roll_s"))
sdn = (ev.filter(pl.col("arch") == "SHARK_DIST")
         .select("cisin", "era").unique())
bars = {}
for era in ("TRAIN", "TEST"):
    x = roll.join(sdn.filter(pl.col("era") == era), left_on=["isin",
                  "era"], right_on=["cisin", "era"], how="inner")
    s_med = float(x["roll_s"].drop_nulls().median())
    zr = float(x["zret"].median())
    ma = float(x["mabs"].median())
    okn = x["roll_s"].drop_nulls().len()
    bars[era] = s_med
    print(f" {era}: Roll round-trip median {1e4*s_med:.0f}bp "
          f"({okn}/{x.height} names w/ cov<0) | zero-ret {100*zr:.1f}%"
          f" | med|ret| {1e4*ma:.0f}bp")

print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICT (TEST era)")
pt, lo, hi, pv = res["TEST"]
bar = 1e4 * bars["TEST"]
print(f" SHARK_DIST excess CAR20 {pt:+.0f}bp, CI [{lo:+.0f},{hi:+.0f}],"
      f" p={pv:.3f} | friction bar {bar:.0f}bp")
if lo > 0 and pt > bar:
    v = ("TAIL HARVESTABLE — reversal clears the round-trip friction "
         "bar; the economics scale where the liquidity is thin")
elif lo > 0:
    v = ("REAL BUT UNHARVESTABLE — the reversal exists in the tail but"
         " does not clear its own friction; limits-to-arbitrage "
         "confirmed at both ends of the liquidity spectrum")
else:
    v = "NO TAIL EFFECT — the reversal does not generalize to the tail"
print("\nVERDICT:", v)
print(f" (liquid-universe comparison: TEST excess ~+33bp event-study /"
      f" +49bp regression)")
print("=" * 70)
