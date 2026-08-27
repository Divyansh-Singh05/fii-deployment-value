from fii.paths import ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 17B · FACTORIAL HMM — DESCRIPTIVES + AGREEMENT WITH THE NAIVE HMM
#
# One claim: characterize WHAT the factorial model learned, and HOW its
# end-to-end archetypes relate to the naive HMM + frozen-threshold
# labels (stockday_states_calibrated.parquet).  No economics here —
# that is 17C.  Mirrors the 3C/13A-D1/D2 diagnostics so the comparison
# columns line up with the published ones.
#
# PRE-REGISTERED CHECKS (descriptive gates; PASS/FAIL printed):
#  B1 JOIN INTEGRITY : >= 99% of FHMM stock-days join 1:1 to the
#     calibrated naive states on (cisin, TR_DATE) — same feature store,
#     same eligibility, so the universes must coincide.
#  B2 BACKBONE AGREEMENT : direction chain vs naive backbone kappa
#     >= 0.60 in both eras (both claim the same SELL/NEUTRAL/BUY axis;
#     if they disagree wildly, one of them is not measuring direction).
#  B3 EPISODE STRUCTURE : mean archetype run lengths within 2x of the
#     naive HMM's (regime persistence must be comparable for the
#     episode-anchored validation battery to be meaningful on these
#     labels).
# ============================================================================
import numpy as np
import polars as pl

IN_FHMM  = str(ISIN_MAPPING / "stockday_states_fhmm.parquet")
IN_NAIVE = str(ISIN_MAPPING / "stockday_states_calibrated.parquet")

fh = pl.read_parquet(IN_FHMM).sort(["cisin", "TR_DATE"])
nv = (pl.read_parquet(IN_NAIVE)
        .select("cisin", "TR_DATE",
                pl.col("state").alias("n_state"),
                pl.col("archetype").alias("n_arch"))
        .sort(["cisin", "TR_DATE"]))
print(f"FHMM states : {fh.shape}")
print(f"naive states: {nv.shape}")

# ---- B1: join integrity ------------------------------------------------------
j = fh.join(nv, on=["cisin", "TR_DATE"], how="inner")
rate = j.height / fh.height
b1 = rate >= 0.99
print(f"\nB1 JOIN INTEGRITY: {j.height:,}/{fh.height:,} rows "
      f"({100*rate:.2f}%) -> {'PASS' if b1 else 'FAIL'}")

# ---- chain descriptives --------------------------------------------------------
print("\n=== chain D census by era ===")
print(j.group_by("era", "state").agg(pl.len().alias("n"))
      .with_columns((pl.col("n") / pl.col("n").sum().over("era"))
                    .round(4).alias("share")).sort(["state", "era"]))
print("\n=== chain C census by era ===")
print(j.group_by("era", "cstate").agg(pl.len().alias("n"))
      .with_columns((pl.col("n") / pl.col("n").sum().over("era"))
                    .round(4).alias("share")).sort(["cstate", "era"]))

print("\n=== empirical chain transitions (decoded), per era ===")
for col in ("state", "cstate"):
    for era in ("TRAIN", "TEST"):
        e = j.filter(pl.col("era") == era).sort(["cisin", "TR_DATE"])
        t = (e.with_columns(pl.col(col).shift(-1).over("cisin")
                            .alias("nxt"))
              .drop_nulls("nxt").group_by(col, "nxt")
              .agg(pl.len().alias("n"))
              .with_columns((pl.col("n") / pl.col("n").sum().over(col))
                            .round(3).alias("p"))
              .pivot(values="p", index=col, on="nxt").sort(col))
        print(f"\n{col} | {era}:")
        print(t)

print("\n=== joint direction x concentration occupancy (TRAIN) ===")
print(j.filter(pl.col("era") == "TRAIN")
      .group_by("state", "cstate").agg(pl.len().alias("n"))
      .with_columns((pl.col("n") / pl.col("n").sum()).round(4)
                    .alias("share"))
      .pivot(values="share", index="state", on="cstate").sort("state"))

# ---- B2: backbone agreement ----------------------------------------------------
def kappa(df, a, b, cats):
    po = float((df[a] == df[b]).mean())
    pe = sum(float((df[a] == c).mean()) * float((df[b] == c).mean())
             for c in cats)
    return po, (po - pe) / (1 - pe)

print("\n=== B2 backbone agreement: FHMM chain D vs naive backbone ===")
b2 = True
for era in ("TRAIN", "TEST"):
    e = j.filter(pl.col("era") == era)
    po, k = kappa(e, "state", "n_state",
                  ["SELL_REGIME", "NEUTRAL", "BUY_REGIME"])
    ok = k >= 0.60
    b2 = b2 and ok
    print(f"  {era:5s} agreement {100*po:.1f}%  kappa {k:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")

# ---- archetype agreement (reported, not gated: labels NEED not agree —
#      the FHMM tags concentration by latent state, not by threshold) ----------
CATS = ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT",
        "UNTAGGED_DIRECTIONAL"]
print("\n=== archetype agreement: FHMM (end-to-end) vs naive (threshold) ===")
po, k = kappa(j, "archetype", "n_arch", CATS)
print(f"overall agreement {100*po:.1f}%  Cohen's kappa {k:.3f}")
for c in ("SHARK_DIST", "SHARK_ACC", "HOSTAGE", "ROBOT"):
    hf = j.filter(pl.col("archetype") == c).height
    hn = j.filter(pl.col("n_arch") == c).height
    ov = j.filter((pl.col("archetype") == c)
                  & (pl.col("n_arch") == c)).height
    print(f"  {c:11s} FHMM n={hf:7,d}  naive n={hn:7,d}  "
          f"overlap {ov:7,d} ({100*ov/max(hn,1):.0f}% of naive)")

# ---- B3: episode structure -------------------------------------------------------
def mean_run(df, col, val):
    d = df.sort(["cisin", "TR_DATE"]).with_columns(
        ((pl.col(col) != pl.col(col).shift(1)).fill_null(True))
        .cum_sum().over("cisin").alias("_r"))
    runs = d.filter(pl.col(col) == val).group_by("cisin", "_r").len()
    return float(runs["len"].mean()) if runs.height else float("nan")

print("\n=== B3 episode structure (mean run, days) ===")
b3 = True
for c in ("SHARK_DIST", "SHARK_ACC", "HOSTAGE", "ROBOT"):
    rf = mean_run(j, "archetype", c)
    rn = mean_run(j, "n_arch", c)
    ok = (np.isfinite(rf) and np.isfinite(rn)
          and 0.5 <= rf / rn <= 2.0)
    b3 = b3 and ok
    print(f"  {c:11s} FHMM {rf:6.2f} | naive {rn:6.2f} "
          f"-> {'PASS' if ok else 'FAIL'}")

print("\n" + "=" * 66)
print(f"17B GATES: B1 {'PASS' if b1 else 'FAIL'} | "
      f"B2 {'PASS' if b2 else 'FAIL'} | B3 {'PASS' if b3 else 'FAIL'}")
print("=" * 66)
print("Next: module17c_fhmm_table1.py")
