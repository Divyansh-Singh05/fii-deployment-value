# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 3C · DESCRIPTIVE STATISTICS & HYPOTHESIS-TESTING BATTERY
#
# Design rule: with n in the hundreds of thousands, EVERYTHING is
# "significant" — a p-value alone is decoration. Every test here reports the
# EFFECT SIZE first (Cohen's d, KS distance D, Cliff-style ratios), p second.
#
# Battery:
#   1. Feature contrasts between archetypes (Cohen's d + KS D + Mann-Whitney)
#      — headline contrast: HOSTAGE vs SHARK_DIST (both persistent-sell; the
#      entity axis is the ONLY thing separating them → d should be large on
#      F_entity_s and modest elsewhere).
#   2. Trade-size distribution tests across archetypes (KS on blockiness),
#      per user request — the classic paper table.
#   3. Permutation test for Hostage episode clustering (shuffle tags within
#      each stock's sell-regime days) — upgrades the informal "2.7x" ratio
#      to an exact p with a proper null band. Run per era: TRAIN and TEST
#      separately (OOS replication of the clustering effect).
#   4. Dwell/run-length summary per state per era.
#
# Input: stockday_states_calibrated.parquet  (from 3B)
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats

parquet_path = ISIN_MAPPING
IN_CALIB = str(parquet_path / "stockday_states_calibrated.parquet")
FEATS    = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]
N_PERM   = 200
SEED     = 42
rng = np.random.default_rng(SEED)

both = pl.read_parquet(IN_CALIB).sort(["cisin", "TR_DATE"])

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / sp

def contrast(df, arch_a, arch_b, feats):
    print(f"\n=== {arch_a} vs {arch_b} ===")
    A = df.filter(pl.col("archetype") == arch_a)
    B = df.filter(pl.col("archetype") == arch_b)
    print(f"n = {A.height:,} vs {B.height:,}")
    print(f"{'feature':<16}{'Cohen d':>10}{'KS D':>8}{'MW p':>12}   read")
    for f in feats:
        a, b = A[f].drop_nulls().to_numpy(), B[f].drop_nulls().to_numpy()
        d  = cohens_d(a, b)
        ks = stats.ks_2samp(a, b)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        size = ("large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5
                else "small" if abs(d) >= 0.2 else "negligible")
        print(f"{f:<16}{d:>+10.2f}{ks.statistic:>8.3f}{mw.pvalue:>12.1e}   {size}")
    print("(huge n → p is always tiny; judge by d and KS D)")

# ── 1+2. archetype contrasts, per era ──────────────────────────────────────
for era in ("TRAIN", "TEST"):
    e = both.filter(pl.col("era") == era)
    print(f"\n{'█'*20}  ERA: {era}  {'█'*20}")
    # headline: the two persistent-sell archetypes — entity axis is the only divider
    contrast(e, "HOSTAGE", "SHARK_DIST", FEATS)
    # sharks vs robots: blockiness + concentration should separate
    contrast(e, "SHARK_ACC", "ROBOT", FEATS)
    # hostage vs robot: persistence + dispersion
    contrast(e, "HOSTAGE", "ROBOT", FEATS)

# ── 3. permutation test: Hostage episode clustering, per era ───────────────
print(f"\n{'█'*20}  EPISODE-CLUSTERING PERMUTATION TEST  {'█'*20}")
for era in ("TRAIN", "TEST"):
    e = (both.filter(pl.col("era") == era).sort(["cisin", "TR_DATE"])
             .select("cisin", (pl.col("state") == "SELL_REGIME").alias("sell"),
                     (pl.col("archetype") == "HOSTAGE").alias("h")))
    cis  = e["cisin"].to_numpy(); sell = e["sell"].to_numpy(); hh = e["h"].to_numpy()
    new_stock = np.r_[True, cis[1:] != cis[:-1]]
    def mean_run(hv):
        starts = hv & (np.r_[True, ~hv[:-1]] | new_stock)
        ns = starts.sum()
        return hv.sum() / ns if ns else 0.0
    obs = mean_run(hh)
    idx_bounds = np.flatnonzero(new_stock).tolist() + [len(cis)]
    sell_slices = [np.flatnonzero(sell[a:b]) + a for a, b in zip(idx_bounds[:-1], idx_bounds[1:])]
    null = np.empty(N_PERM)
    for p_ in range(N_PERM):
        hp = hh.copy()
        for sl in sell_slices:
            if sl.size: hp[sl] = hh[sl][rng.permutation(sl.size)]
        null[p_] = mean_run(hp)
    pval = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    print(f"  {era}: observed mean run = {obs:.2f}d | null = {null.mean():.2f} ± {null.std():.2f}d "
          f"| ratio = {obs/null.mean():.2f}x | perm p = {pval:.4f}  "
          f"({'clustering real' if pval < 0.05 else 'NOT significant'})")

# ── 4. run-length summary per archetype per era ────────────────────────────
print(f"\n{'█'*20}  EPISODE LENGTHS BY ARCHETYPE  {'█'*20}")
runs = (both.with_columns(
            ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
            .cum_sum().over("cisin").alias("_run"))
        .group_by("era", "cisin", "_run")
        .agg(pl.col("archetype").first(), pl.len().alias("len")))
print(runs.group_by("era", "archetype")
          .agg(pl.len().alias("episodes"),
               pl.col("len").mean().round(2).alias("mean_len"),
               pl.col("len").median().alias("med_len"),
               pl.col("len").quantile(0.9).alias("p90_len"),
               pl.col("len").max().alias("max_len"))
          .sort(["archetype", "era"]))
print("\nDone. Read: (1) do TRAIN effect sizes replicate in TEST? "
      "(2) is HOSTAGE vs SHARK_DIST separation carried by F_entity_s as designed? "
      "(3) does episode clustering survive OOS?")
