# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 2 · v4 — FINAL: DIRECTIONAL HMM BACKBONE + ARCHETYPE OVERLAY RULES
#
# Continuation cell: run AFTER v1/v2/v3 in the same Colab session (reuses
# `panel`, `cc2`, `FEATS2`, `sweep` from v3 in memory).
#
# Architecture decision (the pre-registered fork, taken after v3 proved the
# Hostage is not a temporal regime): use the HMM for what it CAN see — the
# directional flow regime (sell/neutral/buy) — and overlay rules on the
# smoothed entity coordinates for what is episodic (concentration/dispersion).
# This is arguably closer to the economics anyway: fire-sales are EPISODES
# within selling regimes, not regimes themselves.
#
# Validated by: episode-clustering (2.7x the i.i.d. noise baseline) and
# face-validity against known market events (taper tantrum, commodity crash,
# demonetization, RCom/HDIL pre-distress). See FII_Module2_hmm_log.md §5-6.
#
# Output: stockday_states_final.parquet — the current model output.
# ============================================================================
import numpy as np, polars as pl
from hmmlearn.hmm import GaussianHMM
from pathlib import Path

parquet_path = ISIN_MAPPING
OUT_FINAL = str(parquet_path / "stockday_states_final.parquet")
TH = 0.5                                   # overlay threshold (probit units)

model3 = sweep[3][0]                       # best k=3 from the v3 sweep
means  = model3.means_
sell_s = int(np.argsort(means[:, 0])[0]); buy_s = int(np.argsort(means[:, 0])[2])
neut_s = int(np.argsort(means[:, 0])[1])
sname  = {sell_s: "SELL_REGIME", neut_s: "NEUTRAL", buy_s: "BUY_REGIME"}

Xa = cc2.select(FEATS2).to_numpy()
La = cc2.group_by("cisin", maintain_order=True).agg(pl.len())["len"].to_list()
st = model3.predict(Xa, La)

final = cc2.with_columns(
    pl.Series("state", [sname[s] for s in st])
).with_columns(
    pl.when((pl.col("state") == "SELL_REGIME") & (pl.col("F_entity_s") < -TH))
      .then(pl.lit("HOSTAGE"))
     .when((pl.col("state") == "SELL_REGIME") & (pl.col("F_entity_s") > TH))
      .then(pl.lit("SHARK_DIST"))
     .when((pl.col("state") == "BUY_REGIME") & (pl.col("F_entity_buy_s") > TH))
      .then(pl.lit("SHARK_ACC"))
     .when(pl.col("state") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL"))
      .alias("archetype")
)

print("=== ARCHETYPE CENSUS ===")
print(final.group_by("state", "archetype").agg(pl.len().alias("n"))
           .with_columns((pl.col("n") / final.height).alias("share"))
           .sort(["state", "n"], descending=[False, True]))

print("\n=== HOSTAGE EPISODE STRUCTURE (consecutive hostage days per stock) ===")
eps = (final.with_columns((pl.col("archetype") == "HOSTAGE").alias("h"))
            .with_columns(((pl.col("h") != pl.col("h").shift(1)).fill_null(True))
                          .cum_sum().over("cisin").alias("run"))
            .filter(pl.col("h"))
            .group_by("cisin", "run").agg(pl.len().alias("ep_len"),
                                          pl.col("TR_DATE").min().alias("start")))
print(eps.select(pl.len().alias("n_episodes"),
                 pl.col("ep_len").median().alias("med_len"),
                 pl.col("ep_len").quantile(0.9).alias("p90_len"),
                 pl.col("ep_len").max().alias("max_len")))

print("\n=== ARCHETYPE SHARE BY YEAR (2024-25 coverage-confounded) ===")
print(final.with_columns(pl.col("TR_DATE").dt.year().alias("yr"))
           .group_by("yr", "archetype").agg(pl.len().alias("n"))
           .with_columns((pl.col("n") / pl.col("n").sum().over("yr")).alias("s"))
           .pivot(values="s", index="yr", on="archetype").sort("yr"))

final.write_parquet(OUT_FINAL)
print(f"\nSaved → {OUT_FINAL}")

# ── FACE-VALIDITY CHECK — top Hostage episodes vs known market events ──────
names = (panel.group_by("cisin").agg(pl.col("name").drop_nulls().last().alias("name"))
         if "name" in panel.columns else None)

top = (final.with_columns((pl.col("archetype") == "HOSTAGE").alias("h"))
        .with_columns(((pl.col("h") != pl.col("h").shift(1)).fill_null(True))
                      .cum_sum().over("cisin").alias("run"))
        .filter(pl.col("h"))
        .group_by("cisin", "run")
        .agg(pl.len().alias("ep_len"),
             pl.col("TR_DATE").min().alias("start"),
             pl.col("TR_DATE").max().alias("end"),
             pl.col("F_entity_s").mean().alias("avg_dispersion"),
             pl.col("F_persist").mean().alias("avg_persist"))
        .sort("ep_len", descending=True)
        .head(25))
top = top.join(names, on="cisin", how="left") if names is not None else top
print("\n=== TOP 25 HOSTAGE EPISODES ===")
print(top.select([c for c in
      ["cisin","name","start","end","ep_len","avg_dispersion","avg_persist"]
      if c in top.columns]))
# ISINs were then mapped by hand against NSE EQUITY_L.csv (SYMBOL, NAME OF
# COMPANY, ISIN NUMBER) since `name` wasn't populated in the panel.

# RESULT (as run):
#   CENSUS: ROBOT 42.4% | SHARK_DIST 11.4% | SHARK_ACC 9.7% | HOSTAGE 6.9%
#           | UNTAGGED_DIRECTIONAL 29.6%   (776,068 decoded stock-days)
#   EPISODES: 55,530 Hostage days → 15,495 episodes, mean 3.6d, median 2,
#           p90 8d, max 53d. i.i.d. null (23.7% of sell-regime days tagged)
#           would give mean run 1.31d, median 1 → observed clustering ≈2.7x
#           the noise baseline. PASS.
#   BY YEAR: Hostage share drifts 7.7% (2011) → 5.3% (2025), tracking the
#           ID-missingness rise — NOT interpreted as economic (coverage
#           confound, standing caveat).
#   FACE VALIDITY (top episodes mapped via EQUITY_L.csv):
#     53d JSW Steel        2013-05-29→08-13  starts 7d after taper-tantrum speech
#     42d Vedanta           2015-10→12        global commodity crash / Glencore panic
#     40d HDIL               2014-09→11        leveraged-realty distress (later PMC-fraud)
#     39d Reliance Comm.     2016-07→09        debt spiral into Jio launch; later insolvent
#     38d Vedanta(Sesa)      2013-01→03        iron-ore bans + CAD crisis
#     27d Ambuja Cements     2016-12→2017-01   demonetization (cash-exposed sector)
#     27d Embassy Dev.       2014-09→11        same window as HDIL — sector-wide realty
#   Two of the top four names (RCom, HDIL) subsequently went bankrupt/distressed.
#   → Module 2 COMPLETE. Next: Module 3/4 forward-return + INNOV validation.
