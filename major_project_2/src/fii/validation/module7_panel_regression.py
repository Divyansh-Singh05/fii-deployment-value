# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 7 · PANEL REGRESSION (main specs) — via linearmodels.PanelOLS
#
# THE MODEL (one explicit, auditable call per spec):
#   PanelOLS(y, X, entity_effects=True, time_effects=True)
#       .fit(cov_type="clustered", cluster_entity=True, clusters=month)
#   y      = post-END 20d market-adjusted CAR (bp), per episode-END event
#   entity = stock (cisin)  -> stock fixed effects
#   time   = episode end DATE -> date fixed effects (stricter than month FE:
#            absorbs every daily market-wide shock)
#   SE     = two-way clustered: stock AND calendar month
#   X      = archetype dummies (UNTAGGED omitted) + controls
#
# Specs per era:
#   R0  archetypes only
#   R1  + characteristics, NO pre20  -> TOTAL predictive effect
#   R2  + pre20                      -> conservative LOWER BOUND (pre20 is
#        partly a mediator of the pressure phase = "bad control")
# Controls (past-only, internal; no fundamentals -> value/sector are stated
# paper limitations): beta120, momentum(t-126..t-21), Amihud, log turnover,
# vol20, relvol, log price, log episode length.
# LANGUAGE: "predicts conditional on controls" — NOT causal.
# ============================================================================
import polars as pl
import pandas as pd
from pathlib import Path

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA

# ---- panel + past-only characteristics --------------------------------------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret")
       .sort(["isin", "date"]))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-0.5, 0.5).fill_null(0.0).alias("ar"),
    pl.col("ret_adj").clip(-0.5, 0.5).alias("r"),
    (pl.col("close") * pl.col("volume")).alias("to"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))

W = 120  # beta window: beta = cov(r, mkt) / var(mkt), rolling, lagged
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
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                 pl.col("cum").last().over("isin"))
     - pl.col("cum")).alias("post20"),
    pl.col("ar").rolling_std(window_size=20).over("isin")
      .shift(1).alias("vol20"),
    (pl.col("ar").abs() / (pl.col("to") + 1.0)).alias("_ilq"),
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1) * 1e9
     + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anchors = p.select("isin", "date", "pre20", "post20", "vol20", "relvol",
                   "logclose", "beta120", "mom", "amihud", "logto")

# ---- episode END events ------------------------------------------------------
states = (pl.read_parquet(DRIVE / "states_v3.parquet")
            .select("cisin", "TR_DATE", "era", "archetype")
            .sort(["cisin", "TR_DATE"]))
runs = states.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
ev = runs.join(anchors, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("post20")).alias("y"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a in ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"]:
    ev = ev.with_columns((pl.col("archetype") == a).cast(pl.Float64)
                         .alias("D_" + a))

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL1 = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
        "logclose", "logeplen"]
CTL2 = CTL1 + ["pre20bp"]

def run_spec(sub, xcols, tag):
    need = ["y", "cisin", "ed", "month"] + xcols
    d = sub.drop_nulls(need).select(need).to_pandas()
    d = d.set_index(["cisin", "ed"])          # (entity, time) panel index
    mod = PanelOLS(d["y"], d[xcols],
                   entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True,
                  clusters=d[["month"]])
    print("\n " + tag + f"  (n={res.nobs}, "
          f"stocks={d.index.get_level_values(0).nunique()}, "
          f"dates={d.index.get_level_values(1).nunique()})")
    print(" " + "var".ljust(14) + "coef".rjust(9) + "se".rjust(8)
          + "t".rjust(7) + "p".rjust(8))
    for v in xcols:
        pv = float(res.pvalues[v])
        star = "***" if pv < .01 else "**" if pv < .05 else \
               "*" if pv < .10 else ""
        print(" " + v.ljust(14)
              + ("%.1f" % float(res.params[v])).rjust(9)
              + ("%.1f" % float(res.std_errors[v])).rjust(8)
              + ("%.2f" % float(res.tstats[v])).rjust(7)
              + ("%.3f" % pv).rjust(8) + " " + star)

for era in ("TRAIN", "TEST"):
    sub = ev.filter(pl.col("era") == era)
    print("\n" + "=" * 70)
    print("ERA:", era, "| PanelOLS: dep = post-END CAR20 (bp),")
    print("entity(stock) FE + time(date) FE, SE two-way clustered")
    print("(stock x month) | omitted category = UNTAGGED_DIRECTIONAL")
    print("=" * 70)
    run_spec(sub, DUM, "R0: archetypes + FE only")
    run_spec(sub, DUM + CTL1, "R1: + characteristics (TOTAL effect)")
    run_spec(sub, DUM + CTL2, "R2: + pre20 (bad-control LOWER BOUND)")

print("""
READ:
 1. Coefs = bp vs an UNTAGGED episode, same stock, same date.
 2. R1 vs R2 bracket SHARK_DIST (pre20 is partly a mediator ->
    R2 is a conservative lower bound). Survival in R2 = signal
    beyond generic loser-reversal.
 3. Expected: SHARK_DIST > 0 both eras; HOSTAGE ~ 0; ROBOT ~ 0
    (placebo); SHARK_ACC <= 0 (give-back).
 4. Language: 'predicts conditional on controls' — not causal.
 5. Robustness next (7b): non-overlap subsample, CAR10/30/60,
    continuous F_entity_s dose-response.
""")
