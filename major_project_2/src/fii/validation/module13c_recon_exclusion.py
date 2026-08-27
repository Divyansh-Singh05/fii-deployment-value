# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 13C · PASSIVE-MECHANICS CONFOUND — could the reversal be index/
# ETF rebalancing rather than institutional liquidity demand?
# (referee objection #3, the testable half)
#
# Instrument: index_recon_dates.csv — 150 MSCI/FTSE review dates +
# approximate NIFTY reconstitution dates, 2011-2025. The file is
# MARKET-WIDE (no symbols), so the test is date-window exclusion:
# drop every episode whose END falls within +/-W calendar days of ANY
# recon/review date, then re-run the Table-1 R2 regression on the
# surviving "clean" episodes.
#
# PRE-REGISTERED (before results):
#   PRIMARY window W = 7 calendar days (passive funds concentrate
#   trading in the few days around effectiveness); SECONDARY W = 3.
#   VERDICT "PASSIVE MECHANICS EXCLUDED" if, on the W=7 clean subset,
#   D_SHARK_DIST keeps |t| >= 2 in BOTH eras with coefficient within
#   50% of the full-sample value. Retention % printed — power loss is
#   expected and reported, not hidden.
#   If the clean-subset effect DISAPPEARS -> the reversal is
#   substantially an index-rebalancing artifact; report as such.
#
# NOT covered (stated, goes to limitations): stock-level recon
# membership (file has no symbols); ESG/mandate exits (untestable
# here; handled by transitory/permanent language, not causal claims).
# ============================================================================
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA

rec = pd.read_csv(DRIVE / "index_recon_dates.csv")
rec["date"] = pd.to_datetime(rec["date"])
print("recon events:", len(rec), "| by type:")
print(rec["event"].value_counts().to_string())
rdates = rec["date"].dt.date.unique()

# ---- panel characteristics (module7b construction) ---------------------------
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
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"),
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                 pl.col("cum").last().over("isin"))
     - pl.col("cum")).alias("post20"))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
     * 1e9 + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anch = p.select("isin", "date", "pre20", "post20", "vol20", "relvol",
                "logclose", "beta120", "mom", "amihud", "logto")

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
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a in ("HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"):
    ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                         .alias("D_" + a))

# ---- flag episodes near recon dates ------------------------------------------
ed = ev["ed"].to_numpy().astype("datetime64[D]")
rd = np.array(sorted(rdates), dtype="datetime64[D]")
idx = np.searchsorted(rd, ed)
idx_lo = np.clip(idx - 1, 0, len(rd) - 1)
idx_hi = np.clip(idx, 0, len(rd) - 1)
dist = np.minimum(np.abs((ed - rd[idx_lo]).astype(int)),
                  np.abs((rd[idx_hi] - ed).astype(int)))
ev = ev.with_columns(pl.Series("recon_dist", dist))

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]

def run_r2(sub, era, tag):
    need = ["y20", "cisin", "ed", "month"] + DUM + CTL
    d = (sub.filter(pl.col("era") == era).drop_nulls(need)
            .select(need).to_pandas().set_index(["cisin", "ed"]))
    res = PanelOLS(d["y20"], d[DUM + CTL], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=d[["month"]])
    print(f"\n {tag} {era} (n={res.nobs})")
    out = {}
    for v in DUM:
        pv = float(res.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        out[v] = (float(res.params[v]), float(res.tstats[v]))
        print(f"   {v:14s}{out[v][0]:+9.1f}  t={out[v][1]:+6.2f} {s}")
    return out

print("\n=== full sample (reference) vs recon-clean subsets ===")
res = {}
for era in ("TRAIN", "TEST"):
    res[("ALL", era)] = run_r2(ev, era, "ALL episodes  ")
for w in (7, 3):
    clean = ev.filter(pl.col("recon_dist") > w)
    keep = clean.height / ev.height
    sd = clean.filter(pl.col("arch") == "SHARK_DIST").height
    sd0 = ev.filter(pl.col("arch") == "SHARK_DIST").height
    print(f"\n--- window +/-{w}cd: retained {clean.height}/{ev.height}"
          f" episodes ({100*keep:.0f}%), SHARK_DIST {sd}/{sd0} ---")
    for era in ("TRAIN", "TEST"):
        res[(w, era)] = run_r2(clean, era, f"CLEAN w={w}    ")

print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICT (primary window = 7 calendar days)")
ok = True
for era in ("TRAIN", "TEST"):
    ca, ta = res[("ALL", era)]["D_SHARK_DIST"]
    cc, tc = res[(7, era)]["D_SHARK_DIST"]
    keep = abs(tc) >= 2 and abs(cc - ca) <= .5 * abs(ca)
    ok = ok and keep
    print(f"  {era:5s} SHARK_DIST all {ca:+.1f}(t{ta:+.2f}) -> "
          f"clean {cc:+.1f}(t{tc:+.2f}) "
          f"{'HOLDS' if keep else 'DEGRADED'}")
print("\nVERDICT:", "PASSIVE MECHANICS EXCLUDED — reversal is not an"
      " index-rebalancing artifact" if ok else
      "NOT EXCLUDED at this power — report degradation honestly")
print("=" * 70)
