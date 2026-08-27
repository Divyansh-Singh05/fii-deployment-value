# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5B-STEP2 · APPLY CA ADJUSTMENT -> returns_panel_v2 + GATES
#
# Base return = close-to-close per ISIN (NOT prev_close: Gate 0 showed
# prev_close is raw; and on relist days NSE sets prev_close to a base price,
# so close-to-close across the gap is the true holding return anyway).
# On a factor ex-date: ret_adj = (1 + ret_cc) * factor - 1.
#
# Application policy (from Step-1 verdict):
#   APPLY   factor if confirmed, or unverifiable (no price row on ex-day —
#           mostly suspensions; the factor then belongs to the NEXT traded
#           day's gap return, handled by forward as-of mapping below).
#   EXCLUDE the 7 confirmed-wrong events (DRREDDY/NTPC debenture bonuses,
#           postponed ex-dates, DPSCLTD cap casualty): ex-day return NULLED,
#           never adjusted with a factor the tape contradicts.
#
# Ex-dates falling on non-trading days for that stock are mapped FORWARD to
# the stock's next traded row (join_asof by symbol) — a split during a
# suspension must adjust the relist gap return.
#
# GATES AT THE END (this step's verdict):
#   G-A  ex-day gate re-run: median |ret_adj| on confirmed ex-days must be
#        ordinary (<5%) where raw |ret_cc| was ~50%. Must flip to PASS.
#   G-B  extreme-return audit: |ret_adj|>25% days per year, vs raw counts.
# No model contact in this step. CARs come only after G-A passes.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
OUT = DRIVE / "returns_panel_v2.parquet"

# ---- load ------------------------------------------------------------------
panel = pl.read_parquet(DRIVE / "returns_panel.parquet").sort(["isin", "date"])
fac = pl.read_parquet(DRIVE / "ca_adjustment_factors.parquet")

apply_ev = fac.filter(pl.col("confirmed") | pl.col("obs_ratio").is_null())
excl_ev = fac.filter(pl.col("confirmed") == False)  # noqa: E712
print("factor events: apply =", apply_ev.height,
      "| exclude (null the day) =", excl_ev.height)

# ---- base close-to-close return per ISIN ------------------------------------
panel = panel.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("isin") - 1)
    .alias("ret_cc")
)

# ---- map each event to the stock's next traded row (forward as-of) ---------
rows = panel.select("symbol", "date").unique().sort("date")
ev = apply_ev.select("symbol", "ex_date", "factor").sort("ex_date")
mapped = ev.join_asof(rows, left_on="ex_date", right_on="date",
                      by="symbol", strategy="forward")
n_unmapped = mapped.filter(pl.col("date").is_null()).height
n_gap = mapped.filter(pl.col("date") > pl.col("ex_date")).height
print("events mapped to a traded row:", mapped.height - n_unmapped,
      "| gap-mapped (ex-date while suspended):", n_gap,
      "| unmappable (delisted before ex-date):", n_unmapped)
per_day = (mapped.filter(pl.col("date").is_not_null())
                 .group_by("symbol", "date")
                 .agg(pl.col("factor").product().alias("adj_factor")))

# ---- apply ------------------------------------------------------------------
panel = panel.join(per_day, on=["symbol", "date"], how="left")
panel = panel.with_columns(
    ((1 + pl.col("ret_cc")) * pl.col("adj_factor").fill_null(1.0) - 1)
    .alias("ret_adj")
)
n_applied = panel.filter(pl.col("adj_factor").is_not_null()).height
print("stock-days with a factor applied:", n_applied)

# APPLICATION GUARD (from 5B-2d): if a factor was applied and the adjusted
# ex-day return is STILL implausible, the event's crash is not inside this
# ISIN's chain (symbol migration around the event, e.g. UNOMINDA 2022 whose
# split happened under MINDAIND; or split-minted new ISIN starting at the
# ex-date). Applying the factor there fabricates a huge fake return -> null
# the day instead, and log every case.
GUARD = 0.50
g_bad = panel.filter(pl.col("adj_factor").is_not_null()
                     & pl.col("ret_adj").is_not_null()
                     & (pl.col("ret_adj").abs() > GUARD))
print("application guard: factor days still implausible after adjustment")
print("(|ret_adj| >", GUARD, ") -> nulled:", g_bad.height)
print(g_bad.select("symbol", "date", "close", "ret_cc",
                   "adj_factor", "ret_adj").head(12))
panel = panel.with_columns(
    pl.when(pl.col("adj_factor").is_not_null()
            & (pl.col("ret_adj").abs() > GUARD))
    .then(None).otherwise(pl.col("ret_adj"))
    .alias("ret_adj")
)

# null the excluded (confirmed-wrong) event days
panel = panel.join(
    excl_ev.select("symbol", pl.col("ex_date").alias("date"),
                   pl.lit(True).alias("_kill")),
    on=["symbol", "date"], how="left")
panel = panel.with_columns(
    pl.when(pl.col("_kill")).then(None).otherwise(pl.col("ret_adj"))
    .alias("ret_adj")
).drop("_kill")

# market-adjusted version
panel = panel.with_columns(
    (pl.col("ret_adj") - pl.col("nifty50_ret")).alias("ret_adj_mktadj")
)

# ---- GATE A: ex-day gate must flip to PASS ----------------------------------
print("")
print("=== GATE A: ex-day returns after adjustment ===")
conf = fac.filter(pl.col("confirmed") == True)  # noqa: E712
g = conf.join(panel.select("symbol", "date", "ret_cc", "ret_adj"),
              left_on=["symbol", "ex_date"], right_on=["symbol", "date"],
              how="inner")
g = g.filter(pl.col("ret_cc").is_not_null() & pl.col("ret_adj").is_not_null())
med_raw = g["ret_cc"].abs().median()
med_adj = g["ret_adj"].abs().median()
frac_ok = float((g["ret_adj"].abs() < 0.20).mean())
print("confirmed ex-days checked:", g.height)
print("median |ret_cc| (raw)     :", round(float(med_raw), 4))
print("median |ret_adj| (adjusted):", round(float(med_adj), 4))
print("share of ex-days with |ret_adj| < 20% :", round(100 * frac_ok, 1), "%")
ok = (med_adj is not None) and float(med_adj) < 0.05 and float(med_raw) > 0.15
print("GATE A:", "PASS — splits/bonuses neutralized" if ok
      else "FAIL — STOP, inspect before any CAR work")

# ---- GATE B: extreme-return audit, raw vs adjusted --------------------------
print("")
print("=== GATE B: |ret| > 25% days per year, raw vs adjusted ===")
gb = (panel.group_by(pl.col("date").dt.year().alias("yr"))
           .agg((pl.col("ret_cc").abs() > 0.25).sum().alias("raw_gt25"),
                (pl.col("ret_adj").abs() > 0.25).sum().alias("adj_gt25"))
           .sort("yr"))
print(gb)
print("read: adj counts should drop vs raw (splits removed); what remains")
print("is circuits, relists after suspension, and genuine crash days.")
print("largest surviving |ret_adj| — eyeball for leftover artifacts:")
print(panel.filter(pl.col("ret_adj").is_not_null())
           .sort(pl.col("ret_adj").abs(), descending=True)
           .select("symbol", "date", "close", "ret_cc", "adj_factor", "ret_adj")
           .head(12))

# ---- write ------------------------------------------------------------------
panel = panel.drop("ret_mktadj")  # v1 column, built on unadjusted ret
panel.write_parquet(OUT)
print("")
print("wrote", OUT.name, panel.shape)
print("NEXT (5B-3, only if GATE A passed): forward-return CARs on ret_adj_mktadj,")
print("one anchor at a time, starting with episode-START drift test.")
