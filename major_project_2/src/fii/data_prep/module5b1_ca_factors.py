# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5B-STEP1 · CA ADJUSTMENT FACTORS — BUILD & VERIFY (no model contact)
#
# Gate 0 proved bhavcopy prev_close is RAW (unadjusted): on split ex-days
# |ret| == |ret_cc| == ~50%. So we build the adjustment ourselves.
# THIS STEP ONLY: parse split/bonus factors from the NSE CA file and verify
# each one against the observed ex-day price ratio. Returns untouched.
#
# Factor = price divisor on the ex-date:
#   FV split Rs.10 -> Re.1        factor = 10/1 = 10
#   Bonus a:b (a new per b held)  factor = (a+b)/b   (1:2 -> 1.5)
#   Same-day split+bonus          factors multiply
#   Dividends: ignored (small; standard at this noise level)
#   Rights: skipped (needs issue price) — counted for the record
#
# Verify: observed_ratio = prev_close / close on the ex-day (both raw)
# should be close to the parsed factor (band allows same-day market moves).
# ============================================================================
import math
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
OUT = DRIVE / "ca_adjustment_factors.parquet"
BAND = 1.30
LOG_BAND = math.log(BAND)
DATE_FMT = "%d-%b-%Y"
# v2: split numbers anchored AFTER the split keyword. v1 took the first two
# numbers in the whole purpose string, so "Bonus 1:1 And Face Value Split
# Rs.10 To Re.1" parsed the bonus ratio as face values -> the split factor
# was silently dropped on every combined bonus+split row (TITAN, ONGC, ...).
SPLIT_PAT = r"(?i)spl\D*?(\d+(?:\.\d+)?)\D+?(\d+(?:\.\d+)?)"

# ---- load NSE CA, parse ex-dates -------------------------------------------
ca = pl.read_csv(DRIVE / "nse_corporate_actions.csv", infer_schema_length=0)
ca = ca.rename({c: c.strip() for c in ca.columns})
ca = ca.with_columns(
    pl.col("EX-DATE").str.strip_chars()
      .str.to_date(DATE_FMT, strict=False).alias("ex_date"),
    pl.col("SYMBOL").str.strip_chars().alias("symbol"),
    pl.col("PURPOSE").alias("purpose"),
)
ca = ca.filter(pl.col("ex_date").is_not_null())
print("NSE CA rows with valid ex-date:", ca.height)

print("\npurpose census (keyword hits):")
for kw in ["split", "bonus", "div", "rights", "amalgam", "demerger",
           "scheme", "consolidat", "reduction", "buy back"]:
    n = ca.filter(pl.col("purpose").str.contains("(?i)" + kw)).height
    print("  " + kw.ljust(12) + str(n).rjust(7))

# ---- SPLITS: "Fv Split Rs.10 To Re.1" (spelling variants: Splt/Split) ------
sp_cand = ca.filter(pl.col("purpose").str.contains(r"(?i)spl?i?t"))
sp = sp_cand.with_columns(
    pl.col("purpose").str.extract(SPLIT_PAT, 1).cast(pl.Float64).alias("fv_old"),
    pl.col("purpose").str.extract(SPLIT_PAT, 2).cast(pl.Float64).alias("fv_new"),
)
sp = sp.filter(
    pl.col("fv_old").is_not_null() & pl.col("fv_new").is_not_null()
    & (pl.col("fv_new") > 0) & (pl.col("fv_old") > pl.col("fv_new"))
)
sp = sp.with_columns(
    (pl.col("fv_old") / pl.col("fv_new")).alias("factor"),
    pl.lit("split").alias("kind"),
)
sp = sp.filter((pl.col("factor") > 1.01) & (pl.col("factor") <= 50))
sp = sp.select("symbol", "ex_date", "kind", "factor", "purpose")
print("\nsplits:", sp_cand.height, "candidate rows ->", sp.height, "parsed")
print("unparsed split samples (check nothing important is lost):")
unparsed = sp_cand.join(sp.select("symbol", "ex_date"),
                        on=["symbol", "ex_date"], how="anti")
print(unparsed.select("symbol", "ex_date", "purpose").head(8))

# ---- BONUSES: "Bonus 1:2" ---------------------------------------------------
bo_cand = ca.filter(pl.col("purpose").str.contains(r"(?i)bonus"))
bo = bo_cand.with_columns(
    pl.col("purpose").str.extract(r"(\d+)\s*:\s*(\d+)", 1)
      .cast(pl.Float64).alias("a"),
    pl.col("purpose").str.extract(r"(\d+)\s*:\s*(\d+)", 2)
      .cast(pl.Float64).alias("b"),
)
bo = bo.filter(pl.col("a").is_not_null() & (pl.col("b") > 0))
bo = bo.with_columns(
    ((pl.col("a") + pl.col("b")) / pl.col("b")).alias("factor"),
    pl.lit("bonus").alias("kind"),
)
bo = bo.filter((pl.col("factor") > 1.01) & (pl.col("factor") <= 20))
bo = bo.select("symbol", "ex_date", "kind", "factor", "purpose")
print("\nbonuses:", bo_cand.height, "candidate rows ->", bo.height, "parsed")

n_rights = ca.filter(pl.col("purpose").str.contains("(?i)rights")).height
print("rights issues (NOT adjusted, needs issue price):", n_rights)

# ---- combine; same symbol+day events multiply ------------------------------
# guard: at most ONE split and ONE bonus per (symbol, ex_date) — a duplicated
# CA row must not multiply the factor twice
ev = pl.concat([sp, bo]).unique(subset=["symbol", "ex_date", "kind"], keep="first")
factors = ev.group_by("symbol", "ex_date").agg(
    pl.col("factor").product().alias("factor"),
    pl.col("kind").first().alias("kind"),
    pl.len().alias("n_events"),
    pl.col("purpose").first().alias("purpose"),
)
n_multi = factors.filter(pl.col("n_events") > 1).height
print("\ncombined factor table:", factors.height, "events;",
      n_multi, "same-day multi-events (factors multiplied)")
print("factor distribution:")
print(factors["factor"].describe())

# ---- VERIFY against observed ex-day price ratios ---------------------------
panel = pl.read_parquet(DRIVE / "returns_panel.parquet")
panel = panel.select("symbol", "date", "close", "prev_close")
chk = factors.join(panel, left_on=["symbol", "ex_date"],
                   right_on=["symbol", "date"], how="left")
chk = chk.with_columns((pl.col("prev_close") / pl.col("close")).alias("obs_ratio"))
found = chk.filter(pl.col("obs_ratio").is_not_null() & (pl.col("obs_ratio") > 0))
print("\nevents with an ex-day price row in the panel:",
      found.height, "/", factors.height)
print("(rest: not NSE-listed that day / outside 2011-2025 / suspended)")

found = found.with_columns(
    ((pl.col("obs_ratio") / pl.col("factor")).log().abs() < LOG_BAND)
    .alias("confirmed")
)
n_ok = int(found["confirmed"].sum())
pct = 100.0 * n_ok / max(found.height, 1)
print("CONFIRMED (obs ratio within x/", BAND, "of factor):",
      n_ok, "/", found.height, "=", round(pct, 1), "%")
print("by kind:")
print(found.group_by("kind").agg(
    pl.len().alias("n"),
    (100 * pl.col("confirmed").mean()).round(1).alias("confirmed_pct"),
).sort("n", descending=True))

print("\nworst disagreements — eyeball these:")
bad = found.filter(~pl.col("confirmed"))
bad = bad.with_columns((pl.col("obs_ratio") / pl.col("factor")).alias("obs_over_factor"))
bad = bad.sort(pl.col("obs_over_factor").log().abs(), descending=True)
print(bad.select("symbol", "ex_date", "kind", "factor",
                 "obs_ratio", "obs_over_factor", "purpose").head(15))

# ---- write factor table (with verification columns) ------------------------
out = factors.join(found.select("symbol", "ex_date", "obs_ratio", "confirmed"),
                   on=["symbol", "ex_date"], how="left")
out.write_parquet(OUT)
print("\nwrote", OUT.name, out.shape)
print("")
print("VERDICT TO READ:")
print(" - confirmed pct is the number that matters. >=90: parser sound;")
print("   unconfirmed tail = postponed/cancelled ex-dates + same-day crashes")
print("   (check the worst-disagreement table).")
print(" - if <80: we FIX THE PARSER before touching any returns.")
print(" - nothing downstream was modified in this step.")
print("")
print("NEXT (5B-2, only after verdict): apply confirmed factors ->")
print("adjusted returns v2 -> re-run ex-day gate (must flip to PASS).")
