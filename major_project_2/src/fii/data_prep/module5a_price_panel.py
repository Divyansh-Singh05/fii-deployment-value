# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5A · STAGE-1 BUILD — ADJUSTED RETURN PANEL + COVERAGE GATES
#
# Input:  VALIDATION_DATA/bhavcopy_parquets/prices_2011..2025.parquet
#         VALIDATION_DATA/nse_corporate_actions.csv   (cross-check only)
#         VALIDATION_DATA/{nifty50,sp500,usdinr,india_vix}.parquet
#         ISIN_MAPPING/stockday_states_calibrated.parquet  (model output)
# Output: VALIDATION_DATA/returns_panel.parquet
#
# What it does, in order:
#   R1. Repair the 0020-vs-2020 date bug found by the 4C audit (two-digit
#       years parsed literally — poisons every date join in 2020).
#   R2. Backfill 2011's null ISINs via symbol → next-observed-ISIN.
#   R3. Dedupe (isin, date) preferring EQ > BE > BZ series rows.
#   RET. Daily returns = close/prev_close − 1. NSE's prev_close is restated
#       on ex-dates (post-split basis), so this is CA-ADJUSTED by the
#       exchange itself. We VERIFY that claim: days where close-to-close
#       and prev_close returns diverge >2% should line up with split/bonus
#       ex-dates in the NSE CA file. High explained-share = adjustment works.
#   MAC. Join macro series; detect + repair the yfinance timezone date shift
#       (nifty50/india_vix starting on a SUNDAY = dates one day early).
#   G1. Gate: delisted distress names survive with real trading histories.
#   G2. Gate: model stock-days (stockday_states_calibrated) join the price
#       panel — match rate overall / per era / per archetype. THE gate:
#       if HOSTAGE days can't find prices, Module 5 is dead on arrival.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE  = VALIDATION_DATA
BHAV   = DRIVE / "bhavcopy_parquets"
MODELD = ISIN_MAPPING
OUT    = DRIVE / "returns_panel.parquet"

# ── load & stack yearly parquets ────────────────────────────────────────────
files = sorted(BHAV.glob("prices_*.parquet"))
print(f"=== loading {len(files)} yearly parquets ===")
px = pl.concat([pl.read_parquet(f) for f in files])
print(f"raw rows: {px.height:,}")

# ── R1: repair two-digit-year dates (0020-07-13 → 2020-07-13) ───────────────
bad = px.filter(pl.col("date").dt.year() < 1900)
print(f"\n=== R1: date repair ===")
print(f"rows with year < 1900: {bad.height:,}  "
      f"(years seen: {sorted(bad['date'].dt.year().unique().to_list()) if bad.height else '—'})")
px = px.with_columns(
    pl.when(pl.col("date").dt.year() < 100)
      .then(pl.col("date").dt.offset_by("2000y"))
      .otherwise(pl.col("date"))
      .alias("date"))
still_bad = px.filter(pl.col("date").dt.year() < 1990).height
print(f"after repair, rows with year < 1990: {still_bad}  "
      f"({'PASS' if still_bad == 0 else 'FAIL — inspect manually'})")
print("per-year row counts and date spans after repair:")
print(px.group_by(pl.col("date").dt.year().alias("yr"))
        .agg(pl.len().alias("rows"), pl.col("date").min().alias("min_d"),
             pl.col("date").max().alias("max_d")).sort("yr"))

# ── R2: backfill null ISINs (2011 problem) via symbol ───────────────────────
print(f"\n=== R2: ISIN backfill ===")
null_by_yr = (px.group_by(pl.col("date").dt.year().alias("yr"))
                .agg(pl.col("isin").is_null().sum().alias("null_isin")).sort("yr"))
print("null ISINs per year BEFORE:"); print(null_by_yr.filter(pl.col("null_isin") > 0))
px = (px.sort(["symbol", "date"])
        .with_columns(pl.col("isin").backward_fill().over("symbol")))
n_left = px["isin"].is_null().sum()
print(f"null ISINs remaining after symbol-backfill: {n_left:,} "
      f"(symbols that vanished before ISINs appeared — dropped)")
px = px.filter(pl.col("isin").is_not_null())

# ── R3: dedupe (isin,date), prefer EQ > BE > BZ ─────────────────────────────
px = px.with_columns(pl.col("series").str.strip_chars())
pri = pl.when(pl.col("series") == "EQ").then(0).when(pl.col("series") == "BE").then(1).otherwise(2)
n0 = px.height
px = (px.with_columns(pri.alias("_pri"))
        .sort(["isin", "date", "_pri"])
        .unique(subset=["isin", "date"], keep="first")
        .drop("_pri"))
print(f"\n=== R3: dedupe === {n0 - px.height:,} duplicate (isin,date) rows removed → {px.height:,}")

# ── RET: returns two ways + empirical verification of prev_close adjustment ─
print(f"\n=== RET: daily returns ===")
px = (px.sort(["isin", "date"])
        .with_columns(
            pl.when(pl.col("prev_close") > 0)
              .then(pl.col("close") / pl.col("prev_close") - 1)
              .otherwise(None).alias("ret"),
            (pl.col("close") / pl.col("close").shift(1).over("isin") - 1).alias("_ret_cc")))

# divergent days = candidate corporate-action days (prev_close was restated)
div = px.filter((pl.col("ret").is_not_null()) & (pl.col("_ret_cc").is_not_null())
                & ((pl.col("_ret_cc") - pl.col("ret")).abs() > 0.02))
print(f"days where close-to-close vs prev_close returns diverge >2%: {div.height:,}")

# cross-check those days against NSE CA ex-dates (split/bonus/rights/etc.)
ca = pl.read_csv(DRIVE / "nse_corporate_actions.csv", infer_schema_length=0)
ca = ca.rename({c: c.strip() for c in ca.columns})
ca = (ca.with_columns(pl.col("EX-DATE").str.strip_chars().str.to_date("%d-%b-%Y", strict=False).alias("ex_date"))
        .filter(pl.col("ex_date").is_not_null())
        .select(pl.col("SYMBOL").str.strip_chars().alias("symbol"), "ex_date",
                pl.col("PURPOSE").alias("purpose"))
        .unique(subset=["symbol", "ex_date"]))
div_j = div.join(ca, left_on=["symbol", "date"], right_on=["symbol", "ex_date"], how="left")
explained = div_j["purpose"].is_not_null().sum()
print(f"  of those, matched to an NSE CA ex-date (same symbol+day): {explained:,} "
      f"({100*explained/max(div.height,1):.1f}% explained)")
print("  read: high % ⇒ prev_close-based returns ARE exchange-adjusted; the")
print("  unexplained remainder is mostly dividends-only days (small gaps),")
print("  BSE-only actions, and relists. Sample of LARGE unexplained divergences:")
print(div_j.filter(pl.col("purpose").is_null())
          .with_columns((pl.col("_ret_cc") - pl.col("ret")).abs().alias("gap"))
          .sort("gap", descending=True)
          .select("symbol", "date", "close", "prev_close", "ret", "_ret_cc", "gap")
          .head(10))

# extreme-return sanity: |ret|>25% frequency by year (should be rare, crisis-clustered)
print("\n|ret| > 25% days per year (prev_close basis — CA-adjusted):")
print(px.filter(pl.col("ret").abs() > 0.25)
        .group_by(pl.col("date").dt.year().alias("yr")).agg(pl.len().alias("n")).sort("yr"))

# ── MAC: macro joins with timezone-shift detection & repair ─────────────────
print(f"\n=== MAC: macro series ===")
trading_days = set(px["date"].unique().to_list())

def load_macro(name, col):
    m = pl.read_parquet(DRIVE / f"{name}.parquet").sort("date")
    wk = m.filter(pl.col("date").dt.weekday() >= 6).height   # Sat=6, Sun=7
    frac_wk = wk / m.height
    if frac_wk > 0.05:   # systematic tz shift: dates are one day early
        m = m.with_columns(pl.col("date").dt.offset_by("1d"))
        # roll any date that still isn't a trading day forward is NOT safe;
        # after +1d re-check and just report
        wk2 = m.filter(pl.col("date").dt.weekday() >= 6).height
        print(f"  {name}: {100*frac_wk:.1f}% weekend dates → tz-shift detected, "
              f"applied +1 day (weekend rows now {wk2})")
    else:
        print(f"  {name}: dates look clean ({wk} weekend rows)")
    in_cal = m.filter(pl.col("date").is_in(list(trading_days))).height
    print(f"    {in_cal}/{m.height} dates match the bhavcopy trading calendar")
    return m.with_columns((pl.col(col) / pl.col(col).shift(1) - 1).alias(f"{col}_ret"))

nifty = load_macro("nifty50", "nifty50")
vix   = load_macro("india_vix", "india_vix")
sp    = load_macro("sp500", "sp500")
inr   = load_macro("usdinr", "usdinr")

# S&P enters LAGGED (US closes after India: day-t India reacts to US t−1).
sp = sp.with_columns(pl.col("sp500_ret").shift(1).alias("sp500_ret_lag")).select("date", "sp500_ret_lag")

panel = (px.drop("_ret_cc")
           .join(nifty.select("date", "nifty50_ret"), on="date", how="left")
           .join(sp, on="date", how="left")
           .join(inr.select("date", "usdinr_ret"), on="date", how="left")
           .join(vix.select("date", "india_vix"), on="date", how="left"))
for c in ("nifty50_ret", "sp500_ret_lag", "usdinr_ret", "india_vix"):
    cov = panel[c].is_not_null().mean()
    print(f"  panel coverage {c}: {100*cov:.1f}%")
panel = panel.with_columns((pl.col("ret") - pl.col("nifty50_ret")).alias("ret_mktadj"))

# ── G1: delisted distress names must be present (survivorship gate) ─────────
print(f"\n=== G1: survivorship gate — distress/delisted names ===")
for sym in ["RCOM", "HDIL", "DHFL", "JETAIRWAYS", "RELCAPITAL", "YESBANK", "SUZLON", "JPASSOCIAT"]:
    d = panel.filter(pl.col("symbol") == sym)
    if d.height:
        print(f"  {sym:<12} {d.height:>5} days   {d['date'].min()} → {d['date'].max()}   "
              f"isin(s): {d['isin'].unique().to_list()}")
    else:
        print(f"  {sym:<12} *** ABSENT — investigate before trusting Hostage tests ***")

# ── G2: model ↔ price-panel join coverage (the make-or-break gate) ──────────
print(f"\n=== G2: model join coverage ===")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
j = states.join(panel.select(pl.col("isin"), pl.col("date"), pl.col("ret")),
                left_on=["cisin", "TR_DATE"], right_on=["isin", "date"], how="left")
print(f"overall stock-days with a same-day price row: "
      f"{100 * j['ret'].is_not_null().mean():.1f}%")
print(j.group_by("era").agg((100 * pl.col("ret").is_not_null().mean()).round(1).alias("match_%"),
                            pl.len().alias("n")).sort("era"))
print(j.group_by("archetype").agg((100 * pl.col("ret").is_not_null().mean()).round(1).alias("match_%"),
                                  pl.len().alias("n")).sort("archetype"))
print("read: if match% is low, cisin (canonical ISIN) ≠ bhavcopy same-day ISIN —")
print("      we then need the ISIN active/inactive mapping before 5B. Files available there:")
for f in sorted(MODELD.glob("*")):
    if f.is_file(): print(f"        {f.name}")

# ── write ───────────────────────────────────────────────────────────────────
panel.write_parquet(OUT)
print(f"\nwrote {OUT.name}: {panel.shape}")
print("\nNEXT (5B): forward-return validation — CARs after archetype episodes")
print("(Shark-acc drifts up, Hostage reverses, Robot ~flat). Only run once G2 passes.")
