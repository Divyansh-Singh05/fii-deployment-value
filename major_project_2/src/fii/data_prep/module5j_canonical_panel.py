# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5J · v3 CANONICAL PANEL + DUAL-SIDE CLOSURE  (tape AND model)
#
# Closure map (built once, applied to BOTH bhavcopy and the model states so
# they share ONE company key):
#   ccanon(k) = isin_lookup[k]                     (CA-based, class-aware)
#               else issuer terminal (latest-trading ISIN of the issuer code)
#                    IF k does not co-exist with that terminal
#                    (overlap < 180d => chain/split -> collapse)
#               else k                              (co-existing DVR/partly-
#                                                    paid & singletons: keep)
# Terminal uses LATEST TRADE DATE, not the active-list flag (5K showed the
# list is incomplete). Overlap guard keeps Bharti partly-paid separate while
# merging Tata Steel / Bajaj / Alok / Ruchi / Vaibhav / Chola fragments.
#
# Outputs: returns_panel_v3.parquet (isin = canonical), states_v3.parquet
# (cisin merged). Supersedes v2 + calibrated states for CAR work.
# ============================================================================
import datetime as dt
from collections import defaultdict
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
GUARD, OVERLAP_D = 0.50, 180

v2 = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
fac = pl.read_parquet(DRIVE / "ca_adjustment_factors.parquet")
lk = pl.read_parquet(MODELD / "isin_lookup.parquet")
lkmap = dict(zip(lk["old_isin"].to_list(), lk["canonical_isin"].to_list()))
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")

# ---- per-ISIN date span from the tape --------------------------------------
dts = v2.group_by("isin").agg(pl.col("date").min().alias("f"),
                              pl.col("date").max().alias("l"))
first = dict(zip(dts["isin"].to_list(), dts["f"].to_list()))
last = dict(zip(dts["isin"].to_list(), dts["l"].to_list()))
panel_isins = set(v2["isin"].unique().to_list())
model_cisins = set(states["cisin"].unique().to_list())

def iss(k):
    return k[3:7] if isinstance(k, str) and len(k) >= 7 else None

# terminal per issuer code = the priced ISIN with the latest last-trade
grp = defaultdict(list)
for k in panel_isins:
    grp[iss(k)].append(k)
terminal = {c: max(m, key=lambda x: last.get(x, dt.date(1900, 1, 1)))
            for c, m in grp.items()}

def overlap_days(a, b):
    if a not in first or b not in first:
        return -99999
    lo = max(first[a], first[b]); hi = min(last[a], last[b])
    return (hi - lo).days

# ---- AUDIT ITEM 2 (extended): the closure is now POINT-IN-TIME -------------
# The pre-audit `ccanon` was a full-sample function applied to all history, in
# two separate ways, and it ran AFTER module1 -- so fixing module1 alone would
# have removed the hindsight and then re-injected it here.
#
#   1. `lkmap` (isin_lookup.parquet) is old_isin -> canonical_isin with no
#      effective date, exactly the defect measured in module1.
#   2. `terminal[c]` picks, per issuer code, the priced ISIN with the LATEST
#      last-trade date -- a full-sample argmax. It selects the line that turns
#      out to survive longest, which is knowable only at the end of the sample,
#      and `overlap_days` then compares whole-sample spans.
#
# Replaced by an as-of join against the dated interval map: a row on date t is
# collapsed only by corporate actions already effective at t. The undated
# issuer-code fallback is NOT point-in-time reconstructible from what is on
# disk, so it is OFF by default; ISINs it used to merge now stay separate until
# a dated CA merges them. That costs coverage, and the cost is measured and
# printed below rather than assumed away.
from fii.data_prep.isin_pit_map import OUT as PIT_MAP  # noqa: E402

USE_UNDATED_ISSUER_FALLBACK = False   # full-sample; see note above

pit = (pl.read_parquet(PIT_MAP)
         .select("old_isin", "effective_from", "canonical_isin")
         .sort("effective_from"))
print(f"[pit] identity map: {pit.height:,} intervals over "
      f"{pit['old_isin'].n_unique():,} ISINs "
      f"| undated issuer fallback: "
      f"{'ON' if USE_UNDATED_ISSUER_FALLBACK else 'OFF (point-in-time)'}")


def pit_close(df, key, datecol):
    """Attach `ccanon` by backward as-of join on the corporate-action date."""
    out = (df.sort(datecol)
             .join_asof(pit.rename({"old_isin": key}).sort("effective_from"),
                        left_on=datecol, right_on="effective_from",
                        by=key, strategy="backward")
             .with_columns(pl.coalesce("canonical_isin", key).alias("ccanon"))
             .drop(["canonical_isin", "effective_from"]))
    if USE_UNDATED_ISSUER_FALLBACK:
        fb = {k: terminal[iss(k)] for k in panel_isins
              if iss(k) in terminal and k != terminal[iss(k)]
              and overlap_days(k, terminal[iss(k)]) < OVERLAP_D}
        if fb:
            fbdf = pl.DataFrame({"ccanon": list(fb), "_fb": list(fb.values())})
            out = (out.join(fbdf, on="ccanon", how="left")
                      .with_columns(pl.coalesce("_fb", "ccanon").alias("ccanon"))
                      .drop("_fb"))
    return out


_n_tape = (pit.filter(pl.col("canonical_isin") != pl.col("old_isin"))
              ["old_isin"].is_in(list(panel_isins)).sum())
_n_model = (pit.filter(pl.col("canonical_isin") != pl.col("old_isin"))
               ["old_isin"].is_in(list(model_cisins)).sum())
print(f"dated CA intervals touching tape ISINs : {_n_tape:,}")
print(f"dated CA intervals touching model cisins: {_n_model:,}")

# ---- TAPE: apply closure, dedup, returns, factors, Gate A ------------------
p = pit_close(v2, "isin", "date").rename({"isin": "isin_raw"})
p = p.with_columns((pl.col("isin_raw") == pl.col("ccanon")).alias("_c"))
n0 = p.height
p = (p.sort(["ccanon", "date", "_c", "volume"],
            descending=[False, False, True, True])
       .unique(subset=["ccanon", "date"], keep="first").drop("_c"))
print("\ntape dedup removed", n0 - p.height, "overlap rows ->", p.height)
p = p.rename({"ccanon": "isin"}).sort(["isin", "date"])
p = p.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("isin") - 1).alias("ret_cc"))

apply_ev = fac.filter(pl.col("confirmed") | pl.col("obs_ratio").is_null())
excl_ev = fac.filter(pl.col("confirmed") == False)  # noqa: E712
rows = p.select("symbol", "date").unique().sort("date")
mapd = (apply_ev.select("symbol", "ex_date", "factor").sort("ex_date")
        .join_asof(rows, left_on="ex_date", right_on="date", by="symbol",
                   strategy="forward"))
per_day = (mapd.filter(pl.col("date").is_not_null())
               .group_by("symbol", "date")
               .agg(pl.col("factor").product().alias("adj_factor")))
p = p.join(per_day, on=["symbol", "date"], how="left")
p = p.with_columns(((1 + pl.col("ret_cc"))
                    * pl.col("adj_factor").fill_null(1.0) - 1).alias("ret_adj"))
nb = p.filter(pl.col("adj_factor").is_not_null()
              & (pl.col("ret_adj").abs() > GUARD)).height
p = p.with_columns(pl.when(pl.col("adj_factor").is_not_null()
                           & (pl.col("ret_adj").abs() > GUARD)).then(None)
                   .otherwise(pl.col("ret_adj")).alias("ret_adj"))
p = p.join(excl_ev.select("symbol", pl.col("ex_date").alias("date"),
                          pl.lit(True).alias("_k")), on=["symbol", "date"],
           how="left")
p = p.with_columns(pl.when(pl.col("_k")).then(None)
                   .otherwise(pl.col("ret_adj")).alias("ret_adj")).drop("_k")
p = p.with_columns((pl.col("ret_adj") - pl.col("nifty50_ret"))
                   .alias("ret_adj_mktadj"))
print("application guard nulled:", nb)

conf = fac.filter(pl.col("confirmed") == True)  # noqa: E712
gg = conf.join(p.select("symbol", "date", "ret_adj"),
               left_on=["symbol", "ex_date"], right_on=["symbol", "date"],
               how="inner").filter(pl.col("ret_adj").is_not_null())
med = float(gg["ret_adj"].abs().median())
print("\nGATE A: confirmed ex-days", gg.height, "| median |ret_adj|",
      round(med, 4), "->", "PASS" if med < 0.05 else "FAIL")

# ---- MODEL: apply the SAME closure to states -> states_v3 ------------------
sv3 = (pit_close(states, "cisin", "TR_DATE")
             .drop("cisin").rename({"ccanon": "cisin"}))
d0 = sv3.height
sv3 = sv3.unique(subset=["cisin", "TR_DATE"], keep="first")
print("\nstates: merged fragments, dropped", d0 - sv3.height,
      "overlap-day dups -> cisins now:", sv3["cisin"].n_unique(),
      "(pre-audit full-sample closure gave 946)")

# ---- coverage on v3 --------------------------------------------------------
j = sv3.join(p.select("isin", "date", "ret_adj_mktadj"),
             left_on=["cisin", "TR_DATE"], right_on=["isin", "date"],
             how="left")
print("\nmodel match on v3:",
      round(100 * float(j["ret_adj_mktadj"].is_not_null().mean()), 2),
      "% (v2 90.4%)")
print(j.group_by("era").agg(
    (100 * pl.col("ret_adj_mktadj").is_not_null().mean()).round(2).alias("m%"),
    pl.len().alias("n")).sort("era"))
print(j.group_by("archetype").agg(
    (100 * pl.col("ret_adj_mktadj").is_not_null().mean()).round(2).alias("m%"))
    .sort("archetype"))
print("2011 coverage (backfill sanity):",
      round(100 * float(j.filter(pl.col("TR_DATE").dt.year() == 2011)
                        ["ret_adj_mktadj"].is_not_null().mean()), 2), "%")

p.write_parquet(DRIVE / "returns_panel_v3.parquet")
sv3.write_parquet(DRIVE / "states_v3.parquet")
print("\nwrote returns_panel_v3.parquet", p.shape,
      "and states_v3.parquet", sv3.shape)
print("NEXT: point 5B-4 at returns_panel_v3 + states_v3, run END-anchor test.")
