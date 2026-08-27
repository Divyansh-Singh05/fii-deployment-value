# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5G · ISIN PROVENANCE (read-only) — WHAT is each dataset, really?
#
# The worry: I've been calling returns_panel_v2 "the tape" (3,595 ISINs) but
# never checked what ISIN_MAPPING/YYYY.parquet are. If THEY also hold ~3,595
# ISINs, are we conflating two different things? And is the model's 946 a
# correct reduction or a lossy accident? This resolves identity first, then
# counts every ISIN universe, then traces 3,595 -> 946.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
BHAV = DRIVE / "bhavcopy_parquets"
pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_width_chars(120)

def isin_col(cols):
    for c in ("isin", "ISIN", "cisin", "CISIN"):
        if c in cols:
            return c
    return None

# ---------------------------------------------------------------------------
print("=" * 74)
print("SECTION A - IDENTITY: show schema + head so we SEE what each file is")
print("=" * 74)
samples = [
    ("BHAVCOPY  prices_2011", BHAV / "prices_2011.parquet"),
    ("BHAVCOPY  prices_2020", BHAV / "prices_2020.parquet"),
    ("ISIN_MAP  2011.parquet", MODELD / "2011.parquet"),
    ("ISIN_MAP  2020.parquet", MODELD / "2020.parquet"),
    ("MODEL     stockday_states_calibrated",
     MODELD / "stockday_states_calibrated.parquet"),
    ("FEATURES  stockday_features_v2", MODELD / "stockday_features_v2.parquet"),
]
for label, fp in samples:
    print("\n----", label, "----")
    if not fp.exists():
        print("   (absent)"); continue
    d = pl.read_parquet(fp)
    print("   shape:", d.shape)
    print("   cols :", d.columns)
    # classify by columns present
    cset = set(c.lower() for c in d.columns)
    kind = "?"
    if {"close", "prev_close"} & cset:
        kind = "PRICE tape (has close/prev_close)"
    elif {"buy_value", "sell_value", "buy_count", "quantity",
          "tr_type"} & cset:
        kind = "FII TRADE data (has buy/sell/quantity)"
    elif {"state", "archetype"} & cset:
        kind = "MODEL output (has state/archetype)"
    elif any("f_" in c for c in cset):
        kind = "FEATURE store (has F_ features)"
    elif {"old_isin", "canonical_isin", "new_isin"} & cset:
        kind = "ISIN MAPPING table"
    print("   -> looks like:", kind)
    print(d.head(4))

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("SECTION B - EVERY ISIN UNIVERSE, COUNTED")
print("=" * 74)
panel = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
bhav_isins = set(panel["isin"].unique().to_list())
print("bhavcopy price tape unique ISINs:", len(bhav_isins))

# ISIN_MAPPING yearwise
im_years = sorted(MODELD.glob("20??.parquet"))
im_isins = set()
im_col = None
for fp in im_years:
    d = pl.read_parquet(fp)
    c = isin_col(d.columns)
    if c is None:
        print("  ", fp.name, "-> no isin-like column! cols:", d.columns)
        continue
    im_col = c
    im_isins |= set(d[c].unique().to_list())
print("ISIN_MAPPING yearwise files:", len(im_years),
      "| isin col:", im_col, "| unique ISINs:", len(im_isins))

states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
model_cisins = set(states["cisin"].unique().to_list())
print("model (states) unique cisins:", len(model_cisins))

fs = MODELD / "stockday_features_v2.parquet"
fs_cisins = set()
if fs.exists():
    f = pl.read_parquet(fs)
    fc = isin_col(f.columns)
    fs_cisins = set(f[fc].unique().to_list()) if fc else set()
    print("feature-store v2 unique", fc, ":", len(fs_cisins))

print("\n-- SET RELATIONSHIPS --")
print("bhavcopy == ISIN_MAPPING-yearwise ?",
      "IDENTICAL SET" if bhav_isins == im_isins else "DIFFERENT")
print("  in both:", len(bhav_isins & im_isins),
      "| only bhavcopy:", len(bhav_isins - im_isins),
      "| only ISIN_MAP:", len(im_isins - bhav_isins))
print("model cisins inside bhavcopy tape:",
      len(model_cisins & bhav_isins), "/", len(model_cisins))
if fs_cisins:
    print("model cisins == feature-store cisins ?",
          "IDENTICAL" if model_cisins == fs_cisins else
          f"overlap {len(model_cisins & fs_cisins)} "
          f"(states-only {len(model_cisins - fs_cisins)}, "
          f"fs-only {len(fs_cisins - model_cisins)})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("SECTION C - TRACE THE 3,595 -> 946 REDUCTION")
print("=" * 74)
print("Hypothesis: the ~3,595 is the PRICE/market universe; the model's 946")
print("is the FII-traded, feature-complete, canonicalized subset. Steps:")

# how many of the 3595 price names ever map to a model cisin, via isin_lookup
lk = MODELD / "isin_lookup.parquet"
if lk.exists():
    L = pl.read_parquet(lk).select("old_isin", "canonical_isin")
    # canonical of every bhavcopy isin (itself if not in lookup)
    bmap = (pl.DataFrame({"isin": list(bhav_isins)})
              .join(L, left_on="isin", right_on="old_isin", how="left")
              .with_columns(pl.coalesce("canonical_isin", "isin")
                            .alias("canon")))
    n_canon = bmap["canon"].n_unique()
    print("bhavcopy raw ISINs:", len(bhav_isins),
          "-> distinct canonical after chain-collapse:", n_canon,
          "(", len(bhav_isins) - n_canon, "collapsed by splits/renames)")
    hit = bmap.filter(pl.col("canon").is_in(list(model_cisins)))["isin"].n_unique()
    print("raw price ISINs whose canonical IS a model cisin:", hit)
else:
    print("(isin_lookup.parquet absent — chain-collapse step skipped)")

# liquidity: model names vs the rest of the price universe
mm = MODELD / "isin_master_clean.parquet"
if mm.exists():
    M = pl.read_parquet(mm).select(
        pl.col("ISIN").alias("isin"), "total_trades")
    M = M.filter(pl.col("isin").is_in(list(bhav_isins)))
    M = M.with_columns(pl.col("isin").is_in(list(model_cisins)).alias("in_model"))
    print("\nliquidity of price-universe names, model vs non-model:")
    print(M.group_by("in_model").agg(
        pl.len().alias("n"),
        pl.col("total_trades").median().alias("median_trades"),
        pl.col("total_trades").quantile(0.25).alias("q25_trades"),
        pl.col("total_trades").max().alias("max_trades")).sort("in_model"))
    print("read: if in_model=true names have FAR higher median_trades, the")
    print("946 is exactly the liquid FII-relevant subset — reduction is")
    print("justified, not a bug.")

print("""
VERDICT:
 - SECTION A tells you what each file ACTUALLY is (price vs trade vs map).
   If ISIN_MAP/YYYY.parquet has close/prev_close, it is PRICE data (same
   universe as bhavcopy) -> 'the tape' label is fine, just duplicated, and
   the true FII source is upstream (its universe = the feature store's 946).
 - SECTION B: if bhavcopy and ISIN_MAP yearwise are the IDENTICAL set, that
   CONFIRMS they are the same price universe (your ~3,595 observation), not
   two different things being conflated.
 - SECTION C: the 3,595->946 drop should decompose into chain-collapse
   (splits merge ISINs) + liquidity/feature filter (FIIs only trade a liquid
   subset). If model names are the high-trade names, 946 is correct.
 - If instead ISIN_MAP/YYYY.parquet turns out to be FII TRADE data with
   3,595 ISINs while the model only kept 946, THEN we have a real question:
   which ~2,600 FII-traded names were dropped and why. Section C's liquidity
   split answers that too.
""")
