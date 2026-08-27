# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 6 · BLOCK/BULK/SHORT-DEAL CORROBORATION OF THE ARCHETYPES
#
# Economic reframing (Module 5): concentrated FII flow (SHARK) = liquidity-
# demanding BLOCK trades -> temporary price impact + reversal; dispersed flow
# (HOSTAGE) = no block footprint. This tests that mechanism with NSE's own
# block/bulk/short deal records (independent of the flow data).
#
# Headline test: SHARK_DIST and HOSTAGE are BOTH sell-regime days; the only
# difference is concentration. If SHARK_DIST days coincide with block/bulk
# SELL deals FAR more than HOSTAGE days -> concentrated selling = real large
# trades, confirming the block-impact reading.
# Also: SHARK_ACC should be enriched for BUY deals; ROBOT is the placebo.
#
# Deal data keyed by SYMBOL+DATE -> bridged to cisin via the v3 panel.
# ============================================================================
import math
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA

def find_folder(*keys):
    for p in DRIVE.iterdir():
        if p.is_dir() and all(k in p.name.lower() for k in keys):
            return p
    return None

# ---- parse block + bulk (same schema) and short (no side) ------------------
def load_dir_bb(folder, kind):
    out = []
    for fp in sorted(folder.glob("*.csv")):
        d = pl.read_csv(fp, infer_schema_length=0)
        d = d.rename({c: c.strip() for c in d.columns})
        cols = {c.lower(): c for c in d.columns}
        try:
            out.append(d.select(
                pl.col(cols["date"]).str.strip_chars()
                  .str.to_date("%d-%b-%Y", strict=False).alias("date"),
                pl.col(cols["symbol"]).str.strip_chars().alias("symbol"),
                pl.col(cols["buy / sell"]).str.strip_chars()
                  .str.to_uppercase().alias("side"),
                pl.col(cols["quantity traded"]).str.strip_chars()
                  .str.replace_all(",", "").cast(pl.Float64, strict=False)
                  .alias("qty")).with_columns(pl.lit(kind).alias("kind")))
        except Exception as e:
            print("  parse issue", fp.name, e)
    return pl.concat(out).filter(pl.col("date").is_not_null()) if out else None

def load_short(folder):
    out = []
    for fp in sorted(folder.glob("*.csv")):
        d = pl.read_csv(fp, infer_schema_length=0)
        d = d.rename({c: c.strip() for c in d.columns})
        cols = {c.lower(): c for c in d.columns}
        out.append(d.select(
            pl.col(cols["date"]).str.strip_chars()
              .str.to_date("%d-%b-%Y", strict=False).alias("date"),
            pl.col(cols["symbol"]).str.strip_chars().alias("symbol"),
            pl.col(cols["quantity"]).str.strip_chars()
              .str.replace_all(",", "").cast(pl.Float64, strict=False)
              .alias("qty")))
    return pl.concat(out).filter(pl.col("date").is_not_null()) if out else None

blk = load_dir_bb(find_folder("block"), "block")
bulk = load_dir_bb(find_folder("bulk"), "bulk")
short = load_short(find_folder("short"))
deals = pl.concat([blk, bulk])
print("block deals:", blk.height, "| bulk deals:", bulk.height,
      "| short-sell rows:", short.height)
print("side split:", deals.group_by("side").agg(pl.len()).sort("side").to_dicts())

# ---- bridge symbol+date -> cisin via v3 panel ------------------------------
panel = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
           .select("symbol", "date", "isin").unique(subset=["symbol", "date"]))
deals = deals.join(panel, on=["symbol", "date"], how="left")
short = short.join(panel, on=["symbol", "date"], how="left")
mr = 100 * float(deals["isin"].is_not_null().mean())
print("block/bulk deals mapped to a priced stock-day:", round(mr, 1), "%")

# ---- per (cisin,date) deal flags -------------------------------------------
flags = (deals.filter(pl.col("isin").is_not_null())
              .group_by("isin", "date")
              .agg((pl.col("side") == "SELL").any().alias("has_sell"),
                   (pl.col("side") == "BUY").any().alias("has_buy")))
sflags = (short.filter(pl.col("isin").is_not_null())
               .group_by("isin", "date").agg(pl.len().alias("_s"))
               .with_columns(pl.lit(True).alias("has_short")).drop("_s"))

# ---- join to model states --------------------------------------------------
states = pl.read_parquet(DRIVE / "states_v3.parquet").select(
    "cisin", "TR_DATE", "era", "archetype")
sd = (states.join(flags, left_on=["cisin", "TR_DATE"],
                  right_on=["isin", "date"], how="left")
            .join(sflags, left_on=["cisin", "TR_DATE"],
                  right_on=["isin", "date"], how="left")
            .with_columns(pl.col("has_sell").fill_null(False),
                          pl.col("has_buy").fill_null(False),
                          pl.col("has_short").fill_null(False)))

ARCHS = ["HOSTAGE", "SHARK_DIST", "SHARK_ACC", "ROBOT",
         "UNTAGGED_DIRECTIONAL"]
print("\n" + "=" * 66)
print("DEAL-COINCIDENCE RATE BY ARCHETYPE (% of stock-days)")
print("=" * 66)
tab = (sd.group_by("archetype").agg(
    pl.len().alias("n"),
    (100 * pl.col("has_sell").mean()).round(2).alias("sell_deal%"),
    (100 * pl.col("has_buy").mean()).round(2).alias("buy_deal%"),
    (100 * pl.col("has_short").mean()).round(2).alias("short%"))
    .sort("sell_deal%", descending=True))
print(tab)

def rates(arch, col):
    r = sd.filter(pl.col("archetype") == arch)
    return float(r[col].mean()), r.height

def ztest(a, b, col):
    p1, n1 = rates(a, col); p2, n2 = rates(b, col)
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0
    return p1, p2, (p1 / p2 if p2 else float("inf")), z

print("\n" + "=" * 66)
print("HEADLINE: concentrated vs dispersed SELLING (block-impact mechanism)")
print("=" * 66)
p1, p2, ratio, z = ztest("SHARK_DIST", "HOSTAGE", "has_sell")
print(f"SELL-deal rate  SHARK_DIST {100*p1:.2f}%  vs  HOSTAGE {100*p2:.2f}%")
print(f"  enrichment ratio {ratio:.2f}x   z = {z:.1f}",
      "(>3 => strongly enriched)")
print("  read: SHARK_DIST >> HOSTAGE confirms concentrated selling shows up")
print("  as real block/bulk SELL deals; dispersed HOSTAGE does not -> the")
print("  reversal is a block-impact phenomenon, on the concentration axis.")

print("\nBUY side (SHARK_ACC should lead):")
p1, p2, ratio, z = ztest("SHARK_ACC", "ROBOT", "has_buy")
print(f"BUY-deal rate  SHARK_ACC {100*p1:.2f}%  vs  ROBOT {100*p2:.2f}%"
      f"   ratio {ratio:.2f}x  z {z:.1f}")

print("\nplacebo: SHARK_DIST sell-deal rate vs ROBOT (should be enriched too):")
p1, p2, ratio, z = ztest("SHARK_DIST", "ROBOT", "has_sell")
print(f"  {100*p1:.2f}% vs {100*p2:.2f}%  ratio {ratio:.2f}x  z {z:.1f}")

print("\nby era (does the enrichment replicate TRAIN vs TEST?):")
print(sd.group_by("era", "archetype").agg(
    (100 * pl.col("has_sell").mean()).round(2).alias("sell%"))
    .filter(pl.col("archetype").is_in(["SHARK_DIST", "HOSTAGE"]))
    .sort(["era", "archetype"]))

print("""
VERDICT:
 If SHARK_DIST sell-deal rate is several x HOSTAGE (large z, replicating
 across eras), the block/bulk data INDEPENDENTLY confirms that the
 concentrated-selling regime = identifiable large trades -> the +68/+33bp
 reversal is liquidity-demanding block impact, not noise. That upgrades the
 archetype from a flow-statistics artifact to a mechanism with an external,
 exchange-reported footprint.
NEXT (6b): condition the SHARK_DIST forward reversal on block-deal presence
 -- episodes WITH a block sell deal should reverse MORE (dose-response).
""")
