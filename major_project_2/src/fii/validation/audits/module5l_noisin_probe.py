# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5L · NOISIN / INVALID-KEY PROBE (read-only) — just show them
#
# Surfaces the fabricated placeholder keys (NOISIN<digits>) and any other
# invalid-format ISINs across FII trades, feature store, and model states.
# Answers: how many, do they map to ONE consistent SCRIP_NAME (recoverable)
# or many (noise), and did any leak into the 946 model cisins.
# ============================================================================
import polars as pl
from pathlib import Path

MODELD = ISIN_MAPPING
VALID = r"^IN[A-Z0-9]{10}$"   # 12 chars, starts IN (loose valid-ISIN shape)

# ---- scan FII trade data for invalid keys (+ their names) ------------------
parts = []
for fp in sorted(MODELD.glob("20??.parquet")):
    d = pl.read_parquet(fp, columns=["ISIN", "SCRIP_NAME"])
    inv = d.filter(pl.col("ISIN").is_null()
                   | ~pl.col("ISIN").str.contains(VALID))
    if inv.height:
        parts.append(inv.with_columns(
            pl.col("ISIN").fill_null("<NULL>").alias("ISIN")))
if parts:
    inv = pl.concat(parts)
else:
    inv = pl.DataFrame({"ISIN": [], "SCRIP_NAME": []})
print("invalid-key trade records (null or non-IN-format):", inv.height)

# per distinct invalid key: record count + distinct names
byk = (inv.group_by("ISIN")
          .agg(pl.len().alias("records"),
               pl.col("SCRIP_NAME").n_unique().alias("n_names"),
               pl.col("SCRIP_NAME").unique().alias("names"))
          .sort("records", descending=True))
print("distinct invalid keys:", byk.height)

# NOISIN family specifically
noisin = byk.filter(pl.col("ISIN").str.contains("(?i)NOISIN"))
print("\ndistinct NOISIN<...> codes:", noisin.height,
      "| total NOISIN records:", int(noisin["records"].sum()) if noisin.height else 0)
print("\n--- sample NOISIN codes (code | #records | #distinct names | a name) ---")
print(noisin.head(25).with_columns(
    pl.col("names").list.first().alias("sample_name")).select(
    "ISIN", "records", "n_names", "sample_name"))

# recoverability: does each NOISIN map to exactly ONE name?
if noisin.height:
    one = noisin.filter(pl.col("n_names") == 1).height
    print("\nNOISIN codes mapping to EXACTLY ONE name (recoverable via name):",
          one, "/", noisin.height,
          "(", round(100 * one / noisin.height, 1), "% )")
    print("NOISIN codes with MANY names (ambiguous/noise):",
          noisin.filter(pl.col("n_names") > 1).height)
    print("\nexamples with MANY names (why ambiguous):")
    print(noisin.filter(pl.col("n_names") > 1)
                .select("ISIN", "records", "n_names", "names").head(6))

# other (non-NOISIN) invalid keys, for completeness
other = byk.filter(~pl.col("ISIN").str.contains("(?i)NOISIN"))
print("\n--- other invalid keys (non-NOISIN): ", other.height, "---")
print(other.head(15).with_columns(
    pl.col("names").list.first().alias("sample_name")).select(
    "ISIN", "records", "n_names", "sample_name"))

# ---- did any invalid key leak into the MODEL cisins (the 946)? -------------
print("\n" + "=" * 66)
print("MODEL CONTAMINATION CHECK")
print("=" * 66)
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
cis = states["cisin"]
bad_c = states.filter(pl.col("cisin").is_null()
                      | ~pl.col("cisin").str.contains(VALID))
print("model cisins that are invalid-format:", bad_c["cisin"].n_unique(),
      "| stock-days on them:", bad_c.height,
      "(", round(100 * bad_c.height / states.height, 2), "% of 804,958 )")
if bad_c.height:
    print(bad_c.group_by("cisin").agg(pl.len().alias("stock_days"))
              .sort("stock_days", descending=True).head(20))
    print("^ these are FAKE or unrecoverable companies in the 946 -> decide:")
    print("  recover (if the cisin's NOISIN maps to 1 name) or drop+count.")
else:
    print("NONE -> the 946 cisins are all valid-format ISINs (good; the")
    print("NOISIN placeholders live only in raw trades, filtered before the")
    print("model). Then NOISIN only affects upstream feature coverage, not")
    print("the label universe.")

# feature store too
fs = MODELD / "stockday_features_v2.parquet"
if fs.exists():
    f = pl.read_parquet(fs, columns=["cisin"])
    fb = f.filter(pl.col("cisin").is_null()
                  | ~pl.col("cisin").str.contains(VALID))
    print("\nfeature-store cisins invalid-format:", fb["cisin"].n_unique(),
          "| rows:", fb.height)
