# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5F · UNIVERSE & COUNT AUDIT (read-only)
#
# Answers two worries directly with the files on disk:
#   Q1. Why 3,595 tape ISINs vs 946 model ISINs -- is 946 a clean SUBSET of
#       the tape, or is there a key mismatch? And is the gap just liquidity
#       (model = FII-traded liquid names; tape = whole NSE market)?
#   Q2. Is the 804,958 labeled-day count correct? Verified by internal
#       integrity (no dup keys), reconciliation to the Module-3 documented
#       split (509,185 train + 295,773 test), the archetype/state census
#       summing to the total, sane per-stock coverage, the May-Jun 2021
#       mask being present, and reconciliation to feature-store v2.
# No writes.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
pl.Config.set_tbl_rows(20)

panel = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")

tape_isins = set(panel["isin"].unique().to_list())
model_isins = set(states["cisin"].unique().to_list())

print("=" * 74)
print("Q1 - IS THE 946 MODEL UNIVERSE A CLEAN SUBSET OF THE 3,595 TAPE?")
print("=" * 74)
print("tape unique ISINs :", len(tape_isins))
print("model unique cisins:", len(model_isins))
inter = model_isins & tape_isins
absent = model_isins - tape_isins
print("model cisins FOUND in tape :", len(inter),
      "(", round(100 * len(inter) / len(model_isins), 1), "% )")
print("model cisins ABSENT from tape:", len(absent),
      "  <- these are the only genuinely unjoinable names")

# name + liquidity context from the master table
try:
    m = pl.read_parquet(MODELD / "isin_master_clean.parquet")
    m = m.select(pl.col("ISIN").alias("isin"),
                 pl.col("Name Of Issuer").alias("name"),
                 pl.col("total_trades"))
    if absent:
        print("\nthe absent model cisins (with names) -- inspect these:")
        print(m.filter(pl.col("isin").is_in(list(absent))).head(30))
        print("(absent + no master row => canonical ISIN the tape never had;")
        print(" likely BSE-primary or SME names outside NSE bhavcopy)")
    # liquidity contrast: model names vs the tape's OTHER names
    m2 = m.with_columns(
        pl.when(pl.col("isin").is_in(list(model_isins)))
          .then(pl.lit("MODEL")).otherwise(pl.lit("tape_only"))
          .alias("grp"))
    print("\nliquidity contrast (why the tape has ~3.8x more names):")
    print(m2.filter(pl.col("isin").is_in(list(tape_isins)))
            .group_by("grp")
            .agg(pl.len().alias("n"),
                 pl.col("total_trades").median().alias("median_total_trades"),
                 pl.col("total_trades").mean().round(0).alias("mean_total_trades"))
            .sort("grp"))
    print("read: MODEL names should have FAR higher median trades -> the")
    print("946 is the liquid, FII-traded subset; the other ~2,600 tape names")
    print("are thin/small/delisted stocks FIIs barely touched (correctly")
    print("outside the model).")
except Exception as e:
    print("(master table read failed, skipping liquidity contrast:", e, ")")

print("\n" + "=" * 74)
print("Q2 - IS THE 804,958 LABELED-DAY COUNT CORRECT?")
print("=" * 74)
n = states.height
nuniq = states.unique(subset=["cisin", "TR_DATE"]).height
print("rows in states file          :", n)
print("unique (cisin, TR_DATE) keys :", nuniq,
      "->", "OK no dups" if n == nuniq else "!! DUPLICATE KEYS INFLATE COUNT")

print("\nera split (should be 509,185 + 295,773 = 804,958 per Module-3 log):")
esplit = states.group_by("era").agg(pl.len().alias("rows")).sort("era")
print(esplit)
tot = esplit["rows"].sum()
print("era rows sum:", tot, "->", "matches total" if tot == n else "!! mismatch")

print("\narchetype census (must sum to", n, "):")
ac = states.group_by("archetype").agg(pl.len().alias("rows")).sort("rows",
                                                                    descending=True)
print(ac); print("sum:", ac["rows"].sum())
print("\nstate census (must sum to", n, "):")
sc = states.group_by("state").agg(pl.len().alias("rows")).sort("rows",
                                                               descending=True)
print(sc); print("sum:", sc["rows"].sum())

print("\ncoverage geometry:")
ncis = states["cisin"].n_unique()
nd = states["TR_DATE"].n_unique()
dmin, dmax = states["TR_DATE"].min(), states["TR_DATE"].max()
print("unique stocks:", ncis, "| unique dates:", nd,
      "| span:", dmin, "->", dmax)
# trading days available in the tape over the same span
tape_days = (panel.filter(pl.col("date").is_between(dmin, dmax))
                  ["date"].n_unique())
print("tape trading days in span:", tape_days,
      "| naive ceiling 946 x days =", ncis * tape_days)
print("actual/ceiling fill rate :",
      round(100 * n / (ncis * tape_days), 1), "%",
      "(well under 100% is EXPECTED: stocks enter/exit, thin FII days and",
      "incomplete-feature days are dropped, masks applied)")

print("\nMay-Jun 2021 mask check (Module-2/3 masked this window -> ~0 rows):")
masked = states.filter(pl.col("TR_DATE").is_between(
    pl.date(2021, 5, 1), pl.date(2021, 6, 30))).height
print("rows in 2021-05-01..2021-06-30:", masked,
      "->", "mask present" if masked == 0 else "mask NOT applied (investigate)")

print("\nper-stock day-count distribution (sane band = tens..few thousand):")
perstock = states.group_by("cisin").agg(pl.len().alias("days"))
print(perstock["days"].describe())

print("\nrows per year (should track market growth, dip at 2021 mask,")
print("partial 2025):")
print(states.group_by(pl.col("TR_DATE").dt.year().alias("yr"))
            .agg(pl.len().alias("rows")).sort("yr"))

# reconcile against feature store v2 if present
fs = MODELD / "stockday_features_v2.parquet"
if fs.exists():
    print("\nreconcile vs feature-store v2:")
    f = pl.read_parquet(fs)
    kcol = "cisin" if "cisin" in f.columns else (
        "isin" if "isin" in f.columns else None)
    dcol = "TR_DATE" if "TR_DATE" in f.columns else (
        "date" if "date" in f.columns else None)
    print("feature-store rows:", f.height, "| key cols:", kcol, dcol)
    if kcol and dcol:
        j = states.join(f.select([kcol, dcol]).unique(),
                        left_on=["cisin", "TR_DATE"],
                        right_on=[kcol, dcol], how="left", join_nulls=False)
        # crude: how many states rows have a feature-store twin
        fk = set((a, b) for a, b in
                 f.select([kcol, dcol]).unique().iter_rows())
        hit = sum(1 for a, b in states.select(["cisin", "TR_DATE"]).iter_rows()
                  if (a, b) in fk)
        print("states rows also in feature store:", hit,
              "(", round(100 * hit / n, 1), "% )")
        print("(states is built from >=60-day sequences over the feature")
        print(" store, so <100% and count != 788k are expected, not bugs)")

print("""
VERDICT:
 Q1 PASS if 'model cisins FOUND in tape' ~ 100% and MODEL median trades
    >> tape_only -> 946 is the liquid FII-traded subset of the 3,595-name
    market, not a key mismatch.
 Q2 PASS if: no dup keys, era split = 509,185/295,773, censuses sum to
    804,958, mask window ~0, per-stock counts sane. Then 804,958 is the
    correct, internally-consistent labeled-day total.
""")
