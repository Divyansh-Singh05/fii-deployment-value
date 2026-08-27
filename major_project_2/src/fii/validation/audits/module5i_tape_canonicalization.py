# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5I · CAN TAPE-SIDE CANONICALIZATION RECOVER MISSED ROWS? (read-only)
#
# 5D tested whether MODEL cisins appear in the tape (Cause A = 0.5%). It did
# NOT test the reverse: model companies whose PRE-SPLIT price history sits in
# the tape under an OLD ISIN string the canonical cisin never matches. Those
# are date-level (Cause B) misses that a TAPE-SIDE canonicalization
# (tape isin -> canonical, THEN join) could recover. This measures how many.
# No rebuild; just quantifies the prize so we can decide if it's worth it.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING

panel = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
lk = pl.read_parquet(MODELD / "isin_lookup.parquet").select(
    "old_isin", "canonical_isin")

tape_isins = set(panel["isin"].unique().to_list())
model = set(states["cisin"].unique().to_list())
unused = tape_isins - model
print("tape ISINs:", len(tape_isins), "| model cisins:", len(model))
print("unused tape ISINs (in tape, not a model cisin):", len(unused))

# canonical of each unused tape isin (itself if not in lookup)
lkmap = dict(zip(lk["old_isin"].to_list(), lk["canonical_isin"].to_list()))
def canon(i): return lkmap.get(i, i)

rows = []
for i in unused:
    rows.append((i, canon(i)))
u = pl.DataFrame(rows, schema=["isin", "canon"], orient="row")
u = u.with_columns(pl.col("canon").is_in(list(model)).alias("is_model_oldform"))
n_oldform = int(u["is_model_oldform"].sum())
print("")
print("=== classification of the", len(unused), "unused tape ISINs ===")
print("(a) OUT OF SCOPE (canonical not a model name; non-FII market):",
      len(unused) - n_oldform)
print("(b) OLD-FORM of a MODEL company (pre-split price under retired ISIN):",
      n_oldform, "  <- these carry recoverable price history")
print("\nsample of population (b):")
print(u.filter(pl.col("is_model_oldform")).head(15))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RECOVERY TEST: re-join model stock-days with a CANONICALIZED tape key")
print("=" * 70)
# canonical price key on the tape; dedup (ccanon,date) preferring the row
# whose isin already IS canonical, then higher volume
p = panel.with_columns(
    pl.col("isin").replace(lkmap, default=pl.col("isin")).alias("ccanon"))
p = p.with_columns((pl.col("isin") == pl.col("ccanon")).alias("_is_canon"))
p = (p.sort(["ccanon", "date", "_is_canon", "volume"],
            descending=[False, False, True, True])
       .unique(subset=["ccanon", "date"], keep="first"))

# baseline (current) join: cisin -> raw isin
j_old = states.join(panel.select("isin", "date",
                                 pl.lit(True).alias("m_old")),
                    left_on=["cisin", "TR_DATE"],
                    right_on=["isin", "date"], how="left")
# new join: cisin -> canonicalized tape key
j_new = j_old.join(p.select("ccanon", "date", pl.lit(True).alias("m_new")),
                   left_on=["cisin", "TR_DATE"],
                   right_on=["ccanon", "date"], how="left")

old_rate = 100 * float(j_new["m_old"].fill_null(False).mean())
new_rate = 100 * float(j_new["m_new"].fill_null(False).mean())
recovered = j_new.filter(pl.col("m_old").is_null()
                         & pl.col("m_new").fill_null(False))
print("current match (cisin -> raw isin) :", round(old_rate, 2), "%")
print("canonicalized (cisin -> canonical):", round(new_rate, 2), "%")
print("rows RECOVERED by tape-canonicalization:", recovered.height,
      "(", round(100 * recovered.height / states.height, 2), "% of all)")

print("\nrecovery by era (expect early/TRAIN to gain more -- older splits):")
print(j_new.group_by("era").agg(
    (100 * pl.col("m_old").fill_null(False).mean()).round(2).alias("old_%"),
    (100 * pl.col("m_new").fill_null(False).mean()).round(2).alias("new_%"),
    pl.len().alias("n")).sort("era"))

print("\nrecovery by archetype (does HOSTAGE gain?):")
print(j_new.group_by("archetype").agg(
    (100 * pl.col("m_old").fill_null(False).mean()).round(2).alias("old_%"),
    (100 * pl.col("m_new").fill_null(False).mean()).round(2).alias("new_%"))
    .sort("archetype"))

print("""
VERDICT:
 - population (a) = out-of-scope market names: correctly unused, no loss
   (the tape is a lookup table; extras are fine).
 - population (b) = old-form ISINs of model companies: the ONLY recoverable
   history. RECOVERY row-count above says how much the current join misses
   because of it.
 - If recovered % is tiny (<0.5%), the direct join is fine as-is and 5D's
   'accept the 9.6%' stands. If it's material (>1-2%, or concentrated in
   TRAIN/HOSTAGE), it is worth REBUILDING the price join on a canonical key
   (map tape isin -> canonical, carry CA factors across the boundary) before
   the final CARs -- a real correction, not cosmetic.
""")
