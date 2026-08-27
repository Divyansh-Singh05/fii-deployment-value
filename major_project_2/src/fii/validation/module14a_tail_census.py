# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 14A · ILLIQUID-TAIL CENSUS — read-only feasibility gate
#
# Question: is the excluded ~75% (FII-traded names NOT in the 946 model
# universe) measurable enough to run the concentration/reversal test?
# NO labels, NO economics here — census only, gates pre-registered.
#
# What it measures:
#  C1 universe: tail cisins, stock-days, activity by trade-count floor
#  C2 feature computability at relaxed floor N>=2 (raw columns null rates:
#     persistence_raw, entity_hhi_raw — the two the rule backbone needs)
#  C3 price-tape coverage (join to canonical returns_panel_v3), zero-vol
#     share, delisting truncation exposure, median |ret| (staleness/jump)
#  C4 friction bar: raw bhavcopy schema for HIGH/LOW; if present, a
#     Corwin-Schultz-style high-low half-spread preview, tail vs model
#
# PRE-REGISTERED VIABILITY GATES (all must PASS for 14B to proceed):
#  G1 >= 100,000 tail stock-days with N>=2 AND persistence_raw AND
#     entity_hhi_raw non-null (enough raw material for rule labels)
#  G2 price coverage of those days >= 70% (tail will be worse than the
#     model universe's 98.5%; below 70% the event study is not credible)
#  G3 >= 30,000 such days in TEST era (an OOS verdict must be possible)
# Friction (C4) is REPORTED, not gated — it sets the bar 14C must clear.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING

# ---- C1 · universe -----------------------------------------------------------
f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
st = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
model = set(st["cisin"].unique().to_list())
allc = set(f["cisin"].unique().to_list())
tail = allc - model
print("=" * 70)
print(f"C1 · cisins: all {len(allc)} | model {len(model)} | "
      f"TAIL {len(tail)}")
t = f.filter(pl.col("cisin").is_in(list(tail)))
t = t.with_columns(
    pl.when(pl.col("TR_DATE") <= pl.date(2021, 4, 30))
      .then(pl.lit("TRAIN"))
      .when(pl.col("TR_DATE") >= pl.date(2021, 7, 1))
      .then(pl.lit("TEST")).otherwise(pl.lit("MASK")).alias("era"))
print("tail stock-days total:", t.height)
for floor in (1, 2, 5):
    n = t.filter(pl.col("N") >= floor).height
    print(f"  N>={floor}: {n:8d} stock-days")
print(t.group_by("era").len().sort("era"))

# ---- C2 · raw-feature computability at N>=2 ----------------------------------
print("\nC2 · raw-column availability on tail days with N>=2:")
t2 = t.filter(pl.col("N") >= 2)
for c in ("persistence_raw", "entity_conc_raw", "entity_conc_buy_raw",
          "blockiness_raw", "imbalance_raw"):
    ok = float(t2[c].is_not_null().mean()) if c in t2.columns else -1
    print(f"  {c:22s} non-null {100*ok:5.1f}%")
core = t2.filter(pl.col("persistence_raw").is_not_null()
                 & pl.col("entity_conc_raw").is_not_null())
n_core = core.height
n_test = core.filter(pl.col("era") == "TEST").height
print(f"  CORE (persist & hhi non-null): {n_core} days "
      f"({core['cisin'].n_unique()} names) | TEST {n_test}")

# ---- C3 · price coverage on the canonical tape -------------------------------
print("\nC3 · price-tape coverage for CORE tail days:")
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "ret_adj", "volume"))
j = core.select("cisin", "TR_DATE", "era").join(
    p, left_on=["cisin", "TR_DATE"], right_on=["isin", "date"],
    how="left")
cov = float(j["ret_adj"].is_not_null().mean())
print(f"  ret_adj coverage {100*cov:.1f}%  (gate G2 >= 70%)")
print(j.group_by("era").agg(
    (100 * pl.col("ret_adj").is_not_null().mean()).round(1)
    .alias("cov_%")).sort("era"))
zv = float((j["volume"].fill_null(0) == 0).mean())
mret = float(j["ret_adj"].abs().median())
print(f"  zero/null-volume share {100*zv:.1f}% | "
      f"median |ret_adj| {mret:.4f}")
# forward-window survival: does a price exist 20 trading days later?
psort = p.sort(["isin", "date"]).with_columns(
    pl.col("date").shift(-20).over("isin").alias("d20"))
j2 = core.select("cisin", "TR_DATE").join(
    psort.select("isin", "date", "d20"),
    left_on=["cisin", "TR_DATE"], right_on=["isin", "date"], how="inner")
tr20 = float(j2["d20"].is_null().mean())
print(f"  20d-forward truncation share {100*tr20:.1f}% "
      "(delisting/suspension exposure — reported, not gated)")

# ---- C4 · friction bar (high/low spread proxy) -------------------------------
print("\nC4 · friction: raw bhavcopy schema + high-low half-spread")
raw = pl.read_parquet(DRIVE / "bhavcopy_parquets" / "prices_2019.parquet")
print("  raw columns:", raw.columns)
hl = [c for c in raw.columns if c.upper() in ("HIGH", "LOW")]
if len(hl) == 2:
    H, L = sorted(hl, key=lambda c: c.upper() != "HIGH")
    r = raw.filter((pl.col(L) > 0) & (pl.col(H) >= pl.col(L)))
    r = r.with_columns(
        ((pl.col(H) - pl.col(L)) / ((pl.col(H) + pl.col(L)) / 2))
        .alias("hlrng"))
    # crude half-spread proxy: median daily high-low range / 2
    # (upper bound on CS; good enough to SET the bar, refined in 14C)
    isin_col = next(c for c in raw.columns if "ISIN" in c.upper())
    agg = r.group_by(isin_col).agg(
        pl.col("hlrng").median().alias("mrng"), pl.len().alias("nd"))
    agg = agg.filter(pl.col("nd") >= 30)
    tl = agg.filter(pl.col(isin_col).is_in(list(tail)))
    md = agg.filter(pl.col(isin_col).is_in(list(model)))
    print(f"  2019 median daily HL-range: model "
          f"{1e4*float(md['mrng'].median()):.0f}bp | tail "
          f"{1e4*float(tl['mrng'].median()):.0f}bp "
          f"(names: {md.height}/{tl.height})")
    print("  -> 14C round-trip bar ~ HL-range x0.5 x2 (half-spread both"
          " ways) + impact allowance; refined per-episode in 14C")
else:
    print("  HIGH/LOW NOT in raw parquets -> 14C falls back to Roll /")
    print("  |ret| quantile friction proxy (weaker; flagged)")

# ---- verdict ------------------------------------------------------------------
print("\n" + "=" * 70)
g1 = n_core >= 100_000
g2 = cov >= 0.70
g3 = n_test >= 30_000
print(f"G1 core tail days >=100k: {n_core} "
      f"{'PASS' if g1 else 'FAIL'}")
print(f"G2 price coverage >=70%: {100*cov:.1f}% "
      f"{'PASS' if g2 else 'FAIL'}")
print(f"G3 TEST-era core days >=30k: {n_test} "
      f"{'PASS' if g3 else 'FAIL'}")
print("\nVERDICT:", "TAIL VIABLE — proceed to 14B (rule labels)"
      if g1 and g2 and g3 else
      "NOT VIABLE at pre-registered thresholds — report as the honest"
      " boundary of measurability; do NOT lower gates post hoc")
print("=" * 70)
