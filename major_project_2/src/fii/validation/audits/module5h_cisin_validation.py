# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5H · ARE THE 946 MODEL CISINs CORRECT? (read-only cross-validation)
#
# cisin = a CONSTRUCTED canonical identity (collapse a company's old/new ISIN
# strings into one). No absolute guarantee exists; correctness is checked by
# corroborating against sources the mapping did NOT use:
#   (i)  the PRICE TAPE's own symbol assignments (NSE assigns symbols
#        independently -> old & canonical ISIN sharing a symbol = external
#        confirmation they are the same listing), and
#   (ii) name consistency within each collapsed chain (false-merge guard).
# Scope: the 946 cisins the model actually uses (not the whole 5,960).
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(120)

panel = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
model = set(states["cisin"].unique().to_list())
print("model cisins:", len(model))

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("S1 - WHY 5,960 RAW FII ISINs: prefix breakdown (non-standard forms")
print("     are the ones that MUST be canonicalized)")
print("=" * 74)
fii_isins = set()
for fp in sorted(MODELD.glob("20??.parquet")):
    d = pl.read_parquet(fp, columns=["ISIN"])
    fii_isins |= set(d["ISIN"].drop_nulls().unique().to_list())
def prefix(s):
    if s is None or len(s) < 3: return "other"
    if s.startswith("INE"): return "INE (standard NSE eq)"
    if s.startswith("IN9"): return "IN9 (partly-paid/special)"
    if s.startswith("IN8"): return "IN8 (BSE/old-form)"
    if s.startswith("IN0") or s.startswith("IN1"): return "IN0/IN1 (newer)"
    return "other"
pf = (pl.DataFrame({"isin": list(fii_isins)})
        .with_columns(pl.col("isin").map_elements(prefix,
                      return_dtype=pl.Utf8).alias("form"))
        .group_by("form").agg(pl.len().alias("n")).sort("n", descending=True))
print("raw FII unique ISINs:", len(fii_isins))
print(pf)
print("read: IN8/IN9/IN0 forms are the SAME companies under different")
print("strings -> collapsing them is exactly what canonicalization does.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("S2 - THE 946: format, presence in tape, status, chain vs singleton")
print("=" * 74)
lk_fp = MODELD / "isin_lookup.parquet"
lk = pl.read_parquet(lk_fp) if lk_fp.exists() else None
# which of the 946 are chain-heads (something maps INTO them) vs singletons
if lk is not None:
    canon_with_chain = set(lk["canonical_isin"].unique().to_list())
else:
    canon_with_chain = set()
n_chain = len(model & canon_with_chain)
n_single = len(model) - n_chain
print("chain-heads (>=1 old ISIN collapses into them):", n_chain)
print("singletons (never needed mapping -> identity trivially correct):",
      n_single)
print("-> only the", n_chain, "chain-heads carry ANY mapping risk.")

tape_isins = set(panel["isin"].unique().to_list())
fmt_ok = sum(1 for c in model if isinstance(c, str) and len(c) == 12
             and c[:2] == "IN")
print("valid 12-char IN* format:", fmt_ok, "/", len(model))
print("present in price tape:", len(model & tape_isins), "/", len(model))

# status from master/nsdl
mm = MODELD / "isin_master_clean.parquet"
if mm.exists():
    M = pl.read_parquet(mm).select(pl.col("ISIN").alias("cisin"),
                                   pl.col("ISIN Status").alias("status"))
    st = (pl.DataFrame({"cisin": list(model)})
            .join(M, on="cisin", how="left")
            .group_by("status").agg(pl.len().alias("n")).sort("n",
                                                              descending=True))
    print("\nISIN status of the 946 (canonical should be ACTIVE/surviving):")
    print(st)

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("S3 - INDEPENDENT CHECK: do collapsed chains share a TAPE symbol?")
print("=" * 74)
# isin -> set of symbols in the price tape
sym = (panel.select("isin", "symbol").unique()
            .group_by("isin").agg(pl.col("symbol").unique().alias("syms")))
symmap = dict(zip(sym["isin"].to_list(), sym["syms"].to_list()))

if lk is not None:
    chains = lk.filter(pl.col("canonical_isin").is_in(list(model)))
    chains = chains.select("old_isin", "canonical_isin", "old_name",
                           "match_type", "confidence",
                           *[c for c in ("exchange_verified",)
                             if c in lk.columns])
    both_in_tape = chains.filter(
        pl.col("old_isin").is_in(list(tape_isins))
        & pl.col("canonical_isin").is_in(list(tape_isins)))
    print("collapse links feeding the 946:", chains.height,
          "| both ISINs in tape (checkable):", both_in_tape.height)
    def share_symbol(o, c):
        so, sc = symmap.get(o), symmap.get(c)
        if not so or not sc: return None
        return len(set(so) & set(sc)) > 0
    checked = both_in_tape.with_columns(
        pl.struct("old_isin", "canonical_isin").map_elements(
            lambda s: share_symbol(s["old_isin"], s["canonical_isin"]),
            return_dtype=pl.Boolean).alias("same_symbol"))
    nconf = int(checked["same_symbol"].fill_null(False).sum())
    print("links where old & canonical SHARE a tape symbol (CONFIRMED by")
    print("NSE's independent symbol assignment):", nconf, "/", both_in_tape.height,
          "=", round(100 * nconf / max(both_in_tape.height, 1), 1), "%")
    print("\nlinks that do NOT share a symbol (rename OR possible bad merge):")
    print(checked.filter(~pl.col("same_symbol").fill_null(False))
                 .head(15))
    print("(different symbol is often a legit rename; inspect for any pair")
    print(" whose two names are clearly DIFFERENT companies = false merge.)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("S4 - FALSE-MERGE GUARD: name consistency within each canonical group")
print("=" * 74)
if lk is not None and "old_name" in lk.columns and "new_name" in lk.columns:
    def toks(s):
        return set(str(s).upper().replace(".", " ").split()) if s else set()
    grp = lk.filter(pl.col("canonical_isin").is_in(list(model)))
    grp = grp.with_columns(
        pl.struct("old_name", "new_name").map_elements(
            lambda s: (len(toks(s["old_name"]) & toks(s["new_name"]))
                       / max(len(toks(s["old_name"]) | toks(s["new_name"])), 1)),
            return_dtype=pl.Float64).alias("name_jaccard"))
    weird = grp.filter(pl.col("name_jaccard") < 0.2)
    print("collapse links with LOW name overlap (<0.2 Jaccard) -- candidate")
    print("false merges to eyeball:", weird.height, "of", grp.height)
    print(weird.select("old_isin", "old_name", "canonical_isin", "new_name",
                       "match_type", "confidence", "name_jaccard")
              .sort("name_jaccard").head(20))
    print("(legit: FV-change renames keep the core name -> high overlap.")
    print(" true different-company pairs here = real errors to fix.)")

# manual-review file, if the ISIN workstream already flagged uncertain ones
mr = MODELD / "isin_manual_review.csv"
if mr.exists():
    R = pl.read_csv(mr, infer_schema_length=0)
    print("\nisin_manual_review.csv exists:", R.shape, "cols:", R.columns)
    print("  how many of THOSE touch a model cisin:")
    rc = isinf = None
    for c in R.columns:
        if "isin" in c.lower(): rc = c; break
    if rc:
        touch = R.filter(pl.col(rc).is_in(list(model))).height
        print("  ", touch, "flagged rows involve a model cisin")

print("""
VERDICT (guarantee on the 946):
 - S2 n_single = cisins that never needed mapping -> identity is trivially
   correct (no construction). Only the chain-heads carry risk.
 - S3 %same-symbol = the INDEPENDENT corroboration: chains NSE also treats
   as one listing. High % = the construction agrees with the exchange.
 - S4 low-name-overlap rows = the ONLY places a false merge could hide;
   if that list is ~empty (all FV-rename look-alikes), no false merges.
 - Residual risk = (non-confirmed S3 links) + (S4 flags) + manual-review
   rows. If that set is small and names are consistent, the 946 cisins are
   corroborated by two independent sources, which is the strongest
   guarantee available short of the registrar's master file.
""")
