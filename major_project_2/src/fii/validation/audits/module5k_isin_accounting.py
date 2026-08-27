# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5K · ISIN ACCOUNTING + STRUCTURE + ENTITY-BOUNDARY AUDIT (read-only)
#
# Now that degenerate keys are ruled out (5L: MF/junk, not in model), this
# works over the VALID-EQUITY ISIN subset. Three questions:
#  Q1 ACCOUNTING: do active + inactive partition the valid-equity FII
#     universe? What's unclassified (the silent orphans you doubted)?
#  Q2 STRUCTURE: group by ISSUER CODE (chars 4-7 = stable company identity,
#     robust across IN8->INE and FV splits). Multi-ISIN codes = value-
#     preserving chains; terminal = the active member.
#  Q3 ENTITY BOUNDARY: legit closure lives WITHIN one issuer code. Any
#     isin_lookup link CROSSING issuer codes = merger/acq treated as rename
#     = false-merge candidate (should be a DEATH). Flag + name them.
# No writes.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(120)
VALID = r"^IN[A-Z0-9]{10}$"

def issuer(i):   # chars 4-7 (index 3:7) = 4-char issuer code
    return i[3:7] if isinstance(i, str) and len(i) >= 7 else None

# ---- valid-equity universes ------------------------------------------------
fii = set()
for fp in sorted(MODELD.glob("20??.parquet")):
    d = pl.read_parquet(fp, columns=["ISIN"])
    d = d.filter(pl.col("ISIN").is_not_null()
                 & pl.col("ISIN").str.contains(VALID))
    fii |= set(d["ISIN"].unique().to_list())
bhav = set(pl.read_parquet(DRIVE / "returns_panel_v2.parquet")["isin"]
           .unique().to_list())
bhav = {b for b in bhav if b and len(b) == 12 and b[:2] == "IN"}
allu = fii | bhav
print("valid-equity FII ISINs:", len(fii), "| bhavcopy:", len(bhav),
      "| union:", len(allu))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Q1 - ACCOUNTING: does active + inactive partition valid-equity FII?")
print("=" * 72)
def load_set(name, guess):
    fp = MODELD / name
    if not fp.exists():
        print("  (missing", name, ")"); return set(), None
    d = pl.read_csv(fp, infer_schema_length=0)
    col = next((c for c in d.columns if guess in c.lower()), d.columns[0])
    return set(d[col].drop_nulls().to_list()), col
active, ac = load_set("active_isins.csv", "active")
inactive, ic = load_set("inactive_isins_v2.csv", "inactive")
print("active list:", len(active), "(", ac, ") | inactive:", len(inactive),
      "(", ic, ") | overlap(err):", len(active & inactive))
for label, U in [("valid-equity FII", fii), ("union", allu)]:
    a = len(U & active); i = len(U & inactive); nei = len(U - active - inactive)
    print(f"\n  universe {label} (n={len(U)}):")
    print(f"    active {a} | inactive {i} | UNCLASSIFIED {nei}",
          "->", "CLOSED" if nei == 0 else "NOT CLOSED")
unclass = sorted(fii - active - inactive)
print("\nsample UNCLASSIFIED valid-equity FII ISINs (in neither list):")
print(unclass[:25])
# are unclassified ones actually in the tape / model? (do they matter?)
model = set(pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
            ["cisin"].unique().to_list())
print("unclassified that are MODEL cisins:", len(set(unclass) & model),
      "| that are in bhavcopy:", len(set(unclass) & bhav))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Q2 - STRUCTURE: group by ISSUER CODE (chars 4-7)")
print("=" * 72)
u = (pl.DataFrame({"isin": sorted(allu)})
       .with_columns(pl.col("isin").map_elements(issuer, return_dtype=pl.Utf8)
                     .alias("iss")))
g = (u.group_by("iss").agg(pl.len().alias("n"), pl.col("isin").alias("members"))
       .sort("n", descending=True))
print("distinct issuer codes (~distinct companies):", g.height)
print("issuer codes with >1 ISIN (value-preserving chains):",
      g.filter(pl.col("n") > 1).height)
print("chain-length distribution:")
print(g.group_by("n").agg(pl.len().alias("num_issuers")).sort("n").head(10))
def n_act(ms): return sum(1 for m in ms if m in active)
gm = (g.filter(pl.col("n") > 1)
        .with_columns(pl.col("members").map_elements(n_act,
                      return_dtype=pl.Int64).alias("n_active")))
print("\nmulti-ISIN chains by #active members (healthy = exactly 1):")
print(gm.group_by("n_active").agg(pl.len().alias("chains")).sort("n_active"))
print("  0=company fully dead | 1=clean (map olds->terminal) | >1=ambiguous")
print("\nsample >1-active chains (need manual terminal pick):")
print(gm.filter(pl.col("n_active") > 1).select("iss", "n", "members").head(6))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Q3 - ENTITY BOUNDARY: cross-issuer lookup links = false-merge flags")
print("=" * 72)
lk = pl.read_parquet(MODELD / "isin_lookup.parquet").select(
    "old_isin", "canonical_isin")
lk = lk.with_columns(
    pl.col("old_isin").map_elements(issuer, return_dtype=pl.Utf8).alias("oi"),
    pl.col("canonical_isin").map_elements(issuer, return_dtype=pl.Utf8).alias("ci"))
cross = lk.filter((pl.col("oi") != pl.col("ci")) | pl.col("oi").is_null()
                  | pl.col("ci").is_null())
print("lookup links:", lk.height, "| SAME issuer (legit):",
      lk.height - cross.height, "| CROSS issuer (suspect):", cross.height)
mm = MODELD / "isin_master_clean.parquet"
if mm.exists():
    M = pl.read_parquet(mm).select(pl.col("ISIN").alias("k"),
                                   pl.col("Name Of Issuer").alias("nm"))
    nmap = dict(zip(M["k"].to_list(), M["nm"].to_list()))
    cross = cross.with_columns(
        pl.col("old_isin").map_elements(lambda x: nmap.get(x, ""),
                                        return_dtype=pl.Utf8).alias("old_nm"),
        pl.col("canonical_isin").map_elements(lambda x: nmap.get(x, ""),
                                              return_dtype=pl.Utf8).alias("can_nm"))
print("\ncross-issuer links (legit rename w/ new issuer code, or wrong")
print("merge? compare names):")
print(cross.head(20))
print("\ncross-issuer links that feed a MODEL cisin (highest stakes):")
print(cross.filter(pl.col("canonical_isin").is_in(list(model))).head(20))

print("""
VERDICT:
 Q1: UNCLASSIFIED>0 = active/inactive don't partition the valid-equity FII
     universe; those must be classified (esp. any that are model cisins or
     in bhavcopy) before the closure is trustworthy.
 Q2: '1 active' chains -> safe to close olds to terminal. '0 active' = dead.
     '>1 active' -> manual terminal pick.
 Q3: SAME-issuer links = safe closures; CROSS-issuer = merger/acq -> should
     be deaths, each printed one to confirm or kill (those feeding a model
     cisin matter most -- a wrong one contaminated the label universe).
Then: rebuild crosswalk = issuer-bounded closure (map within issuer code to
terminal active; cross-issuer = dead) -> v3 price panel on that key + the
guarded 2011 backfill.
""")
