# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5M · IS 946 THE CORRECT COMPANY COUNT? (read-only integrity check)
#
# Fragmentation risk: if the model's OWN canonicalization split a company's
# ISIN chain, one company appears as >1 cisin in the 946 -> 946 overcounts
# and each fragment has a truncated history (corrupted features/labels).
# Detector: group the 946 by ISSUER CODE (chars 4-7 = company). Two cisins
# sharing an issuer code are the same company UNLESS they co-exist in time
# (DVR/dual-class). Discriminator = DISJOINT dates (fragmentation) vs
# OVERLAP (legit distinct securities).
# ============================================================================
import polars as pl
from pathlib import Path

MODELD = ISIN_MAPPING

states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
per = (states.group_by("cisin").agg(
    pl.len().alias("days"),
    pl.col("TR_DATE").min().alias("first"),
    pl.col("TR_DATE").max().alias("last"))
    .with_columns(pl.col("cisin").str.slice(3, 4).alias("issuer")))
print("model cisins:", per.height,
      "| distinct issuer codes:", per["issuer"].n_unique())
print("-> if distinct issuers << 946, the extra cisins are fragments or",
      "dual-class")

# names + active status
mm = MODELD / "isin_master_clean.parquet"
nmap, amap = {}, {}
if mm.exists():
    M = pl.read_parquet(mm)
    nmap = dict(zip(M["ISIN"].to_list(), M["Name Of Issuer"].to_list()))
    amap = dict(zip(M["ISIN"].to_list(), M["ISIN Status"].to_list()))
active = set(pl.read_csv(MODELD / "active_isins.csv", infer_schema_length=0)
             ["active_isin"].drop_nulls().to_list())

# ---- issuer codes carrying MORE THAN ONE model cisin -----------------------
multi = (per.group_by("issuer").agg(pl.len().alias("n_cisin"),
                                    pl.col("cisin").alias("cisins"))
            .filter(pl.col("n_cisin") > 1).sort("n_cisin", descending=True))
print("\nissuer codes with >1 model cisin:", multi.height,
      "(", int(multi["n_cisin"].sum()), "cisins involved)")

pr = dict(zip(per["cisin"].to_list(),
              zip(per["first"].to_list(), per["last"].to_list(),
                  per["days"].to_list())))

def classify(cisins):
    # sort by first date; disjoint if each starts after previous ends
    cs = sorted(cisins, key=lambda c: pr[c][0])
    disjoint = True
    for a, b in zip(cs, cs[1:]):
        if pr[b][0] <= pr[a][1]:        # b starts before a ends -> overlap
            disjoint = False
    return "FRAGMENTATION" if disjoint else "co-exist(DVR?)"

print("\n--- each multi-cisin issuer (name | cisin | days | first->last) ---")
frag_ct = coex_ct = 0
for row in multi.iter_rows(named=True):
    cs = row["cisins"]
    kind = classify(cs)
    if kind == "FRAGMENTATION": frag_ct += 1
    else: coex_ct += 1
    nm = nmap.get(cs[0], "?")
    print(f"\n[{kind}] issuer {row['issuer']}  ({nm})")
    for c in sorted(cs, key=lambda x: pr[x][0]):
        f, l, dd = pr[c]
        act = "ACTIVE" if c in active else amap.get(c, "?")
        print(f"    {c}  {str(dd).rjust(5)}d  {f} -> {l}  [{act}]")

print("\n" + "=" * 60)
print("issuer codes flagged FRAGMENTATION (disjoint dates):", frag_ct)
print("issuer codes flagged co-exist / DVR (overlapping):", coex_ct)
true_companies = per["issuer"].n_unique()
print("\nnaive cisin count :", per.height)
print("distinct issuers  :", true_companies,
      "  (= true company count if every multi is DVR)")
print("fragmentation inflation:", per.height - true_companies,
      "extra cisins from shared issuers (of which", frag_ct,
      "issuer-groups look like true fragmentation)")

# ---- terminal check: are model cisins the ACTIVE leaf of their chain? -------
n_act = sum(1 for c in per["cisin"].to_list() if c in active)
n_ina = per.height - n_act
print("\nmodel cisins that are ACTIVE (terminal):", n_act,
      "| NOT active (old-form used as canonical?):", n_ina)
if n_ina:
    print("sample non-active model cisins (model may have keyed an old ISIN):")
    na = per.filter(~pl.col("cisin").is_in(list(active))).head(15)
    for c in na["cisin"].to_list():
        f, l, dd = pr[c]
        print(f"    {c}  {nmap.get(c,'?')[:34]:34}  {dd}d  {f}->{l}  "
              f"[{amap.get(c,'?')}]")

print("""
VERDICT on the 946:
 - FRAGMENTATION count = issuer codes where 2+ model cisins trade in
   DISJOINT periods = one company split across cisins (bad: truncated
   histories). If 0 -> 946 is NOT inflated by fragmentation.
 - co-exist/DVR = legitimately distinct securities sharing an issuer ->
   NOT a problem, correctly separate.
 - non-active model cisins = the model may have used a pre-split ISIN as
   the identity; if such a cisin's newer active form is ALSO a cisin, that
   pair shows up in fragmentation above.
 If FRAGMENTATION ~ 0 and non-active is small/benign, 946 is confirmed as
 the company count (modulo a few DVR pairs) and v3 can proceed. If
 FRAGMENTATION is material, the model universe itself needs an upstream
 fix before any economic result.
""")
