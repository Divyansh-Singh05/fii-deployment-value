# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5D · JOIN-ATTRITION DIAGNOSTIC (read-only; diagnose + discover)
#
# Goal: understand the ~9% G2 join loss BEFORE fixing it. Two mechanically
# distinct causes, which a flat per-archetype average cannot separate:
#
#   CAUSE A (crosswalk-fixable): the model's cisin (canonical ISIN) never
#     appears in the price tape at all -> the stock trades under a DIFFERENT
#     historical ISIN. A canonical<->historical crosswalk recovers these.
#   CAUSE B (NOT crosswalk-fixable): cisin IS in the tape, just not on that
#     date -> genuine non-trading (holiday/pre-listing/post-delisting).
#
# The fixable prize is CAUSE A only. This script sizes A vs B, checks
# whether distress names are hit harder, checks the END anchor (5B-4's
# reversal test) specifically, and inspects the existing ISIN_MAPPING files
# to see if the crosswalk already exists. NO writes, NO recovery join yet.
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
pl.Config.set_tbl_width_chars(120)
pl.Config.set_tbl_rows(30)

panel = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
panel_isins = set(panel["isin"].unique().to_list())
print("panel unique isins:", len(panel_isins))
print("model unique cisins:", states["cisin"].n_unique())

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 1 - DECOMPOSE THE ATTRITION: crosswalk-fixable (A) vs not (B)")
print("=" * 78)
j = states.join(
    panel.select("isin", "date", "ret_adj_mktadj"),
    left_on=["cisin", "TR_DATE"], right_on=["isin", "date"], how="left")
j = j.with_columns(
    pl.col("ret_adj_mktadj").is_not_null().alias("matched"),
    pl.col("cisin").is_in(list(panel_isins)).alias("cisin_in_tape"))

overall = 100 * float(j["matched"].mean())
print("overall match rate:", round(overall, 1), "%  (unmatched:",
      round(100 - overall, 1), "%)")

unm = j.filter(~pl.col("matched"))
print("\nof the UNMATCHED rows:")
a_share = 100 * float((~unm["cisin_in_tape"]).mean())
print("  CAUSE A (cisin absent from tape -> crosswalk-FIXABLE):",
      round(a_share, 1), "%")
print("  CAUSE B (cisin in tape, wrong date -> not fixable):",
      round(100 - a_share, 1), "%")

print("\nunmatched rows split by archetype x cause:")
tab = (unm.group_by("archetype")
          .agg(pl.len().alias("unmatched_n"),
               (100 * (~pl.col("cisin_in_tape")).mean())
               .round(1).alias("causeA_pct_fixable"))
          .sort("unmatched_n", descending=True))
print(tab)

print("\nMATCH RATE per archetype, and the CEILING if CAUSE A were fixed:")
cur = (j.group_by("archetype")
         .agg((100 * pl.col("matched").mean()).round(1).alias("match_now"),
              # ceiling = matched OR (unmatched only because cisin absent,
              # i.e. would be recoverable IF the alt-isin traded that day)
              pl.len().alias("n")))
# a conservative ceiling proxy: rows that are matched already; CAUSE-A rows
# are candidates but only recover if the alt isin traded that date (unknown
# until crosswalk is applied). We report the candidate pool size instead of
# a fake ceiling number.
cand = (unm.filter(~pl.col("cisin_in_tape"))
           .group_by("archetype").agg(pl.len().alias("causeA_candidates")))
print(cur.join(cand, on="archetype", how="left").sort("archetype"))

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 2 - ARE DISTRESS NAMES HIT HARDER? (the real worry)")
print("=" * 78)
DISTRESS = ["RCOM", "HDIL", "DHFL", "JETAIRWAYS", "RELCAPITAL",
            "YESBANK", "SUZLON", "JPASSOCIAT", "RELCAPITAL", "IL&FS",
            "COX&KINGS", "RELINFRA", "PMC", "SREINFRA"]
dfi = (panel.filter(pl.col("symbol").is_in(DISTRESS))
            .group_by("symbol")
            .agg(pl.col("isin").unique().alias("tape_isins"),
                 pl.col("date").min().alias("first"),
                 pl.col("date").max().alias("last")))
print("distress names present in tape:")
print(dfi.sort("symbol"))
# does the model carry these, and under which cisin?
dist_isins = set()
for row in dfi["tape_isins"].to_list():
    dist_isins.update(row)
in_model = states.filter(pl.col("cisin").is_in(list(dist_isins)))
print("\nmodel stock-days whose cisin is one of these distress isins:",
      in_model.height)
if in_model.height:
    dm = in_model.join(
        panel.select("isin", "date", "ret_adj_mktadj"),
        left_on=["cisin", "TR_DATE"], right_on=["isin", "date"], how="left")
    print("distress-name match rate:",
          round(100 * float(dm["ret_adj_mktadj"].is_not_null().mean()), 1),
          "%  vs overall", round(overall, 1), "%")
    print("  (if MUCH lower than overall -> distress selection bias real)")
    print("  (if similar -> the 9% is not distress-concentrated)")
print("NOTE: distress names whose canonical cisin is NOT in dist_isins")
print("would be invisible here -> that itself is a CAUSE-A signal, chased")
print("properly once the crosswalk is loaded (Part 4).")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 3 - END ANCHOR (5B-4 reversal test): does IT lose more?")
print("=" * 78)
runs = (states.sort(["cisin", "TR_DATE"])
        .with_columns(
            ((pl.col("archetype") != pl.col("archetype").shift(1))
             .fill_null(True)).cum_sum().over("cisin").alias("_run"))
        .group_by("cisin", "_run")
        .agg(pl.col("archetype").first(), pl.col("era").first(),
             pl.col("TR_DATE").first().alias("start_date"),
             pl.col("TR_DATE").last().alias("end_date")))

pk = panel.select(pl.col("isin").alias("cisin"),
                  pl.col("date"), pl.lit(True).alias("has"))
sm = (runs.join(pk, left_on=["cisin", "start_date"],
                right_on=["cisin", "date"], how="left")
          .rename({"has": "start_has"}))
sm = (sm.join(pk, left_on=["cisin", "end_date"],
              right_on=["cisin", "date"], how="left")
        .rename({"has": "end_has"}))
prof = (sm.group_by("era", "archetype")
          .agg(pl.len().alias("episodes"),
               (100 * pl.col("start_has").fill_null(False).mean())
               .round(1).alias("START_match"),
               (100 * pl.col("end_has").fill_null(False).mean())
               .round(1).alias("END_match"))
          .sort(["era", "archetype"]))
print("episode anchor match rates (START vs END), per era:")
print(prof)
print("read: if HOSTAGE END_match << START_match, the reversal test's own")
print("anchors are dropping out where restructuring changes the ISIN.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 4 - CROSSWALK DISCOVERY: does a canonical<->historical map")
print("         already exist in ISIN_MAPPING/?")
print("=" * 78)
candidates = ["isin_mapping_final.csv", "isin_unified_mapping.csv",
              "isin_mapping_v2.csv", "restructure_recovered.csv",
              "restructure_diagnostic.csv", "nsdl_isin_status.csv",
              "isin_master_clean.parquet", "isin_lookup.parquet",
              "active_isins.csv", "inactive_isins_v2.csv"]
for name in candidates:
    fp = MODELD / name
    if not fp.exists():
        print("\n--", name, "(absent)"); continue
    print("\n--", name, "--")
    try:
        if fp.suffix == ".parquet":
            d = pl.read_parquet(fp)
        else:
            d = pl.read_csv(fp, infer_schema_length=0)
        print("  shape:", d.shape, "| cols:", d.columns)
        print(d.head(4))
    except Exception as e:
        print("  could not read:", e)

print("\n" + "=" * 78)
print("VERDICT TO READ:")
print(" - Part 1 CAUSE-A share = the crosswalk-fixable fraction of the 9%.")
print("   If A is small, a crosswalk barely helps and the 9% is mostly")
print("   genuine non-trading (accept it). If A is large, build the fix.")
print(" - Part 2/3 say WHETHER the loss concentrates in distress / at the")
print("   END anchor -> whether it biases 5B-4 specifically.")
print(" - Part 4 shows which existing file (if any) IS the crosswalk:")
print("   look for a file with BOTH a canonical-ISIN col AND a historical/")
print("   old-ISIN col (or ISIN + status + successor). That schema drives")
print("   module5e (the recovery join) - NOT written until we see it here.")
