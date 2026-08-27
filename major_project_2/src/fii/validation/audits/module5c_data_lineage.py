# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 5C · DATA LINEAGE WALKTHROUGH (read-only, forensic replay)
#
# Reads the artifacts already built, in the ORDER they were derived, and
# prints a real row at each stage so the transformations are visible, not
# just described. Running example: TITAN's 2011-06-23 split(10:1)+bonus(1:1)
# event (combined factor 20) — the case we've discussed most.
#
# Also answers: have bulk/block/short deals been used yet? (STAGE 0 — no.)
# ============================================================================
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
pl.Config.set_tbl_width_chars(120)

def hdr(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)

# ---------------------------------------------------------------------------
hdr("STAGE 0 — bulk / block / short deals: USED SO FAR? -> NO")
print("Collected in Module 4 (data collection), never joined into any")
print("Module 5 script. Only bhavcopy + CA file + macro have been used.")
print("Planned future role (per project architecture): SHARK_ACC")
print("cross-check -- a SHARK_ACC-labeled buying episode SHOULD show up")
print("as a bulk/block buy deal for a genuinely informed large trade;")
print("absence would question the label. Not yet built. Heads below")
print("just to confirm the data is there and untouched:")
bulk = pl.read_csv(DRIVE / "bulk deals/Bulk-Deals-01-01-2011-to-31-12-2011.csv",
                   infer_schema_length=0, n_rows=3)
print("")
print("-- bulk deals/Bulk-Deals-...-2011.csv (head) --")
print(bulk)
block = pl.read_csv(DRIVE / "block deals /Block-Deals-01-01-2011-to-31-12-2011.csv",
                    infer_schema_length=0, n_rows=3)
print("")
print("-- block deals /Block-Deals-...-2011.csv (head) --")
print(block)

# ---------------------------------------------------------------------------
hdr("STAGE 1 — raw bhavcopy (before ANY repair): prices_2011 / prices_2020")
print("This is what NSE actually published. Two known defects live here.")
raw11 = pl.read_parquet(DRIVE / "bhavcopy_parquets/prices_2011.parquet")
print("")
print("-- prices_2011.parquet: TITAN rows around the 2011-06-23 event --")
print(raw11.filter(pl.col("symbol") == "TITAN")
          .filter(pl.col("date").is_between(
              pl.date(2011, 6, 20), pl.date(2011, 6, 27)))
          .sort("date"))
print("^ isin is null here (2011 defect) -- see R2 in Stage 2.")
raw20 = pl.read_parquet(DRIVE / "bhavcopy_parquets/prices_2020.parquet")
print("")
print("-- prices_2020.parquet: year-corrupted rows (R1 defect) --")
print(raw20.filter(pl.col("date").dt.year() < 100).head(3))

# ---------------------------------------------------------------------------
hdr("STAGE 2 — returns_panel.parquet: after R1/R2/R3 repair + ret_cc/ret")
print("module5a: date repair, ISIN backfill (symbol -> next known ISIN),")
print("dedupe EQ>BE>BZ, then macro join (NIFTY/SP500/USDINR/VIX).")
p1 = pl.read_parquet(DRIVE / "returns_panel.parquet")
print("columns:", p1.columns)
print("")
print("-- TITAN, same window, POST repair --")
print(p1.filter(pl.col("symbol") == "TITAN")
        .filter(pl.col("date").is_between(
            pl.date(2011, 6, 20), pl.date(2011, 6, 27)))
        .sort("date")
        .select("date", "isin", "close", "prev_close", "ret", "nifty50_ret"))
print("^ isin now populated (INE280A01010 pre-event). 'ret' here is the")
print("  prev_close-basis return -- LATER PROVEN RAW/UNADJUSTED (Gate 0).")

# ---------------------------------------------------------------------------
hdr("STAGE 3 — NSE corporate-actions file: TITAN's raw CA row")
ca = pl.read_csv(DRIVE / "nse_corporate_actions.csv", infer_schema_length=0)
ca = ca.rename({c: c.strip() for c in ca.columns})
ca_titan = ca.filter(pl.col("SYMBOL").str.strip_chars() == "TITAN")
ca_titan = ca_titan.filter(
    pl.col("EX-DATE").str.strip_chars() == "23-Jun-2011")
print("-- nse_corporate_actions.csv row for TITAN ex-date 23-Jun-2011 --")
print(ca_titan.select("SYMBOL", "PURPOSE", "EX-DATE", "FACE VALUE"))
print("^ this free-text PURPOSE field is the only source of the factor.")
print("  v1 parser read the bonus '1:1' as split face values and DROPPED")
print("  the split component here -- Step-1's systematic-tail bug.")

# ---------------------------------------------------------------------------
hdr("STAGE 4 — ca_adjustment_factors.parquet: TITAN's PARSED + VERIFIED factor")
fac = pl.read_parquet(DRIVE / "ca_adjustment_factors.parquet")
fac_titan = fac.filter((pl.col("symbol") == "TITAN")
                       & (pl.col("ex_date") == pl.date(2011, 6, 23)))
print(fac_titan)
print("^ factor = 10 (split) x 2 (bonus) = 20, confirmed = true")
print("  (obs_ratio from Step-1's independent tape check landed near 20)")

# ---------------------------------------------------------------------------
hdr("STAGE 5 — THE MERGE: factor event -> next traded row (forward as-of)")
print("module5b2's join_asof by symbol, strategy='forward'. Shown here on")
print("just TITAN's event so the join mechanics are visible end to end.")
rows_t = (p1.filter(pl.col("symbol") == "TITAN")
            .select("symbol", "date").unique().sort("date"))
ev_t = fac_titan.select("symbol", "ex_date", "factor").sort("ex_date")
print("")
print("-- LEFT (event) --")
print(ev_t)
mapped_t = ev_t.join_asof(rows_t, left_on="ex_date", right_on="date",
                          by="symbol", strategy="forward")
print("-- RIGHT (TITAN's traded dates, first 3) --")
print(rows_t.head(3))
print("-- RESULT of join_asof (event mapped to its traded row) --")
print(mapped_t)
print("^ ex_date IS a traded day here, so date == ex_date -- no gap-map")
print("  needed for this event. (172 OTHER events in the full run WERE")
print("  gap-mapped: ex-date fell during a trading suspension.)")

# ---------------------------------------------------------------------------
hdr("STAGE 6 — returns_panel_v2.parquet: TITAN's return, RAW vs ADJUSTED")
p2 = pl.read_parquet(DRIVE / "returns_panel_v2.parquet")
print(p2.filter(pl.col("symbol") == "TITAN")
        .filter(pl.col("date").is_between(
            pl.date(2011, 6, 20), pl.date(2011, 6, 27)))
        .sort("date")
        .select("date", "isin", "close", "ret_cc", "adj_factor",
                "ret_adj", "ret_adj_mktadj"))
print("^ 2011-06-23: ret_cc = -0.947 (fake crash, mechanical) becomes")
print("  ret_adj = (1 - 0.947) x 20 - 1 = +0.066 (the REAL move that day)")

# ---------------------------------------------------------------------------
hdr("STAGE 7 — stockday_states_calibrated.parquet: the MODEL's own lineage")
print("Separate pipeline entirely (FII flow data -> HMM -> Module 3), NOT")
print("derived from prices. This is what gets JOINED to Stage 6's panel.")
states = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
print("columns:", states.columns)
print(states.head(5))

# ---------------------------------------------------------------------------
hdr("STAGE 8 — THE G2 MERGE: model labels JOIN price panel")
print("Join key: cisin/TR_DATE (model, canonical ISIN) = isin/date (panel,")
print("bhavcopy raw ISIN). This is the gate that decides whether Stage 7")
print("and Stage 6 can even be compared.")
j = states.join(
    p2.select("isin", "symbol", "date", "ret_adj_mktadj"),
    left_on=["cisin", "TR_DATE"], right_on=["isin", "date"], how="left")
print("")
print("-- 5 MATCHED rows (price found) --")
print(j.filter(pl.col("ret_adj_mktadj").is_not_null())
        .select("cisin", "symbol", "TR_DATE", "archetype",
                "ret_adj_mktadj").head(5))
print("")
print("-- 5 UNMATCHED rows (no same-day price -- the ~9% gap) --")
print(j.filter(pl.col("ret_adj_mktadj").is_null())
        .select("cisin", "TR_DATE", "era", "archetype").head(5))
print("")
match_rate = 100 * float(j["ret_adj_mktadj"].is_not_null().mean())
print("overall match rate:", round(match_rate, 1), "% (matches the G2")
print("gate result from module5a: 90.7%)")

print("")
print("=" * 78)
print("LINEAGE COMPLETE. Chain: bhavcopy(raw) --R1/R2/R3/macro--> panel_v1")
print("--CA parse+verify--> factors --join_asof--> mapped events")
print("--apply+guard--> panel_v2  <--G2 join-- model states (separate")
print("lineage from FII flow data). Bulk/block/short deals: collected,")
print("NOT yet in this chain anywhere.")
