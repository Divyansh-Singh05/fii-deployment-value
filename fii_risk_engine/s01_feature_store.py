# [migrated from Colab: paths now come from paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 1 · FEATURE STORE v2 — 10 flow-only features on the stock-day panel
#
# This is the script that actually produced stockday_features_v2.parquet.
# Supersedes the very first panel-build attempt (which had a 278-row
# null-ISIN phantom-stock bug, fixed here via the ISIN length==12 filter)
# and the initial 3-feature-only version (this adds 7 more).
#
# All memory features: per-stock (.over("cisin")), past-only (.shift(1)),
# row-based 20-day windows (gap-skipping — naturally steps over the masked
# May-Jun 2021 window and any illiquid holes). Same-day features are
# end-of-day knowable. Tail: liquidity floor → within-day rank → probit.
#
# Depends on: yearly raw trade parquets, and isin_pit_map.parquet (from
# isin_pit_map) for POINT-IN-TIME canonical-ISIN keying.
#
# ---------------------------------------------------------------------------
# AUDIT REBUILD (items 1 and 2). Two changes to the research object itself;
# the byte-for-byte pre-audit original is preserved at
# legacy/colab_modules/module1_feature_store_v2.py.
#
# ITEM 2 — identity is now point-in-time. Previously `canon_map` was built
# from the static active_isins.csv / inactive_isins.csv pair, which carry each
# ISIN's EVENTUAL successor with no effective date, and was applied uniformly
# across 2011-2025. That re-keyed 1,819,703 trade rows (7.59% of the tape,
# 593 ISINs) into identities that did not exist on the trade date -- row-
# weighted mean lead time 3,178 days. Identity decides which rows form a
# stock's series, so the hindsight propagated into every trailing window and
# every HMM sequence. Now an as-of join applies only mappings whose corporate
# -action date is on or before the trade date; an unmapped ISIN stays its own
# sequence.
#
# ITEM 1 — the concentration axis is now the documented one. `book_hhi()`
# computed the Herfindahl of each ENTITY's book ACROSS STOCKS, then value-
# weighted it into the stock-day. docs/02_feature_engineering.md:45 specifies
# the Herfindahl ACROSS PARTICIPANTS WITHIN the stock-side-day. Measured on
# 244k sell-side stock-days the two rank-correlate -0.23, and the implemented
# one tracked participant book width (its tails hold participants trading a
# median of 5 vs 81.5 stocks) while participant COUNTS barely moved (4 vs 3).
# Both are now computed:
#     F_entity_conc  within-stock-day participant HHI  <- the documented
#                    measure; ENTERS the HMM subset
#     F_bookfocus    the previous measure, renamed to what it measures;
#                    retained as an auxiliary and robustness axis
# ============================================================================
import polars as pl
import numpy as np
from scipy.special import ndtri
import datetime as dt
from pathlib import Path

# ------------------------------------------------------------- CONFIG -------
parquet_path = ISIN_MAPPING
TRADES_GLOB  = str(parquet_path / "20[0-9][0-9].parquet")
OUT_PANEL    = str(parquet_path / "stockday_features_v2.parquet")

BASE_WIN, MIN_SAMPLES = 20, 15
MIN_TRADES            = 5
COVERAGE_MIN          = 0.50
BAD_START, BAD_END    = dt.date(2021, 5, 1), dt.date(2021, 6, 30)
MISSING               = ["(null)"]

# ── canonical-ISIN map — POINT-IN-TIME (audit item 2) ──────────────────────
# Interval table: one row per (old_isin, [effective_from, effective_to)) with
# the identity in force during that window. Read by a backward as-of join, so
# a trade on day t can only ever see a corporate action dated on or before t.
# No interval covering t -> the ISIN was not yet remapped and stays itself.
from isin_pit_map import OUT as PIT_MAP  # noqa: E402

if not PIT_MAP.exists():
    raise SystemExit(
        f"missing {PIT_MAP.name} — build it first:\n"
        f"    python -m isin_pit_map")
pit = (pl.read_parquet(PIT_MAP)
         .select(pl.col("old_isin").alias("ISIN"), "effective_from",
                 "canonical_isin", "confidence", "chain_depth")
         .sort("effective_from"))
print(f"[pit] identity map: {pit.height:,} intervals over "
      f"{pit['ISIN'].n_unique():,} ISINs")

# ── per-format FII entity id (early: strip trailing YYYYMM; late: raw id) ──
# See entity audit findings — IDs re-mint ~monthly, no cross-month identity;
# this decodes the two ID-format eras but the resulting id is ONLY valid
# within a single month.
def entity_id(col):
    raw = pl.col(col); L = raw.str.len_chars()
    return (pl.when(raw.is_null() | raw.is_in(MISSING)).then(None)
              .when(L.is_in([17, 18, 19])).then(raw.str.replace(r"\d{6}$", ""))
              .otherwise(raw).alias("eid"))

# ------------------------------------------------------------- LOAD ---------
lf = (pl.scan_parquet(TRADES_GLOB)
      .with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))        # cat→str: avoids cross-file cache error
      .filter(pl.col("TR_TYPE").is_in([1, 4]) & (pl.col("RATE") > 0) &
              (pl.col("RFDE_INSTR_TYPE") == "REG_DL_INSTR_EQ"))
      .filter(pl.col("ISIN").is_not_null() & (pl.col("ISIN").str.len_chars() == 12))  # drop 278 malformed rows
      .filter(~pl.col("TR_DATE").is_between(BAD_START, BAD_END))    # mask 2021 gap before any window
      # POINT-IN-TIME identity: backward as-of on the corporate-action date.
      # A row dated t sees only mappings effective at or before t; anything
      # unmapped as of t keeps its own ISIN as its own sequence.
      .sort("TR_DATE")
      .join_asof(pit.lazy(), left_on="TR_DATE", right_on="effective_from",
                 by="ISIN", strategy="backward")
      .with_columns(pl.coalesce(["canonical_isin", "ISIN"]).alias("cisin"),
                    entity_id("FII"))
      .with_columns(
          pl.when(pl.col("TR_TYPE") == 1).then(pl.col("VALUE_INR")).otherwise(0.0).alias("buy_val"),
          pl.when(pl.col("TR_TYPE") == 4).then(pl.col("VALUE_INR")).otherwise(0.0).alias("sell_val"),
      ))

# ------------------------------------------------------------- BASE PANEL (+ per-trade stats)
panel = (
    lf.group_by(["cisin", "TR_DATE"])
      .agg(pl.col("buy_val").sum().alias("buy_value"),
           pl.col("sell_val").sum().alias("sell_value"),
           (pl.col("TR_TYPE") == 1).sum().alias("n_buys"),
           (pl.col("TR_TYPE") == 4).sum().alias("n_sells"),
           pl.col("QUANTITY").sum().alias("total_qty"),
           pl.col("VALUE_INR").std().alias("trade_size_std"),
           pl.col("eid").filter(pl.col("eid").is_not_null()).n_unique().alias("n_entities"),
           pl.col("VALUE_INR").filter(pl.col("eid").is_not_null()).sum().alias("valid_gross"))
      .collect()
      .with_columns(pl.col("buy_value").fill_null(0.0), pl.col("sell_value").fill_null(0.0))
      .with_columns((pl.col("buy_value") - pl.col("sell_value")).alias("NET"),
                    (pl.col("buy_value") + pl.col("sell_value")).alias("GROSS"),
                    (pl.col("n_buys") + pl.col("n_sells")).alias("N"))
      .with_columns((pl.col("GROSS") / pl.col("N")).alias("mean_trade_size"),
                    (pl.col("valid_gross") / pl.col("GROSS")).alias("id_coverage"))
      .sort(["cisin", "TR_DATE"])
)
print(f"Base panel: {panel.height:,} stock-days | {panel['cisin'].n_unique():,} stocks | "
      f"{panel['TR_DATE'].min()} → {panel['TR_DATE'].max()}")

# ------------------------------------------------------------- ENTITY HHI — SELL side & BUY side
def side_concentration(tr_type, val_col, conc_name, focus_name, cov_name):
    """Both concentration measures for one side of the book, in one pass.

    conc  (audit item 1, the DOCUMENTED measure — enters the HMM)
              HHI across PARTICIPANTS within this stock-side-day:
                  sum_e ( v_{e,i,t} / sum_e' v_{e',i,t} ) ^ 2
              High = a few entities account for the day's flow in this name.
              Low  = the flow is spread across many entities.
              This is what "concentrated vs dispersed" has always meant in
              the write-ups, and it is what the archetype names claim.

    focus (the PREVIOUS measure, renamed to what it actually measures)
              value-weighted mean, over the participants, of each entity's
              OWN book concentration across the stocks it traded that day:
                  hhi_e,t  = sum_j ( v_{e,j,t} / sum_j v_{e,j,t} ) ^ 2
                  focus    = sum_e v_{e,i,t} * hhi_e,t / sum_e v_{e,i,t}
              Since hhi_e ~ 1 / (stocks e traded that day) — correlation 0.94,
              and 60.7% of entity-days are single-stock and take 1.0 exactly —
              this is a BASKET-vs-FOCUSED measure of who is trading, not a
              measure of how concentrated the trading is. Kept as an auxiliary
              because it is a coherent construct in its own right and because
              the pre-audit results were built on it.

    Both share the same attribution-coverage denominator, so the existing
    COVERAGE_MIN gate applies unchanged to each.
    """
    side = (lf.filter((pl.col("TR_TYPE") == tr_type) & pl.col("eid").is_not_null())
              .select("cisin", "TR_DATE", "eid", pl.col(val_col).alias("v0")))
    # entity x stock x day value — the shared building block
    en  = side.group_by(["eid", "TR_DATE", "cisin"]).agg(pl.col("v0").sum().alias("v"))

    # --- documented measure: shares taken WITHIN the stock-day -------------
    conc = (en.group_by(["cisin", "TR_DATE"])
              .agg((pl.col("v") ** 2).sum().alias("_sq"),
                   pl.col("v").sum().alias(cov_name))
              .with_columns((pl.col("_sq") / pl.col(cov_name) ** 2).alias(conc_name))
              .drop("_sq"))

    # --- previous measure: shares taken WITHIN THE ENTITY'S OWN BOOK -------
    et  = en.group_by(["eid", "TR_DATE"]).agg(pl.col("v").sum().alias("tot"))
    eh  = (en.join(et, on=["eid", "TR_DATE"])
             .with_columns((pl.col("v") / pl.col("tot")).alias("s"))
             .group_by(["eid", "TR_DATE"]).agg((pl.col("s") ** 2).sum().alias("hhi")))
    focus = (en.join(eh, on=["eid", "TR_DATE"])
               .group_by(["cisin", "TR_DATE"])
               .agg((pl.col("v") * pl.col("hhi")).sum().alias("wsum"),
                    pl.col("v").sum().alias("_val"))
               .with_columns((pl.col("wsum") / pl.col("_val")).alias(focus_name))
               .drop(["wsum", "_val"]))

    return conc.join(focus, on=["cisin", "TR_DATE"], how="left").collect()

hhi_sell = side_concentration(4, "sell_val", "entity_conc_raw",
                              "bookfocus_raw",     "valid_sell")
hhi_buy  = side_concentration(1, "buy_val",  "entity_conc_buy_raw",
                              "bookfocus_buy_raw", "valid_buy")

panel = (panel
    .join(hhi_sell, on=["cisin", "TR_DATE"], how="left")
    .join(hhi_buy,  on=["cisin", "TR_DATE"], how="left")
    .with_columns((pl.col("valid_sell") / pl.col("sell_value")).alias("a3_cov_sell"),
                  (pl.col("valid_buy")  / pl.col("buy_value")).alias("a3_cov_buy"))
    .with_columns(
        pl.when(pl.col("a3_cov_sell") >= COVERAGE_MIN).then(pl.col("entity_conc_raw")).otherwise(None).alias("entity_conc_raw"),
        pl.when(pl.col("a3_cov_buy")  >= COVERAGE_MIN).then(pl.col("entity_conc_buy_raw")).otherwise(None).alias("entity_conc_buy_raw"),
        pl.when(pl.col("a3_cov_sell") >= COVERAGE_MIN).then(pl.col("bookfocus_raw")).otherwise(None).alias("bookfocus_raw"),
        pl.when(pl.col("a3_cov_buy")  >= COVERAGE_MIN).then(pl.col("bookfocus_buy_raw")).otherwise(None).alias("bookfocus_buy_raw"),
        pl.when(pl.col("id_coverage") >= COVERAGE_MIN).then(pl.col("n_entities")).otherwise(None).alias("n_entities"),
    ))

# ------------------------------------------------------------- SAME-DAY RAWS
panel = panel.with_columns(
    (pl.col("NET") / pl.col("GROSS")).alias("imbalance_raw"),
    (pl.col("trade_size_std") / pl.col("mean_trade_size")).alias("size_disp_raw"),
    pl.col("n_entities").cast(pl.Float64).alias("breadth_raw"),
)

# ------------------------------------------------------------- MARKET AGGREGATE (for flow beta)
mkt = (panel.group_by("TR_DATE")
            .agg((pl.col("NET").sum() / pl.col("GROSS").sum()).alias("mkt_imb")))
panel = panel.join(mkt, on="TR_DATE", how="left").sort(["cisin", "TR_DATE"])

# ------------------------------------------------------------- TRAILING RAWS (per-stock, past-only, row-based)
W, MS = BASE_WIN, MIN_SAMPLES
x, y = pl.col("imbalance_raw"), pl.col("mkt_imb")
panel = panel.with_columns(
    (pl.col("NET").sign().rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("pers_signed"),
    ((pl.col("NET").abs() / pl.col("GROSS")).rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("intensity"),
    (pl.col("mean_trade_size").rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("mts_base"),
    (pl.col("GROSS").rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("gross_base"),
    ((x * y).rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("_mxy"),
    (x.rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("_mx"),
    (y.rolling_mean(W, min_samples=MS).shift(1)).over("cisin").alias("_my"),
    (x.rolling_std(W, min_samples=MS).shift(1)).over("cisin").alias("_sx"),
    (y.rolling_std(W, min_samples=MS).shift(1)).over("cisin").alias("_sy"),
)

# streak of same-sign NET as of YESTERDAY — two separate .over() calls
# (nesting one window inside another throws "window expression not allowed
# in aggregation" in polars; must materialize the inner window first)
panel = (panel
    .with_columns(pl.col("NET").sign().alias("_sgn"))
    .with_columns(((pl.col("_sgn") != pl.col("_sgn").shift(1)).fill_null(True))
                  .cum_sum().over("cisin").alias("_run"))
    .with_columns(((pl.int_range(0, pl.len()).over(["cisin", "_run"]) + 1)
                   * pl.col("_sgn")).alias("_streak_now"))
    .with_columns(pl.col("_streak_now").shift(1).over("cisin").alias("streak_raw"))
    .drop(["_sgn", "_run", "_streak_now"]))

panel = panel.with_columns(
    (pl.col("pers_signed") * pl.col("intensity")).alias("persistence_raw"),
    (pl.col("mean_trade_size") / pl.col("mts_base")).alias("blockiness_raw"),
    (pl.col("GROSS") / pl.col("gross_base")).alias("activity_raw"),
    pl.when((pl.col("_sx") > 0) & (pl.col("_sy") > 0))
      .then((pl.col("_mxy") - pl.col("_mx") * pl.col("_my")) / (pl.col("_sx") * pl.col("_sy")))
      .otherwise(None).alias("flowbeta_raw"),
).drop(["_mxy", "_mx", "_my", "_sx", "_sy"])

# ------------------------------------------------------------- FLOOR + WITHIN-DAY PROBIT (all 10)
panel = panel.with_columns((pl.col("N") >= MIN_TRADES).alias("eligible"))

FEATS = {  # raw column            → model feature
    "persistence_raw":    "F_persist",    # Axis 1: sustained direction (Robot low)
    "blockiness_raw":     "F_block",      # Axis 2: size surprise (Shark high)
    # Axis 3 — participant concentration WITHIN the stock-side-day (item 1).
    # High = few entities carry the day's flow; low = spread across many.
    "entity_conc_raw":     "F_entity_conc",      # sell side — ENTERS the HMM
    "entity_conc_buy_raw": "F_entity_conc_buy",  # buy side  — ENTERS the HMM
    # Auxiliary — the pre-audit measure, renamed to what it measures: are the
    # participants running broad baskets or focused books today?
    "bookfocus_raw":       "F_bookfocus",
    "bookfocus_buy_raw":   "F_bookfocus_buy",
    "breadth_raw":        "F_breadth",    # crowdedness (Robot high)
    "size_disp_raw":      "F_sizedisp",   # lumpy vs uniform slicing (Shark high, Robot low)
    "flowbeta_raw":       "F_flowbeta",   # co-move with aggregate tide (Robot high)
    "activity_raw":       "F_activity",   # activity surprise (regime onset) — NOTE: r=0.88 w/ F_block
    "streak_raw":         "F_streak",     # signed run length as of yesterday — NOTE: r=0.56 w/ F_persist
    "imbalance_raw":      "F_imbal",      # today's direction
}
for raw, feat in FEATS.items():
    masked = pl.when(pl.col("eligible")).then(pl.col(raw)).otherwise(None)
    panel = panel.with_columns(masked.alias("_m"))
    n_valid = pl.col("_m").is_not_null().sum().over("TR_DATE")
    rnk     = pl.col("_m").rank(method="average").over("TR_DATE")
    panel = panel.with_columns((rnk / (n_valid + 1)).alias("_p"))
    p = panel["_p"].to_numpy()
    pr = np.full(p.shape, np.nan); mm = ~np.isnan(p); pr[mm] = ndtri(p[mm])
    panel = panel.with_columns(pl.Series(feat, pr).fill_nan(None)).drop(["_m", "_p"])  # NaN→null: else warm-up masquerades as present

# ------------------------------------------------------------- DIAGNOSTICS + SAVE
core3 = ["F_persist", "F_block", "F_entity_conc"]
all10 = list(FEATS.values())   # now 12: the audit added F_entity_conc_buy
                              # (documented axis) and split out F_bookfocus*
elig  = panel.filter(pl.col("eligible")).height
comp3 = panel.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in core3]))
comp10= panel.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in all10]))
print(f"\nEligible                    : {elig:,}")
print(f"Complete on core 3          : {comp3.height:,}  ({100*comp3.height/elig:.1f}%)")
print(f"Complete on all {len(all10)}          : {comp10.height:,}  ({100*comp10.height/elig:.1f}%)")

print("\n=== NULL RATE per feature (eligible rows) ===")
print(panel.filter(pl.col("eligible"))
           .select([pl.col(f).is_null().mean().alias(f) for f in all10]))

print(f"\n=== {len(all10)}x{len(all10)} CORRELATION (complete cases) — flag |r| > 0.6 ===")
cm = comp10.select(all10).corr()
print(cm)
arr = cm.to_numpy()
for i in range(len(all10)):
    for j in range(i + 1, len(all10)):
        if abs(arr[i, j]) > 0.6:
            print(f"  REDUNDANCY FLAG: {all10[i]} × {all10[j]} = {arr[i, j]:.2f}")

print("\n=== SUMMARY (each ~N(0,1)) ===")
print(comp10.select(all10).describe())

# ---- AUDIT CHECK (item 1): the two concentration measures are NOT the same
_c = panel.filter(pl.col("F_entity_conc").is_not_null()
                  & pl.col("F_bookfocus").is_not_null())
if _c.height:
    _r = _c.select(pl.corr("F_entity_conc", "F_bookfocus")).item()
    _rb = _c.filter(pl.col("F_breadth").is_not_null()).select(
        pl.corr("F_entity_conc", "F_breadth")).item()
    print(f"\n=== AUDIT ITEM 1 — concentration vs book-focus ===")
    print(f"  corr(F_entity_conc, F_bookfocus) = {_r:+.4f}   "
          f"(pre-audit measure vs documented measure)")
    print(f"  corr(F_entity_conc, F_breadth)   = {_rb:+.4f}   "
          f"(must be strongly NEGATIVE: more participants = less concentrated)")
    print(f"  VERDICT: {'PASS - conc behaves as documented' if _rb < -0.5 else 'FAIL - check construction'}")

panel.write_parquet(OUT_PANEL)
print(f"\nSaved → {OUT_PANEL}   {panel.shape}")

# RESULT (as run):
#   Base panel: 2,423,212 stock-days | 3,812 stocks
#   Eligible: 1,151,518 | complete-on-3: 929,195 (80.7%) | complete-on-10: 788,312 (68.5%)
#   REDUNDANCY FLAG: F_block × F_activity = 0.88 (structural — gross surprise
#     arrives via bigger tickets, not more trades; keep both, never together in a model)
#   F_streak × F_persist = 0.56 (expected — two encodings of persistence)
#   All other |r| < 0.36, economically coherent signs (e.g. F_entity × F_imbal
#     = -0.27: dispersed-seller days are sell-imbalanced — the Hostage corner)
#   All 10 features ~N(0,1) post-probit (mean≈0, std≈0.93-0.99); F_imbal has
#     compressed tails (std 0.83) from tie-mass at exactly ±1 (all-buy/all-sell days)
#   → HMM subset used downstream: F_persist, F_block, F_entity, F_entity_buy
