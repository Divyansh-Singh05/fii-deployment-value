# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 12B · BASELINE STRATEGIES (NO HMM) — locked before HMM variants
#
# Three respectable daily long-short strategies on the model universe
# (939 canonical names, CA-adjusted returns_panel_v3). Signal at close t,
# executed with a 1-day lag (trade close t+1, earn day t+2). Costs 15 bps
# one-way on turnover (0 and 30 bps also reported in the summary).
#
#   S1 REV20   classic 20d loser reversal: alpha = -sum(mktadj ret, 20d)
#   S2 FLOW10  FII flow-following: alpha = 10d sum of buy/sell imbalance
#   S3 PROXY   mechanical no-HMM analog of the regime idea:
#              LONG  climax-reversal proxy (20d loser, relvol>1.1,
#                    5d FII net selling)  [SHARK_DIST proxy]
#              SHORT quiet-persistent-seller proxy (10d net selling,
#                    relvol<0.9, 20d loser) [HOSTAGE proxy]
#              signal smoothed over 10d (event hold with decay)
#
# Eras: TRAIN <= 2021-04-30, TEST >= 2021-07-01 (May-Jun 2021 masked).
# PRE-REGISTERED here: nothing is tuned after seeing TEST-era output;
# these baselines are RECORDED, then frozen, then compared in 12C.
# Sanity gates: coverage / finiteness / turnover only (not performance).
#
# Requires: module12a run in this session (engine gates PASSED).
# Writes: bt12_baselines.parquet (daily gross pnl + turnover per strategy).
# ============================================================================
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from fii.backtest.engine import *  # noqa: F401,F403

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
LAG, TCOST, MAXW = 1, 15.0, 0.05

# ---- model universe + price grid --------------------------------------------
st = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
uni = st["cisin"].unique().to_list()
print("model universe cisins:", len(uni))

pan = pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
print("panel columns:", pan.columns)
rcol = "ret_adj" if "ret_adj" in pan.columns else "ret_adj_mktadj"
print("pnl return column:", rcol,
      "(mktadj fallback is fine for a $-neutral book)" if
      rcol != "ret_adj" else "")
pan = (pan.filter(pl.col("isin").is_in(uni))
          .select("isin", "date", rcol, "ret_adj_mktadj", "volume")
          .sort(["isin", "date"]))
pdfp = pan.to_pandas()

RET = pdfp.pivot(index="date", columns="isin", values=rcol)
RET = RET.clip(-0.5, 0.5)
MAD = pdfp.pivot(index="date", columns="isin",
                 values="ret_adj_mktadj").clip(-0.5, 0.5)
VOL = pdfp.pivot(index="date", columns="isin", values="volume")
dates = RET.index
print("grid:", RET.shape[0], "days x", RET.shape[1], "stocks,",
      str(dates[0].date()), "->", str(dates[-1].date()))

# ---- FII flow imbalance matrix ----------------------------------------------
f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
print("features columns:", f.columns)
vb = next((c for c in ["buy_val", "val_buy", "buy_value", "BUY_VAL"]
           if c in f.columns), None)
vs = next((c for c in ["sell_val", "val_sell", "sell_value", "SELL_VAL"]
           if c in f.columns), None)
if vb and vs:
    print("imbalance from VALUES:", vb, vs)
    f = f.with_columns(((pl.col(vb) - pl.col(vs))
                        / (pl.col(vb) + pl.col(vs) + 1e-9)).alias("imb"))
else:
    print("imbalance from COUNTS: n_buys / n_sells (value cols absent)")
    f = f.with_columns(((pl.col("n_buys") - pl.col("n_sells"))
                        / (pl.col("n_buys") + pl.col("n_sells") + 1e-9))
                       .alias("imb"))
fp = f.select("cisin", "TR_DATE", "imb").to_pandas()
IMB = (fp.pivot(index="TR_DATE", columns="cisin", values="imb")
         .reindex(index=dates, columns=RET.columns))
IMB = IMB.fillna(0.0)                     # no FII trade that day = 0 flow

# ---- signals (all use info through close t only) -----------------------------
inuni = RET.notna()                       # tradable that day
l20 = MAD.fillna(0.0).rolling(20, min_periods=15).sum()
rv = VOL / VOL.rolling(20, min_periods=10).mean()
f5 = IMB.rolling(5, min_periods=3).sum()
f10 = IMB.rolling(10, min_periods=5).sum()

A1 = (-l20).where(inuni)                                       # S1 REV20
A2 = f10.where(inuni)                                          # S2 FLOW10
loser = l20.rank(axis=1, pct=True) < 0.2
sig = (loser & (rv > 1.1) & (f5 < 0)).astype(float) \
    - ((f10 < 0) & (rv < 0.9) & (l20 < 0)).astype(float)
A3 = sig.rolling(10, min_periods=1).mean().where(inuni)        # S3 PROXY

# ---- eras --------------------------------------------------------------------
d = pd.Series(dates)
era = np.where(d <= pd.Timestamp("2021-04-30"), "TRAIN",
               np.where(d >= pd.Timestamp("2021-07-01"), "TEST", "MASK"))
Rm = RET.to_numpy()

out = []
print("\n" + "=" * 70)
for name, A in [("S1_REV20", A1), ("S2_FLOW10", A2), ("S3_PROXY", A3)]:
    Am = A.to_numpy()
    cov = np.isfinite(Am).sum(axis=1)
    bt = backtest(Am, Rm, lag=LAG, tcost_bps=TCOST, maxw=MAXW)
    ic, ict, icn = daily_ic(Am, MAD.to_numpy(), lag=LAG)
    print(f"\n{name}  (lag={LAG}, cost={TCOST}bps, maxw={MAXW}) "
          f"IC {ic:+.4f} (t={ict:+.1f}, {icn}d)")
    for e in ("TRAIN", "TEST"):
        m = era == e
        show(f"{e:5s} net", metrics(bt["pnl_net"][m], bt["turnover"][m]))
        show(f"{e:5s} gross", metrics(bt["pnl_gross"][m],
                                      bt["turnover"][m]))
    # sanity gates
    g_cov = np.median(cov[cov > 0]) >= (200 if name != "S3_PROXY" else 20)
    g_fin = np.all(np.isfinite(bt["pnl_net"]))
    g_to = bt["turnover"].mean() < 2.0
    print(f"  gates: coverage {'PASS' if g_cov else 'FAIL'} "
          f"(median {np.median(cov[cov>0]):.0f}/d) | finite "
          f"{'PASS' if g_fin else 'FAIL'} | turnover "
          f"{'PASS' if g_to else 'FAIL'} ({bt['turnover'].mean():.2f}/d)")
    out.append(pd.DataFrame({"date": dates, "strat": name, "era": era,
                             "pnl_gross": bt["pnl_gross"],
                             "turnover": bt["turnover"]}))

res = pl.from_pandas(pd.concat(out, ignore_index=True))
res.write_parquet(DRIVE / "bt12_baselines.parquet")
print("\nwrote bt12_baselines.parquet", res.shape)
print("=" * 70)
print("VERDICT: BASELINES RECORDED & FROZEN (if all sanity gates PASS).")
print("These numbers may not be revisited after 12C. Next: 12C (HMM).")
print("=" * 70)
