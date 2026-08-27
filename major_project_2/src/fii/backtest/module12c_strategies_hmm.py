# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 12C · HMM-CONDITIONED STRATEGIES + WITH/WITHOUT COMPARISON
#
# Each 12B baseline gets an HMM twin (same engine, lag, costs, universe):
#   S1H  REV20 gated: reversal magnitude kept ONLY on names within 20d
#        after a SHARK_DIST end (long side) / SHARK_ACC end (short side)
#   S2H  FLOW10 filtered: flow signal zeroed on stock-days currently in a
#        concentrated (transitory) regime SHARK_DIST/SHARK_ACC
#   S3H  pure regime book: long 20d after SHARK_DIST end, short 20d after
#        SHARK_ACC end, short while currently HOSTAGE
# Real-time discipline: an episode END is only knowable at the close of the
# first day AFTER the run -> end impulse at t requires arch[t-1]=X, arch[t]!=X.
#
# PRE-REGISTERED VERDICT (per pair, decided on TEST era, net of 15bps):
#   dSharpe = Sharpe(HMM) - Sharpe(base) on paired daily net pnl;
#   moving-block bootstrap (block=20d, B=2000) 95% CI on dSharpe.
#   CI entirely > 0  -> "HMM ADDS VALUE"
#   CI entirely < 0  -> "HMM SUBTRACTS"
#   else             -> "INCONCLUSIVE AT THIS POWER"
#
# Requires: module12a run in this session; 12B already recorded
# (reads bt12_baselines.parquet). Self-contained data prep otherwise.
# ============================================================================
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from fii.backtest.engine import *  # noqa: F401,F403

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
LAG, TCOST, MAXW = 1, 15.0, 0.05

# ---- shared prep (identical to 12B) ------------------------------------------
st = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
uni = st["cisin"].unique().to_list()
pan = pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
rcol = "ret_adj" if "ret_adj" in pan.columns else "ret_adj_mktadj"
pan = (pan.filter(pl.col("isin").is_in(uni))
          .select("isin", "date", rcol, "ret_adj_mktadj", "volume")
          .sort(["isin", "date"]))
pdfp = pan.to_pandas()
RET = pdfp.pivot(index="date", columns="isin", values=rcol).clip(-.5, .5)
MAD = pdfp.pivot(index="date", columns="isin",
                 values="ret_adj_mktadj").clip(-.5, .5)
dates = RET.index
f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
vb = next((c for c in ["buy_val", "val_buy", "buy_value"]
           if c in f.columns), None)
vs = next((c for c in ["sell_val", "val_sell", "sell_value"]
           if c in f.columns), None)
if vb and vs:
    f = f.with_columns(((pl.col(vb) - pl.col(vs))
                        / (pl.col(vb) + pl.col(vs) + 1e-9)).alias("imb"))
else:
    f = f.with_columns(((pl.col("n_buys") - pl.col("n_sells"))
                        / (pl.col("n_buys") + pl.col("n_sells") + 1e-9))
                       .alias("imb"))
IMB = (f.select("cisin", "TR_DATE", "imb").to_pandas()
        .pivot(index="TR_DATE", columns="cisin", values="imb")
        .reindex(index=dates, columns=RET.columns).fillna(0.0))
inuni = RET.notna()
l20 = MAD.fillna(0.0).rolling(20, min_periods=15).sum()
f10 = IMB.rolling(10, min_periods=5).sum()

# ---- archetype day-masks on the price grid -----------------------------------
sp = (st.select("cisin", "TR_DATE", "archetype").to_pandas()
        .pivot(index="TR_DATE", columns="cisin", values="archetype")
        .reindex(index=dates, columns=RET.columns))
print("archetype grid days x stocks:", sp.shape,
      "| labeled cells:", int(sp.notna().to_numpy().sum()))


def endsig(mask, hold=20):
    """real-time end impulse: in-state yesterday, not today; held 'hold'd."""
    m = mask.fillna(False)
    imp = (m.shift(1, fill_value=False) & ~m).astype(float)
    return imp.rolling(hold, min_periods=1).mean()

SD, SA = (sp == "SHARK_DIST"), (sp == "SHARK_ACC")
HO = (sp == "HOSTAGE")
sd_end, sa_end = endsig(SD), endsig(SA)
inconc = (SD | SA).fillna(False)

# ---- HMM variants -------------------------------------------------------------
A1H = ((-l20).where(sd_end.gt(0) | sa_end.gt(0))
       .where(inuni))                                     # gated reversal
A2H = f10.mask(inconc, 0.0).where(inuni)                  # filtered flow
A3H = (sd_end - sa_end - HO.fillna(False).astype(float)) \
    .where(inuni)                                         # pure regime book

d = pd.Series(dates)
era = np.where(d <= pd.Timestamp("2021-04-30"), "TRAIN",
               np.where(d >= pd.Timestamp("2021-07-01"), "TEST", "MASK"))
Rm = RET.to_numpy()

base = (pl.read_parquet(DRIVE / "bt12_baselines.parquet").to_pandas()
          .assign(date=lambda x: pd.to_datetime(x["date"])))


def block_boot_dsharpe(x, y, block=20, B=2000, seed=11):
    """95% CI of Sharpe(x)-Sharpe(y) via paired moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = int(np.ceil(n / block))
    ds = np.empty(B)
    for b in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in
                              rng.integers(0, n, k)])[:n]
        xs, ys = x[idx], y[idx]
        sx = xs.mean() / xs.std(ddof=1) * np.sqrt(252)
        sy = ys.mean() / ys.std(ddof=1) * np.sqrt(252)
        ds[b] = sx - sy
    return np.percentile(ds, [2.5, 97.5])

pairs = [("S1_REV20", "S1H_gated_reversal", A1H),
         ("S2_FLOW10", "S2H_filtered_flow", A2H),
         ("S3_PROXY", "S3H_regime_book", A3H)]
rows, hml = [], []
print("\n" + "=" * 70)
for bname, hname, A in pairs:
    bt = backtest(A.to_numpy(), Rm, lag=LAG, tcost_bps=TCOST, maxw=MAXW)
    print(f"\n{hname}  vs  {bname}")
    for e in ("TRAIN", "TEST"):
        m = era == e
        show(f"{e:5s} HMM  net", metrics(bt["pnl_net"][m],
                                         bt["turnover"][m]))
        bb = base[(base["strat"] == bname) & (base["era"] == e)]
        bnet = (bb["pnl_gross"] - TCOST * 1e-4 * bb["turnover"]).to_numpy()
        show(f"{e:5s} base net", metrics(bnet, bb["turnover"].to_numpy()))
        lo, hi = block_boot_dsharpe(bt["pnl_net"][m], bnet)
        verdict = ("HMM ADDS VALUE" if lo > 0 else
                   "HMM SUBTRACTS" if hi < 0 else
                   "INCONCLUSIVE AT THIS POWER")
        print(f"  {e} dSharpe 95% CI [{lo:+.2f}, {hi:+.2f}] -> {verdict}"
              + ("   << DECISIVE ERA" if e == "TEST" else ""))
        rows.append((hname, e, lo, hi, verdict))
    hml.append(pd.DataFrame({"date": dates, "strat": hname, "era": era,
                             "pnl_gross": bt["pnl_gross"],
                             "turnover": bt["turnover"]}))

hres = pl.from_pandas(pd.concat(hml, ignore_index=True))
hres.write_parquet(DRIVE / "bt12_hmm.parquet")
print("\nwrote bt12_hmm.parquet", hres.shape)

print("\n" + "=" * 70)
print("SUMMARY (pre-registered: TEST-era net-of-cost CI decides)")
for hname, e, lo, hi, v in rows:
    if e == "TEST":
        print(f"  {hname:22s} dSharpe CI [{lo:+.2f},{hi:+.2f}]  {v}")
print("=" * 70)
