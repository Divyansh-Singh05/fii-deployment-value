# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 12E · S4 STYLE-SWITCH — trend in some regimes, reversion in others
#
# User-specified design: trend-follow certain regimes, mean-revert others,
# hold (flat) the rest. The model-faithful "arc" book (Module 6B):
#
#   S4H (HMM states):    in SHARK_DIST -> short (ride pressure)
#                        20d after SHARK_DIST end -> long (reversal)
#                        in SHARK_ACC  -> long  (ride pressure)
#                        20d after SHARK_ACC end -> short (give-back)
#                        in HOSTAGE    -> short (permanent info decline)
#                        ROBOT/UNTAGGED -> flat
#   S4B (no HMM twin, same raw ingredients, mechanical rules):
#                        climax-sell proxy (5d net sell & relvol>1.1) -> short
#                        that proxy just switched off -> long 20d
#                        climax-buy proxy -> long; switched off -> short 20d
#                        quiet-seller proxy (10d net sell & relvol<0.9)-> short
#                        else flat
#
# Scale fix vs S3H (disclosed): S3H's end-legs used a decaying-mean hold
# (~0.05 magnitude) against a 1.0-magnitude HOSTAGE leg, so it was
# effectively a HOSTAGE-short book. Here ALL legs use equal-magnitude
# sustained holds (20d rolling max), summed and clipped to [-1, +1].
#
# Real-time discipline unchanged: state of day t known at close t; an
# episode END is knowable only at the close of the first day after the
# run; lag=1 execution; costs 15bps; both books run once, no re-tuning.
#
# PRE-REGISTERED VERDICT (before results): TEST era, net of 15bps,
# paired dSharpe (S4H - S4B), 20d-block bootstrap 95% CI:
#   CI>0 HMM ADDS VALUE | CI<0 HMM SUBTRACTS | else INCONCLUSIVE.
# Gross dSharpe CI + breakeven costs reported as diagnostics (12D-style).
#
# Requires module12a run in this session. Writes bt12_style.parquet.
# ============================================================================
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from fii.backtest.engine import *  # noqa: F401,F403

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING
LAG, TCOST, MAXW, HOLD = 1, 15.0, 0.05, 20

# ---- shared prep (identical to 12B/12C) --------------------------------------
st = pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
uni = st["cisin"].unique().to_list()
pan = pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
pan = (pan.filter(pl.col("isin").is_in(uni))
          .select("isin", "date", "ret_adj", "ret_adj_mktadj", "volume")
          .sort(["isin", "date"]))
pdfp = pan.to_pandas()
RET = pdfp.pivot(index="date", columns="isin",
                 values="ret_adj").clip(-.5, .5)
dates = RET.index
VOL = pdfp.pivot(index="date", columns="isin", values="volume")
f = pl.read_parquet(MODELD / "stockday_features_v2.parquet")
f = f.with_columns(((pl.col("buy_value") - pl.col("sell_value"))
                    / (pl.col("buy_value") + pl.col("sell_value") + 1e-9))
                   .alias("imb"))
IMB = (f.select("cisin", "TR_DATE", "imb").to_pandas()
        .pivot(index="TR_DATE", columns="cisin", values="imb")
        .reindex(index=dates, columns=RET.columns).fillna(0.0))
inuni = RET.notna()
rv = VOL / VOL.rolling(20, min_periods=10).mean()
f5 = IMB.rolling(5, min_periods=3).sum()
f10 = IMB.rolling(10, min_periods=5).sum()
sp = (st.select("cisin", "TR_DATE", "archetype").to_pandas()
        .pivot(index="TR_DATE", columns="cisin", values="archetype")
        .reindex(index=dates, columns=RET.columns))


def endhold(mask, hold=HOLD):
    """1.0 for 'hold' days after a run of True ends (real-time legal)."""
    m = mask.fillna(False)
    imp = (m.shift(1, fill_value=False) & ~m).astype(float)
    return imp.rolling(hold, min_periods=1).max()

SD, SA = (sp == "SHARK_DIST"), (sp == "SHARK_ACC")
HO = (sp == "HOSTAGE")
A4H = ((-SD.fillna(False).astype(float))          # trend: ride sell pressure
       + SA.fillna(False).astype(float)           # trend: ride buy pressure
       - HO.fillna(False).astype(float)           # trend: permanent decline
       + endhold(SD)                              # reversion: post-SD long
       - endhold(SA))                             # reversion: post-SA short
A4H = A4H.clip(-1, 1).where(inuni)

csell = (f5 < 0) & (rv > 1.1)
cbuy = (f5 > 0) & (rv > 1.1)
quiet = (f10 < 0) & (rv < 0.9)
A4B = ((-csell.astype(float)) + cbuy.astype(float)
       - quiet.astype(float)
       + endhold(csell) - endhold(cbuy))
A4B = A4B.clip(-1, 1).where(inuni)

d = pd.Series(dates)
era = np.where(d <= pd.Timestamp("2021-04-30"), "TRAIN",
               np.where(d >= pd.Timestamp("2021-07-01"), "TEST", "MASK"))
Rm = RET.to_numpy()


def boot(fn, args, block=20, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    n = len(args[0])
    k = int(np.ceil(n / block))
    v = np.empty(B)
    for b in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in
                              rng.integers(0, n, k)])[:n]
        v[b] = fn(*[a[idx] for a in args])
    return np.percentile(v, [2.5, 97.5])


def shp(x):
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(252) if sd > 0 else np.nan

print("=" * 70)
print("MODULE 12E · S4 style-switch (trend/reversion/hold by regime)")
print("=" * 70)
bH = backtest(A4H.to_numpy(), Rm, lag=LAG, tcost_bps=TCOST, maxw=MAXW)
bB = backtest(A4B.to_numpy(), Rm, lag=LAG, tcost_bps=TCOST, maxw=MAXW)
out = []
for e in ("TRAIN", "TEST"):
    m = era == e
    print(f"\n--- {e} ---")
    show("S4H net  ", metrics(bH["pnl_net"][m], bH["turnover"][m]))
    show("S4H gross", metrics(bH["pnl_gross"][m], bH["turnover"][m]))
    show("S4B net  ", metrics(bB["pnl_net"][m], bB["turnover"][m]))
    show("S4B gross", metrics(bB["pnl_gross"][m], bB["turnover"][m]))
    lo, hi = boot(lambda a, b: shp(a) - shp(b),
                  (bH["pnl_net"][m], bB["pnl_net"][m]))
    glo, ghi = boot(lambda a, b: shp(a) - shp(b),
                    (bH["pnl_gross"][m], bB["pnl_gross"][m]))
    beH = 1e4 * bH["pnl_gross"][m].mean() / max(bH["turnover"][m].mean(),
                                                1e-9)
    beB = 1e4 * bB["pnl_gross"][m].mean() / max(bB["turnover"][m].mean(),
                                                1e-9)
    print(f"  net dSharpe 95% CI [{lo:+.2f}, {hi:+.2f}]"
          + ("   << DECISIVE ERA" if e == "TEST" else ""))
    print(f"  gross dSharpe 95% CI [{glo:+.2f}, {ghi:+.2f}] (diagnostic)")
    print(f"  breakeven cost: S4H {beH:+.1f}bps  S4B {beB:+.1f}bps")
    if e == "TEST":
        v = ("HMM ADDS VALUE" if lo > 0 else
             "HMM SUBTRACTS" if hi < 0 else "INCONCLUSIVE AT THIS POWER")
        print(f"  PRE-REGISTERED VERDICT (net, TEST): {v}")

for nm, bt in [("S4B_style_proxy", bB), ("S4H_style_hmm", bH)]:
    out.append(pd.DataFrame({"date": dates, "strat": nm, "era": era,
                             "pnl_gross": bt["pnl_gross"],
                             "turnover": bt["turnover"]}))
res = pl.from_pandas(pd.concat(out, ignore_index=True))
res.write_parquet(DRIVE / "bt12_style.parquet")
print("\nwrote bt12_style.parquet", res.shape)
print("=" * 70)
