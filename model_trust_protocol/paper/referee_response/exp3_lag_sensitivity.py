"""EXP3. Is the two-day reporting lag a knife-edge?

critics.txt 1.6: the t-2 information set is a DEPLOYMENT ASSUMPTION, not an
econometric result; nothing in the programme documents the settlement/vendor
latency. If the C4 risk finding lives only at exactly t-2, the survivor's
significance half is fragile. If it is flat across neighbouring lags, the
assumption is not load-bearing.

Re-runs the pre-registered C4 specification -- |nifty(t+1)| ~ NF(t-k),
NEG(t-k) + 5 vol lags + india_vix, Newey-West 10 -- for k = 0..5, using the
project's own estimator and its canonical unobserved-flow rule (exclude).
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, "/Users/divyanshsingh/Desktop/model_trust_protocol/src")
from convgap.replication.aggregate_flow import (  # noqa: E402
    build_frame, _ols_hac, TRAIN_END, TEST_START, N_TARGET_LAGS, NW_LAGS)

VD = Path("/Users/divyanshsingh/Desktop/Major Project 2/data/VALIDATION_DATA")
IM = Path("/Users/divyanshsingh/Desktop/Major Project 2/data/ISIN_MAPPING")

df = build_frame(VD, IM)
for k in range(6):
    df = df.with_columns(**{f"nf_L{k}": pl.col("nf").shift(k)})
    df = df.with_columns(**{f"neg_L{k}": pl.min_horizontal(pl.col(f"nf_L{k}"), pl.lit(0.0))})

controls = [f"vol_lag{j}" for j in range(N_TARGET_LAGS)] + ["india_vix"]
eras = {"FULL": df,
        "TRAIN": df.filter(pl.col("date") <= TRAIN_END),
        "TEST": df.filter(pl.col("date") >= TEST_START)}

print("=" * 78)
print("EXP3 · C4 RISK CELL ACROSS INFORMATION LAGS  (Newey-West %d)" % NW_LAGS)
print("=" * 78)
print("bar: |t| >= 2.50 full (Bonferroni, 4 cells) AND same sign |t| >= 1.5 both eras\n")
print(f"{'lag':>5s}{'FULL t':>10s}{'n':>7s}{'TRAIN t':>10s}{'TEST t':>10s}{'verdict':>10s}")

rows = []
for k in range(6):
    regs = [f"nf_L{k}", f"neg_L{k}", *controls]
    out = {}
    for era, frame in eras.items():
        sub = frame.select(["y_vol", "date", *regs]).drop_nulls()
        sub = sub.filter(pl.col(f"nf_L{k}") != 0.0)
        y = sub["y_vol"].to_numpy()
        x = sub.select(regs).to_numpy()
        coef, t, p = _ols_hac(y, x, focus=0)
        out[era] = (coef, t, sub.height)
    tf, tr, te = out["FULL"][1], out["TRAIN"][1], out["TEST"][1]
    ok = (abs(tf) >= 2.50 and np.sign(tr) == np.sign(te)
          and abs(tr) >= 1.5 and abs(te) >= 1.5)
    rows.append((k, tf, tr, te, ok))
    print(f"{'t-'+str(k):>5s}{tf:+10.2f}{out['FULL'][2]:7d}{tr:+10.2f}{te:+10.2f}"
          f"{'PASS' if ok else 'fail':>10s}")

print("\n" + "=" * 78)
passing = [k for k, *_, ok in rows if ok]
print(f"Lags clearing the pre-registered bar: {passing if passing else 'none'}")
k2 = dict((r[0], r) for r in rows)[2]
print(f"The deployed lag (t-2): FULL t = {k2[1]:+.2f}  -> "
      f"{'PASS' if k2[4] else 'FAIL'}")
print("Knife-edge?" , "NO - neighbouring lags also clear" if len(passing) > 1
      else "YES - only one lag clears the bar")
