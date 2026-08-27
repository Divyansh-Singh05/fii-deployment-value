"""
MODULE 24 · WEEKLY (h=5) MARKET RISK ENGINE — Stage 1 of PREREG_WEEKLY_ENGINE.md

Stage 0 (the pre-registered screen, run 2026-08-23) passed on exactly
one of three cells: spx5, the trailing 5-day S&P500 return known at the
last US close before India's close (t=-3.50 full, -2.72 TRAIN, -4.39
TEST). Flow persistence and USDINR failed the screen, so per the
prereg this engine tilts on spx5 ONLY.

    R5(t) = r(t+1)+...+r(t+5) = s * m(t) * sigma5_GJR(t) * T_nu
    m(t)  = clamp(exp(theta * spx5(t)), 0.5, 2.0)

sigma5_GJR: the walk-forward GJR(1,1) daily variance iterated 5 steps
(var5 = h_{t+1}*A + w*(5-A)/(1-phi), A = 1+phi+..+phi^4, phi = a+g/2+b)
under the vintage in force at t. Tilt (s, nu, theta) fit by Student-t
QMLE on the expanding past with a 5-DAY GUARD: training pairs end at
t <= b0-5 so no pair's target window overlaps the scoring block.
Overlapping daily scoring; DM Newey-West 10 lags; the five
non-overlapping offset subseries are reported alongside.

Gate (fixed in the prereg): B1 vs B0 CRPS DM t <= -2.0 full (NW10),
better in both eras (era keyed on forecast date t, as in Stage 0),
Kupiec p > 0.01 at 95% and 99%. Exits 0 either way; FAIL is a result.

Run:  python -m fii.validation.module24_weekly_engine
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import minimize

from fii.paths import VALIDATION_DATA, VALIDATION_OUT
from fii.validation.module21_market_flow_engine import (
    CLAMP, FIRST_FIT, MIN_PAIRS, REFIT, TEST_START, TRAIN_END, _nll,
    christoffersen, kupiec, score)
from fii.validation.module22_market_flow_engine_v2 import gjr_walk_forward

H = 5
NW_LAGS = 10
MODELS = {"B0_gjr5": [], "B1_spx5": [0]}


def dm_nw(la: np.ndarray, lb: np.ndarray, lags: int) -> tuple[float, int]:
    d = la - lb
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, n
    d0 = d - d.mean()
    g = [float((d0 * d0).mean())]
    for l in range(1, lags + 1):
        g.append(float((d0[l:] * d0[:-l]).mean()))
    var = g[0] + 2.0 * sum((1 - l / (lags + 1)) * g[l] for l in range(1, lags + 1))
    return float(d.mean() / np.sqrt(max(var, 1e-30) / n)), n


def fit_wf_guard(Z: np.ndarray, Xf: np.ndarray, cols: list[int],
                 valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """module21.fit_walk_forward with training pairs ending at b0-H."""
    n = len(Z)
    X = Xf[:, cols]
    mult = np.full(n, np.nan)
    nu_d = np.full(n, np.nan)
    p = np.r_[0.0, np.log(4.0), np.zeros(len(cols))]
    for b0 in range(FIRST_FIT, n, REFIT):
        cut = max(b0 - H, 0)
        past = valid[:cut]
        if int(past.sum()) < MIN_PAIRS:
            continue                       # leave the block unscored (R6 rule)
        args = (Z[:cut][past], X[:cut][past])
        r = minimize(_nll, p, args=args, method="BFGS",
                     options=dict(gtol=1e-9, maxiter=1000))
        r2 = minimize(_nll, r.x if np.isfinite(r.fun) else p, args=args,
                      method="Nelder-Mead",
                      options=dict(maxiter=4000, xatol=1e-9, fatol=1e-11))
        best = min((r, r2), key=lambda q: q.fun)
        if np.isfinite(best.fun):
            p = best.x
        s, nu = np.exp(p[0]), 2.0 + np.exp(p[1])
        b1 = min(b0 + REFIT, n)
        tilt = np.exp(X[b0:b1] @ p[2:]) if cols else np.ones(b1 - b0)
        mult[b0:b1] = s * np.clip(tilt, *CLAMP)
        nu_d[b0:b1] = nu
    return mult, nu_d


def main() -> None:
    t0 = time.time()
    mk = (pl.scan_parquet(VALIDATION_DATA / "returns_panel_v3.parquet")
            .select(["date", "nifty50_ret", "india_vix"])
            .unique(subset=["date"]).sort("date").collect()
            .with_columns(pl.col("india_vix").forward_fill())
            .filter(pl.col("nifty50_ret").is_not_null()
                    & pl.col("india_vix").is_not_null()))
    spx = (pl.read_parquet(VALIDATION_DATA / "sp500.parquet").sort("date")
             .with_columns(lr=(pl.col("sp500") / pl.col("sp500").shift(1))
                              .log() * 100)
             .with_columns(spx5=pl.col("lr").rolling_sum(5, min_samples=5),
                           avail=pl.col("date").dt.offset_by("1d"))
             .select("avail", "spx5").drop_nulls().sort("avail"))
    df = mk.join_asof(spx, left_on="date", right_on="avail",
                      strategy="backward")
    df = df.drop([c for c in df.columns if c.startswith("avail")])
    n = df.height
    dates = df["date"].to_numpy()
    ret = df["nifty50_ret"].to_numpy().astype(float)
    spx5 = np.nan_to_num(df["spx5"].to_numpy().astype(float))
    print(f"frame: {n} trading days {dates[0]} .. {dates[-1]}  "
          f"[{time.time()-t0:.1f}s]")

    sig1, hist = gjr_walk_forward(ret)
    # 5-step-ahead total sigma under the vintage in force at t
    sigma5 = np.full(n, np.nan)
    blocks = {b0: (w, a, g, b) for b0, w, a, g, b in hist}
    for b0, (w, a, g, b) in blocks.items():
        b1 = min(b0 + REFIT, n)
        phi = min(a + 0.5 * g + b, 0.9999)
        A = sum(phi ** k for k in range(H))
        h1 = sig1[b0:b1] ** 2
        sigma5[b0:b1] = np.sqrt(h1 * A + w * (H - A) / (1.0 - phi))
    print(f"GJR base: {len(hist)} vintages  [{time.time()-t0:.1f}s]")

    # forward 5-day return known through t+5
    r5 = np.full(n, np.nan)
    for t in range(n - H):
        r5[t] = ret[t + 1:t + 1 + H].sum()
    ok = np.isfinite(sigma5) & np.isfinite(r5)
    Z = np.where(ok, r5 / np.where(sigma5 > 0, sigma5, np.nan), np.nan)
    valid = np.isfinite(Z)
    Xf = spx5[:, None]

    res = {}
    for name, cols in MODELS.items():
        mult, nu_d = fit_wf_guard(np.nan_to_num(Z), Xf, cols, valid)
        scale = mult * sigma5
        sc = np.isfinite(scale) & valid
        crps, pin5, pin1, hit5, hit1 = score(
            np.where(sc, r5, np.nan), np.where(sc, scale, np.nan), nu_d)
        res[name] = dict(crps=crps, pin5=pin5, pin1=pin1, hit5=hit5,
                         hit1=hit1, nu=nu_d, scale=scale)
        print(f"  fitted {name:9s} scored={int(np.isfinite(crps).sum())} "
              f"[{time.time()-t0:.1f}s]")

    era = {"FULL": np.ones(n, bool),
           "TRAIN": dates <= np.datetime64(TRAIN_END),
           "TEST": dates >= np.datetime64(TEST_START)}

    print("\n== mean losses (x100), 5-day target ==")
    print(f"{'model':10s}" + "".join(
        f"{e+'-CRPS':>12s}{e+'-pin5':>12s}{e+'-pin1':>12s}" for e in era))
    for name, r in res.items():
        row = f"{name:10s}"
        for e, m in era.items():
            for k in ("crps", "pin5", "pin1"):
                row += f"{np.nanmean(r[k][m])*100:12.4f}"
        print(row)

    print("\n== VaR backtests (FULL, overlapping) ==")
    kup = {}
    for name, r in res.items():
        h5_, k5 = kupiec(r["hit5"], 0.05)
        h1_, k1 = kupiec(r["hit1"], 0.01)
        kup[name] = (k5, k1)
        print(f"  {name:10s} hit5={h5_*100:.2f}% (p={k5:.3f})  "
              f"hit1={h1_*100:.2f}% (p={k1:.3f})  "
              f"chris1={christoffersen(r['hit1']):.3f}")

    print("\n== DM on CRPS, B1 vs B0 (negative = spx5 tilt better) ==")
    dm_full = np.nan
    for e, m in era.items():
        t, nn = dm_nw(res["B1_spx5"]["crps"][m], res["B0_gjr5"]["crps"][m],
                      NW_LAGS)
        if e == "FULL":
            dm_full = t
        print(f"  {e:6s} t={t:+6.2f} (n={nn}, NW{NW_LAGS})")
    print("  non-overlapping offsets (NW2):")
    idx = np.arange(n)
    for off in range(H):
        m = (idx % H) == off
        t, nn = dm_nw(res["B1_spx5"]["crps"][m], res["B0_gjr5"]["crps"][m], 2)
        print(f"    offset {off}: t={t:+6.2f} (n={nn})")

    d = {e: np.nanmean(res["B1_spx5"]["crps"][m]) -
            np.nanmean(res["B0_gjr5"]["crps"][m]) for e, m in era.items()}
    k5, k1 = kup["B1_spx5"]
    ga = np.isfinite(dm_full) and dm_full <= -2.0
    gb = d["TRAIN"] < 0 and d["TEST"] < 0
    gc = np.isfinite(k5) and np.isfinite(k1) and k5 > 0.01 and k1 > 0.01
    print("\n== PRE-REGISTERED GATE (docs/PREREG_WEEKLY_ENGINE.md, Stage 1) ==")
    print(f"  (a) FULL DM t = {dm_full:+.2f} (need <= -2.0)  "
          f"{'PASS' if ga else 'FAIL'}")
    print(f"  (b) both eras better: TRAIN {d['TRAIN']*100:+.4f} "
          f"TEST {d['TEST']*100:+.4f} (x100)  {'PASS' if gb else 'FAIL'}")
    print(f"  (c) calibration: Kupiec p5={k5:.3f} p1={k1:.3f}  "
          f"{'PASS' if gc else 'FAIL'}")
    print(f"  GATE: {'PASS' if (ga and gb and gc) else 'FAIL'}")

    out = pl.DataFrame(
        {"date": dates, "ret5_next": r5, "sigma5_gjr": sigma5, "spx5": spx5,
         **{f"{k}_{nm}": res[nm][k] for nm in MODELS
            for k in ("crps", "pin5", "pin1", "hit5", "hit1", "scale", "nu")}})
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(VALIDATION_OUT / "market_engine_v4_weekly_scores.parquet")
    print(f"\nwrote market_engine_v4_weekly_scores.parquet "
          f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
