"""
MODULE 22 · MARKET FLOW RISK ENGINE v2  (GJR base + full aggregate flow set)

v1 (module 21) passed its pre-registered gate but on a bare EWMA base and
with a single flow number per day. v2 answers the two objections a
referee (or a desk) would raise, with design fixed in
docs/PREREG_MARKET_ENGINE_V2.md BEFORE this module ran:

  base:  GJR-GARCH(1,1) with leverage, Gaussian QMLE, refit every 21
         days walk-forward — the competitive benchmark.
  flow:  five aggregate features at deployable lag, all new-to-project
         except nf±:  nf+/nf− (v1), nf5 persistence (t−6..t−3),
         breadth deviation, gross-intensity log-ratio.

Models: G0 gjr_t | G1 +vix | G2 +vix+flow(5) | G2a +vix+nf± (attribution)
        | G3 +flow only.  Gate: G2 vs G1, same bar as v1.

Exits 0 whether the gate passes or fails; a FAIL is a result.

Run:  python -m fii.validation.module22_market_flow_engine_v2
"""
from __future__ import annotations

import glob
import time

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import minimize

from fii.paths import ISIN_MAPPING, VALIDATION_DATA, VALIDATION_OUT
from fii.validation.module21_market_flow_engine import (
    BURN, FIRST_FIT, GROSS_W, REFIT, TEST_START, TRAIN_END,
    christoffersen, dm_test, fit_walk_forward, kupiec, score)

MODELS = {                       # name -> feature column indices into Xf
    "G0_gjr_t":    [],
    "G1_vix":      [0],
    "G2_vix_flow": [0, 1, 2, 3, 4, 5],
    "G2a_vix_nf":  [0, 1, 2],
    "G3_flow":     [1, 2, 3, 4, 5],
}
GJR_REFIT = 21


# ---- data ------------------------------------------------------------------
def build_flow2() -> pl.DataFrame:
    """Per-date aggregate net, gross, and cross-sectional breadth."""
    files = sorted(glob.glob(str(ISIN_MAPPING / "20[0-9][0-9].parquet")))
    sd = (pl.scan_parquet(files)
            .with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))
            .filter(pl.col("TR_TYPE").is_in([1, 4])
                    & (pl.col("RATE") > 0)
                    & (pl.col("RFDE_INSTR_TYPE") == "REG_DL_INSTR_EQ"))
            .group_by(["TR_DATE", "ISIN"])
            .agg(net=(pl.when(pl.col("TR_TYPE") == 1)
                        .then(pl.col("VALUE_INR"))
                        .otherwise(-pl.col("VALUE_INR"))).sum(),
                 gross=pl.col("VALUE_INR").sum()))
    return (sd.group_by("TR_DATE")
              .agg(net=pl.col("net").sum(),
                   gross=pl.col("gross").sum(),
                   breadth=((pl.col("net") > 0).sum().cast(pl.Float64)
                            - (pl.col("net") < 0).sum())
                           / pl.len())
              .rename({"TR_DATE": "date"})
              .sort("date")).collect()


def build_frame() -> pl.DataFrame:
    mk = (pl.scan_parquet(VALIDATION_DATA / "returns_panel_v3.parquet")
            .select(["date", "nifty50_ret", "india_vix"])
            .unique(subset=["date"]).sort("date").collect()
            .with_columns(pl.col("india_vix").forward_fill())
            .filter(pl.col("nifty50_ret").is_not_null()
                    & pl.col("india_vix").is_not_null()))
    df = mk.join(build_flow2(), on="date", how="left")
    hw = GROSS_W // 2
    df = df.with_columns(
        gmean=pl.col("gross").rolling_mean(GROSS_W, min_samples=hw).shift(1),
        bmean=pl.col("breadth").rolling_mean(GROSS_W, min_samples=hw).shift(1))
    df = df.with_columns(
        nf=pl.when(pl.col("net").is_not_null() & pl.col("gmean").is_not_null())
             .then(pl.col("net") / pl.col("gmean")).otherwise(0.0),
        brd=(pl.col("breadth") - pl.col("bmean")).fill_null(0.0),
        gid=(pl.col("gross") / pl.col("gmean")).log().fill_null(0.0))
    return df.with_columns(
        nf2=pl.col("nf").shift(2).fill_null(0.0),
        nf5=pl.col("nf").rolling_sum(4, min_samples=4).shift(3).fill_null(0.0),
        br2=pl.col("brd").shift(2).fill_null(0.0),
        gi2=pl.col("gid").shift(2).fill_null(0.0))


# ---- GJR-GARCH(1,1) base ---------------------------------------------------
def _gjr_filter(r: np.ndarray, w: float, a: float, g: float, b: float,
                h0: float) -> np.ndarray:
    """h[i] = conditional variance of r[i]; h has len(r)+1 entries."""
    n = len(r)
    h = np.empty(n + 1)
    h[0] = h0
    for i in range(n):
        h[i + 1] = w + (a + g * (r[i] < 0.0)) * r[i] * r[i] + b * h[i]
    return h


def _gjr_nll(p: np.ndarray, r: np.ndarray, h0: float) -> float:
    w, a, g, b = p
    if w <= 0 or a < 0 or g < 0 or b < 0 or a + 0.5 * g + b >= 0.9999:
        return 1e12
    h = _gjr_filter(r, w, a, g, b, h0)[:-1]
    h = h[BURN:]
    v = 0.5 * float(np.sum(np.log(h) + r[BURN:] ** 2 / h))
    return v if np.isfinite(v) else 1e12


def gjr_walk_forward(ret: np.ndarray) -> tuple[np.ndarray, list]:
    """sig_fc[t] = sqrt(h[t+1]) under the vintage in force at t; OOS only."""
    n = len(ret)
    h0 = float(np.var(ret[:BURN]))
    uv = float(np.var(ret[:FIRST_FIT]))
    p = np.array([uv * 0.05, 0.05, 0.10, 0.85])
    p[0] = uv * (1 - p[1] - 0.5 * p[2] - p[3])
    sig_fc = np.full(n, np.nan)
    hist = []
    for b0 in range(FIRST_FIT, n, GJR_REFIT):
        r = minimize(_gjr_nll, p, args=(ret[:b0], h0), method="Nelder-Mead",
                     options=dict(maxiter=400, xatol=1e-10, fatol=1e-8))
        if np.isfinite(r.fun) and r.fun < 1e11:
            p = r.x
        b1 = min(b0 + GJR_REFIT, n)
        h = _gjr_filter(ret[:b1], *p, h0)
        sig_fc[b0:b1] = np.sqrt(h[b0 + 1:b1 + 1])
        hist.append((b0, *p))
    return sig_fc, hist


# ---- economic metrics ------------------------------------------------------
def econ_metrics(y, scale, nu, mask):
    m = mask & np.isfinite(scale) & np.isfinite(nu)
    var99 = scale[m] * stats.t.ppf(0.01, nu[m])
    yy = y[m]
    c = np.quantile(yy / var99, 0.99)          # constant for exact 1% coverage
    br = yy < var99
    esgap = float(np.mean(var99[br] - yy[br])) if br.sum() else np.nan
    return (100 * float(np.mean(-c * var99)),  # matched-coverage capital, %
            100 * esgap)                       # mean excess loss beyond VaR99


# ---- main ------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    df = build_frame()
    dates = df["date"].to_numpy()
    ret = df["nifty50_ret"].to_numpy().astype(float)
    vix = df["india_vix"].to_numpy().astype(float)
    n_all = df.height
    print(f"frame: {n_all} trading days {dates[0]} .. {dates[-1]}   "
          f"[{time.time()-t0:.1f}s]")

    sig, hist = gjr_walk_forward(ret)
    w, a, g, b = hist[-1][1:]
    print(f"GJR walk-forward: {len(hist)} vintages, last params "
          f"w={w:.2e} a={a:.3f} g={g:.3f} b={b:.3f} "
          f"(persist={a+0.5*g+b:.3f})   [{time.time()-t0:.1f}s]")

    Xf = np.column_stack([
        np.log(vix / 20.0),
        np.maximum(df["nf2"].to_numpy(), 0.0),
        np.minimum(df["nf2"].to_numpy(), 0.0),
        df["nf5"].to_numpy(),
        df["br2"].to_numpy(),
        df["gi2"].to_numpy()])
    y = np.r_[ret[1:], np.nan]
    ok = np.isfinite(sig) & np.isfinite(y) & np.isfinite(Xf).all(axis=1)
    Z = np.where(ok, y / np.where(sig > 0, sig, np.nan), np.nan)
    valid = np.isfinite(Z)

    res = {}
    for name, cols in MODELS.items():
        mult, nu_d = fit_walk_forward(np.nan_to_num(Z), Xf, cols, valid)
        scale = mult * sig
        sc = np.isfinite(scale) & valid
        crps, pin5, pin1, hit5, hit1 = score(
            np.where(sc, y, np.nan), np.where(sc, scale, np.nan), nu_d)
        res[name] = dict(crps=crps, pin5=pin5, pin1=pin1, hit5=hit5,
                         hit1=hit1, nu=nu_d, scale=scale)
        print(f"  fitted {name:12s} scored={int(np.isfinite(crps).sum())} "
              f"[{time.time()-t0:.1f}s]")

    tgt = np.r_[dates[1:], dates[-1]]
    era = {"FULL": np.ones(n_all, bool),
           "TRAIN": tgt <= np.datetime64(TRAIN_END),
           "TEST": tgt >= np.datetime64(TEST_START)}

    print("\n== mean losses (x100) ==")
    print(f"{'model':14s}" + "".join(
        f"{e+'-CRPS':>12s}{e+'-pin5':>12s}{e+'-pin1':>12s}" for e in era))
    for name, r in res.items():
        row = f"{name:14s}"
        for e, m in era.items():
            for k in ("crps", "pin5", "pin1"):
                row += f"{np.nanmean(r[k][m])*100:12.4f}"
        print(row)

    print("\n== VaR backtests (FULL) + economic metrics ==")
    print(f"{'model':14s}{'hit5%':>8s}{'kupiec':>8s}{'hit1%':>8s}{'kupiec':>8s}"
          f"{'chris1':>8s}{'cap@1%':>9s}{'ESgap':>8s}")
    kup_g2 = {}
    for name, r in res.items():
        h5, k5 = kupiec(r["hit5"], 0.05)
        h1, k1 = kupiec(r["hit1"], 0.01)
        c1 = christoffersen(r["hit1"])
        cap, esg = econ_metrics(y, r["scale"], r["nu"], era["FULL"] & valid)
        if name == "G2_vix_flow":
            kup_g2 = dict(k5=k5, k1=k1)
        print(f"{name:14s}{h5*100:7.2f}%{k5:8.3f}{h1*100:7.2f}%{k1:8.3f}"
              f"{c1:8.3f}{cap:8.3f}%{esg:7.3f}%")

    print("\n== DM tests on CRPS (negative t = first model better) ==")
    pairs = [("G1_vix", "G0_gjr_t"), ("G2_vix_flow", "G1_vix"),
             ("G2_vix_flow", "G0_gjr_t"), ("G2a_vix_nf", "G1_vix"),
             ("G2_vix_flow", "G2a_vix_nf"), ("G3_flow", "G0_gjr_t")]
    dm_gate = np.nan
    for a_, b_ in pairs:
        row = f"{a_} vs {b_:12s}"
        for e, m in era.items():
            t, nn = dm_test(res[a_]["crps"][m], res[b_]["crps"][m])
            row += f"  {e}: t={t:+6.2f} (n={nn})"
            if (a_, b_, e) == ("G2_vix_flow", "G1_vix", "FULL"):
                dm_gate = t
        print(row)

    mtr = {e: np.nanmean(res["G2_vix_flow"]["crps"][m]) -
              np.nanmean(res["G1_vix"]["crps"][m]) for e, m in era.items()}
    ga = np.isfinite(dm_gate) and dm_gate <= -2.0
    gb = mtr["TRAIN"] < 0 and mtr["TEST"] < 0
    gc = all(np.isfinite(v) and v > 0.01 for v in kup_g2.values())
    print("\n== PRE-REGISTERED GATE (docs/PREREG_MARKET_ENGINE_V2.md) ==")
    print(f"  (a) G2 vs G1 FULL DM t = {dm_gate:+.2f}  (need <= -2.0)  "
          f"{'PASS' if ga else 'FAIL'}")
    print(f"  (b) G2 better both eras: TRAIN {mtr['TRAIN']*100:+.4f} "
          f"TEST {mtr['TEST']*100:+.4f} (x100)  {'PASS' if gb else 'FAIL'}")
    print(f"  (c) G2 Kupiec p5={kup_g2.get('k5', float('nan')):.3f} "
          f"p1={kup_g2.get('k1', float('nan')):.3f} (need > 0.01)  "
          f"{'PASS' if gc else 'FAIL'}")
    print(f"  GATE: {'PASS' if (ga and gb and gc) else 'FAIL'}")

    out = pl.DataFrame(
        {"date": dates, "target_date": tgt, "ret_next": y, "sigma_gjr": sig,
         **{f"{k}_{n}": res[n][k] for n in MODELS
            for k in ("crps", "pin5", "pin1", "hit5", "hit1", "scale", "nu")}})
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(VALIDATION_OUT / "market_engine_v2_scores.parquet")
    print(f"\nwrote market_engine_v2_scores.parquet "
          f"[{time.time()-t0:.1f}s total]")


if __name__ == "__main__":
    main()
