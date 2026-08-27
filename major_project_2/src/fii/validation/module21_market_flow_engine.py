"""
MODULE 21 · MARKET-LEVEL FLOW RISK ENGINE

Converts the one pre-registered, both-era, deployable finding of this
project -- aggregate FII net inflow at t-2 predicts lower next-day Nifty
|return| beyond VIX (PREREG_AGGREGATE_FLOW.md, cell C4: t=-4.78 full,
-2.49 TRAIN / -2.51 TEST) -- into an actual out-of-sample forecasting
engine, and tests whether the regression finding survives the conversion.

Design (fixed in docs/PREREG_MARKET_ENGINE.md BEFORE this module ran):

    r(t+1) = s * m(t) * sigma_EWMA(t) * T_nu
    m(t)   = clamp(exp(theta . x(t)), 0.5, 2.0)

with four fixed feature sets

    M0 ewma_t    x = []
    M1 vix       x = [ln(VIX(t)/20)]
    M2 vix_flow  x = [ln(VIX(t)/20), NF+(t-2), NF-(t-2)]
    M3 flow      x = [NF+(t-2), NF-(t-2)]        (diagnostic)

NF is the C4 construction exactly: (buy - sell VALUE_INR summed over all
stocks, TR_TYPE in {1,4}, RATE>0, REG_DL_INSTR_EQ) divided by the
trailing 250-day mean of daily gross flow ending t-1. Days with no flow
observation (masked 2021-05/06; TR_TYPE-null 2023-06/09/11) enter as
NF=0 -- "no information", never imputed direction.

All parameters (s, nu, theta) are fit by Student-t MLE on the expanding
past only, first fit after 750 scored days, refit every 21 trading days,
applied out-of-sample to the next block.

Decisive gate (pre-registered): M2 beats M1 on CRPS with full-sample DM
t <= -2.0 AND a lower mean CRPS in TRAIN and TEST separately, without
breaking VaR calibration (Kupiec at 1%). A gate FAIL is a scientific
result, not a broken pipeline: this module always exits 0 and prints the
verdict.

Run:  python -m fii.validation.module21_market_flow_engine
"""
from __future__ import annotations

import datetime as dt
import glob
import time

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import minimize

from fii.paths import ISIN_MAPPING, VALIDATION_DATA, VALIDATION_OUT

# ---- pre-registered constants (PREREG_MARKET_ENGINE.md) --------------------
LAM = 0.94                 # RiskMetrics EWMA
BURN = 60                  # EWMA burn-in, days
FIRST_FIT = 750            # first vintage after this many scored days
REFIT = 21                 # refit cadence, trading days
GROSS_W = 250              # trailing window for the flow scale
CLAMP = (0.5, 2.0)         # tilt clamp at forecast time
TAUS = np.arange(0.01, 1.00, 0.01)
NW_LAGS = 5
TRAIN_END = dt.date(2021, 4, 30)
TEST_START = dt.date(2021, 7, 1)

MODELS = {                 # name -> feature column indices into X
    "M0_ewma_t":   [],
    "M1_vix":      [0],
    "M2_vix_flow": [0, 1, 2],
    "M3_flow":     [1, 2],
}


# ---- data ------------------------------------------------------------------
def build_flow() -> pl.DataFrame:
    """Aggregate daily FII net and gross rupee flow (C4 construction)."""
    files = sorted(glob.glob(str(ISIN_MAPPING / "20[0-9][0-9].parquet")))
    return (pl.scan_parquet(files)
              .with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))
              .filter(pl.col("TR_TYPE").is_in([1, 4])
                      & (pl.col("RATE") > 0)
                      & (pl.col("RFDE_INSTR_TYPE") == "REG_DL_INSTR_EQ"))
              .group_by("TR_DATE")
              .agg(net=(pl.when(pl.col("TR_TYPE") == 1)
                          .then(pl.col("VALUE_INR"))
                          .otherwise(-pl.col("VALUE_INR"))).sum(),
                   gross=pl.col("VALUE_INR").sum())
              .rename({"TR_DATE": "date"})
              .sort("date")
            ).collect()


def build_frame() -> pl.DataFrame:
    mk = (pl.scan_parquet(VALIDATION_DATA / "returns_panel_v3.parquet")
            .select(["date", "nifty50_ret", "india_vix"])
            .unique(subset=["date"]).sort("date").collect()
            .with_columns(pl.col("india_vix").forward_fill())
            .filter(pl.col("nifty50_ret").is_not_null()
                    & pl.col("india_vix").is_not_null()))
    fl = build_flow()
    df = mk.join(fl, on="date", how="left")
    # trailing 250-day mean of gross flow, ending t-1, over trading days
    df = df.with_columns(
        gmean=pl.col("gross").rolling_mean(GROSS_W, min_samples=GROSS_W // 2)
                .shift(1))
    df = df.with_columns(
        nf=pl.when(pl.col("net").is_not_null() & pl.col("gmean").is_not_null())
             .then(pl.col("net") / pl.col("gmean"))
             .otherwise(0.0))
    return df


# ---- likelihood ------------------------------------------------------------
def _nll(p: np.ndarray, Z: np.ndarray, X: np.ndarray) -> float:
    """-loglik of Z ~ s * exp(X theta) * T_nu.  p = [log s, log(nu-2), theta]."""
    s = np.exp(p[0])
    nu = 2.0 + np.exp(p[1])
    scale = s * np.exp(X @ p[2:]) if X.shape[1] else np.full(len(Z), s)
    v = np.log(scale).sum() - stats.t.logpdf(Z / scale, nu).sum()
    return v if np.isfinite(v) else 1e12


MIN_PAIRS = 500      # no parameter is applied OOS on fewer training pairs.
# Code-review correction (2026-08-23): without this guard, engines whose
# base sigma only exists from FIRST_FIT (the GJR engines) fit their first
# tilt vintages on 0..500 pairs and those noisy parameters were SCORED,
# asymmetrically handicapping feature-rich models. Days whose vintage
# lacks MIN_PAIRS are now left unscored -- symmetrically for every model
# including the featureless base. v1 (EWMA base) is unaffected: its first
# vintage already had ~690 pairs. See AUDIT_RETRACTIONS.md R6.


def fit_walk_forward(Z: np.ndarray, Xf: np.ndarray, cols: list[int],
                     valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-day (scale multiplier s*clamp(m), nu), NaN before FIRST_FIT."""
    n = len(Z)
    X = Xf[:, cols]
    mult = np.full(n, np.nan)
    nu_d = np.full(n, np.nan)
    p = np.r_[0.0, np.log(4.0), np.zeros(len(cols))]     # s=1, nu=6, theta=0
    for b0 in range(FIRST_FIT, n, REFIT):
        past = valid[:b0]
        if int(past.sum()) < MIN_PAIRS:
            continue                       # leave the block unscored
        args = (Z[:b0][past], X[:b0][past])
        r = minimize(_nll, p, args=args, method="BFGS",
                     options=dict(gtol=1e-9, maxiter=1000))
        # Nelder-Mead polish: BFGS on a flat small-edge likelihood can stop
        # at path-dependent points (found in the R6 code review); the polish
        # makes the fit warm-start-independent to ~1e-10 in nll.
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


# ---- scoring ---------------------------------------------------------------
def score(y: np.ndarray, scale: np.ndarray, nu: np.ndarray):
    """CRPS (quantile decomposition), pinball at 5%/1%, VaR hits."""
    n = len(y)
    crps = np.full(n, np.nan)
    pin5 = np.full(n, np.nan)
    pin1 = np.full(n, np.nan)
    hit5 = np.full(n, np.nan)
    hit1 = np.full(n, np.nan)
    for v in np.unique(nu[np.isfinite(nu)]):
        m = np.isfinite(nu) & (nu == v) & np.isfinite(scale)
        tq = stats.t.ppf(TAUS, v)                        # (99,)
        q = scale[m, None] * tq[None, :]                 # (n_m, 99)
        d = y[m, None] - q
        pin = np.maximum(TAUS[None, :] * d, (TAUS[None, :] - 1.0) * d)
        crps[m] = 2.0 * pin.mean(axis=1)
        pin5[m] = pin[:, 4]                              # tau = 0.05
        pin1[m] = pin[:, 0]                              # tau = 0.01
        hit5[m] = (y[m] < q[:, 4]).astype(float)
        hit1[m] = (y[m] < q[:, 0]).astype(float)
    return crps, pin5, pin1, hit5, hit1


def dm_test(la: np.ndarray, lb: np.ndarray) -> tuple[float, int]:
    """Diebold-Mariano t (NW), negative = A better."""
    d = la - lb
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, n
    d0 = d - d.mean()
    g = [float((d0 * d0).mean())]
    for l in range(1, NW_LAGS + 1):
        g.append(float((d0[l:] * d0[:-l]).mean()))
    var = g[0] + 2.0 * sum((1.0 - l / (NW_LAGS + 1)) * g[l]
                           for l in range(1, NW_LAGS + 1))
    return float(d.mean() / np.sqrt(max(var, 1e-30) / n)), n


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    h = hits[np.isfinite(hits)]
    n, x = len(h), h.sum()
    if n == 0:
        return np.nan, np.nan
    ph = x / n
    def ll(q):
        return (n - x) * np.log(max(1 - q, 1e-12)) + x * np.log(max(q, 1e-12))
    lr = -2.0 * (ll(p) - ll(max(ph, 1e-12)))
    return float(ph), float(stats.chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> float:
    h = hits[np.isfinite(hits)].astype(int)
    if len(h) < 30:
        return np.nan
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    def sl(n_, p_):
        return n_ * np.log(max(p_, 1e-12))
    l1 = sl(n00, 1 - p01) + sl(n01, p01) + sl(n10, 1 - p11) + sl(n11, p11)
    l0 = sl(n00 + n10, 1 - p1) + sl(n01 + n11, p1)
    return float(stats.chi2.sf(-2.0 * (l0 - l1), 1))


# ---- main ------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    df = build_frame()
    n_all = df.height
    dates = df["date"].to_numpy()
    ret = df["nifty50_ret"].to_numpy().astype(float)
    vix = df["india_vix"].to_numpy().astype(float)
    nf = df["nf"].to_numpy().astype(float)
    n_flow = int((df["net"].is_not_null()).sum())
    print(f"frame: {n_all} trading days {dates[0]} .. {dates[-1]}, "
          f"{n_flow} with flow observations   [{time.time()-t0:.1f}s]")

    # EWMA variance, burn-in seed
    var = np.full(n_all, np.nan)
    var[BURN - 1] = float(np.var(ret[:BURN]))
    for i in range(BURN, n_all):
        var[i] = LAM * var[i - 1] + (1 - LAM) * ret[i] ** 2
    sig = np.sqrt(var)

    # forecast pairs: at close of day t predict r(t+1)
    # feature matrix at t: [ln(VIX_t/20), NF+(t-2), NF-(t-2)]
    nf2 = np.r_[0.0, 0.0, nf[:-2]]
    Xf = np.column_stack([np.log(vix / 20.0),
                          np.maximum(nf2, 0.0), np.minimum(nf2, 0.0)])
    y = np.r_[ret[1:], np.nan]                     # y[t] = r(t+1)
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
        res[name] = dict(crps=crps, pin5=pin5, pin1=pin1,
                         hit5=hit5, hit1=hit1, nu=nu_d, mult=mult)
        print(f"  fitted {name:12s} scored={int(np.isfinite(crps).sum())} "
              f"nu(last)={nu_d[np.isfinite(nu_d)][-1]:.2f} "
              f"[{time.time()-t0:.1f}s]")

    # eras keyed on the TARGET date r(t+1)
    tgt = np.r_[dates[1:], dates[-1]]
    era = {"FULL": np.ones(n_all, bool),
           "TRAIN": tgt <= np.datetime64(TRAIN_END),
           "TEST": tgt >= np.datetime64(TEST_START)}

    print("\n== mean losses (x100) ==")
    hdr = f"{'model':14s}" + "".join(f"{e+'-CRPS':>12s}{e+'-pin5':>12s}{e+'-pin1':>12s}"
                                     for e in era)
    print(hdr)
    for name, r in res.items():
        row = f"{name:14s}"
        for e, m in era.items():
            for k in ("crps", "pin5", "pin1"):
                x = r[k][m]
                row += f"{np.nanmean(x)*100:12.4f}"
        print(row)

    print("\n== VaR backtests (FULL) ==")
    print(f"{'model':14s}{'hit5%':>8s}{'kupiec':>8s}{'chris':>8s}"
          f"{'hit1%':>8s}{'kupiec':>8s}{'chris':>8s}")
    kup_m2 = {}
    for name, r in res.items():
        h5, k5 = kupiec(r["hit5"], 0.05); c5 = christoffersen(r["hit5"])
        h1, k1 = kupiec(r["hit1"], 0.01); c1 = christoffersen(r["hit1"])
        if name == "M2_vix_flow":
            kup_m2 = dict(k5=k5, k1=k1)
        print(f"{name:14s}{h5*100:7.2f}%{k5:8.3f}{c5:8.3f}"
              f"{h1*100:7.2f}%{k1:8.3f}{c1:8.3f}")

    print("\n== DM tests on CRPS (negative t = first model better) ==")
    pairs = [("M1_vix", "M0_ewma_t"), ("M2_vix_flow", "M1_vix"),
             ("M2_vix_flow", "M0_ewma_t"), ("M3_flow", "M0_ewma_t")]
    dm_gate = np.nan
    for a, b in pairs:
        row = f"{a} vs {b:12s}"
        for e, m in era.items():
            t, nn = dm_test(res[a]["crps"][m], res[b]["crps"][m])
            row += f"  {e}: t={t:+6.2f} (n={nn})"
            if (a, b, e) == ("M2_vix_flow", "M1_vix", "FULL"):
                dm_gate = t
        print(row)

    # ---- pre-registered gate ----
    mtr = {e: np.nanmean(res["M2_vix_flow"]["crps"][m]) -
              np.nanmean(res["M1_vix"]["crps"][m]) for e, m in era.items()}
    ga = np.isfinite(dm_gate) and dm_gate <= -2.0
    gb = mtr["TRAIN"] < 0 and mtr["TEST"] < 0
    gc = all(np.isfinite(v) and v > 0.01 for v in kup_m2.values())
    print("\n== PRE-REGISTERED GATE (docs/PREREG_MARKET_ENGINE.md) ==")
    print(f"  (a) M2 vs M1 FULL DM t = {dm_gate:+.2f}  (need <= -2.0)  "
          f"{'PASS' if ga else 'FAIL'}")
    print(f"  (b) M2 better in both eras: TRAIN {mtr['TRAIN']*100:+.4f} "
          f"TEST {mtr['TEST']*100:+.4f} (x100, need both < 0)  "
          f"{'PASS' if gb else 'FAIL'}")
    print(f"  (c) M2 VaR calibration not rejected (Kupiec p > 0.01): "
          f"p5={kup_m2.get('k5', np.nan):.3f} p1={kup_m2.get('k1', np.nan):.3f}  "
          f"{'PASS' if gc else 'FAIL'}")
    print(f"  GATE: {'PASS' if (ga and gb and gc) else 'FAIL'}")

    out = pl.DataFrame(
        {"date": dates, "target_date": tgt, "ret_next": y,
         "sigma_ewma": sig, "nf_t2": nf2,
         **{f"{k}_{n}": res[n][k] for n in MODELS
            for k in ("crps", "pin5", "pin1", "hit5", "hit1")},
         **{f"scale_{n}": res[n]["mult"] * sig for n in MODELS},
         **{f"nu_{n}": res[n]["nu"] for n in MODELS}})
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(VALIDATION_OUT / "market_engine_scores.parquet")
    print(f"\nwrote {VALIDATION_OUT / 'market_engine_scores.parquet'} "
          f"[{time.time()-t0:.1f}s total]")


if __name__ == "__main__":
    main()
