"""
MODULE 23 · MARKET FLOW RISK ENGINE v3 — GLOBAL FORCES

Third engine attempt in a disclosed sequence (v1 PASS on EWMA base,
v2 FAIL on GJR base — see PREREG_MARKET_ENGINE_V2.md result section).
v3 tests a different hypothesis, fixed in PREREG_MARKET_ENGINE_V3.md
BEFORE this module ran: the global dollar / rate / US-equity conditions
that DRIVE FII flows may carry more forecastable magnitude for Nifty
risk than the flow echo itself.

  H0 = GJR(1,1)-t walk-forward base
  H1 = H0 + [inr5, spx5, spxn1]                    <- gate: H1 vs H0
  H2 = H0 + [inr5, spx5, spxn1, dxy5, ddif]        (declared secondary)
  H3 = H2 + [nf+, nf-]                             (flow beyond globals)

Data: local usdinr/sp500 parquets + FRED snapshots in
data/VALIDATION_DATA/external_macro/ (DTWEXBGS, DFF, IRSTCI01INM156N).
All availability lags are deployable and documented in the prereg.

Exits 0 whether the gate passes or fails; a FAIL is a result.

Run:  python -m fii.validation.module23_market_flow_engine_v3
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy import stats

from fii.paths import VALIDATION_DATA, VALIDATION_OUT
from fii.validation.module21_market_flow_engine import (
    GROSS_W, TEST_START, TRAIN_END, build_flow, christoffersen, dm_test,
    fit_walk_forward, kupiec, score)
from fii.validation.module22_market_flow_engine_v2 import (
    econ_metrics, gjr_walk_forward)

EXT = VALIDATION_DATA / "external_macro"

MODELS = {                     # name -> columns of Xf
    "H0_gjr_t":      [],
    "H1_global3":    [0, 1, 2],
    "H2_global5":    [0, 1, 2, 3, 4],
    "H3_glob_flow":  [0, 1, 2, 3, 4, 5, 6],
}


def _fred(name: str, col: str) -> pl.DataFrame:
    return (pl.read_csv(EXT / f"{name}.csv")
              .rename({"observation_date": "date", name: col})
              .with_columns(pl.col("date").str.to_date(),
                            pl.col(col).cast(pl.Float64, strict=False))
              .drop_nulls().sort("date"))


def _ret5(df: pl.DataFrame, col: str, avail_days: int) -> pl.DataFrame:
    """1d and 5d log returns in %, tagged with availability date."""
    return (df.with_columns(lr=(pl.col(col) / pl.col(col).shift(1)).log() * 100)
              .with_columns(r5=pl.col("lr").rolling_sum(5, min_samples=5),
                            avail=pl.col("date").dt.offset_by(f"{avail_days}d"))
              .select("avail", "lr", "r5").drop_nulls().sort("avail"))


def build_features(mk: pl.DataFrame) -> pl.DataFrame:
    """mk: (date, nifty50_ret, india_vix, nf) on India trading days."""
    inr = _ret5(pl.read_parquet(VALIDATION_DATA / "usdinr.parquet")
                  .sort("date"), "usdinr", 0)
    spx = _ret5(pl.read_parquet(VALIDATION_DATA / "sp500.parquet")
                  .sort("date"), "sp500", 1)
    dxy = _ret5(_fred("DTWEXBGS", "dxy"), "dxy", 1)
    dff = (_fred("DFF", "ff")
           .with_columns(avail=pl.col("date").dt.offset_by("1d"))
           .select("avail", "ff").sort("avail"))
    # India call rate for month m enters from the 15th of month m+1
    icr = (_fred("IRSTCI01INM156N", "icr")
           .with_columns(avail=pl.col("date").dt.offset_by("1mo")
                                 .dt.offset_by("14d"))
           .select("avail", "icr").sort("avail"))

    df = mk.sort("date")
    rights = [inr.rename({"r5": "inr5"}).select("avail", "inr5"),
              spx.rename({"r5": "spx5", "lr": "spx1"})
                 .select("avail", "spx5", "spx1"),
              dxy.rename({"r5": "dxy5"}).select("avail", "dxy5"),
              dff, icr]
    for rt in rights:
        df = df.join_asof(rt, left_on="date", right_on="avail",
                          strategy="backward")
        df = df.drop([c for c in df.columns if c.startswith("avail")])
    return (df.with_columns(spxn1=pl.min_horizontal(pl.col("spx1"), 0.0),
                            diff=pl.col("icr") - pl.col("ff"))
              .with_columns(ddif=pl.col("diff") - pl.col("diff").shift(63))
              .with_columns(nf2=pl.col("nf").shift(2).fill_null(0.0)))


def main() -> None:
    t0 = time.time()
    mk = (pl.scan_parquet(VALIDATION_DATA / "returns_panel_v3.parquet")
            .select(["date", "nifty50_ret", "india_vix"])
            .unique(subset=["date"]).sort("date").collect()
            .with_columns(pl.col("india_vix").forward_fill())
            .filter(pl.col("nifty50_ret").is_not_null()
                    & pl.col("india_vix").is_not_null()))
    fl = build_flow()
    mk = (mk.join(fl, on="date", how="left")
            .with_columns(gmean=pl.col("gross")
                          .rolling_mean(GROSS_W, min_samples=GROSS_W // 2)
                          .shift(1))
            .with_columns(nf=pl.when(pl.col("net").is_not_null()
                                     & pl.col("gmean").is_not_null())
                             .then(pl.col("net") / pl.col("gmean"))
                             .otherwise(0.0)))
    df = build_features(mk)
    n_all = df.height
    dates = df["date"].to_numpy()
    ret = df["nifty50_ret"].to_numpy().astype(float)
    feats = ["inr5", "spx5", "spxn1", "dxy5", "ddif"]
    Xg = np.column_stack([df[c].to_numpy().astype(float) for c in feats])
    Xg = np.nan_to_num(Xg)                     # early-sample nulls -> neutral
    nf2 = df["nf2"].to_numpy().astype(float)
    Xf = np.column_stack([Xg, np.maximum(nf2, 0), np.minimum(nf2, 0)])
    cov = {c: int(df[c].is_not_null().sum()) for c in feats}
    print(f"frame: {n_all} trading days {dates[0]} .. {dates[-1]}; "
          f"feature coverage {cov}   [{time.time()-t0:.1f}s]")

    sig, hist = gjr_walk_forward(ret)
    w, a, g, b = hist[-1][1:]
    print(f"GJR walk-forward: {len(hist)} vintages, last "
          f"w={w:.2e} a={a:.3f} g={g:.3f} b={b:.3f}   [{time.time()-t0:.1f}s]")

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
        print(f"  fitted {name:13s} scored={int(np.isfinite(crps).sum())} "
              f"[{time.time()-t0:.1f}s]")

    tgt = np.r_[dates[1:], dates[-1]]
    era = {"FULL": np.ones(n_all, bool),
           "TRAIN": tgt <= np.datetime64(TRAIN_END),
           "TEST": tgt >= np.datetime64(TEST_START)}

    print("\n== mean losses (x100) ==")
    print(f"{'model':15s}" + "".join(
        f"{e+'-CRPS':>12s}{e+'-pin5':>12s}{e+'-pin1':>12s}" for e in era))
    for name, r in res.items():
        row = f"{name:15s}"
        for e, m in era.items():
            for k in ("crps", "pin5", "pin1"):
                row += f"{np.nanmean(r[k][m])*100:12.4f}"
        print(row)

    print("\n== VaR backtests (FULL) + economic metrics ==")
    print(f"{'model':15s}{'hit5%':>8s}{'kupiec':>8s}{'hit1%':>8s}"
          f"{'kupiec':>8s}{'chris1':>8s}{'cap@1%':>9s}{'ESgap':>8s}")
    kup = {}
    for name, r in res.items():
        h5, k5 = kupiec(r["hit5"], 0.05)
        h1, k1 = kupiec(r["hit1"], 0.01)
        c1 = christoffersen(r["hit1"])
        cap, esg = econ_metrics(y, r["scale"], r["nu"], era["FULL"] & valid)
        kup[name] = (k5, k1)
        print(f"{name:15s}{h5*100:7.2f}%{k5:8.3f}{h1*100:7.2f}%{k1:8.3f}"
              f"{c1:8.3f}{cap:8.3f}%{esg:7.3f}%")

    print("\n== DM tests on CRPS (negative t = first model better) ==")
    pairs = [("H1_global3", "H0_gjr_t"), ("H2_global5", "H0_gjr_t"),
             ("H2_global5", "H1_global3"), ("H3_glob_flow", "H2_global5")]
    dm = {}
    for a_, b_ in pairs:
        row = f"{a_} vs {b_:12s}"
        for e, m in era.items():
            t, nn = dm_test(res[a_]["crps"][m], res[b_]["crps"][m])
            dm[(a_, b_, e)] = t
            row += f"  {e}: t={t:+6.2f} (n={nn})"
        print(row)

    def gate(a_, b_):
        d = {e: np.nanmean(res[a_]["crps"][m]) - np.nanmean(res[b_]["crps"][m])
             for e, m in era.items()}
        t = dm[(a_, b_, "FULL")]
        ga = np.isfinite(t) and t <= -2.0
        gb = d["TRAIN"] < 0 and d["TEST"] < 0
        k5, k1 = kup[a_]
        gc = np.isfinite(k5) and np.isfinite(k1) and k5 > 0.01 and k1 > 0.01
        return ga, gb, gc, t, d

    print("\n== PRE-REGISTERED GATE (docs/PREREG_MARKET_ENGINE_V3.md) ==")
    ga, gb, gc, t, d = gate("H1_global3", "H0_gjr_t")
    print(f"  PRIMARY H1 vs H0: (a) DM t={t:+.2f} (need <=-2.0) "
          f"{'PASS' if ga else 'FAIL'}; (b) eras TRAIN {d['TRAIN']*100:+.4f} "
          f"TEST {d['TEST']*100:+.4f} {'PASS' if gb else 'FAIL'}; "
          f"(c) calibration {'PASS' if gc else 'FAIL'}")
    print(f"  PRIMARY GATE: {'PASS' if (ga and gb and gc) else 'FAIL'}")
    ga2, gb2, gc2, t2, d2 = gate("H2_global5", "H0_gjr_t")
    print(f"  SECONDARY H2 vs H0: (a) DM t={t2:+.2f} {'PASS' if ga2 else 'FAIL'}; "
          f"(b) eras TRAIN {d2['TRAIN']*100:+.4f} TEST {d2['TEST']*100:+.4f} "
          f"{'PASS' if gb2 else 'FAIL'}; (c) {'PASS' if gc2 else 'FAIL'}")
    print(f"  SECONDARY: {'PASS' if (ga2 and gb2 and gc2) else 'FAIL'}")

    out = pl.DataFrame(
        {"date": dates, "target_date": tgt, "ret_next": y, "sigma_gjr": sig,
         **{c: df[c].to_numpy() for c in feats}, "nf2": nf2,
         **{f"{k}_{n}": res[n][k] for n in MODELS
            for k in ("crps", "pin5", "pin1", "hit5", "hit1", "scale", "nu")}})
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(VALIDATION_OUT / "market_engine_v3_scores.parquet")
    print(f"\nwrote market_engine_v3_scores.parquet [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
