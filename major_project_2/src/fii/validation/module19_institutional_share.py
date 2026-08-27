"""
MODULE 19 · INSTITUTIONAL SHARE OF TURNOVER AND NEXT-DAY VOLATILITY

THE QUESTION.  Every flow feature in this project is a within-day
cross-sectional RANK, and every outcome is divided by an EWMA sigma. Between
them those two normalisations destroy exactly the information most likely to
matter: the LEVEL of institutional participation, and the LEVEL of
volatility. Once the archetype/concentration axis had been shown null on
every moment of the outcome distribution, the remaining question was whether
the COMPOSITION of a stock's trading -- how much of it is institutional --
says anything about its next-day volatility.

THE SPECIFICATION.  For stock i on day t,

    share_it = log(FII gross flow / trailing-60d mean)
             - log(turnover      / trailing-60d mean)
    vol_it   = log(turnover      / trailing-60d mean)

both DEMEANED WITHIN DAY, regressed on z_it^2 where z is the engine's own
standardised outcome for t -> t+1. Because z has already been divided by the
stock's EWMA sigma, any remaining predictability is volatility the engine's
sigma MISSED. Standard errors are clustered by date throughout.

Each control kills a specific objection:
  * date fixed effects  -> the comparison is between stocks ON THE SAME DAY,
    so no aggregate or market-wide story can explain it;
  * log(turnover)       -> volume-volatility (Karpoff 1987) is controlled, and
    enters strongly positive, so this is not that relationship in disguise;
  * stock fixed effects -> not driven by which names are in the sample;
  * TRAIN/TEST split    -> not an artifact of searching the whole sample.

WHAT IT DOES NOT SHOW.  The effect dies completely at a two-day lag. With
custodian reporting lags (1.5% of value same-day, 67% by T+1) it is therefore
NOT tradeable, and this module reports that as a headline rather than a
footnote. Direction of causality is also not identified: news at t could
drive both retail participation at t and volatility at t+1. The defensible
claim is predictive and compositional, not causal.

INPUTS   data/VALIDATION_DATA/fii_stockday_intensity.parquet  (module 20)
         outputs/phase3/outcome_z.parquet                     (C4)
NOTE     outcome_z must be built WITHOUT the C4 share-sigma correction
         (the default); otherwise this module measures its own residual.

Run:  python -m fii.validation.module19_institutional_share
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from fii.paths import OUTPUTS, VALIDATION_DATA

CLIP = 10.0          # winsorise z at the density grid edge; robustness varies it
NORM_W = 60          # must match module 20
MIN_N = 1000


# ----------------------------------------------------------------- helpers
def _demean(x, g, n=None):
    n = n or (int(g.max()) + 1)
    s = np.bincount(g, weights=x, minlength=n)
    c = np.bincount(g, minlength=n)
    return x - (s / np.maximum(c, 1))[g]


def _demean2(x, g1, g2, iters=12):
    """Iterated two-way demeaning — stock and date fixed effects."""
    for _ in range(iters):
        x = _demean(_demean(x, g1), g2)
    return x


def _cluster_reg(y, cols, g):
    """OLS with standard errors clustered on `g`. Returns [(b, t), ...]."""
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1],) * 2)
    for k in np.unique(g):
        m = g == k
        z = X[m].T @ e[m]
        meat += np.outer(z, z)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return [(b[i + 1], b[i + 1] / se[i + 1]) for i in range(len(cols))]


def load():
    it = pl.read_parquet(VALIDATION_DATA / "fii_stockday_intensity.parquet")
    it = it.filter((pl.col("gnorm") > 0) & (pl.col("tnorm") > 0)
                   & (pl.col("fii_gross") > 0) & (pl.col("turnover") > 0))
    it = it.with_columns(
        ((pl.col("fii_gross") / pl.col("gnorm")).log()
         - (pl.col("turnover") / pl.col("tnorm")).log()).alias("share"),
        (pl.col("turnover") / pl.col("tnorm")).log().alias("vol"))
    oz = pl.read_parquet(OUTPUTS / "phase3" / "outcome_z.parquet",
                         columns=["cisin", "TR_DATE", "era", "z_h1"])
    oz = oz.drop_nulls("z_h1").filter(pl.col("z_h1").is_finite())
    j = oz.join(it.select(pl.col("isin").alias("cisin"), "TR_DATE",
                          "share", "vol", "fii_gross", "turnover"),
                on=["cisin", "TR_DATE"], how="inner").sort(["cisin", "TR_DATE"])
    vix = (pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet",
                           columns=["date", "india_vix"])
             .drop_nulls().unique(subset=["date"]).rename({"date": "TR_DATE"}))
    return j.join(vix, on="TR_DATE", how="left")


def main() -> None:
    t0 = time.time()
    j = load()
    j = j.with_columns(pl.col("share").shift(2).over("cisin").alias("share_L2"),
                       pl.col("vol").shift(2).over("cisin").alias("vol_L2"),
                       (pl.col("z_h1") ** 2).shift(1).over("cisin").alias("z2_L1"),
                       (pl.col("z_h1") ** 2).shift(2).over("cisin").alias("z2_L2"))
    d = j.to_pandas()
    _, day = np.unique(d["TR_DATE"].values, return_inverse=True)
    _, stk = np.unique(d["cisin"].values, return_inverse=True)
    era = d["era"].values
    z2 = np.clip(d["z_h1"].values, -CLIP, CLIP) ** 2

    print("=" * 74)
    print("MODULE 19 · institutional share of turnover -> next-day volatility")
    print("=" * 74)
    print(f"stock-days: {len(d):,} | stocks {len(np.unique(stk)):,} | "
          f"dates {len(np.unique(day)):,}")
    for e in ("TRAIN", "TEST"):
        print(f"  {e}: {(era == e).sum():,}")

    def spec(lab, ycol, xcols, two_way=False, sub=None):
        m = np.ones(len(d), bool) if sub is None else sub
        cols = [d[c].values if isinstance(c, str) else c for c in xcols]
        ok = m & np.isfinite(ycol) & np.all([np.isfinite(c) for c in cols], axis=0)
        if ok.sum() < MIN_N:
            print(f"  {lab:<34}{'--':>10}{'--':>8}{ok.sum():>9,}")
            return
        dd, ss = day[ok], stk[ok]
        f = ((lambda x: _demean2(x, day[ok], stk[ok])) if two_way
             else (lambda x: _demean(x, dd)))
        r = _cluster_reg(f(ycol[ok]), [f(c[ok]) for c in cols], dd)
        print(f"  {lab:<34}{r[0][0]:>10.4f}{r[0][1]:>8.2f}{ok.sum():>9,}"
              f"   vol {r[1][0]:+.4f} (t {r[1][1]:+.1f})" if len(r) > 1 else "")

    print("\n1 · MAIN SPECIFICATIONS   (dependent variable: z^2)")
    print(f"  {'spec':<34}{'b(share)':>10}{'t':>8}{'n':>9}")
    for e, lab in ((None, "FULL"), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        sub = None if e is None else (era == e)
        spec(f"date FE                  [{lab}]", z2, ["share", "vol"], sub=sub)
    print()
    for e, lab in ((None, "FULL"), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        sub = None if e is None else (era == e)
        spec(f"stock + date FE          [{lab}]", z2, ["share", "vol"],
             two_way=True, sub=sub)
    print()
    for e, lab in ((None, "FULL"), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        sub = None if e is None else (era == e)
        spec(f"+ own lagged z^2         [{lab}]", z2,
             ["share", "vol", "z2_L1", "z2_L2"], sub=sub)
    print("\n  DEPLOYABILITY — the same test with the share known only at t-2")
    for e, lab in ((None, "FULL"), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        sub = None if e is None else (era == e)
        spec(f"share at t-2             [{lab}]", z2, ["share_L2", "vol_L2"],
             sub=sub)

    print("\n2 · ROBUSTNESS")
    print(f"  {'variant':<34}{'b(share)':>10}{'t':>8}{'n':>9}")
    for C in (4.0, 6.0, 10.0, np.inf):
        y = np.clip(d["z_h1"].values, -C, C) ** 2
        spec(f"winsorise |z| at {'none' if np.isinf(C) else C:<17}", y,
             ["share", "vol"], two_way=True)
    az = np.abs(np.clip(d["z_h1"].values, -CLIP, CLIP))
    for e, lab in ((None, "FULL"), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        sub = None if e is None else (era == e)
        spec(f"|z| instead of z^2       [{lab}]", az, ["share", "vol"],
             two_way=True, sub=sub)

    print("\n3 · MARKET REGIME — India VIX split (date FE)")
    v = d["india_vix"].values
    med = np.nanmedian(v)
    print(f"  {'regime':<34}{'b(share)':>10}{'t':>8}{'n':>9}")
    spec(f"LOW  VIX (< {med:.1f})", z2, ["share", "vol"], sub=v <= med)
    spec(f"HIGH VIX (>= {med:.1f})", z2, ["share", "vol"], sub=v > med)

    print("\n4 · ECONOMIC SIZE — within-day quintiles of institutional share")
    sh = d["share"].values
    q = np.empty(len(sh))
    for k in np.unique(day):
        m = day == k
        q[m] = np.searchsorted(np.quantile(sh[m], [.2, .4, .6, .8]), sh[m], "right")
    print(f"  {'quintile':<12}{'n':>9}{'FII/turnover':>14}{'E[z^2]':>10}"
          f"{'implied sigma':>15}")
    rows = []
    for k in range(5):
        m = q == k
        r = float(np.nanmean(d["fii_gross"].values[m] / d["turnover"].values[m]))
        e2 = float(np.nanmean(z2[m]))
        rows.append(e2)
        print(f"  Q{k + 1:<11}{int(m.sum()):>9,}{r:>14.3f}{e2:>10.4f}"
              f"{np.sqrt(e2):>15.4f}")
    print(f"\n  Q1 - Q5 spread in E[z^2]: {rows[0] - rows[4]:+.4f}"
          f"   -> lowest-share stock-days are "
          f"{100 * (np.sqrt(rows[0] / rows[4]) - 1):+.1f}% more volatile")

    print("\n" + "=" * 74)
    print("READ.  A negative b(share) that holds in BOTH eras under stock and")
    print("date fixed effects, with log(turnover) controlled, is the claim:")
    print("the institutional share of a stock's trading carries cross-sectional")
    print("information about its next-day volatility beyond total volume and")
    print("beyond the stock's own EWMA sigma. The t-2 row is the honest limit:")
    print("if it is null, the finding is economic, not tradeable.")
    print(f"total {time.time() - t0:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
