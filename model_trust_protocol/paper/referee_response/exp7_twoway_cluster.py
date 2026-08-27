"""EXP7. Equation (1) under two-way clustering.

Round-2 review 2.1: date clustering protects the cross-section but not
within-instrument serial dependence; the panel has both, so the reviewer asks
for two-way inference or a bootstrap.

Cameron-Gelbach-Miller: V_2way = V_date + V_instrument - V_white, since each
(instrument, date) cell holds exactly one observation, so the intersection
clustering is White. Reported alongside the published date-clustered t and a
date-block bootstrap.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/divyanshsingh/Desktop/Major Project 2/src")
from fii.validation import module19_institutional_share as m19  # noqa: E402
import polars as pl  # noqa: E402

j = m19.load()
j = j.with_columns(pl.col("share").shift(2).over("cisin").alias("share_L2"),
                   pl.col("vol").shift(2).over("cisin").alias("vol_L2"))
d = j.to_pandas()
_, day = np.unique(d["TR_DATE"].values, return_inverse=True)
_, stk = np.unique(d["cisin"].values, return_inverse=True)
era = d["era"].values
z2 = np.clip(d["z_h1"].values, -m19.CLIP, m19.CLIP) ** 2


def _meat(X, e, g):
    M = np.zeros((X.shape[1],) * 2)
    for k in np.unique(g):
        m = g == k
        z = X[m].T @ e[m]
        M += np.outer(z, z)
    return M


def fit(ycol, xcols, sub=None):
    m = np.ones(len(d), bool) if sub is None else sub
    cols = [d[c].values for c in xcols]
    ok = m & np.isfinite(ycol) & np.all([np.isfinite(c) for c in cols], axis=0)
    dd, ss = day[ok], stk[ok]
    f = lambda x: m19._demean2(x, dd, ss)          # stock + date FE
    y = f(ycol[ok])
    X = np.column_stack([np.ones(ok.sum())] + [f(c[ok]) for c in cols])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V_d = XtXi @ _meat(X, e, dd) @ XtXi
    V_s = XtXi @ _meat(X, e, ss) @ XtXi
    V_w = XtXi @ (X * (e**2)[:, None]).T @ X @ XtXi
    V_2 = V_d + V_s - V_w
    out = {}
    for lab, V in (("date", V_d), ("instrument", V_s), ("two-way", V_2), ("white", V_w)):
        se = np.sqrt(np.maximum(np.diag(V), 0))
        out[lab] = b[1] / se[1] if se[1] > 0 else np.nan
    # date-block bootstrap on the coefficient
    rng = np.random.default_rng(7)
    dates = np.unique(dd); nd = len(dates)
    idx_by_date = {k: np.flatnonzero(dd == k) for k in dates}
    bs = []
    for _ in range(300):
        sel, t = [], rng.integers(nd)
        while len(sel) < nd:
            L = rng.geometric(1/20.0)
            sel.extend(dates[(t + q) % nd] for q in range(L)); t = rng.integers(nd)
        rows = np.concatenate([idx_by_date[k] for k in sel[:nd]])
        Xb, yb = X[rows], y[rows]
        bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        bs.append(bb[1])
    out["block-boot"] = b[1] / np.std(bs, ddof=1)
    return b[1], out, ok.sum()


print("=" * 86)
print("EXP7 · EQUATION (1): t-STATISTIC ON share UNDER FOUR VARIANCE ESTIMATORS")
print("=" * 86)
print(f"{'spec':<26}{'coef':>10}{'date':>9}{'instr':>9}{'two-way':>9}{'white':>9}{'blockBS':>9}{'n':>10}")
for lab, e_ in (("FULL", None), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
    sub = None if e_ is None else (era == e_)
    c, o, n = fit(z2, ["share", "vol"], sub)
    print(f"{'contemporaneous '+lab:<26}{c:>10.4f}{o['date']:>9.2f}{o['instrument']:>9.2f}"
          f"{o['two-way']:>9.2f}{o['white']:>9.2f}{o['block-boot']:>9.2f}{n:>10,}")
for lab, e_ in (("FULL", None), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
    sub = None if e_ is None else (era == e_)
    c, o, n = fit(z2, ["share_L2", "vol_L2"], sub)
    print(f"{'t-2 '+lab:<26}{c:>10.4f}{o['date']:>9.2f}{o['instrument']:>9.2f}"
          f"{o['two-way']:>9.2f}{o['white']:>9.2f}{o['block-boot']:>9.2f}{n:>10,}")
