"""WorldQuant-BRAIN-style daily cross-sectional L/S backtest engine.

Extracted verbatim from module12a_bt_engine.py (functions unchanged).
Correctness gates live in fii/backtest/engine_gates.py and run as a
pipeline stage; the engine must never be modified without re-running
them (G1 exactness, G2 alignment/no-look-ahead, G3 cost accounting).
"""
import numpy as np

ANN = 252.0


def weights_from_alpha(A, maxw=0.05, iters=4):
    """A: T x N alpha, NaN = out of universe. Returns weights:
    demeaned per day over valid names, sum|w| = 1, |w| <= maxw."""
    W = A.astype(float).copy()
    valid = np.isfinite(W)
    W[~valid] = np.nan
    for _ in range(iters):
        mu = np.nanmean(W, axis=1, keepdims=True)
        W = W - mu
        g = np.nansum(np.abs(W), axis=1, keepdims=True)
        g[g == 0] = np.nan
        W = W / g
        W = np.clip(W, -maxw, maxw)
    # final renorm after last clip
    g = np.nansum(np.abs(W), axis=1, keepdims=True)
    g[g == 0] = np.nan
    W = W / g
    W[~valid] = np.nan
    return W


def backtest(A, R, lag=1, tcost_bps=15.0, maxw=0.05):
    """A: T x N alpha (signal known at close of day t).
    R: T x N close-to-close return OF day t (close t-1 -> close t).
    Held weights on day t are the weights formed at close t-1-lag.
    Returns dict with daily series (pnl gross/net, turnover)."""
    T, N = A.shape
    W = weights_from_alpha(A, maxw=maxw)
    Wf = np.nan_to_num(W, nan=0.0)
    H = np.zeros((T, N))
    s = 1 + lag
    if T > s:
        H[s:] = Wf[:T - s]
    Rf = np.nan_to_num(R, nan=0.0)
    pnl_g = np.sum(H * Rf, axis=1)
    to = np.sum(np.abs(np.diff(H, axis=0, prepend=np.zeros((1, N)))),
                axis=1)
    pnl_n = pnl_g - (tcost_bps * 1e-4) * to
    return {"pnl_gross": pnl_g, "pnl_net": pnl_n, "turnover": to,
            "H": H, "W": W, "lag": lag, "tcost_bps": tcost_bps}


def metrics(pnl, to, live=None):
    """pnl, to: daily arrays. live: boolean mask of evaluated days."""
    if live is None:
        live = np.abs(to) + np.abs(pnl) > 0
    x = pnl[live]
    t = to[live]
    n = len(x)
    if n < 20:
        return {"n": n}
    mu, sd = x.mean(), x.std(ddof=1)
    dn = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    cum = np.cumsum(x)
    dd = cum - np.maximum.accumulate(cum)
    mdd = -dd.min()
    annret = mu * ANN
    shp = mu / sd * np.sqrt(ANN) if sd > 0 else np.nan
    srt = mu / dn * np.sqrt(ANN) if dn > 0 else np.nan
    cal = annret / mdd if mdd > 0 else np.nan
    tavg = t.mean()
    marg = (mu / tavg * 1e4) if tavg > 0 else np.nan
    fit = shp * np.sqrt(abs(annret) / max(tavg, 0.125)) \
        if np.isfinite(shp) else np.nan
    return {"n": n, "annret_%": 100 * annret, "annvol_%":
            100 * sd * np.sqrt(ANN), "sharpe": shp, "sortino": srt,
            "maxDD_%": 100 * mdd, "calmar": cal, "turnover": tavg,
            "margin_bps": marg, "fitness": fit,
            "hit_%": 100 * float((x > 0).mean())}


def show(tag, m):
    if m.get("n", 0) < 20:
        print(f"  {tag}: n={m.get('n',0)} (too few days)")
        return
    print(f"  {tag}: Sharpe {m['sharpe']:+.2f}  Sortino {m['sortino']:+.2f}"
          f"  ann {m['annret_%']:+.1f}%  vol {m['annvol_%']:.1f}%")
    print(f"      maxDD {m['maxDD_%']:.1f}%  Calmar "
          f"{('%+.2f' % m['calmar']) if np.isfinite(m['calmar']) else 'na'}"
          f"  turnover {m['turnover']:.2f}/d  margin "
          f"{m['margin_bps']:+.1f}bps  fitness {m['fitness']:+.2f}"
          f"  hit {m['hit_%']:.1f}%  n={m['n']}")


def daily_ic(A, R, lag=1):
    """mean daily Spearman corr of alpha(t) with return of day t+1+lag."""
    from scipy.stats import spearmanr
    T = A.shape[0]
    s = 1 + lag
    ics = []
    for t in range(T - s):
        a, r = A[t], R[t + s]
        m = np.isfinite(a) & np.isfinite(r)
        if m.sum() >= 30:
            ics.append(spearmanr(a[m], r[m]).statistic)
    ics = np.array(ics, float)
    ics = ics[np.isfinite(ics)]
    tt = ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))
    return ics.mean(), tt, len(ics)


