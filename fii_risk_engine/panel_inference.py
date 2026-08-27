"""
PANEL-VALID INFERENCE AND CONDITIONAL CALIBRATION  —  audit items 10 and 11

Shared by C6 and C7 so the two cannot report different numbers for the same
quantity.

ITEM 10 — WHY THE OLD p-VALUES DO NOT HOLD.  Loss differentials on this panel
are dependent in two directions at once: every stock shares a market factor on
a given day, and an h-day forward window overlaps the next h-1 of them by
construction. Measured on the pre-audit build, at h=1:

    procedure                                   t-statistic
    naive iid over pooled stock-days              -76.62
    date means + Newey-West(h+5)   [C6's]          -9.66
    stationary block bootstrap over dates          -8.02

Naive pooling inflates by 9.6x. C6's date-averaging plus HAC survives that
comparison and is retained; what does NOT survive is the DISTRIBUTIONAL
machinery:

  * PIT chi-square. The naive statistic was 47.8 against a textbook 9-df
    critical value of 16.9, which would reject. Under a date-block bootstrap
    null the statistic's own median is 209.7 and its 95th percentile 818.1 --
    the reference distribution is wrong by roughly 23x at the median, and 47.8
    rejects nothing. C7 patched this by rescaling bin counts by
    neff/len(u), a linear correction that is approximately right for a mean
    and not for a chi-square of binned counts.
  * Kupiec with fractional counts. `kupiec_p(rate * neff, neff, a)` feeds
    non-integer successes into a binomial likelihood ratio. It is a
    plausibility adjustment, not a test.

Both are replaced here by bootstrap distributions of the statistic itself, so
the critical value comes from the data's own dependence rather than from an
assumption it violates.

ITEM 11 — CONDITIONAL CALIBRATION.  A pooled headline is not evidence of
calibration. On the pre-audit build the pooled 1% breach rate was 1.04% while
the COVID window ran at 7.22%, and breach rates fell monotonically from 6.18%
to 3.72% across EWMA-volatility quintiles at the 5% level -- a gradient that
should not exist at all, since outcomes are already divided by that same
volatility. `conditional_report` makes those strata mandatory rather than
optional.
"""
from __future__ import annotations

import numpy as np

BLOCK = 20          # mean block length in trading days for the stationary
                    # bootstrap; ~one month, comfortably longer than h=20's
                    # overlap and than the observed volatility clustering
NBOOT = 2000


def _date_blocks(n_dates, rng, block=BLOCK):
    """Stationary (Politis-Romano) bootstrap index over the date axis."""
    sel, t = [], rng.integers(n_dates)
    while len(sel) < n_dates:
        L = rng.geometric(1.0 / block)
        sel.extend((t + j) % n_dates for j in range(L))
        t = rng.integers(n_dates)
    return np.asarray(sel[:n_dates])


def block_bootstrap(values, day_id, stat=np.mean, nboot=NBOOT, seed=7,
                    block=BLOCK):
    """Bootstrap distribution of `stat` under date-block resampling.

    Resampling whole DATES keeps same-day cross-sectional dependence intact
    (every stock on a resampled date travels together); drawing them in blocks
    keeps serial dependence and the overlap of forward windows.

    For the mean — which is every statistic this module needs (breach rates,
    CRPS differentials) — the draw is computed from PER-DATE SUMS AND COUNTS
    rather than by materialising the resampled rows. A resampled mean is
    sum(date sums) / sum(date counts), which is algebraically identical and
    turns an O(rows) inner loop into O(dates): on this panel that is ~550,000
    down to ~2,000 per draw. Any other `stat` falls back to the explicit path.
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, d = v[ok], np.asarray(day_id)[ok]
    if len(v) == 0:
        return np.array([]), np.nan
    ud, di = np.unique(d, return_inverse=True)
    nd = len(ud)
    rng = np.random.default_rng(seed)

    if stat is np.mean:
        sums = np.bincount(di, weights=v, minlength=nd)
        cnts = np.bincount(di, minlength=nd).astype(float)
        out = np.empty(nboot)
        for b in range(nboot):
            sel = _date_blocks(nd, rng, block)
            c = cnts[sel].sum()
            out[b] = sums[sel].sum() / c if c > 0 else np.nan
        return out, float(v.mean())

    idx = [np.where(di == k)[0] for k in range(nd)]
    out = np.empty(nboot)
    for b in range(nboot):
        take = np.concatenate([idx[k] for k in _date_blocks(nd, rng, block)])
        out[b] = stat(v[take])
    return out, float(stat(v))


def boot_ci(values, day_id, stat=np.mean, alpha=0.05, **kw):
    """Point estimate and percentile CI under date-block resampling."""
    dist, point = block_bootstrap(values, day_id, stat, **kw)
    if not len(dist):
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "p_two_sided": np.nan}
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    # two-sided p for "the statistic is zero", by inversion
    p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(1.0, p)), "boot_sd": float(dist.std())}


def breach_rate_ci(breach, day_id, alpha_level, **kw):
    """Breach rate with a dependence-aware interval, replacing Kupiec.

    Returns the rate, its CI, and whether the nominal level sits inside it --
    the honest version of "the model passes at 1%".
    """
    b = np.asarray(breach, float)
    r = boot_ci(b, day_id, np.mean, **kw)
    r["nominal"] = alpha_level
    r["covers_nominal"] = bool(r["lo"] <= alpha_level <= r["hi"])
    r["ratio_to_nominal"] = (r["point"] / alpha_level
                             if alpha_level else np.nan)
    return r


def pit_chi2_bootstrap(u, day_id, bins=10, nboot=500, seed=11, block=BLOCK):
    """PIT uniformity, scored against a NULL-CENTRED block distribution.

    THE BUG THIS REPLACES.  The previous version block-resampled the OBSERVED
    PIT bin counts and took the 95th percentile of the resulting chi-squares
    as the critical value. That reproduces whatever non-uniform distribution
    the model actually produced, so the reference distribution is centred on
    the alternative rather than on the null and the critical value rises in
    lockstep with the statistic. It has essentially zero power. Measured on
    synthetic data with 600,000 draws over 2,000 dates:

        PIT sample                              chi2        crit95   reject?
        uniform                                 16.8          41.0     no
        u^2 (mass piled at 0)                  343,485      345,970    no
        Beta(0.5, 0.5)                         169,469      171,381    no
        all mass in the bottom half            600,006      600,029    no
        ALL MASS IN ONE BIN                  5,400,000    5,400,000    no

    It could not reject a PIT concentrated entirely in a single bin. Every
    "PIT not rejected" verdict produced with it is void.

    THE REPLACEMENT.  Bin counts are decomposed per date into an expected
    count under uniformity (n_d / bins) and a residual. The residuals are
    then RECENTRED across dates, which imposes H0 by construction, and it is
    the recentred residuals that are block-resampled. The null therefore
    carries the panel's real dependence — same-day cross-sectional clustering
    and serial correlation — while being centred where the null says it
    should be. A systematic departure from uniformity now shows up as a large
    observed statistic against a null that does not move with it.
    """
    u = np.asarray(u, float)
    ok = np.isfinite(u)
    u, d = u[ok], np.asarray(day_id)[ok]
    if len(u) < 200:
        return {"chi2": np.nan, "crit95": np.nan, "reject": None, "n": len(u)}

    ud, di = np.unique(d, return_inverse=True)
    nd = len(ud)
    b_idx = np.clip((u * bins).astype(int), 0, bins - 1)
    cnt = np.zeros((nd, bins))
    np.add.at(cnt, (di, b_idx), 1.0)

    n_d = cnt.sum(axis=1, keepdims=True)          # observations per date
    exp_d = n_d / bins                            # expected under uniformity
    resid = cnt - exp_d                           # zero-sum across bins
    # observed statistic: total residual per bin, scaled by total expectation
    R = resid.sum(axis=0)
    E = float(exp_d.sum())
    stat = float((R * R).sum() / E) if E > 0 else np.nan

    # H0 imposed: recentre so each bin's residual has mean zero across dates
    resid0 = resid - resid.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    null = np.empty(nboot)
    for b in range(nboot):
        sel = _date_blocks(nd, rng, block)
        Rb = resid0[sel].sum(axis=0)
        Eb = float(exp_d[sel].sum())
        null[b] = (Rb * Rb).sum() / Eb if Eb > 0 else np.nan
    crit = float(np.nanquantile(null, 0.95))
    return {"chi2": stat, "crit95": crit,
            "null_median": float(np.nanmedian(null)),
            "reject": bool(stat > crit), "n": int(len(u)),
            "textbook_crit95": 16.919}


# --------------------------------------------------------------------------
# ITEM 11 · conditional calibration
# --------------------------------------------------------------------------
def quintiles(x, labels=None):
    """Index 0-4 by quintile of x; -1 where x is not finite."""
    x = np.asarray(x, float)
    out = np.full(len(x), -1)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return out
    q = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    out[ok] = np.searchsorted(q, x[ok], side="right")
    return out


def expanding_quintiles(x, group, min_obs=60):
    """Quintile of x within its group, using only that group's OWN PAST.

    `quintiles` above cuts on the WHOLE sample, which for a volatility
    stratum means asking "is today in the calmest fifth of this stock's
    history, including its future?" That is forward-looking information, and
    a causal model cannot price it: knowing a day sits in the bottom fifth of
    a window that extends past it genuinely predicts that volatility will
    rise. Conditioning on it manufactures a gradient in a model that has none.

    Measured on this engine at h=1, 5%: the full-sample stratum shows breach
    rates of 1.16x nominal in Q1 falling to 0.85x in Q5 (spread 0.30), while
    the causal stratum below shows 1.02 / 1.05 / 1.03 / 0.99 / 1.06 — spread
    -0.04, i.e. flat. Same engine, same rows, same forecasts. The difference
    is entirely in what the stratum is allowed to know.

    Both are reported. The full-sample cut is a legitimate ex-post question
    ("how did it do in what turned out to be quiet regimes?"); it is just not
    a statement about calibration.
    """
    x = np.asarray(x, float)
    out = np.full(len(x), -1)
    g = np.asarray(group)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        v = x[idx]
        ok = np.isfinite(v)
        if ok.sum() < min_obs:
            continue
        vi = idx[ok]
        w = x[vi]
        # share of the group's own past strictly below today, inclusive of t
        below = (w[None, :] < w[:, None]).sum(axis=1)
        tri = (np.arange(len(w))[None, :] <= np.arange(len(w))[:, None])
        below = ((w[None, :] < w[:, None]) & tri).sum(axis=1)
        frac = below / np.arange(1, len(w) + 1)
        out[vi] = np.clip((frac * 5).astype(int), 0, 4)
    return out


def crisis_mask(dates, windows):
    """True where a date falls inside any crisis window.

    `windows` is an iterable of (label, start, end) — the project already
    carries these in data/VALIDATION_DATA/crisis_windows.csv.
    """
    d = np.asarray(dates)
    m = np.zeros(len(d), dtype=bool)
    for _lab, s, e in windows:
        m |= (d >= np.datetime64(s)) & (d <= np.datetime64(e))
    return m


def conditional_report(breach, day_id, dates, strata: dict, alpha_level,
                       min_n=500, **kw):
    """Breach rate with a bootstrap CI inside every stratum.

    `strata` maps a label to a boolean mask. Every stratum is reported
    regardless of outcome; a stratum smaller than `min_n` is reported with its
    size and no interval rather than silently dropped.
    """
    rows = []
    b = np.asarray(breach, float)
    for label, mask in strata.items():
        mask = np.asarray(mask, bool)
        n = int((mask & np.isfinite(b)).sum())
        if n < min_n:
            rows.append({"stratum": label, "n": n, "rate": np.nan,
                         "note": f"below min_n={min_n}"})
            continue
        r = breach_rate_ci(b[mask], np.asarray(day_id)[mask], alpha_level, **kw)
        rows.append({"stratum": label, "n": n, "rate": r["point"],
                     "lo": r["lo"], "hi": r["hi"],
                     "ratio_to_nominal": r["ratio_to_nominal"],
                     "covers_nominal": r["covers_nominal"]})
    return rows


def render_conditional(rows, alpha_level, title):
    L = [f"\n{title}  (nominal {alpha_level:.0%})",
         f"  {'stratum':34s}{'n':>9}{'rate':>9}{'95% CI':>19}"
         f"{'x nominal':>11}  flag"]
    for r in rows:
        if not np.isfinite(r.get("rate", np.nan)):
            L.append(f"  {r['stratum']:34s}{r['n']:>9,}"
                     f"        —  {r.get('note',''):>28s}")
            continue
        flag = "" if r["covers_nominal"] else "MISCALIBRATED"
        L.append(f"  {r['stratum']:34s}{r['n']:>9,}{r['rate']:>9.4f}"
                 f"   [{r['lo']:.4f}, {r['hi']:.4f}]"
                 f"{r['ratio_to_nominal']:>11.2f}  {flag}")
    return "\n".join(L)
