"""
MODULE C6 · PRE-REGISTERED VALIDATION  —  Phase III

Runs the tests locked in docs/PHASE3_PREREG.md Amendments 2, 3 and 3a.

STRATA (locked before any outcome was computed)
    PRIMARY   pa_max < 0.70    archetype-uncertain days
    CO        p_max  < 0.70    state-uncertain days
    SECONDARY full panel
    TERTIARY  pa_max in [0.70, 0.90]   monotonicity check

COMPARISONS
    soft_roll vs hard_roll   does soft labelling beat a forced hard call?
    soft_roll vs clim_roll   does conditioning on the archetype help at all?
    soft_roll vs normal      does the apparatus beat an EWMA-normal?
    soft_roll vs t_ewma      ... and a FAIR parametric law that already knows
                             the tail is fat (Student-t, nu fitted per vintage)
    clim_roll vs t_ewma      empirical-vs-parametric with the flow signal off,
                             which is the comparison that isolates the density

WHY NOT A TEXTBOOK DIEBOLD-MARIANO.  The loss differentials live on a panel
and are dependent in two directions at once: cross-sectionally (every stock
shares a market factor on a given day) and serially (an h-day forward window
overlaps the next h-1 of them by construction). A DM test applied to the
pooled row-level differential would treat ~45,000 correlated observations as
independent and produce a t-statistic inflated by roughly the square root of
the average cluster size. Here the differential is first averaged WITHIN each
date, which absorbs the cross-sectional dependence, and the resulting daily
series is then tested with a Newey-West HAC variance at lag h+5, which
absorbs the overlap. The unit of inference is the day, not the stock-day.

WHY CHRISTOFFERSEN ONLY AT h=1.  The independence test asks whether VaR
violations cluster in time. At h=5 and h=20 consecutive forward windows share
h-1 days of returns, so violations MUST cluster whatever the model does —
the test would reject a perfect model. It is therefore run at h=1 only, where
windows are disjoint, and reported as not-applicable elsewhere.

MULTIPLICITY.  The primary family is 3 comparisons x 3 horizons on the
PRIMARY stratum = 9 tests, Bonferroni alpha = 0.05/9 = 0.00556. All other
strata are reported without correction and are explicitly secondary.

Input : outputs/phase3/predictive.parquet
Output: outputs/phase3/c6_results.parquet  + full console report
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.stats import chi2, norm

from panel_inference import breach_rate_ci, pit_chi2_bootstrap

from paths import OUTPUTS

HORIZONS = [1, 5, 20]
PAIRS = [("soft_roll", "hard_roll"), ("soft_roll", "clim_roll"),
         ("soft_roll", "normal"), ("soft_roll", "t_ewma"),
         ("clim_roll", "t_ewma")]
ALPHA_FAMILY = 0.05 / 9
MIN_DAY_N = 3

t0 = time.time()
d = pl.read_parquet(OUTPUTS / "phase3" / "predictive.parquet")
print(f"predictive rows {d.height:,}")

pa = d["pa_max"].to_numpy()
pm = d["p_max"].to_numpy()
STRATA = {
    "PRIMARY  pa_max<0.70": pa < 0.70,
    "CO       p_max <0.70": pm < 0.70,
    "TERTIARY pa 0.70-0.90": (pa >= 0.70) & (pa < 0.90),
    "SECONDARY full panel": np.ones(d.height, bool),
}
for k, m in STRATA.items():
    print(f"  {k:24s} n = {m.sum():>8,}")

dates = d["TR_DATE"].to_numpy()
udates, day_id = np.unique(dates, return_inverse=True)
ND = len(udates)


def nw_var(x, lag):
    """Newey-West long-run variance of a mean."""
    n = len(x)
    xc = x - x.mean()
    g = (xc @ xc) / n
    for l in range(1, min(lag, n - 1) + 1):
        gl = (xc[l:] @ xc[:-l]) / n
        g += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    return max(g, 1e-300) / n


def dm_daily(diff, mask, lag):
    """Average the loss differential within each date, then HAC-test the
    resulting daily series. Returns (mean_diff, t, p, n_days)."""
    ok = mask & np.isfinite(diff)
    if ok.sum() < 50:
        return np.nan, np.nan, np.nan, 0
    s = np.bincount(day_id[ok], weights=diff[ok], minlength=ND)
    c = np.bincount(day_id[ok], minlength=ND)
    keep = c >= MIN_DAY_N
    if keep.sum() < 30:
        return np.nan, np.nan, np.nan, int(keep.sum())
    dd = s[keep] / c[keep]
    m = float(dd.mean())
    se = np.sqrt(nw_var(dd, lag))
    t = m / se
    return m, t, 2.0 * (1.0 - norm.cdf(abs(t))), int(keep.sum())


def icc_neff(ind, mask):
    """Effective sample size for a clustered binary indicator, via the
    one-way (by date) intraclass correlation."""
    ok = mask & np.isfinite(ind)
    n = int(ok.sum())
    if n < 100:
        return n, 1.0
    g = day_id[ok]
    cnt = np.bincount(g, minlength=ND)
    live = cnt > 0
    sm = np.bincount(g, weights=ind[ok], minlength=ND)
    mbar = cnt[live].mean()
    grand = ind[ok].mean()
    msb = (cnt[live] * (sm[live] / cnt[live] - grand) ** 2).sum() \
        / max(live.sum() - 1, 1)
    msw = ((ind[ok] - (sm / np.maximum(cnt, 1))[g]) ** 2).sum() \
        / max(n - live.sum(), 1)
    rho = (msb - msw) / max(msb + (mbar - 1) * msw, 1e-12)
    rho = float(np.clip(rho, 0.0, 0.99))
    deff = 1.0 + (mbar - 1.0) * rho
    return n / deff, rho


def kupiec(x, n, a):
    if n <= 0:
        return np.nan, np.nan, np.nan
    pi = x / n
    if x == 0:
        lr = -2.0 * n * np.log(1 - a)
    elif x == n:
        lr = -2.0 * n * np.log(a)
    else:
        lr = -2.0 * ((n - x) * np.log(1 - a) + x * np.log(a)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return pi, lr, 1.0 - chi2.cdf(lr, 1)


def christoffersen_ind(seq):
    """LR test that violations are serially independent (per stock)."""
    a, b = seq[:-1], seq[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0.0, 1.0) or p01 in (0.0,) or p11 in (0.0, 1.0):
        return np.nan, np.nan
    ll_r = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_u = (n00 * np.log(1 - p01) + n01 * np.log(p01)
            + n10 * np.log(1 - p11) + n11 * np.log(p11))
    lr = -2.0 * (ll_r - ll_u)
    return lr, 1.0 - chi2.cdf(lr, 1)


rows = []

# ============================ 1 · DIEBOLD-MARIANO ===========================
print("\n" + "=" * 78)
print("1 · DIEBOLD-MARIANO ON CRPS  (positive mean diff = the SECOND system "
      "is worse,")
print("    i.e. soft_roll wins).  Daily-aggregated, Newey-West HAC lag h+5.")
print(f"    Primary family Bonferroni alpha = {ALPHA_FAMILY:.5f}")
for sname, smask in STRATA.items():
    print(f"\n  --- {sname}  (n={smask.sum():,}) ---")
    print(f"    {'comparison':26s}{'h':>4}{'mean diff':>13}{'t':>9}"
          f"{'p':>10}{'days':>7}  verdict")
    for h in HORIZONS:
        for A, B in PAIRS:
            ca = d[f"{A}_h{h}_crps"].to_numpy()
            cb = d[f"{B}_h{h}_crps"].to_numpy()
            diff = cb - ca                       # >0 means A (soft) is better
            m, t, p, nd = dm_daily(diff, smask, h + 5)
            prim = sname.startswith("PRIMARY")
            thr = ALPHA_FAMILY if prim else 0.05
            if not np.isfinite(p):
                verdict = "insufficient"
            elif p < thr:
                verdict = f"{A} WINS" if m > 0 else f"{B} WINS"
            else:
                verdict = "no difference"
            print(f"    {A + ' vs ' + B:26s}{h:>4}{m:>13.6f}{t:>9.2f}"
                  f"{p:>10.4f}{nd:>7}  {verdict}")
            rows.append(dict(test="DM", stratum=sname, horizon=h,
                             a=A, b=B, stat=t, pval=p, effect=m, n=nd))

# ============================ 2 · KUPIEC ====================================
print("\n" + "=" * 78)
print("2 · KUPIEC UNCONDITIONAL COVERAGE  (VaR breach rate vs nominal)")
print("    p_naive treats stock-days as independent; p_adj uses the effective")
print("    sample size after the by-date intraclass correlation.")
for sname, smask in STRATA.items():
    print(f"\n  --- {sname} ---")
    print(f"    {'system':11s}{'h':>4}{'a':>6}{'breach':>9}{'nominal':>9}"
          f"{'n':>9}{'ICC':>7}{'n_eff':>9}{'p_naive':>9}{'p_adj':>8}"
          f"{'  block-bootstrap 95% CI':>26}")
    for h in HORIZONS:
        for sysn in ("soft_roll", "hard_roll"):
            for a, tag in ((0.05, "05"), (0.01, "01")):
                z = d[f"z_h{h}"].to_numpy()
                v = d[f"{sysn}_h{h}_var{tag}"].to_numpy()
                ok = smask & np.isfinite(z) & np.isfinite(v)
                ind = np.where(ok, (z < v).astype(float), np.nan)
                n = int(ok.sum())
                if n < 100:
                    continue
                x = int(np.nansum(ind))
                # AUDIT ITEM 10 — the binding statement is the block-bootstrap
                # interval; kupiec(x * neff / n, neff, a) below feeds a
                # FRACTIONAL success count into a binomial likelihood ratio and
                # is retained only for continuity with the pre-audit tables.
                bci = breach_rate_ci(ind[ok], day_id[ok], a,
                                     nboot=1000, seed=7)
                pi, lr, pv = kupiec(x, n, a)
                neff, rho = icc_neff(ind, smask)
                _, lr2, pv2 = kupiec(x * neff / n, neff, a)
                cov = "" if bci["covers_nominal"] else " MISCAL"
                print(f"    {sysn:11s}{h:>4}{a:>6.2f}{pi:>9.4f}{a:>9.2f}"
                      f"{n:>9,}{rho:>7.3f}{neff:>9,.0f}{pv:>9.4f}{pv2:>8.4f}"
                      f"   [{bci['lo']:.4f}, {bci['hi']:.4f}]{cov}")
                rows.append(dict(test="kupiec", stratum=sname, horizon=h,
                                 a=sysn, b=f"alpha{tag}", stat=lr, pval=pv2,
                                 effect=pi - a, n=n,
                                 boot_lo=bci["lo"], boot_hi=bci["hi"],
                                 boot_covers_nominal=bci["covers_nominal"]))

# ============================ 3 · CHRISTOFFERSEN ============================
print("\n" + "=" * 78)
print("3 · CHRISTOFFERSEN INDEPENDENCE  (h=1 only — at h=5/20 the forward")
print("    windows overlap by construction, so violations must cluster and")
print("    the test would reject a perfect model)")
cis = d["cisin"].to_numpy()
for sname, smask in list(STRATA.items()):
    for sysn in ("soft_roll", "hard_roll"):
        z = d["z_h1"].to_numpy()
        v = d[f"{sysn}_h1_var05"].to_numpy()
        ok = smask & np.isfinite(z) & np.isfinite(v)
        stks, cnt = np.unique(cis[ok], return_counts=True)
        floor = 250 if smask.all() else 60
        elig = stks[cnt >= floor]
        ps = []
        for s in elig:
            m = ok & (cis == s)
            idx = np.argsort(dates[m])
            seq = (z[m] < v[m]).astype(int)[idx]
            _, p = christoffersen_ind(seq)
            if np.isfinite(p):
                ps.append(p)
        if ps:
            ps = np.array(ps)
            rej = float((ps < 0.05).mean())
            print(f"  {sname:24s} {sysn:11s} stocks>={floor} tested {len(ps):>4} | "
                  f"reject@5% {rej:6.1%} (nominal 5.0%) | median p "
                  f"{np.median(ps):.3f}")
            rows.append(dict(test="christoffersen", stratum=sname, horizon=1,
                             a=sysn, b="ind", stat=rej,
                             pval=float(np.median(ps)), effect=rej - 0.05,
                             n=len(ps)))

# ============================ 4 · PIT UNIFORMITY ============================
print("\n" + "=" * 78)
print("4 · PIT UNIFORMITY  (10 bins; chi-square on the effective sample)")
for sname, smask in STRATA.items():
    print(f"\n  --- {sname} ---")
    print(f"    {'system':11s}{'h':>4}{'mean':>8}{'P(u<.05)':>10}"
          f"{'P(u>.95)':>10}{'n_eff':>9}{'chi2':>9}{'p_adj':>9}"
          f"{'chi2_raw':>11}{'crit95_bs':>11}  verdict")
    for h in HORIZONS:
        for sysn in ("soft_roll", "hard_roll", "clim_roll", "normal",
                     "t_ewma"):
            u = d[f"{sysn}_h{h}_pit"].to_numpy()
            ok = smask & np.isfinite(u)
            n = int(ok.sum())
            if n < 200:
                continue
            uu = u[ok]
            cnts, _ = np.histogram(uu, bins=10, range=(0, 1))
            binned = np.full(len(u), np.nan)
            binned[ok] = np.floor(np.clip(uu, 0, 0.999999) * 10) / 9.0
            neff, _ = icc_neff(binned, smask)
            scale = neff / n
            exp = neff / 10.0
            c2 = float((((cnts * scale) - exp) ** 2 / exp).sum())
            pv = 1.0 - chi2.cdf(c2, 9)
            # AUDIT ITEM 10 — chi2(9) is the wrong reference under panel
            # dependence (measured null median 209.7, 95th pct 818.1 against a
            # textbook 16.9). Score the statistic against its own null.
            _pb = pit_chi2_bootstrap(u[ok], day_id[ok], nboot=300, seed=11)
            _rej = "REJECT" if _pb["reject"] else ""
            print(f"    {sysn:11s}{h:>4}{uu.mean():>8.3f}"
                  f"{(uu < .05).mean():>10.3f}{(uu > .95).mean():>10.3f}"
                  f"{neff:>9,.0f}{c2:>9.1f}{pv:>9.4f}"
                  f"{_pb['chi2']:>11.1f}{_pb['crit95']:>11.1f}  {_rej}")
            rows.append(dict(test="pit", stratum=sname, horizon=h, a=sysn,
                             b="uniform", stat=c2, pval=pv,
                             effect=float(uu.mean() - 0.5), n=n,
                             chi2_raw=_pb["chi2"], crit95_boot=_pb["crit95"],
                             reject_boot=_pb["reject"]))

# ================= 5 · THE FORCED HARD CALL =================================
print("\n" + "=" * 78)
print("5 · WHAT A FORCED HARD CALL COSTS")
print("    CRPS(hard) - CRPS(soft) by archetype confidence. If hedging pays,")
print("    this should rise as pa_max falls.")
W = d.select([c for c in d.columns if c.startswith("pa_")
              and c != "pa_max"]).to_numpy() if False else None
buckets = [(0.0, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 0.90),
           (0.90, 0.99), (0.99, 1.01)]
for h in HORIZONS:
    print(f"\n    h = {h}")
    print(f"      {'pa_max band':>14}{'n':>9}{'CRPS soft':>12}{'CRPS hard':>12}"
          f"{'hard-soft':>12}{'rel %':>9}{'t':>8}{'p':>9}")
    for lo, hi in buckets:
        m = (pa >= lo) & (pa < hi)
        ca = d[f"soft_roll_h{h}_crps"].to_numpy()
        cb = d[f"hard_roll_h{h}_crps"].to_numpy()
        ok = m & np.isfinite(ca) & np.isfinite(cb)
        if ok.sum() < 200:
            continue
        diff = cb - ca
        mm, t, p, nd = dm_daily(diff, m, h + 5)
        print(f"      {f'{lo:.2f}-{hi:.2f}':>14}{ok.sum():>9,}"
              f"{np.nanmean(ca[ok]):>12.5f}{np.nanmean(cb[ok]):>12.5f}"
              f"{mm:>12.6f}{100 * mm / np.nanmean(ca[ok]):>9.3f}"
              f"{t:>8.2f}{p:>9.4f}")
        rows.append(dict(test="forced_hard", stratum=f"pa {lo:.2f}-{hi:.2f}",
                         horizon=h, a="soft_roll", b="hard_roll", stat=t,
                         pval=p, effect=mm, n=int(ok.sum())))

# how different are the two weight vectors where it matters?
print("\n    how far apart are the soft and hard weight vectors?")
paq = d.select([c for c in d.columns if c.startswith("pa_")]).columns
print(f"      {'pa_max band':>14}{'n':>9}{'mean L1 dist':>15}"
      f"{'mean pa_max':>13}")
for lo, hi in buckets:
    m = (pa >= lo) & (pa < hi)
    if m.sum() < 200:
        continue
    # L1 between the probability vector and its own one-hot = 2(1 - pa_max)
    print(f"      {f'{lo:.2f}-{hi:.2f}':>14}{m.sum():>9,}"
          f"{2 * (1 - pa[m]).mean():>15.4f}{pa[m].mean():>13.4f}")

pl.DataFrame(rows).write_parquet(OUTPUTS / "phase3" / "c6_results.parquet")
print(f"\nwrote c6_results.parquet | total {time.time() - t0:.1f}s")
