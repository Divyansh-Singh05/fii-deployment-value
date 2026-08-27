"""
MODULE C4 · PAST-ONLY OUTCOME DENSITIES  —  Phase III

For each parameter vintage, each archetype and each horizon, estimates the
distribution of the VOLATILITY-STANDARDISED forward return using only
observations whose outcome was already realised at that vintage's asof date.

    z_{i,t,h} = R_{i, t+1 -> t+h} / (sigma_{i,t} * sqrt(h))

    R      cumulative log return over the next h TRADING days, taken from the
           price panel (a stock has returns on days it has no FII activity,
           and those days are part of the outcome).
    sigma  EWMA(halflife 20d) volatility of daily log returns through day t
           INCLUSIVE. Known at EOD t, so it may standardise an outcome that
           starts at t+1.

THE EMBARGO.  An observation at day s with horizon h is not fully observed
until day s+h. A density used to score day t is therefore built only from
observations with  s + h <= asof(vintage of t)  <  t.  That embargo is what
stops a density from containing outcomes overlapping the day being scored.
It is the most important causality property in C4 (gate I0).

WINDOWS
    expanding  all admissible history (maximises tail sample)
    roll5y     trailing 5 years, matching the C1 parameter window

WEIGHTS
    soft   each past day contributes weight pa_a to archetype a
    hard   each past day contributes 1 to its argmax archetype
Both run through identical machinery, so C6 can isolate whether any gain comes
from soft labelling at scoring time, at estimation time, or both.

RETURNS.  ret_adj (corporate-action adjusted). 63 stock-days in the modelled
universe carry |ret_adj| > 0.5 from a broken adjustment factor (CHEMPLASTS
2021-08-24: close 535.6 vs prev_close 541.0, a -1.0% move, recorded as
+3507%). Data errors, not market moves; they would dominate any tail estimate.
Excluded, with invalidity PROPAGATED to every forward window containing one.

Gates
    I0  embargo     no density contains an outcome realised at or after the
                    asof of the vintage that owns it (brute-force re-checked)
    I1  proper CDF  non-decreasing, endpoints 0 and 1
    I2  sample      effective sample size per cell, hard floor
    I3  vol model   pooled z centred with sd ~ 1

Input : outputs/phase3/archetype_probs.parquet
        outputs/phase3/vintages.parquet
        data/VALIDATION_DATA/returns_panel_v3.parquet
Output: outputs/phase3/outcome_densities.npz
        outputs/phase3/outcome_z.parquet
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from fii.paths import OUTPUTS, VALIDATION_DATA
from fii.lineage import write_lineage  # audit item 12
# ---- truncation hook (used only by module_c9_lookahead_audit) --------------
# P3_TRUNC=<ISO date> makes the module behave as if today were that date:
# every dated input is cut there and every vintage fitted after it is dropped.
# P3_SUFFIX renames the outputs so a truncated run never overwrites the real
# one. With both unset this is a no-op and the module is unchanged.
import os as _os, datetime as _dt
_TRUNC = _os.environ.get("P3_TRUNC")
_TRUNC = _dt.date.fromisoformat(_TRUNC) if _TRUNC else None
_SUF = _os.environ.get("P3_SUFFIX", "")


def _cut(df, col):
    return df.filter(pl.col(col) <= _TRUNC) if _TRUNC is not None else df


ARCH = ["HOSTAGE", "SELL_MID", "SHARK_DIST", "ROBOT",
        "DISPERSED_ACC", "BUY_MID", "SHARK_ACC"]
NA = len(ARCH)
HORIZONS = [1, 5, 20]
PUBLIC_H = (1, 5)       # horizons the engine actually serves; the ESS
                        # frontier gate is binding only for these
HL = 20                 # EWMA halflife for volatility, trading days
MIN_VOL_OBS = 60        # prior observations required before sigma is trusted
HL_SLOW = 250           # slow EWMA halflife (~1y) — the stock's own long-run
                        # volatility level, the anchor the fast estimate is
                        # shrunk toward (see the L-VOL block below)
K_CLAMP = 0.60          # |k| ceiling on the parametric shrinkage exponent
MIN_K_OBS = 20000       # matured observations required before k is estimated
# Non-parametric g(dev). Knots are a FIXED regular grid, declared here and
# never derived from the data, so the per-bin cumulative sums stay causal and
# a truncated world bins identically to the full one. 1.19% / 0.09% of
# stock-days fall in the two catch-all tails.
DEV_EDGES = np.arange(-1.2, 0.91, 0.15)
G_TAU = 2000.0          # pseudo-count shrinking each bin toward the power law
G_CLAMP = 2.0           # |log| ceiling on g, as a ratio
VOL_BURN = HL           # LIMITATION L4 — seed the EWMA variance with an
                        # equal-weighted mean over this many observations
                        # instead of with a single squared return
BAD_RET = 0.5           # |ret_adj| above this is a data error, not a move
GRID = np.linspace(-10.0, 10.0, 1001)
ROLL_D = 5 * 252        # trailing window for the roll5y variant
MIN_ESS = 1000          # hard floor on effective sample size per cell.
                        # 100 was indefensible: at alpha=0.01 it buys ONE
                        # effective tail observation. Measured cost of
                        # raising it to 1000 is 4-6 vintages of extra
                        # warm-up (frontier 2016-03-31 -> 2016-07-29) out
                        # of a 9.5-year sample. The per-LEVEL floor that
                        # actually protects the 1% tail is applied per row
                        # in C5 (MIN_TAIL_EFF), because a row's tail is
                        # supported by its MIXTURE, not by any one cell.

t_start = time.time()

ap = pl.read_parquet(OUTPUTS / "phase3" / f"archetype_probs{_SUF}.parquet")
vt = _cut(pl.read_parquet(OUTPUTS / "phase3" / "vintages.parquet")
          .sort("asof"), "asof")
rp = _cut(pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet"),
          "date")
print(f"archetype rows {ap.height:,} | vintages {vt.height}")

# ------------------------------------------------ calendar & price matrix ---
stocks = np.sort(ap["cisin"].unique().to_numpy())
sidx = {s: i for i, s in enumerate(stocks)}
# Calendar comes from PRICES over the full panel, never from the FII feed —
# see the note in C2. Every calendar day must actually price, or multi-day
# outcome windows spanning it are silently destroyed.
cal = np.sort(
    (rp.group_by("date").agg(pl.col("ret_adj").is_finite().sum().alias("n"))
       .filter(pl.col("n") >= 50)["date"]).to_numpy())
didx = {d: i for i, d in enumerate(cal)}
assert set(ap["TR_DATE"].unique().to_list()) <= set(cal.tolist()), \
    "an archetype row falls on a non-pricing day; rerun C2"
rp = rp.filter(pl.col("isin").is_in(stocks.tolist())
               & pl.col("date").is_in(cal.tolist()))
NS, ND = len(stocks), len(cal)
print(f"price matrix {NS} stocks x {ND} trading days")

r_adj = rp["ret_adj"].to_numpy()
bad = ~np.isfinite(r_adj) | (np.abs(r_adj) > BAD_RET)
print(f"excluded returns |ret_adj|>{BAD_RET} or non-finite: {bad.sum():,} "
      f"of {len(r_adj):,} ({bad.mean():.4%})")

R = np.full((NS, ND), np.nan)
ri = np.array([sidx[s] for s in rp["isin"].to_numpy()])
ci = np.array([didx[d] for d in rp["date"].to_numpy()])
R[ri, ci] = np.where(bad, np.nan, np.log1p(np.clip(r_adj, -0.999, None)))
valid = np.isfinite(R)
print(f"valid stock-days in price matrix: {valid.sum():,} "
      f"({valid.mean():.1%} of the grid)")

# ------------------------------------------------ EWMA volatility -----------
lam = 0.5 ** (1.0 / HL)
lam_s = 0.5 ** (1.0 / HL_SLOW)
SIG = np.full((NS, ND), np.nan)
SIG_SLOW = np.full((NS, ND), np.nan)
v_slow = np.zeros(NS)
v_run = np.zeros(NS)
c_run = np.zeros(NS, dtype=np.int32)
seen = np.zeros(NS, bool)
for j in range(ND):
    r = R[:, j]
    ok = np.isfinite(r)
    r2 = np.where(ok, r * r, 0.0)
    # LIMITATION L4 — the variance state used to be seeded with the stock's
    # FIRST squared return. With HL=20 the weight on that single day is
    # lam^k, so 12.94% of the variance was still one day's r^2 at
    # observation 60, the moment MIN_VOL_OBS releases sigma for use. First
    # days are not ordinary: median r^2 is 1.53x the stock's own median,
    # p99 292x, max 74,388x. That inflated sigma by >10% for 14.4% of
    # stocks at the gate, which suppresses z and UNDERSTATES risk.
    #
    # The state is now seeded with an equal-weighted running mean over the
    # first VOL_BURN observations, then handed to the EWMA. Still strictly
    # causal -- observation j uses r_1..r_j and nothing later -- and the
    # first day's weight at observation 60 falls to (1/20) * lam^40 =
    # 1.25%, a tenfold reduction.
    n_prior = c_run
    warm = (v_run * n_prior + r2) / np.maximum(n_prior + 1, 1)
    ewma = lam * v_run + (1 - lam) * r2
    v_run = np.where(ok, np.where(n_prior < VOL_BURN, warm, ewma), v_run)
    warm_s = (v_slow * n_prior + r2) / np.maximum(n_prior + 1, 1)
    ewma_s = lam_s * v_slow + (1 - lam_s) * r2
    v_slow = np.where(ok, np.where(n_prior < VOL_BURN, warm_s, ewma_s), v_slow)
    seen |= ok
    c_run += ok
    usable = seen & (c_run >= MIN_VOL_OBS)
    SIG[:, j] = np.where(usable, np.sqrt(v_run), np.nan)
    SIG_SLOW[:, j] = np.where(usable, np.sqrt(v_slow), np.nan)
print(f"sigma available on {np.isfinite(SIG).sum():,} stock-days "
      f"(needs {MIN_VOL_OBS} prior observations)")

# ------------------------------------------------ forward returns -----------
Rz = np.where(valid, R, 0.0)
CS = np.concatenate([np.zeros((NS, 1)), np.cumsum(Rz, axis=1)], axis=1)
CV = np.concatenate([np.zeros((NS, 1)),
                     np.cumsum(valid.astype(np.int32), axis=1)], axis=1)
j0 = np.arange(ND)
FWD = {}
for h in HORIZONS:
    lo = np.minimum(j0 + 1, ND)
    hi = np.minimum(j0 + 1 + h, ND)
    tot = CS[:, hi] - CS[:, lo]
    nval = CV[:, hi] - CV[:, lo]
    f = np.where((nval == h) & ((j0 + h) < ND)[None, :], tot, np.nan)
    FWD[h] = f
    print(f"  h={h:>2}: {np.isfinite(f).sum():,} complete forward windows")

# C1 chose asof dates off the FII calendar, so some land on non-trading days
# (2016-10-30 is a Sunday). Map each to the last trading day at or before it.
# This can only tighten the embargo, never loosen it.
_asof_raw = vt["asof"].to_numpy()
asof_pos = np.searchsorted(cal, _asof_raw, side="right") - 1
_nonbiz = int((cal[asof_pos] != _asof_raw).sum())
print(f"\nvintage asof dates snapped back to the last trading day: {_nonbiz} "
      f"of {len(asof_pos)}")
assert (asof_pos >= 0).all()
NV, G = len(asof_pos), len(GRID)

# ------------------------------------------------ per-row outcomes ----------
row_s = np.array([sidx[s] for s in ap["cisin"].to_numpy()])
row_d = np.array([didx[d] for d in ap["TR_DATE"].to_numpy()])
sig_row = SIG[row_s, row_d]
N = ap.height
Z = {}

# ============================================================================
# L-VOL · MEAN-REVERSION SHRINKAGE OF THE VOLATILITY SCALE
#
# THE DEFECT. Breach rates ran ~1.36x nominal in the calmest EWMA-volatility
# quintile and ~0.72x in the wildest, at h=1. Outcomes are already divided by
# that same volatility, so the gradient should not exist. Decomposed:
#
#     quintile basis                  sd(z) Q1 -> Q5     breach x nominal
#     stock TYPE (avg sigma)          1.056 -> 1.085     1.02 -> 0.96   flat
#     WITHIN-STOCK over time          1.218 -> 0.935     1.36 -> 0.72   <-- here
#
# It is not that volatile stocks are mishandled; it is that a stock which is
# calm RELATIVE TO ITS OWN HISTORY has an over-dispersed outcome, and one that
# is agitated has an under-dispersed one. That is volatility mean reversion:
# EWMA(20) estimates volatility NOW, while the outcome depends on volatility
# over (t, t+h]. When today is unusually quiet for this stock, tomorrow
# reverts upward and |z| comes in larger than the pooled density implies.
#
# THE CORRECTION. With dev = log(sigma_fast / sigma_slow), the measured
# scale error is log-linear in dev:
#
#     rms(z) ~ exp(k * dev),   k = -0.168 (h=1), -0.229 (h=5)
#
# so shrinking the deviation by b = 1 + k removes it:
#
#     sigma* = sigma_slow * (sigma_fast / sigma_slow) ** (1 + k)
#            = sigma_fast * exp(k * dev)
#
# A constant multiplicative error would be harmless — the empirical density
# absorbs any fixed scale. Only the dev-DEPENDENT part matters, which is
# exactly what k captures and all that is corrected here.
#
# CAUSALITY. k is estimated per vintage from matured observations only
# (day + h <= asof of that vintage), by OLS of log z^2 on dev over the whole
# price matrix, using cumulative-by-day sufficient statistics so the estimate
# at vintage v sees nothing after its own asof. Before MIN_K_OBS matured
# observations exist, k = 0 and no correction is applied. This is the same
# embargo the densities themselves use.
# ============================================================================
SIG_ADJ, G_BY_VINTAGE = {}, {}
DEV_M = np.log(SIG) - np.log(SIG_SLOW)          # (NS, ND), NaN where unusable
dev_row = DEV_M[row_s, row_d]
K_BY_VINTAGE = {}
print()
for h in HORIZONS:
    # The regression runs over SCORED ROWS ONLY, never over the whole price
    # matrix. The matrix is indexed by `stocks`, which is drawn from the
    # archetype panel — so in a truncated world it silently contains a
    # different stock set, and summing over it leaks the identity of names
    # that only enter the universe later. The look-ahead audit catches this
    # (L3/L4 fail with rows differing in the millions). Scored rows at a given
    # date are identical in both worlds by construction, which is exactly what
    # gate L5 certifies, so accumulating over them is safe.
    # k targets the SECOND MOMENT, not E[log z^2]. Breach rates are driven by
    # E[z^2 | dev]; regressing log z^2 estimates E[log z^2 | dev], and with
    # heavy tails the two slopes differ by about a factor of two — the first
    # attempt under-shrank by exactly that much and left half the gradient
    # standing. Here the slope of log E[z^2 | dev] is taken to first order as
    #     d log E[z^2] / d dev  =  slope(z^2 ~ dev) / mean(z^2)
    # which is exactly estimable from cumulative sufficient statistics.
    # z is clipped to the density grid before squaring, so one -32 sigma day
    # cannot set the correction for the whole panel.
    z_raw_row = FWD[h][row_s, row_d] / (sig_row * np.sqrt(h))
    w_row = np.clip(z_raw_row, GRID[0], GRID[-1]) ** 2
    good = np.isfinite(w_row) & np.isfinite(dev_row)
    d0 = np.where(good, dev_row, 0.0)
    y0 = np.where(good, w_row, 0.0)
    bc = lambda w: np.bincount(row_d, weights=w, minlength=ND).astype(float)
    per_day = [bc(good.astype(float)), bc(d0), bc(y0),
               bc(d0 * d0), bc(d0 * y0)]
    C1, Cx, Cy, Cxx, Cxy = [np.concatenate([[0.0], np.cumsum(a)])
                            for a in per_day]
    kv = np.zeros(NV)
    for vi in range(NV):
        end = asof_pos[vi] - h + 1        # cumulative index: days j <= asof-h
        if end <= 0:
            continue
        n, sx, sy = C1[end], Cx[end], Cy[end]
        sxx, sxy = Cxx[end], Cxy[end]
        den = n * sxx - sx * sx
        if n < MIN_K_OBS or den <= 0:
            continue
        slope = (n * sxy - sx * sy) / den      # d E[z^2] / d dev
        mean_w = sy / n                        # E[z^2]
        if mean_w <= 0:
            continue
        kv[vi] = float(np.clip(0.5 * slope / mean_w, -K_CLAMP, K_CLAMP))
    K_BY_VINTAGE[h] = kv

    # ---- non-parametric g(dev) -------------------------------------------
    # The power law above removes most of the gradient but not all of it: the
    # relationship is not exactly log-linear in dev at the extremes. g is the
    # relative second moment measured per dev-bin,
    #     g(bin)^2 = E[z^2 | bin] / E[z^2]
    # so dividing by it flattens E[z*^2 | dev] by construction rather than by
    # assuming a functional form. Bins are shrunk toward the power-law fit
    # with a pseudo-count, so a sparse bin degrades to exp(k*dev) rather than
    # to noise or to 1 — the fallback matters because the sparse bins are the
    # extremes, which is exactly where the residual lived. Between knots g is
    # linearly interpolated, so the correction is continuous in dev.
    NBIN = len(DEV_EDGES) + 1
    cen = np.empty(NBIN)
    cen[0] = DEV_EDGES[0] - 0.075
    cen[1:-1] = 0.5 * (DEV_EDGES[:-1] + DEV_EDGES[1:])
    cen[-1] = DEV_EDGES[-1] + 0.075
    bin_row = np.searchsorted(DEV_EDGES, np.where(good, dev_row, 0.0))
    flat = row_d * NBIN + bin_row
    cnt_db = np.bincount(flat[good], minlength=ND * NBIN).reshape(ND, NBIN)
    sw_db = np.bincount(flat[good], weights=y0[good],
                        minlength=ND * NBIN).reshape(ND, NBIN)
    Ccnt = np.vstack([np.zeros(NBIN), np.cumsum(cnt_db.astype(float), axis=0)])
    Csw = np.vstack([np.zeros(NBIN), np.cumsum(sw_db, axis=0)])

    GMAT = np.ones((NV, NBIN))
    for vi in range(NV):
        end = asof_pos[vi] - h + 1
        prior_g = np.exp(kv[vi] * cen)
        if end <= 0:
            GMAT[vi] = prior_g
            continue
        nb, sb = Ccnt[end], Csw[end]
        n_tot, s_tot = nb.sum(), sb.sum()
        if n_tot < MIN_K_OBS or s_tot <= 0:
            GMAT[vi] = prior_g
            continue
        mu = s_tot / n_tot
        prior_w = mu * prior_g ** 2
        wbar = (sb + G_TAU * prior_w) / (nb + G_TAU)
        GMAT[vi] = np.clip(np.sqrt(np.maximum(wbar, 1e-12) / mu),
                        1.0 / G_CLAMP, G_CLAMP)
    G_BY_VINTAGE[h] = GMAT

    vid_row = ap["vintage_id"].to_numpy()
    dv_safe = np.where(np.isfinite(dev_row), dev_row, 0.0)
    g_row = np.ones(N)
    for vi in range(NV):                       # 106 vectorised interpolations
        mv = vid_row == vi
        if mv.any():
            g_row[mv] = np.interp(dv_safe[mv], cen, GMAT[vi])
    g_row = np.where(np.isfinite(dev_row), g_row, 1.0)
    sig_adj = sig_row * g_row
    SIG_ADJ[h] = sig_adj
    z = FWD[h][row_s, row_d] / (sig_adj * np.sqrt(h))
    Z[h] = z
    ok = np.isfinite(z)
    live = kv[kv != 0]
    print(f"  h={h:>2}: standardised outcome on {ok.sum():,}/{N:,} rows "
          f"({ok.mean():.1%})  |  k median {np.median(live) if len(live) else 0:+.4f}"
          f"  range [{live.min() if len(live) else 0:+.4f}, "
          f"{live.max() if len(live) else 0:+.4f}]"
          f"  ({len(live)}/{NV} vintages corrected)"
          f"  |  g range [{GMAT.min():.3f}, {GMAT.max():.3f}]")

# ---------------------------- INSTITUTIONAL-SHARE VOLATILITY CORRECTION -----
# THE FINDING THIS IMPLEMENTS.  Within a trading day, across stocks, the
# INSTITUTIONAL SHARE of a stock's turnover predicts its next-day volatility
# negatively, controlling for total volume. Measured on 578k stock-days under
# stock AND date fixed effects: b = -0.030 (t = -5.53 full, -4.73 TRAIN,
# -2.95 TEST), monotone across quintiles, implied sigma falling 1.0923 ->
# 1.0345 from the lowest to the highest share. Robust to the normalisation
# window (20/60/120d), to winsorisation (including none, t = -4.30), and
# larger on |z| than on z^2 (t = -7.37).
#
# That is residual predictability the engine's own sigma does not capture, so
# it belongs in sigma. Both regressors are demeaned WITHIN DAY, matching the
# date-fixed-effect specification the coefficient was estimated under, and
# making the correction purely cross-sectional: it re-allocates volatility
# between stocks on a day, it never changes the market-wide level.
#
#     m^2 = 1 + b*share + c*vol      sigma* = sigma * m
#
# b and c are re-fitted PER VINTAGE on matured observations only
# (day + h <= asof), the same embargo the densities use.
#
# DEFAULT OFF, and that is a RESULT, not a convenience. Applying it removes
# the gradient it targets almost exactly (Q1-Q5 spread in E[z^2] falls from
# +0.1178 to -0.0016, 101% removed) and yet makes the engine WORSE:
# Diebold-Mariano on daily CRPS, corrected minus baseline, h=1 -> +0.001196
# with NW t = +8.94; h=5 -> +0.000174, t = +2.51. The parametric benchmark
# degrades by almost the same amount, which locates the damage in sigma
# rather than in the density.
#
# Why a real effect hurts: the share effect is CROSS-SECTIONAL and small
# (a 5.6% spread in sigma), but exploiting it moves sigma for every stock
# every day off a noisily estimated coefficient -- the multiplier ranged over
# [0.500, 1.508] at h=1 to chase a 5.6% effect. It also mismatches: the
# coefficient is identified under date fixed effects, so it is only valid for
# RELATIVE re-allocation between stocks, while CRPS scores each stock's
# ABSOLUTE forecast. Set P3_SHARE_SIGMA=1 to switch it on and reproduce that
# comparison; module19 quantifies the underlying finding.
SHARE_OFF = not bool(_os.environ.get("P3_SHARE_SIGMA"))
M2_CLIP = (0.25, 4.0)
MIN_SHARE_OBS = 20000

_fi = VALIDATION_DATA / "fii_stockday_intensity.parquet"
SHARE_ROW = VOLI_ROW = None
if not SHARE_OFF and _fi.exists():
    _int = _cut(pl.read_parquet(_fi), "TR_DATE").filter(
        (pl.col("gnorm") > 0) & (pl.col("tnorm") > 0)
        & (pl.col("fii_gross") > 0) & (pl.col("turnover") > 0))
    _int = _int.with_columns(
        ((pl.col("fii_gross") / pl.col("gnorm")).log()
         - (pl.col("turnover") / pl.col("tnorm")).log()).alias("_sh"),
        (pl.col("turnover") / pl.col("tnorm")).log().alias("_vo"))
    _k = ap.select("cisin", "TR_DATE").join(
        _int.select(pl.col("isin").alias("cisin"), "TR_DATE", "_sh", "_vo"),
        on=["cisin", "TR_DATE"], how="left")
    SHARE_ROW = _k["_sh"].to_numpy().astype(float)
    VOLI_ROW = _k["_vo"].to_numpy().astype(float)
    # demean within day -> exactly the date-fixed-effect specification
    _fin = np.isfinite(SHARE_ROW) & np.isfinite(VOLI_ROW)
    for _v in (SHARE_ROW, VOLI_ROW):
        _s = np.bincount(row_d[_fin], weights=_v[_fin], minlength=ND)
        _c = np.bincount(row_d[_fin], minlength=ND)
        _v[_fin] -= (_s / np.maximum(_c, 1))[row_d[_fin]]
        _v[~_fin] = 0.0
    print(f"\n[S] institutional-share sigma correction — "
          f"{int(_fin.sum()):,}/{len(_fin):,} rows carry a share "
          f"({100*_fin.mean():.1f}%)")
else:
    print("\n[S] institutional-share sigma correction OFF"
          + (" (default; set P3_SHARE_SIGMA=1 to enable — it is off because it"
             " DEGRADES CRPS, see the header)" if SHARE_OFF
             else " (intensity file absent)"))

SHARE_COEF = {}
if SHARE_ROW is not None:
    for h in HORIZONS:
        zr = Z[h]                       # z under the pre-correction sigma
        w = np.clip(zr, GRID[0], GRID[-1]) ** 2
        good = np.isfinite(w) & _fin
        bc = lambda v: np.bincount(row_d, weights=v, minlength=ND).astype(float)
        # cumulative-by-day sufficient statistics for a causal per-vintage OLS
        S1 = bc(good.astype(float))
        Sx = bc(np.where(good, SHARE_ROW, 0.0))
        Sy = bc(np.where(good, VOLI_ROW, 0.0))
        Sw = bc(np.where(good, w, 0.0))
        Sxx = bc(np.where(good, SHARE_ROW ** 2, 0.0))
        Syy = bc(np.where(good, VOLI_ROW ** 2, 0.0))
        Sxy = bc(np.where(good, SHARE_ROW * VOLI_ROW, 0.0))
        Sxw = bc(np.where(good, SHARE_ROW * w, 0.0))
        Syw = bc(np.where(good, VOLI_ROW * w, 0.0))
        C = [np.concatenate([[0.0], np.cumsum(a)])
             for a in (S1, Sx, Sy, Sw, Sxx, Syy, Sxy, Sxw, Syw)]
        coef = np.zeros((NV, 2))
        for vi in range(NV):
            e = asof_pos[vi] - h + 1
            if e <= 0:
                continue
            n, sx, sy, sw, sxx, syy, sxy, sxw, syw = (c[e] for c in C)
            if n < MIN_SHARE_OBS:
                continue
            A = np.array([[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]])
            rhs = np.array([sw, sxw, syw])
            try:
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            if sol[0] <= 1e-9:
                continue
            coef[vi] = sol[1:] / sol[0]        # scale to a multiplier on m^2
        SHARE_COEF[h] = coef
        m2 = 1.0 + coef[ap["vintage_id"].to_numpy()][:, 0] * SHARE_ROW \
                 + coef[ap["vintage_id"].to_numpy()][:, 1] * VOLI_ROW
        m2 = np.clip(np.where(_fin, m2, 1.0), *M2_CLIP)
        mult = np.sqrt(m2)
        SIG_ADJ[h] = SIG_ADJ[h] * mult
        Z[h] = FWD[h][row_s, row_d] / (SIG_ADJ[h] * np.sqrt(h))
        live = coef[np.any(coef != 0, axis=1)]
        print(f"   h={h:>2}: b(share) median {np.median(live[:, 0]):+.4f}  "
              f"c(vol) median {np.median(live[:, 1]):+.4f}  "
              f"({len(live)}/{NV} vintages)  "
              f"m in [{mult.min():.3f}, {mult.max():.3f}]")

# ------------------------------------------------ Student-t benchmark -------
# A FAIR parametric benchmark. `normal` is a straw man: standardised equity
# returns are fat-tailed by construction, so a Gaussian loses on the tail
# before any flow information is involved. The honest question is whether an
# EMPIRICAL density beats a parametric law that already knows the tail is
# fat. Same EWMA volatility, same rows, same dates — only the assumed shape
# differs, exactly as for `normal`.
#
# nu is estimated by maximum likelihood on the standardised outcomes that had
# already MATURED at each vintage's asof (day + h <= asof), which is the same
# embargo the empirical densities use. z carries unit variance by
# construction, so the t is rescaled by sqrt(nu/(nu-2)) to match.
from scipy.optimize import minimize_scalar
from scipy.stats import t as _tdist

NU_BY_VINTAGE = {}
print("\n[T] Student-t benchmark — nu fitted per vintage, past-only")
for h in HORIZONS:
    z_all = Z[h]
    ok_all = np.isfinite(z_all)
    order = np.argsort(row_d, kind="stable")
    zo_s = z_all[order]
    rd_o = row_d[order]
    ok_o = ok_all[order]
    nu_v = np.full(NV, np.nan)
    for vi in range(NV):
        cut = asof_pos[vi] - h
        j = int(np.searchsorted(rd_o, cut, side="right"))
        zz = zo_s[:j][ok_o[:j]]
        if len(zz) < 5000:
            continue
        if len(zz) > 200_000:                       # cap cost; deterministic
            zz = zz[:: max(1, len(zz) // 200_000)]

        def _nll(nu):
            if nu <= 2.05:
                return 1e18
            sc = np.sqrt(nu / (nu - 2.0))
            return -float(np.sum(_tdist.logpdf(zz * sc, nu) + np.log(sc)))

        r = minimize_scalar(_nll, bounds=(2.1, 60.0), method="bounded",
                            options={"xatol": 1e-3})
        nu_v[vi] = float(r.x) if r.success else np.nan
    NU_BY_VINTAGE[h] = nu_v
    fin = nu_v[np.isfinite(nu_v)]
    print(f"   h={h:>2}: nu median {np.median(fin):.2f} "
          f"range [{fin.min():.2f}, {fin.max():.2f}] "
          f"({len(fin)}/{NV} vintages fitted)")

# ------------------------------------------------ EVT tail benchmark --------
# PEAKS-OVER-THRESHOLD. The empirical density reads its 1% quantile straight
# off a histogram, which is why C5 has to WITHHOLD a 1% number on ~33% of h=5
# rows: below the threshold there simply are not enough past observations to
# read. Extreme-value theory says the exceedances over a high threshold
# converge to a Generalised Pareto regardless of the parent distribution, so
# the tail can be EXTRAPOLATED rather than counted.
#
# Construction, per vintage, past-only (day + h <= asof, the same embargo the
# densities use): pool the matured standardised outcomes, take the 5th
# percentile as the threshold u, fit a GPD to the exceedances below it, and
# splice
#     F(g) = p_u * (1 - G_gpd(u - g))     for g <  u
#     F(g) = empirical CDF(g)             for g >= u
#
# The body is the POOLED empirical distribution, i.e. the same information
# `clim` uses. So `evt` vs `clim` isolates one thing: does extrapolating the
# tail beat counting it? And xi, the fitted shape parameter, is reportable —
# a histogram has no such summary.
from scipy.stats import genpareto

EVT_Q = 0.05            # threshold: the 5th percentile of past outcomes
EVTCDF, XI_BY_VINTAGE = {}, {}
print("\n[E] EVT tail benchmark — GPD fitted per vintage, past-only")
for h in HORIZONS:
    z_all = Z[h]
    ok_all = np.isfinite(z_all)
    order = np.argsort(row_d, kind="stable")
    zo_s, rd_o, ok_o = z_all[order], row_d[order], ok_all[order]
    cdf_v = np.full((NV, G), np.nan, dtype=np.float32)
    xi_v = np.full(NV, np.nan)
    for vi in range(NV):
        cut = asof_pos[vi] - h
        j = int(np.searchsorted(rd_o, cut, side="right"))
        zz = np.sort(zo_s[:j][ok_o[:j]])
        if len(zz) < 5000:
            continue
        u = float(np.quantile(zz, EVT_Q))
        exc = u - zz[zz < u]
        exc = exc[exc > 0]
        if len(exc) < 200:
            continue
        try:
            xi, loc, sc = genpareto.fit(exc, floc=0.0)
        except Exception:
            continue
        if not np.isfinite(xi) or sc <= 0:
            continue
        xi_v[vi] = float(xi)
        F = np.searchsorted(zz, GRID, side="right") / len(zz)   # empirical body
        below = GRID < u
        F[below] = EVT_Q * (1.0 - genpareto.cdf(u - GRID[below], xi, 0.0, sc))
        F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
        cdf_v[vi] = F.astype(np.float32)
    EVTCDF[h], XI_BY_VINTAGE[h] = cdf_v, xi_v
    fx = xi_v[np.isfinite(xi_v)]
    print(f"   h={h:>2}: xi median {np.median(fx):+.4f} "
          f"range [{fx.min():+.4f}, {fx.max():+.4f}] "
          f"({len(fx)}/{NV} vintages fitted)"
          f"   {'heavy-tailed (xi>0)' if np.median(fx) > 0 else 'bounded tail'}")

# ---- I3 · volatility model sanity ------------------------------------------
# The gate is on DISPERSION only. An EWMA volatility model standardises the
# width of the outcome distribution; it makes no claim about its location.
# A non-zero mean z is market drift, which is an economic fact about the
# window rather than a fault in the standardisation — over 2016-02 to
# 2017-06, for instance, the Nifty rose 27.7% and mean z at h=20 is +0.22 in
# BOTH a full-sample run and a run truncated at that date. Gating on the mean
# would fail any genuinely trending sub-period. The mean is reported for
# information and never asserted.
print("\n[I3] vol-model sanity — gate on dispersion (sd ~ 1); mean is drift, "
      "reported only")
i3 = True
for h in HORIZONS:
    z = Z[h][np.isfinite(Z[h])]
    m, s = float(np.mean(z)), float(np.std(z))
    ok = 0.7 < s < 1.4
    i3 &= ok
    print(f"     h={h:>2}: sd {s:.4f} {'ok' if ok else 'OUT OF RANGE'}  |  "
          f"mean {m:+.4f} (drift, informational)  "
          f"kurtosis {float(np.mean(((z - m) / s) ** 4)):6.1f}")
print(f"     {'PASS' if i3 else 'FAIL'}")
assert i3, "volatility standardisation is not behaving"

# ------------------------------------------------ density estimation --------
W = {"soft": ap.select([f"pa_{a}" for a in ARCH]).to_numpy(),
     "hard": np.eye(NA)[np.array([ARCH.index(a)
                                  for a in ap["arch_hard"].to_numpy()])]}

# avail_pos = row_d + h, a constant offset, so sorting by row_d also sorts by
# availability. Both the "has become available" and the "has fallen out of the
# 5y window" conditions are then prefix conditions on one order, which makes
# the whole sweep O(N) per (horizon, weight) instead of O(N x vintages).
order = np.argsort(row_d, kind="stable")
rd_s = row_d[order]

DENS, ESS, MASS = {}, {}, {}
CWT, CSQ, CNT = {}, {}, {}
print("\nbuilding densities ...")
for h in HORIZONS:
    fin_s = np.isfinite(Z[h])[order]
    bin_s = np.clip(np.searchsorted(GRID, np.nan_to_num(Z[h], nan=0.0),
                                    side="left"), 0, G - 1)[order]
    for wname, Wm in W.items():
        Wo = Wm[order]
        Hadd = np.zeros((NA, G))
        Hrem = np.zeros((NA, G))
        sw_a = np.zeros(NA); sq_a = np.zeros(NA)
        sw_r = np.zeros(NA); sq_r = np.zeros(NA)
        pa = pr = 0
        cdf = {"expanding": np.zeros((NV, NA, G), dtype=np.float32),
               "roll5y": np.zeros((NV, NA, G), dtype=np.float32)}
        ess = {"expanding": np.zeros((NV, NA)),
               "roll5y": np.zeros((NV, NA))}
        mass = {"expanding": np.zeros((NV, NA)),
                "roll5y": np.zeros((NV, NA))}
        # AUDIT ITEM 6 — per-bin material for DIRECT tail support, so C5 can
        # ask "how many observations actually sit below this row's VaR"
        # instead of inferring it from ess * alpha. Cumulative weight is the
        # unnormalised CDF; cumulative squared weight gives a Kish ESS of
        # exactly the tail.
        #
        # The raw count used to be stored ONCE per vintage, on the argument
        # that under soft weighting the same past day feeds all seven
        # components. True for soft, false for hard: a hard density is built
        # from one archetype's observations, so certifying its tail with a
        # count pooled over all seven let a thin hard-archetype tail pass the
        # gate on the strength of the other six. The count is now per
        # archetype, incremented wherever that archetype carries any weight —
        # which reproduces the pooled total under soft weighting and gives the
        # single owning archetype's count under hard.
        NB = len(GRID)      # histograms span grid POSITIONS, not intervals
        cwt = {t: np.zeros((NV, NA, NB), dtype=np.float32)
               for t in ("expanding", "roll5y")}
        csq = {t: np.zeros((NV, NA, NB), dtype=np.float32)
               for t in ("expanding", "roll5y")}
        cnt = {t: np.zeros((NV, NA, NB), dtype=np.int32)
               for t in ("expanding", "roll5y")}
        Qadd = np.zeros((NA, NB)); Qrem = np.zeros((NA, NB))
        Cadd = np.zeros((NA, NB), dtype=np.int64)
        Crem = np.zeros((NA, NB), dtype=np.int64)
        max_avail = np.full(NV, -1)
        for vi in range(NV):
            cut_add = asof_pos[vi] - h          # s + h <= asof
            while pa < N and rd_s[pa] <= cut_add:
                if fin_s[pa]:
                    w = Wo[pa]
                    Hadd[:, bin_s[pa]] += w
                    Qadd[:, bin_s[pa]] += w * w
                    Cadd[:, bin_s[pa]] += (w > 0)
                    sw_a += w; sq_a += w * w
                    max_avail[vi] = max(max_avail[vi], rd_s[pa] + h)
                pa += 1
            if vi:
                max_avail[vi] = max(max_avail[vi], max_avail[vi - 1])
            cut_rem = asof_pos[vi] - ROLL_D
            while pr < N and rd_s[pr] < cut_rem:
                if fin_s[pr]:
                    w = Wo[pr]
                    Hrem[:, bin_s[pr]] += w
                    Qrem[:, bin_s[pr]] += w * w
                    Crem[:, bin_s[pr]] += (w > 0)
                    sw_r += w; sq_r += w * w
                pr += 1
            for tag, Hc, sw, sq, Qc, Cc in (
                    ("expanding", Hadd, sw_a, sq_a, Qadd, Cadd),
                    ("roll5y", Hadd - Hrem, sw_a - sw_r, sq_a - sq_r,
                     Qadd - Qrem, Cadd - Crem)):
                cwt[tag][vi] = np.cumsum(Hc, axis=1).astype(np.float32)
                csq[tag][vi] = np.cumsum(Qc, axis=1).astype(np.float32)
                cnt[tag][vi] = np.cumsum(Cc, axis=1).astype(np.int32)
                tot = Hc.sum(axis=1, keepdims=True)
                cdf[tag][vi] = (np.cumsum(Hc, axis=1)
                                / np.where(tot > 0, tot, 1.0)).astype(np.float32)
                # LIMITATION L3 — h-day forward windows are sampled EVERY
                # day, so consecutive outcomes overlap by h-1 days and the
                # Kish ESS counts them as if independent. It therefore
                # overstates the information behind a cell by ~h. The
                # marginal density is unaffected (measured: the h=5 tail
                # quantiles are within 0.012 of a non-overlapping estimate,
                # against a 0.22 spread across disjoint offsets) -- what is
                # wrong is the SAMPLE SIZE, and every floor built on it.
                # Deflating here makes MIN_ESS in this module and
                # MIN_TAIL_EFF in s09 both count independent observations,
                # so a 5-day number now has to clear the same evidential bar
                # a 1-day number does. h=1 is unchanged by construction.
                ess[tag][vi] = np.where(
                    sq > 0, sw * sw / np.where(sq > 0, sq, 1), 0.0) / h
                mass[tag][vi] = sw
        for tag in ("expanding", "roll5y"):
            DENS[(h, wname, tag)] = cdf[tag]
            ESS[(h, wname, tag)] = ess[tag]
            MASS[(h, wname, tag)] = mass[tag]
            CWT[(h, wname, tag)] = cwt[tag]
            CSQ[(h, wname, tag)] = csq[tag]
            CNT[(h, wname, tag)] = cnt[tag]
        print(f"  h={h:>2} {wname:4s} done  (rows consumed {pa:,})")
    globals()[f"_maxavail_{h}"] = max_avail

# ---- I0 · embargo ----------------------------------------------------------
# Brute-force, independent of the pointer logic: for every vintage and horizon,
# recompute the admissible set from scratch and check the latest outcome date
# it contains is strictly before the earliest date that vintage ever scores.
print("\n[I0] embargo — brute-force re-check of every vintage x horizon")
v_row = ap["vintage_id"].to_numpy()
first_scored = np.full(NV, ND + 1)
for vi in range(NV):
    m = v_row == vi
    if m.any():
        first_scored[vi] = row_d[m].min()
worst = -10**9
bad_cells = 0
for h in HORIZONS:
    for vi in range(NV):
        adm = np.isfinite(Z[h]) & ((row_d + h) <= asof_pos[vi])
        if not adm.any():
            continue
        latest_outcome = int((row_d[adm] + h).max())
        slack = int(first_scored[vi]) - latest_outcome
        worst = max(worst, -slack)
        bad_cells += latest_outcome >= first_scored[vi]
print(f"     cells where an outcome reaches the scored day: {bad_cells}  "
      f"{'PASS' if bad_cells == 0 else 'FAIL'}")
print(f"     tightest margin between latest outcome and first scored day: "
      f"{-worst} trading days")
assert bad_cells == 0, "embargo violated: a density sees its own scoring day"

# ---- I1 · proper CDF -------------------------------------------------------
print("\n[I1] proper CDF")
i1 = True
for k, c in DENS.items():
    live = ESS[k] >= 1.0
    if not live.any():
        continue
    sub = c[live]
    mono = float(np.diff(sub, axis=-1).min())
    e0 = float(sub[:, 0].max())
    e1 = float(np.abs(sub[:, -1] - 1.0).max())
    ok = mono >= -1e-6 and e1 < 1e-5
    i1 &= ok
    if not ok:
        print(f"     {k} mono {mono:.2e} start {e0:.2e} end-err {e1:.2e} FAIL")
print(f"     all {len(DENS)} density blocks non-decreasing with endpoints "
      f"0/1: {'PASS' if i1 else 'FAIL'}")
assert i1

# ---- I2 · effective sample size, with an explicit warm-up frontier ---------
# A density built from almost no embargoed history is not wrong, it is
# untrustworthy. Rather than lower the floor, each block declares the first
# vintage from which EVERY archetype clears MIN_ESS and never falls back
# below it. Earlier vintages are marked unusable and must not be scored.
# The cost is reported in months, per horizon, so it is visible rather than
# absorbed.
print("\n[I2] effective sample size and warm-up frontier")
print(f"{'horizon':>8}{'weights':>9}{'window':>11}{'valid_from':>12}"
      f"{'asof':>13}{'min ESS after':>15}{'median':>10}")
VALID_FROM = {}
i2 = True
for h in HORIZONS:
    for wname in ("soft", "hard"):
        for tag in ("expanding", "roll5y"):
            e = ESS[(h, wname, tag)]
            ok = (e >= MIN_ESS).all(axis=1)
            # first index from which ok holds for the entire remaining tail
            tail_ok = np.flip(np.cumprod(np.flip(ok.astype(int))))
            idx = int(np.argmax(tail_ok == 1)) if tail_ok.any() else NV
            VALID_FROM[(h, wname, tag)] = idx
            good = e[idx:]
            # LIMITATION L3 — the floor now counts INDEPENDENT observations,
            # so a long horizon can legitimately never reach an adequate
            # sample. That is a real finding about h=20, not a broken stage,
            # and it must not be fatal for a horizon the engine does not
            # serve. The gate stays binding for the PUBLIC horizons.
            if h in PUBLIC_H:
                i2 &= idx < NV
            if idx >= NV:
                print(f"{h:>8}{wname:>9}{tag:>11}{'never':>12}"
                      f"{'—':>13}{'—':>15}{'—':>10}")
            else:
                print(f"{h:>8}{wname:>9}{tag:>11}{idx:>12}"
                      f"{str(vt['asof'][idx]):>13}{good.min():>15.0f}"
                      f"{np.median(good):>10.0f}")
_never = [k for k, v in VALID_FROM.items() if v >= NV]
if _never:
    print(f"     NOTE — {len(_never)} block(s) never clear the floor: "
          f"{sorted({k[0] for k in _never})} (horizons). Under the L3 horizon "
          f"scaling this is expected for h=20 and is why it is not served.")
print(f"     {'PASS' if i2 else 'FAIL'} — every PUBLIC-horizon block "
      f"{PUBLIC_H} reaches a stable frontier (floor {MIN_ESS})")
assert i2, ("a public-horizon density block never reaches an adequate "
            "sample: " + str([k for k in _never if k[0] in PUBLIC_H]))
for h in HORIZONS:
    idx = max(VALID_FROM[(h, w, t)] for w in ("soft", "hard")
              for t in ("expanding", "roll5y"))
    if idx >= NV:
        print(f"     h={h:>2}: NEVER reaches an adequate independent sample "
              f"in this window — not servable")
    else:
        print(f"     h={h:>2}: scoring may begin at vintage {idx} "
              f"({vt['asof'][idx]}) — warm-up cost {idx} vintages")

# ------------------------------------------------ save ----------------------
out_npz = OUTPUTS / "phase3" / f"outcome_densities{_SUF}.npz"
np.savez_compressed(
    out_npz, grid=GRID, arch=np.array(ARCH), horizons=np.array(HORIZONS),
    asof_pos=asof_pos,
    **{f"cdf__{h}__{w}__{t}": DENS[(h, w, t)] for (h, w, t) in DENS},
    **{f"ess__{h}__{w}__{t}": ESS[(h, w, t)] for (h, w, t) in ESS},
    **{f"cwt__{h}__{w}__{t}": CWT[(h, w, t)] for (h, w, t) in CWT},
    **{f"csq__{h}__{w}__{t}": CSQ[(h, w, t)] for (h, w, t) in CSQ},
    **{f"cnt__{h}__{w}__{t}": CNT[(h, w, t)] for (h, w, t) in CNT},
    **{f"validfrom__{h}__{w}__{t}": np.array(VALID_FROM[(h, w, t)])
       for (h, w, t) in VALID_FROM},
    **{f"mass__{h}__{w}__{t}": MASS[(h, w, t)] for (h, w, t) in MASS},
    **{f"nu__{h}": NU_BY_VINTAGE[h] for h in HORIZONS},
    **{f"evtcdf__{h}": EVTCDF[h] for h in HORIZONS},
    **{f"xi__{h}": XI_BY_VINTAGE[h] for h in HORIZONS},
    **{f"sharecoef__{h}": SHARE_COEF[h] for h in HORIZONS}
    if SHARE_COEF else {})

zo = ap.select(["cisin", "TR_DATE", "era", "vintage_id", "pa_max", "p_max",
                "arch_soft", "arch_hard", "stock_out_of_window"]).with_columns(
    [pl.Series(f"z_h{h}", Z[h]) for h in HORIZONS]
    # `sigma_ewma` stays the RAW fast EWMA — it is the volatility STATE, and
    # s11's conditional-calibration strata are cut on it. The mean-reversion
    # shrinkage is horizon-specific, so the scale that actually standardised
    # each outcome is carried separately and is what converts a z-space VaR
    # back into return space downstream.
    + [pl.Series(f"sigma_adj_h{h}", SIG_ADJ[h]) for h in HORIZONS]
    + [pl.Series("sigma_ewma", sig_row)])
zo.write_parquet(OUTPUTS / "phase3" / f"outcome_z{_SUF}.parquet")
for _art in (OUTPUTS / "phase3" / f"outcome_densities{_SUF}.npz",
             OUTPUTS / "phase3" / f"outcome_z{_SUF}.parquet"):
    write_lineage(_art,
                  inputs=[OUTPUTS / "phase3" / f"archetype_probs{_SUF}.parquet",
                          VALIDATION_DATA / "returns_panel_v3.parquet"],
                  extras={"module": "C4", "suffix": _SUF,
                          "min_ess": MIN_ESS, "grid": len(GRID)})

# ------------------------------------------------ report --------------------
print("\n" + "=" * 72)
print("C4 SUMMARY")
print(f"  densities  {len(DENS)} blocks "
      f"({len(HORIZONS)} horizons x 2 weightings x 2 windows)")
print(f"  each       {NV} vintages x {NA} archetypes x {G} grid points")
print(f"  saved      {out_npz.name} "
      f"({out_npz.stat().st_size / 1e6:.1f} MB)")

print("\n  standardised-outcome coverage and dispersion by archetype (h=5, "
      "pooled, descriptive only):")
z5 = Z[5]
hard_lab = ap["arch_hard"].to_numpy()
print(f"    {'archetype':16s}{'n':>9}{'mean':>9}{'sd':>8}{'p5':>8}{'p95':>8}")
for a in ARCH:
    m = (hard_lab == a) & np.isfinite(z5)
    if m.sum():
        zz = z5[m]
        print(f"    {a:16s}{m.sum():>9,}{zz.mean():>9.3f}{zz.std():>8.3f}"
              f"{np.percentile(zz, 5):>8.2f}{np.percentile(zz, 95):>8.2f}")
print(f"\n  total {time.time() - t_start:.1f}s")
print("=" * 72)
