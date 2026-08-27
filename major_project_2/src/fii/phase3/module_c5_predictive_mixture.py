"""
MODULE C5 · PREDICTIVE MIXTURE AND RISK ENGINE  —  Phase III

Combines each stock-day's archetype probabilities (C3) with the corresponding
vintage's past-only outcome densities (C4) into a predictive distribution for
the standardised forward return, then reads risk quantities off it.

    F_{i,t,h}(z) = sum_a  w_{a,i,t} * F_{a,h}(z | vintage v(t))

    w   the day's archetype weights — SOFT (the C3 probabilities) or HARD
        (one-hot on the argmax). This is the ONLY difference between the two
        systems being compared; both draw on densities built the same way,
        so any gap is attributable to labelling at scoring time.

Everything is computed in standardised units and converted back to return
space with  R = z * sigma_{i,t} * sqrt(h),  where sigma is the past-only EWMA
volatility from C4. A VaR expressed in return space is therefore a genuine
one-day-ahead statement made with information available at EOD t.

SYSTEMS SCORED
    soft_roll   w = C3 probabilities,  densities soft/roll5y   (primary)
    hard_roll   w = one-hot argmax,    densities hard/roll5y   (the baseline
                                                    the thesis must beat)
    soft_exp    w = C3 probabilities,  densities soft/expanding (robustness
                                                    to the density window)
    clim_roll   w = the vintage's own archetype mass shares — identical
                machinery with the CONDITIONING REMOVED. Isolates how much of
                any skill comes from knowing the archetype at all, versus
                from the shape of the pooled outcome distribution.
    normal      standard normal in standardised space. Because outcomes are
                already divided by EWMA volatility, N(0,1) IS the "EWMA
                volatility model" benchmark, stated exactly.

Rows before a horizon's C4 warm-up frontier are not scored (NaN), never
scored with a thin density.

Gates
    J0  frontier    no row scored by a vintage below its horizon's valid_from
    J1  proper CDF  every mixture non-decreasing with endpoints 0 and 1
    J2  degenerate  a one-hot weight reproduces that archetype's CDF exactly
    J3  PIT range   PIT within [0,1]; grid-truncation rate reported
    J4  coherence   VaR(1%) <= VaR(5%) and ES(a) <= VaR(a)
    J5  CRPS check  the `normal` system's numerically integrated CRPS matches
                    the closed form  z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)
                    — an independent validation of the grid integration that
                    every other system depends on

Input : outputs/phase3/archetype_probs.parquet
        outputs/phase3/outcome_z.parquet
        outputs/phase3/outcome_densities.npz
Output: outputs/phase3/predictive.parquet
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.stats import norm

from fii.paths import OUTPUTS
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
ALPHAS = [0.01, 0.05]
MIN_TAIL_RAW = 200      # AUDIT ITEM 6 — raw observations that must actually
                        # lie below the row's own VaR before it is reported.
                        # Measured directly, not inferred. On the pre-audit
                        # build the median cell carried 2,426 raw observations
                        # below its 1% quantile (p10 ESS 48), so this floor
                        # binds only where support is genuinely absent.
MIN_TAIL_EFF = 50       # effective observations required IN THE TAIL before a
                        # VaR/ES at level alpha is reported for a row.
                        # A row's tail is supported by its MIXTURE, not by any
                        # single archetype cell, and the variance of a mixture
                        # CDF at a point goes like sum_a w_a^2 / ESS_a. So the
                        # governing quantity is the mixture ESS
                        #     ess_mix = 1 / sum_a (w_a^2 / ESS_a)
                        # and the requirement is ess_mix * alpha >= MIN_TAIL_EFF
                        # (>=5,000 at 1%, >=1,000 at 5%). AUDIT ITEM 6: this
                        # is now a cheap PRE-SCREEN only. It is a proxy -- it
                        # assumes effective observations land in the tail in
                        # proportion to alpha -- and the binding gate is the
                        # DIRECT measurement below the realised VaR. The audit
                        # found the proxy accurate in aggregate (implied 223.6
                        # vs measured 227.4 median tail ESS), which is why it
                        # is kept as a screen rather than deleted, but an
                        # aggregate agreement is not a per-row guarantee.
                        # Rows that fail get NaN
                        # rather than a number nobody should trust. CRPS and PIT
                        # are NOT suppressed: they are whole-density scores, and
                        # suppressing them for the empirical systems but not for
                        # the parametric benchmark would bias every DM test.
FULL = ("soft_roll", "hard_roll")        # systems getting full risk metrics
LIGHT = ("soft_exp", "clim_roll", "normal", "t_ewma", "evt")  # benchmarks
# evt — pooled empirical body with a GPD tail spliced below the 5th
# percentile, fitted per vintage on matured past outcomes in C4. Against
# `clim_roll`, which uses the same body and reads its tail off the histogram,
# this isolates whether EXTRAPOLATING the tail beats counting it.
# t_ewma — the FAIR parametric benchmark. Same EWMA volatility and the same
# rows as every other system; only the assumed shape differs. nu is fitted per
# vintage on matured past outcomes in C4, so a Gaussian is not the only
# parametric law the empirical density has to beat.

t_start = time.time()

ap = pl.read_parquet(OUTPUTS / "phase3" / f"archetype_probs{_SUF}.parquet")
zo = pl.read_parquet(OUTPUTS / "phase3" / f"outcome_z{_SUF}.parquet")
D = np.load(OUTPUTS / "phase3" / f"outcome_densities{_SUF}.npz",
            allow_pickle=True)
GRID = D["grid"]
G = len(GRID)
DZ = float(GRID[1] - GRID[0])
MID = 0.5 * (GRID[1:] + GRID[:-1])
N = ap.height
print(f"rows {N:,} | grid {G} points, step {DZ:g} | horizons {HORIZONS}")
assert zo.height == N

v_row = ap["vintage_id"].to_numpy()
NV = int(v_row.max()) + 1
Wsoft = ap.select([f"pa_{a}" for a in ARCH]).to_numpy()
Whard = np.eye(NA)[np.array([ARCH.index(a) for a in ap["arch_hard"].to_numpy()])]
sigma = zo["sigma_ewma"].to_numpy()
Zr = {h: zo[f"z_h{h}"].to_numpy() for h in HORIZONS}

# climatology weights: the vintage's own archetype mass, conditioning removed
MASS = {h: D[f"mass__{h}__soft__roll5y"] for h in HORIZONS}
CLIMW = {h: MASS[h] / np.where(MASS[h].sum(axis=1, keepdims=True) > 0,
                               MASS[h].sum(axis=1, keepdims=True), 1.0)
         for h in HORIZONS}

NORMCDF = norm.cdf(GRID).astype(np.float32)
from scipy.stats import t as _tdist
NU = {h: D[f"nu__{h}"] for h in HORIZONS}
EVTCDF = {h: D[f"evtcdf__{h}"] for h in HORIZONS}

SPEC = {
    "soft_roll": ("soft", "roll5y", "wsoft"),
    "hard_roll": ("hard", "roll5y", "whard"),
    "soft_exp": ("soft", "expanding", "wsoft"),
    "clim_roll": ("soft", "roll5y", "wclim"),
}

# ---- J0 · warm-up frontier --------------------------------------------------
VF = {}
n_thin = {}   # (system, horizon, alpha) -> rows suppressed by J6
n_direct = {} # rows the DIRECT tail measure caught that the proxy missed
for h in HORIZONS:
    VF[h] = max(int(D[f"validfrom__{h}__{w}__{t}"])
                for w in ("soft", "hard") for t in ("expanding", "roll5y"))
print("\n[J0] warm-up frontier — rows below it are not scored")
for h in HORIZONS:
    n_ex = int((v_row < VF[h]).sum())
    print(f"     h={h:>2}: valid from vintage {VF[h]}, excluded {n_ex:,} rows "
          f"({n_ex / N:.3%})")

out = {}
j1 = j2 = j4 = True
trunc = {}

for h in HORIZONS:
    zz = Zr[h]
    have_z = np.isfinite(zz)
    trunc[h] = int((np.abs(zz[have_z]) > GRID[-1]).sum())
    zc = np.clip(np.nan_to_num(zz, nan=0.0), GRID[0], GRID[-1])
    zi = np.clip(np.searchsorted(GRID, zc, side="left"), 0, G - 1)

    for sysname in list(SPEC) + ["normal", "t_ewma", "evt"]:
        crps = np.full(N, np.nan, dtype=np.float32)
        pit = np.full(N, np.nan, dtype=np.float32)
        full = sysname in FULL
        if full:
            mu = np.full(N, np.nan, dtype=np.float32)
            sd = np.full(N, np.nan, dtype=np.float32)
            var = {a: np.full(N, np.nan, dtype=np.float32) for a in ALPHAS}
            es = {a: np.full(N, np.nan, dtype=np.float32) for a in ALPHAS}

        for vi in range(VF[h], NV):
            m = np.flatnonzero(v_row == vi)
            if not len(m):
                continue
            if sysname == "normal":
                M = np.broadcast_to(NORMCDF, (len(m), G))
            elif sysname == "evt":
                Ce = EVTCDF[h][vi]
                if not np.isfinite(Ce).all():
                    continue
                M = np.broadcast_to(Ce, (len(m), G))
            elif sysname == "t_ewma":
                nu = float(NU[h][vi])
                if not np.isfinite(nu):
                    continue
                # z carries unit variance, so the t is rescaled to match
                sc = np.sqrt(nu / (nu - 2.0))
                M = np.broadcast_to(
                    _tdist.cdf(GRID * sc, nu).astype(np.float32), (len(m), G))
            else:
                wkind, win, wsrc = SPEC[sysname]
                C = D[f"cdf__{h}__{wkind}__{win}"][vi]          # (NA, G)
                if wsrc == "wsoft":
                    Wv = Wsoft[m]
                elif wsrc == "whard":
                    Wv = Whard[m]
                else:
                    Wv = np.broadcast_to(CLIMW[h][vi], (len(m), NA))
                M = (Wv.astype(np.float32) @ C)
                E = D[f"ess__{h}__{wkind}__{win}"][vi]          # (NA,)
                ess_mix = 1.0 / np.maximum(
                    (Wv ** 2 / np.maximum(E, 1e-12)[None, :]).sum(axis=1),
                    1e-30)

            # J1 · proper CDF
            if np.diff(M, axis=1).min() < -1e-5 or \
               abs(float(M[:, -1].max()) - 1.0) > 1e-4:
                j1 = False

            pdf = np.diff(M, axis=1)
            pdf = np.clip(pdf, 0.0, None)
            tot = pdf.sum(axis=1, keepdims=True)
            pdf = pdf / np.where(tot > 0, tot, 1.0)

            # CRPS via the kernel identity
            #     CRPS(F, z) = E|Z - z| - (1/2) E|Z - Z'| ,
            #     (1/2) E|Z - Z'| = integral of F(1-F) dx
            # Integrating (F - 1{x>=z})^2 on a grid instead puts the step's
            # kink between nodes and carries an O(dz/2) = 0.01 bias, which is
            # 2-5% of a typical CRPS and would swamp the system differences
            # this module exists to measure. Here |Z - z| is evaluated exactly
            # at the support points and F(1-F) is smooth, so both terms are
            # O(dz^2). Validated against the closed form in J5.
            c = (pdf * np.abs(MID[None, :] - zc[m][:, None])).sum(axis=1)
            c -= np.trapezoid(M * (1.0 - M), dx=DZ, axis=1)
            c[~have_z[m]] = np.nan
            crps[m] = c
            p = M[np.arange(len(m)), zi[m]]
            p[~have_z[m]] = np.nan
            pit[m] = p

            if full:
                m1 = pdf @ MID
                m2 = pdf @ (MID ** 2)
                mu[m] = m1
                sd[m] = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
                cpdf = np.cumsum(pdf, axis=1)
                cmom = np.cumsum(pdf * MID[None, :], axis=1)
                for a in ALPHAS:
                    k = np.clip((cpdf < a).sum(axis=1), 0, G - 2)
                    ar = np.arange(len(m))
                    var[a][m] = MID[k]
                    es[a][m] = cmom[ar, k] / a
                    # J6 · tail-support floor (parametric benchmark exempt:
                    # it has no empirical cell behind it)
                    if sysname != "normal":
                        # (i) cheap proxy pre-screen
                        thin = (ess_mix * a) < MIN_TAIL_EFF
                        # (ii) AUDIT ITEM 6 — the binding, direct measurement:
                        # how much support actually sits below THIS row's VaR.
                        # Cumulative weight / squared weight give a Kish ESS of
                        # exactly the tail; the raw count is archetype-
                        # independent under soft weighting.
                        key = f"cwt__{h}__{wkind}__{win}"
                        if key in D.files:
                            cw = D[key][vi]                    # (NA, NB)
                            cs = D[f"csq__{h}__{wkind}__{win}"][vi]
                            cn = D[f"cnt__{h}__{wkind}__{win}"][vi]  # (NA, NB)
                            kb = np.clip(k, 0, cw.shape[1] - 1)
                            Wt = np.einsum("na,na->n", Wv, cw[:, kb].T)
                            St = np.einsum("na,na->n", Wv * Wv, cs[:, kb].T)
                            # LIMITATION L3 — both quantities below count
                            # OVERLAPPING h-day windows. `ess_mix` above is
                            # already horizon-corrected because s08 deflates
                            # the stored ESS, but these two are recomputed
                            # here from the cumulative arrays and are not.
                            # Deflate the Kish ESS by h and raise the raw bar
                            # by h, so both floors count independent
                            # observations and a 5-day tail has to clear the
                            # same evidential bar a 1-day tail does.
                            ess_t = np.where(
                                St > 0, Wt * Wt / np.maximum(St, 1e-30),
                                0.0) / h
                            # same weights as the density itself: one-hot
                            # under hard weighting picks that archetype's own
                            # count, soft reproduces the pooled total
                            n_raw = np.einsum("na,na->n", Wv, cn[:, kb].T)
                            direct_thin = ((ess_t < MIN_TAIL_EFF)
                                           | (n_raw < MIN_TAIL_RAW * h))
                            n_direct[(sysname, h, a)] = (
                                n_direct.get((sysname, h, a), 0)
                                + int((direct_thin & ~thin).sum()))
                            thin = thin | direct_thin
                        if thin.any():
                            var[a][m[thin]] = np.nan
                            es[a][m[thin]] = np.nan
                            n_thin[(sysname, h, a)] = (
                                n_thin.get((sysname, h, a), 0) + int(thin.sum()))

        out[f"{sysname}_h{h}_crps"] = crps
        out[f"{sysname}_h{h}_pit"] = pit
        if full:
            out[f"{sysname}_h{h}_mean"] = mu
            out[f"{sysname}_h{h}_sd"] = sd
            for a in ALPHAS:
                tag = f"{int(a * 100):02d}"
                out[f"{sysname}_h{h}_var{tag}"] = var[a]
                out[f"{sysname}_h{h}_es{tag}"] = es[a]
                # A cell with NO served VaR is the tail floor doing its job,
                # not a gate failure -- at h=20 under the L3 horizon scaling
                # every row is withheld, and an empty slice must read as
                # "nothing to check" rather than crash the stage.
                _fin = np.isfinite(var[a]) & np.isfinite(es[a])
                if _fin.any() and np.min(var[a][_fin] - es[a][_fin]) < -1e-3:
                    j4 = False
        print(f"  h={h:>2} {sysname:10s} done")

    for s in FULL:
        v1 = out[f"{s}_h{h}_var01"]
        v5 = out[f"{s}_h{h}_var05"]
        ok = np.isfinite(v1) & np.isfinite(v5)
        if ok.any() and (v1[ok] - v5[ok]).max() > 1e-6:
            j4 = False

# ---- J6 · tail-support floor ------------------------------------------------
# How many rows were denied a VaR/ES because their mixture could not support
# the level? Reported, never absorbed silently. At 1% the requirement is a
# mixture ESS of 5,000; at 5% it is 1,000.
print(f"\n[J6] tail-support floor (MIN_TAIL_EFF={MIN_TAIL_EFF}) — rows "
      f"suppressed of {N:,}")
if not n_thin:
    print("     none: every scored row's mixture supports both levels")
else:
    print(f"     {'system':12s}{'h':>4}{'alpha':>7}{'suppressed':>13}{'share':>9}")
    for (sysn, h, a) in sorted(n_thin):
        c = n_thin[(sysn, h, a)]
        print(f"     {sysn:12s}{h:>4}{a:>7.2f}{c:>13,}{c / N:>9.3%}")

print(f"\n[J1] every mixture a proper CDF: {'PASS' if j1 else 'FAIL'}")
assert j1

# ---- J2 · degenerate weight ------------------------------------------------
vtest = NV - 1
Ctest = D["cdf__5__soft__roll5y"][vtest]
worst = 0.0
for a in range(NA):
    w = np.zeros((1, NA), dtype=np.float32)
    w[0, a] = 1.0
    worst = max(worst, float(np.abs((w @ Ctest) - Ctest[a]).max()))
print(f"[J2] one-hot weight reproduces the archetype CDF: max|diff| "
      f"{worst:.3e}  {'PASS' if worst < 1e-6 else 'FAIL'}")
assert worst < 1e-6

# ---- J3 · PIT range and grid truncation ------------------------------------
print("[J3] PIT range and grid truncation")
j3 = True
for h in HORIZONS:
    for s in list(SPEC) + ["normal", "t_ewma", "evt"]:
        p = out[f"{s}_h{h}_pit"]
        p = p[np.isfinite(p)]
        if len(p) and (p.min() < -1e-6 or p.max() > 1 + 1e-6):
            j3 = False
    nz = int(np.isfinite(Zr[h]).sum())
    print(f"     h={h:>2}: |z| beyond grid edge +/-{GRID[-1]:.0f}: "
          f"{trunc[h]:,} of {nz:,} ({trunc[h] / max(nz, 1):.5%})")
print(f"     PIT within [0,1] for all systems: {'PASS' if j3 else 'FAIL'}")
assert j3

# ---- J4 · coherence --------------------------------------------------------
print(f"[J4] VaR(1%) <= VaR(5%) and ES <= VaR: {'PASS' if j4 else 'FAIL'}")
assert j4

# ---- J5 · CRPS validated against the closed form ---------------------------
print("[J5] numerical CRPS vs closed form for N(0,1)")
j5 = True
for h in HORIZONS:
    z = Zr[h]
    ok = np.isfinite(z) & np.isfinite(out[f"normal_h{h}_crps"])
    zc = np.clip(z[ok], GRID[0], GRID[-1])
    exact = zc * (2 * norm.cdf(zc) - 1) + 2 * norm.pdf(zc) - 1 / np.sqrt(np.pi)
    # LIMITATION L3 — a horizon with no served rows has nothing to validate.
    if not ok.any():
        print(f"     h={h:>2}: no served rows — nothing to validate (the "
              f"horizon-scaled tail floor withheld every forecast)")
        continue
    err = float(np.abs(out[f"normal_h{h}_crps"][ok] - exact).max())
    good = err < 5e-3
    j5 &= good
    print(f"     h={h:>2}: max|numeric - closed form| {err:.2e} over "
          f"{ok.sum():,} rows  {'ok' if good else 'FAIL'}")
print(f"     {'PASS' if j5 else 'FAIL'}")
assert j5, "grid integration of CRPS is inaccurate"

# ------------------------------------------------ assemble ------------------
res = zo.select(["cisin", "TR_DATE", "era", "vintage_id", "p_max", "pa_max",
                 "arch_soft", "arch_hard", "sigma_ewma", "stock_out_of_window"]
                + [f"z_h{h}" for h in HORIZONS]
                + [f"sigma_adj_h{h}" for h in HORIZONS])
res = res.with_columns([pl.Series(k, v) for k, v in out.items()])
# Risk numbers in LOG-return space, primary system and horizon. C4 builds
# forward LOG returns, so z * sigma * sqrt(h) is a log return and naming it
# `_ret` implied a simple return the field never held — the query layer
# applies expm1 for display, so the stored field and the served number
# disagreed in units. Named for what it is; convert with expm1 if a simple
# return is wanted.
for h in HORIZONS:
    res = res.with_columns(
        # the scale that standardised THIS horizon's outcome, not the raw
        # state — otherwise the return-space number contradicts the z-space one
        (pl.col(f"soft_roll_h{h}_var05") * pl.col(f"sigma_adj_h{h}")
         * np.sqrt(h)).alias(f"soft_roll_h{h}_var05_logret"),
        (pl.col(f"soft_roll_h{h}_es05") * pl.col(f"sigma_adj_h{h}")
         * np.sqrt(h)).alias(f"soft_roll_h{h}_es05_logret"))
OUT = OUTPUTS / "phase3" / f"predictive{_SUF}.parquet"
res.write_parquet(OUT)
write_lineage(OUT, inputs=[OUTPUTS / "phase3" / f"archetype_probs{_SUF}.parquet",
                          OUTPUTS / "phase3" / f"outcome_densities{_SUF}.npz"],
              extras={"module": "C5", "suffix": _SUF, "rows": res.height,
                      "min_tail_eff": MIN_TAIL_EFF, "min_tail_raw": MIN_TAIL_RAW})

# ------------------------------------------------ report --------------------
print("\n" + "=" * 72)
print("C5 SUMMARY  (descriptive — all testing happens in C6)")
print(f"  rows {res.height:,} | columns {len(res.columns)} | "
      f"{OUT.stat().st_size / 1e6:.0f} MB")

print("\n  mean CRPS by system (lower is better) — NOT yet a test:")
print(f"    {'system':12s}" + "".join(f"{'h=' + str(h):>12}" for h in HORIZONS))
for s in list(SPEC) + ["normal", "t_ewma", "evt"]:
    row = "".join(f"{np.nanmean(out[f'{s}_h{h}_crps']):>12.5f}"
                  for h in HORIZONS)
    print(f"    {s:12s}{row}")

print("\n  PIT calibration, h=5 (a perfect model gives 0.500 / 0.050 / 0.050):")
print(f"    {'system':12s}{'mean':>9}{'P(u<.05)':>11}{'P(u>.95)':>11}")
for s in list(SPEC) + ["normal", "t_ewma", "evt"]:
    p = out[f"soft_roll_h5_pit"] if False else out[f"{s}_h5_pit"]
    p = p[np.isfinite(p)]
    print(f"    {s:12s}{p.mean():>9.3f}{(p < .05).mean():>11.3f}"
          f"{(p > .95).mean():>11.3f}")

print("\n  risk output, h=5 soft_roll, by hard archetype "
      "(standardised units):")
ha = res["arch_hard"].to_numpy()
v5 = out["soft_roll_h5_var05"]
e5 = out["soft_roll_h5_es05"]
sdm = out["soft_roll_h5_sd"]
print(f"    {'archetype':16s}{'n':>9}{'VaR5%':>9}{'ES5%':>9}{'mix sd':>9}")
for a in ARCH:
    m = (ha == a) & np.isfinite(v5)
    if m.sum():
        print(f"    {a:16s}{m.sum():>9,}{np.nanmean(v5[m]):>9.3f}"
              f"{np.nanmean(e5[m]):>9.3f}{np.nanmean(sdm[m]):>9.3f}")

print(f"\n  total {time.time() - t_start:.1f}s")
print("=" * 72)
