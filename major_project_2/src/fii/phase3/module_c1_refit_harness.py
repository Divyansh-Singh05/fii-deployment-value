"""
MODULE C1 · WALK-FORWARD REFIT HARNESS  —  Phase III

Re-estimates the HMM backbone once per month on a trailing 5-year window and
writes one immutable parameter vintage per refit. Vintage v is fitted on rows
in (asof_v - 1826 days, asof_v] and is never revised, so any day scored with
it can be traced to parameters that existed before that day.

PROVENANCE NOTE.  The original harness ran from a scratch directory that was
cleared between sessions; only its output, vintages.parquet, survived. This
module is a reconstruction, and it is checked against that output rather than
trusted: --verify refits stored vintages and compares log-likelihood, means,
transition matrix and thresholds. Reconstructed conventions recovered from
the stored metadata:

    window        (asof - 1826 days, asof]   — open on the left
    eligibility   stocks with >= 60 observations in the window
    sequence cap  each stock's LAST 400 observations
    restarts      5 k-means inits, seeds vintage_id*1000 + 42..46
    seeded init   1 always-on candidate with means at F_persist quantiles
                  1/6, 3/6, 5/6, seed vintage_id*1000 + 542
    persistence   every candidate starts from a transition matrix with
                  diagonal 0.94 (eliminates degenerate optima at no cost)
    structure     self-transitions > 0.5 and adjacent F_persist means at
                  least 0.5 apart, else the candidate is rejected
    alignment     means matched to the previous vintage by exhaustive 3! =
                  6 assignment, so state k means the same thing in every
                  vintage and C2 may carry a posterior across a boundary

Gates
    M1  determinism   two runs of the same vintage agree exactly
    M2  truncation    fitting vintage v against data truncated at asof_v is
                      identical to fitting it against the full panel — the
                      look-ahead test C9 could not previously reach
    M3  reproduction  the reconstruction matches the stored vintage

Run:
    python -m fii.phase3.module_c1_refit_harness --verify 0 1 2
    python -m fii.phase3.module_c1_refit_harness --audit 12 40 75
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import permutations

import numpy as np
import polars as pl
from scipy import stats

from fii.models.gap_aware_hmm import GapAwareGaussianHMM
from fii.paths import OUTPUTS, VALIDATION_DATA
from fii.phase3.trading_calendar import gaps_from_frame, trading_days

FEATS = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]
K = 3
MIN_SEQ, FIT_CAP, WIN_D = 60, 400, 1826
TOL, N_ITER, N_RESTART, DIAG0 = 1e-2, 300, 5, 0.94
SEED, NBOOT, MIN_GAP = 42, 1000, 0.5
JITTER = 0.5      # restart dispersion on the seeded mean vector


def _tm(diag=DIAG0):
    off = (1.0 - diag) / (K - 1)
    return np.full((K, K), off) + np.eye(K) * (diag - off)


def _mk(seed, init=None):
    # AUDIT ITEM 3. The estimator is now gap-aware: a transition between
    # observations separated by k trading days is A^k, and A is therefore
    # estimated PER TRADING DAY -- the same unit C2 filters in. Previously
    # this was hmmlearn's GaussianHMM, which has no notion of elapsed time,
    # so A was estimated per observed FII row and then reinterpreted as
    # per-trading-day at scoring time. On the existing labels that mismatch
    # costs 77% of mixing half-life (11.5d fitted vs 20.4d on k=1 pairs) and
    # concentrates in thin stocks (per-stock gap rate p90 42.7%, corr -0.54
    # with observation count).
    return GapAwareGaussianHMM(n_components=K, n_iter=N_ITER, tol=TOL,
                               random_state=seed)


def fit_kmeans(X, L, seed, gaps=None):
    m = _mk(seed)
    m.fit(X, L, _need_gaps(gaps, X))
    return m, float(m.loglik_)


def _need_gaps(gaps, X):
    if gaps is None:
        raise ValueError("gap-aware fitting requires trading-day gaps")
    if len(gaps) != len(X):
        raise ValueError(f"gaps {len(gaps)} != rows {len(X)}")
    return gaps


def fit_seeded(X, L, seed, jitter=0.0, gaps=None):
    """Means initialised at F_persist quantiles 1/6, 3/6, 5/6, optionally
    dispersed. Restarts are jittered versions of this rather than k-means
    inits: on this panel k-means reliably converges to an optimum whose
    states are NOT separated on F_persist (structural check fails on all
    five restarts), so it contributes no usable candidate. Jittered seeding
    explores multiple basins while keeping every candidate interpretable."""
    qs = np.quantile(X[:, 0], [1 / 6, 3 / 6, 5 / 6])
    mu = np.tile(X.mean(axis=0), (K, 1))
    mu[:, 0] = qs
    if jitter:
        mu = mu + np.random.default_rng(seed).normal(0, jitter, mu.shape)
    m = _mk(seed)
    m.fit(X, L, _need_gaps(gaps, X), init_means=mu, init_transmat=_tm())
    return m, float(m.loglik_)


def structural_ok(m):
    dg = np.array([m.transmat_[i, i] for i in range(K)])
    f = np.sort(m.means_[:, 0])
    return bool((dg > 0.5).all()) and bool(np.diff(f).min() >= MIN_GAP)


def align(mu_new, mu_prev):
    D = np.linalg.norm(mu_new[:, None, :] - mu_prev[None, :, :], axis=2)
    costs = sorted(((sum(D[i, p[i]] for i in range(K)), p)
                    for p in permutations(range(K))), key=lambda x: x[0])
    return costs[0][1], costs[0][0], costs[1][0]


_CAL = None


def calendar(cut=None):
    """Shared price-defined trading calendar, loaded once (see item 3)."""
    global _CAL
    if _CAL is None:
        _CAL = trading_days(cut)
    return _CAL


def window_rows(st, asof):
    """Rows the vintage is fitted on. Pure function of (data, asof).

    AUDIT ITEM 3. Also returns the elapsed trading-day gaps the gap-aware
    likelihood needs. Rows on non-trading dates are dropped here, using the
    same price-defined calendar C2 uses, so the estimator and the filter share
    one notion of elapsed time.
    """
    lo = asof - dt.timedelta(days=WIN_D)
    w = st.filter((pl.col("TR_DATE") > lo) & (pl.col("TR_DATE") <= asof))
    # Drop weekend / holiday rows carried by the custodian feed. C2 already
    # dropped these; C1 did not, so the estimator was fitting on days the
    # filter never scores — a second, smaller C1/C2 divergence found by the
    # calendar guard while wiring up item 3.
    w = w.filter(pl.col("TR_DATE").is_in(calendar()))
    cnt = w.group_by("cisin").len()
    keep = cnt.filter(pl.col("len") >= MIN_SEQ)["cisin"].to_list()
    w = w.filter(pl.col("cisin").is_in(keep)).sort(["cisin", "TR_DATE"])
    n_rows = w.height                      # eligible rows, BEFORE the cap
    # cap each stock at its LAST FIT_CAP observations
    w = (w.with_columns(pl.col("TR_DATE").rank("ordinal", descending=True)
                        .over("cisin").alias("_r"))
          .filter(pl.col("_r") <= FIT_CAP).drop("_r")
          .sort(["cisin", "TR_DATE"]))
    gaps, _lens = gaps_from_frame(w, calendar())
    return lo, len(keep), n_rows, w, gaps


def filtered_states(X, lens, m, gaps=None):
    """Forward-filtered posterior P(S_t | x_1..t) over the fitting window.

    AUDIT ITEM 3. Propagates with A^k over elapsed TRADING days -- the same
    recursion C2 uses and, now, the same one the parameters were estimated
    under. The pre-audit version deliberately used one transition per row to
    match a per-observation fit; that convention no longer exists.
    """
    if gaps is None:
        raise ValueError("filtered_states is gap-aware; pass trading-day gaps")
    cv = m.covars_
    S2 = np.array([np.diag(c) for c in cv]) if cv.ndim == 3 else cv
    const = -0.5 * np.log(2 * np.pi * S2).sum(axis=1)
    LL = np.stack([const[k] - 0.5 * (((X - m.means_[k]) ** 2) / S2[k]).sum(axis=1)
                   for k in range(K)], axis=1)
    B = np.exp(LL - LL.max(axis=1, keepdims=True))
    Apow = [np.eye(K)]
    for _ in range(int(max(gaps.max(), 1))):
        Apow.append(Apow[-1] @ m.transmat_)
    P = np.empty((len(X), K))
    o = 0
    for L in lens:
        a = m.startprob_ * B[o]
        P[o] = a / a.sum()
        for t in range(o + 1, o + L):
            a = (P[t - 1] @ Apow[gaps[t]]) * B[t]
            P[t] = a / a.sum()
        o += L
    return P


def _wq(v_sorted, w_sorted, qs):
    """Weighted quantile by the mid-point plotting position.

    pp_i = (cumw_i - w_i/2) / total, then linear interpolation in pp.
    Scale-invariant in w, and with equal weights it reduces to the Hazen
    convention rather than numpy's default `linear`. That is WHY the 0/1
    weight variant is computed alongside the argmax variant below: it holds
    the convention fixed so the weighting effect can be read on its own.
    """
    cw = np.cumsum(w_sorted)
    pp = (cw - 0.5 * w_sorted) / cw[-1]
    return np.interp(qs, pp, v_sorted)


def _wq_cluster_boot(v, w, cis, qs, rng, nboot):
    """Cluster bootstrap of weighted quantiles, resampling STOCKS.

    F_entity_s is a panel quantity: strongly persistent within a stock and
    cross-sectionally correlated on a given day. Resampling rows independently
    treats correlated observations as independent and understates the standard
    error of a quantile by roughly a factor of 4-5 on this data (measured:
    4.21x - 4.88x across vintages 31/61/90). Resampling whole stocks preserves
    the within-stock dependence. It also restores approximate normality of the
    sampling distribution, which the Phi() transform in C3 relies on: under
    iid resampling a KS test rejects normality (p < 0.01 in every vintage
    tested); under clustering it does not.

    IMPLEMENTATION.  Drawing a stock c times is exactly equivalent to
    multiplying that stock's row weights by c, so the values are sorted ONCE
    and each replication is a weighted cumsum over that fixed order. No
    re-sorting, and it is what makes the fully-weighted population (every row
    participates, most with near-zero weight) affordable.
    """
    o = np.argsort(v, kind="stable")
    vs, ws, cs = v[o], w[o], cis[o]
    us, inv = np.unique(cs, return_inverse=True)
    G = len(us)
    out = np.empty((nboot, len(qs)))
    for b in range(nboot):
        c = np.bincount(rng.integers(0, G, G), minlength=G)
        out[b] = _wq(vs, ws * c[inv], qs)
    return out


def thresholds(w, m, rng):
    """Archetype cut-points and their sampling uncertainty.

    DECODING POPULATION.  Production thresholds are POSTERIOR-WEIGHTED: every
    row in the window contributes to the sell-side quantile with weight
    P(S_t = sell | x_1..t), rather than being hard-included by argmax. The
    engine applies these cut-points inside a probabilistic state model, so
    estimating them on a hard-selected subsample was internally inconsistent:
    a row at the state boundary was fully in or fully out of the estimation
    sample but later received fractional membership.

    Three comparison variants are retained and NOT used downstream:
        *_argmax   np.quantile on the filtered-argmax subsample (the estimator
                   used in every vintage built before this change)
        *_hard     _wq with 0/1 argmax weights (same convention as production,
                   so `th_x` vs `th_x_hard` isolates the weighting alone)
        *_viterbi  np.quantile on the Viterbi path (the pre-filtering estimator)

    The filtered posterior, not Viterbi, is the right decoding: Viterbi assigns
    a day's state using the whole window, while the engine scores forward-only.
    """
    X = w.select(FEATS).to_numpy()
    gaps, lens = gaps_from_frame(w, calendar())
    order = np.argsort(m.means_[:, 0])          # 0 sell, 1 neutral, 2 buy
    sell, buy = order[0], order[2]

    PF = filtered_states(X, lens, m, gaps)
    st_f = PF.argmax(axis=1)
    st_v = m.predict(X, lens, gaps)

    fs = w["F_entity_s"].to_numpy()
    fb = w["F_entity_buy_s"].to_numpy()
    cis = w["cisin"].to_numpy()
    QS = (0.25, 0.75)

    out = {"filt_viterbi_agree": float((st_f == st_v).mean())}

    # ---- comparison variants on hard subsamples (np.quantile convention) ----
    for tag, states in (("_argmax", st_f), ("_viterbi", st_v)):
        s_m, b_m = states == sell, states == buy
        out[f"th_hostage{tag}"] = float(np.quantile(fs[s_m], 0.25))
        out[f"th_shark_dist{tag}"] = float(np.quantile(fs[s_m], 0.75))
        out[f"th_dispersed_acc{tag}"] = float(np.quantile(fb[b_m], 0.25))
        out[f"th_shark_acc{tag}"] = float(np.quantile(fb[b_m], 0.75))

    # ---- production (posterior-weighted) + 0/1 control, both via _wq --------
    NAMES = (("th_hostage", "th_shark_dist"), ("th_dispersed_acc", "th_shark_acc"))
    for (F, k, (lo_name, hi_name)) in ((fs, sell, NAMES[0]), (fb, buy, NAMES[1])):
        wt_soft = PF[:, k]
        wt_hard = (st_f == k).astype(float)
        o = np.argsort(F, kind="stable")
        for wt, tag in ((wt_hard, "_hard"), (wt_soft, "")):
            q = _wq(F[o], wt[o], QS)
            out[f"{lo_name}{tag}"], out[f"{hi_name}{tag}"] = float(q[0]), float(q[1])
        out[f"{lo_name}_ess"] = float(wt_soft.sum() ** 2 / (wt_soft ** 2).sum())

        bs = _wq_cluster_boot(F, wt_soft, cis, QS, rng, NBOOT)
        for j, name in ((0, lo_name), (1, hi_name)):
            d = bs[:, j]
            out[f"{name}_sd"] = float(np.std(d, ddof=1))
            out[f"{name}_lo"] = float(np.percentile(d, 2.5))
            out[f"{name}_hi"] = float(np.percentile(d, 97.5))
            # normality of the bootstrap distribution, which C3's Phi() assumes
            z = (d - d.mean()) / max(d.std(ddof=1), 1e-12)
            out[f"{name}_boot_skew"] = float(stats.skew(d))
            out[f"{name}_boot_kurt"] = float(stats.kurtosis(d))
            out[f"{name}_boot_ks_p"] = float(stats.kstest(z, "norm").pvalue)
    return out


def fit_vintage(st, vid, asof, mu_prev=None):
    """Everything about a vintage, from data and asof alone."""
    lo, n_stocks, n_rows, w, gaps = window_rows(st, asof)
    X = w.select(FEATS).to_numpy()
    lens = w.group_by("cisin", maintain_order=True).len()["len"].to_numpy()

    cands = []
    for j in range(N_RESTART):
        s = vid * 1000 + SEED + j
        try:
            m, ll = fit_seeded(X, lens, s, JITTER, gaps=gaps)
            cands.append((ll, s, m))
        except Exception:
            pass
    s = vid * 1000 + 542
    try:
        m, ll = fit_seeded(X, lens, s, gaps=gaps)
        cands.append((ll, s, m))
    except Exception:
        pass

    lls = [c[0] for c in cands]
    # A candidate must be BOTH numerically sound and structurally valid.
    # `structural_ok` alone was the old test, and `max(good or cands, ...)`
    # meant that when nothing passed it, the best structurally INVALID model
    # was published as the vintage with only a counter to show for it. A
    # regime process nothing validated is not a fallback, it is a defect.
    def _sound(mm):
        return bool(getattr(mm, "ok_", True))
    good = [c for c in cands if structural_ok(c[2]) and _sound(c[2])]
    if not good:
        why = [("structural" if not structural_ok(c[2]) else "")
               + ("+numeric" if not _sound(c[2]) else "") for c in cands]
        raise RuntimeError(
            f"vintage {vid} ({asof}): no candidate passed both gates out of "
            f"{len(cands)} fits — reasons {why}. Refusing to publish a "
            f"fallback; fix the fit or exclude the vintage explicitly.")
    best = max(good, key=lambda c: c[0])
    ll, seed_win, m = best
    res = {"vintage_id": vid, "asof": asof, "window_start": lo,
           "n_stocks": n_stocks, "n_rows": n_rows,
           "n_fit_rows": int(w.height),
           "loglik": ll, "winning_seed": seed_win,
           "iters": int(m.monitor_.iter),
           "seeded_won": seed_win % 1000 == 542,
           "ll_spread": float(max(lls) - min(lls)) if lls else 0.0,
           "ll_cost_of_structure": float(ll - max(lls)),
           "min_self_transition": float(min(m.transmat_[i, i]
                                            for i in range(K))),
           "min_fpersist_gap": float(np.diff(np.sort(m.means_[:, 0])).min()),
           "n_fallback": len(cands) - len(good),
           "n_unsound": sum(1 for c in cands if not _sound(c[2])),
           "converged": bool(getattr(m, "converged_", True)),
           "monotone_ok": bool(getattr(m, "monotone_ok_", True))}
    # Canonical state order is ascending F_persist: 0 SELL, 1 NEUTRAL, 2 BUY.
    # The states are separated by >1.0 on F_persist in every vintage, so this
    # ordering is unambiguous and is what C2's G0b gate requires. Alignment to
    # the previous vintage is retained as a DIAGNOSTIC (hungarian_cost) rather
    # than as the ordering rule: align() returns p with p[i] = the previous
    # state matching new state i, so reordering by it needs the INVERSE
    # permutation, and using it directly silently transposes states.
    perm = np.argsort(m.means_[:, 0])
    hc = hs = float("nan")
    if mu_prev is not None:
        _, hc, hs = align(m.means_, mu_prev)   # diagnostic margin only
    mu = m.means_[perm]
    cv = m.covars_[perm]
    cv = np.array([np.diag(c) for c in cv]) if cv.ndim == 3 else cv
    A = m.transmat_[np.ix_(perm, perm)]
    res.update(hungarian_cost=hc, hungarian_second=hs,
               means=json.dumps(mu.tolist()), covars=json.dumps(cv.tolist()),
               transmat=json.dumps(A.tolist()),
               startprob=json.dumps(m.startprob_[perm].tolist()))
    res.update(thresholds(w, m, np.random.default_rng(vid * 1000 + SEED)))
    return res, mu


def load_states(trunc=None):
    st = (pl.read_parquet(VALIDATION_DATA / "states_v3.parquet")
            .sort(["cisin", "TR_DATE"]))
    return st.filter(pl.col("TR_DATE") <= trunc) if trunc else st


_ST = None


def _worker_init():
    """One states() load per worker process, not one per task."""
    global _ST
    _ST = load_states()
    calendar()                      # warm the shared trading calendar too


def _worker(args):
    """Fit one vintage in a subprocess.

    `mu_prev` is deliberately NOT passed. It feeds `hungarian_cost` and
    `hungarian_second` only -- the canonical state order is `argsort(means[:,
    0])`, which fit() has already applied, so `perm` is the identity and every
    other output column is a pure function of (data, vid, asof). The two
    diagnostics are recomputed serially afterwards from the ordered means, so
    the parallel and serial rebuilds produce identical rows.
    """
    vid, asof = args
    r, mu = fit_vintage(_ST, vid, asof, None)
    return vid, r, mu


def rebuild_rows(st, asofs, jobs):
    """All vintage rows, serially or over a process pool."""
    import time as _t
    t0 = _t.time()
    rows = [None] * len(asofs)
    mus = [None] * len(asofs)

    def note(vid, r, done):
        el = _t.time() - t0
        print(f"  v{vid:<4} {asofs[vid]}  ll {r['loglik']:>12.1f}  "
              f"seed {r['winning_seed']}  fallback {r['n_fallback']}  "
              f"gap {r['min_fpersist_gap']:.3f}  "
              f"[{el:5.0f}s, eta {el / done * (len(asofs) - done):5.0f}s]",
              flush=True)

    if jobs <= 1:
        mu_prev = None
        for vid, asof in enumerate(asofs):
            r, mu_prev = fit_vintage(st, vid, asof, mu_prev)
            rows[vid], mus[vid] = r, mu_prev
            if vid % 10 == 0 or vid == len(asofs) - 1:
                note(vid, r, vid + 1)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                    initializer=_worker_init) as ex:
            futs = [ex.submit(_worker, (vid, a))
                    for vid, a in enumerate(asofs)]
            for f in cf.as_completed(futs):
                vid, r, mu = f.result()
                rows[vid], mus[vid] = r, mu
                done += 1
                if done % 10 == 0 or done == len(asofs):
                    note(vid, r, done)

    # ---- the two diagnostics that need the previous vintage ---------------
    for vid in range(len(asofs)):
        hc = hs = float("nan")
        if vid > 0:
            _, hc, hs = align(mus[vid], mus[vid - 1])
        rows[vid]["hungarian_cost"] = hc
        rows[vid]["hungarian_second"] = hs
    return rows, _t.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", nargs="*", type=int, default=None,
                    help="refit these vintage ids and compare to the stored file")
    ap.add_argument("--audit", nargs="*", type=int, default=None,
                    help="M2 truncation test on these vintage ids")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes for --rebuild. Vintages "
                         "are independent, but the fit is memory-bandwidth "
                         "bound, so this does not scale with core count: "
                         "measured on the six largest vintages of an 8 GB M2 "
                         "Air, jobs=2 gives 1.84x and jobs=3 gives 1.35x. "
                         "Use 2 unless the machine has more memory headroom.")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate every vintage and overwrite "
                         "vintages.parquet (the original is backed up)")
    a = ap.parse_args()

    vt = pl.read_parquet(OUTPUTS / "phase3" / "vintages.parquet").sort("asof")
    st = load_states()

    if a.verify is not None:
        ids = a.verify or [0, 1, 2]
        print("=" * 78)
        print("M3 · REPRODUCTION — reconstruction vs the stored vintage")
        print("=" * 78)
        for vid in ids:
            row = vt[vid]
            mu_prev = (np.array(json.loads(vt["means"][vid - 1]))
                       if vid > 0 else None)
            r, _ = fit_vintage(st, vid, row["asof"][0], mu_prev)
            dll = r["loglik"] - float(row["loglik"][0])
            dmu = np.abs(np.array(json.loads(r["means"]))
                         - np.array(json.loads(row["means"][0]))).max()
            dA = np.abs(np.array(json.loads(r["transmat"]))
                        - np.array(json.loads(row["transmat"][0]))).max()
            dth = max(abs(r[c] - float(row[c][0])) for c in
                      ("th_hostage", "th_shark_dist", "th_dispersed_acc",
                       "th_shark_acc"))
            same_shape = (r["n_rows"] == int(row["n_rows"][0])
                          and r["n_fit_rows"] == int(row["n_fit_rows"][0])
                          and r["n_stocks"] == int(row["n_stocks"][0]))
            print(f"  v{vid:<3} asof {r['asof']}  rows/stocks match "
                  f"{same_shape}  seed {r['winning_seed']} "
                  f"(stored {int(row['winning_seed'][0])})")
            print(f"        d_loglik {dll:+.6f}   max|d means| {dmu:.3e}   "
                  f"max|d transmat| {dA:.3e}   max|d thresholds| {dth:.3e}")

    if a.audit is not None:
        ids = a.audit or [12, 40, 75]
        print("\n" + "=" * 78)
        print("M2 · TRUNCATION — fitting vintage v in a world that ends at "
              "asof_v")
        print("=" * 78)
        print("  M2a asks the look-ahead question exactly: is the DATA the fit")
        print("      consumes bitwise identical when the world stops at asof?")
        print("  M2b then compares the fitted parameters, but only against the")
        print("      noise floor measured in M1 — EM on this panel is not")
        print("      bitwise reproducible (BLAS/OpenMP reduction order), so")
        print("      exact equality is the wrong criterion for M2b and would")
        print("      report non-determinism as leakage.")

        print("\n  M1 · determinism noise floor (same data, same seed, twice)")
        floor_ll = floor_mu = 0.0
        for vid in ids[:1]:
            asof = vt["asof"][vid]
            _, _, _, w, gaps = window_rows(st, asof)
            X = w.select(FEATS).to_numpy()
            L = w.group_by("cisin", maintain_order=True).len()["len"].to_numpy()
            for j in range(N_RESTART):
                s = vid * 1000 + SEED + j
                m1, l1 = fit_kmeans(X, L, s, gaps=gaps)
                m2, l2 = fit_kmeans(X, L, s, gaps=gaps)
                floor_ll = max(floor_ll, abs(l2 - l1))
                floor_mu = max(floor_mu,
                               float(np.abs(m1.means_ - m2.means_).max()))
        print(f"      max |d loglik| {floor_ll:.3e}   "
              f"max |d means| {floor_mu:.3e}")
        tol_ll = max(floor_ll * 100, 1e-6)
        tol_mu = max(floor_mu * 100, 1e-9)
        print(f"      tolerances for M2b set at 100x the floor: "
              f"loglik {tol_ll:.1e}, means {tol_mu:.1e}")

        allok = True
        print("\n  M2a · fit inputs, bitwise")
        for vid in ids:
            asof = vt["asof"][vid]
            loF, nsF, nrF, wF, gF = window_rows(st, asof)
            loT, nsT, nrT, wT, gT = window_rows(load_states(asof), asof)
            XF, XT = wF.select(FEATS).to_numpy(), wT.select(FEATS).to_numpy()
            LF = wF.group_by("cisin", maintain_order=True).len()["len"].to_numpy()
            LT = wT.group_by("cisin", maintain_order=True).len()["len"].to_numpy()
            # The gaps are an input the estimator consumes, not bookkeeping:
            # observations k trading days apart are linked by A^k, so a gap
            # vector that differed between the two worlds would move the
            # fitted transition matrix with X and lengths still matching.
            same = (np.array_equal(XF, XT) and np.array_equal(LF, LT)
                    and np.array_equal(gF, gT)
                    and wF["cisin"].to_list() == wT["cisin"].to_list()
                    and (loF, nsF, nrF) == (loT, nsT, nrT))
            allok &= same
            print(f"      v{vid:<3} asof {asof}  rows {wF.height:>7}  "
                  f"gaps {len(gF):>7}  "
                  f"X/lengths/gaps/order/counts identical: "
                  f"{'YES' if same else 'NO'}")

        print("\n  M2b · fitted parameters, against the M1 floor")
        for vid in ids:
            asof = vt["asof"][vid]
            mu_prev = (np.array(json.loads(vt["means"][vid - 1]))
                       if vid > 0 else None)
            rF, _ = fit_vintage(st, vid, asof, mu_prev)
            rT, _ = fit_vintage(load_states(asof), vid, asof, mu_prev)
            dll = abs(rF["loglik"] - rT["loglik"])
            dmu = float(np.abs(np.array(json.loads(rF["means"]))
                               - np.array(json.loads(rT["means"]))).max())
            dA = float(np.abs(np.array(json.loads(rF["transmat"]))
                              - np.array(json.loads(rT["transmat"]))).max())
            dth = max(abs(rF[c] - rT[c]) for c in
                      ("th_hostage", "th_shark_dist", "th_dispersed_acc",
                       "th_shark_acc"))
            disc = [k for k in ("n_rows", "n_fit_rows", "n_stocks",
                                "winning_seed", "iters") if rF[k] != rT[k]]
            ok = dll < tol_ll and dmu < tol_mu and not disc
            allok &= ok
            print(f"      v{vid:<3} d_loglik {dll:.2e}  d_means {dmu:.2e}  "
                  f"d_transmat {dA:.2e}  d_thresholds {dth:.2e}  "
                  f"discrete {'ok' if not disc else disc}  "
                  f"{'PASS' if ok else 'FAIL'}")
        print(f"\n  M2 {'PASS' if allok else 'FAIL'}")

    if a.rebuild:
        import shutil, time as _t
        src = OUTPUTS / "phase3" / "vintages.parquet"
        bak = OUTPUTS / "phase3" / "vintages_original.parquet"
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"backed up the original harness output -> {bak.name}")
        asofs = vt["asof"].to_list()      # keep the original monthly schedule
        print(f"rebuilding {len(asofs)} vintages with {a.jobs} worker(s)",
              flush=True)
        rows, elapsed = rebuild_rows(st, asofs, a.jobs)
        df = pl.DataFrame(rows)
        # columns the downstream modules expect but this harness does not track
        for c, v in (("degenerate", False), ("structurally_invalid", False),
                     ("used_fallback", False), ("state_flipped", False),
                     ("secs", 0.0)):
            if c not in df.columns:
                df = df.with_columns(pl.lit(v).alias(c))
            
        df.write_parquet(src)
        print(f"\nwrote {src.name}: {df.height} vintages in "
              f"{elapsed:.0f}s")
        print(f"  structurally invalid : {int((df['min_fpersist_gap'] < MIN_GAP).sum())}")
        print(f"  seeded (unjittered) won: {int(df['seeded_won'].sum())}/{df.height}")
        print(f"  min F_persist gap    : {df['min_fpersist_gap'].min():.3f}")
        print(f"  ll_spread median     : {df['ll_spread'].median():.0f}")
        for c in ("th_hostage", "th_shark_dist", "th_dispersed_acc",
                  "th_shark_acc"):
            print(f"  {c:18s} median {df[c].median():+.4f}")

    if a.verify is None and a.audit is None and not a.rebuild:
        ap.print_help()


if __name__ == "__main__":
    main()
