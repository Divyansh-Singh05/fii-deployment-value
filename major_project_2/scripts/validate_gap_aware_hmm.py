"""
VALIDATION — gap-aware Gaussian HMM (audit item 3)

Four gates, each with its rule stated before the result.

  G1  degenerate  with every gap = 1 the estimator must reproduce hmmlearn's
                  GaussianHMM: same log-likelihood to 1e-4 relative, same
                  means to 1e-3. If it does not, the new code is not an
                  extension of the old one, it is a different model.
  G2  likelihood  the scaled forward-backward log-likelihood must match a
                  direct, independent computation (brute-force enumeration of
                  state paths on a short sequence).
  G3  monotone    every EM iteration must increase the observed-data
                  log-likelihood.
  G4  recovery    THE DECISIVE ONE. Simulate a Markov chain at DAILY
                  frequency with a known A, then observe it only on scattered
                  days. The gap-aware fit must recover the daily A; a
                  per-observation fit (what C1 does today) must NOT, and must
                  fail in the measured direction — under-persistent.

Run:  python scripts/validate_gap_aware_hmm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fii.models.gap_aware_hmm import (GapAwareGaussianHMM,  # noqa: E402
                                      trading_day_gaps)

K, D = 3, 2
rng = np.random.default_rng(7)


def half_life(A):
    ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    l2 = ev[1]
    return np.log(0.5) / np.log(l2) if 0 < l2 < 1 else np.inf


def simulate(A_daily, means, sd, n_stocks, n_days, keep_prob, seed):
    """A latent chain that moves EVERY day; observations on a random subset."""
    r = np.random.default_rng(seed)
    Xs, lens, gaps_all, states = [], [], [], []
    for _ in range(n_stocks):
        s = r.integers(K)
        path = np.empty(n_days, dtype=int)
        for t in range(n_days):
            path[t] = s
            s = r.choice(K, p=A_daily[s])
        obs_days = np.where(r.random(n_days) < keep_prob)[0]
        if len(obs_days) < 30:
            continue
        st = path[obs_days]
        x = means[st] + sd * r.standard_normal((len(obs_days), D))
        Xs.append(x)
        lens.append(len(obs_days))
        gaps_all.append(np.concatenate([[0], np.diff(obs_days)]))
        states.append(st)
    return (np.vstack(Xs), lens, np.concatenate(gaps_all),
            np.concatenate(states))


print("=" * 72)
print("G1 · DEGENERATE — all gaps = 1 must reproduce hmmlearn")
print("=" * 72)
A1 = np.array([[.94, .04, .02], [.03, .94, .03], [.02, .04, .94]])
means = np.array([[-1.2, 0.4], [0.0, -0.3], [1.3, 0.5]])
X, lens, _, _ = simulate(A1, means, 0.6, 40, 400, 1.0, seed=11)
g1 = np.zeros(len(X), dtype=np.int64)
pos = 0
for n in lens:
    g1[pos + 1:pos + n] = 1
    pos += n

m_new = GapAwareGaussianHMM(K, n_iter=120, tol=1e-6, random_state=3).fit(X, lens, g1)
try:
    from hmmlearn.hmm import GaussianHMM
    m_ref = GaussianHMM(n_components=K, covariance_type="diag",
                        n_iter=120, tol=1e-6, random_state=3)
    m_ref.fit(X, lens)
    ll_ref = m_ref.score(X, lens)
    o = np.argsort(m_ref.means_[:, 0])
    dmu = np.abs(m_new.means_ - m_ref.means_[o]).max()
    rel = abs(m_new.loglik_ - ll_ref) / abs(ll_ref)
    print(f"  gap-aware loglik {m_new.loglik_:,.3f} | hmmlearn {ll_ref:,.3f}"
          f" | rel diff {rel:.2e}")
    print(f"  max |mean difference| {dmu:.2e}")
    g1_pass = rel < 1e-4 and dmu < 1e-3
except ImportError:
    g1_pass = False
    print("  hmmlearn unavailable")
print(f"  G1: {'PASS' if g1_pass else 'FAIL'}\n")

print("=" * 72)
print("G2 · LIKELIHOOD — scaled recursion vs brute-force path enumeration")
print("=" * 72)
m = GapAwareGaussianHMM(K)
m.means_, m.covars_ = means, np.full((K, D), 0.36)
m.startprob_ = np.array([0.2, 0.5, 0.3])
m.transmat_ = A1
Xs = rng.standard_normal((6, D))
gs = np.array([0, 1, 3, 1, 7, 2], dtype=np.int64)
Apow = m._powers(A1, int(gs.max()))
logB = m._log_emissions(Xs)
tot = 0.0
for path in np.ndindex(*([K] * 6)):
    lp = np.log(m.startprob_[path[0]]) + logB[0, path[0]]
    for t in range(1, 6):
        lp += np.log(Apow[gs[t]][path[t - 1], path[t]]) + logB[t, path[t]]
    tot += np.exp(lp)
brute = np.log(tot)
scaled = m._e_step(Xs, [6], gs, A1, Apow)[3]
print(f"  brute force {brute:.10f} | scaled recursion {scaled:.10f}")
g2_pass = abs(brute - scaled) < 1e-8
print(f"  G2: {'PASS' if g2_pass else 'FAIL'}  (|diff| {abs(brute-scaled):.2e})\n")

print("=" * 72)
print("G3 · MONOTONE — EM must never decrease the log-likelihood")
print("=" * 72)
A_true = np.array([[.966, .024, .010], [.015, .970, .015], [.010, .024, .966]])
X, lens, gaps, st = simulate(A_true, means, 0.6, 60, 900, 0.55, seed=21)
mg = GapAwareGaussianHMM(K, n_iter=150, tol=1e-7, random_state=5).fit(X, lens, gaps)
h = np.array(mg.history_)
drops = int((np.diff(h) < -1e-6).sum())
print(f"  iterations {len(h)} | loglik {h[0]:,.1f} -> {h[-1]:,.1f} "
      f"| decreases {drops}")
g3_pass = drops == 0 and mg.monotone_ok_
print(f"  G3: {'PASS' if g3_pass else 'FAIL'}\n")

print("=" * 72)
print("G4 · RECOVERY — known daily A observed at irregular times")
print("=" * 72)
print(f"  simulated {len(X):,} observations over {len(lens)} stocks; "
      f"{(gaps > 1).mean():.1%} of transitions span k > 1 "
      f"(the real panel: 14.65%)")
naive_gaps = np.where(gaps > 0, 1, 0).astype(np.int64)   # what C1 does today
mn = GapAwareGaussianHMM(K, n_iter=150, tol=1e-7, random_state=5).fit(
    X, lens, naive_gaps)

print(f"\n  {'':22s}{'mean diag':>12}{'lambda2':>10}{'half-life':>12}")


def line(lbl, A):
    ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    print(f"  {lbl:22s}{np.mean(np.diag(A)):>12.4f}{ev[1]:>10.4f}"
          f"{half_life(A):>12.2f}")


line("TRUE daily A", A_true)
line("gap-aware fit", mg.transmat_)
line("per-observation fit", mn.transmat_)

err_g = np.abs(mg.transmat_ - A_true).max()
err_n = np.abs(mn.transmat_ - A_true).max()
hl_t, hl_g, hl_n = half_life(A_true), half_life(mg.transmat_), half_life(mn.transmat_)
print(f"\n  max |A - A_true|   gap-aware {err_g:.4f} | per-observation {err_n:.4f}")
print(f"  half-life error    gap-aware {100*(hl_g-hl_t)/hl_t:+.1f}% | "
      f"per-observation {100*(hl_n-hl_t)/hl_t:+.1f}%")

g4_pass = err_g < 0.02 and err_g < err_n / 3 and hl_n < hl_t
print(f"\n  G4: {'PASS' if g4_pass else 'FAIL'}"
      f"  (gap-aware recovers A; per-observation is under-persistent as "
      f"predicted)\n")

print("=" * 72)
print("G5 · BATCHED E-STEP — the vectorised rewrite equals the reference loop")
print("=" * 72)
print("  `_e_step` steps every sequence forward together instead of one")
print("  Python iteration per observation. That is a performance change only,")
print("  so it has to reproduce `_e_step_loop` to floating-point noise on")
print("  panel-shaped data with real gap structure, under arbitrary")
print("  parameters -- not just at the fitted optimum.")

_A5 = np.array([[0.960, 0.030, 0.010],
                [0.025, 0.950, 0.025],
                [0.008, 0.032, 0.960]])
X5, L5, g5, _ = simulate(_A5, np.array([[-1.2, 0.3], [0.0, 0.0], [1.2, -0.3]]),
                         0.9, n_stocks=300, n_days=900, keep_prob=0.55,
                         seed=505)
_m5 = GapAwareGaussianHMM(n_components=3)
_r5 = np.random.default_rng(505)
w_gam = w_xi = w_sm = w_ll = 0.0
t_loop = t_bat = 0.0
for _ in range(3):
    _m5.means_ = np.array([np.quantile(X5, q, axis=0) for q in (.2, .5, .8)]) \
        + 0.05 * _r5.standard_normal((3, X5.shape[1]))
    _m5.covars_ = np.tile(X5.var(axis=0), (3, 1)) * _r5.uniform(0.7, 1.4) + 1e-3
    _sp = _r5.uniform(0.2, 1.0, 3)
    _m5.startprob_ = _sp / _sp.sum()
    _Ar = _r5.uniform(0.005, 0.05, (3, 3))
    np.fill_diagonal(_Ar, 0.0)
    _Ar += np.diag(1.0 - _Ar.sum(axis=1))
    _Ap = _m5._powers(_Ar, int(g5.max()))
    _t = time.time(); _a = _m5._e_step_loop(X5, L5, g5, _Ar, _Ap); t_loop += time.time() - _t
    _t = time.time(); _b = _m5._e_step(X5, L5, g5, _Ar, _Ap);      t_bat += time.time() - _t
    w_gam = max(w_gam, np.abs(_a[0] - _b[0]).max())
    w_xi = max(w_xi, np.abs(_a[1] - _b[1]).max() / max(np.abs(_a[1]).max(), 1))
    w_sm = max(w_sm, np.abs(_a[2] - _b[2]).max())
    w_ll = max(w_ll, abs(_a[3] - _b[3]) / abs(_a[3]))

print(f"\n  {len(X5):,} observations / {len(L5)} sequences / max gap {g5.max()}"
      f" / {100*(g5[g5>0]>1).mean():.1f}% of steps span k > 1")
print(f"  3 random parameter draws:")
print(f"    posteriors  max|diff|    {w_gam:.3e}")
print(f"    Xi          max rel diff {w_xi:.3e}")
print(f"    startprob   max|diff|    {w_sm:.3e}")
print(f"    loglik      max rel diff {w_ll:.3e}")
print(f"  timing        loop {t_loop/3:.3f}s   batched {t_bat/3:.3f}s"
      f"   speedup {t_loop/max(t_bat,1e-9):.1f}x")

g5_pass = (w_gam < 1e-10 and w_xi < 1e-10 and w_sm < 1e-9 and w_ll < 1e-12)
print(f"\n  G5: {'PASS' if g5_pass else 'FAIL'}\n")

print("=" * 72)
allp = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
for nm, p in (("G1 degenerate", g1_pass), ("G2 likelihood", g2_pass),
              ("G3 monotone", g3_pass), ("G4 recovery", g4_pass),
              ("G5 batched E-step", g5_pass)):
    print(f"  {'PASS' if p else 'FAIL'}  {nm}")
print("=" * 72)
sys.exit(0 if allp else 1)
