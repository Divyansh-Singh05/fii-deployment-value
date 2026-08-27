"""Factorial HMM core (Ghahramani & Jordan, 1997) — exact EM via a
constrained product-space GaussianHMM.

Two independent latent Markov chains emit jointly:

  chain D (direction),     K_D states, transition A_D, start pi_D
  chain C (concentration), K_C states, transition A_C, start pi_C

  x_t | (d_t, c_t)  ~  N( mu_D[d_t] + mu_C[c_t],  diag(sigma2) )

Because K_D*K_C is small (9 here), inference is EXACT on the
equivalent product HMM with S = K_D*K_C states, product index
s = d*K_C + c, transition A = kron(A_D, A_C) and means
M[(d,c)] = mu_D[d] + mu_C[c].  The estimator subclasses
hmmlearn.GaussianHMM (the repo's standard HMM library, so the E-step
is the same certified machinery as the naive backbone) and overrides
ONLY the M-step, re-imposing the factorial structure each iteration:

  * A_D / A_C   <- row-normalized chain-marginal pairwise posteriors
  * mu_D / mu_C <- weighted least squares on the additive-means design
                   (identified by centering mu_C at zero occupancy-
                   weighted mean; the global level lives in mu_D)
  * sigma2      <- shared diagonal residual variance (GJ 1997)

Why this model here: the flat HMM cannot form concentration states
because the persistent direction axis captures the likelihood mass
(Module 2 finding).  The factorial structure gives concentration its
OWN transition matrix and its OWN additive contribution to the
emission mean, so the direction chain cannot absorb it.  Whether the
concentration chain then earns non-degenerate states is exactly the
experiment (gate G3 in module17a).

`numpy_product_loglik` is an INDEPENDENT, pure-numpy forward pass used
as the implementation cross-check (exactness gate G2): hand-rolled
algebra as a cross-check only, per the repo's design philosophy.
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import ConvergenceMonitor

_VAR_FLOOR = 1e-4
_TRANS_FLOOR = 1e-8
_EPS = 1e-300


class _FullHistoryMonitor(ConvergenceMonitor):
    """ConvergenceMonitor that keeps the WHOLE loglik path (the stock
    monitor keeps only the last two values) — needed for gate G1
    (EM monotonicity)."""

    def __init__(self, tol, n_iter, verbose=False):
        super().__init__(tol, n_iter, verbose)
        self.full_history: list[float] = []

    def report(self, log_prob: float) -> None:
        self.full_history.append(float(log_prob))
        super().report(log_prob)


class FactorialGaussianHMM(GaussianHMM):
    """GaussianHMM on the product space with a factorial M-step."""

    def __init__(self, k_d: int = 3, k_c: int = 3, n_iter: int = 200,
                 tol: float = 1e-4, random_state: int = 0):
        super().__init__(n_components=k_d * k_c, covariance_type="diag",
                         n_iter=n_iter, tol=tol,
                         random_state=random_state,
                         init_params="", params="stmc")
        self.k_d, self.k_c = k_d, k_c
        self.monitor_ = _FullHistoryMonitor(tol, n_iter)
        # chain-level parameters (kept in sync with product arrays)
        self.a_d = self.a_c = None
        self.pi_d = self.pi_c = None
        self.mu_d = self.mu_c = None
        self.sigma2 = None

    # ---- chain -> product sync -----------------------------------------
    def _sync_product(self) -> None:
        self.startprob_ = np.kron(self.pi_d, self.pi_c)
        self.transmat_ = np.kron(self.a_d, self.a_c)
        self.means_ = (self.mu_d[:, None, :]
                       + self.mu_c[None, :, :]).reshape(
                           self.n_components, -1)
        self._covars_ = np.tile(np.maximum(self.sigma2, _VAR_FLOOR),
                                (self.n_components, 1))

    # ---- quantile-seeded initialization (EM may move away freely) ------
    def seed_init(self, x: np.ndarray, dir_dims: list[int],
                  conc_dims: list[int], jitter_seed: int) -> None:
        rng = np.random.default_rng(jitter_seed)
        f = x.shape[1]
        gmean = x.mean(axis=0)
        self.mu_d = np.tile(gmean, (self.k_d, 1))
        self.mu_c = np.zeros((self.k_c, f))
        dkey = x[:, dir_dims].mean(axis=1)
        ckey = x[:, conc_dims].mean(axis=1)
        qs = np.quantile(dkey, np.linspace(0, 1, self.k_d + 1))
        for j in range(self.k_d):
            sel = (dkey >= qs[j]) & (dkey <= qs[j + 1])
            self.mu_d[j] = x[sel].mean(axis=0) if sel.sum() >= 10 else gmean
        qs = np.quantile(ckey, np.linspace(0, 1, self.k_c + 1))
        for j in range(self.k_c):
            sel = (ckey >= qs[j]) & (ckey <= qs[j + 1])
            dev = (x[sel].mean(axis=0) - gmean) if sel.sum() >= 10 else 0.0
            self.mu_c[j, conc_dims] = np.asarray(dev)[conc_dims] \
                if sel.sum() >= 10 else 0.0
        self.mu_c -= self.mu_c.mean(axis=0, keepdims=True)
        jit = 0.05 * x.std(axis=0)
        self.mu_d += rng.normal(0, 1, self.mu_d.shape) * jit
        self.mu_c += rng.normal(0, 1, self.mu_c.shape) * jit
        self.sigma2 = np.maximum(x.var(axis=0), _VAR_FLOOR)
        for attr, k in (("a_d", self.k_d), ("a_c", self.k_c)):
            a = np.full((k, k), 0.10 / (k - 1))
            np.fill_diagonal(a, 0.90)
            setattr(self, attr, a)
        self.pi_d = np.full(self.k_d, 1.0 / self.k_d)
        self.pi_c = np.full(self.k_c, 1.0 / self.k_c)
        self._sync_product()

    # ---- constrained M-step ---------------------------------------------
    def _do_mstep(self, stats: dict) -> None:
        kd, kc = self.k_d, self.k_c
        # transitions: chain-marginal expected pairwise counts
        xi4 = stats["trans"].reshape(kd, kc, kd, kc)
        a_d = xi4.sum(axis=(1, 3)) + _TRANS_FLOOR
        a_c = xi4.sum(axis=(0, 2)) + _TRANS_FLOOR
        self.a_d = a_d / a_d.sum(axis=1, keepdims=True)
        self.a_c = a_c / a_c.sum(axis=1, keepdims=True)
        # start probabilities
        g14 = stats["start"].reshape(kd, kc)
        self.pi_d = g14.sum(axis=1) + _EPS
        self.pi_d /= self.pi_d.sum()
        self.pi_c = g14.sum(axis=0) + _EPS
        self.pi_c /= self.pi_c.sum()
        # additive means: (kd+kc) x (kd+kc) normal equations, F rhs
        w_s = stats["post"]                       # (S,)
        b_sf = stats["obs"]                       # (S, F)
        f = b_sf.shape[1]
        w4 = w_s.reshape(kd, kc)
        nmat = np.zeros((kd + kc, kd + kc))
        nmat[:kd, :kd] = np.diag(w4.sum(axis=1))
        nmat[kd:, kd:] = np.diag(w4.sum(axis=0))
        nmat[:kd, kd:] = w4
        nmat[kd:, :kd] = w4.T
        b4 = b_sf.reshape(kd, kc, f)
        rhs = np.vstack([b4.sum(axis=1), b4.sum(axis=0)])
        sol, *_ = np.linalg.lstsq(nmat, rhs, rcond=None)
        mu_d, mu_c = sol[:kd], sol[kd:]
        wc = w4.sum(axis=0)
        shift = (wc @ mu_c) / (wc.sum() + _EPS)
        self.mu_c = mu_c - shift
        self.mu_d = mu_d + shift
        # shared diagonal covariance
        m = (self.mu_d[:, None, :] + self.mu_c[None, :, :]).reshape(
            self.n_components, -1)
        obs2 = stats["obs**2"]                    # (S, F)
        t_tot = w_s.sum()
        self.sigma2 = np.maximum(
            (obs2.sum(axis=0) - 2 * (m * b_sf).sum(axis=0)
             + (w_s[:, None] * m ** 2).sum(axis=0)) / t_tot,
            _VAR_FLOOR)
        self._sync_product()

    # ---- serializable frozen parameters ----------------------------------
    def chain_params(self) -> dict:
        return {"k_d": self.k_d, "k_c": self.k_c,
                "mu_d": self.mu_d.tolist(), "mu_c": self.mu_c.tolist(),
                "sigma2": self.sigma2.tolist(),
                "a_d": self.a_d.tolist(), "a_c": self.a_c.tolist(),
                "pi_d": self.pi_d.tolist(), "pi_c": self.pi_c.tolist()}

    @classmethod
    def from_chain_params(cls, p: dict) -> "FactorialGaussianHMM":
        m = cls(k_d=p["k_d"], k_c=p["k_c"])
        m.mu_d = np.array(p["mu_d"]); m.mu_c = np.array(p["mu_c"])
        m.sigma2 = np.array(p["sigma2"])
        m.a_d = np.array(p["a_d"]); m.a_c = np.array(p["a_c"])
        m.pi_d = np.array(p["pi_d"]); m.pi_c = np.array(p["pi_c"])
        m.n_features = np.array(p["mu_d"]).shape[1]
        m._sync_product()
        return m


def numpy_product_loglik(startprob, transmat, means, covars_diag,
                         x, lengths) -> float:
    """Independent scaled-forward log-likelihood (pure numpy).
    Cross-checks the hmmlearn-based estimator: gate G2."""
    log2pi = np.log(2.0 * np.pi)
    inv = 1.0 / covars_diag                       # (S, F)
    cst = -0.5 * (np.log(covars_diag).sum(axis=1)
                  + covars_diag.shape[1] * log2pi)  # (S,)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    ll = 0.0
    for i in range(len(lengths)):
        seg = x[bounds[i]:bounds[i + 1]]
        quad = ((seg ** 2) @ inv.T - 2.0 * seg @ (means * inv).T
                + ((means ** 2) * inv).sum(axis=1))
        logb = cst - 0.5 * quad                   # (T, S)
        shift = logb.max(axis=1)
        b = np.exp(logb - shift[:, None])
        alpha = startprob * b[0]
        s0 = alpha.sum() + _EPS
        alpha /= s0
        ll_i = np.log(s0) + shift[0]
        for t in range(1, seg.shape[0]):
            alpha = (alpha @ transmat) * b[t]
            st = alpha.sum() + _EPS
            alpha /= st
            ll_i += np.log(st) + shift[t]
        ll += ll_i
    return float(ll)
