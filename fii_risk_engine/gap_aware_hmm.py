"""
GAP-AWARE GAUSSIAN HMM  —  audit item 3

A Gaussian HMM whose transition matrix is expressed PER TRADING DAY, and whose
likelihood accounts for the elapsed time between consecutive observations: two
observations separated by k trading days are linked by A^k, not by A.

WHY THIS EXISTS.  The pre-audit chain estimated A with one transition per
observed FII row (hmmlearn has no notion of elapsed time) and then deployed it
in C2 as though it were a per-trading-day matrix, propagating belief with A^k.
Those are different units. Measured on the existing state labels:

    transitions used              mean diag   lambda2   half-life
    pooled (what the fit saw)       0.9491     0.9416     11.5 d
    k = 1 only (a true daily A)     0.9648     0.9665     20.4 d

a 77% error in mixing half-life, concentrated in thin stocks: per-stock gap
rate has median 22.4% and p90 42.7%, and correlates -0.54 with observation
count. 14.65% of consecutive observations span k > 1.

The fix is not to change C2 to match C1 -- that would keep a per-observation
matrix and lose the gap handling that is genuinely correct. It is to estimate
the parameters under the SAME likelihood the filter uses.

THE MODEL.  Observations x_1..x_T for one stock, with elapsed trading-day gaps
k_2..k_T between them (k_t >= 1). Latent S_t in {1..K}.

    P(S_t = j | S_{t-1} = i)  =  (A^{k_t})_{ij}
    x_t | S_t = j             ~  N(mu_j, diag(sigma^2_j))

A is the ONE-TRADING-DAY transition matrix -- the quantity C2 needs and the
quantity the pre-audit code never estimated.

THE M-STEP.  means, covariances and startprob keep their closed forms. A does
not: the expected complete-data log-likelihood contribution is

    Q(A)  =  sum_k  sum_{i,j}  Xi_k[i,j] * log (A^k)_{ij}

where Xi_k accumulates posterior transition mass over all pairs separated by k
days. For k = 1 this collapses to the textbook row-normalisation; for k > 1 it
has no closed form, so it is maximised numerically over a softmax
parametrisation of the rows (K = 3, so 9 free parameters -- cheap). EM
monotonicity is asserted every iteration rather than assumed.

PERFORMANCE.  The recursion is sequential in t and cannot be parallelised
along it, but the ~860 stocks in a vintage are independent, so `_e_step` steps
all of them together and the Python loop runs once per LONGEST SEQUENCE (400)
instead of once per observation (231,711). `_e_step_loop` is kept as the
reference definition and gate G5 holds the two against each other. Measured on
the largest vintage: 379s of E-step per vintage became 21s, and the whole
vintage 647s -> 29s. The remaining cost is real array traffic, not interpreter
overhead, which is why a GPU does not help here -- each of the 800 sequential
steps does one 3x3 contraction per stock, far below the arithmetic intensity a
kernel dispatch pays for.

Gates (see validate_gap_aware_hmm.py)
    G1  degenerate    all gaps = 1 reproduces hmmlearn's GaussianHMM
    G2  likelihood    scaled forward-backward matches a direct computation
    G3  monotone      every EM iteration increases the observed log-likelihood
    G4  recovery      on synthetic data with known daily A and irregular
                      sampling, this estimator recovers A while a
                      per-observation fit does not
    G5  batched       the vectorised E-step equals the reference loop to
                      floating-point noise under arbitrary parameters
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["GapAwareGaussianHMM", "trading_day_gaps"]

MIN_COVAR = 1e-3
KCAP = 250          # A^k beyond this is stationary to <1e-8; matches C2's cap


class _Monitor:
    """Minimal stand-in for hmmlearn's ConvergenceMonitor (see fit)."""

    __slots__ = ("iter", "converged", "history")

    def __init__(self, iter_, converged, history):
        self.iter = iter_
        self.converged = converged
        self.history = history


def trading_day_gaps(day_index: np.ndarray, lengths: list[int]) -> np.ndarray:
    """Elapsed trading days between consecutive rows, 0 at each sequence start.

    `day_index` is the position of each observation on the trading calendar,
    so a stock that does not trade for a week has a gap of 5, not 7.
    """
    g = np.zeros(len(day_index), dtype=np.int64)
    pos = 0
    for n in lengths:
        if n > 1:
            g[pos + 1:pos + n] = np.diff(day_index[pos:pos + n])
        pos += n
    if (g < 0).any():
        raise ValueError("day_index must be non-decreasing within a sequence")
    return np.clip(g, 0, KCAP)


class GapAwareGaussianHMM:
    def __init__(self, n_components=3, n_iter=200, tol=1e-4, random_state=0,
                 verbose=False):
        self.K = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------- helpers
    def _log_emissions(self, X):
        """(N, K) log N(x | mu_j, diag(var_j))."""
        var = self.covars_
        d = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (np.sum(d * d / var[None, :, :]
                              + np.log(2 * np.pi * var)[None, :, :], axis=2))

    def _powers(self, A, kmax):
        """[I, A, A^2, ... A^kmax].

        Written into a preallocated array rather than appended to a list and
        stacked: the multiply order is unchanged (P[i] @ A, ascending), so the
        result is bit-identical, but the per-call list build and the
        `np.array` stack of ~250 small arrays go away. L-BFGS-B's numerical
        gradient calls this ~25,000 times per vintage, which is what makes it
        worth caring about.
        """
        P = np.empty((kmax + 1, self.K, self.K))
        P[0] = np.eye(self.K)
        for i in range(kmax):
            P[i + 1] = P[i] @ A
        return P

    # ------------------------------------------------------- forward-backward
    def _e_step_loop(self, X, lengths, gaps, A, Apow):
        """Reference implementation: one Python step per observation.

        Retained verbatim as the definition of the E-step. `_e_step` below is
        an algebraically identical rewrite that steps every stock forward
        together; `validate_gap_aware_hmm.py` gate G5 holds the two against
        each other on real vintage data.
        """
        logB = self._log_emissions(X)
        mx = logB.max(axis=1)
        B = np.exp(logB - mx[:, None])

        N, K = B.shape
        alpha = np.empty((N, K))
        beta = np.empty((N, K))
        cs = np.empty(N)
        gamma = np.empty((N, K))
        kmax = int(gaps.max()) if len(gaps) else 1
        Xi = np.zeros((kmax + 1, K, K))
        start_mass = np.zeros(K)

        pos = 0
        ll = 0.0
        for n in lengths:
            s, e = pos, pos + n
            b = B[s:e]
            g = gaps[s:e]

            # ---- forward
            a0 = self.startprob_ * b[0]
            c0 = a0.sum()
            alpha[s] = a0 / c0
            cs[s] = c0
            for t in range(1, n):
                pred = alpha[s + t - 1] @ Apow[g[t]]
                at = pred * b[t]
                ct = at.sum()
                alpha[s + t] = at / ct
                cs[s + t] = ct

            # ---- backward
            beta[e - 1] = 1.0
            for t in range(n - 2, -1, -1):
                nxt = Apow[g[t + 1]] @ (b[t + 1] * beta[s + t + 1])
                beta[s + t] = nxt / cs[s + t + 1]

            # ---- posteriors
            gm = alpha[s:e] * beta[s:e]
            gm /= gm.sum(axis=1, keepdims=True)
            gamma[s:e] = gm
            start_mass += gm[0]

            for t in range(1, n):
                k = g[t]
                x = (alpha[s + t - 1][:, None] * Apow[k]
                     * (b[t] * beta[s + t])[None, :]) / cs[s + t]
                Xi[k] += x

            ll += np.log(cs[s:e]).sum() + mx[s:e].sum()
            pos = e

        return gamma, Xi, start_mass, ll

    # --------------------------------------------- forward-backward (batched)
    @staticmethod
    def _pad_layout(lengths):
        """Row-major (N,) <-> padded (S, T) layout for a set of sequences.

        The recursion is sequential in t and cannot be parallelised along it,
        but the sequences are independent, so all S of them can take step t
        together. That turns the Python loop from one iteration per
        OBSERVATION (231,711 on the largest vintage) into one per LONGEST
        SEQUENCE (400), each doing a batched 3x3 contraction. Everything the
        padding touches is made a no-op: emissions pad to 1, gaps pad to 0 so
        A^0 = I, and the scaling factors pad to 1 so log c = 0. Belief
        therefore freezes past the end of a sequence and contributes nothing
        to the likelihood, the posteriors or Xi.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        S = len(lengths)
        T = int(lengths.max())
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        valid = np.arange(T)[None, :] < lengths[:, None]
        flat = np.where(valid, starts[:, None] + np.arange(T)[None, :], 0)
        return S, T, valid, flat

    def _e_step(self, X, lengths, gaps, A, Apow, layout=None):
        K = self.K
        logB = self._log_emissions(X)
        mx = logB.max(axis=1)
        Bflat = np.exp(logB - mx[:, None])

        S, T, valid, flat = layout or self._pad_layout(lengths)
        vf = valid[:, :, None]
        B = np.where(vf, Bflat[flat], 1.0)
        G = np.where(valid, np.asarray(gaps, np.int64)[flat], 0)
        MX = np.where(valid, mx[flat], 0.0)

        alpha = np.empty((S, T, K))
        beta = np.empty((S, T, K))
        cs = np.ones((S, T))

        a0 = self.startprob_[None, :] * B[:, 0, :]
        c0 = a0.sum(axis=1)
        alpha[:, 0] = a0 / c0[:, None]
        cs[:, 0] = c0
        for t in range(1, T):
            at = np.einsum("sk,skj->sj", alpha[:, t - 1], Apow[G[:, t]]) * B[:, t]
            ct = at.sum(axis=1)
            alpha[:, t] = at / ct[:, None]
            cs[:, t] = ct

        beta[:, T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.einsum(
                "sij,sj->si", Apow[G[:, t + 1]],
                B[:, t + 1] * beta[:, t + 1]) / cs[:, t + 1][:, None]

        gm = alpha * beta
        gm /= gm.sum(axis=2, keepdims=True)
        gamma = np.empty((len(X), K))
        gamma[flat[valid]] = gm[valid]
        start_mass = gm[:, 0, :].sum(axis=0)

        kmax = int(np.max(gaps)) if len(gaps) else 1
        Xi = np.zeros((kmax + 1, K, K))
        if T > 1:
            # Xi[k]_ij = sum over the steps with gap k of
            #     alpha_{t-1}[i] * (A^k)_ij * (b_t[j] beta_t[j]) / c_t
            # A^k does not depend on the sample, so it factors out of the sum
            # and each gap's contribution is one (n_k, K)^T @ (n_k, K) product.
            # Materialising the full (S, T, K, K) numerator instead costs K
            # times the memory -- 25 MB on the largest vintage, reallocated
            # every EM iteration -- which is what makes the fit swap on an 8 GB
            # machine when several vintages run at once.
            ok = valid[:, 1:].ravel()
            gk = np.where(valid[:, 1:], G[:, 1:], 0).ravel()[ok]
            Lk = (alpha[:, :-1] / cs[:, 1:][:, :, None]).reshape(-1, K)[ok]
            Rk = (B[:, 1:] * beta[:, 1:]).reshape(-1, K)[ok]
            order = np.argsort(gk, kind="stable")
            gs = gk[order]
            bounds = np.searchsorted(gs, np.arange(kmax + 2))
            for k in range(1, kmax + 1):     # k = 0 only ever holds padding
                lo, hi = bounds[k], bounds[k + 1]
                if hi > lo:
                    sl = order[lo:hi]
                    Xi[k] = Apow[k] * (Lk[sl].T @ Rk[sl])

        ll = float((np.log(cs) * valid).sum() + MX.sum())
        return gamma, Xi, start_mass, ll

    # ------------------------------------------------------------- M-step: A
    @staticmethod
    def _softmax_rows(theta, K):
        Z = theta.reshape(K, K)
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _m_step_A(self, Xi, A0):
        """Maximise sum_k sum_ij Xi_k[i,j] log (A^k)_ij over row-stochastic A.

        k = 1 alone has the closed-form row normalisation; anything else does
        not, so the whole thing is optimised numerically. Starting from the
        closed-form k=1 solution keeps the optimiser near the answer.
        """
        K = self.K
        live = [k for k in range(Xi.shape[0]) if Xi[k].sum() > 0]
        if live == [1]:                       # every gap is one day
            A = Xi[1] / np.maximum(Xi[1].sum(axis=1, keepdims=True), 1e-300)
            return A
        kmax = max(live)
        # stack the live slices once: the objective is then a single reduction
        # instead of a 200-plus-iteration Python loop per function evaluation,
        # and L-BFGS-B's numerical gradient calls it ten times per step
        lk = np.asarray(live, dtype=np.int64)
        Xl = Xi[lk]

        def neg_Q(theta):
            A = self._softmax_rows(theta, K)
            P = self._powers(A, kmax)
            return -float(np.sum(Xl * np.log(np.maximum(P[lk], 1e-300))))

        theta0 = np.log(np.maximum(A0, 1e-8)).ravel()
        r = minimize(neg_Q, theta0, method="L-BFGS-B",
                     options={"maxiter": 300, "ftol": 1e-12})
        # The optimiser's answer was previously taken on trust. L-BFGS-B can
        # return a non-converged iterate, or one WORSE than where it started,
        # and accepting that silently breaks EM monotonicity and publishes a
        # transition matrix nothing ever validated. Accept only a converged
        # step that actually improved the objective; otherwise keep the
        # incoming A, which leaves Q unchanged and so keeps EM monotone.
        q0, q1 = neg_Q(theta0), neg_Q(r.x)
        if (not r.success) or (not np.isfinite(q1)) or (q1 > q0 + 1e-8):
            self.mstep_failed_ = getattr(self, "mstep_failed_", 0) + 1
            return A0
        return self._softmax_rows(r.x, K)

    # ------------------------------------------------------------------- fit
    def fit(self, X, lengths, gaps, init_means=None, init_transmat=None):
        X = np.asarray(X, dtype=float)
        gaps = np.asarray(gaps, dtype=np.int64)
        N, D = X.shape
        K = self.K
        rng = np.random.default_rng(self.random_state)

        if init_means is not None:
            self.means_ = np.asarray(init_means, dtype=float).copy()
        else:
            # k-means-ish init on the first feature, matching the chain's
            # convention that state order is increasing in F_persist
            q = np.quantile(X[:, 0], np.linspace(0.1, 0.9, K))
            self.means_ = np.array(
                [X[np.argsort(np.abs(X[:, 0] - qq))[:max(N // K, 1)]].mean(axis=0)
                 for qq in q])
            self.means_ += 1e-6 * rng.standard_normal(self.means_.shape)
        self.covars_ = np.tile(X.var(axis=0), (K, 1)) + MIN_COVAR
        self.startprob_ = np.full(K, 1.0 / K)
        if init_transmat is not None:
            self.transmat_ = np.asarray(init_transmat, dtype=float).copy()
        else:
            A = np.full((K, K), 0.06 / (K - 1))
            np.fill_diagonal(A, 0.94)
            self.transmat_ = A

        prev, self.history_ = -np.inf, []
        self.mstep_failed_ = 0
        self.monotone_ok_ = True
        layout = self._pad_layout(lengths)      # depends on lengths alone
        for it in range(self.n_iter):
            Apow = self._powers(self.transmat_, int(max(gaps.max(), 1)))
            gamma, Xi, start_mass, ll = self._e_step(
                X, lengths, gaps, self.transmat_, Apow, layout)
            self.history_.append(ll)

            if ll < prev - 1e-6:              # G3
                self.monotone_ok_ = False
            # ABSOLUTE improvement, matching hmmlearn's `tol` semantics and
            # therefore C1's existing TOL=1e-2. A relative test here would be
            # ~7,000 log-likelihood units wide on this panel and would stop
            # after about four iterations.
            if ll - prev < self.tol:
                prev = ll
                break
            prev = ll

            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            d2 = (X[:, None, :] - self.means_[None, :, :]) ** 2
            self.covars_ = np.maximum(
                np.einsum("nk,nkd->kd", gamma, d2) / w[:, None], MIN_COVAR)
            self.startprob_ = start_mass / start_mass.sum()
            self.transmat_ = self._m_step_A(Xi, self.transmat_)

            if self.verbose and it % 10 == 0:
                print(f"    iter {it:3d}  loglik {ll:,.2f}")

        self.n_iter_ = len(self.history_)
        self.loglik_ = prev
        self.monotone_ok_ = getattr(self, "monotone_ok_", True)
        self.converged_ = self.n_iter_ < self.n_iter
        # A single flag the caller can gate on. Previously `monotone_ok_` was
        # recorded and then ignored by every consumer, so a fit whose
        # likelihood went backwards, or which ran out of iterations, was
        # indistinguishable from a clean one.
        self.ok_ = bool(self.monotone_ok_ and self.converged_
                        and self.mstep_failed_ == 0)
        # hmmlearn API compatibility: C1 reads m.monitor_.iter, and keeping the
        # shape of the object it replaces means the harness needs no changes
        # beyond passing gaps.
        self.monitor_ = _Monitor(self.n_iter_,
                                 self.n_iter_ < self.n_iter,
                                 list(self.history_))
        self._order_states()
        return self

    def _order_states(self):
        """Sort states by mean of feature 0, as the rest of the chain assumes."""
        o = np.argsort(self.means_[:, 0])
        self.means_ = self.means_[o]
        self.covars_ = self.covars_[o]
        self.startprob_ = self.startprob_[o]
        self.transmat_ = self.transmat_[np.ix_(o, o)]

    # -------------------------------------------------------------- inference
    def score(self, X, lengths, gaps):
        Apow = self._powers(self.transmat_, int(max(np.max(gaps), 1)))
        return self._e_step(np.asarray(X, float), lengths,
                            np.asarray(gaps, np.int64), self.transmat_, Apow)[3]

    def predict_proba(self, X, lengths, gaps):
        """Smoothed posteriors P(S_t | all observations of the sequence)."""
        Apow = self._powers(self.transmat_, int(max(np.max(gaps), 1)))
        return self._e_step(np.asarray(X, float), lengths,
                            np.asarray(gaps, np.int64), self.transmat_, Apow)[0]

    def predict(self, X, lengths, gaps):
        """Gap-aware Viterbi path.

        Retained only because C1 reports a Viterbi threshold variant for
        comparison; production thresholds and C2 both use the filtered
        posterior. Transitions use log(A^k), matching the fitted likelihood.
        """
        X = np.asarray(X, float)
        gaps = np.asarray(gaps, np.int64)
        Apow = self._powers(self.transmat_, int(max(gaps.max(), 1)))
        logA = np.log(np.maximum(Apow, 1e-300))
        logB = self._log_emissions(X)
        out = np.empty(len(X), dtype=int)
        pos = 0
        for n in lengths:
            s, e = pos, pos + n
            delta = np.log(np.maximum(self.startprob_, 1e-300)) + logB[s]
            psi = np.zeros((n, self.K), dtype=int)
            for t in range(1, n):
                sc = delta[:, None] + logA[gaps[s + t]]
                psi[t] = sc.argmax(axis=0)
                delta = sc.max(axis=0) + logB[s + t]
            out[e - 1] = int(delta.argmax())
            for t in range(n - 1, 0, -1):
                out[s + t - 1] = psi[t][out[s + t]]
            pos = e
        return out

    def filter_proba(self, X, lengths, gaps):
        """CAUSAL posteriors P(S_t | x_1..t) — what C2 needs.

        Same recursion as the forward pass, so the filter and the estimator
        agree by construction rather than by convention.
        """
        X = np.asarray(X, float)
        gaps = np.asarray(gaps, np.int64)
        Apow = self._powers(self.transmat_, int(max(gaps.max(), 1)))
        logB = self._log_emissions(X)
        B = np.exp(logB - logB.max(axis=1)[:, None])
        out = np.empty_like(B)
        pos = 0
        for n in lengths:
            s, e = pos, pos + n
            a = self.startprob_ * B[s]
            out[s] = a / a.sum()
            for t in range(1, n):
                a = (out[s + t - 1] @ Apow[gaps[s + t]]) * B[s + t]
                out[s + t] = a / a.sum()
            pos = e
        return out
