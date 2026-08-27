# Models

Phase: `model`. Stages: `hmm_overlay_design`, `hmm_train_oos`,
`threshold_calibration`, `model_descriptives`. Interface:
`src/fii/models/` (`base.py`, `registry.py`, `_template.py`).

## 1. The main model: hybrid HMM regime detector

### 1.1 Hidden Markov model — mathematical foundations

A hidden Markov model assumes an unobserved state sequence
$S_1, \dots, S_T \in \{1,\dots,K\}$ evolving as a first-order Markov chain,

$$P(S_t = j \mid S_{t-1} = i) = A_{ij}, \qquad P(S_1 = i) = \pi_i,$$

with observations (here: a stock's daily feature vector
$\mathbf{x}_t \in \mathbb{R}^d$) emitted conditionally on the state:

$$\mathbf{x}_t \mid S_t = j \;\sim\; \mathcal{N}(\boldsymbol{\mu}_j, \Sigma_j),
\qquad \Sigma_j \ \text{diagonal}.$$

Diagonal covariance is a deliberate restriction: rare states (the
concentration corners) cannot support full covariance estimation.

The likelihood marginalizes over all state paths via the forward recursion.
With $\alpha_t(j) = P(\mathbf{x}_{1:t}, S_t = j)$:

$$\alpha_1(j) = \pi_j\, b_j(\mathbf{x}_1), \qquad
\alpha_{t+1}(j) = b_j(\mathbf{x}_{t+1}) \sum_{i=1}^K \alpha_t(i)\, A_{ij},$$

where $b_j(\cdot)$ is the state-$j$ Gaussian density, and
$P(\mathbf{x}_{1:T}) = \sum_j \alpha_T(j)$.

**Estimation (Baum–Welch / EM).** With the backward variables
$\beta_t(i) = P(\mathbf{x}_{t+1:T} \mid S_t = i)$, the E-step computes state
and transition posteriors

$$\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{\sum_k \alpha_t(k)\beta_t(k)},
\qquad
\xi_t(i,j) = \frac{\alpha_t(i) A_{ij} b_j(\mathbf{x}_{t+1}) \beta_{t+1}(j)}
                  {\sum_{k,l}\alpha_t(k) A_{kl} b_l(\mathbf{x}_{t+1}) \beta_{t+1}(l)},$$

and the M-step re-estimates
$\hat A_{ij} = \sum_t \xi_t(i,j) / \sum_t \gamma_t(i)$ and the Gaussian
moments as $\gamma$-weighted averages. EM increases the likelihood
monotonically to a local optimum (hence seeded, fixed-init fits).

**Decoding (Viterbi).** The reported state sequence maximizes the joint path
probability via dynamic programming:

$$\delta_{t+1}(j) = b_j(\mathbf{x}_{t+1}) \max_i \delta_t(i) A_{ij},$$

with backtracking from $\arg\max_j \delta_T(j)$.

### 1.2 Why the model is hybrid (a negative result that shaped design)

Module 2 established that **no HMM with $k = 3\ldots6$ ever forms a
persistent-sell + dispersed state**: state formation requires temporal
coherence, and the persistence feature (20-day window, autocorrelation
≈ 0.93) dominates the likelihood; single-day concentration snapshots are
HMM-invisible. The regime space is two-dimensional — flow direction ×
participant concentration — with rare corners that never earn states.

**Final architecture**: a $k=3$ directional backbone (SELL / NEUTRAL / BUY,
dwell ~13–17 days) fitted by the HMM, plus **overlay rules** on frozen
thresholds for the concentration archetypes:

| Archetype | Rule |
|---|---|
| HOSTAGE | SELL backbone ∧ $F^{entity}_s < -0.513$ (dispersed) |
| SHARK_DIST | SELL backbone ∧ $F^{entity}_s > +0.877$ (concentrated) |
| SHARK_ACC | BUY backbone ∧ $F^{entity}_{buy} > +0.795$ |
| ROBOT | NEUTRAL backbone |

### 1.3 Threshold calibration with built-in falsification

Thresholds come from the TRAIN era only. The first-choice method (GMM on the
concentration marginal) **failed its own pre-registered stability test**
(49/51 coin-flip component assignment, boundary drift 0.665) and was
discarded; a quantile rule was adopted and confirmed by three-way convergence
(k-means −0.441, test-era GMM −0.510, train-q25 −0.513). The failure is
retained in the log as a methods exhibit.

### 1.4 Out-of-sample protocol

Everything frozen at 2021-04-30; May–June 2021 masked; test = 2021-07-01
onward decoded with frozen parameters. Backbone signatures, census shares and
transition matrices replicate near-exactly OOS (effect-size drift ≤ 0.15).

## 2. LightGBM challenger

Gradient-boosted trees minimize, over additive trees $f_m$,

$$\mathcal{L} = \sum_i \ell\big(y_i,\ \hat y_i^{(m-1)} + f_m(\mathbf{x}_i)\big) + \Omega(f_m),$$

with second-order (gradient/hessian) approximation at each boost; LightGBM
grows leaves best-first with histogram binning. Target: forward 20-day
abnormal CAR; features: all 10 probit features; attribution: SHAP values via
`pred_contrib` (exact for trees).

**Role**: the pre-registered "is the HMM the bottleneck?" test — full detail
in `04_validation_framework.md`. Verdict: dynamic increment below the
pre-registered bar (non-overlap spread t = 1.48 < 2); regimes stand.

## 3. Adding a new model

Copy `src/fii/models/_template.py`, set `name`, implement
`train/predict/evaluate/save_outputs`. The registry auto-discovers it;
`python pipeline.py --model <name>` runs it. The contract (frozen split,
feature-store-only inputs, (cisin, TR_DATE)-keyed outputs, pre-registered
bars) is stated in the template docstring. **User commitment recorded in the
research log: any successor model receives the same validation battery as
the HMM.**
