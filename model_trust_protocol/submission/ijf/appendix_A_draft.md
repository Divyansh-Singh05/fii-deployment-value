# APPENDIX A. SPECIFICATION OF THE LATENT-STATE AND HAZARD MODELS

*Draft, pending confirmation that IJF excludes appendices from the 6,000-word
body count. Every figure below is read from the frozen parameter file
`hmm_backbone_params.json` or from the producing module, not restated from
project documents.*

## A.1 The regime backbone

**Observation vector.** Four stock-day features, each a within-day
cross-sectional rank mapped through the inverse normal, so every feature is
standardised across instruments on the day it is measured:

- `F_persist` — signed persistence of the day's net flow direction
- `F_block` — mean trade size relative to the instrument's trailing base
- `F_entity_s` — participant concentration on the sell side, 5-day smoothed
- `F_entity_buy_s` — the same on the buy side

**Model.** A three-state Gaussian hidden Markov model with diagonal covariance.
For state *s* the emission is *N*(μ_s, diag σ²_s), and the latent chain is
first-order with transition matrix *A* and initial distribution π.

**Estimation.** Fitted by Expectation-Maximisation on the training era only
(observations through 30 April 2021), tolerance 1e-4, maximum 200 iterations.
To avoid a recency-biased sample, each instrument contributes one randomly
located contiguous block of at most 400 days rather than its most recent 400.
Five restarts are run at seeds 42-46 and the highest-likelihood solution is
retained; the selected fit has log-likelihood −870,760.

**Label switching.** States are ordered by the fitted mean of `F_persist`,
lowest to highest, and named SELL_REGIME, NEUTRAL, BUY_REGIME. The ordering is
a deterministic function of the fitted parameters, so labels are reproducible
across restarts.

**Frozen parameters.** The fitted quantities, applied unchanged to both eras:

Transition matrix *A* (rows = state at *t*, columns = state at *t*+1):

| from \ to | BUY | SELL | NEUTRAL | implied dwell |
|---|---:|---:|---:|---:|
| BUY | 0.9537 | 0.0378 | 0.0084 | 21.6 d |
| SELL | 0.0404 | 0.9521 | 0.0075 | 20.9 d |
| NEUTRAL | 0.0113 | 0.0109 | 0.9778 | 45.0 d |

Emission means μ_s, in probit units:

| state | F_persist | F_block | F_entity_s | F_entity_buy_s |
|---|---:|---:|---:|---:|
| BUY | +0.928 | −0.054 | +0.776 | +0.401 |
| SELL | −0.854 | −0.049 | +0.405 | +0.781 |
| NEUTRAL | −0.012 | +0.105 | −0.778 | −0.821 |

Emission variances σ²_s:

| state | F_persist | F_block | F_entity_s | F_entity_buy_s |
|---|---:|---:|---:|---:|
| BUY | 0.366 | 1.188 | 0.525 | 0.526 |
| SELL | 0.388 | 1.168 | 0.556 | 0.529 |
| NEUTRAL | 0.358 | 0.554 | 0.407 | 0.406 |

Initial distribution π = (0.473, 0.345, 0.182) over (BUY, SELL, NEUTRAL).

**Real-time decoding.** The state reported for day *t* is the filtered
posterior P(S_t | x_1..t), computed by the forward recursion alone under the
frozen parameters. No smoothing pass is run and no parameter is re-estimated,
so no label depends on an observation after *t*. Filtered labels agree with a
full-sequence Viterbi decode on 91.3% of stock-days; where they differ, the
filtered label reaches a new regime a median of one day later. A
pre-registered invariance check re-estimated the episode-level results on the
filtered labels and confirmed both era-wise coefficients within tolerance.

**Episodes.** An episode is a maximal run of identical labels for one
instrument. Runs are broken at calendar gaps exceeding 21 days, since a
suspension or a masked month cannot weld two episodes. An episode's end is
knowable only at the close of the first day after the run concludes, and every
downstream use respects that timestamp.

## A.2 The episode-end hazard layer

**Target.** Standing at the close of day *t* inside an episode, the model
predicts 1[the episode's last day falls within the next *k* days], for
*k* ∈ {1, 3, 5}. At *k* = 1 the event is that today was the last day.

**Features**, all measured on information through *t*: episode age; the four
state features above; the filtered posterior `p_sell`; concentration drift
against the episode start and against three days back; relative volume; and
log turnover.

**Estimator.** Gradient-boosted trees (LightGBM), 200 estimators, learning
rate 0.05, 31 leaves, minimum 50 samples per leaf, seed 7. Hyperparameters are
fixed across all folds and are not tuned on any evaluation fold.

**Walk-forward protocol.** Evaluation years run 2014 through 2024 plus a
partial 2025, the data ending 28 March 2025. For each evaluation year the
training set is every episode-day dated at least ten days before that year
begins, so the window expands and never overlaps the fold being scored. The
model is refit once per fold.

**Censoring.** Runs whose end is unobserved at the instrument's last labelled
day are right-censored and contribute no targets they cannot label.

**Baselines.** An age-only discrete hazard estimated by Kaplan-Meier on the
training window, with ages capped at 15 and over, composed to a *k*-day
probability; and a constant predictor at the training base rate.

**Pre-registered bar.** The model must beat the age-only baseline on pooled
out-of-window log loss with a paired daily *t* of at least 2, for both *k* = 1
and *k* = 3. Results at *k* = 5 are reported but do not gate.
