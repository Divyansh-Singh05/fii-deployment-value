# 09 · Instrument Families

For every validation question the framework poses, the **family of instruments
the field offers** — not only the ones this programme happened to use.

Three columns matter throughout:
- **Used** — what the source projects deployed.
- **Siblings** — the rest of the family, and when each is preferred.
- **Can it fail?** — the D3 note: the condition under which this instrument is
  structurally incapable of returning a negative, or is being misapplied.

Entries marked **[GAP]** are instruments the framework says should have been
used and were not. They are listed deliberately; they are the paper's own
application of its own method, and they are the package's roadmap.

---

## A · Discrimination and probability quality (L3, L5)

*Used: AUC-ROC, Brier score, log-loss, reliability curves.*

| Instrument | What it measures | When preferred | Can it fail? |
|---|---|---|---|
| **ROC-AUC** *(used: 0.797 vs 0.569 age-only)* | Threshold-free ranking quality | Balanced-ish classes; when the whole ranking matters | **Invariant to any monotone transform of the score — it cannot detect miscalibration at all.** Also invariant to prevalence, which flatters rare-event models. An AUC alone is never a sufficient verdict |
| **PR-AUC / average precision** **[GAP]** | Ranking quality weighted toward the positive class | **Rare events** — episode ends, defaults, crashes. Should have been reported alongside AUC here | Sensitive to prevalence, so not comparable across samples with different base rates |
| **Brier score** *(used: 0.121 vs 0.262)* | Mean squared probability error | A single proper summary | Conflates calibration and discrimination |
| **Murphy decomposition of Brier** **[GAP]** | Splits Brier into *reliability − resolution + uncertainty* | Whenever a Brier score is reported. Separates "well-calibrated but uninformative" from "informative but overconfident" — a distinction the raw score hides | The decomposition itself is binning-dependent; report the binning |
| **Log score** *(used: 0.182 vs 0.532)* | Proper, local scoring rule | When tail probabilities matter | Unbounded — one confident error dominates. Never report alone on small samples |
| **Reliability diagram / ECE / MCE** *(used)* | Calibration across the probability range | Always, alongside any discrimination metric | Aggregate calibration can pass while a region fails — which is exactly what happened here (excellent at 0.99, overconfident at 0.70) |
| **Spiegelhalter's z** **[GAP]** | Formal calibration test | When a yes/no calibration verdict is needed rather than a picture | Low power on small samples |
| **DeLong test** **[GAP]** | Compares two **correlated** AUCs | Whenever two models are ranked by AUC on the same data — which is exactly the comparison made here | Assumes asymptotic normality; bootstrap the difference on small samples |
| **Harrell's C-index / time-dependent ROC** **[GAP]** | Discrimination under **right-censoring** | Directly applicable to the hazard model, where episodes are censored. Plain AUC on censored data is biased | — |
| **Decision Curve Analysis (net benefit)** **[GAP]** | Converts probabilities into decision value across a range of decision thresholds | **The missing bridge between L5 and L6 in this programme.** It is the formal version of the finding that a model with AUC 0.797 lost to a three-line age rule | Requires a stated preference range; state it |
| KS statistic, Gini, lift/gain curves | Credit-risk conventions | Cross-industry comparability | Gini = 2·AUC − 1 exactly; reporting both is reporting one thing twice |

**The discipline the family implies.** Discrimination, calibration, and decision
value are three separate properties, and a model can hold any one without the
others. This programme measured all three and found exactly that dissociation —
strong discrimination, good calibration where the mass lives, no decision value.
Reporting only the first would have supported a claim the other two refute.

---

## B · Out-of-sample protocol and backtest inference (L5, L6)

*Used: frozen split with a masked embargo, expanding-window walk-forward with
yearly refits, immutable dated parameter vintages, paired moving-block bootstrap
ΔSharpe, Diebold–Mariano, cost grid with breakevens.*

| Instrument | What it does | When preferred | Can it fail? |
|---|---|---|---|
| **Frozen train/test split** *(used)* | One clean out-of-sample era | Establishing that structure generalises | Does not reproduce deployment (parameters are frozen where a real system refits) and does not *demonstrate* absence of look-ahead — it assumes it |
| **Walk-forward with dated vintages** *(used)* | Reproduces deployment; every parameter has an `asof` | Any claim about what a system would have done | This is the macro-forecasting **real-time / vintage data** discipline (Croushore–Stark) under another name. Worth citing as such |
| **Purged k-fold with embargo** **[GAP]** | Removes training observations whose labels overlap the test window | **Overlapping labels** — which this programme has everywhere (20-day forward CARs, h-day forward returns) | Requires knowing the label span; purging the wrong span is worse than not purging |
| **Combinatorial purged CV (CPCV)** **[GAP]** | Produces a *distribution* of backtest paths rather than one | When you want a sampling distribution of the performance statistic | Expensive; and it does not fix a biased signal, only measures the variance of the estimate |
| **Probability of Backtest Overfitting (PBO)** **[GAP]** | Probability the selected configuration underperforms the median out of sample | **When any selection happened among strategy variants — which it did here (eight strategies, four engine variants)** | Requires the full set of trials to have been recorded. Its value depends entirely on honest trial accounting |
| **Deflated / haircut Sharpe** **[GAP]** | Adjusts a Sharpe for the number of trials and for non-normality | Any reported Sharpe following a search | Same dependency: it needs the true trial count |
| **White's Reality Check / Hansen's SPA** **[GAP]** | Tests the *best* of many strategies against a benchmark, controlling for data snooping | Directly applicable to the eight-strategy comparison | SPA is more powerful than Reality Check under poor alternatives; prefer it |
| **Romano–Wolf stepdown** **[GAP]** | Multiple testing with dependence, less conservative than Bonferroni | The nine-test pre-registered family here used Bonferroni | — |
| **Model Confidence Set** **[GAP]** | Returns the *set* of models not distinguishable from the best | When several engine variants are compared and one is chosen | Set size grows with noise — a large MCS is itself the finding |
| **Diebold–Mariano** *(used)* | Compares two forecasts' loss differentials | Non-nested competitors | **Invalid for nested models under the null** — the loss differential degenerates. A parametric benchmark nested inside a richer specification needs **Clark–West** or Giacomini–White instead. Worth auditing which comparisons here are nested |
| **Giacomini–White** **[GAP]** | *Conditional* predictive ability | When you want to know *when* one model beats another, not just on average | — |
| **Pesaran–Timmermann** **[GAP]** | Directional accuracy independent of magnitude | When sign, not size, drives the decision | — |
| **Moving-block bootstrap on paired ΔSharpe** *(used)* | Serial-dependence-robust interval on a paired comparison | Strategy comparison | Block length is a judgement; report it and show sensitivity |

**The gap this table exposes.** The programme's out-of-sample *protocol* is
strong — vintages, embargo, truncation audit — while its **selection accounting**
is comparatively thin. Eight strategies, four market-engine variants and several
model classes were tried; the corrections that price exactly that (PBO, deflated
Sharpe, SPA, MCS) were not applied. That is a real gap and belongs in the paper's
own limitations, not hidden.

---

## C · Sequential and state inference (L2, L3, L5)

*Used: Gaussian HMM by Baum–Welch; Viterbi decoding; forward filtering for
causal labels; gap-aware transition propagation over elapsed trading days; a
factorial HMM as structural challenger; frozen-parameter decode across eras.*

| Instrument | What it does | When preferred | Can it fail? |
|---|---|---|---|
| **Forward filtering** `P(S_t \| x_{1:t})` *(used)* | Causal state estimate | Anything feeding a downstream decision | The correct default. Its cost — an onset lag, here a median of one day — must be measured, not assumed |
| **Forward–backward smoothing** *(used in Phase I, then replaced)* | `P(S_t \| x_{1:T})` | Descriptive labelling only | **The state at *t* sees the future.** Any inference built on smoothed labels inherits that, which is why the causal-relabelling gate exists |
| **Viterbi MAP path** *(used)* | Most likely *sequence* | When path coherence matters | Not the same as the sequence of marginal modes, and a sticky transition prior dominates it — the source of the circular-persistence finding |
| **Fixed-lag smoothing** **[GAP]** | Allows exactly *L* days of hindsight | **When the decision itself has a known lag** — matches the information set to the decision instead of choosing between two extremes | Requires the decision lag to be fixed and stated |
| **Hidden semi-Markov model (HSMM)** **[GAP]** | Explicit dwell-time distribution rather than geometric | **Directly applicable here.** The geometric dwell implied by an HMM was a known weakness, and episode *ends* were economically central — a separate hazard model was built to forecast exactly what an HSMM parameterises natively | More parameters; harder to fit; but it would have unified two models into one |
| **Sticky HDP-HMM** **[GAP]** | Bayesian nonparametric; infers the number of states | **When model order must be chosen and the usual criterion cannot discriminate** — precisely the situation the BIC sweep hit at n = 2.7M | Sampling cost; sensitivity to concentration priors |
| **Factorial HMM** *(used as challenger)* | Independent chains emitting additively | When the state space is a product of semi-independent factors | The right challenger, correctly deployed here: it established that the flat-HMM starvation was a property of the class, not the data |
| **Input–output HMM / TVTP** | Transitions depend on covariates | When an exogenous variable should govern persistence rather than level | Was pre-registered here and withdrawn once the motivating deficit proved to be a unit error |
| **Kalman / RTS smoother** | Linear-Gaussian latent state | Continuous latent states | Same filtering/smoothing distinction applies |
| **Particle filter / SMC** | Nonlinear, non-Gaussian | Heavy tails, nonlinear dynamics | Degeneracy; needs resampling diagnostics (ESS — the same effective-sample-size discipline as Q4) |
| **Bayesian online change-point detection; CUSUM; PELT** **[GAP]** | An entirely different family answering the same question | When regimes are better framed as *breaks* than as recurring states | A useful adversary for a regime model: if a change-point detector reproduces the episodes, the state model is a convenience |
| **Bai–Perron / Quandt–Andrews sup-Wald** **[GAP]** | Formal structural-break tests | Establishing that a break exists before masking or splitting on it | The frozen split here masks a period asserted to be a structural break; a formal test would substantiate that |

---

## D · Event studies and causal identification (L4)

*Used: excess CAR as difference-in-differences against a labelled baseline;
START/END anchors; delisting truncation; PanelOLS with unit + date FE; two-way
clustering; specification ladder with a deliberate bad control; continuous
dose–response; non-overlap subsample; horizon ladder; mechanical-date exclusion;
hindsight-advantaged control; PIN as an independent construct.*

| Instrument | What it does | When preferred | Can it fail? |
|---|---|---|---|
| **CAR against a labelled baseline (DiD)** *(used)* | Abnormal return net of a drifting comparator | Always, on a trending universe | Testing against **zero** instead is the failure this replaced — it made the placebo significant |
| **Market-model / FF3 / FF5 / characteristic-matched benchmarks** **[GAP]** | Alternative abnormal-return definitions | Robustness: the result should not depend on the benchmark model | A beta-1 market adjustment (used here) is the crudest of these; the paper states it as a bounded assumption but does not test alternatives |
| **BHAR vs CAR** **[GAP]** | Buy-and-hold vs cumulative | BHAR is closer to an investor's experience over long windows | BHAR has known skewness problems in inference; report both or justify one |
| **Calendar-time portfolio** **[GAP]** | Aggregates events into a time series | **Robust to cross-sectional correlation of event dates** — the exact problem when episodes cluster in stress windows | Lower power; weights differently across time |
| **Patell / BMP standardised tests** **[GAP]** | Handle event-induced variance increases | Events that themselves raise volatility — fire sales certainly do | — |
| **Corrado rank / generalised sign tests** **[GAP]** | Nonparametric event-study inference | Heavy-tailed returns, which these are extremely | — |
| **Kolari–Pynnönen adjustment** **[GAP]** | Corrects for cross-sectional correlation in clustered event dates | Directly applicable: episodes concentrate in macro-stress windows | — |
| **Two-way clustered SE** *(used)* | Unit and time dependence | Panels with both | Few clusters in either dimension breaks it — check the count |
| **Driscoll–Kraay** **[GAP]** | Robust to general cross-sectional dependence | When cross-sectional dependence is strong and clusters are few | — |
| **Wild cluster bootstrap** **[GAP]** | Inference with few clusters | If the month-cluster count is small in any subsample | — |
| **Staggered-adoption DiD estimators** **[GAP]** — Goodman-Bacon decomposition, Callaway–Sant'Anna, Sun–Abraham, de Chaisemartin–D'Haultfœuille | Correct for the bias of two-way FE when treatment turns on at different times with heterogeneous effects | **This design is closer to staggered adoption than to a two-period DiD** — episodes start and end at different times for each stock. The recent literature shows TWFE is biased in exactly this setting. **The most substantive methodological gap in the programme** | Requires a clean definition of treatment timing per unit, which episodes do supply |
| **Dose–response on the continuous variable** *(used)* | Sidesteps estimated-regressor attenuation entirely | When labels are model-generated | Dilutes a tail effect into a linear slope — which is what happened, and was read correctly |
| **Horizon ladder as a signature test** *(used)* | Discriminates mechanisms: build-and-persist vs decay vs flip | Any dynamic effect with a mechanism claim | Only discriminates if the competing mechanisms predict different profiles — state them first |
| **Placebo-in-time / permutation (randomisation) inference** *(used in part)* | Null constructed by shuffling | Always | **A placebo whose event is mechanically different from the treatment's is not a placebo** — the failure diagnosed here |
| **Oster's δ / Altonji–Elder–Taber** **[GAP]** | Bounds omitted-variable bias from coefficient movement across specifications | The specification ladder already produces the inputs — this is nearly free given what was run | Requires an assumption on the R² bound |
| **Rosenbaum bounds / E-value** **[GAP]** | How strong must an unobserved confounder be to overturn the result | Any observational claim | Presentational rather than identifying, but it makes fragility explicit |
| **PIN / structural microstructure estimator** *(used)* | Independent construct, no shared inputs with the return tests | External validation | Its loading was not *specific* to the hypothesised group — the test that mattered was the contrast, not the level. Modern alternatives: **VPIN, and the Duarte–Young adjustment for the PIN estimator's known biases** **[GAP]** |

---

## E · Risk-model and density validation (L5)

*Used: Kupiec POF, Christoffersen independence, PIT with a block bootstrap,
CRPS, pinball loss, ES coverage, ES/VaR ratio, Diebold–Mariano.*

| Instrument | What it does | When preferred | Can it fail? |
|---|---|---|---|
| **Kupiec POF** *(used)* | Unconditional coverage of VaR breaches | The minimum bar | Coverage can be exact while breaches cluster |
| **Christoffersen independence / conditional coverage** *(used)* | Breach clustering | Always alongside Kupiec | Tests only first-order dependence |
| **Engle–Manganelli Dynamic Quantile (DQ) test** **[GAP]** | Regression of the hit sequence on lagged hits and covariates | **More powerful than Christoffersen** and can test dependence on *any* covariate — including the volatility state where a gradient was suspected | Needs a covariate set specified in advance |
| **Berkowitz LR / Berkowitz tail test** **[GAP]** | Transforms PIT to normal and tests jointly for mean, variance and autocorrelation; the tail variant censors above a threshold | **The tail version is the right instrument for a tail claim** — a full-support PIT test spends its power on the middle | Assumes the transform is valid |
| **PIT independence** **[GAP]** | PITs must be i.i.d. uniform, not merely uniform | **Uniformity alone was tested here.** Serial dependence in the PITs is dynamic misspecification, and breach clustering already suggests it is present | — |
| **Acerbi–Székely Z1/Z2/Z3; Du–Escanciano** **[GAP]** | Direct **ES backtests** | ES is the reported risk measure; it was validated here only through a coverage *ratio*, which is not a test | Different Z-statistics have different power profiles; report more than one |
| **Fissler–Ziegel joint (VaR, ES) scoring** **[GAP]** | ES is **not elicitable alone**; the pair is jointly elicitable, giving a consistent scoring function | **The correct way to compare two engines on ES.** Directly fixes the weakest part of the current comparison | Requires choosing a member of the class |
| **CRPS** *(used)* | Proper score over the whole distribution | General density comparison | Dominated by the centre. A 0.55% CRPS margin is **not** primarily a tail statement |
| **Threshold-weighted CRPS** **[GAP]** | CRPS with a weight function emphasising the tail | **The right instrument for the claim actually being made here.** The engine's case is a tail case; the score used is a whole-distribution score | Weight choice must be pre-registered |
| **Murphy diagram** **[GAP]** | Dominance across the entire class of consistent scoring functions | **Settles whether a ranking depends on the loss chosen.** Given how small the CRPS margin is, this is the decisive exhibit | Visual; pair with a formal dominance test |
| **Reliability–sharpness paradigm** | Maximise sharpness subject to calibration | Framing the whole comparison | An engine can win calibration by being vague; sharpness must be reported with it |
| **Filtered Historical Simulation (McNeil–Frey)** | Standardise by a conditional volatility model, resample standardised residuals empirically | — | **This is what the engine is.** It is a named method with a canonical reference, and the paper must cite it rather than present the construction as novel |
| **EVT peaks-over-threshold with a GPD tail** **[GAP]** | Parametric far tail beyond the empirical support | **Exactly the abstention problem**: 6.3% of rows get no 1% forecast because the empirical tail is too thin. A GPD tail is the standard remedy | Threshold selection is the hard part and is itself a validation question |

---

## F · Attribution and model comparison (L2, L3)

*Used: SHAP on the gradient-boosted challenger; Cohen's κ for label agreement;
BIC sweep for model order.*

| Instrument | When preferred | Can it fail? |
|---|---|---|
| **SHAP** *(used)* | Exact for trees; consistent local attributions | Correlated features split credit arbitrarily among themselves — with a deliberately correlated feature block, group-level attribution is the honest unit |
| **Permutation / drop-column importance** **[GAP]** | Model-agnostic; drop-column is the ground truth at high cost | Plain permutation on correlated features evaluates off-manifold; use **conditional permutation** |
| **Mean decrease in impurity** | — | Biased toward high-cardinality features. Avoid |
| **Partial dependence / ALE** **[GAP]** | Shape of a relationship, not just its magnitude | PDP is misleading under correlation; ALE is the fix |
| **Cohen's κ** *(used: 0.891)* | Agreement between two labelings with **aligned** label sets | Requires the alignment. If two models partition without a natural correspondence, κ is undefined or arbitrary |
| **Adjusted Rand index / normalised mutual information** **[GAP]** | Partition agreement **without** requiring a label alignment | The more appropriate instrument for comparing an unsupervised state partition to a rule-based one, since state indices are arbitrary |
| **AIC / BIC / HQIC** *(used)* | Model order | **BIC's penalty is negligible at large n** — it declined monotonically at n = 2.7M and could not discriminate. This is the paper's cleanest example of an instrument incapable of failing |
| **Cross-validated likelihood; WAIC / PSIS-LOO** **[GAP]** | Model order without an asymptotic penalty | The correct replacement when an information criterion cannot discriminate; needs a fold scheme respecting dependence |
| **Vuong test** **[GAP]** | Formal comparison of **non-nested** models | Puts a p-value on a comparison usually made by eye | Assumes both models are strictly non-nested |

---

## G · Effective sample size and dependence (cross-cutting)

| Instrument | Used | Note |
|---|---|---|
| Kish effective sample size, deflated by overlap | ✔ | The core of Q4 |
| Intra-class correlation → n_eff | ✔ | — |
| Newey–West with stated lag | ✔ | Lag selection rule should be stated (e.g. Andrews) |
| Moving-block / circular-block bootstrap | ✔ | Block length sensitivity should be shown |
| **Stationary bootstrap (Politis–Romano)** | **[GAP]** | Random block length; removes the sensitivity to a single block choice |
| Date-block bootstrap with the null imposed | ✔ | The construction that fixed the zero-power test |
| **Subsampling / self-normalisation** | **[GAP]** | Alternatives when block-length choice is contentious |

---

## The paper's own gap list

Applying the framework to the programme produces a specific, checkable list.
Ordered by how much a referee would care:

| # | Gap | Why it matters |
|---|---|---|
| 1 | **Staggered-adoption DiD estimators** | The design has heterogeneous treatment timing; two-way FE is known to be biased in that setting. The most substantive methodological gap |
| 2 | **Selection accounting** — PBO, deflated Sharpe, White/Hansen SPA, Model Confidence Set | Eight strategies and four engine variants were tried; nothing prices that search |
| 3 | **Proper ES backtests** — Acerbi–Székely, Du–Escanciano, and Fissler–Ziegel joint scoring | ES is the headline risk measure and was validated only by a coverage ratio |
| 4 | **Threshold-weighted CRPS and a Murphy diagram** | The claim is a tail claim; the score is a whole-distribution score with a 0.55% margin |
| 5 | **PIT independence, and the Berkowitz tail test** | Only uniformity was tested; breach clustering suggests serial dependence is present |
| 6 | **Diebold–Mariano on nested pairs** | DM is invalid under the null for nested models; Clark–West is the correction. Audit which comparisons are nested |
| 7 | **Purged CV with embargo** | Overlapping labels are everywhere; only the frozen split and walk-forward protect against them |
| 8 | **Filtered Historical Simulation and McNeil–Frey must be cited** | The engine is a named method; presenting the construction without the citation is a positioning error |
| 9 | **Hidden semi-Markov model** | Would have unified the state model and the separate episode-end hazard model |
| 10 | **ARI/NMI instead of κ** for partition agreement; **Decision Curve Analysis** to formalise the forecast-versus-decision gap | Both are the more appropriate instrument for a comparison already made |

**This list is an asset, not an embarrassment.** A framework paper that cannot
generate a concrete list of what its own case study should have done differently
has not demonstrated that the framework does anything. Section §11 of the spine
is where it goes.
