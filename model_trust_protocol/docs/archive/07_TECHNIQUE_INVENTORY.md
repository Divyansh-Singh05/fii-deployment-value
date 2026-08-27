# 07 · Technique Inventory

Everything in the two source projects that constitutes genuine methodological
engineering — excluding data preparation and implementation work. Each entry:
what was done, why it is non-obvious, and what it generalises to.

**Reading note.** Coefficient values quoted from the pre-audit thesis are
marked *(superseded)* — the construct correction changed the numbers. The
*methods* are unaffected and are what this inventory is about.

---

## L1 · Measurement — validating a number before any model sees it

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 1.1 | **Pre-registered existence criterion.** Before the first fit: *"the target state exists iff some fitted state shows clearly negative mean on axis A **and** clearly negative mean on axis B"* | Written as a falsifiable condition on the *fitted object*, not as a hypothesis about the world. It is what prevented rationalising whatever clusters EM happened to produce | Any unsupervised model. State, in advance, the structure whose absence would refute you |
| 1.2 | **Within-day rank→probit normalisation**, `F = Φ⁻¹(rank/(n+1))` | Buys three things at once: invariance to participation growth and reporting-regime changes, exact N(0,1) marginals, and removal of the market-wide component by construction. The cost is explicit — market-level information is forfeited, and the cross-sectional census is pinned | Any feature measured on a growing, regime-changing panel where levels are not comparable across time |
| 1.3 | **Coverage gate: missing attribution must produce missing data, never mismeasurement.** Null the measure wherever attributable value falls below 50% of the day's total, checked **value-weighted** rather than row-weighted | The row-weighted version overstates the problem and invites a bad fix. The distinction between "unmeasurable here" and "measured as zero" is the whole point | Any derived measure with incomplete attribution |
| 1.4 | **Trailing statistics end yesterday**, and structural breaks are masked *before* any window touches them | Standard, but the second clause is not: a window that spans a known break silently mixes two regimes | Any rolling feature |
| 1.5 | **Liquidity floor on the measure itself** — concentration computed from two prints is noise, so require a minimum count | Distinguishes a measure that is *unreliable* from one that is *extreme* | Any concentration, dispersion or ratio statistic |

## L2 · Model class — deciding what kind of model the question needs

This is the layer with the most transferable content, and the one most papers skip entirely.

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 2.1 | **Autocorrelation feasibility diagnostic, run before fitting.** Measure the lag-1 autocorrelation of every input; a state-space model can only allocate states along temporally coherent axes, because persistence is what makes self-transition probabilities informative | Measured: 0.926 on the persistence axis against 0.290–0.334 on the concentration axes. The model was being asked to find structure on an axis it was **structurally incapable of seeing** — not a tuning problem, an inductive-bias mismatch. Diagnosed *before* re-engineering, not after | **Any latent-variable or sequence model.** Check that your features' time scale matches your model class's inductive bias before blaming the fit |
| 2.2 | **The smoothing fix, and its legality argument.** A 5-day trailing mean of daily snapshots raised autocorrelation to 0.767 and made the axis visible — and it averages *measurements*, not entities, so it required no cross-period identity | The fix is chosen to respect a constraint established at L1. Most feature engineering does not carry a legality argument | Any transformation applied to make a feature model-visible. State what the transformation assumes |
| 2.3 | **Information criteria are not discriminating at large n — read solution structure instead.** At n ≈ 2.7M, BIC's `p·log n` penalty is negligible against any real likelihood gain, so BIC declines monotonically in k | "BIC chose k=5" would have been reported as model selection. It is not evidence at that n. What settled it was *inspecting what the extra states were*: a persistence ladder with transient noise states, never the hypothesised structure | Any model-order selection on large samples. State whether your criterion can discriminate at your n |
| 2.4 | **The dissection test: is the structure absent from the data, or invisible to the class?** Before concluding the phenomenon does not exist, count it directly — 25.8% of persistent-sell days had dispersed books, 36.8% concentrated | Separates two conclusions with completely different consequences: *"there is nothing there"* versus *"this class cannot carve it"*. Only the second justifies a hybrid architecture | Any negative result from an unsupervised model |
| 2.5 | **Pre-registered architectural fork.** Written before the fits: *if the criterion fails at every k, use the model for what it can see and overlay deterministic rules for what is episodic* | The fallback is committed to in advance, so adopting it is not a post-hoc rescue | Any project where a negative result would require redesign |
| 2.6 | **The structural challenger (factorial HMM).** The starvation result was tested against the structural fix — two independent chains, direction and concentration, emitting additively, exact EM on the 9-state product space. Pre-registered gates: EM monotonicity; **exactness against an independently written forward pass** (rel. 5.9e-16); the channel must *earn* states (census ≥5% each, loading spread ≥0.20); OOS drift bars | The finding is a clean scope statement: the starvation is a property of **flat** HMMs, not of the data. Given its own transition matrix, the concentration channel formed three well-populated states. All five seeded inits reached the same optimum — the solution is not an initialisation artifact | Any "model X cannot represent Y" claim. Test the structural fix before generalising the limitation |
| 2.7 | **The mechanical twin.** A three-line rule backbone — training-era quantile cuts, census-matched so both tag the same fraction of days — with identical downstream overlays reproduces the main table within band. Label agreement 92.7%, **Cohen's κ = 0.891** | Demotes the sophisticated model to what it actually buys: episode smoothness, not identification. Rule-based runs are ~14% shorter, producing ~48% more episode ends with the same economics. The finding is **model-class-independent**, which is a stronger claim than the model's | **Any fitted model.** Build the dumbest thing that produces the same output type, match it on the confound, and report agreement |
| 2.8 | **Pre-registered escalation bar.** A gradient-boosted challenger was fitted with all features; the bar for justifying a sequence model was a *decomposition* bar — does the within-unit demeaned signal carry information? Observed t = 1.48 against a bar of 2 | The escalation question is not "is the bigger model better" but "is the thing the bigger model supplies the binding constraint". Decomposing the challenger's edge into regime information + static characteristics + timing residual is what answers it | Any decision to move up a model class |

## L3 · Fitted model — validating an unsupervised model with no ground truth

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 3.1 | **Decompose "accuracy" into five falsifiable sub-claims** when no oracle exists: structural stability, statistical reality of the labels, reproducibility, external validity, economic content | "The model is 93% accurate" is unavailable and would be meaningless. "Every measurable property replicates on a disjoint era and a 45%-different universe" is available and is a stronger claim than an accuracy number | **Every unsupervised model in finance.** This decomposition is the inventory's single most portable item |
| 3.2 | **Frozen-parameter decode on a disjoint era.** Signatures replicate almost digit-for-digit across eras on a substantially different universe | The universe difference is a feature: TEST is not a repeat sample | Any frozen-split protocol |
| 3.3 | **Identifying which replication rows are incapable of failing** — and excluding them from evidence. Three were found: the census is pinned by within-day ranking, so it *cannot* look unstable; the transition diagonal is circular, because a Viterbi decode using a frozen 0.95-sticky prior reports ~0.95 regardless; the effect size on the variable the labels are *defined* by is tautological | This is the discipline that recurs everywhere in the projects. **The only row that was both non-tautological and non-circular was the signature row** — and saying so is worth more than the other three rows combined | Any replication table. Ask of each row: could this have come out differently? |
| 3.4 | **Isolating the prior's contribution.** Four methods compared, including per-day classification on the frozen emission parameters alone with **no transition matrix at all** — the one measurement simultaneously look-ahead-free and untouched by any test-fitted parameter. Genuine test-era persistence: **0.888**, not 0.95, against a 0.346 no-persistence chance floor | Quantifies exactly how much of an apparent out-of-sample result is the prior restating itself: +0.061, over half the remaining headroom | Any Bayesian or regularised model reporting an out-of-sample property the prior also encodes |
| 3.5 | **A rejected fix, recorded with its reason.** Refitting the transition matrix by EM on the test era was tried and rejected: Baum–Welch is forward-*backward*, so it uses the entire test period at once — look-ahead *within* the test era — on top of violating the frozen-split discipline | Records a tempting, plausible, wrong idea and why. That is more useful to a reader than the correct method alone | Any protocol document |
| 3.6 | **Effect-sizes-first, because p-values are decoration at large n.** Cohen's d with a pre-registered cross-era drift bar of ≤0.15 | At n ≈ 800k everything is significant. The falsifiable statement is about *stability of magnitude*, not about rejection | Any large-panel descriptive comparison |
| 3.7 | **The episode-clustering permutation test.** Null: 200 within-stock shuffles preserving each stock's label *count* while destroying temporal adjacency; statistic = mean run length; p = (1 + #{null ≥ obs})/(1+B). Observed 3.64 d against a 1.47 d null — **2.48×**, p = 0.005 — replicating at 2.38× out of sample and robust to the threshold choice | The null is constructed to hold everything constant except the property being claimed. This is what makes "the labels form real episodes" a testable statement rather than a visual impression | **Any claim that discrete labels cluster** — regimes, events, states |
| 3.8 | **Reproducibility as a measured quantity.** The chain re-executed on different hardware, a different OS and a different library generation: the feature store regenerated **byte-identical**; the census matched exactly on all four active archetypes; **three stock-days in 804,958** flipped between control categories — EM tie-breaking at machine precision, 4×10⁻⁶ of the panel | For a pipeline with an EM fit inside it, this is the strongest available form of an accuracy claim: **the model is a deterministic function of the data and the frozen protocol, not of the session that produced it** | Any stochastic-fit pipeline |
| 3.9 | **Face validity checked before the outcome data existed.** The longest episodes were mapped to companies and dates *before any price data touched the project* — landing on canonical macro-stress windows and on two firms that later went bankrupt, in their documented distress windows | Ordering is the whole point: after seeing the price results, face validity is unfalsifiable storytelling. Before, it is a necessary condition that would have ended the project cheaply | Any labelling model. Do it first or do not claim it |
| 3.10 | **Calibration with built-in falsification.** The first-choice threshold method (a mixture model on the marginal) **failed its own pre-registered stability test** — a 49/51 component split with boundary drift 0.665 — and was discarded. The replacement quantile rule was confirmed by **three-way convergence** across independent methods (−0.441, −0.510, −0.513) | A calibration step that *can* fail, did fail, and the failure is retained as a methods exhibit. Convergence across methods that share no machinery is much stronger evidence than any one method's fit | Any threshold, cut-point or hyperparameter selection |

## L4 · Causal claim — econometric identification

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 4.1 | **Difference-in-differences against a labelled baseline, not against zero.** Excess CAR = CAR(treated) − CAR(baseline) with date-clustered bootstrap intervals | Adopted because testing against zero under a drifting baseline made the **placebo** significant. "Different from zero" is meaningless when the baseline drifts +52 bp | Any event study on a trending universe |
| 4.2 | **Anchor choice as an identification decision.** START (drift while the condition is active) versus END (behaviour after it stops) answer different questions and are reported separately | The reversal claim requires the END anchor; using START would measure something else and look like the same thing | Any event study on states with duration |
| 4.3 | **Delisting truncation rather than exclusion.** Events kept, windows truncated at last trade, with truncation percentages printed per group | Dropping delisted names is survivorship bias; the direction of the residual understatement is stated and shown to *strengthen* the permanence finding | Any long-window event study |
| 4.4 | **Two-way clustered standard errors** (unit × calendar month) with the conservative direction argued, plus **date fixed effects that absorb every market-wide variable by construction** — which is why a macro variable can only enter as an interaction | The FE structure determines what questions are even askable. A macro *level* is not identifiable here, and the paper says so rather than reporting a spurious coefficient | Any panel with contemporaneous cross-sectional dependence |
| 4.5 | **A deliberate bad control as a conservative lower bound.** The pre-episode return is included in the strictest specification, knowingly absorbing part of the effect, and reported as a floor | Inverts the usual incentive: the strictest spec is the headline, not the friendliest | Any specification ladder |
| 4.6 | **Estimated-regressor attenuation argued explicitly.** Labels are estimated, so dummies attenuate toward zero and the estimates are conservative — and then **sidestepped entirely** by a continuous dose–response on the underlying variable with no thresholds and no labels | Two independent routes to the same claim, one of which does not depend on the labelling at all | Any regression on model-generated regressors |
| 4.7 | **Placebo validity is itself testable.** The pre-registered placebo failed; decomposing *why* showed its event was, by construction, a directional transition — structurally invalid at that anchor. It was replaced by the control that differs from the treatment in exactly one dimension | A failed placebo is usually treated as a failed study. Diagnosing it produced a **stronger** control than the original | Any placebo or falsification test |
| 4.8 | **The horizon ladder as a signature test, not a robustness check.** Transitory impact must *build and persist*; a momentum artifact would decay; a microstructure bounce would flip. Reported at 10/20/30/60 days | The horizon profile discriminates between competing mechanisms — it is identification, not sensitivity analysis | Any dynamic effect with a mechanism claim |
| 4.9 | **Non-overlapping subsample by greedy spacing.** Episodes kept ≥28 calendar days apart within unit; the effect **strengthened** | Kills the double-counting objection in the direction that matters | Any overlapping-event design |
| 4.10 | **Mechanical-date exclusion.** Every episode within ±7 calendar days of any of 150 index-review dates dropped — discarding 41% of episodes — with a ±3-day variant reported | Rules out passive/rebalancing mechanics as the driver, at a large and stated cost in sample | Any equity result that could be index mechanics |
| 4.11 | **The hindsight-advantaged control.** A flow-surprise yardstick fitted on the **full sample deliberately**, with the rationale on record: hindsight makes it the best-case competitor, so the treatment surviving it is the stronger result. Its own forward performance is labelled a decomposition finding, never a signal | Inverts the usual leakage concern into an argument. Requires stating clearly that the control is not a result | Any "does X survive controlling for Y" test where Y is hard to construct causally |
| 4.12 | **Microstructure decomposition rather than dismissal.** Bid-ask bounce quantified by re-running on bounce-free windows (t+3…t+22): the bounce component is 11% and 24% by era, and **every headline is quoted with its bounce-free companion thereafter** | Does not claim the bounce is absent; measures it, keeps it attached, and notes that the spread concession *is* the mechanism in miniature | Any short-horizon return result |
| 4.13 | **"Significant versus not significant" is not a test.** The headline contrast was re-stated as a direct test of the coefficient difference, and reported with its power bound: an effect below ~40 bp would be undetectable here | The single most common inferential error in applied finance, caught in self-review and corrected | Any two-group comparison stated as a contrast |
| 4.14 | **The free-alternative head-to-head.** A reversal proxy computable from public data alone entered alongside the proprietary measure; and a three-rung incremental ladder — public → +conventional flow → +the new block — with paired daily t at each rung | Answers "does this data need to be bought" rather than "is this data significant" | Any proprietary-data claim |
| 4.15 | **An independent construct with no shared inputs.** A structural microstructure model estimated by MLE on daily counts — inputs sharing nothing with the return tests | Genuine independence is rare and is the strongest form of external validation available without new data | Any claim that could be corroborated by a differently-sourced construct |
| 4.16 | **Pre-committing the *reading* of a conflict before running the test.** When two results appeared to contradict each other, the module header stated in advance what each possible outcome would mean, then ran three tests — specificity, within-unit persistence, and the decisive moderation test | Prevents the resolution being chosen after seeing which way it went. The verdict — that the corroborating construct **cannot arbitrate the question in either direction** — is a genuine third option that post-hoc reasoning rarely reaches | Any apparent contradiction between two results |
| 4.17 | **Power bounds stated on null results.** Interaction standard errors of ≈27 bp give 95% intervals of roughly [−26,+80] and [−79,+30]; the claim made is *"no evidence of moderation"*, explicitly not *"moderation is absent"* | A null without a power bound is not a finding | Every null result |

## L5 · Forecast — walk-forward and distributional validation

Fully specified in `01_PROTOCOL.md` as the nine questions. Additional items from the state-forecasting layer:

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 5.1 | **Filtered, not smoothed.** Labels used for inference were re-derived as forward-filtered posteriors `P(S_t | x_{1:t})` from frozen parameters, because full-sequence decoding lets the state at *t* see the future. Gate: **economic invariance** — the main table must survive on causal labels | Most regime papers report smoothed states and out-of-sample claims in the same breath. The onset lag is measured (median 1 day) rather than assumed negligible | Any latent-state model whose labels feed a downstream claim |
| 5.2 | **The mandatory comparator is persistence.** With self-transition ≈0.95, "predict no change" is brutal and is required as a baseline for every forecast | Beating a naive baseline is the minimum bar, and on sticky series it is a high one | Any state or regime forecast |
| 5.3 | **A baseline caught being quasi-circular, and amended before rerunning.** The first persistence baseline conditioned on yesterday's *smoothed* label — unknowable at *t* and near-circular with the smoothed target. Replaced with the information-set-fair version: persistence of yesterday's *filtered* state. The original gate failure is kept on the record | The amendment is documented *before* the rerun, which is what separates it from moving the goalposts | Any baseline construction |
| 5.4 | **Calibration reported where it fails, not only in aggregate.** Excellent where 79% of the probability mass lives (0.99 → 0.994), overconfident mid-range (0.70 → 0.55), stated as a consumption caveat | A single Brier score hides both | Any probabilistic output |
| 5.5 | **Discrete-time hazard with walk-forward refits**, right-censoring handled, runs broken on calendar gaps >21 days, every parameter learned in-window and frozen before application | Episode *ends* are the economically relevant event, and modelling them as a hazard rather than a classification handles censoring correctly | Any duration or regime-exit forecast |
| 5.6 | **Out-of-sample stronger than in-sample as a memorisation check.** AUC 0.809 test against 0.786 train | Not proof, but a check that costs nothing and would have flagged overfitting | Any walk-forward model |

## L6 · Decision — backtest engineering

| # | Technique | Why it is non-obvious | Generalises to |
|---|---|---|---|
| 6.1 | **The cheat-alpha gate — a positive control for the backtest engine.** Feed the engine tomorrow's return as the signal. It **must** produce Sharpe > 10 at zero lag and collapse at lag 1. Observed: **+211.5 → −0.5** | This is the best single idea in the projects. It tests the engine's *alignment machinery* by constructing a case whose correct answer is known in advance. A backtester that cannot detect a deliberate look-ahead cannot be trusted to be free of an accidental one | **Every backtest engine ever written.** Cheap, decisive, and almost never done |
| 6.2 | **Exactness gate against a naive implementation.** The vectorised engine must equal a naive per-day loop | Vectorisation bugs are silent and directional | Any optimised numerical pipeline |
| 6.3 | **Cost-accounting gate on a constructed book.** A deliberately constructed flip book whose turnover must be exactly 2.0, verifying `net == gross − c·TO` | Turnover accounting is where cost assumptions quietly become wrong | Any cost model |
| 6.4 | **The engine may not change without re-passing its gates** | Makes the engine an audited artifact rather than a script | Any shared research infrastructure |
| 6.5 | **Execution timing written as an equation.** Signal formed at close *t*, traded at close *t+L*, earning the close-to-close return of *t+L+1* — and, for state events, the rule that an episode end is knowable only at the close of the first day *after* the run | The second clause is where most regime backtests leak. Writing the timing as an equation makes it auditable | Any event-driven backtest |
| 6.6 | **Baselines frozen before the model twins are built** | Removes the option of tuning the comparison | Any paired model-versus-baseline comparison |
| 6.7 | **Paired ΔSharpe with a moving-block bootstrap** (block 20 d, B = 2000), pre-registered as the verdict rule | Paired, blocked, and pre-registered — three separate defences against the standard backtest inference failure | Any strategy comparison |
| 6.8 | **Gross and net diagnosed as separate layers.** Gross answers "is the signal real"; net answers "can it be harvested". Reported separately with different conclusions | Collapsing them loses the finding: the signal was real *and* unharvestable, which is the interesting result | Any cost-sensitive strategy |
| 6.9 | **Breakeven cost as the headline, not a point verdict.** 2–8 bps one-way, against a grid of 0/5/10/15/30 | A breakeven is a fact about the strategy; a net Sharpe at one cost assumption is a fact about the assumption | Any net-of-cost claim |
| 6.10 | **Margin in bp per dollar traded**, compared directly against the cost assumption | Puts the two comparable quantities beside each other | Any turnover-heavy strategy |
| 6.11 | **A construction flaw in the authors' own strategy reported rather than fixed silently** — unequal leg magnitudes turned one book into a churn book | The verdict that depended on it is discounted accordingly | Any multi-leg strategy |
| 6.12 | **The trivial-rule head-to-head, paired on common events.** A three-line age rule beat the fitted hazard model by −15 bp (t = −2.96) on 2,764 common episodes | Both were positive against zero. Only the paired comparison on shared events revealed the ordering | Any model proposed for a decision |
| 6.13 | **Limits-to-arbitrage as the reading of a negative backtest.** Breakevens below institutional costs *explain why the regularity persists* rather than refuting it | Converts a failed strategy into an economic finding, legitimately, provided the gross result is established first | Any real-but-unharvestable effect |

---

## What this inventory is not

Excluded deliberately: data cleaning, identifier resolution, corporate-action
handling, and every implementation defect. Also excluded: techniques that are
standard and carry no design decision — computing a Sharpe ratio, fitting an
OLS, running a bootstrap.

## Density

**59 techniques** across six layers. Roughly a third are standard practice
applied carefully; the rest involve a design decision that a competent
practitioner could plausibly get wrong or skip, and that changes what the
result means.

The ones that would survive on their own, in any field, ranked:

1. **The cheat-alpha positive control** (6.1) — universal, trivial, decisive.
2. **The five-way decomposition of "accuracy" for unsupervised models** (3.1).
3. **The autocorrelation feasibility diagnostic** (2.1) — check the model class
   can see the axis before blaming the fit.
4. **Identifying replication rows incapable of failing** (3.3) and its siblings
   throughout (the tautological effect size, the circular diagonal, the
   zero-power null, the quasi-circular baseline, the structurally invalid
   placebo).
5. **The mechanical twin** (2.7) — match the dumb model on the confound and
   report κ.
6. **The episode-clustering permutation test** (3.7) — a null that holds
   everything constant except the claimed property.
7. **Reproducibility as a measured quantity** (3.8).
8. **Pre-committing the reading of a conflict** (4.16).
