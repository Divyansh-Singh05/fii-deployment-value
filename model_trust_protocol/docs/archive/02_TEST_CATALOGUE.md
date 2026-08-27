# 02 · Test Catalogue

The instruments the protocol calls for. Scoped to walk-forward validation of a
distributional or risk forecast.

`Have` = a working reference implementation exists in the case codebase and can
be generalised into the library.

---

## Q1 · Alignment

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T1.1 | Unit assertions at write and serve boundaries | Log/simple, bp/decimal, level/return mismatches | Every persisted field asserts its unit | partial |
| T1.2 | Calendar-arithmetic assertion on every scored row | Outcome dated by calendar rather than trading days; gaps and halts | The realised outcome's date is asserted against the origin + h trading days | ✔ `trading_calendar.py` |
| T1.3 | **Identical-mask assertion before any ratio of aggregates** | Paired quantities over different populations | Assertion in code before the division, not in review | ✖ build |
| T1.4 | **Parameter unit register** | Estimation/deployment time-base mismatch | A table of every parameter with its unit at both ends; reviewed as an artifact | ✖ build |
| T1.5 | Moment-target check on any correction | Correcting `E[log Y]` where the metric consumes `E[Y]` | The estimated moment is the one the metric consumes | ✔ (as fixed) |
| T1.6 | Feature-domain declaration | Query times at which a feature is undefined rather than stale | The interface refuses queries outside the domain | ✔ (declared) |

## Q2 · Fit admissibility

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T2.1 | Convergence gate on every refit | Non-converged steps entering production | Step accepted only if the optimiser converged **and** the objective improved; otherwise incoming parameters retained | ✔ `s05_refit_harness.py` |
| T2.2 | No-fallback rule | Publication of the best inadmissible candidate | Raise, naming the vintage and per-candidate reason | ✔ |
| T2.3 | Cold/warm start agreement | Path dependence on flat likelihoods | Parameters agree to stated tolerance from both starts; tight tolerance plus derivative-free polish | ✔ modules 21–24 |
| T2.4 | **Symmetric maturity floor** | Asymmetric handicapping of higher-parameter models | Minimum training observations enforced identically for every competitor; immature blocks unscored for all | ✔ |
| T2.5 | Near-miss re-estimation sweep | Defect noise read as suppressed signal | Every \|t\| in [1.5, 2.5] re-run after T2.3–T2.4 | ✔ |

## Q3 · Causality

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T3.1 | Static causal gates | Ordinary look-ahead | `shift(1)`, explicit embargo `s + h ≤ asof`, forward-only filtering | ✔ |
| T3.2 | **Truncation audit** | Any full-sample quantity reaching an earlier row | Re-execute on data ending at *T*; every row ≤ *T* identical to the full run at **exact zero**; multiple truncation dates | ✔ `s13_lookahead_audit.py` — the most portable artifact in the codebase |
| T3.3 | **Availability equality clause** | Finite→NaN flips slipping through a both-finite comparison | Availability mismatch is a failure | ✔ |
| T3.4 | Immutable dated vintages | Restated parameters; hidden staleness | Append-only, `asof` strictly prior, staleness served as a field | ✔ |
| T3.5 | **Causal stratification** | Evaluation-set look-ahead | Every conditional diagnostic reported on both a full-sample and an expanding own-past cut; the causal one is the calibration statement | ✔ `panel_inference.expanding_quintiles` |
| T3.6 | Audit-scope statement | Overstated coverage | Stages in and out of scope named in the verdict | ✔ |
| T3.7 | Publication-lag declaration | Data available later than its event date | Declared as unfalsifiable absent dated vendor snapshots | ✔ |

## Q4 · Effective evidence

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T4.1 | Kish ESS with overlap deflation | Row count mistaken for information | Every ESS deflated by the overlap factor at source | ✔ `s08` |
| T4.2 | Intra-class correlation / n_eff | Panel dependence in a scored series | Reported alongside every n | ✔ `icc_neff` |
| T4.3 | **Overlap-vs-decimation probe** | Whether overlap distorts *shape* or only *precision* | Compare overlapping quantiles to h disjoint decimated series **and** to the spread across offsets. Diagnosis determines the remedy | ✔ |
| T4.4 | Independent-observation support floors | Floors stated in rows | Minimum ESS and minimum tail counts expressed in independent observations | ✔ `s08`/`s09` |
| T4.5 | Per-component support counts | Pooled counts certifying component densities | Counts per component, combined with the estimate's own weights | ✔ |
| T4.6 | Burn-in weight report | Warm-up contamination of a recursive state | Residual first-observation weight reported at the point the state is first used; equal-weighted burn-in seed | ✔ |
| T4.7 | Abstention accounting | Unreported coverage limits | Abstention rate reported per horizon and level, with where it binds | ✔ |
| T4.8 | **Abstention-selection audit** | Adversarial withholding flattering the metrics | Outcome distribution of withheld vs served rows compared on sd, low quantiles, minimum and exceedance rates | ✔ `LIMITATIONS.md` L2 |

## Q5 · Test power

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T5.1 | **Synthetic size/power harness** | A test statistic that cannot reject | Labelled must-reject and must-not-reject samples; size ≈ nominal, power ≈ 1 on degenerate alternatives, **before** any real verdict | partial — exists as a bespoke script; **must become the library's first module** |
| T5.2 | Null-centred block bootstrap | A null that tracks the alternative | Null expectation + recentred residuals, block-resampled | ✔ `pit_chi2_bootstrap` |
| T5.3 | Date-block bootstrap CI | Panel dependence in coverage statistics | Block length stated and justified | ✔ `block_bootstrap`, `breach_rate_ci` |
| T5.4 | Kupiec unconditional coverage | Breach frequency | LR against nominal, pooled and per unit | ✔ `s10_validation.py` |
| T5.5 | **Christoffersen independence** | Breach clustering at correct frequency | Reported alongside coverage, always | ✔ |
| T5.6 | PIT uniformity against a valid null | Distributional shape | Scored against T5.2, never a textbook χ² | ✔ |
| T5.7 | Diebold–Mariano on daily aggregates, Newey–West | Dependence in paired score comparisons | Aggregate to date level; state the lag | ✔ `dm_daily`, `nw_var` |
| T5.8 | Multiplicity adjustment | Family-wise error | Family size declared before results | ✔ |
| T5.9 | Prior-stripped replication metric | Metrics that embed what they verify | Classify on the emission model alone before claiming out-of-sample persistence | ✔ |

## Q6 · Mechanism

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T6.1 | **Pre-registered mechanism ablation** | Wins attributed to the wrong cause | Identical machinery, mechanism removed, everything else fixed. Bar and power pre-registered | ✔ (C6) |
| T6.2 | Ablation power statement | "No difference" without detectability | Report the effect size the design was powered to detect | ✔ (0.03% of CRPS) |
| T6.3 | Stratified ablation | Mechanisms that matter only where they are uncertain | Ablate within strata, not only pooled | ✔ |
| T6.4 | Mechanical twin agreement | Model class taking credit for the measure | Rule-based backbone matched on the scored output; report κ | ✔ (κ = 0.89) |
| T6.5 | Magnitude alongside significance | Significant-but-negligible differences | Both reported; stability arguments preferred where accuracy gaps are negligible | ✔ |
| T6.6 | Benchmark-depth symmetry | Flattered benchmarks | Q5 run on the benchmark's diagnostics too | ✔ |

## Q7 · Conditional failure

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T7.1 | Conditional calibration report over causal strata | Gradients hidden by pooling | Strata cut per T3.5 | ✔ `conditional_report` |
| T7.2 | **Block-bootstrap CI on every stratum** | Point estimates read without their precision | An interval covering nominal is not evidence of miscalibration | ✔ |
| T7.3 | Crisis-window scoring | Systemic-shock behaviour | Scored separately, with the structural limit stated | ✔ `crisis_mask` |
| T7.4 | Per-unit pass rates | Pooled passes over failing minorities | Reported for coverage and independence | ✔ |
| T7.5 | Horizon-wise reporting | Calibration generalised across horizons | Every metric reported per horizon | ✔ |

## Q8 · Boundary

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T8.1 | Interface retirement rule | Serving a configuration that fails its gate | Unreachable from the query interface without an explicit override | ✔ `s12_risk_engine.py` |
| T8.2 | Serve-vs-artifact re-derivation at startup | Drift between the validated object and the served answer | Re-derive from raw densities and compare; ours 1.14e-07 | ✔ |
| T8.3 | Out-of-claim event statement | Scorecards implying coverage they do not have | Named, with the realised versus forecast magnitude | ✔ |
| T8.4 | Measurement-truncation statement | Tails the recording convention cannot contain | Declared with its sign and bound | ✔ |
| T8.5 | Assumption register | Fragility scattered across sections | Each assumption marked tested / bounded / maintained, with bias direction | ✔ |

## Q9 · Economic meaning

| ID | Instrument | Detects | Pass rule | Have |
|---|---|---|---|---|
| T9.1 | Capital restatement at matched coverage | Score-unit improvements nobody can act on | Capital and loss-beyond-VaR, at matched breach rates | ✔ |
| T9.2 | Calibration/prediction separation statement | Distributional results read as directional | Stated explicitly in the results section | ✔ |
| T9.3 | Density-score vs regression-significance comparison | Predictors significant in a mean regression and worthless in a density | Both reported for any proposed predictor | ✔ |
| T9.4 | Deployable-lag re-run | Non-implementable timing | Recomputed at the lag a decision can use | ✔ |
| T9.5 | Cost grid and breakeven | Costless-world verdicts | Breakeven reported, not a point verdict | ✔ |

---

## Build list

Ordered by value to the paper.

| ID | Why |
|---|---|
| **T5.1** synthetic size/power harness, generalised | The paper's central demonstration and the library's first module. Currently a bespoke script |
| **T1.3** identical-mask assertion | Two lines of code; catches a sign-inverting failure class that no statistical test can see |
| **T1.4** parameter unit register | Same class, same cost, catches a 77%-magnitude error |
| Generalised **T3.2** truncation harness | Works, but is written against this pipeline's stage names. Needs a model-agnostic interface |
| **Validation-suite self-check** (design requirement D1) | Assert that every declared gate emitted a verdict this run. Without it the battery's coverage is unverified |
