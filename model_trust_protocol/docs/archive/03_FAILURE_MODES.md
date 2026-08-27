# 03 · Failure-Mode Register

Every failure mode the protocol tests for, stated as a property of the
evaluation design, with the magnitude it induced on the case engine and the
question that detects it.

**Direction** is the direction of the error in the *reported* result:
`FLATTER` = the model looks better than it is; `PENALISE` = worse than it is;
`INVERT` = the sign or the reading reverses; `VOID` = the number carries no
information either way.

---

## F1 · Alignment (Q1)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F1.1 | A ratio of two aggregates computed over different row populations — mean *predicted* ES over mean *realised* VaR | Tail-fatness at h = 20 reported **1.131** (below the Gaussian's 1.1457, i.e. thinner tails) against a correctly paired **1.428**. The economic reading reverses from "tails Gaussianise with horizon" to "tails fatten with horizon" | INVERT |
| F1.2 | A parameter estimated in one time unit and deployed in another — transitions per observed row, propagated per elapsed trading day | Mixing half-life **8.7 d** reported against **33.8 d**; a **77%** error, and the reported figure was below the benchmark where the true figure is above it | INVERT |
| F1.3 | A correction estimated on the wrong moment — `E[log z²\|x]` by OLS where the metric consumes `E[z²\|x]` | Under heavy tails the two slopes differ by ≈2×; the correction under-shrinks by that factor and leaves roughly half the targeted defect standing | PENALISE |
| F1.4 | Stored field units disagreeing with served units (log returns served through a simple-return transform) | Silent; detected by unit assertion, not by any statistical test | VOID |
| F1.5 | A feature whose structural domain is narrower than its query interface — a within-day cross-sectional rank queried intraday | The cross-section does not exist mid-session; the feature is undefined rather than stale | VOID |
| F1.6 | A universe-relative feature treated as an instrument property | Two runs under different eligibility floors are not comparable, and a backfilled universe reprices history for every surviving instrument | VOID |

## F2 · Fit admissibility (Q2)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F2.1 | A non-converged optimiser step accepted as a scored parameter set | Structural: in a walk-forward design no human inspects any individual fit, so admissibility must be enforced, not observed | VOID |
| F2.2 | Fallback publication — `best(passing) or best(all)` — publishing the best inadmissible candidate when none passes | A period with no admissible fit is served badly rather than left unserved | FLATTER |
| F2.3 | Path-dependent optimisation on a flat small-edge likelihood | One diagnostic moved from **t = −2.79** to **t = −0.90** under tightened tolerances plus a derivative-free polish. Three "narrow misses" widened into clear failures | FLATTER |
| F2.4 | Estimation-maturity handicap — parameters scored before the training window supports them, **asymmetrically** across model sizes | Affected ~**18%** of the scored window; a 2-parameter base is nearly unaffected while 5–8-parameter models are badly penalised, so the defect handicaps exactly the models under test | PENALISE |

## F3 · Causality (Q3)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F3.1 | Look-ahead asserted by static gates rather than demonstrated end-to-end | The truncation audit is the demonstration: exact zeros across 12 risk and 24 scoring columns at three truncation dates, up to 291,778 rows | — |
| F3.2 | An audit that compares only where both runs produce a finite value | A quantity that flips finite→NaN between the full and truncated runs passes silently. Requiring availability equality closes the most common loophole in look-ahead audits | FLATTER |
| F3.3 | An aggregation whose index set is drawn from a later panel | Detected in development: over **one million rows** differed between the full and truncated worlds. Invisible to every static gate | FLATTER |
| F3.4 | **Evaluation-set look-ahead — stratifying a diagnostic on a full-sample quantile** | Identical forecasts: breach-rate spread across within-name volatility quintiles **0.30** on a full-sample cut, **−0.04** on a causal expanding cut. Roughly half the originally reported 0.64 gradient was a genuine effect; the residual was the diagnostic conditioning on the future | INVERT |
| F3.5 | An audit that truncates on event date, in a feed with a publication lag | Undetectable from inside a single vendor delivery; must be declared as an unfalsifiable assumption | FLATTER |
| F3.6 | Audit scope overstated — a partial rebuild reported as certifying the pipeline | Four of nine stages rebuilt; the remaining five uncertified | FLATTER |

## F4 · Effective evidence (Q4)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F4.1 | Support floors stated in rows on overlapping h-day windows | Every density's evidence overstated by ≈ h: ~200 independent windows at h = 5 where the floor counted 1,000; ~50 at h = 20 | FLATTER |
| F4.2 | **Misdiagnosing what overlap costs.** The standard claim — overlap narrows tails and creates overconfidence — is false | Overlapping vs decimated quantile gap at p0.1 = **0.012**; spread across five disjoint offsets = **0.22**, twenty times larger. Overlap costs efficiency, not width. The remedy is to withhold thin forecasts, not to widen served ones | — |
| F4.3 | Correcting F4.1 (the cost of doing it right) | h = 5 improved on every measure (5% breach 0.0535→0.0518, 1% 0.0106→0.0091, ES coverage 1.025→1.015, PIT χ² 2.4→1.3) — while the 1% forecast disappeared from **33%** of 5-day rows and h = 20 became unservable from first principles | — |
| F4.4 | Pooled support counts certifying a component-specific density | Per-component counts suppressed **63.5%** of hard-labelled 5-day 1% rows; the survivors were newly flagged miscalibrated | FLATTER |
| F4.5 | Recursive volatility state seeded with the first squared return | At the observation where the state is first used, **12.94%** of the variance is one day's squared return; first-observation r² runs median 1.53× the name's own median, p99 292×. Inflated σ suppresses standardised outcomes. A burn-in seed reduces the residual weight to **1.25%** | FLATTER |
| F4.6 | Abstention unreported | **6.3%** of rows receive no 1% forecast, concentrated where history is shortest. A coverage limit, not a defect — but only once F4.7 is measured | FLATTER |
| F4.7 | Abstention selection unmeasured | Withheld rows are **calmer** than served rows on every measure (sd 1.026 vs 1.063; p0.1 −4.95 vs −5.29; P(z<−4) 0.227% vs 0.266%). The selection is real, non-random, and its sign is the opposite of the standard concern — so the reported metrics are mildly conservative | PENALISE |

## F5 · Test power (Q5)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F5.1 | **A bootstrap null built by resampling the observed statistic's inputs.** The critical value rises in lockstep with the statistic | Zero power. On 600,000 synthetic draws over 2,000 dates the test failed to reject a PIT with **all mass in one bin** (χ² 5,400,000 against a critical value of 5,400,000), mass piled at zero, and mass entirely in the lower half. Every verdict it produced was void — for the model and the benchmark alike | VOID |
| F5.2 | The replacement: null imposed by construction (null expectation + recentred residuals, block-resampled) | Rejects all three degenerate cases; size **5.0%** on date-clustered uniform data, conservative on independent draws | — |
| F5.3 | A textbook i.i.d. reference distribution applied to a contemporaneously correlated panel | χ²(9) critical value 16.9 against a correct date-block value of **584** at h = 1 — wrong by ~36×, in the direction that manufactures rejections | PENALISE |
| F5.4 | Correcting F5.1 and F5.3 changes the *benchmark's* verdict more than the model's | The model stays not-rejected at h = 1 (41.8 vs 584.1) — conclusion survives, evidence does not. The benchmark moves from passing to **rejecting at every horizon** (h = 1: 18,961 vs 641) | PENALISE (of the benchmark) |
| F5.5 | Unconditional coverage reported without an independence test | Breach frequency at nominal while breaches cluster: independence rejects for **10.9%** of names against a 5% nominal | FLATTER |
| F5.6 | A replication metric that embeds the quantity it verifies | A state decode using a frozen 0.95-persistent prior reports ~0.95 persistence almost regardless of the data; prior-free classification gives **0.888** | FLATTER |
| F5.7 | A family of pre-registered tests reported at nominal α | Family of 9 requires α = 0.00556; the headline comparison clears it (t = 9.81) but marginal results in the family do not | FLATTER |

## F6 · Mechanism (Q6)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F6.1 | **The distinguishing mechanism never ablated.** The model is compared to an external benchmark but never to itself with its own thesis removed | The conditioning mechanism is **statistically indistinguishable from its removal** at every horizon and in every stratum, in a design powered to detect above **0.03% of CRPS**. The win is attributable to distributional shape, not to the conditioning variable | INVERT (of the attribution) |
| F6.2 | Significance reported without magnitude | Soft versus hard component weighting: **1.5 × 10⁻⁵** CRPS on the full panel — significant, negligible, material only in the **5.5%** of rows where the label is genuinely uncertain. The defensible argument there is day-over-day VaR stability, not accuracy | FLATTER |
| F6.3 | No mechanical twin | A census-matched rule backbone agrees with the fitted state model at **κ = 0.89** — the fitted class is a convenience, and the contribution is the measure | FLATTER |
| F6.4 | The benchmark's diagnostics audited less deeply than the model's | See F5.4: under the corrected null the benchmark's position worsens materially, which strengthens the model's relative result while invalidating the original evidence for it | — |

## F7 · Conditional failure (Q7)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F7.1 | Pooled calibration reported without strata | Pooled 5% breach at 1.03× nominal concealed a monotone gradient of 0.64 across volatility strata | FLATTER |
| F7.2 | **Strata reported without interval width** | A crisis window at **3.19× nominal** has an interval of **[0.0479, 0.2619]** that covers nominal — 11,456 highly dependent rows carry far less information than the count implies. The volatility gradient's intervals were tight and excluded nominal. **The gradient is the finding; the crisis is not** — the reverse of what the point estimates suggest | Both directions |
| F7.3 | Crisis behaviour tuned rather than declared | Through the 2020 crash both engines break (14.29% breach against a 5% target); VaR moves −9.2% → −26.7%, adapting *behind* the event. This is the structural limit of a backward-looking volatility model, not a tunable defect | — |
| F7.4 | Pooled pass rates without per-unit rates | Per-name Kupiec pass 90.4% vs 60.4% is more informative than either pooled breach rate | — |

## F8 · Boundary (Q8)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F8.1 | A failing configuration left on the served interface | h = 20 breaches **11.53%** at the 5% level, serves no 1% forecast under an independent-evidence floor, and loses CRPS to the Gaussian (0.647 vs 0.609). Retired from the query interface | FLATTER |
| F8.2 | Deployed answers not re-derived from the validated artifact | Maximum discrepancy **1.14 × 10⁻⁷** once enforced; without the check the served engine can drift from the object that was validated | FLATTER |
| F8.3 | A scorecard implying coverage of events outside the model's claim | A −40.4% single-day move in a name with 1.7% daily volatility: ES(1%) at −3.37σ against a realised −30.41σ. The claim is calibration in the 3–6σ band, where ~27% more capital is held than the Gaussian | FLATTER |
| F8.4 | Measurement-truncated tails not declared | A name suspended at −60% that ultimately paid nothing is recorded at −60%. A bounded, one-directional understatement of the left tail | FLATTER |

## F9 · Economic meaning (Q9)

| # | Failure mode | Measured magnitude | Direction |
|---|---|---|---|
| F9.1 | Improvement reported only in score units | Restated: ~**2%** less capital at matched 99% coverage; average loss beyond VaR99 **0.770% → 0.613%**, a 20% reduction | — |
| F9.2 | Calibration presented as prediction | A better-calibrated distribution supports no directional claim. The engine forecasts distributions, not direction | INVERT (of the claim) |
| F9.3 | **Regression significance treated as forecasting value** | Measured four independent times: conditional-mean volatility predictors with regression t of 3–5 failed to improve a well-specified GARCH-t density forecast, succeeding only against weak baselines. A real 1–2% modulation of σ can sit below density-score detectability | FLATTER |
| F9.4 | Effects reported at an undeployable lag | t = −5.53 contemporaneously, **t = −0.32** with the signal known only at t−2 | FLATTER |

---

## What the register says in aggregate

**38 failure modes** (the register also carries 8 rows that state a corrected
practice or a structural limit rather than a failure; those are excluded from
the counts).

| Direction | Count | Reading |
|---|---:|---|
| FLATTER — the model looks better than it is | 22 | The default bias of an unaudited backtest is optimistic, and the mechanisms are structural rather than adversarial |
| INVERT — the sign or the reading reverses | 5 | The most dangerous class: the result is not weakened, it is wrong |
| PENALISE — the model or its benchmark looks worse than it is | 5 | Including two that penalised the *benchmark*, which is how a model can be credited for its comparator's handicap |
| VOID — the number carries no information either way | 5 | Undetectable by any statistical test on the number itself |
| Both directions, depending on the stratum | 1 | F7.2 |

**Where they sit.** **Twenty of the 38 fall in Q1–Q4** — alignment, fit
admissibility, causality and evidence accounting. None of those requires a new
statistic, more data, or a stronger model. They are properties of how the
backtest is wired, and every one is checkable before a single p-value is
computed. The remaining eighteen are distributed across test power, mechanism
attribution, conditional reporting, boundary declaration and economic
restatement.

**The three with the largest measured consequence** are, in order:

1. **F6.1** — the distinguishing mechanism was never ablated, so the win was
   attributed to the wrong cause. It changes what the result *is*, not how
   large it is.
2. **F5.1** — a test with zero power, which voided every distributional-shape
   verdict for the model and its benchmark simultaneously.
3. **F3.4** — evaluation-set look-ahead, which invented a model defect that two
   rounds of estimator redesign then chased.

None of the three is detectable by adding out-of-sample data, lengthening the
sample, or strengthening the model. That is the protocol's argument for
existing.
