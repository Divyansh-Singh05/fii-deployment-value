# 01 · The Protocol

Nine questions, in order. Each must be answered before the next is worth
asking. Each carries a **void scope**: what a failure at that step invalidates
downstream.

Every failure mode below is a property of the evaluation design — something a
competent practitioner following standard practice will produce on correct code
and clean data. None is presented as an implementation slip.

```
Q9  Economic meaning      is the improvement worth holding capital differently?
Q8  Boundary              does the model know and declare where it stops?
Q7  Conditional failure   where does it fail, and is that finding or noise?
Q6  Mechanism             does the model's distinguishing feature contribute?
Q5  Test power            can the backtest reject?
Q4  Effective evidence    how much independent information is behind each number?
Q3  Causality             could each number have been produced on its date?
Q2  Fit admissibility     did the scored parameters come from an admissible fit?
Q1  Alignment             is the forecast scored against the outcome it predicts?
```

The ordering rule is **void scope, not importance**. An alignment failure voids
every number in the scorecard including the benchmark's; a conditional-stratum
failure voids one table. Answering them out of order means redoing the expensive
work.

---

## Q1 · Alignment — "Is the forecast being scored against the outcome it actually predicts?"

**Voids: the entire scorecard, for every engine in the comparison.**

Alignment failures do not add noise. They produce well-behaved numbers of the
right magnitude and the wrong meaning, and they survive every downstream test
because every downstream test consumes the same misalignment.

| # | Check | Failure mode | Requirement |
|---|---|---|---|
| 1.1 | Units at every boundary | A field stored in log returns and served through a simple-return transform; basis points against decimals | Assert units at the write boundary, not in review. The field name carries the unit |
| 1.2 | Forecast origin vs outcome date | The h-day outcome for an origin at the close of day *t* lands on the *t+h*-th **trading** day, not the *t+h*-th calendar day. Gaps, halts and holidays break the mapping | Explicit calendar arithmetic; assert the realised outcome's date on every scored row |
| 1.3 | **Row populations of paired quantities** | Any ratio of two aggregates computed over different masks. Mean *predicted* ES over mean *realised* VaR is the canonical case | Assert an identical row mask before any ratio of means |
| 1.4 | Parameter time base | A transition matrix estimated with one transition per *observed row* and deployed with one per *elapsed trading day*. These are different units, not an approximation | A register of every parameter's unit at estimation and at deployment |
| 1.5 | Moment targeted by a correction | Estimating `E[log z² \| x]` by OLS on `log z²` when the metric depends on `E[z² \| x]`. Under heavy tails those slopes differ | Estimate the moment the metric consumes |
| 1.6 | Structural domain of a feature | A within-day cross-sectional rank is defined at the close and **undefined** intraday; it is also universe-relative, so it is a property of the eligible set, not of the instrument | State the query times at which the feature exists, and refuse the others |

**Measured on the case engine.** 1.3: tail fatness at h = 20 reported as
**1.131** — thinner than the Gaussian's structural 1.1457, i.e. tails
Gaussianising with horizon. Correctly paired: **1.428**. The sign inverted, and
the corrected number reverses the economic reading. 1.4: mixing half-life
reported at **8.7 days** against **33.8 days** under a time-consistent
estimator — a **77% error**, large enough that a research programme was
specified to repair a deficit that did not exist. 1.5: with heavy tails the two
slopes differ by roughly a factor of two, so a variance correction estimated on
the wrong moment under-shrinks by exactly that factor and leaves half the defect
standing.

---

## Q2 · Fit admissibility — "Did the parameters being scored come from a fit that is allowed to be scored?"

**Voids: every vintage, and therefore every forecast built on it.**

In a walk-forward design the model is refitted hundreds of times without a human
looking at any individual fit. Admissibility must therefore be a property the
system enforces, not one an analyst inspects.

| # | Check | Failure mode | Requirement |
|---|---|---|---|
| 2.1 | Convergence is a gate, not a diagnostic | An optimiser's return value consumed without reading its success flag; a monotonicity indicator recorded and never read | A non-converged step is rejected and the incoming parameters retained, preserving the objective's monotonicity |
| 2.2 | No fallback publication | Selecting `best(passing) or best(all)` publishes the best *inadmissible* candidate when none passes | Raise, naming the vintage and the per-candidate reason. A period with no admissible fit is unserved, not served badly |
| 2.3 | Path independence | Warm-started quasi-Newton on a flat small-edge likelihood stops at trajectory-dependent points; identical training data reaches different parameters | Tight tolerances plus a derivative-free polish; verify cold-start and warm-start agreement |
| 2.4 | **Estimation maturity** | Parameters scored before the training window can support them. This is *asymmetric*: a 2-parameter base barely suffers, an 8-parameter model suffers badly — so the defect handicaps precisely the models under test | A minimum-training-observations floor applied **symmetrically**; immature blocks unscored for every competitor |
| 2.5 | Near-miss re-estimation | Any \|t\| in roughly [1.5, 2.5] re-run after 2.3 and 2.4 | A near-miss that moves under a numerical fix was never evidence |

**Measured on the case engine.** Under 2.4, base volatility existed only from
day 750, so early tilt vintages were fit on 0–500 pairs and scored anyway, over
**~18%** of the scored window. Under 2.3 and 2.4 jointly: three engine variants
whose pre-registered gates had "narrowly missed" widened into clear failures,
and one apparently recovered positive at **t = −2.79** resolved to **t = −0.90**.
The one engine whose gate genuinely passed was **unchanged to every printed
digit** — which is what makes its pass credible.

**Design principle.** Small-edge engine results can be manufactured by
estimation artifacts alone. A gate that passes only before 2.3–2.4 are applied
has not passed.

---

## Q3 · Causality — "Could each number have been produced on the date it carries — in the model *and* in the evaluation?"

**Voids: the out-of-sample label.**

Causality is audited in four places. Published practice generally audits the
first and asserts the rest.

| # | Check | Failure mode | Requirement |
|---|---|---|---|
| 3.1 | Data flow | Trailing statistics that include the current observation; a scoring window that overlaps the estimation window | `shift(1)` on every trailing statistic; an explicit embargo (`s + h ≤ asof`); forward-only filtering |
| 3.2 | **End-to-end proof, not assertion** | Static causal gates are assumptions about the code, and a train/test split neither demonstrates absence of look-ahead nor reproduces deployment | **Truncation audit**: re-execute the production chain against data ending at *T*; compare every row dated ≤ *T* against the full-sample run |
| 3.3 | Availability equality | A quantity that is finite in the full run and NaN in the truncated run compared only where both are finite | A finite→NaN flip is a **failure**, not a skipped row. This is the loophole most look-ahead audits leave open |
| 3.4 | Parameter immutability | Vintages restated when the model is refitted; staleness carried as hidden state | Dated snapshots, `asof` strictly prior to the rows they price, append-only. Staleness is an output field |
| 3.5 | **Evaluation-set causality** | Stratifying a calibration diagnostic on a full-sample quantile of the conditioning variable. "The calmest quintile of this name's history" includes its future, and knowing a day sits there genuinely predicts that volatility will rise | Every conditional diagnostic computed twice — full-sample cut and expanding own-past cut. The **causal** cut is the calibration statement; the full-sample cut is a legitimate ex-post question and must be labelled as one |
| 3.6 | Audit scope declaration | An audit that rebuilds four of nine stages reported as certifying the pipeline | Name the stages in scope and out |
| 3.7 | Availability lag | Truncating on **event** date cannot detect data published later than the event it describes | Declared as an unfalsifiable assumption unless vendor snapshots at different times exist |

**Measured on the case engine.** 3.2–3.3 pass at three truncation dates, up to
291,778 rows, 12 risk columns and 24 scoring columns, at
`max |diff| = 0.000e+00` — exact zeros, not tolerances. The same harness caught
a genuine leak during development: a variance-correction estimator aggregated
over a matrix indexed by a stock set drawn from a later panel, so the truncated
world silently contained different names; over a million rows differed and the
gate fired immediately.

3.5 is the finding worth the most outside this project. A conditional breach
gradient of **0.30** across within-name volatility quintiles — read as a model
defect, and the subject of two rounds of estimator redesign — measures **−0.04**
on the *same forecasts* under a causal expanding stratum. Roughly half the
original gradient was a real effect (volatility mean reversion, correctable);
the residual was the diagnostic conditioning on the future.

---

## Q4 · Effective evidence — "How much independent information is behind each number the model serves?"

**Voids: every tail quantile, every support floor, every abstention decision.**

| # | Check | Failure mode | Requirement |
|---|---|---|---|
| 4.1 | Overlap accounting | Building h-day forward returns from a sliding one-day window. Consecutive windows share h−1 days, so the independent information is ≈ 1/h of the row count | Deflate every effective-sample-size measure by the overlap factor **at source** |
| 4.2 | Correct diagnosis of what overlap costs | The usual claim — overlapping windows narrow tails and create overconfidence — is wrong. Overlapping windows remain valid draws from the marginal h-day distribution; the empirical CDF stays consistent | Test it: compare overlapping quantiles against the mean of h disjoint decimated series **and** against the spread across those offsets. If the overlap gap is far smaller than the between-offset spread, overlap costs **efficiency, not width** — and the remedy is to withhold thin forecasts, not to widen served ones |
| 4.3 | Support floors count independent observations | A minimum-sample floor stated in rows overstates the evidence behind every h-day density by ≈ h | Every floor — minimum ESS, minimum tail count — expressed in independent observations. An h-day number must clear the same evidential bar a 1-day number does |
| 4.4 | Per-component support | A mixture component's density built from one component's observations, certified by a count pooled over all components | Support counts per component, combined with the same weights the estimate uses |
| 4.5 | Warm-up contamination of the state | Seeding a recursive variance state with the first squared return. First observations are not ordinary, and inflated σ suppresses standardised outcomes and **understates** risk | Seed with an equal-weighted mean over a burn-in block; report the residual first-observation weight at the point the state is first used |
| 4.6 | Abstention as a declared coverage limit | A model that withholds a forecast where support is thin is scored only on rows it chose to answer | Report the abstention rate and where it binds. Abstention is a coverage limit, not a defect — provided 4.7 holds |
| 4.7 | **Abstention-selection audit** | Withheld rows might be systematically the dangerous ones, flattering every metric | Compare the outcome distribution of withheld against served rows on sd, low quantiles, minimum, and exceedance rates. The direction must be **measured**, never argued |

**Measured on the case engine.** 4.2: at p0.1 the overlapping-versus-decimated
gap is **0.012** while the spread across five disjoint offsets is **0.22** —
roughly twenty times larger. No detectable narrowing. 4.3: correcting the floors
improved h = 5 calibration on *every* measure (5% breach 0.0535 → 0.0518, 1%
breach 0.0106 → 0.0091, ES coverage 1.025 → 1.015, PIT χ² 2.4 → 1.3) while
removing the 1% forecast from **33%** of 5-day rows, and made h = 20 unservable
from first principles rather than by measured over-breaching. 4.4: per-component
counts suppressed **63.5%** of hard-labelled 5-day 1% rows and the survivors were
newly flagged miscalibrated. 4.5: with a 20-day half-life, **12.94%** of the
variance at observation 60 — the point the state is first used — was still a
single day's squared return; first-observation squared returns run a median 1.53×
the name's own median and p99 292×. A burn-in seed reduced that weight to 1.25%.
4.7: withheld rows are **calmer** than served rows on every measure (sd 1.026 vs
1.063, p0.1 −4.95 vs −5.29, P(z<−4) 0.227% vs 0.266%) — the selection is real,
non-random, and its sign is the opposite of the concern.

---

## Q5 · Test power — "Can this backtest reject?"

**Voids: every p-value and every PASS verdict.** Note the asymmetry: a Q5
failure can leave a *conclusion* standing while destroying the *evidence* for
it. Both must be stated.

| # | Check | Failure mode | Requirement |
|---|---|---|---|
| 5.1 | **Size and power on labelled synthetic samples** | A test statistic deployed without ever having been shown a case it must reject | Construct must-reject and must-not-reject samples with known ground truth; measure size and power before any real verdict is issued |
| 5.2 | The null must not track the alternative | Building a critical value by resampling the **observed** statistic's inputs. The reference distribution then rises in lockstep with the statistic | Impose H₀ by construction: decompose into the null expectation plus a residual, recentre the residuals across blocks, and resample the recentred residuals. Dependence is preserved; the null is not |
| 5.3 | Reference validity under dependence | Using a textbook i.i.d. reference distribution on a panel of contemporaneously correlated series | Measure the correct critical value by a dependence-preserving bootstrap. Do not assume the inflation factor |
| 5.4 | Coverage is not independence | Reporting unconditional coverage (Kupiec) alone. Breach frequency can be exactly nominal while breaches arrive in clusters | Christoffersen independence alongside coverage, per unit and pooled |
| 5.5 | Scoring dependence | A paired t on daily loss differentials across a correlated panel | Aggregate to the date level; Newey–West with a stated lag; state the block length and its justification |
| 5.6 | Multiplicity | A family of pre-registered tests reported at nominal α | Declare the family size **before** results; adjust |
| 5.7 | **Metrics that cannot fail** | A replication statistic that embeds the quantity it claims to verify. A state decode using a frozen highly-persistent prior reports high persistence almost regardless of the data | Strip the prior from the metric — classify on the emission model alone — before calling the result out-of-sample validation |

**Measured on the case engine.** 5.2 is the paper's demonstration. A PIT
uniformity test whose bootstrap resampled the observed bin counts was scored on
600,000 synthetic draws over 2,000 dates:

| PIT sample | χ² | critical value | rejected? |
|---|---:|---:|---|
| uniform | 16.8 | 41.0 | no ✓ |
| mass piled at zero | 343,485 | 345,970 | **no ✗** |
| all mass in the bottom half | 600,006 | 600,029 | **no ✗** |
| **all mass in one bin** | 5,400,000 | 5,400,000 | **no ✗** |

Zero power. The replacement null-centred construction rejects all three
degenerate cases and holds size at **5.0%** on date-clustered uniform data.

The verdicts this changed: the engine remained not-rejected at h = 1 (41.8
against 584.1) — the conclusion survived, the evidence for it did not. The
**benchmark**, previously passing, now **rejects at every horizon** (h = 1:
18,961 against 641). 5.3: the textbook χ²(9) critical value is 16.9; the correct
date-block value at h = 1 is **584** — an error of ~36×, and in the direction
that manufactures rejections. 5.4: violations cluster — independence rejects for
**10.9%** of names against a 5% nominal, a defect coverage alone cannot see.
5.7: a transition diagonal reported as replicating at **0.95** out of sample
measures **0.888** once the sticky prior is removed from the metric.

---

## Q6 · Mechanism — "Does the thing that makes this model different contribute anything?"

**Voids: the attribution of the win, though not the win.**

This is the question a model's authors are least likely to ask and a user is
most likely to need answered. A risk model wins for a reason. Establishing
*which* reason requires scoring the model against itself.

| # | Check | Requirement |
|---|---|---|
| 6.1 | **Mechanism ablation** | Re-run the identical machinery with the distinguishing mechanism removed and everything else held fixed. Pre-register the bar and the power |
| 6.2 | State the power of the ablation | "No difference" is only a result if the test could have found one. Report the effect size the design was powered to detect |
| 6.3 | Ablate across strata, not only pooled | A mechanism that matters only where it is uncertain will be invisible pooled and visible in the uncertain stratum |
| 6.4 | Mechanical twin | Compare the fitted model against a rule-based backbone matched on whatever the output is scored against; report agreement |
| 6.5 | Separate significance from magnitude | A difference can be statistically significant and economically irrelevant. Report both, and prefer a stability argument to an accuracy argument where the accuracy gap is negligible |
| 6.6 | Benchmark audited to the same depth | Run Q5 on the benchmark's diagnostics too. A benchmark held to a weaker standard flatters the model |

**Measured on the case engine.** 6.1: the flow conditioning — the mechanism the
engine exists to exploit — is **statistically indistinguishable from its
removal** at every horizon and in every stratum, in a design powered to detect
differences above **0.03% of CRPS**. 6.5: soft versus hard component weighting
differs by **1.5 × 10⁻⁵** CRPS on the full panel: significant, negligible, and
material only in the **5.5%** of rows where the label is genuinely uncertain —
where the better argument is day-over-day VaR stability, not accuracy. 6.4: a
census-matched rule backbone agrees with the fitted state model at **κ = 0.89**.

**The consequence for the headline.** The engine's advantage is attributable to
the empirical **shape** of the standardised distribution, not to the conditioning
variable. The defensible claim is therefore *the Gaussian is wrong for this
market* — a claim about the benchmark — and not *this conditioning variable
improves risk forecasts*. Q6 is what separates those two sentences, and no other
question in the protocol can.

---

## Q7 · Conditional failure — "Where does it fail, and is that a finding or noise?"

**Voids: the pooled pass.**

| # | Check | Requirement |
|---|---|---|
| 7.1 | Strata, not pools | Breach and coverage across volatility, liquidity, size, era and horizon strata — each cut **causally** (Q3.5) |
| 7.2 | **Interval width decides what is a finding** | Report block-bootstrap CIs on every stratum. A stratum whose interval covers nominal is not evidence of miscalibration however extreme the point estimate |
| 7.3 | Crisis windows | Score the systemic-shock windows separately, and state what the model can structurally do about them |
| 7.4 | Per-unit pass rates | Pooled calibration can pass while a large minority of individual names fail |
| 7.5 | Horizon | Calibration at one horizon says nothing about another |

**Measured on the case engine.** 7.2 is the discipline that matters. A crisis
window breaching at **3.19× nominal** looks like a catastrophic failure; its
interval is **[0.0479, 0.2619]**, which covers nominal, because 11,456 highly
dependent rows carry far less information than the row count suggests. The
volatility gradient's intervals were tight and excluded nominal. **The gradient
was the finding; the crisis was not** — the reverse of what the point estimates
suggest. 7.3: through the 2020 crash both engines break (14.29% breaches against
a 5% target) and VaR moves from −9.2% to −26.7% — the model adapts, *behind* the
event, which is the structural limit of any backward-looking volatility model
and should be stated as such rather than tuned away. 7.4: per-name Kupiec pass
rates of 90.4% against 60.4% are a more informative comparison than either
pooled breach rate.

---

## Q8 · Boundary — "Does the model know where it stops, and does the system say so?"

**Voids nothing. Its absence voids trust.**

| # | Check | Requirement |
|---|---|---|
| 8.1 | Retire what fails | A configuration that fails its calibration gate is removed from the served interface, reachable only by explicit in-code override |
| 8.2 | Abstain rather than extrapolate | Where support is below the floor, serve nothing (with Q4.6–4.7 accounting) |
| 8.3 | Serve-vs-artifact re-derivation | Re-derive the served number from the validated artifact at startup and compare | 
| 8.4 | State what the model cannot forecast | Name the events outside the model's claim rather than letting a scorecard imply coverage of them |
| 8.5 | State what the measurement cannot contain | Tails truncated by the recording convention; bounded, one-directional understatements declared with their sign |
| 8.6 | Assumption register | Every assumption listed once, each marked **tested / bounded / maintained**, with the bias direction for bounded ones |

**Measured on the case engine.** 8.1: h = 20 breaches **11.53%** against a 5%
target, has no servable 1% forecast under an independent-evidence floor, and
loses CRPS to the Gaussian — it is unreachable from the query interface. 8.3:
maximum discrepancy between the served number and the validated artifact
**1.14 × 10⁻⁷**, so the deployed engine cannot silently drift from the object it
was validated on. 8.4: a −40.4% single-day move in a name with 1.7% daily
volatility was not forecast and is not claimed — ES(1%) stood at −3.37σ against a
realised −30.41σ. The engine's claim is calibration in the 3–6σ range where
losses cluster, and it holds ~27% more capital against that range than the
Gaussian. 8.5: a name suspended at −60% that ultimately paid nothing is recorded
at −60%; the left tail is truncated at the suspension price, a bounded
one-directional understatement.

---

## Q9 · Economic meaning — "Is the improvement worth changing anything for?"

**Voids: the reason to adopt.**

| # | Check | Requirement |
|---|---|---|
| 9.1 | Restate in capital | Convert score improvements into capital at matched coverage and into loss beyond VaR. A CRPS delta is not a decision input |
| 9.2 | Match on coverage before comparing capital | Comparing capital held at unmatched breach rates compares two different risk appetites |
| 9.3 | Distinguish calibration from prediction | A better-calibrated distribution is not a directional forecast, and no return-predictability claim follows from a calibration result |
| 9.4 | Significance versus forecasting value | A predictor with a strong regression t-statistic can leave a well-specified density forecast unimproved |
| 9.5 | Deployable lag and cost, where a decision is claimed | Recompute at the lag the decision can actually use; report breakevens across a cost grid rather than a point verdict |

**Measured on the case engine.** 9.1: at matched 99% coverage the flow-tilted
market engine holds ~**2%** less capital and average loss beyond VaR99 falls
**0.770% → 0.613%**, a 20% reduction — statable to a risk committee in a way
that "ΔCRPS = −0.0019" is not. 9.4 is the case's most generalisable substantive
result: measured four independent times, conditional-mean volatility predictors
with regression t-statistics of 3–5 **failed** to improve a well-specified
GARCH-t density forecast, succeeding only against weak baselines. A ~1–2%
modulation of σ can be real, survive every control, and still sit below density-score
detectability. 9.5: an effect at t = −5.53 contemporaneously measures **t = −0.32**
when the signal is known only at t−2.

---

## Design requirements the protocol assumes

These are properties of the validation system rather than questions about the
model. They are stated once and are not part of the sequence.

| | Requirement | Rationale |
|---|---|---|
| D1 | **Every declared gate emits a verdict every run**, and the run asserts that the verdict count matches the gate count | An exception is a failure, never a skipped check. Otherwise the battery's coverage is unverified |
| D2 | **No data and bad data are distinguishable** at every gate | A gate that crashes on an empty slice cannot report "nothing served" |
| D3 | **Verdict rules are fixed before results exist**, and a gate's target is fixed with them | Re-aiming a gate after seeing results is exploratory, permanently, and must be labelled so |
| D4 | **A missed bar is reported, not re-specified** | The failure branch is written into the pre-registration alongside the pass branch |
| D5 | **Rejected artifacts are recorded** so a later run cannot silently reinstate them | Falsification must bind in code, not in a notebook |
| D6 | **The withdrawal record is kept alongside the result** | Superseded numbers are preserved and marked, not overwritten |
