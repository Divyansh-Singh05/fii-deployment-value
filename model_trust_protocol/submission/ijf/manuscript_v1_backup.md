# What Is an Alternative Dataset Worth? Mapping the Friction Boundary of Institutional Flow Data

## ABSTRACT

**Purpose.** This study measured what a large alternative dataset was worth after its statistical content had already been established.

**Design/Methodology/Approach.** A fourteen-year record of foreign institutional flow in Indian equities, covering 25.2 million trades, was applied at seven points in a modelling stack: a cross-sectional volatility regression, a latent regime model with a hazard layer, a stock-day predictive density, two market-level density forecasts differing only in their baseline, a gradient-boosted feature block, and a long-short book. Each faced six ascending constraints: existence, out-of-sample replication, availability at the decision lag, competition against the baseline it would be deployed against, detectability in the metric the decision consumes, and execution cost. Every threshold was fixed before its results existed. A predictor from outside the dataset passed through the same machinery as a control.

**Findings.** None failed at existence or replication. All seven cleared conventional inference and held on a frozen out-of-sample era. Six then failed, at four different constraints. One survived: the record aggregated to market level, entering a density forecast at a two-day reporting lag against a baseline that did not already contain it. The control failed at the same constraint as two of the seven, locating the difficulty in the evaluation, not the data.

**Practical Implications.** The binding constraint depends on the intended use. Practitioners should establish the deployable lag, the competing baseline and the breakeven cost before establishing significance.

**Originality/Value.** Existing work prices one constraint across many predictors. This study priced six, in sequence, across one dataset.

**Keywords:** alternative data, institutional flow, forecast evaluation, transaction costs, replication, risk models

**JEL Classification:** G12, G14, G17, C52, C58

## INTRODUCTION

Most published return predictors do not survive re-testing. Hou et al. (2020) re-examined 452 anomalies and found that 65% failed a conventional significance hurdle once microcaps were handled properly, rising to 82% under a multiple-testing threshold. Jensen et al. (2023) reached a more optimistic verdict on a broader factor set, and the disagreement between the two remains open. What neither settles is what becomes of the findings that do pass.

This study addresses that remainder. It takes one dataset whose statistical content is never in question, and asks how far that content carries once the tests stop being statistical.

The object of study is the data, not the models built on it. A cross-sectional regression, a regime model, a density engine and a book of strategies appear below, and none is offered as a contribution. Each exists to impose one kind of real-world constraint on the same underlying record, so that the record can be asked a question it could not otherwise be asked. When an ablation shows that a regime model's conditioning adds nothing to a risk density, that is a measurement about the flow data, not a verdict on hidden Markov models.

Six constraints organise the analysis, and a result must clear each to reach the next. Existence asks whether the effect is present under conventional inference. Replication asks whether it holds on an era the design never touched. Availability asks whether it is knowable when the decision must be taken. Competition asks whether it beats the baseline the application would be deployed against. Detectability asks whether it registers in the metric the decision consumes. Execution asks whether the edge exceeds the cost of capturing it. The ordering follows institutional reality rather than statistical severity: the first two concern the sample, the remaining four concern deployment.

Because the sequence is ordered, the level at which an application stops is a complete statement of what the data was worth in that use. Read across applications, the levels are not properties of the market but of the use. The same record, aggregated differently and scored against a different baseline, meets a different binding constraint.

The study contributes three things. First, a boundary rather than a verdict: because one application survives, the result maps where the record's value stops rather than finding it has none. Second, evidence that the binding constraint is a property of the use, since six applications fail at four different levels and no single constraint explains the pattern. Third, a provenance protocol under which every reported figure is bound to the artifact that produced it, to a digest of that artifact, and to an extractor that reads the value back out of it.

The design cannot show that the ordering generalises. One dataset in one market does not support that, and no such claim is made. The mechanisms are offered as portable; the magnitudes are not.

## REVIEW OF LITERATURE

### Replication and the Credibility of Published Findings

The scale of non-replication in the cross-section of returns is now documented. Hou et al. (2020) re-examined 452 anomalies and reported that 65% failed a conventional significance hurdle once microcaps were handled through appropriate breakpoints and value weighting, rising to 82% under a multiple-testing threshold and to 96% within the trading-frictions category. Jensen et al. (2023) reached the opposite conclusion on a broader factor set, finding that most factors replicate once they are modelled jointly and that the pessimistic reading overstates the problem. The disagreement is unresolved, and it turns on the criterion applied rather than on the data.

Work around that dispute has sharpened what replication means. Bowles et al. (2024) showed that when an anomaly's returns occur, relative to the date of its discovery, changes what the evidence supports. Chai et al. (2026) surveyed finance researchers directly and documented how frequently replication fails in practice, including on the researchers' own work. Chopra et al. (2023) measured the publication penalty attached to null findings, which explains why studies reporting what did not survive remain uncommon.

Two features of this literature bear on the present study. It concentrates on whether an effect exists, and it evaluates many predictors under a single criterion. Both are appropriate to its purpose and both leave a question open: what becomes of the findings that pass.

### Constraints Between Statistical Significance and Practical Use

A second strand prices a single friction across many candidate findings. Muravyev et al. (2025) applied short-sale costs to 162 anomalies and found that an average long-short return of 0.14% per month before borrow fees became −0.01% after, with no profitability even before fees once the highest-fee 12% of stock-dates were excluded. Kaplanski (2023) examined the cost imposed by slow trading as competitors race to exploit a documented effect. Pénasse (2022) modelled the decay of an edge following its discovery, separating arbitrage from the possibility that the effect was never real to begin with.

Machine-learning research has begun to price design choices rather than predictors. Bagnara (2024) reviewed the field critically and identified transaction costs and research-design decisions as the recurring weak points. Sak et al. (2024) pruned the factor zoo under a portfolio criterion rather than a t-statistic, finding that most surviving factors carry risk prices indistinguishable from zero. Azevedo et al. (2023) tested anomalies internationally across developed, emerging and frontier markets, and Li et al. (2024) replicated and digested anomalies in the Chinese A-share market. Each of these varies the predictor set while holding the evaluation criterion fixed.

### Evaluating Forecasts, Densities, and Backtests

The forecast-evaluation literature supplies the instruments a deployment test requires. Allen et al. (2025) set out tail calibration for probabilistic forecasts, and Waghmare and Ziegel (2026) surveyed proper scoring rules, both bearing directly on the choice of metric when a claim concerns the tail rather than the centre of a distribution. Arnold et al. (2024) decomposed the continuous ranked probability score into interpretable components, which allows a score difference to be attributed rather than merely reported. Gonçalves et al. (2025) established predictive-ability inference under data revision, the inferential counterpart of a vintage-based design. Arian et al. (2024) compared out-of-sample testing methods in a synthetic environment where the truth was known, showing how sharply false discovery depends on the set of competing models retained.

### Alternative Data and Its Valuation

Coyle and Manley (2024) reviewed the empirical methods available for valuing data as an asset, and Sun et al. (2024) surveyed alternative data applications in finance and business. Both establish that the question of what a dataset is worth is recognised and that methods exist to address it. Neither offers a worked case in which one dataset is priced under successive deployment constraints. Related measurement work is relevant to the instruments used below: Ghachem and Ersan (2025) advanced estimation of informed-trading models and documented biases in the standard maximum-likelihood approach, and Ding et al. (2025) applied regime-switching models to realised volatility across eight markets.

### Foreign Institutional Flow in the Indian Market

Foreign institutional activity has been studied extensively in this market, and the most directly relevant recent work concerns whether such trades carry information. Raizada and Nawn (2025) examined the trade informativeness of foreign investors in India and found positive-feedback trading consistent with an informational disadvantage rather than an advantage. Thapa et al. (2026) showed how policy uncertainty shapes foreign institutional trading behaviour. Dey et al. (2024) examined how institutional investors in this market respond to risk, return and volatility. Maheshwari and Naik (2026) documented the shift from foreign to domestic institutional dominance and its behavioural consequences.

This literature works almost entirely from aggregate or holdings-level data. Transaction-level records with participant attribution are held by depositories and rarely reach researchers, which limits what can be established about the composition of flow rather than its direction and size.

### Research Gap

Three gaps follow. First, the replication literature establishes whether effects exist and stops there; what happens to the surviving minority under deployment constraints is not systematically documented. Second, the friction literature prices one constraint across many predictors, which cannot reveal whether the binding constraint varies across uses of the same information, because the constraint is held fixed by design. Third, the alternative-data literature establishes that valuation methods exist without applying them to a single dataset through an ordered sequence of realistic constraints.

### Objectives of the Study

The study pursued three objectives, each expressed as a question the design can answer.

RQ1. Holding the data constant and varying the use, what is a transaction-level record of foreign institutional flow worth at each of several points in a modelling stack?

RQ2. For each application, at which constraint is that worth lost, and is the binding constraint a property of the data or of the use to which it is put?

RQ3. Is the observed attrition peculiar to this record, or does a predictor drawn from outside the dataset attrite in the same way under identical machinery?

RQ1 is answered by the attrition reported in Figure 2. RQ2 is answered by the spread of failures across constraints and, most directly, by the controlled pair in Table 2, in which one predictor passes and fails according to the baseline it is scored against. RQ3 is answered by the external control.

## METHODOLOGY

### The Record and Its Governing Constraint

The object of study is a transaction-level record of foreign institutional activity in Indian equities, sourced from depository settlement data. It covers 1 January 2011 to 31 March 2025 and 25,155,785 trades across 5,960 instrument identifiers. Each record carries a date, an instrument, a masked participant identifier, a side and a traded value. Published foreign institutional data for this market is a daily aggregate; this record resolves to the instrument, the day and the participant, which permits statistics about the composition of flow that aggregate reporting cannot support.

Masked participant identifiers appear persistent and are not. The test applied is the one such data requires: the persistence statistic is computed on a control population whose real-world persistence is known independently. Brokers and clearing members in this market number in the hundreds. Over a 171-month span they present as 22,262 identifiers with a median lifetime of one month. Masked participant identifiers behave identically, at 363,663 identifiers and the same median. Since the control cannot have that turnover, the identifiers are re-minted at monthly boundaries. Every entity statistic below is therefore within-day, and no measurement tracks an entity across a month.

Returns come from exchange settlement prices. Corporate-action factors are verified against observed price ratios before application, with a guard that nulls any post-adjustment daily return above 50% in magnitude. Instrument identity is resolved through a point-in-time map with issuer-bounded closure. The modelling universe is 1,028 instruments over 802,806 instrument-days.

### Frozen Split and Missing Data

Parameters, thresholds and specifications are fixed on data through 30 April 2021. The test era begins 1 July 2021. The intervening two months are absent from the record at source, and the gap doubles as an embargo across a structural break. The eras are not a repeat sample: the training era covers 618 instruments across 508,563 instrument-days, the test era 851 across 294,243, and only 441 instruments appear in both. Three further months carry a wholly null direction flag, comprising 3.004% of trades and falling entirely within the test era. They are excluded, never imputed.

Figure 1 sets out the resulting design. Parameters are sealed into dated vintages, each fitted on data preceding its own date and applied forward without restatement, so that no figure reported for a given day uses information unavailable on that day.

*Figure 1. Frozen Split, Embargo, and the Walk-Forward Vintage Design*

[[FIGURE:figure_1_design.png]]

*Source. Authors' computation.*

### The Six Constraints and the Test at Each

Each application pairs two measurements made on the same object and the same data: what a conventional significance test reported, and what a value test reported once one constraint was imposed. An entry whose two halves come from different samples establishes nothing and is not admitted.

Existence is assessed by panel regression. For instrument *i* on day *t*, with *z* the return standardised by a past-only volatility estimate, the specification is

[[EQ:1]]

where the first two right-hand terms are instrument and date fixed effects, the third is the institutional share of that instrument's turnover, and the fourth is a control vector containing log turnover. Date fixed effects absorb all market-wide variation by construction, so a market-level variable can enter only as an interaction. Standard errors are clustered on instrument and calendar month. With samples in the hundreds of thousands, significance is close to automatic and is treated as necessary rather than sufficient. Replication uses the frozen split, with both eras reported for every quantity and a sign flip between them counted as failure regardless of significance. Availability re-estimates equation (1) with the share taken at a two-day lag, which is when it becomes knowable to a decision maker. Nothing else changes.

At market level the flow variable is constructed from the raw records as

[[EQ:2]]

where the numerator sums buy less sell value over settled equity trades on the day, and the denominator is the trailing 250-day mean of daily gross flow ending on the previous day. The pre-registered risk cell regresses next-day absolute index return on the deployable lag of that variable,

[[EQ:3]]

in which the second flow term is the negative part of the first and is the single permitted asymmetry, five lags of the target enter as controls alongside the implied volatility index, and standard errors are Newey-West at ten lags. Days carrying no flow observation are excluded rather than entered as zero; the alternative reading is reported alongside.

Competition scores an application against the baseline it would be deployed against rather than against zero. Three forms appear: a well-specified conditional-variance model, where the question is whether the information is already in the instrument's own history; a trivial rule capturing the same idea, compared paired on the events both act upon; and a public proxy, where the question is whether the proprietary record is needed.

Detectability asks whether the effect registers in the metric the decision consumes, which is generally not the metric that established significance. Where an application's own mechanism is the claim, this is tested by ablation: the identical machinery re-run with the mechanism removed and everything else fixed.

Execution reports the breakeven one-way cost at which net profit is zero,

[[EQ:4]]

the ratio of expected gross profit to expected turnover, measured in basis points per unit traded and compared against a realistic institutional cost. The breakeven is a property of the strategy; a net figure at an assumed cost is a property of the assumption, and a reader whose costs differ can use the first and cannot use the second.

### Backtest Construction and Pre-Registration

Where a constraint requires a simulated book, timing is specified rather than described. A signal formed at the close of day *t* is traded at the close of *t + L* and earns the close-to-close return of *t + L + 1*, with *L* = 1. For state-derived events, an episode's end is knowable only at the close of the first day after the run concludes. The engine is gated before use and cannot be modified without re-passing its gates: a vectorised implementation must equal a naive per-day loop; an oracle signal equal to the next day's return must produce an extreme ratio at zero lag and collapse at lag one; and a constructed book with known turnover must satisfy net equals gross less cost times turnover.

Each application's threshold was written into its stage header before its numbers existed, together with the branch to be taken if the threshold were missed. An application that misses its threshold is reported as a negative result and is not re-specified. A gate's target is fixed with its threshold, so re-aiming a test at whichever quantity carries the effect is exploratory and is labelled as such.

### Verification of Reported Figures

Every figure in this paper clears four gates before admission. It must be fresh, meaning the artifact is not older than the code that writes it. It must be primary, meaning the source is the computation rather than a document describing it or a copy of its output. It must be traced, meaning the artifact digest matches a recorded value. And it must be extracted, meaning a locator identifying exactly one number reads the value back out of the artifact and confirms it.

The protocol found three errors in the source programme that had already been written down. A metrics table had been quoted from a copy six weeks behind its origin. Two pre-registered gating regressions had no producing stage and were re-derived here from their specifications. One stage had never been re-executed after an audit rebuilt the object it describes, and its reported effect size was a third larger than the current object supports. None changed a verdict.

## RESULTS AND DISCUSSION

### Where the Applications Failed

Figure 2 reports the attrition. Seven applications of the record, plus one external control, are placed at the lowest constraint each fails to survive; the control is excluded from the counts and discussed separately.

*Figure 2. Attrition of Seven Applications Across Six Ascending Constraints*

[[FIGURE:figure_2_attrition.png]]

*Source. Authors' computation.*

Nothing was lost at the first two constraints. Every application produced an effect that was statistically present and replicated on an out-of-sample era covering a substantially different cross-section, in which only 441 of the two eras' 1,028 instruments are common. By the standard the replication literature applies, all seven pass. Attrition began where institutional constraint did, and it spread across four different levels.

### Availability: Information That Arrives Too Late

The institutional share of a stock's turnover predicts its next-day volatility. The dependent variable is the squared standardised return, so this is variation that the model's own volatility estimate has already failed to explain. Across 577,245 instrument-days, with instrument and date fixed effects and log turnover controlled, the coefficient carries *t* = −5.53 in the full sample, −4.42 in training and −3.25 in test.

Taking the same predictor at the lag a decision could use collapses it. Table 1 reports both specifications side by side.

*Table 1. The Same Specification at Two Information Sets*

| Information set | Full sample | Training era | Test era |
|---|---|---|---|
| Share known contemporaneously | −5.53 | −4.42 | −3.25 |
| Share known only at a two-day lag | −0.32 | −0.30 | −0.13 |

*Source. Authors' computation. Entries are t-statistics on the share coefficient in equation (1), with instrument and date fixed effects and log turnover controlled.*

Nothing differs between the rows of Table 1 except the information set: same panel, same fixed effects, same controls, same clustering. The effect does not weaken, it disappears. That places the finding as a statement about contemporaneous market structure rather than a forecast, and it is the reading the source module states of itself.

### Competition: The Baseline Already Holds the Information

Two applications and the control failed here, against three different kinds of baseline.

The cleanest case is a controlled pair. Two market-level density forecasts share their significance evidence and differ in one respect: the baseline density the flow tilt is asked to improve. The underlying screen, equation (3) applied to the flow variable of equation (2), gives *t* = −4.83 in the full sample, −2.53 in training and −2.51 in test, clearing both legs of its threshold.

*Table 2. The Same Predictor Scored Against Two Baselines*

| Measure | Exponentially weighted baseline | Asymmetric GARCH baseline |
|---|---|---|
| Score differential, full sample (threshold ≤ −2.0) | −2.35 (pass) | −0.95 (fail) |
| Mean score advantage ×100, training | −0.0019 | −0.0008 |
| Mean score advantage ×100, test | −0.0004 | +0.0004 (fail) |
| Coverage test probability, 5% and 1% levels | 0.225 and 0.161 (pass) | Not reached |
| Scored days | 2,739 | 2,235 |

*Source. Authors' computation.*

Against the simpler baseline the tilt clears every leg. Against the better-specified one it fails two of three, and the test-era sign flips, which fails the threshold regardless of magnitude. Incremental value is a property of the pair, not of the predictor. The difference in scored days belongs to the same mechanism: the richer baseline needs a longer burn-in, so it both explains more and is asked to explain fewer days.

The second case is a trivial rule. A walk-forward discrete-time hazard model forecasts whether a flow episode ends today, with yearly refits and censoring handled. Table 3 reports its skill and its decision value.

*Table 3. Forecast Skill and Decision Value of the Episode-End Hazard Model*

| Measure | Model | Age-only baseline |
|---|---|---|
| Area under the curve, pooled | 0.647 | 0.569 |
| Area under the curve, training | 0.637 | Not applicable |
| Area under the curve, test | 0.657 | Not applicable |
| Paired statistic on out-of-window log loss | 14.42 | Not applicable |
| Anticipation gain, test era, basis points per episode | −18 | +16 |
| Paired difference on 402 common episodes, basis points | −31 | Not applicable |

*Source. Authors' computation.*

The skill is real. It beats the duration-only baseline decisively and discriminates better out of sample than in, which rules out memorisation. The decision layer then loses to a three-line rule on the episodes both act upon. A weaker design would have compared each rule against zero separately and declared the model the better of two positives.

Table 3 also carries a finding about the programme rather than the data. Its figures come from a re-execution on the current object. Project records report an area under the curve of 0.797 and a test-era gain of +23 basis points, from a run predating an audit that rebuilt the underlying states; the stage was never re-executed afterwards. Both verdicts survive, but the skill is a third smaller than recorded and the decision result is worse, having moved from positive to negative.

The control is a five-day foreign index return predicting next-week realised volatility of the domestic index. It is not drawn from the dataset under study. Its screen gives *t* = −3.86, −2.91 and −4.46, among the most era-stable coefficients in the programme and stronger out of sample than in. The engine built on it fails, with a score differential of +1.58 against a threshold of −2.0, the positive sign meaning the tilted engine scores worse than its base. A predictor from outside the record, more era-stable than anything the record produced, died at the same constraint and by the same mechanism.

### Detectability: Effects Below the Resolution of the Metric

Two applications survived competition and failed because the quantity a user would consult cannot see them.

A participant-composition feature block produces a top-minus-bottom quintile spread of 74.8 basis points, at *t* = 2.86 on non-overlapping episodes. Its incremental information coefficient over conventional flow features is 0.0012 against a threshold of 0.005, at *t* = 0.67 against a threshold of 2. Both are computed on the same block and the same run. The extremes carry a spread; the average carries nothing. The coefficient a user would consult before adding the block to an existing model weights the half that is empty.

The second case tests a mechanism rather than a predictor. Episode labels cluster far beyond a within-instrument shuffled null that preserves each instrument's label count and destroys only temporal adjacency: observed mean runs of 3.61 days against a null of 1.40 in training, and 4.05 against 1.53 in test, both at *p* = 0.005. The labels are real and the clustering replicates. Table 4 reports what they contribute to the density built on them.

*Table 4. Ablation of the Conditioning Mechanism in the Stock-Day Density*

| Stratum | Test statistic | Probability | Score effect |
|---|---|---|---|
| Pre-registered primary stratum, uncertain label identity | −0.82 | 0.41 | −0.00004 |
| Full panel | +2.90 | 0.0037 | +0.00005 |

*Source. Authors' computation. Positive statistics favour the conditioned system.*

In the stratum the pre-registration nominated, where the mechanism should bind hardest, the conditioning is indistinguishable from its own removal. On the full panel it is statistically favoured, by five parts in a hundred thousand of a score whose level is about 0.55, or roughly one part in eleven thousand. Comparison against an external benchmark establishes that a system is better; only ablation establishes why, and the answer here is the empirical shape of the standardised distribution rather than the flow record the system is named for.

### Execution: The Edge and Its Cost Are the Same Order

A mechanical long-short book built on the concentration measure records a gross ratio of 1.36 in training and 1.51 in test, the highest of eight books in the test era. The book carries no fitted model, so what is priced here is the data rather than an architecture.

Its breakeven one-way cost, computed by equation (4), is 7.33 basis points in training and 7.41 in test. A realistic institutional one-way cost of 15 basis points is roughly double that, and at that cost the same book records −1.42 and −1.53. The two figures answer different questions. The breakeven is a property of the strategy and transfers to any reader; the net ratio is a property of the cost assumption and transfers to nobody whose costs differ. Reporting the breakeven also makes the failure legible: the edge and the cost of capturing it are the same order of magnitude, and the cost is the larger.

### What Survived

One application cleared every constraint: the record aggregated to market level, entering a Student-t density at a two-day reporting lag against a baseline that does not already contain it, as reported in the first column of Table 2. Two features make the pass credible. It is the only engine in its family unchanged under two later estimation corrections that widened its three siblings from narrow misses into clear failures. And its conditions are the ones the six failures violate: aggregation to a level where the measurement is stable, availability at a deployable lag, a magnitude the consuming metric can resolve, and a baseline that has not already absorbed it.

## MANAGERIAL AND PRACTICAL IMPLICATIONS

Six applications of one record failed at four different constraints. No sentence about whether this data carries information is well posed without naming the use, and three practical consequences follow for anyone evaluating an alternative dataset.

*Establish the Binding Constraint Before Establishing Significance.* The significance question was answered affirmatively for all seven applications here and settled nothing. The diagnostics that did settle matters are available earlier and cost less. Re-running a regression at a deployable lag takes one line and determines whether a finding is a forecast at all. Scoring against a better-specified baseline rather than a weak one costs one additional model. Comparing against a trivial rule, paired on the events both act upon, costs a join. Ablating a mechanism costs one re-run of machinery already written. Each of these removed an application that no amount of additional data or model capacity would have touched.

*Name the Baseline, Not the Benchmark.* A vendor evaluation that scores a dataset against a weak alternative is measuring the alternative. Table 2 shows the same tilt passing against one baseline and failing against another, with nothing about the data changed between them. Buyers of alternative data should specify the production baseline the dataset must improve, and should treat a demonstration against anything weaker as uninformative.

*Aggregate to Where the Measurement Is Stable.* The single surviving use is the most aggregated one. The stock-day applications failed at availability, at detectability and at cost; the market-level application cleared every constraint. This is consistent with the identifier constraint documented in the methodology, under which entity structure is measurable within a day and not across a month, and it suggests that the frequency at which an alternative dataset is reported need not be the frequency at which it is informative.

*Implications for the Indian Market.* Three readings apply to participants in this market in particular. First, the surviving application is a market-level risk forecast, which is where a domestic risk manager or an exchange monitoring aggregate exposure would use it, rather than a stock-selection input. Second, the stock-day results place foreign institutional composition as contemporaneous market structure rather than as a forecast, which is consistent with the informational-disadvantage reading Raizada and Nawn (2025) report for foreign investor trades in this market. Third, the breakeven costs of 7.33 and 7.41 basis points sit below plausible institutional execution costs here, so the concentration measure is usable as free information by an agent whose trading costs are already committed, and not as a standalone strategy. As Maheshwari and Naik (2026) document a shift from foreign to domestic institutional dominance in this market, the aggregate flow variable that survived here should be re-estimated on domestic institutional flow before it is relied upon in future periods.

For risk practice specifically, the ablation in Table 4 carries a further implication. A risk model that beats an external benchmark has established that it is better, not why. Where a system is named for a conditioning mechanism, that mechanism should be ablated and the resulting effect size reported. The practice is routine in machine learning and close to absent from risk-model validation, where comparison is almost always against an outside alternative rather than against the same system with its own thesis removed.

## CONCLUSION

A fourteen-year record of foreign institutional flow was applied at seven points in a modelling stack and subjected to six ascending constraints. Every application cleared conventional inference and replicated on an out-of-sample era covering a substantially different cross-section. Six then failed to reach a use.

One failed because the effect was not knowable at the lag a decision requires. Two failed because the baseline they would improve already contained the information, and one of those two is the same predictor that passes against a simpler baseline, which shows that incremental value is a property of the pair rather than of the data. Two failed because the effect lay below the resolution of the metric a user would consult, one of them the conditioning mechanism the system carrying it is named for. One failed because its breakeven cost of 7.41 basis points is roughly half a realistic institutional cost.

One survived, with conditions that are the ones the six failures violate. A control predictor drawn from outside the dataset, more era-stable than anything the record produced, failed at the same constraint and by the same mechanism, which places the difficulty in the transition from statistical evidence to institutional use rather than in this data.

Applying the verification protocol to the source programme also located a metrics table quoted from a stale copy, two pre-registered gating regressions with no producing stage, and one stage whose reported effect size was a third larger than its current object supports. None changed a verdict; all changed a figure that had already been written down.

## LIMITATIONS OF THE STUDY AND SCOPE FOR FURTHER RESEARCH

The ordering of the constraints is argued from logical dependence, since a failure at a lower constraint invalidates those above it by construction. The distribution of failures across constraints, however, is a fact about seven applications of one record in one market, and nothing here establishes that it generalises. The mechanisms are offered as portable; the magnitudes are not. A single external control shows that the attrition is not peculiar to flow data, but cannot establish the shape of attrition for datasets in general.

Six constraints were used because six were encountered. Regulatory limits, capacity, and post-publication decay of the kind Pénasse (2022) models are real frictions that this programme had no occasion to impose. Availability and execution are each supported by one application, which is an illustration rather than a distribution.

Four instruments that the framework identifies were not applied, and each carries a different exposure.

The episode design has heterogeneous treatment timing, which is the setting in which two-way fixed effects is biased, and the corrections compared by Aguilar-Loyo (2025) and developed by de Chaisemartin and D'Haultfœuille (2026) were not used. The direction of that bias depends on how already-treated units are weighted and cannot be signed in advance. The exposure is bounded, however: episode-level estimation enters the attribution application reported in Table 4 and does not enter the density or portfolio applications, so the surviving result is unaffected.

Eight strategy books and four engine variants were examined without a correction pricing that search, of the kind Arian et al. (2024) evaluate. Here the exposure is asymmetric and points the wrong way. A multiplicity correction raises the bar, which makes a failure more likely rather than less, so it cannot rescue the six applications that failed. It can only threaten the one that passed. That the surviving application cleared a three-leg threshold fixed before its results existed, and was unchanged under two later estimation corrections that turned its three siblings from narrow misses into clear failures, is the available defence, and it is weaker than a formal correction would be.

Expected shortfall was assessed through a coverage ratio, which is a comparison rather than a test, so a direct backtest could reject where the ratio appeared satisfactory. The distributional claims concern the tail while the score used weights the centre; applying the tail-calibration framework of Allen et al. (2025) or the score decompositions of Arnold et al. (2024) could move the comparison in either direction, and given that the engine's advantage is concentrated in the tail it may widen rather than narrow. Neither of these bears on the conclusion, which concerns the flow record rather than the engine's own calibration.

Closing these is the natural extension of this work, and the second is the one to close first.

The underlying transaction data is proprietary and cannot be redistributed. Derived artifacts are retained, and every reported figure is bound to a specific artifact, a locator within it and a recorded digest, so a reader with access can check any figure without reading the analysis code. That access is not general. Reproducibility is also machine-local: every figure regenerates from a recorded command, which establishes that the reported numbers descend from computations the authors ran, not that a different environment would produce identical figures.

## Author's Contribution

The author conceived the research design, assembled and validated the dataset, implemented the modelling stack and the verification protocol, conducted all analyses, and wrote the manuscript.

## Conflict of Interest

The author certifies that there is no conflict of interest with any financial or non-financial interest in the subject matter discussed in this manuscript.

## Funding Acknowledgement

The author received no financial support for the research, authorship, or publication of this article.

## REFERENCES

Aguilar-Loyo, J. (2025). A comparative analysis of two-way fixed effects estimators in staggered treatment designs. Journal of Econometrics, 251, 106059. https://doi.org/10.1016/j.jeconom.2025.106059

Allen, S., Koh, J., Segers, J., & Ziegel, J. (2025). Tail Calibration of Probabilistic Forecasts. Journal of the American Statistical Association, 120(552), 2796-2808. https://doi.org/10.1080/01621459.2025.2506194

Arian, H., Norouzi Mobarekeh, D., & Seco, L. (2024). Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment. Knowledge-Based Systems, 305, 112477. https://doi.org/10.1016/j.knosys.2024.112477

Arnold, S., Walz, E. M., Ziegel, J., & Gneiting, T. (2024). Decompositions of the mean continuous ranked probability score. Electronic Journal of Statistics, 18(2). https://doi.org/10.1214/24-ejs2316

Azevedo, V., Kaiser, G. S., & Mueller, S. (2023). Stock market anomalies and machine learning across the globe. Journal of Asset Management, 24(5), 419-441. https://doi.org/10.1057/s41260-023-00318-z

Bagnara, M. (2024). Asset Pricing and Machine Learning: A critical review. Journal of Economic Surveys, 38(1), 27-56. https://doi.org/10.1111/joes.12532

Bowles, B., Reed, A. V., Ringgenberg, M. C., & Thornock, J. R. (2024). Anomaly Time. The Journal of Finance, 79(5), 3543-3579. https://doi.org/10.1111/jofi.13372

Chai, D., Ali, S., Brosnan, M., & Hasso, T. (2026). Understanding researchers' perceptions and experiences in finance research replication studies: A pre-registered study. Pacific-Basin Finance Journal, 96, 103061. https://doi.org/10.1016/j.pacfin.2026.103061

Chopra, F., Haaland, I., Roth, C., & Stegmann, A. (2023). The Null Result Penalty. The Economic Journal, 134(657), 193-219. https://doi.org/10.1093/ej/uead060

Coyle, D., & Manley, A. (2024). What is the value of data? A review of empirical methods. Journal of Economic Surveys, 38(4), 1317-1337. https://doi.org/10.1111/joes.12585

de Chaisemartin, C., & D’Haultfœuille, X. (2026). Difference-in-Differences Estimators of Intertemporal Treatment Effects. Review of Economics and Statistics, 108(4), 863-880. https://doi.org/10.1162/rest_a_01414

Dey, M., Mishra, S., & De, S. (2024). A Study on How Institutional Investors Respond to Risk, Return and Volatility: Evidence from the Indian Stock Market. Journal of the Knowledge Economy, 15(1), 5072-5093. https://doi.org/10.1007/s13132-023-01718-7

Ding, Y., Kambouroudis, D., & McMillan, D. G. (2025). Forecasting realised volatility using regime-switching models. International Review of Economics & Finance, 101, 104171. https://doi.org/10.1016/j.iref.2025.104171

Ghachem, M., & Ersan, O. (2025). Estimation of the probability of informed trading models via an expectation-conditional maximization algorithm. Financial Innovation, 11(1). https://doi.org/10.1186/s40854-024-00729-w

Gonçalves, S., McCracken, M. W., & Yao, Y. (2025). Bootstrapping out-of-sample predictability tests with real-time data. Journal of Econometrics, 247, 105916. https://doi.org/10.1016/j.jeconom.2024.105916

Hou, K., Xue, C., & Zhang, L. (2020). Replicating Anomalies. The Review of Financial Studies, 33(5), 2019-2133. https://doi.org/10.1093/rfs/hhy131

Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is There a Replication Crisis in Finance? The Journal of Finance, 78(5), 2465-2518. https://doi.org/10.1111/jofi.13249

Kaplanski, G. (2023). The race to exploit anomalies and the cost of slow trading. Journal of Financial Markets, 62, 100754. https://doi.org/10.1016/j.finmar.2022.100754

Li, Z., Liu, L. X., Liu, X., & John Wei, K. C. (2024). Replicating and Digesting Anomalies in the Chinese A-Share Market. Management Science, 70(8), 5066-5090. https://doi.org/10.1287/mnsc.2023.4904

Maheshwari, S., & Naik, D. R. (2026). From FII Dependence to DII Dominance: Behavioral Dynamics and Minskyan Risk in India’s Stock Market. Journal of Risk and Financial Management, 19(5), 315. https://doi.org/10.3390/jrfm19050315

Muravyev, D., Pearson, N. D., & Pollet, J. M. (2025). Anomalies and Their Short-Sale Costs. The Journal of Finance, 80(6), 3639-3694. https://doi.org/10.1111/jofi.13501

Pénasse, J. (2022). Understanding Alpha Decay. Management Science, 68(5), 3966-3973. https://doi.org/10.1287/mnsc.2022.4353

Raizada, G., & Nawn, S. (2025). Trade informativeness of foreign investors in India. Journal of Asset Management, 26(1), 83-90. https://doi.org/10.1057/s41260-024-00387-8

Sak, H., Huang, T., & Chng, M. T. (2024). Exploring the factor zoo with a machine-learning portfolio. International Review of Financial Analysis, 96, 103599. https://doi.org/10.1016/j.irfa.2024.103599

Sun, Y., Liu, L., Xu, Y., Zeng, X., Shi, Y., Hu, H., Jiang, J., & Abraham, A. (2024). Alternative data in finance and business: emerging applications and theory analysis (review). Financial Innovation, 10(1). https://doi.org/10.1186/s40854-024-00652-0

Thapa, C., Neupane, B., Shrestha, C., & Bhattarai, N. P. (2026). Policy information uncertainty and foreign institutional investors trading behavior: evidence from India. Review of Quantitative Finance and Accounting, 67(1), 45-79. https://doi.org/10.1007/s11156-025-01448-8

Waghmare, K., & Ziegel, J. (2026). Proper Scoring Rules for Estimation and Forecast Evaluation. Annual Review of Statistics and Its Application, 13(1), 271-296. https://doi.org/10.1146/annurev-statistics-042424-050626

