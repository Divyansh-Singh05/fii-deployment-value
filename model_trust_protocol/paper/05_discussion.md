# 5 · Discussion

---

## 5.1 The binding constraint is a property of the use, not of the data

Six applications of one record fail, at four different levels. That distribution
is the paper's central observation, and it is not available to a design that
holds the friction fixed.

The replication literature is breadth-first by construction. Hou, Xue and Zhang
price multiple testing and portfolio construction across 452 anomalies;
Muravyev, Pearson and Pollet price short-sale costs across 162. Both establish
how *often* one friction is decisive across a population of predictors. Neither
can observe that the same information, differently aggregated and differently
scored, meets a different binding constraint — because in a breadth design the
constraint is the constant.

Read across our seven applications, the record is:

- **unusable at stock-day frequency for a decision**, because the measurement
  that carries the information is contemporaneous with the outcome (§4.2);
- **redundant against a well-specified variance model**, but *not* against a
  simpler one — the same tilt passes and fails depending only on its baseline
  (§4.3.1);
- **non-incremental to a trivial rule** at the decision layer, despite genuine
  and out-of-sample-replicating forecast skill (§4.3.2);
- **present in the extremes and absent on the average**, so the summary
  statistic decides the answer (§4.4);
- **below the resolution of a distributional score** where its own mechanism is
  ablated (§4.4);
- **economically real and unharvestable**, at a breakeven roughly half a
  realistic cost (§4.5);
- **and worth about what it claims**, once aggregated to market level against a
  baseline that does not already contain it (§4.6).

No single sentence about "whether FII flow carries information" is adequate to
that list. The question is not well posed without naming the use.

## 5.2 What the control establishes, and what it does not

A3 is drawn from outside the dataset entirely, is among the most era-stable
coefficients the programme produced, and dies at the same level and by the same
mechanism as two of the flow applications. That rules out a reading in which
this particular record is unusually weak: the friction is in the transition, not
in the data.

It does not establish the converse. One control cannot show that every dataset
would attrite the same way, and we do not claim it. What it supports is the
narrower and sufficient claim that the attrition documented here is not a
property peculiar to institutional flow.

## 5.3 Statistical significance is a filter aimed where little fails

The most uncomfortable number in this paper is a zero: **no application was lost
at existence or replication.** Every effect cleared conventional inference and
held on an era covering a substantially different cross-section.

This is precisely the population at which the replication literature's filter
stops. Hou, Xue and Zhang find that 65% of published anomalies cannot clear
|t| ≥ 1.96 and 82% cannot clear a multiple-testing hurdle. Our applications are
in the surviving minority — and six of seven still fail to reach a use.

The two literatures are therefore complementary rather than competing, and
together they suggest an unflattering arithmetic. If a large majority of
candidate findings fail the statistical filter, and a large majority of the
survivors fail the deployment filters, then the conditional probability that a
published, replicated finding is *usable* is small — and almost all of the
field's methodological attention is directed at the first filter.

We do not claim to have estimated that probability. Seven applications of one
dataset cannot. What the design does establish is that the second set of
filters is not a formality: applied in order, to findings that all passed the
first, it removed six of seven.

## 5.4 The order matters, and it is cheap in the right direction

The ladder is ordered by institutional reality rather than statistical severity,
and one practical consequence is worth stating.

The cheapest checks are near the top. Re-running a regression at a deployable
lag (§4.2) costs one line and settles whether a finding is a forecast at all.
Scoring a predictor against a better-specified baseline rather than a weak one
(§4.3.1) costs one additional model. Comparing against a trivial rule *paired on
common events* (§4.3.2) costs a join. Ablating a mechanism (§4.4) costs one
re-run of machinery already written. None requires new data, a larger sample, or
a stronger model — and each of them killed an application here that a
significance test could not touch.

The expensive part of empirical finance is generally taken to be the
identification and the inference. On this evidence the decisive part is neither.

## 5.5 A note on the ablation

Of the six failures, one is different in kind. A1 through A6 report that an
effect is too late, too small, too redundant or too costly. A7 reports that the
mechanism the system is named for contributes nothing to it.

That distinction matters for how a result should be stated. Our engine beats a
Gaussian benchmark decisively in the deep tail; a model-centred reading would
report that as a success for flow-conditioned risk modelling. The ablation says
otherwise: the identical machinery with the conditioning removed is
statistically indistinguishable from it in the stratum where the mechanism
should bind hardest, and differs by one part in eleven thousand on the full
panel. The defensible claim is about the *benchmark* — that a Gaussian is the
wrong distributional assumption for this market — and not about the flow record.

Comparison against an external benchmark establishes that a system is better.
Only ablation establishes why. The practice is standard in machine learning and
close to absent from risk-model validation, where comparison is almost always
against an outside alternative and almost never against the same system with its
own thesis removed. On this evidence that absence is expensive: no amount of
additional out-of-sample data would have revealed what one ablation did.

## 5.6 Implications for alternative data

The record analysed here is a strong case: transaction-level, entity-attributed,
of a class that does not normally leave regulators, and covering fourteen years.
Whatever it is worth is close to an upper bound on what a commercial equivalent
would be worth on the same market.

Three practical readings follow.

**Ask which friction binds before asking whether the signal is real.** The
significance question was answered affirmatively for every application here and
determined nothing. The useful diagnostic — deployable lag, baseline
redundancy, metric resolution, breakeven cost — is available before a
significance test is run, and is cheaper.

**Name the baseline, not the benchmark.** A vendor evaluation that scores a
dataset against a weak alternative is measuring the alternative. The same tilt
in §4.3.1 passes against an EWMA base and fails against a GJR base; nothing
about the data changed.

**Aggregate to where the measurement is stable.** The one surviving use is the
most aggregated one. The stock-day applications fail at availability, at
detectability, and at cost; the market-level one clears every level. That is
consistent with the constraint documented in §2.2 — identifiers are re-minted
monthly, so entity structure is measurable within a day and not across one —
and it suggests that the frequency at which an alternative dataset is *reported*
is not necessarily the frequency at which it is *informative*.
