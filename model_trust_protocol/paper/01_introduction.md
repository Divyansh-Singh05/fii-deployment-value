# 1 · Introduction

*Draft. marks numerals not yet traced to a primary artifact.*

---

## 1.1 The question the replication literature stops at

Hou, Xue and Zhang (2020) re-tested 452 published stock-return anomalies and
found that 65% could not clear a conventional |t| ≥ 1.96 hurdle once microcaps
were mitigated through NYSE breakpoints and value weighting. Raising the bar to
a multiple-testing threshold of 2.78 lifts the failure rate to 82.1%. In the
trading-frictions category the figure is 96%. Harvey (2017) had already argued
that the incentive structure of the field makes precisely this outcome
predictable.

That literature is decisive about one thing and silent about another. It tells
us that most published predictors do not survive re-testing. It does not tell us
what becomes of the eighteen to thirty-five per cent that do.

This paper is about that remainder. We take a single dataset — one whose
statistical content is not in question at any point in what follows — and ask
how far it carries once the tests stop being statistical.

## 1.2 The object of study is the data, not the model

A methodological commitment governs everything below, and it is worth stating
before the results rather than after.

**The dataset is the object of study. Every model applied to it is an
instrument.** A cross-sectional regression, a hidden Markov regime model, a
predictive density engine, a book of long/short strategies — none of these is
proposed as a contribution, and none is evaluated on its own merits. Each
exists to impose one particular kind of real-world constraint on the same
underlying record, so that the record can be asked a question it could not be
asked otherwise. When we report that a regime model's conditioning contributes
nothing to a risk density, that is a statement about the flow data, not about
hidden Markov models.

This commitment is not stylistic. It determines what counts as a result. Under
a model-centred reading, an ablation showing that the conditioning mechanism is
indistinguishable from its own removal would be a failure of the architecture,
inviting a better architecture. Under a data-centred reading it is a
measurement: at that layer, on that metric, the record carries no incremental
information, and no architecture will change that.

## 1.3 The friction ladder

We organise the analysis around six ascending levels of constraint. A finding
must clear each to reach the next, so the level at which an application stops is
a complete statement of what the data was worth in that use.

| | Level | The question |
|---|---|---|
| L0 | existence | Is the effect present under conventional inference? |
| L1 | replication | Does it hold on an era the design never touched? |
| L2 | availability | Is it knowable when the decision must be made? |
| L3 | competition | Does it beat the baseline it would be deployed against? |
| L4 | detectability | Is it large enough to register in the metric the decision consumes? |
| L5 | execution | Does the edge exceed the cost of capturing it? |

The ordering is by institutional reality rather than statistical severity. L0
and L1 are questions about the sample; L2 through L5 are questions about
deployment, and each is a constraint an institution actually faces rather than
a robustness check a referee might request.

Two features of the ladder matter for what it can establish. First, it is
**ordered**, so attrition is cumulative and interpretable: an application that
dies at L5 has, by construction, survived four earlier filters that killed
others. Second, the levels are **not properties of the market** but of the use
to which the data is put. The same record, aggregated differently and scored
against a different baseline, meets a different binding constraint. That the
binding constraint varies across applications of one dataset is, we will argue,
the paper's central finding.

## 1.4 What we find

Seven applications of the record, plus one non-flow control run through
identical machinery.

**Nothing is lost at L0 or L1.** Every application produced an effect
that was statistically present and replicated on a frozen out-of-sample era
covering a substantially different universe of instruments. On the standard by
which the replication literature judges published work, the entire programme
passes.

All attrition occurs above L1. One application fails at availability: the
effect is measured contemporaneously and is absent at the lag a decision could
act on. Two fail at competition: a well-specified conditional-variance baseline
already contains the information, and in one case a three-line rule outperforms
the fitted model head to head on the events both act upon. Two fail at
detectability: the effect is present in the extremes and absent on average, or
present in principle and below the resolution of a distributional score. One
fails at execution: its breakeven one-way cost is roughly half a realistic
institutional cost. One survives every level, and we state what it is worth in
capital rather than in score units.

The control fails at the same level as the flow applications. Since it is
drawn from outside the dataset — and is among the most era-stable coefficients
the programme measured — this indicates the friction is a property of the
transition from statistical evidence to institutional use, not a peculiarity of
this record.

## 1.5 Contribution, and its boundary

Existing work in this tradition is **breadth-first**: many predictors, one
friction. Hou, Xue and Zhang price multiple testing and portfolio construction
across 452 anomalies. Muravyev, Pearson and Pollet (2025) price short-sale
costs across 162, finding an average long-short return of +0.14% per month
before borrow fees and −0.01% after — and, tellingly, no profitability even
before fees once the highest-fee 12% of stock-dates are removed.

This paper is **depth-first**: one dataset, six frictions, ordered. The two
designs answer different questions and neither substitutes for the other. What
depth buys is the observation that the binding constraint differs across uses
of the same data, which a breadth design cannot see because it holds the
friction fixed.

We claim three things.

1. **A boundary, not a verdict.** Because one application survives, the result
   is a map of where the record's value stops rather than a finding that it has
   none. This is a stronger and more falsifiable claim than a null result, and
   it is the reason the surviving application is reported at equal length to the
   failures.

2. **The friction that binds is a property of the use.** Six of seven
   applications fail, at four different levels. No single constraint explains
   the pattern.

3. **A provenance protocol.** Every number reported is bound to the artifact
   that produced it, to a digest of that artifact, and to an extractor that
   reads the value back out of it. A paper whose subject is the gap between a
   statistic and its evidentiary weight should be able to say which of its own
   numerals have been traced. We report that count.

We are explicit about what this design cannot establish. One dataset in one
market cannot show that the ladder's ordering is general, and we do not claim
it. The magnitudes are specific; the mechanisms are what we offer as portable.

## 1.6 Roadmap

Section 2 describes the record, its provenance, and the constraint that
governs every measurement built on it. Section 3 sets out the ladder, the
instrument used at each level, the pre-registration discipline, and the
verification protocol. Section 4 reports the eight applications in ladder order.
Section 5 discusses what the pattern implies for the use of alternative data.
Section 6 states the limitations, including a list of instruments the framework
identifies as ones this programme should have used and did not.
