# Abstract

<!-- provenance -->
*Every numeral below is traced to a primary artifact by `convgap verify`, with
the artifact's digest recorded and the value read back out of it. 8/8
applications, 32/32 data-section claims.*
<!-- /provenance -->

---

The replication literature in empirical finance has established that a large
majority of published return predictors fail conventional significance
hurdles when re-tested. Less is known about the minority that pass. This paper
asks what happens to a financial dataset *after* it clears that hurdle, using a
fourteen-year, 25.2-million-trade masked transaction record of foreign
institutional flow in Indian equities as the object of study.

We apply the record at seven points in a modelling stack — a cross-sectional
volatility regression, a latent regime model with a walk-forward hazard layer,
a stock-day predictive density, two market-level density forecasts differing
only in their baseline, a gradient-boosted feature block, and a long/short book
— and subject each application to six
ascending levels of institutional friction: statistical existence, out-of-sample
replication, availability at the decision lag, competition against the baseline
the application would actually be deployed against, detectability in the metric
the decision consumes, and execution cost. Each level's bar was fixed before
its results existed. The models are stress-test environments, not objects of
study; they exist to impose a particular constraint on the same underlying data.

**No application was lost at the first two levels.** Every effect the
record produced was statistically present and replicated on a frozen
out-of-sample era in which only 441 of the two eras' 1,028 instruments are
common. All attrition occurs under deployment friction. One application dies at
availability: a *t*-statistic of −5.53 becomes −0.32 when the predictor is taken
at the lag a decision could use. Two die at competition, one of them a
controlled pair in which the same flow tilt clears its pre-registered gate
against an EWMA baseline (Diebold–Mariano *t* = −2.35) and fails it against a
GJR-GARCH baseline (*t* = −0.95). Two die at detectability, one of them by
ablation: the conditioning mechanism a risk density is named for is
indistinguishable from its own removal in the stratum where it should bind
hardest, and differs by one part in eleven thousand of the score on the full
panel. One dies at execution, with a breakeven cost of 7.4 bp against a
realistic 15 bp. One of seven survives every level.

A non-flow control predictor — among the most era-stable coefficients the
programme measured, at *t* = −3.86 / −2.91 / −4.46 — fails at the same level
and by the same mechanism, indicating that the friction is a property of the
evaluation transition rather than of this dataset.

The contribution is a boundary rather than a verdict. Existing replication work
prices one friction across many predictors; we price six, in order, across one
dataset, and find that *the level at which value is lost is a property of the
use rather than of the data*. We provide the ladder, a per-level accounting for
each application, and a provenance protocol under which every reported number
is bound to the artifact that produced it, to a digest of that artifact, and to
an extractor that reads the value back out of it. Applying that protocol to the
source programme located two pre-registered gating regressions with no
producing stage — both re-derived here — and one stage whose reported effect
size was a third larger than the current object supports, because it was never
re-executed after an audit rebuilt that object.

**Keywords:** alternative data, institutional flow, forecast evaluation,
transaction costs, replication, negative results
