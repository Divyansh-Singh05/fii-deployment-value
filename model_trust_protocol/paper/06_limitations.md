# 6 · Limitations

The framework is applied to this paper. Section 6.1 lists the instruments it
identifies as ones this programme should have used and did not; §6.2 states the
limitations of the design itself.

---

## 6.1 Instruments the framework identifies as missing

Ordered by how much a referee should care.

**Staggered-adoption difference-in-differences.** The episode design has
heterogeneous treatment timing — episodes begin and end at different dates for
each instrument — with plausibly heterogeneous effects. That is the setting in
which two-way fixed effects is known to be biased, and the corrections
(Goodman-Bacon decomposition, Callaway–Sant'Anna, Sun–Abraham,
de Chaisemartin–D'Haultfœuille) were not applied. This is the most substantive
methodological gap in the programme. It bears on the episode-level results
rather than on the density and portfolio applications, but it bears on them
directly.

**Selection accounting.** Eight strategy books, four market-engine variants and
several model classes were tried. Nothing prices that search. The probability of
backtest overfitting, the deflated Sharpe ratio, White's Reality Check, Hansen's
superior predictive ability test and the model confidence set are all applicable
and none was computed. The pre-registration limits the damage — bars were fixed
in advance and misses were reported rather than re-specified — but a
pre-registration is not a multiplicity correction.

**Proper expected-shortfall backtests.** Expected shortfall is a reported risk
measure in the stock-day application and was validated only through a coverage
ratio, which is not a test. Acerbi–Székely and Du–Escanciano are the direct
instruments; and because expected shortfall is not elicitable alone, a
comparison of two systems on it should use the jointly elicitable (VaR, ES)
scoring function of Fissler–Ziegel rather than a ratio.

**Tail-weighted scoring.** The distributional claims concern the tail; CRPS is
a whole-distribution score dominated by the centre. A threshold-weighted CRPS
and a Murphy diagram — which shows whether a ranking survives across the whole
class of consistent scoring functions — would settle whether the ordering
depends on the loss chosen. Given how small some of the margins here are, this
is not a formality.

**PIT independence.** Distributional calibration was assessed for uniformity.
The standard also requires the probability integral transforms to be serially
independent, and the observed clustering of violations suggests they are not.
The Berkowitz tail test would additionally concentrate power where the claim is.

**Nested-model comparison.** Diebold–Mariano is invalid under the null for
nested models, where the loss differential degenerates. Several comparisons here
are nested by construction; Clark–West is the correction, and which of our
comparisons require it has not been audited.

**Purged cross-validation with embargo.** Overlapping labels are pervasive —
twenty-day forward returns, h-day forward windows — and only the frozen split
and the walk-forward vintages protect against the resulting leakage. Purged
k-fold would be the standard additional protection.

**A hidden semi-Markov formulation.** The state model's geometric dwell-time
implication was a known weakness, and episode *ends* were economically central
enough to warrant a separate hazard model. A hidden semi-Markov model
parameterises dwell time natively and would have unified two models into one.

**Partition agreement.** Agreement between the fitted state model and a
rule-based backbone was reported as Cohen's κ, which requires an alignment
between label sets. The adjusted Rand index or normalised mutual information
would be the appropriate instrument for comparing partitions whose indices are
arbitrary.

**Decision curve analysis.** The gap between forecast skill and decision value
(§4.3.2) was established empirically. Net-benefit analysis across a range of
decision thresholds is the formal instrument for it.

**Citation.** The stock-day engine is filtered historical simulation — a
conditional volatility model with empirically resampled standardised residuals —
which is a named method with a canonical reference. It is cited as such.

## 6.2 Limitations of this design

**One dataset, one market.** The ordering of the ladder is argued from void
scope: a failure at a lower level invalidates the levels above it by
construction, so the order does not depend on how often each level binds. But
the *distribution* of failures across levels is a fact about seven applications
of one record in one market, and nothing here establishes that it generalises.
The mechanisms are what we offer as portable; the magnitudes are not.

**The record is proprietary.** The underlying transaction data cannot be
redistributed. The derived artifacts are retained and every reported number is
bound to a specific artifact, a locator within it and a recorded digest, so a
reader with access can check any figure without reading the analysis code — but
that access is not general.

**One control.** A3 establishes that the attrition is not peculiar to flow data.
One control cannot establish the shape of the attrition for datasets in general.

**The friction levels are not exhaustive.** Six were used because six were
encountered. Regulatory constraint, capacity, and the decay of a signal after
publication are all real frictions that this programme had no occasion to
impose, and their absence from the ladder reflects the case rather than the
concept.

**Two applications rest on a single measured instance each.** A1 (availability)
and A5 (execution) are each supported by one application at that level. An
instance is an illustration; it is not a distribution.

**We both generated and classified the failures.** The placement of each
application on the ladder is a judgement about which constraint bound first, made
by the people who ran the applications. Where an application could plausibly be
placed at more than one level, we placed it at the lowest — the earliest point
at which it demonstrably fails — but the judgement is ours and is not
independently adjudicated.

**Reproducibility is machine-local.** Every figure regenerates from this
repository by a recorded command, and the digests of the resulting artifacts are
committed. That establishes that our numbers descend from computations we ran;
it does not establish that a different machine, with different library versions,
would produce identical figures. The one quantity we re-derived from
specification rather than inheriting — the aggregate-flow screen — differs from
the inherited figure in the second decimal, and we report the sensitivity that
explains it (§4.3.1, and the unobserved-flow reading documented with it) rather
than reconciling it away.
