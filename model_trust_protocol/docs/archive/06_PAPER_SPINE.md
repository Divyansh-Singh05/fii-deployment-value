# 06 · Paper Spine

**Structure only.** Built against the matrix in `08_FRAMEWORK.md`, not against
the nine questions — those are now the L5 row.

Title candidates:
- *Instruments That Cannot Fail: a validation framework for quantitative finance*
- *Could This Test Have Said No?*
- *Six Layers, Three Disciplines: engineering rigour in financial modelling*

---

## §1 Introduction

Opens on a validated research programme: a regime model replicating
digit-for-digit out of sample, an econometric battery surviving eleven
alternative specifications, a gated backtest engine, and a risk engine beating
the industry benchmark on every metric of the standard battery over 578,481
out-of-sample observations. By every convention of the field, this is validated
work. Then poses the paper's question of each instrument in turn — *could it
have returned a different answer?* — and states that eleven could not.

**E1** — the four models and their headline validation claims, as published.

## §2 Related work and the honest calibration

Three adjacent literatures, and what each does not cover: multiplicity and
selection (Harvey–Liu–Zhu; White; Hansen); backtest overfitting (Bailey–López de
Prado; PBO; deflated Sharpe); risk-model backtesting (Kupiec; Christoffersen;
Berkowitz; the elicitability / ES-backtestability strand; proper scoring rules).
All three price *the strength of evidence*. None asks whether the instrument
producing the evidence was capable of producing the opposite. States plainly
that no statistic in the paper is new.

## §3 The framework

Six layers × three disciplines. The layers argued as properties of what
modelling is; the disciplines as what each layer is attacked with. Ordering
argued from **void scope** — a lower-layer failure invalidates every layer
above it — not from frequency.

**E2** — the matrix, populated.

## §4 D3 · Instrument validation — the paper's core

The column the literature has no name for. Two halves:

**§4.1 Negative capability — could the test have said no?** Eleven instruments
across three model types, each shown incapable of returning a different answer,
each with the corrected result. Ordered by how different the corrected answer
was, not by layer.

**§4.2 Positive control — does the test fire when it should?** The cheat-alpha
gate and the constructed cost book. Argued as the mirror discipline, and the
asymmetry noted: practitioners occasionally check for spurious firing and almost
never check for capability to fire.

**E3** — the eleven instruments, with the corrected result for each.
**E4** — the cheat-alpha gate: Sharpe +211.5 at lag 0, −0.5 at lag 1.
**E5** — the synthetic power table (four labelled PIT samples, two null
constructions) → **D1**.

## §5 L1–L2 · Measurement and model class

The layer most papers skip. The autocorrelation feasibility diagnostic as a
pre-fit test of whether the model class can see the axis at all; the dissection
test separating *absent from the data* from *invisible to the class*; the
structural challenger that established the limitation's scope; the mechanical
twin and its κ; the pre-registered escalation bar that declined a richer model
class.

**E6** — feature autocorrelation before and after the transformation, with the
resulting change in what the model could represent.
**E7** — mechanical twin agreement, and what the fitted model actually buys.

## §6 L3 · Validating a fitted model with no ground truth

The five-way decomposition of "accuracy" when no oracle exists. The three
replication rows excluded as incapable of failing, and the one that survived.
The prior's contribution isolated. The permutation test whose null holds
everything constant but the claim. Reproducibility as a measured quantity.
Face validity checked before outcome data existed.

**E8** — the five sub-claims with their tests and verdicts.
**E9** — the permutation test: observed run length against the shuffled null,
both eras.

## §7 L4 · Causal claims and constructed adversaries

Difference-in-differences adopted because the placebo fired. Placebo validity
tested and the placebo replaced. The hindsight-advantaged control. The
free-public-data head-to-head. The deliberate bad control as a floor. The
horizon ladder as a *signature* test discriminating between mechanisms. Power
bounds on nulls. "Significant versus not significant is not a test."

**E10** — the adversary ladder: what each constructed competitor removed.

## §8 L5 · Forecast validation — the nine questions

The deepest worked layer, compressed from `01_PROTOCOL.md`. Leads with the
three findings that no additional data could have produced: evaluation-set
look-ahead (0.30 vs −0.04 on identical forecasts), effective-evidence
accounting (a third of five-day tails withdrawn), and the mechanism ablation.

**E11** — the risk scorecard beside the protocol's return.
**E12** — the truncation audit, six gates, exact zeros → **D3**.
**E13** — mechanism ablation with its powered detection bound.

## §9 L6 · Decision validation

Execution timing as an equation. Baselines frozen before twins. Paired
block-bootstrap verdict rules. Gross and net diagnosed as separate layers.
Breakeven as the headline. The three-line trivial rule that beat the fitted
model paired on common events. Limits-to-arbitrage as the legitimate reading of
a negative backtest.

**E14** — breakeven grid.
**E15** — model versus trivial rule, paired on common events.

## §10 What survives

The discriminating half, and the paper's defence against nihilism. Results
built on instruments that *could* have failed: the episode-clustering
permutation test, the byte-identical reproducibility measurement, the
mechanical twin's κ, the cheat-alpha-gated breakevens, the fat-tail result, the
truncation audit's exact zeros. Names the most fragile survivor before a
referee does.

## §11 The framework applied to itself

Two parts, and the second is the more persuasive.

**§11.1 The gap list.** Ten instruments the framework says this programme should
have used and did not — staggered-adoption DiD estimators, selection accounting
(PBO, deflated Sharpe, SPA, MCS), proper ES backtests and joint (VaR, ES)
scoring, threshold-weighted CRPS with a Murphy diagram, PIT independence and the
Berkowitz tail test, the nested-model correction to Diebold–Mariano, purged CV
with embargo, a hidden semi-Markov formulation, ARI in place of κ, and decision
curve analysis. Each stated with what it would have changed.

**§11.2 Limitations of the paper.** One research programme; instruments and
magnitudes are domain-specific; the matrix's completeness is argued, not proved;
the classification of instruments as capable or incapable of failing is
reasoning, not measurement.

**E16** — the gap list, with the instrument, the question it answers, and the
claim it would have strengthened or weakened.

## §12 Conclusion

The asymmetry: in quantitative finance the productive effort is not spent
establishing a result but constructing the tests that could destroy it — and
then, one level up, establishing that those tests were capable of doing so. A
result that has survived instruments known to be capable of killing it is the
only kind that can be used.

## Appendices

- **A** The full technique inventory, by layer (59 entries)
- **B** The nine forecast questions in full
- **C** The synthetic size/power harness — code and results
- **D** The cheat-alpha harness — code and results
- **E** Reproduction contract
- **F** Assumption register with bias directions

---

## Exhibit inventory

| ID | Exhibit | Source | Status |
|---|---|---|---|
| E1 | Four models and their published validation claims | thesis + `C7_RISK_REPORT.md` | assemble |
| E2 | The matrix, populated | `08_FRAMEWORK.md` | drafted |
| E3 | Eleven instruments incapable of failing | `08_FRAMEWORK.md` | drafted |
| E4 | Cheat-alpha gate | `05_backtesting.md`, `engine_gates.py` | exists |
| E5 | Synthetic PIT power table | `LIMITATIONS.md` R2-1 | **regenerate as D1** |
| E6 | Feature autocorrelation before/after | thesis §5.4 | exists |
| E7 | Mechanical twin agreement | `module13a_backbone_ablation.py` | exists |
| E8 | Five sub-claims of "accuracy" | thesis §7.1 | exists |
| E9 | Permutation test, both eras | `model_descriptives` | exists |
| E10 | Adversary ladder | thesis §13–§15 | assemble |
| E11 | Risk scorecard vs protocol return | `00_FRAMING.md` | drafted |
| E12 | Truncation audit | `s13_lookahead_audit.py` | exists |
| E13 | Mechanism ablation with power bound | C6 | exists |
| E14 | Breakeven grid | `outputs/metrics/backtest_metrics.csv` | exists |
| E15 | Model vs trivial rule, paired | module 16D | exists |
