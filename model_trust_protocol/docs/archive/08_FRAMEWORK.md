# 08 · The Framework

The inventory in `07_TECHNIQUE_INVENTORY.md` is 59 techniques applied to three
completely different model types — an unsupervised regime model, a panel
econometric design, and a walk-forward distributional risk engine — over four
years. This document is the weave: what they have in common, and why that
common structure is the paper.

---

## The observation that organises everything

Across all three model types, the same three disciplines recur, and the *same
three failure modes* recur when they are absent.

The techniques do not organise by statistical family. They organise by **what
is being validated** (six layers) crossed with **how it is being validated**
(three disciplines). Every one of the 59 falls into a cell, and the empty cells
are the paper's recommendations.

## The two axes

**Layers — what object is under test.** Each layer validates a different thing,
and a failure at a lower layer voids every layer above it.

| Layer | Object | The question |
|---|---|---|
| **L1 Measurement** | A number | Is this a valid measurement of the thing it names? |
| **L2 Model class** | A family of models | Can this class of model represent what I am asking it to find? |
| **L3 Fitted model** | One estimated object | Does this fit mean the same thing on data it has not seen? |
| **L4 Causal claim** | An estimated effect | Does the effect survive every alternative I can construct? |
| **L5 Forecast** | An out-of-sample number | Could this number have been produced on the date it carries, and is it what it claims? |
| **L6 Decision** | An action | Is the edge worth acting on, and does it beat the trivial rule? |

**Disciplines — how each layer is attacked.**

| Discipline | Statement | What it prevents |
|---|---|---|
| **D1 Pre-commitment** | The verdict rule exists before the number does — including the failure branch and what it would require | Rationalising whatever the model produced; re-specifying after a miss |
| **D2 Adversarial construction** | Build the strongest competitor you can, and hand it advantages you do not have | Winning against a straw benchmark; mistaking "better than nothing" for "better than the alternative" |
| **D3 Instrument validation** | Prove the test can return a negative, and prove it fires on a known positive | Verdicts from instruments that were never capable of producing any other verdict |

---

## The matrix

Every cell populated from the source projects. Bracketed references are
inventory entries.

| | **D1 Pre-commitment** | **D2 Adversarial construction** | **D3 Instrument validation** |
|---|---|---|---|
| **L1 Measurement** | Existence criterion written before the first fit [1.1] | Convergence of three independent threshold methods [3.10] | Coverage gate: unmeasurable must yield missing, not zero [1.3]; liquidity floor separates *unreliable* from *extreme* [1.5] |
| **L2 Model class** | The architectural fork committed to in advance [2.5] | The mechanical twin [2.7]; the structural challenger [2.6]; the flexible-learner challenger [2.8] | **BIC cannot discriminate at n = 2.7M — an instrument incapable of failing** [2.3]; the dissection test separates "absent" from "invisible" [2.4]; the autocorrelation diagnostic tests class feasibility *before* fitting [2.1] |
| **L3 Fitted model** | Pre-registered gates and drift bars before fitting [2.6]; face validity checked before outcome data existed [3.9] | Frozen-parameter decode on a disjoint era and a 45%-different universe [3.2] | **Three replication rows identified as incapable of failing and excluded from evidence** [3.3]; the prior's contribution isolated [3.4]; the permutation test's null holds everything constant but the claim [3.7]; reproducibility measured, not asserted [3.8] |
| **L4 Causal claim** | Bars and failure conditions written together [L4 passim]; the *reading* of a conflict pre-committed [4.16] | Hindsight-advantaged control [4.11]; free-public-data head-to-head [4.14]; deliberate bad control [4.5]; independent construct with no shared inputs [4.15]; incremental ladder [4.14] | Difference-in-differences adopted because vs-zero made the placebo fire [4.1]; **placebo validity itself tested and the placebo replaced** [4.7]; power bounds on every null [4.17]; "significant vs not significant is not a test" [4.13]; effect-sizes-first because p-values are decoration at n = 800k [3.6] |
| **L5 Forecast** | Pre-registrations with binding bars and failure branches; the amendment documented *before* the rerun [5.3] | Persistence as the mandatory comparator [5.2]; the parametric benchmark; **the mechanism ablation** (`01_PROTOCOL.md` Q6) | **The zero-power null** (Q5); a quasi-circular baseline caught and replaced [5.3]; truncation audit with availability equality (Q3); causal stratification (Q3.5); out-of-sample stronger than in-sample as a memorisation check [5.6] |
| **L6 Decision** | Verdict rule pre-registered as paired block-bootstrap ΔSharpe [6.7]; baselines frozen before the twins [6.6] | The three-line trivial rule, paired on common events [6.12]; the cost grid as adversary [6.9] | **The cheat-alpha positive control** [6.1]; exactness against a naive loop [6.2]; cost accounting on a constructed book [6.3] |

---

## The finding the matrix produces

**Column D3 is where the projects' real contribution lives, and it is the column
the literature does not have a name for.**

Reading down it, the same act recurs eleven times on three unrelated model
types: an instrument was examined for whether it could have returned a
different answer.

| Instrument | Why it could not fail | Consequence |
|---|---|---|
| BIC over model order | At n = 2.7M the complexity penalty is negligible; BIC declines monotonically in k | Model order chosen by inspecting solution *structure* instead |
| Census share replication | Within-day ranking pins cross-sectional shares by construction | Excluded from evidence |
| Transition-diagonal replication | A Viterbi decode with a frozen 0.95-sticky prior reports ~0.95 regardless of the data | True persistence 0.888, not 0.95 |
| Effect size on the defining variable | The labels are *defined* by a cut on it | Excluded from evidence |
| Test-vs-zero in an event study | The baseline drifts +52 bp, so everything is significant | Difference-in-differences adopted — caught because the *placebo* fired |
| The pre-registered placebo | Its event is, by construction, a directional transition | Replaced by a control differing in exactly one dimension |
| "Significant vs not significant" as a contrast | It is not a test of the difference | Direct contrast test, with its power bound |
| Nowcast persistence baseline | Conditioned on a label unknowable at *t*, quasi-circular with the target | Replaced by the information-set-fair version |
| PIT bootstrap null | Constructed by resampling the observed statistic's inputs | Zero power — could not reject a PIT with all mass in one bin |
| Textbook χ²(9) reference | i.i.d. assumption on a correlated panel | Correct critical value 584, not 16.9 |
| Full-sample stratification | Conditions on the future, so a causal model must appear miscalibrated | Gradient 0.30 → −0.04 on identical forecasts |

And its mirror — the **positive control**, an instrument shown to fire on a
case whose answer is known in advance — appears only twice, both in the
backtest engine: the cheat-alpha gate and the constructed flip book. Both are
trivial to implement and decisive. That asymmetry is a finding: practitioners
occasionally check that a test does not fire spuriously, and almost never check
that it fires when it should.

**The single sentence the paper is built on:**

> Every validation instrument — metric, baseline, placebo, null, or gate — must
> be shown capable of returning a negative before its positive is admitted as
> evidence, and shown to fire on a constructed positive before its negative is.

That statement is model-agnostic, cheap to act on, and violated eleven times in
one carefully-run research programme by people actively looking for problems.

---

## Why the framework generalises even though the models do not

The specific instruments here are hyper-specific: an HHI on masked depository
records, a probit rank within a trading day, a Kupiec test on a 578,481-row
panel. None transfers.

What transfers is the **structure of the argument**, and the reason it
transfers is that the six layers are properties of *what modelling is*, not of
what was modelled:

- Every quantitative model consumes a **measurement** whose validity is prior
  to the model (L1).
- Every model belongs to a **class** with an inductive bias that determines
  what it can represent at all (L2) — and this is the layer almost universally
  skipped, because the model is chosen before the question is examined.
- Every fit is one draw from a procedure, and its **stability** is a separate
  question from its fit quality (L3).
- Every **effect** competes with alternative explanations that must be
  constructed, not merely acknowledged (L4).
- Every **out-of-sample number** carries a date it must have been producible on
  (L5).
- Every **decision** faces a trivial rule and a cost (L6).

A reader in a different domain replaces the instruments in every cell and keeps
the matrix. That is the claim, and it is why the paper can be honest about the
data's hyper-specificity without weakening its contribution.

---

## The story, in the order the paper tells it

1. **Open on the scorecard.** A walk-forward risk engine beats the industry
   benchmark on every metric of the standard battery, on 578,481 out-of-sample
   observations. Downstream of it sits a regime model that replicated
   digit-for-digit out of sample, an econometric battery that survived eleven
   alternative specifications, and a backtest whose engine passes its own gates.
   By every convention of the field, this is validated work.

2. **Then ask the D3 question of each instrument in turn.** Eleven of them could
   not have returned a different answer. The failures are not in the
   statistics — every test was computed correctly — they are in whether the
   tests were capable of speaking.

3. **Show what that changes.** A distributional verdict resting on a zero-power
   null. An out-of-sample persistence figure that was largely the prior
   restating itself. A model-order choice made by a criterion that could not
   discriminate. A calibration defect that existed only in the diagnostic. A
   headline attributed to a mechanism that ablates to nothing.

4. **Show what it does not change, and why that matters more.** The results
   built on instruments that *could* have failed survive: the permutation test's
   episode clustering, the reproducibility measurement, the mechanical twin's
   κ, the cheat-alpha-gated backtest's breakevens, the fat-tail result, the
   truncation audit's exact zeros. **The framework discriminates; it does not
   merely destroy.** Without this step the paper is nihilism.

5. **Generalise via the matrix**, with the honest scope statement: six layers ×
   three disciplines, instruments replaced per domain.

6. **Close on the asymmetry.** In quantitative finance the productive effort is
   not spent establishing a result. It is spent constructing the tests that
   could destroy it — and then, one level up, establishing that those tests were
   capable of doing so. A result that has survived instruments known to be
   capable of killing it is the only kind that can be used.

---

## Instrument choice is itself a validation decision

`09_INSTRUMENT_FAMILIES.md` extends each cell of the matrix from *what this
programme used* to *what the field offers for that question*, with a selection
criterion and a D3 note for each.

That extension does three things the inventory alone cannot:

1. **It converts the paper from a memoir into a framework.** "Here is what we
   did" is an anecdote; "here is the family of instruments for this question,
   here is what we chose, here is why, and here is the condition under which
   each is incapable of failing" is a method a reader can apply.

2. **It makes D3 operational rather than aspirational.** Every family entry
   carries the condition under which that instrument cannot return a negative —
   AUC is invariant to monotone transforms so it cannot see miscalibration; BIC's
   penalty vanishes at large n; Diebold–Mariano degenerates under the null for
   nested models; a full-support PIT test spends its power on the middle of a
   distribution when the claim is about the tail. These are properties of the
   instruments, documented in the literature, and checkable in advance.

3. **It generates the paper's own gap list.** Ten instruments the framework says
   should have been used here and were not — staggered-adoption DiD estimators,
   selection accounting (PBO / deflated Sharpe / SPA / MCS), proper ES backtests
   and joint (VaR, ES) scoring, threshold-weighted CRPS and a Murphy diagram,
   PIT independence, the nested-model correction to DM, purged CV, an HSMM, ARI
   instead of κ, and decision-curve analysis.

**The third point is the important one.** A framework paper that cannot generate
a concrete list of what its own case study should have done differently has not
demonstrated that the framework does anything. The gap list is the framework
running on the programme that produced it, and it belongs in the paper.

## What this changes about the current documents

| Document | Status |
|---|---|
| `01_PROTOCOL.md` — the nine questions | **Unchanged, and now correctly placed**: it is the L5 (forecast) row of the matrix, fully instantiated. It stays as the deepest worked layer |
| `03_FAILURE_MODES.md` | Unchanged. It is the L5 failure register |
| `07_TECHNIQUE_INVENTORY.md` | The raw material — 59 techniques used, organised by layer |
| `09_INSTRUMENT_FAMILIES.md` | Each cell extended to the **field's** instrument family, with selection criteria, D3 notes, and the gap list |
| **This document** | The organising claim. Supersedes `00_FRAMING.md`'s scoping of the paper to the risk engine alone |
| `06_PAPER_SPINE.md` | **Needs rewriting** against the matrix rather than against the nine questions |

## What still has to be built

Unchanged in kind, larger in scope:

1. **D1 · the synthetic size/power harness** — now clearly the software
   embodiment of discipline D3. Given a test statistic and labelled samples,
   return size and power. This is the library's first module and the paper's
   central exhibit.
2. **D2 · the evaluation-leakage reproduction** — L5×D3.
3. **D3 · the generalised truncation harness** — L5×D3.
4. **NEW · the cheat-alpha harness** — L6×D3, and the only *positive*-control
   instrument in the inventory. Trivial to generalise: given a backtest engine
   and a return series, inject the oracle signal and assert the Sharpe
   collapses at lag 1.
5. **NEW · the permutation-clustering test** — L3×D3, generalises to any claim
   that discrete labels cluster in time.
6. **Re-verify every magnitude** against a current run before any prose.
