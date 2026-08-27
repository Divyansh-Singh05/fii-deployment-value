# 05 · Open Questions

Decisions that change what gets built. Each has a recommendation.

---

## Q1 · Is one demonstration engine enough?

**The risk.** The protocol's ordering claim and the register's direction counts
come from a single engine on a single market. A referee will ask whether the
ordering reflects this engine's architecture rather than a general one.

**Options.**

| | Approach | Cost | Buys |
|---|---|---|---|
| a | Single engine, ordering argued from **void scope** rather than frequency | 0 | The ordering is a logical claim — an alignment failure voids the scorecard by construction — and does not depend on how often each mode occurs |
| b | Add a **simulated** demonstration: inject each failure mode into a synthetic risk model with known ground truth and show each question detects it | ~1–2 weeks | Proves every question has power (D1 applied to the protocol itself), and makes the magnitudes reproducible by anyone. Does not address ordering |
| c | Apply the protocol to a second, published risk model | 3–6 weeks | Portability evidence; high risk of an uninteresting result |

**Recommendation: (a) + (b).** Argue ordering logically, and add the simulated
suite. (b) is unusually cheap here because a risk model with known ground truth
is easy to simulate — you generate returns from a known distribution, so the
correct VaR is known exactly, which is precisely what a real backtest never has.
That is a strong section in its own right.

---

## Q2 · How prominent is the mechanism-ablation result?

F6.1 says the demonstration engine's distinguishing mechanism contributes
nothing measurable. This is the register's largest finding and it is a negative
result about our own model.

**Options:** (i) lead with it — the paper's most memorable claim is "the model
won for a reason that had nothing to do with the model"; (ii) place it as one
question among nine; (iii) soften it by leading with the surviving fat-tail
result.

**Recommendation: (i).** It is the cleanest possible illustration of the
protocol's value: no amount of additional out-of-sample data would have found
it, no benchmark comparison would have found it, and it changes what the result
*is* rather than how large it is. (iii) would be the reviewer's first
suspicion and is not worth the cost.

---

## Q3 · Which venue, and therefore which register?

| Venue class | Wants | Consequence |
|---|---|---|
| Risk practice (*Risk*, *J. of Risk*, *JPM*) | An adoptable protocol, capital numbers, worked failures | Lead with the scorecard-versus-protocol contrast; the bootstrap detail goes to an appendix |
| Empirical finance methods (*J. Empirical Finance*, *Quantitative Finance*) | Engagement with the backtesting literature — Christoffersen, Berkowitz, Acerbi–Székely on ES backtesting, Gneiting–Raftery on proper scoring | Add a formal section on ES backtestability and elicitability; D1 becomes central |
| Reproducibility / software | The harness and the reproduction contract | The library leads; the engine is a demonstration domain |

**Recommendation: risk practice, with a statistical appendix.** Your stated
intent — engineering discipline as the highlight, not the mathematics — points
there. But note one genuine gap this creates: the ES-backtesting literature
(elicitability, Acerbi–Székely) is directly adjacent to Q5 and Q9 and a
methods referee will expect it. It needs at minimum a positioning paragraph
wherever the paper lands.

---

## Q4 · Are the direction counts (FLATTER / INVERT / …) presented as data?

They are currently a classification by reasoning, not a measurement. Presenting
"22 of 38 flatter the model" as a statistic would be exactly the kind of
unearned quantification the paper argues against.

**Recommendation: present as a classification with the reasoning visible, and
make the *direction of each individual mode* the claim rather than the counts.**
The defensible general statement is the qualitative one: **the failure modes of
an unaudited backtest are not symmetric — they are predominantly optimistic,
and the mechanism is structural, not adversarial.** That sentence survives
scrutiny; "22 of 38" does not.

---

## Q5 · Naming

Working: `model_trust_protocol`, "the protocol", "the nine questions".

Title candidates:
- *Nine Questions Before You Trust a Risk Model*
- *What a Backtest Cannot Tell You*
- *Passing the Battery Is Not Validation*

Library candidates (deferred): short, importable, honest about scope —
`backstop`, `voidscope`, `attest`.

---

## Q6 · Library shape — deferred, but constrain it now

1. **Panel-finance-specific, not general.** The truncation harness, ESS
   deflation, date-block bootstrap and causal stratification all assume a
   date-indexed panel with walk-forward deployment. A general "model
   validation" library would be a worse library and a weaker paper.
2. **First module is the synthetic size/power harness** (T5.1). Self-contained,
   needs none of the case data, and is the paper's strongest exhibit.
3. **Second module is the truncation harness** (T3.2), taking a callable chain
   and a truncation date.

---

## Doubts on the record

1. **Scoping out data preparation is a strong assumption, and our own case
   shows it.** The paper should say once, plainly, that it assumes correct
   inputs and that this assumption is not free — then proceed. Pretending the
   question does not exist would be the weakest available choice; declaring it
   as a premise is defensible.

2. **The protocol's ordering is a logical claim, not an empirical one.** It
   should be argued from void scope. Any frequency-based framing invites the
   objection in Q1 and is not needed.

3. **Q6 (mechanism ablation) has no established name in the risk literature.**
   Ablation is standard in machine learning and essentially absent from risk-model
   validation, where comparison is almost always against an external benchmark
   rather than against the model with its own thesis removed. That gap is an
   opportunity, but it also means there is no citation to lean on and the
   argument must stand alone.

4. **Two questions rest on a single measured instance each** — Q2 (fit
   admissibility, from the market-engine estimation work) and Q8.2 (serve-versus-artifact
   drift). One instance is an illustration, not evidence. Either the simulated
   suite in Q1(b) supplies more, or these are demoted to checks within
   neighbouring questions.

5. **The surviving result should be interrogated as hard as the failures.** The
   engine's tail win has cleared every question — but it has been examined by
   the people who built it. The most fragile survivor is worth naming in the
   paper before a referee does. The candidate is the CRPS margin: **0.5579
   versus 0.5610 is a 0.55% improvement**, and while it is significant at
   t = 9.81 against a Bonferroni-adjusted bar, the *magnitude* is small enough
   that Q6's own standard (F6.2 — significance is not magnitude) applies to it
   as much as to the soft/hard comparison. The defensible headline is the tail
   calibration (1.04% vs 1.62% breach; per-name pass 90.4% vs 60.4%), not the
   CRPS.
