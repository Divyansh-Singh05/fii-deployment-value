# 3 · Methodology

<!-- provenance -->
*No numerals appear in this section.*
<!-- /provenance -->

---

## 3.1 Design

Each **application** pairs two measurements made on the same object and the same
data:

- a **significance result** — what a conventional test reports;
- a **value result** — what the same object delivers once one level of
  institutional friction is imposed.

The pairing is the design. An entry whose two halves come from different samples
establishes nothing and is not admitted. Each application is then placed on the
ladder at the lowest level it does not survive.

The instrument used at each level — regression, regime model, density engine,
portfolio — is chosen because it is the natural way to impose that particular
constraint, not because it is of interest. Where an instrument's own properties
could be confused with the data's, we say so explicitly and, where possible,
measure the difference.

## 3.2 The levels, and the test at each

**L0 · Existence.** Panel regression with instrument and date fixed effects,
standard errors two-way clustered on instrument and calendar month. Date fixed
effects absorb all market-wide variation by construction, so a market-level
variable can enter only as an interaction — a restriction we impose rather than
work around. Effects are reported as coefficients with era-wise t-statistics;
with samples in the hundreds of thousands, statistical significance is close to
automatic and we treat it as a necessary condition, never as a result.

**L1 · Replication.** Frozen split (§2.4), everything estimated on the training
era and applied unchanged. Both eras reported for every quantity. A sign flip
between eras is treated as failure regardless of significance in either.

**L2 · Availability.** The identical specification re-estimated with the
predictor known only at the lag a decision could actually use. Nothing else
changes. A collapse between the contemporaneous and lagged specification
locates the effect as descriptive rather than predictive.

**L3 · Competition.** The application is scored against the baseline it would
be deployed *against*, not against zero and not against a weak alternative.
Three forms appear:

- a well-specified conditional-variance model, where the question is whether the
  new information is already contained in the instrument's own history;
- a *trivial rule* — a two- or three-line heuristic capturing the same idea —
  compared **paired on the events both act upon**, since both can beat zero
  while one is strictly worse;
- a freely available public proxy, where the question is whether the
  proprietary record is needed at all.

**L4 · Detectability.** Whether the effect registers in the metric the decision
consumes, which is generally not the metric the significance test used. A
predictor established by a conditional-mean regression may be evaluated by a
distributional score; the two have different power, and an effect can be real,
survive every control, and sit below the second's resolution. Where the
application's own mechanism is the claim, this level is tested by **ablation**:
the identical machinery re-run with the mechanism removed and everything else
held fixed, with the detectable effect size stated.

**L5 · Execution.** The breakeven one-way cost at which net profit is zero,
compared against a realistic institutional cost. We report the **breakeven**
rather than a net figure at an assumed cost. The breakeven is a property of the
strategy; a net Sharpe is a property of the assumption, and a reader whose costs
differ from ours can use the first and cannot use the second.

## 3.3 Backtest construction

Where a level requires a simulated book, timing is specified as an equation
rather than described: a signal formed at the close of day *t* is traded at the
close of *t + L* and earns the close-to-close return of *t + L + 1*, with
*L* = 1 by default. For state-derived events, the additional constraint is that
an episode's end is knowable only at the close of the first day after the run
concludes.

The engine is gated before use, and may not be modified without re-passing its
gates:

- **exactness** — the vectorised implementation must equal a naive per-day loop;
- **alignment** — a deliberately constructed *oracle signal* equal to the next
  day's return must produce an extreme Sharpe at zero lag and collapse at lag
  one. An engine that cannot detect a deliberate look-ahead cannot be trusted to
  be free of an accidental one;
- **cost accounting** — a constructed book with known turnover must satisfy
  `net = gross − c · turnover` exactly.

Baselines are constructed and frozen **before** their model counterparts, so
the comparison cannot be tuned. Verdicts use paired differences with a
moving-block bootstrap, block length stated.

## 3.4 Pre-registration

Each application's bar was written into its stage header before its numbers
existed, together with the branch to be taken if the bar were missed. An
application that misses its bar is reported as a negative result and is not
re-specified. Where a bar was amended, the amendment is documented before the
re-run and the original verdict is retained on the record.

Two disciplines follow that are worth naming because they bind:

- **A gate's target is fixed with its bar.** Re-aiming a test at whichever
  quantity turns out to carry the effect is exploratory analysis, permanently,
  and is labelled as such wherever it appears.
- **Diagnostics precede fixes and are read-only.** A data repair is never
  bundled with the analysis that motivated it.

## 3.5 Verification protocol

The paper's subject is the distance between a statistic and its evidentiary
weight. It would be incoherent to report our own figures on weaker terms than
we demand of the finding. Every numeral in the results section must clear four
independent gates before it is admitted.

1. **Fresh.** The artifact is not older than the code that writes it, and no
   upstream stage has been re-run since. This gate exists because it was
   violated: a reporting step in the source pipeline copies a metrics table to
   a second location, and the copy had drifted six weeks behind its origin. A
   digest check and a value check both passed on the stale copy.

2. **Primary.** The source is the computation itself — not a project document
   transcribing it, and not a *copy* of the computation's output. A digest match
   on prose establishes only that nobody edited the prose.

3. **Traced.** The artifact's SHA-256 matches a digest recorded in a committed
   lock file. A changed artifact reports `STALE` rather than being silently
   accepted.

4. **Extracted.** Where an extractor is defined, it locates the value inside the
   artifact and compares it against the declared figure, with a locator that
   must identify exactly one number. A disagreement reports `MISMATCH`; the
   declaration is corrected, never the tolerance.

The protocol is implemented as a small package that renders the results table
and reports, alongside it, how many of its own entries have cleared all four
gates. That count appears in the paper.

## 3.6 Scope

Out of scope, and stated once rather than defended repeatedly:

- **Data preparation and identity resolution.** Corporate actions, identifier
  closure and panel assembly are prerequisites, and the constraint they impose
  on entity statistics is stated in §2.2. They are not the subject.
- **Model architecture.** No instrument is proposed, improved, or defended. An
  instrument's failure is reported as a measurement about the data, not as an
  invitation to a better instrument.
- **Causal interpretation.** Where an effect is found, we state what predicts
  what conditional on controls, and no more.
