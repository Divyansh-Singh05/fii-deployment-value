> [!CAUTION]
> **STALE — DO NOT USE (flagged 2026-08-23).**
>
> This document predates the August 2026 engineering audit and states the
> retracted concentration-axis headline (`entity_hhi_raw` measured each
> entity's own-book Herfindahl, not within-stock-day participation shares;
> the two correlate r = −0.2460). Its economic claims about SHARK_DIST are
> **withdrawn**, and its exhibit numbers were generated from pre-audit
> tables now preserved under `outputs/tables.PREAUDIT_20260712/`.
>
> Current results: `docs/paper/FII_final_results_one_pager.md`.
> Full ledger: `docs/AUDIT_RETRACTIONS.md` (R1–R6).

# Foreign Institutional Flow Regimes in Indian Equities
## A complete essay and three-speaker presentation script

*Every number in this document is drawn from the repository's own logs, tables,
and research notes (`outputs/validation/*.log`, `docs/research_log/*.md`,
`docs/paper/FII_thesis.md`), not invented for the speech. Suggested split for
three speakers, each good for 6-9 minutes of natural delivery:*

- **Speaker 1 — the question and the machinery:** Abstract, Introduction,
  Literature Review, Data Sources & Audit
- **Speaker 2 — the model:** Model Architecture, Methodology, HMM Issues &
  Fixes, Regimes vs. Archetypes, the Decision Boundary
- **Speaker 3 — proof and payoff:** Validation Battery, Major Dates & Results,
  Economic Value, Future Scope

Read straight through, this is roughly 20-25 minutes at a natural pace, with
room to breathe and take questions between sections.

---

## Abstract

Every trading day, India's financial press reports one number: net foreign
institutional investor (FII) buying or selling, in crores of rupees. That
number moves markets and headlines, and it is close to the wrong number,
because it collapses a *composition* into a *magnitude*. Two days can show
identical ₹300 crore of net FII selling — on one, a single distressed fund
liquidates a position under duress; on the other, forty independent
institutions each trim a holding for forty independent reasons. Fifty years of
market microstructure theory says these should have opposite price
consequences: the first is a *liquidity event*, temporary and reversible; the
second, if the decisions share an informational cause, should be
*permanent*. Public flow data cannot tell them apart. This project can,
because it works from NSDL's transaction-level FII records — masked but
internally distinct entity identifiers, for every FII trade in every listed
Indian equity, April 2011 through March 2025 — and turns "how many distinct
institutions traded, and how concentrated was their activity" into a daily,
per-stock measurement. A three-state Hidden Markov Model, deliberately
restricted to flow-only, strictly backward-looking features, decodes each
stock-day into a directional regime (sustained FII buying, selling, or
neither); a statistically-calibrated overlay on top of that backbone further
tags the rare, economically loaded corners — concentrated selling
("Shark Distribution," a shark aggressively offloading a position),
concentrated buying ("Shark Accumulation," a shark aggressively building
one), and dispersed, quiet selling ("Hostage," after Coval and Stafford's
fire-sale framework). Twenty-plus
validation modules — panel regressions with two-way clustered standard
errors, an independent Probability-of-Informed-Trading estimator, a machine-
learning challenger, three explicit referee-anticipation tests, an illiquid-
tail extension, and a self-administered "hostile read" — all point the same
way: concentrated FII selling precedes a reversal of roughly 37 to 65 basis
points within a month; dispersed FII selling precedes a decline that does
not come back. The effect is real, replicates out-of-sample on a
substantially different set of stocks, survives every alternative
explanation we could construct against it, and does not clear realistic
trading costs as a standalone strategy. It is nonetheless valuable — as free
information for desks whose costs are already sunk, as an independent lens
for risk and policy monitoring, and as a rare, fully-documented example of
knowing when *not* to escalate to a more complex model.

---

## Part I — Introduction

### 1.1 The starting observation

Every one of us has read a headline like "FIIs sell ₹2,400 crore in Indian
equities today." What that headline hides is composition. Consider two days
on which foreign institutions sell exactly the same amount of the same
stock. On the first, one large foreign desk unwinds a position it needs to
exit *now* — it demands immediacy from the market, and classical
microstructure theory, from Kraus and Stoll's 1972 block-trade study through
Grossman-Miller and Campbell-Grossman-Wang, predicts that kind of pressure
should be at least partly *temporary*: price concedes to the seller, then
drifts back once the selling stops. On the second day, forty distinct
institutions each independently trim the same stock by a small amount. If
those forty decisions share a common cause — a change in the fundamental
picture, a sector-wide reassessment — the resulting price move should be
*permanent*: there is no single urgent seller whose exit the price can
revert from.

The theoretical distinction between temporary, liquidity-driven price impact
and permanent, information-driven price impact is decades old. What has
been missing, everywhere, is the ability to actually *observe* which kind of
day you are looking at. Public flow data reports magnitude — how much was
bought or sold — and never composition — how many distinct hands did the
buying or selling.

### 1.2 The data opportunity, and the question

This project exists because one dataset makes composition partially
observable: NSDL's transaction-level records of FII trades in Indian
equities. For every trade, we see the date, the ISIN, buy or sell, quantity,
price, and a masked identifier for the foreign institution, its sub-account,
and its broker. The masking destroys *who* is trading — deliberately, and,
as the audit work will show, far more thoroughly than we first assumed — but
it preserves, within a given day, *how many* distinct institutions
participated, and how unevenly the day's volume was split among them. That
is exactly the measurement fifty-year-old theory has been waiting for: a
daily, stock-level read on whether flow was *concentrated* — few entities,
skewed participation — or *dispersed* — many entities, roughly even
participation.

The question this project asks is simple to state and hard to answer
honestly: **can the shape of foreign institutional flow — not just its
size — predict what a stock's price does next, and does that prediction
survive every serious attack we can mount against it?**

---

## Part II — Literature Review

The project sits at the intersection of five literatures, and it was built
against a curated set of over sixty peer-reviewed papers from 2023 onward
(`docs/paper/literature_2023plus.md`), alongside the classical foundations.

**Institutional price impact and the transitory/permanent split.** The
theoretical spine — Kraus and Stoll (1972) on block trades, Grossman and
Miller on liquidity provision, Campbell, Grossman and Wang on the volume-
conditioned reversal — argues that price impact from liquidity-driven
trading should revert, while impact from information-driven trading should
persist. Recent work updates this directly to our setting: Glossner et al.
(2025, *Management Science*) find that institutional *downscaling* trades
reverse while *repositioning* trades persist during COVID-19 — precisely
our transitory/permanent decomposition, in a crisis window; Fan et al.
(2025, *Journal of Financial Markets*) distinguish "granular," large-player
shocks from common shocks in a way that closely parallels our
concentrated-versus-dispersed axis; Xu et al. (2025, *Quantitative
Finance*) model transient impact as arising from the *mix* of informed and
uninformed order flow — the theoretical frame our archetypes are built to
measure empirically.

**Fire sales and forced, dispersed selling.** The "Hostage" archetype — a
persistent, one-directional, but *dispersed* sell-off — is named for Coval
and Stafford's fire-sale framework, in which forced, uncoordinated selling
by many distressed holders produces price pressure with no natural buyer
waiting to arrest it. Giannetti et al. (2024, *Review of Financial
Studies*) and Kundu (2023, *Management Science*) extend this to ownership
concentration and covenant-driven forced selling respectively — both treat
*concentration structure*, not flow size, as the state variable that
determines fragility, exactly the design choice this project makes.

**Hidden Markov models and regime detection in finance.** The
methodological literature (Oelschläger and Adam's `fHMM` software paper;
Tampouris et al. on hierarchical HMMs for structural change; Qin et al. on
robust hidden semi-Markov estimation) is unanimous on one point relevant
here: plain, flat HMMs struggle to represent more than one axis of
structure at a time, which motivates exactly the hierarchical or
factor-structured extensions this project's own negative result (Part IV)
independently rediscovers from first principles, and which the project's
own Factorial-HMM challenger (Module 17) directly tests.

**Informed trading.** Easley and O'Hara's Probability-of-Informed-Trading
(PIN) model, and its modern re-estimation via expectation-conditional
maximization (Ghachem and Ersan, 2025, *Financial Innovation*), gives an
entirely independent, order-count-only estimator of informed trading — used
in this project not as a feature, but as an *external check* on whether the
dispersed-selling archetype really does correlate with information-driven
trading, the way theory predicts.

**FII flows in India specifically.** Thapa et al. (2025, *Review of
Quantitative Finance and Accounting*) work with a directly comparable
NSDL-style transaction dataset; Naik et al. and Batra et al. frame the
long-standing "do foreign flows stabilize or destabilize emerging markets"
debate that this project's composition-based decomposition is built to
resolve, rather than merely re-state.

---

## Part III — Data Sources and Audit

### 3.1 The raw material

Two primary sources, spanning **April 2011 to March 2025**: (1) NSDL's FII
transaction feed — roughly 28 million scoped equity buy/sell rows after
restricting to real trades (`TR_TYPE ∈ {buy, sell}`, positive rate,
regular-delivery equity instrument type — this filter alone removes
corporate-action legs and non-equity instruments); and (2) NSE's daily
bhavcopy price tape and corporate-action disclosures. Aggregated to one row
per (canonical stock, trading day), the FII side becomes a panel of
**2,423,212 stock-days across roughly 3,800 stocks**.

### 3.2 The audit that had to happen first: masked entity IDs are not stable identities

Before any feature could be trusted, the project had to answer a
foundational question: do the masked FII/sub-account/broker identifiers
carry a *stable identity across time*? The audit (research log, Module 1,
§3) found the assumed uniform ID format was wrong — six distinct shapes
existed across two eras with a silent transition in between, roughly
2016-2020 (13-character and 14-character IDs with no date suffix in the
later era; 17-character and 19-character IDs carrying an embedded
`YYYYMM` report-month suffix in the earlier one). The decisive test tracked
how many *distinct calendar months* each masked ID appeared in, over spans
of 84 to 98 months. Even for **brokers** — a small, near-fixed real-world
population of perhaps 150 to 320 active firms in any given month, the one
group that should show rock-solid persistence if the ID scheme worked —
**zero percent** of IDs persisted for twelve or more months in either era.
A roughly 300-broker real population fragmented into 12,700 distinct masked
IDs in the early era and 9,221 in the late era. The verdict was
unambiguous: masked IDs are re-minted on a roughly monthly cycle, with no
stable cross-month identity at any level. Consequence: the project could
never track a single fund's liquidation across a quarter — only within a
single day. This is precisely why the "entity concentration" feature had
to be redesigned as a strictly *within-day* measurement, and it is a
finding this project treats as load-bearing, not a footnote.

### 3.3 The corporate-action problem, and a small NLP pipeline to fix it

The bhavcopy price tape's own `prev_close` field was assumed to already be
adjusted for stock splits and bonus issues. It was not: on confirmed
split/bonus ex-dates, the raw close-to-close return showed a median absolute
move of roughly **50 percent** — a mechanical "crash" from an unadjusted
face-value change, not a real price move. The fix required building a
corporate-action adjustment factor table from scratch, by parsing NSE's
free-text corporate-action announcement field with a small regular-
expression-based text parser. The first version of that parser had a real
bug: on combined announcements like *"Bonus 1:1 And Face Value Split
Rs.10 To Re.1,"* it grabbed the first two numbers in the sentence — the
bonus ratio — and silently mis-parsed or dropped the split factor entirely,
for names including TITAN and ONGC. The fixed parser anchors the
split-factor pattern specifically *after* the word "split" in the text.
Every parsed factor was then independently verified against the *observed*
ex-day price ratio: **99.1 percent (805 of 812 testable events) confirmed**,
and once applied, the median absolute split-day return fell from **0.508 to
0.038** — the mechanical crash essentially disappeared, and what remained
was the real market move.

### 3.4 Identity: the ISIN closure problem

A single company can trade under multiple ISINs over its life — after a
demerger, a face-value change, or a symbol migration — and the price tape and
the FII-derived model states can disagree on which ISIN represents "the
same company" on a given day. The project built one shared closure map,
applied identically to both sides, using the issuer code embedded in every
ISIN's structure and an overlap-based rule: if two ISINs under the same
issuer never meaningfully coexist in time, they are the same company under
two names and get merged; if they trade side by side for a long stretch —
a partly-paid share class alongside the ordinary shares, for instance —
they are kept as genuinely distinct instruments. This single step raised
the match rate between the model's states and the price tape from
**90.4 percent to 98.5 percent**.

---

## Part IV — Model Architecture and Methodology

### 4.1 Ten features, four axes, one guiding principle

From the scoped FII trades, the project builds exactly ten stock-day
features (`F_persist`, `F_block`, `F_entity`, `F_entity_buy`, `F_breadth`,
`F_sizedisp`, `F_activity`, `F_streak`, `F_imbal`, `F_flowbeta`), but only
four ever feed the model — one deliberate signature per hypothesized
archetype: **persistence** (a 20-day trailing, signed measure of sustained
directional flow — separates transient "Robot" flow from sustained
conviction), **blockiness** (average trade size relative to a stock's own
baseline — separates large discretionary prints from small, uniform
slices), and two **entity-concentration** axes, on the sell book and the
buy book respectively, measuring whether the day's flow was dominated by a
few institutions or spread across many. Every feature is transformed the
same way: a within-day cross-sectional percentile rank, mapped through the
inverse normal distribution into a probit score. This single step earns
three things at once — it neutralizes fourteen years of non-stationarity
(FII participation breadth alone grew roughly 254 percent over the sample),
it gives a diagonal-covariance Gaussian HMM honest, standard-normal
emissions to fit rather than skewed raw ratios, and it is naturally robust
to outliers, since a data error can move one stock's rank by one slot, not
distort a moment. Every trailing window is strictly backward-looking, and
the model never sees price, index, or macro data at all — a deliberate
negative design choice, so that the later economic validation, which tests
whether these flow-only states predict *price* behavior, is meaningful by
construction rather than by assertion.

### 4.2 Why a Hidden Markov Model

Regimes — sustained buying, sustained selling, or neither — are, by
definition, a *temporal* property: a stock is not permanently a "Shark"
target, it becomes one for a period and then stops. A Hidden Markov Model
is the natural tool for exactly this: it assumes an unobserved state that
persists with some probability and occasionally transitions, and it infers
both the states' characteristic signatures and their persistence directly
from the data, rather than assuming either.

### 4.3 The two-part architecture

The final model has two parts, arrived at only after a real design failure
(next section). A three-state, diagonal-covariance Gaussian HMM fits a
directional backbone — SELL, NEUTRAL, or BUY regime — on the two
best-behaved features. A statistically calibrated *overlay rule* is then
applied on top, using the two entity-concentration features, to further tag
economically distinct sub-populations: within a sustained sell regime,
dispersed selling is tagged **Hostage**; concentrated selling is tagged
**Shark Distribution (SHARK_DIST)** — full form, a "shark" aggressively
*distributing*, i.e. offloading, a position, identifiable because the
selling is concentrated among a few institutions rather than spread
across many. Within a sustained buy regime, concentrated buying is tagged
**Shark Accumulation (SHARK_ACC)** — the mirror image: a shark
aggressively *accumulating*, i.e. building, a position. The two are
distinguished purely by regime direction (SELL vs. BUY) with the *same*
concentration logic applied to each side; both predict a mean-reverting
price move once the concentrated pressure stops (SHARK_DIST → a
reversal/recovery upward; SHARK_ACC → a symmetric give-back downward),
because both are read as temporary, liquidity-driven impact rather than
new information. The neutral regime is tagged "Robot"; anything
directional but structurally unremarkable is honestly left as "Untagged
directional" rather than force-fit into an archetype.

---

## Part V — Issues in the HMM, and How They Were Fixed

This is the part of the project most worth dwelling on, because the
architecture above was not the first design — it is what survived a
documented failure.

### 5.1 The first attempt, and why it failed

The original plan was more ambitious: fit a single HMM directly on all
four features and let it discover all three archetypes as native states.
It did not work. Across every state count from three through six, the
model always found the same thing — a **persistence ladder**: states
ordered purely by how strongly and in which direction flow was sustained
(strongly selling, mildly selling, neutral, mildly buying, strongly
buying), never a state combining "persistently selling" *and*
"dispersed." The pre-registered success criterion for this phase — a
state jointly satisfying persistence clearly below zero *and* entity
concentration clearly below zero — never fired, at any state count. This
was not "no signal": dissecting the persistent-sell state by hand found
that roughly 26 percent of its member days were genuinely dispersed —
the material for a Hostage state plainly existed in the data. The model
simply never carved it out.

### 5.2 Diagnosing the mechanism, not just the symptom

Rather than trying more state counts or more features, the project asked
*why*, using an autocorrelation diagnostic. A Hidden Markov Model's
self-transition probabilities are learned from features that are
themselves temporally smooth — that is what makes a "persistent" state
statistically identifiable in the first place. The median per-stock
lag-1 autocorrelation of `F_persist` — a 20-day trailing average — was
**0.926**: extremely smooth, exactly the kind of feature an HMM can build
a persistent state around. The entity-concentration feature, by contrast,
was a strictly single-day snapshot (a direct consequence of the entity-ID
audit above: no cross-day identity survives, so no cross-day concentration
measurement is possible) — its autocorrelation was **0.334**. The model
had effectively been fed one ultra-smooth feature and one that looked like
noise from one day to the next, and an EM algorithm allocates its finite
states to the axis it can actually see persisting. Axis 3 was not weak; it
was **structurally invisible to this model class**.

### 5.3 The audit-legal fix

The fix had to respect the entity-ID audit's hard limit: no feature could
require tracking an entity across days. The solution was to smooth the
*measurement*, not the entity: a five-day trailing average of the daily
concentration snapshot, re-ranked within day and re-transformed to a
probit score. This requires no cross-day entity identity whatsoever — it
only averages a stock's own daily dispersion readings — so it stays
entirely within the audit's constraint. The effect was immediate: lag-1
autocorrelation on the smoothed concentration axes rose to **0.767 and
0.763**, and refitting on the smoothed features at four states produced
the single most informative result of this stage of the project — the
model spontaneously split its neutral middle state into a
*concentrated-both-sides* state and a *dispersed-both-sides* state. The
regime space, it turned out, is genuinely two-dimensional: direction
**and** concentration. But even here, the rarest, most economically
loaded corners — sell-and-dispersed, buy-and-concentrated — never earned
their own dedicated state at any state count, because an EM algorithm
allocates states by probability mass, and these corners are, by design,
rare. That is precisely the finding that justified the final hybrid
architecture: let the HMM do what it can prove it does well — the
directional backbone — and use a calibrated rule, not a fifth HMM state,
to carve out the rare, structurally invisible corners.

### 5.4 One more fix, upstream of all of this: the fitting sample itself

A separate, easily overlooked bug was found and fixed in the same period:
the very first implementation capped each stock's contribution to model
*fitting* at its four hundred *most recent* rows, to stop a handful of
hyperactive large-cap stocks from dominating the likelihood. That cap
quietly biased the fitted parameters toward late-sample behavior. The
fix, carried into the production model (`module3a_model_split_oos.py`),
replaced "most recent 400" with a **randomly chosen contiguous block** of
up to 400 days per stock — still respecting the requirement that an HMM's
transition structure needs temporally contiguous data, but no longer
silently favoring any particular era.

---

## Part VI — Regimes vs. Archetypes, and How the Decision Boundary Was Set

### 6.1 The distinction

**Regimes** are the three states the HMM itself estimates and decodes —
SELL, NEUTRAL, BUY — the directional backbone, learned entirely from
data, with no thresholds anywhere. **Archetypes** — Robot, Shark Distribution (concentrated selling),
Shark Accumulation (concentrated buying), Hostage (dispersed selling),
and Untagged directional — are
the five *economic labels* produced by combining a regime with a rule
applied to the entity-concentration features. A regime is a hidden state
with a transition probability; an archetype is a name attached to a
region of that state space plus a concentration reading. This distinction
matters because it is honest about what each half of the model can
actually claim: the regime split is model-driven and falls out of a
proper likelihood fit; the archetype split, for the rare corners, is
rule-driven, and its rule had to be calibrated, not eyeballed.

### 6.2 How the threshold was actually decided

The very first version of the overlay used a single hand-picked cutoff —
0.5 in probit units, roughly the 69th percentile — good enough to prove
the concept, but an unexamined free parameter sitting under every later
result. The production redesign replaced it with a data-driven procedure:
fit a Bayesian-Information-Criterion-selected Gaussian mixture model (one,
two, or three components) to the distribution of the concentration
feature *within the training-era sell regime only*, and take the
posterior-probability-one-half boundary of the mixture's "tail" component
as the threshold. Two falsification checks were pre-registered before
looking at the answer: a k-means cross-check for consistency, and — the
one that actually fired — a stability requirement that re-deriving the
same boundary on the held-out test era must not move it by more than 0.25
probit units. The real run failed that stability check (the mixture
model was, in the researchers' own words, close to "a 49/51 coin flip,"
and the train-versus-test boundary moved by 0.665) — and the honest,
pre-registered response was not to keep tuning the mixture model but to
fall back to the alternative rule that had been specified in advance for
exactly this contingency: a simple, frozen training-era quantile cut
(25th/75th percentile). The published, frozen thresholds — roughly
**-0.51 for Hostage, +0.88 for Shark-distribution, and +0.80 for
Shark-accumulation** — are the result of that fallback, derived once on
training data and then never touched again on the test era.

---

## Part VII — The Validation Battery

No single test carries this project's claim. Twenty-plus modules were run,
each with a pre-registered pass/fail bar decided *before* the result was
seen, and each targeting a distinct way the finding could be wrong.

**Event studies (Module 5B4).** Cumulative abnormal returns around
episode start and end, measured *relative to the whole labeled universe's
own contemporaneous baseline* rather than zero — a correction made after
discovering the entire universe drifted meaningfully against the market
benchmark in the test era, which would otherwise make even the neutral
"Robot" placebo look spuriously significant.

**Mechanism tests (Modules 6, 6B).** External corroboration using NSE's
own disclosed block, bulk, and short-sale deal records — concentrated-
selling episodes coincide with disclosed large trades far more often than
dispersed-selling episodes, confirming the "concentrated flow = visible
block trade" story from a data source entirely independent of the FII
feed. A second, purely internal test confirms the full pressure-then-
volume-climax-then-reversal arc, with a pre-committed rule to stop
mechanism-hunting entirely if this test had failed.

**Panel regression — the headline result (Module 7).** The main
econometric specification: a fixed-effects panel regression of
post-episode 20-day abnormal return on archetype dummies, with **stock and
calendar-date fixed effects** and **two-way clustered standard errors**
(by stock and by month). The result: concentrated distribution
("Shark-Dist") predicts a reversal of **+65.4 basis points in the training
era and +48.5 basis points out-of-sample** (both statistically significant
at conventional levels); concentrated accumulation predicts a symmetric
give-back of **-87.9 and -47.6 basis points**; dispersed selling
("Hostage") shows **no significant reversal in either era** — consistent
with a permanent, information-driven decline rather than a temporary
liquidity dislocation.

**Robustness (Module 7B).** The same specification, stress-tested against
non-overlapping episode windows, return horizons from 10 to 60 days, and
a version using the raw continuous concentration measure instead of any
discrete label at all — all confirm the same sign and a broadly similar
magnitude.

**A machine-learning challenger, and knowing when to stop (Modules 8,
8B).** A LightGBM model trained on all ten raw features — including the
six the HMM never uses — was tested against the HMM-regime baseline as a
pre-registered "is the HMM leaving signal on the table" question. It beat
the baseline (test-era rank correlation of +0.0279 versus +0.0117). A
careful follow-up then asked whether that edge was genuine day-to-day
dynamics or just static stock characteristics in disguise — one feature,
crowdedness, turned out to be 55 percent driven by fixed, between-stock
variance rather than real dynamics. Once every feature was de-meaned by
its own stock's training-era average, the dynamics-only version of the
challenger's edge fell just short of its own pre-registered bar. This is
the project's most important negative result by design: a documented,
pre-registered decision **not** to escalate to a more complex sequence
model (an LSTM), because the simpler regime description was already
capturing essentially all of the genuine dynamic signal.

**The flow-surprise control (Module 9, INNOV).** Built the way academic
market-microstructure work usually builds a flow-surprise measure — an
autoregressive residual on scaled net flow, deliberately allowed to use
full-sample information because it is a validation object and never a
model input. Adding this "surprise magnitude" as a control to the main
regression does not make the concentrated-selling reversal disappear —
closing off the last obvious alternative explanation that the archetype
effect is really just flow-surprise size wearing a different name.

**State-dependence (Module 10).** Tests whether the reversal is larger in
high-volatility (India VIX) or high-price-impact conditions. The result
was mixed to null, and reported as such — an honest non-finding that does
not threaten the main, unconditional result.

**An independent informed-trading benchmark (Module 11, PIN).** The
Easley-O'Hara Probability-of-Informed-Trading model, estimated by maximum
likelihood per stock-year from FII order counts alone — this estimator
never sees a price, a return, or an archetype label. Regressing stock-year
PIN on the share of days spent in each archetype found that the dispersed-
selling share loads **2.3 to 3.0 times more heavily** on informed-trading
probability than the concentrated-selling share does (training-era
t-statistics of 7.91 versus 2.87; test-era 4.47 versus 2.54) — a fully
independent estimator, built from a different mathematical model on the
same raw counts, landing on the same permanent-versus-transitory story.

**Three referee-anticipation tests (Modules 13A-C).** Before any reviewer
could ask: is the elaborate HMM machinery actually necessary, does a
trivial rule reproduce the same result? A census-matched rule backbone —
no hidden Markov model at all — was built and run through the identical
regression. It reproduced the headline result closely enough that the
honest conclusion is the HMM's contribution is the *composition measure
itself*, not some deeper structure only a hidden-state model could find.
Is the composition signal just repackaging conventional flow-magnitude
measures? Adding explicit flow-magnitude and imbalance controls left the
concentrated-selling coefficient intact within roughly 11-13 percent of
its original size. Could this simply be passive index-rebalancing
mechanics? Excluding every episode near a known index-review date left the
result standing.

**Extending to the names that were left out (Modules 14A-C).** The core
model universe is a liquid subset; a separate three-stage study asked
whether the same reversal holds in the far larger, thinner "tail" of FII-
traded names excluded from that universe. It does show the same
directional structure — but the honest verdict was that any exploitable
reversal there does not clear the tail's much higher trading friction, so
it was reported as a real but currently unharvestable regularity rather
than a second tradeable result.

**A self-administered hostile read (Module 15).** The last and, in
spirit, the most adversarial module: a fresh attempt to break the
project's own three most exploitable weak points. Is the reversal just
mechanical bid-ask bounce? Delaying the measurement window by two extra
days to skip the bounce-prone period still left a significant, similarly-
signed reversal (roughly **+58 basis points in training, +37 basis points
out-of-sample**, versus the standard-window +71.6 and +48.5). Is
"significant versus not significant" actually a valid contrast between
Hostage and Shark-distribution? A formal statistical test of the
difference — not an eyeballed comparison — confirmed the two archetypes'
effects are themselves significantly different from each other. Does a
well-known public factor, the volume-conditioned reversal already
documented in the literature, absorb the whole effect? Head-to-head, the
composition signal survived controlling for it, and a staged machine-
learning test confirmed composition adds predictive value beyond both
public price/volume information and conventional private flow measures.
All four pre-registered verdicts: pass.

**From validated states to a real-time predictor (Phase II, causal
filtering, calibration, hazard, and decision).** Every result above uses
smoothed state labels, which are legitimate for retrospective validation
but technically see information from after the day being labeled. A
later, separately chartered phase of the project rebuilt the state
estimation as a strictly causal, forward-only filter — using the exact
same frozen parameters, never re-estimated — turning the model's output
into a genuine day-by-day probability rather than a retrospective tag,
and re-verified that the same economics survive on these fully causal
labels. On top of that causal foundation, a hazard model was built to
predict *when* a concentrated-selling episode will end, which beat a
simple age-only statistical baseline; and a final decision-layer test
asked whether *acting early* on that hazard prediction earns more than
simply waiting for confirmation that the episode has ended. It is
statistically forecastable — but, reported just as honestly as every
null result above, it does not yet clearly beat the simplest
naive-timing control once compared rigorously, an outcome recorded as
"forecastable, not yet monetizable" rather than dressed up as a win.

---

## Part VIII — Major Dates, and the Final Results

**The data window: April 2011 to March 2025**, spanning a market that grew
enormously over the period — FII participation breadth alone expanded
roughly 254 percent — and containing a genuine reporting-regime structural
break around **2021**, which the project masks out entirely (May-June
2021) before any feature window can span it, and which is deliberately
reused as the embargo gap between the training and test eras. **Training
era: all stock-days up to April 30, 2021 (509,185 stock-days, 572
stocks). Test era: all stock-days from July 1, 2021 onward (295,773
stock-days, 830 stocks)** — a substantially larger and different set of
names, not a repeat sample, which is what makes out-of-sample replication
a genuine test rather than an echo. Model parameters, state identities,
and overlay thresholds are all derived on the training era alone, frozen,
and never re-estimated on the test era.

**The final results, in one place:** concentrated distributive selling
predicts a **48 to 65 basis point** reversal within roughly a month,
holding out-of-sample and surviving bounce-correction, alternative-
explanation controls, and a formal statistical contrast against its
nearest archetype. Concentrated accumulation shows a symmetric give-back
of similar magnitude. Dispersed, quiet selling shows **no such reversal**
and is instead independently flagged by an external informed-trading
estimator as the more information-laden of the two selling patterns.
Standalone trading strategies built on these signals show one-way
breakeven costs of roughly **0 to 8 basis points**, against a realistic
transaction cost of about 15 basis points — meaning none of the tested
strategies clear costs as an independent trade.

---

## Part IX — Why This Is Valuable, Even Though 7-8bps Isn't Enough to Trade

It is tempting to read "breakevens of 2-8 basis points against 15 basis
points of realistic cost" as a negative result. That reading misses where
the value actually sits.

**First, this was never free money to begin with — and confirming that,
rigorously, is itself worth something.** A large fraction of applied
finance research quietly implies "and therefore you can trade this." This
project instead ran the exact gross-versus-net decomposition needed to
separate "is there a real signal" from "do costs kill it" — and reported
honestly that costs bind. That distinction protects anyone who might
otherwise have deployed capital against a backtest that looked profitable
gross and was not, net.

**Second, the value is real for anyone whose costs are already sunk.** A
desk instructed to sell a large position does not choose *whether* to
trade — it chooses *when*, inside a mandate it cannot avoid executing.
For that desk, the signal is free information, not a new strategy with its
own cost. Concretely: a desk instructed to sell roughly ₹100 crore of a
mid-cap currently caught inside a concentrated-selling episode gives up
close to the episode's own reversal — on the order of **37 to 49 basis
points**, worth roughly **₹37 to ₹49 lakh on that single order** — simply
by executing immediately instead of waiting for the concentrated flow to
exhaust itself. That is real money to a trading desk or portfolio manager,
at literally zero incremental cost, from a variable computable each
evening off data institutions already have.

**Third, it is an independent lens for risk and policy monitoring.** The
composition view inverts the usual panic ordering. Headline-grabbing,
visibly concentrated FII selling is, on this evidence, the *self-
correcting* kind. Quiet, broad, low-volume FII exits are the kind that do
not come back. A regulator or a risk desk watching only the aggregate
daily FII number cannot distinguish these; a regulator watching this
composition measure, computed from data it already possesses, can.

**Fourth, its scientific value does not depend on being tradeable at
all.** This is, as far as this project's literature review could
determine, the first daily-frequency, entity-composition-based
identification of the transitory/permanent price-impact split in an
emerging market, built from a data class that normally never leaves
regulators. Two reusable negative results travel with it: single-day
concentration snapshots are structurally invisible to a plain Hidden
Markov Model, and disclosed block-deal data measures a *visibility
threshold*, not true institutional concentration. And the project is a
rare, fully-documented worked example of the discipline that matters most
in applied machine learning — every claim carried a falsification test
that *could* have fired before it was seen, several of those tests
genuinely did fire along the way, and the two headline discoveries here
are survivals of deliberate attack, not products of search.

---

## Part X — Future Scope

Several directions are already staged in the repository rather than merely
imagined. A **Factorial Hidden Markov Model** — two independent latent
chains, one for direction and one for concentration, learned jointly rather
than imposed by a threshold rule — has already been built and gated as a
parallel challenger to the hand-tuned overlay, asking directly whether a
richer model class can replace the calibrated thresholds rather than needing
them. The causal, real-time layer (Phase II) opens a concrete path toward an
actual execution-timing tool, if the anticipation-versus-confirmation gap
can be closed with a better hazard model, additional causal features, or a
less naive timing control to beat. The illiquid-tail extension suggests the
same reversal exists further down the market-capitalization scale, at a
noisier signal-to-friction ratio — a natural target for improved execution-
cost modeling rather than a dead end. The overlay threshold itself was swept
only lightly (0.3/0.5/0.7) and could be revisited with a fuller robustness
grid, alongside a sensitivity check of the liquidity floor at lower trade-
count minimums. And the composition lens, once validated in one emerging
market, is a natural candidate for replication in other markets with
similarly granular depository-level foreign-flow data — a test of whether
the transitory/permanent split this project measures is a feature of Indian
market structure specifically, or of foreign institutional flow composition
more generally.

---

## Closing line for whichever speaker takes the last word

The headline FII number will keep making headlines. What this project shows
is that the headline number was never the right one to watch — and that,
for the first time, from data institutions already hold, we can tell the
difference between the kind of foreign selling that comes back and the
kind that does not.
