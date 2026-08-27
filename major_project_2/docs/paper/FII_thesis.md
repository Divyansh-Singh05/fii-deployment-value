# Foreign Institutional Flow Regimes in Indian Equities
## A complete account: motivation, construction, mathematics, validation, and economic meaning

*Long-form research document. Unlike the journal draft (`FII_paper_draft.md`),
nothing here is compressed: every design decision carries its thought process,
its mathematics, its reality check, and a citation to the code file, log, or
table in this repository where the claim can be verified.*

**Citation convention.** Code: `src/fii/...` (original Colab scripts preserved
byte-for-byte in `legacy/colab_modules/`). Printed evidence:
`outputs/validation/<stage>.log`. Tables/figures: `outputs/tables/`,
`outputs/figures/`. Narrative history: `docs/research_log/`. Every number in
this document regenerates via `python pipeline.py --all` (~8 minutes).

---

## Table of contents

**Part I — The question and the raw material**
1. The question, and why anyone should care
2. The data and its brutal realities
3. Feature engineering: ten numbers per stock-day, and why exactly these

**Part II — The model**
4. Why regimes, why an HMM, and the mathematics of hidden Markov models
5. Why the HMM alone could not do it: the negative result that fixed the architecture
6. Freezing, thresholds, and the calibration that falsified itself
7. Descriptive statistics and out-of-sample replication: quantifying "accuracy" for an unsupervised model

**Part III — Building prices you can trust**
8. The price panel: repairs, corporate-action mathematics, and gates
9. Identity: the ISIN closure problem

**Part IV — Economic validation**
10. Event studies: mathematics and the inference correction
11. The mechanism: a liquidity shock signed by volume
12. Panel regression: fixed effects, clustering, and the bad control
13. Robustness: overlap, horizons, dose–response, and the placebo that wasn't
14. Alternative explanations and the flow-surprise control (INNOV)
15. The machine-learning challenger: why LightGBM won the battle and lost the war
16. External validation: the PIN model
17. State dependence: the honest nulls (VIX, Kyle's λ)
18. Economic significance: backtests and limits to arbitrage

**Part VI — Phase II: from validated states to prediction (added on completion)**
23. Causal estimation, calibrated forecasting, and the decision layer

**Part V — The ledger**
19. Assumptions, in one place
20. Why this research is valuable, audience by audience
21. The failure log: everything that broke and how we knew
22. Conclusion

---
---

# Part I — The question and the raw material

## 1. The question, and why anyone should care

### 1.1 The starting observation

Every trading day, India's financial press reports a single number: net foreign
institutional investor (FII) buying or selling, in crores of rupees. It moves
markets, headlines, and policy conversations. And it is, we will argue, close
to the wrong number — because it collapses a *composition* into a *magnitude*.

Consider two days on which FIIs sell ₹300 crore of the same stock. On the
first, a single foreign desk unwinds a large position, demanding immediacy
from the market. On the second, forty distinct institutions each trim the name
slightly as part of broader portfolio adjustments. The headline number is
identical. The economics could not be more different: the first is a
*liquidity event* — a price concession extracted by someone who needs to trade
now — and standard microstructure theory (running from Kraus and Stoll's 1972
block-trade study through Grossman–Miller and Campbell–Grossman–Wang) predicts
the price impact should be partly *temporary*, reverting once the pressure
stops. The second aggregates many independent decisions; if those decisions
share an informational cause, the impact should be *permanent*.

The distinction is fifty years old in theory. What has been missing is the
ability to *observe* it: public data reports flow magnitude, never flow
composition. Nobody watching the daily FII number can tell a one-seller day
from a forty-seller day.

### 1.2 The data opportunity

This project exists because of a dataset that makes composition partially
observable: NSDL's transaction-level records of FII trades in Indian
equities, April 2011 – March 2025, with **masked but (within a month)
distinct entity identifiers**. For every FII trade we observe the date, the
ISIN, buy/sell, quantity, price — and a masked code for the foreign
institution, its sub-account, and its broker. The masking destroys *who* is
trading (deliberately, and, as §2 shows, far more thoroughly than we first
assumed), but it preserves, within each day, *how many* distinct institutions
participated and how the day's volume was distributed across them.

That is exactly the measurement the fifty-year-old theory needs: a daily,
stock-level observation of whether flow was **concentrated** (few entities,
skewed participation) or **dispersed** (many entities, flat participation).

### 1.3 The hypothesis as originally posed — and a promise about honesty

We began (see `docs/research_log/FII_Module1_findings.md`, §0) with three
behavioural archetypes, each grounded in a literature:

| Archetype | Economic story | Expected flow signature | Literature anchor |
|---|---|---|---|
| **Robot** | passive / index-rebalance flow | transient, small uniform slices, low persistence | index/passive flow studies |
| **Shark** | informed conviction accumulation | persistent, blocky, **concentrated** book | Kyle (1985)-style informed trading |
| **Hostage** | forced fire-sale under redemptions | persistent selling, fragmented, **dispersed** book | Coval–Stafford (2007) fire sales |

The pre-registered economic predictions were: Shark episodes should *drift*
(informed flow anticipates returns), Hostage episodes should *reverse* (forced
selling overshoots and recovers), Robot episodes should be *transient noise*.

**Both directional predictions turned out to be wrong, and the final result
is their inversion**: concentrated flow (the "Shark") behaves as liquidity
demand and *reverts*; dispersed selling (the "Hostage") behaves as
information and is *permanent*. We flag this at the outset for two reasons.
First, honesty: a reader should know the destination before judging the
route. Second, methodology: the fact that the validation machinery was strong
enough to overturn its designers' expectations — twice, as Part IV recounts —
is the strongest evidence we can offer that the machinery was not built to
confirm anything. The archetype names were kept as fossils of the original
hypothesis; the paper's claims rest on what the data showed, not on what the
names suggest.

### 1.4 Why a *regime* model rather than a daily signal

A natural first instinct is to compute a daily concentration number per stock
and correlate it with returns. We rejected that design for three reasons, each
of which shaped everything downstream.

**First, institutional flow is a campaign, not a coin flip.** Accumulation
programs and liquidations are worked over days to weeks. Empirically, our
persistence feature has lag-1 autocorrelation ≈ 0.93 (a 20-day window updated
daily), and the fitted regimes dwell 13–17 trading days
(`outputs/validation/hmm_train_oos.log`). A daily signal treats each day as
independent and throws away exactly the temporal structure — onset, duration,
exhaustion — that gives the events economic identity. The natural statistical
object for "a persistent latent condition emitting noisy daily observations"
is a hidden Markov model.

**Second, event anchors need episodes.** Our sharpest economic test (Part IV)
compares returns *after flow stops* — the reversal test. "After it stops" is
only defined if there is an "it": an episode with a start and an end. Regimes
give every stock-day a state, and every state run gives an END anchor date.

**Third, discipline against data snooping.** The regime model was built,
frozen, and calibrated **entirely on flow data** — no prices, no index
levels, no VIX, anywhere in the features (§3.6). Price data enters only
afterward, as an out-of-domain validation. Had we optimized a daily signal
against returns from the start, every downstream "finding" would be
in-sample fit dressed up as discovery.

### 1.5 Why this is worth doing — the value question, round one

We will re-answer "why is this valuable" at every stage, as promised. At the
stage of pure conception, the value proposition is: **a measurement that did
not exist**. The transitory/permanent decomposition of institutional price
impact is textbook theory with no daily-frequency, composition-based
identification anywhere in the emerging-market literature, because the
required data (entity-resolved institutional flow) essentially never leaves
regulators and depositories. If the measurement works, everyone downstream of
the daily FII number — execution desks deciding whether to trade into or
around foreign flow, risk managers deciding whether a decline in a holding is
noise or information, regulators deciding whether FII outflows warrant
concern — gains a lens they currently lack. If it fails, the failure itself
documents the limits of masked depository data. Either outcome is knowledge;
that asymmetry is what justified the build.

---

## 2. The data and its brutal realities

### 2.1 The raw feed

**Source.** One parquet per year, `data/ISIN_MAPPING/2011.parquet` …
`2025.parquet` (~450 MB compressed). Grain: one row per reported FII
transaction. Key fields: `FII`, `SUB_ACC`, `BRKER` (masked entity IDs),
`ISIN`, `TR_DATE`, `TR_TYPE`, `RATE`, `QUANTITY`, `VALUE_INR`,
`RFDE_INSTR_TYPE`.

**Scope filter** (applied in `src/fii/features/module1_feature_store_v2.py`;
rationale in `docs/research_log/FII_Module1_findings.md` §1): keep
`TR_TYPE ∈ {1 (buy), 4 (sell)}` **and** `RATE > 0` **and**
`RFDE_INSTR_TYPE = "REG_DL_INSTR_EQ"`. This is not a convenience filter; it
is a correctness filter. Transaction types 7/15/16/17 are corporate-action
legs — bonus credits, split adjustments — and are 100% zero-rate in the
feed; treating them as trades would fabricate enormous phantom "flow" on
every corporate-action date. The instrument filter drops non-equity
paper. What survives: **≈28 million** genuine equity buy/sell records.
Unattributable rows (null/empty ISIN) were 278 of 24M in the v1 build —
0.001%, dropped.

**Reality check that failed later, recorded here:** we initially assumed the
yearly parquets were one dataset among several; the ISIN-provenance audit
(`src/fii/validation/audits/module5g_isin_provenance.py`, log:
`outputs/validation/isin_provenance.log`) later established the full funnel —
**5,960** distinct raw FII-side ISINs → **3,812** canonical identities after
ISIN-lineage collapse → **946** canonical ISINs (**939** true companies) in
the final model universe. The 946 are the liquid subset: median 9,882 trades
per name versus 623 in the excluded tail. The scope this creates — the
model speaks about liquid large/mid-caps, ~25% of FII-traded names — is
stated here once and inherited by every result in this document. (We
originally hedged that fire-sale dynamics "may be stronger" in the
excluded tail; Module 14 later *measured* the attributable tail and
revised this: dislocations there are larger, but the reversal does not
exist — see §13.6.)

### 2.2 The entity audit: the test that nearly killed the project

Everything interesting in this project runs through the masked entity IDs.
Axis 3 — concentration — requires that "entity F387862664940" means *the same
institution* across the observations we aggregate. Before building anything,
we audited that assumption (`docs/research_log/FII_Module1_findings.md` §3).
The audit ran three iterations because our first assumption about the ID
format was itself wrong — a pattern that recurs throughout this project and
that we came to treat as the normal epistemic state: **assume your
assumptions are wrong until a test says otherwise.**

**Step 1 — format heterogeneity.** Character-shape profiling (map letters→A,
digits→9) revealed *six* coexisting ID families, not the single
`F + 10 digits + YYYYMM` pattern an initial sample suggested (that pattern
covers only ~15% of rows):

| Shape | Chars | Rows | Trailing YYYYMM matches report month? |
|---|---|---|---|
| `A999999999999` | 13 | 8.38M | no |
| `A9999999999999` | 14 | 5.44M | no |
| `A9999999999999999` | 17 | 4.29M | yes (~0.91) |
| `A999999999999999999` | 19 | 2.16M | ~0.77 |
| literal `"(null)"` | 6 | 1.73M | missing |
| true null | — | 1.96M | missing |

The scheme *changed over time*: an early era (~2011–2015) of 17/19-character
IDs of the form `PREFIX + core + YYYYMM` (strip the last six characters to
get identity), and a late era (~2021–2025) of plain 13/14-character IDs.
Missingness is ≈13% overall but heavily year-skewed: 0% in 2011 rising to
**30% (2024) and 39% (2025)** — a coverage confound we carry as a stated
limitation for all late-sample results.

**Step 2 — the retention test and why it is insufficient.** Define
month-to-month retention as the share of month-*m* entities that reappear in
month *m+1*. Measured with correct per-format identity, FII retention is
0.24–0.32. But retention *conflates two hypotheses*: "the institution didn't
trade next month" and "the institution's mask was re-minted." A low retention
number cannot distinguish them.

**Step 3 — the decisive test: months-per-entity, with a control group.** The
separation comes from a distributional statistic: for each distinct ID, count
the number of distinct months it ever appears in, over the era's full span.
And crucially, run it on a population whose *true* persistence is known:
**brokers**. Only ~150–320 brokers execute FII trades in any month — a
near-fixed real-world set. If IDs were stable, broker IDs should appear in
essentially *all* months of an 84–98-month span. The result:

| Era (span) | Level | distinct IDs | median months | max months | ≥12 months |
|---|---|---|---|---|---|
| Early (84 mo) | FII | 95,987 | 2 | 10 | **0.0%** |
| Early | BRKER (control) | 12,700 | 2 | 10 | **0.0%** |
| Late (98 mo) | FII | 250,949 | 1 | 2 | **0.0%** |
| Late | BRKER (control) | 9,221 | 2 | **2** | **0.0%** |

A ~300-broker real set fragments into 12,700 (early) / 9,221 (late) distinct
IDs, and **no ID of any kind, in any era, ever appears in twelve or more
months**. The conclusion is unambiguous: *masked IDs are re-minted on
roughly a monthly cycle; there is no stable cross-month entity identity in
this data, at any level.*

**What died, and what survived.** Cross-month entity tracking died — we can
never watch a specific fund liquidate over a quarter, which was the original
Coval–Stafford-style conception of the Hostage. What survived: within a
month, ID counts are sane (~150 brokers, ~3,000 FIIs per month), so IDs are
internally consistent *within* the month, and therefore **within a day**.
Axis 3 was rebuilt as a *single-day* concentration snapshot (§3.3), the
Hostage was re-conceived as "a stock-day whose sell-side book is dispersed,"
and the resulting loss of statistical power for that state was accepted and
documented in advance ("the fragile state"). Two design consequences
followed: entity level = `FII` (the umbrella best captures manager-level
dispersal; `SUB_ACC` as robustness), and a per-stock-day **ID-coverage
floor** (§3.3) so that days where too much value is unattributable are
excluded rather than mismeasured.

**Value at this stage, round two.** The audit is a contribution independent
of everything downstream: anyone who uses masked NSDL identifiers naively —
and the temptation is strong, because the IDs *look* persistent — will
manufacture spurious entity dynamics from mask re-minting. The
months-per-entity-with-a-control-group design is the test we would hand any
researcher facing a masked-panel dataset.

### 2.3 The base panel

From scoped trades, aggregation to the **(canonical ISIN, day)** grain
(`src/fii/features/module1_feature_store_v2.py`):

$$BUY_{it} = \sum \text{VALUE\_INR} \,[\text{TR\_TYPE}=1], \quad
SELL_{it} = \sum \text{VALUE\_INR} \,[\text{TR\_TYPE}=4],$$

plus buy/sell counts, total quantity, and the derived quantities

$$NET_{it} = BUY_{it} - SELL_{it}, \quad GROSS_{it} = BUY_{it} + SELL_{it},
\quad N_{it} = n^{buy}_{it} + n^{sell}_{it}, \quad
\overline{ts}_{it} = GROSS_{it}/N_{it}.$$

Every feature in §3 is a function of these plus the entity-level values. The
final store — `data/ISIN_MAPPING/stockday_features_v2.parquet` — contains
**2,423,212 stock-days across 3,812 canonical ISINs**, 2011-01-03 to
2025-03-28 (verified byte-identical on local regeneration:
`outputs/diagnostics/fingerprints_after.json`).

Two structural facts about this panel drove the normalization strategy in
§3.5: FII participation *breadth* grew ~254% over the sample, and there is a
visible structural break around 2021 (a reporting-regime change). May–June
2021 is masked outright — removed before any trailing window is computed —
so that no feature window spans the break; the same masked period later
serves as the embargo buffer around the model's train/test boundary (§6),
which is not a coincidence but a design alignment.

---

## 3. Feature engineering: ten numbers per stock-day, and why exactly these

Full column-level documentation: `docs/FII_stockday_data_dictionary.md`.
Construction code: `src/fii/features/module1_feature_store_v2.py` (stage
`feature_store`; log `outputs/logs/*_feature_store.log`).

### 3.1 The unit of analysis, defended

The row is a **stock-day**. This was a contested choice worth defending
explicitly (the full argument is §5 of the data dictionary):

1. **The label is a time-varying property of a stock.** "Reliance on
   2020-03-23 looks like a fire sale" is a statement about a stock on a day.
   A static per-stock label or a per-entity label cannot express it.
2. **The phenomena are day-granular.** Block deals and redemption dumps are
   day-level events; the trades are day-dated; weekly aggregation would blur
   exactly the climax days the features exist to catch.
3. **It is the only grain that unifies stock-centric and entity-centric
   archetypes.** Robot and Shark are stories about flow into a name. The
   Hostage is a story about an *entity* (a fund dumping its whole book) — but
   given the audit's verdict, entity trajectories are unobservable. The
   participation-weighted projection of entity book-concentration onto the
   stock-day (§3.3) is what lets all three archetypes inhabit one feature
   space. This projection is lossy, and the loss lands on the Hostage — the
   accepted, documented cost.
4. **Daily is not naively noisy**, because the panel is a hybrid: the output
   grain is daily, but two of the three model axes embed 20-day memory. A
   single wild day cannot dominate a stock-day's coordinates.

One honest cost, flagged before modelling and revisited in §5: pooling ~929k
complete-case stock-days into shared HMM emissions is
*observation-weighted* — hyper-active large-caps contribute more rows, so
fitted emissions tilt toward large-cap behaviour. Within-day ranking
mitigates this; it does not remove it.

### 3.2 The three axes — the theory of the feature set

No single number separates three archetypes; separation requires position on
three axes simultaneously:

- **Axis 1, persistence** — separates Robot (transient) from both persistent
  types. If flow direction is a campaign, its 20-day sign-consistency is high.
- **Axis 2, blockiness** — separates conviction (large prints) from
  fragmented execution (small uniform slices). Measured as *surprise against
  the stock's own norm*, not in absolute rupees, to avoid price-level and
  liquidity confounds.
- **Axis 3, entity concentration** — the hard one and the point of the whole
  dataset: are today's sellers (buyers) of this stock institutions acting
  *concentratedly* (their day's book is mostly this name) or *dispersedly*
  (this name is a sliver of a broad book)?

### 3.3 The ten features, exactly as constructed

All formulas below are lifted from the construction code; the mapping table
(raw → probit name) is at `src/fii/features/module1_feature_store_v2.py`,
lines ~175–184.

**Axis 1 — `F_persist`** (persistence):
$$\text{pers\_signed}_{it} = \Big[\tfrac{1}{20}\textstyle\sum_{k=1}^{20}
\operatorname{sign}(NET_{i,t-k})\Big], \qquad
\text{intensity}_{it} = \Big[\tfrac{1}{20}\textstyle\sum_{k=1}^{20}
\tfrac{|NET_{i,t-k}|}{GROSS_{i,t-k}}\Big],$$
$$\text{persistence\_raw}_{it} = \text{pers\_signed}_{it} \times
\text{intensity}_{it} \in [-1, 1].$$
The sign-mean captures directional consistency; the intensity scaler damps
stock-days where flow was two-sided (a +1 sign on a day that was 51% buys is
not a campaign). Note the `shift(1)`: the window ends *yesterday* — today is
never in its own baseline.

**Axis 2 — `F_block`** (blockiness surprise):
$$\text{blockiness\_raw}_{it} = \frac{\overline{ts}_{it}}
{\frac{1}{20}\sum_{k=1}^{20}\overline{ts}_{i,t-k}}.$$
Self-normalisation (today's mean trade size over the stock's own trailing
mean) makes the feature price-level-neutral and stock-size-neutral: a ₹2 cr
print is blocky for a midcap and routine for Reliance.

**Axis 3 — `F_entity` / `F_entity_buy`** (entity book concentration, the
dataset's raison d'être). For entity $e$ on day $d$, let $v(e,d,s)$ be $e$'s
sell value in stock $s$. The entity's **book Herfindahl** that day:
$$HHI(e,d) = \sum_s \left(\frac{v(e,d,s)}{\sum_{s'} v(e,d,s')}\right)^2
\in (0, 1],$$
equal to 1 if the entity sold exactly one name and →0 as its selling spreads
over many. The stock-day feature is the **participation-weighted average**
over the entities selling $s$ that day:
$$\text{entity\_hhi\_raw}(s,d) = \frac{\sum_e v(e,d,s)\cdot HHI(e,d)}
{\sum_e v(e,d,s)}.$$
> [!CAUTION]
> **CORRECTED (2026-08-23).** The formula above is what the code computed —
> and it is **not** the measure the rest of this thesis interprets. It is a
> participation-weighted average of each entity's *own book* concentration
> across stocks (book focus), whereas every economic claim reads it as the
> concentration of participation *within the stock-day*:
> $$\text{conc}(s,d) = \sum_e \left(\frac{v(e,d,s)}{\sum_{e'} v(e',d,s)}\right)^2 .$$
> `docs/02_feature_engineering.md` documented that second formula; this
> section documented the first; the code implemented the first. The two
> correlate **r = −0.2460**. Since the audit rebuild the HMM is fitted on the
> within-stock-day measure (`entity_conc_raw` → `F_entity_conc`), and book
> focus is retained, unfitted, as `bookfocus_raw` → `F_bookfocus`.
> All Axis-3 economic results in this thesis are withdrawn — see
> `docs/AUDIT_RETRACTIONS.md`.

High ⇒ the sellers hitting this stock today are *focused* on it
(the *book-focus* reading — **not** "concentrated selling", which is what the
within-stock-day measure captures); low ⇒ this stock is incidental to broad
selling programs. The buy-side analog (`entity_hhi_buy_raw`) is
constructed identically on buy values. **Coverage gate:** the feature is null
whenever less than 50% of the day's sell value carries a valid entity ID —
missing attribution must produce a missing feature, never a mismeasured one.
A critical late audit finding: coverage is *value-weighted*, and even in
2024–25 (30–40% of rows missing IDs) attributable *value* clears the floor
on essentially every stock-day, so Axis 3 survives the late sample.

**Supporting features** (context the three axes don't span):

- `F_breadth`: $n$ distinct entities trading the stock-day — crowdedness
  (Robot flow is broad). Later found to be the most characteristic-like
  feature (55% between-stock variance; §15).
- `F_sizedisp`: $\text{trade\_size\_std}/\overline{ts}$ — the coefficient of
  variation of print sizes; uniform VWAP-style slicing (Robot) vs lumpy
  discretionary prints (Shark).
- `F_activity`: $GROSS_{it}/\overline{GROSS}^{20d}_{i,t-1}$ — activity
  surprise, a regime-onset marker. Disclosed redundancy: correlation 0.88
  with `F_block` (both are "today vs own baseline" ratios).
- `F_streak`: signed length of the current run of same-sign NET days, as of
  *yesterday* (shifted). Correlation 0.56 with `F_persist` — related but
  distinct (a 3-day streak and a 15-day streak differ where the sign-mean
  may not).
- `F_imbal`: $NET_{it}/GROSS_{it}$ — today's directional imbalance, the
  "standby axis." Known artefact: probit ties at ±1 (fully one-sided days)
  compress its tails.
- `F_flowbeta`: the trailing co-movement of the stock's daily imbalance with
  the *aggregate* FII imbalance $\text{mkt\_imb}_d = \sum_i NET_{id} /
  \sum_i GROSS_{id}$ — does this name ride the broad FII tide (Robot) or
  march to its own drum?

### 3.4 What was deliberately left out — and why that is load-bearing

**No price. No index. No VIX. No macro.** Anywhere. This is the single most
important negative design decision in the project. The regime model's later
economic validation (Part IV) tests whether flow-defined states *predict
price behaviour*. That test is only meaningful if price information had no
path into the states. The features are flow-only, so the validation is
out-of-domain by construction, not by assertion.

**No flow-surprise (INNOV).** A residual from a flow-expectation model
(AAK-style) was designed early and *explicitly reserved as a validation
object* (`docs/research_log/FII_Module1_findings.md` §2.1): if the same
construct both fed the model and validated it, the validation would mark its
own homework. This decision pays off in §14, where INNOV serves as the
independent control that closes the last alternative explanation.

### 3.5 The normalization: within-day rank → probit, and the mathematics of why

Every raw feature is transformed, within each trading day $d$, over the
eligible cross-section of $n_d$ stock-days:
$$p_{it} = \frac{\operatorname{rank}_d(x_{it})}{n_d + 1} \in (0,1),
\qquad F_{it} = \Phi^{-1}(p_{it}),$$
where $\Phi^{-1}$ is the standard normal quantile (probit). Three distinct
problems are solved by this one transform:

1. **Non-stationarity.** FII breadth grew 254% and the 2021 break shifted
   levels of nearly every raw feature. Cross-sectional ranking makes each
   day self-referential: $F_{it}$ answers *"how unusual is this stock's flow
   relative to today's market"*, a question whose meaning is constant across
   fourteen years. Formally, if $x_{it} = g_t(\tilde{x}_{it})$ for any
   day-specific monotone distortion $g_t$ (level shifts, scale drifts,
   reporting-regime changes), the rank — and hence $F$ — is invariant to
   $g_t$. We buy stationarity by construction and pay by forfeiting
   market-wide level information (each day's census shares are pinned by the
   quantile cut — an accepted cost, disclosed when interpreting census
   stability in §7).
2. **Emission honesty for the HMM.** Part II fits Gaussian emissions. The
   probability integral transform guarantees $p_{it}$ is uniform on (0,1)
   under the within-day empirical distribution, hence $F_{it} =
   \Phi^{-1}(p_{it})$ has *exactly* standard normal marginals — no heavy
   tails, no skew, no outlier leverage. Fitting a Gaussian HMM to raw
   right-skewed HHIs or value ratios would hand the likelihood to a few
   extreme days; fitting it to probits is honest by construction. The
   empirical check (`docs/FII_stockday_data_dictionary.md` §4.3): post-probit
   means ≈ 0 (−0.042, 0.023, −0.001), standard deviations ≈ 0.97, quartiles
   ≈ ±0.67 — the theoretical N(0,1) values.
3. **Outlier robustness.** Ranks are bounded; a data error in one raw value
   moves one rank slot, not a moment.

**Eligibility and null semantics.** Ranking runs over stock-days with
$N \ge 5$ trades (the liquidity floor — a "concentration" measured off two
prints is noise). A feature is null when the day is illiquid, in a stock's
~15-day warm-up, or (Axis 3) below the coverage floor. Nulls are *honest
missing values* that the model masks — a hygiene bug where warm-up NaNs
masqueraded as present values (poisoning correlation and description
statistics) was caught and fixed in the v1 build, recorded in the dictionary
§4.3.

### 3.6 Feature validation: what we checked before any model saw the data

The feature store was validated *statistically* before modelling
(economics comes later and is earned separately):

- **Marginals** ≈ N(0,1) as above (probit correctness, no tie artefacts).
- **Independence.** The three model axes are near-orthogonal — pairwise
  correlations $\rho(\text{persist},\text{block}) = 0.015$,
  $\rho(\text{persist},\text{entity}) = -0.142$,
  $\rho(\text{block},\text{entity}) = 0.195$ — and the two non-zero terms
  point the economically *correct* way: concentrated players also print
  bigger blocks (+0.19); dispersed sellers skew sell-persistent (−0.14).
  Axes that were highly correlated would make the three-axis story an
  illusion of one axis.
- **Known redundancies disclosed, not hidden:** activity×block $r=0.88$;
  streak×persist $r=0.56$; imbalance tie-compression. These mattered later —
  the GBT challenger (§15) had access to all ten features precisely so that
  the "did the HMM leave signal behind" question included the redundant and
  standby features the HMM never used.
- **Sample accounting.** 2,423,212 stock-days total; ~1.15M pass the
  liquidity floor; **929,195 complete cases (80.7% of eligible)** on the
  three v1 axes; the ~19% gap decomposes into warm-up and coverage, both
  measured. The final complete-case store for the 10-feature v2 build feeds
  the model universe of §2.1 (946 canonical ISINs after a minimum-sequence
  filter of 60 stock-days per name, verified count-integral in
  `outputs/validation/universe_audit.log`... see Part II).

**Value at this stage, round three.** Even with no model attached, the
feature store is infrastructure: a leakage-disciplined, stationarised,
entity-audit-gated panel of FII flow microstructure across fourteen years.
Every design element — the shift(1) convention, the mask-then-window rule,
the value-weighted coverage gate, the probit marginals — is reusable by any
successor project on this data, and each exists because a specific failure
mode (lookahead, break-spanning windows, attribution decay, emission
misspecification) was identified *before* it could contaminate results.

*— End of Part I. Part II (§§4–7) covers the HMM mathematics, the negative
result that forced the hybrid architecture, the threshold calibration that
falsified itself, and the descriptive statistics that quantify replication.*

---
---

# Part II — The model

## 4. Why regimes, why an HMM, and the mathematics of hidden Markov models

### 4.1 The modelling bet, stated precisely

§1.4 argued informally that institutional flow is a campaign. The formal
version of that bet: a stock's daily flow-feature vector
$\mathbf{x}_t \in \mathbb{R}^4$ is generated by a **latent state** $S_t$
that (i) takes few values, (ii) persists over days, and (iii) determines the
*distribution* of the observables, not their exact values. If that structure
is real, a hidden Markov model recovers it; if it is not, the HMM's own
diagnostics — degenerate states, one-day dwell times, non-replicating
signatures — expose the failure. Part of why we chose an HMM over softer
clustering is exactly that it fails loudly.

### 4.2 The generative model

For each stock, an unobserved chain $S_1,\dots,S_T \in \{1,\dots,K\}$:

$$P(S_1 = i) = \pi_i, \qquad P(S_t = j \mid S_{t-1} = i) = A_{ij},$$

with first-order Markov dependence (the future depends on the past only
through the present state), and conditionally independent emissions

$$\mathbf{x}_t \mid S_t = j \;\sim\; \mathcal{N}(\boldsymbol{\mu}_j,\,
\Sigma_j), \qquad \Sigma_j \text{ diagonal}.$$

Three model-class decisions and their reasons:

**Gaussian emissions** are honest here *only because of* the probit
normalization (§3.5): each feature has exactly standard-normal marginals by
construction, so a Gaussian emission family is not an approximation imposed
on skewed data but a shape the data actually has.

**Diagonal covariance** ($\Sigma_j = \mathrm{diag}(\sigma^2_{j1},\dots,
\sigma^2_{j4})$) was chosen for rare-state estimability. A full covariance
per state costs $\binom{4}{2}=6$ extra parameters per state; a state
occupying ~7% of the panel must estimate its covariance from proportionally
few effective observations, and off-diagonal terms estimated from thin
samples destabilize EM. The cost is that within-state feature correlations
are ignored — acceptable because the features were built near-orthogonal
(§3.6: pairwise $|\rho| \le 0.195$).

**The feature subset.** Only four of the ten features feed the HMM:
`F_persist`, `F_block`, `F_entity_s`, `F_entity_buy_s` — one direct
signature per hypothesized archetype (see
`docs/research_log/FII_Module2_hmm_log.md` §0). `F_activity` was excluded
for its 0.88 correlation with `F_block` (near-duplicate features double-count
evidence in a likelihood), `F_streak` for its 0.56 correlation with
`F_persist`; the rest were held in reserve. This exclusion decision is what
later gives the LightGBM challenger (§15) its purpose: the GBT gets all ten,
so "did the HMM's four-feature diet leave predictive signal on the table?"
becomes a testable question rather than a regret.

### 4.3 Likelihood, forward–backward, and Baum–Welch

The likelihood of an observed sequence marginalizes over all $K^T$ state
paths — naively intractable, and the reason the forward recursion exists.
Define the **forward variable** $\alpha_t(j) = P(\mathbf{x}_{1:t}, S_t=j)$
and the emission density $b_j(\mathbf{x}) =
\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_j,\Sigma_j)$. Then

$$\alpha_1(j) = \pi_j\, b_j(\mathbf{x}_1), \qquad
\alpha_{t+1}(j) = b_j(\mathbf{x}_{t+1}) \sum_{i=1}^{K} \alpha_t(i) A_{ij},
\qquad P(\mathbf{x}_{1:T}) = \sum_j \alpha_T(j),$$

computed in $O(K^2 T)$. The **backward variable**
$\beta_t(i) = P(\mathbf{x}_{t+1:T} \mid S_t = i)$ satisfies the mirror
recursion, and together they give the two posteriors EM needs:

$$\gamma_t(i) = P(S_t = i \mid \mathbf{x}_{1:T}) =
\frac{\alpha_t(i)\beta_t(i)}{\sum_k \alpha_t(k)\beta_t(k)}, \qquad
\xi_t(i,j) = \frac{\alpha_t(i)\, A_{ij}\, b_j(\mathbf{x}_{t+1})\,
\beta_{t+1}(j)}{\sum_{k,l} \alpha_t(k) A_{kl} b_l(\mathbf{x}_{t+1})
\beta_{t+1}(l)}.$$

**Baum–Welch is EM.** With latent states, the complete-data log-likelihood
is linear in the indicators $\mathbb{1}[S_t=i]$ and
$\mathbb{1}[S_{t-1}=i, S_t=j]$; the E-step replaces them with their
posteriors $\gamma$ and $\xi$; the M-step maximizes the resulting expected
log-likelihood in closed form:

$$\hat{A}_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad
\hat{\boldsymbol{\mu}}_j = \frac{\sum_t \gamma_t(j)\,\mathbf{x}_t}
{\sum_t \gamma_t(j)}, \qquad
\hat{\sigma}^2_{jd} = \frac{\sum_t \gamma_t(j) (x_{td}-\hat{\mu}_{jd})^2}
{\sum_t \gamma_t(j)}.$$

Each iteration provably does not decrease the likelihood; convergence is to
a **local** optimum. Two practical consequences we engineered around
(`src/fii/models/hmm_stages/module3a_model_split_oos.py`): multiple random
initializations (`N_INITS` restarts, keep the best log-likelihood; in the
Module-2 fit two of five inits converged to the same optimum at
−1,424,681 — reassuring), and a fixed seed per init so the entire fit is
deterministic — which is why the local re-run of the pipeline reproduced
the Colab census to within 3 rows in 804,958 (§7.4).

**Decoding.** Reported states are the Viterbi path — the single most
probable state *sequence*, via dynamic programming on
$\delta_{t+1}(j) = b_j(\mathbf{x}_{t+1}) \max_i \delta_t(i) A_{ij}$ with
backtracking. Viterbi (rather than pointwise argmax of $\gamma_t$) respects
the transition structure: it will not produce one-day state flickers that
the transition matrix says are improbable — the correct choice when the
downstream object is the *episode*.

**Label switching.** State indices are arbitrary (the likelihood is
invariant to permuting states), so states are labeled **post hoc from their
fitted signatures** — explicitly, the state whose mean `F_persist` is most
negative is SELL_REGIME, most positive is BUY_REGIME
(`module3a_model_split_oos.py` line ~101). Module 2's labeling code went
further and *refused to bless* labels whose signatures contradicted the
archetype definitions — a warning system that, as §5 recounts, fired
immediately and correctly.

### 4.4 Fitting hygiene: the fit-cap and its own bug-fix

Pooling every stock-day into EM would let hyper-active large-caps dominate
the emissions (§3.1). The mitigation: **fit balanced, decode everything** —
each stock contributes at most 400 complete rows to the *fitting* sample
(278,462 rows from 776,068 in the Module-2 fit), then the fitted model
decodes all full sequences. The v1 implementation of the cap took each
stock's *most recent* 400 rows, quietly biasing emissions toward late-sample
years; Module 3A replaced it with a **random contiguous block** of ≤400 days
per stock (`docs/research_log/FII_Module3_validation_log.md` §1). Contiguity
matters: EM learns the transition matrix from within-sequence adjacencies,
so random *rows* would destroy exactly the temporal structure the model
exists to estimate.

---

## 5. Why the HMM alone could not do it: the negative result that fixed the architecture

This section answers "why choose four regimes" — and the honest answer is
that we did not choose them; we *discovered* the regime space's true
dimensionality by watching a sequence of HMMs fail in an informative
pattern. The full record is `docs/research_log/FII_Module2_hmm_log.md`;
the exploratory scripts are preserved in `legacy/colab_modules/`
(`module2_v1` … `module2_v4`).

### 5.1 The pre-registered criterion

Before the first fit, the success condition was written down: *the Hostage
exists as a regime iff some fitted state shows clearly negative mean
`F_persist` AND clearly negative mean `F_entity` — persistent, dispersed
selling.* Pre-registering this is what prevented rationalizing whatever
clusters EM happened to produce.

### 5.2 First fit: the model found direction, not archetypes

The 3-state fit (`legacy/colab_modules/module2_v1_hmm_first_fit.py`)
produced states with mean `F_persist` of +1.17, +0.01, −1.14 — a
**buy / neutral / sell triad** — and the labeling code's contradiction
warnings fired on both directional states. The reason is visible in the
magnitudes: persistence means of ±1.15 dwarf entity means of ±0.2, so the
likelihood surface is dominated by one axis. Two economically real facts
were extracted from the failure: the persistent-sell state's *sellers were
concentrated on average* (+0.17) — meaning it mixed dispersed and
concentrated selling rather than separating them — and dispersed selling
co-occurred with persistent *buying* (−0.22 in the buy state): broad-book
entities are the liquidity **providers** during accumulation. Dispersion
alone is not distress; the Hostage requires dispersion *and* sell pressure
jointly.

### 5.3 Dissection and the k-sweep: the mixture exists, EM will not carve it

Was the material even there? Dissecting the 220,481 persistent-sell
stock-days: 25.8% had dispersed books (`F_entity` < −0.5), 36.8%
concentrated (> +0.5) — a genuine mixture the model was not separating. A
BIC sweep over $k = 2\ldots6$
(`legacy/colab_modules/module2_v2_dissection_bic_sweep.py`) then showed why
more states would not help. On $n \approx 2.7$M observations, BIC
($-2\log L + p\log n$) declines monotonically in $k$ — the big-$n$ regime
where the $\log n$ penalty is negligible against any real likelihood gain —
with a true elbow at $k=5$. But the $k=5$ and $k=6$ solutions were a
**persistence ladder** (state means −1.52, −0.66, ≈0, ≈0, +0.66, +1.54 on
one axis, plus one-day transient noise states): finer gradations of
direction, never a persist<0 ∧ entity<0 state. The pre-registered criterion
failed at every $k$.

### 5.4 The mechanism, diagnosed before re-engineering

Why is the entity axis invisible to EM? The hypothesis: **an HMM state must
be temporally coherent** — EM allocates states along features whose values
persist, because persistent features are what make the Markov chain's
self-transition probabilities informative. The test — median per-stock lag-1
autocorrelation of each input
(`legacy/colab_modules/module2_v3_autocorr_smoothing_refit.py`):

| feature | lag-1 autocorr |
|---|---|
| `F_persist` | **0.926** |
| `F_block` | 0.290 |
| `F_entity` (single-day snapshot) | 0.334 |
| `F_entity_buy` | 0.295 |

We had fed the model one ultra-smooth feature and three jittery ones. The
entity-audit's single-day constraint (§2.2) had not merely weakened Axis 3 —
it had made it *structurally invisible to this model class*.

**The fix, and why it is audit-legal:** `F_entity_s`, `F_entity_buy_s` =
5-day trailing means of the *daily HHI snapshots*, re-ranked within day and
re-probited. This averages stock-level dispersion **measurements** across
days; it requires no cross-day entity identity whatsoever, so it lives
within the audit's constraint. Lag-1 autocorrelation rose to 0.767/0.763,
and the entity axis became HMM-visible.

### 5.5 The k=4 fit: the regime space is two-dimensional

The refit sweep on the smoothed features produced the single most
informative fit of the project. At $k=4$, EM kept the two directional poles
and split the *neutral* middle by **participant structure**: a
concentrated-both-sides state (entity means +0.56/+0.78, dwell 14 days)
versus a dispersed-both-sides state (−0.45/−0.67, dwell 9.5 days). The
regime space is **flow direction × participant concentration** — genuinely
two-dimensional — and the four *occupied* cells were sell/concentrated,
neutral/concentrated, neutral/dispersed, buy/mildly-dispersed. The two
**empty corners were exactly the original hypotheses**: sell/dispersed (the
Hostage) and buy/concentrated (the accumulating Shark). EM allocates states
by probability mass; rare corners never earn one.

Hence the Module-2 headline finding, which the pre-registered criterion
certified at every $k$, both raw and smoothed:

> Persistent selling in Indian FII flow is, on average,
> concentrated-seller selling. The Coval–Stafford Hostage — persistent
> *and* dispersed selling — does not exist as a temporal regime at the
> stock-day level. Hostage days exist (~26% of persistent-sell days), but
> they are **episodic**, not regime-forming.

### 5.6 The hybrid architecture — and why "four archetypes" is the honest count

The pre-registered fork for this outcome: use the HMM for what it *can* see
(temporal flow direction) and overlay deterministic rules on the entity
coordinates for what is episodic
(`legacy/colab_modules/module2_v4_final_hybrid_overlay.py`; production
implementation `src/fii/models/hmm_stages/module3a` + `module3b`):

| Label | Rule |
|---|---|
| HOSTAGE | SELL_REGIME ∧ `F_entity_s` < threshold₋ |
| SHARK_DIST | SELL_REGIME ∧ `F_entity_s` > threshold₊ |
| SHARK_ACC | BUY_REGIME ∧ `F_entity_buy_s` > threshold₊ᵇ |
| ROBOT | NEUTRAL backbone state |
| UNTAGGED_DIRECTIONAL | directional regime, unremarkable entity structure |

So the direct answer to "why four regimes": **three** is the backbone count
(the only temporal structure the data supports — direction), **four** is
the archetype count because concentration adds one binary distinction on
each directional pole that the data occupies (concentrated-sell,
dispersed-sell, concentrated-buy; dispersed-buy never formed even an
episodic cluster), and there is deliberately a **fifth non-label** —
UNTAGGED, ~30–36% of stock-days — the honest "directional but unremarkable"
bucket that prevents force-fitting every day into a story. Architectures
that lack an "unremarkable" class manufacture archetypes out of noise.

Economically, the hybrid is arguably *truer* than the original design: fire
sales are **episodes within selling regimes**, not regimes themselves — a
fund's redemption dump lasts days, embedded in longer market-level selling
conditions.

**Post-review addendum — the ablation that settles the HMM's role.** A
referee asked the right question: is the HMM central, or sophisticated
machinery on a simpler idea? We tested it
(`src/fii/validation/module13a_backbone_ablation.py`, stage
`backbone_ablation`): a three-line rule backbone — TRAIN-era `F_persist`
quantile cuts (−0.510/+0.539), census-matched to the HMM so both tag the
same fraction of days — with the identical frozen overlays, reproduces the
paper's main table within band on every cell (rule SHARK_DIST
+71.6***/+47.4*** vs HMM +65.4***/+48.5***; SHARK_ACC −78.0***/−52.3*** vs
−87.9***/−47.6***; label agreement 92.7%, Cohen's κ = 0.891). What the HMM
buys is **episode smoothness, not identification**: rule-based runs are
~14% shorter (no transition-matrix persistence prior), producing ~48% more
episode-ends with the same economics. The honest conclusion, adopted
throughout this document: *the contribution is the participant-composition
measure; the HMM is a taxonomy convenience* — retained for its cleaner
episode segmentation, claimed for nothing more. The finding is
model-class-independent.

### 5.7 Face validity: do the labels find real events?

Before any price data touched the project, the longest HOSTAGE episodes
were mapped to companies and dates
(`docs/research_log/FII_Module2_hmm_log.md` §6). The top of the list: JSW
Steel, 53 days, starting seven days after Bernanke's May 2013 taper speech;
Vedanta, 42 days, during the 2015 commodity crash; HDIL and Reliance
Communications — **both later bankrupt** — in their documented distress
windows; Ambuja Cements in the demonetization aftermath. Two reads: the
episodes land on canonical macro-stress windows *despite* within-day
ranking removing the market-wide component (these are the names sold
hardest relative to an already-stressed market — precisely Coval–Stafford's
prediction that fire sales concentrate in commonly-held names), and the
label finds genuine distress names years before their failures. Face
validity is not proof — it is a necessary condition that, had it failed,
would have ended the project cheaply.

---

## 6. Freezing, thresholds, and the calibration that falsified itself

### 6.1 The temporal protocol

Everything downstream rests on one discipline
(`src/fii/models/hmm_stages/module3a_model_split_oos.py`; frozen dates in
`config/config.yaml`): **train = stock-days ≤ 2021-04-30; test =
≥ 2021-07-01; May–June 2021 masked** (the same window already excised from
the features — the structural break doubles as the embargo buffer, so no
trailing window and no fitted parameter ever touches the boundary). HMM
parameters, state identities, and overlay thresholds are derived on TRAIN
only, frozen, and applied unchanged to TEST. The test era is not "more
data": it is 295,773 stock-days across **830 stocks versus TRAIN's 572** —
a ~45% larger and substantially different universe, which makes replication
a real test rather than an echo.

### 6.2 Why ±0.5 had to die

Module 2's overlay threshold (±0.5 probit ≈ 69th percentile) was an
eyeballed placeholder. A threshold defines the archetypes, hence every
downstream event study and regression; leaving it arbitrary would put an
unexamined free parameter under the whole edifice. Module 3B's design
(`src/fii/models/hmm_stages/module3b_threshold_calibration.py`): fit a
BIC-selected Gaussian mixture to the distribution of `F_entity_s` *within
the TRAIN sell regime*, take the posterior-0.5 boundary of the dispersed
component as the HOSTAGE threshold — **with two falsification checks
pre-registered**: a k-means cross-check, and a stability requirement that
re-deriving the boundary on TEST move it by less than 0.25.

### 6.3 What happened: the calibration failed its own test

The TRAIN GMM selected $k=2$ — but with weights **49%/51%** at means
−0.50/+0.84: a coin-flip bisection of the sell regime, not the
rare-tail-versus-majority structure a fire-sale component should have. Its
boundary, **+0.155**, would have labeled mildly *concentrated* days as
HOSTAGE, doubled the Hostage census to ~14%, and eliminated the honest
middle (every sell-regime day would have become either HOSTAGE or
SHARK_DIST). Every independent check disagreed with it *and agreed with
each other*: the k-means boundary was **−0.441**; the TEST-era GMM found a
proper rare tail (weight 0.24, mean −0.84) with boundary **−0.510**; the
stability delta, 0.665, tripped the pre-registered UNSTABLE flag. The
fallback engaged per protocol: **the quantile rule** — the TRAIN 25th
percentile of `F_entity_s` within the sell regime:

$$\text{thr}_{H} = q_{0.25}\big(F^{entity\_s}\,\big|\,\text{SELL, TRAIN}\big)
= -0.513,$$

with the symmetric $q_{0.75}$ rules giving SHARK_DIST > **+0.877** and
SHARK_ACC > **+0.795**. The three-way convergence — k-means −0.441,
test-GMM −0.510, train-quantile −0.513, all within 0.07 of each other and
of the original eyeballed −0.5 — is what elevates the threshold from
"assumed" to "derived, with convergent evidence from methods with different
assumptions."

Per-threshold stability of the final rule (train- vs test-quantile):
HOSTAGE −0.513 vs −0.331 (Δ0.18, and the drift direction carries the
2024–25 ID-coverage fingerprint, §2.2); SHARK_DIST Δ0.09; SHARK_ACC
Δ0.003. The calibrated census restores the honest middle: ROBOT 42.9/43.6%,
UNTAGGED 35.5/35.9%, SHARK_DIST 7.3/8.0%, SHARK_ACC 7.0/7.0%, HOSTAGE
7.3/5.6% (train/test; the Hostage test-dip is the coverage confound, not
economics). Full numbers: `outputs/validation/threshold_calibration.log`,
`outputs/tables/T2_archetype_census.csv`.

**Did the calibrated threshold "improve accuracy"?** The right claim is
sharper: it prevented a *specific, quantified* mislabeling (a +0.155 cut
mislabels mildly-concentrated days as dispersed fire-sales and doubles the
HOSTAGE census with days that contradict its definition), and the key
downstream statistic proved robust to the choice — the episode-clustering
ratio (§7.3) held at ~2.6× even under the rejected threshold. We therefore
do not claim the threshold manufactured the result; we claim it made the
labels mean what their definitions say.

### 6.4 A provenance lesson learned in this repository

One engineering honesty note. The original Colab session *observed* the
UNSTABLE flag and applied the quantile fallback manually; the script as
saved only printed the warning. When this pipeline was first re-run locally,
the calibration silently regenerated the *bad* GMM thresholds — caught
because the archetype census diverged from the frozen fingerprint
(`outputs/diagnostics/fingerprints_before.json`). The falsification rule is
now **binding in code** (`module3b_threshold_calibration.py`: Δ ≥ 0.25 ⇒
quantile rule for all three thresholds), and the local re-run reproduces
the frozen thresholds to the third decimal. The episode is preserved
because it is the project's thesis in miniature: decisions that live in a
person's memory instead of in code are not reproducible, and only a
fingerprint gate caught the difference.

---

## 7. Descriptive statistics and out-of-sample replication: quantifying "accuracy" for an unsupervised model

### 7.1 What "accuracy" can even mean here

A supervised model has ground truth; an unsupervised regime model does not —
there is no oracle file of true fire-sale days. "Accuracy" must therefore be
decomposed into falsifiable sub-claims, each with its own test:

1. **Structural stability** — do the fitted states mean the same thing on
   unseen data? (§7.2)
2. **Statistical reality of the labels** — do labeled days cluster in time
   like genuine episodes rather than i.i.d. noise? (§7.3)
3. **Reproducibility** — does an independent re-run produce the same
   labels? (§7.4)
4. **External validity** — do the labels align with events (face validity,
   §5.7) and with independent constructs (PIN, Part IV §16)?
5. **Economic content** — do the labels *predict price behaviour out of
   sample*? — the entirety of Part IV, and the only test that can make the
   labels matter.

The claim "the model is 93% accurate" is unavailable and would be
meaningless; the claim "every measurable property of the model replicates
on a disjoint era and a 45% different universe" is available, and is what
this section documents.

### 7.2 Backbone replication (log: `outputs/validation/hmm_train_oos.log`)

The frozen TRAIN-fitted model, decoding both eras:

| Check | TRAIN | TEST (frozen parameters) |
|---|---|---|
| SELL signature (persist / entity_s) | −1.04 / +0.18 | −1.07 / +0.30 |
| BUY signature (persist / entity_s) | +1.08 / −0.299 | +1.13 / −0.304 |
| Census (sell / neutral / buy) | 29.1 / 42.9 / 28.0 | 28.5 / 43.6 / 27.9 |
| Transition diagonal | 0.95 / 0.95 / 0.95 | 0.95 / 0.95 / 0.95 |

Signatures and occupancy replicate almost digit-for-digit across disjoint
years on a substantially different stock universe. Two caveats we impose on
ourselves, one per non-signature row:

- **Census** is stable partly *by construction* — within-day ranking pins
  daily cross-sectional shares, so this row cannot fail to look stable and
  carries little independent evidence either way.
- **Transition diagonal** is *circular*, via a different mechanism, and
  this was not caught until pushed on directly: the TEST-era 0.95 comes
  from Viterbi-decoding TEST with the same frozen transition matrix fit on
  TRAIN, which already encodes ~0.95 self-transition as a prior. A strongly
  sticky prior mechanically biases any Viterbi decode toward looking
  sticky, almost regardless of the TEST data's true persistence — so "0.95
  replicated" is largely restating the TRAIN-fitted input, not an
  independent TEST-era measurement. Isolating the prior's contribution
  (`scripts/check_test_era_stickiness.py`; four methods compared, including
  a per-day classification using only the frozen TRAIN emission
  parameters with no transition matrix at all — the one measurement here
  that is simultaneously look-ahead-free and untouched by any TEST-fit
  parameter) puts **genuine, TEST-native persistence at 0.888**, not 0.95,
  against a 0.346 no-persistence chance floor. Real and substantial
  (2.6× chance), but the frozen prior adds a further +0.061 on top —
  over half the remaining headroom to a perfectly sticky 1.0 — that TEST
  data does not independently confirm. (A tempting-looking fix, refitting
  the transition matrix by EM directly on TEST data, was tried and
  rejected: Baum-Welch is forward-*backward*, so it uses the entire TEST
  period at once — look-ahead within the test era itself — on top of
  which it violates the frozen train/test discipline this paper enforces
  everywhere else by fitting a parameter on TEST data at all.)

The only row that is both non-tautological and non-circular is the
signature row.

### 7.3 Effect sizes and the permutation test (log: `outputs/validation/model_descriptives.log`)

With $n \approx 800{,}000$, p-values are decoration — everything is
"significant." The battery is therefore **effect-sizes-first**, using
Cohen's $d = (\bar{x}_1 - \bar{x}_2)\big/ s_{pooled}$:

| Contrast | F_persist | F_block | F_entity_s | F_entity_buy_s |
|---|---|---|---|---|
| HOSTAGE vs SHARK_DIST | +0.22 → +0.26 | −0.19 → −0.38 | *(tautological)* | −0.46 → −0.55 |
| SHARK_ACC vs ROBOT | **+3.19 → +3.20** | ~0 | ~0 | **+1.52 → +1.42** |
| HOSTAGE vs ROBOT | **−2.98 → −3.01** | −0.19 → −0.34 | **−1.40 → −1.43** | −0.74 → −0.71 |

(Arrows: TRAIN → TEST.) The flagged caveat comes first: the
HOSTAGE-vs-SHARK_DIST gap on `F_entity_s` itself ($d≈−5.7$, KS $D=1.0$) is
**tautological** — those labels are *defined* by a cut on that variable —
so it is excluded from evidence. What counts: (i) every non-tautological
effect size moves by ≤0.15 between disjoint eras — the label structure
generalizes; (ii) within the sell regime, dispersed and concentrated
sellers are near-identical on persistence and blockiness — the entity axis
carries independent information, as designed; (iii) a small, correctly
signed bonus: HOSTAGE prints are *smaller* than SHARK_DIST prints (block
$d$ = −0.19/−0.38) — fire-sale fragmentation versus block distribution,
exactly the original framework's sign, honestly reported as small.

**The episode-clustering permutation test** asks: do HOSTAGE days chain
into episodes, or would i.i.d. tagging at the same rate produce the same
runs? Null distribution: 200 within-stock shuffles of the label sequence —
preserving each stock's label *count* while destroying temporal adjacency —
with mean run length as the statistic; $p = (1 + \#\{null \ge obs\})/(1+B)$:

| Era | Observed mean run | Shuffled null | Ratio | p |
|---|---|---|---|---|
| TRAIN | 3.64 d | 1.47 d | **2.48×** | 0.005 |
| TEST | 3.39 d | 1.42 d | **2.38×** | 0.005 |

Replicating out of sample, and robust to the threshold choice (≈2.6× even
under the rejected +0.155 cut). Episode-length distributions are
economically sensible: the active archetypes come in 3-day-median bursts
with heavy tails (max HOSTAGE run: 53 days — the JSW Steel taper-tantrum
event); ROBOT dwells for weeks (median 11 days, max 854).

### 7.4 Reproducibility as a measured quantity

The entire chain — features, HMM fit, calibration — was re-executed from
raw data on different hardware, a different OS, and a different numpy/
hmmlearn generation than the original Colab runs
(`outputs/diagnostics/fingerprints_after.json`): the feature store
regenerated **byte-identical** (2,423,212 rows / 3,812 cisins); the
calibrated archetype census matched **exactly on all four active
archetypes** (HOSTAGE 53,613; SHARK_DIST 60,711; SHARK_ACC 56,275), with
three stock-days in 804,958 flipping between the two control categories
(ROBOT/UNTAGGED) — EM tie-breaking at machine precision, $4\times10^{-6}$
of the panel; and the downstream headline regression reproduced to ±0.1 bp
(§12). For an unsupervised pipeline with an EM fit inside it, this is the
strongest available form of the "accuracy" claim: **the model is a
deterministic function of the data and the frozen protocol, not of the
session that happened to produce it.**

**Value at this stage, round four.** Even a reader who rejects every
economic claim in Part IV inherits, from Parts I–II: an audited masked-ID
dataset; a stationarised, leakage-disciplined feature store; the negative
result that single-day concentration snapshots are HMM-invisible (with the
autocorrelation diagnosis and the audit-legal smoothing fix — directly
reusable by anyone modelling regime structure on jittery microstructure
features); the demonstration that the regime space of institutional flow is
direction × concentration with empty rare corners; and a calibration
protocol whose falsification test fired and was survived. These are
methods-contributions independent of whether concentrated FII selling
reverts.

*— End of Part II. Part III (§§8–9) builds the price panel: the false
assumption about NSE's `prev_close`, the corporate-action mathematics and
its gates, the symbol-migration guard, and the issuer-bounded ISIN closure
that recovered 8% of the panel.*

---
---

# Part III — Building prices you can trust

*The regime model never touched price data; the economics of Part IV depend
entirely on it. This part documents why "just download the prices" consumed
more engineering and more failure-recovery than the model itself — and why
that was the correct allocation of effort. The unabridged incident record is
`docs/research_log/FII_Module5_validation_log.md` §§1–3e.*

## 8. The price panel: repairs, corporate-action mathematics, and gates

### 8.1 What the price data must support, and the central hazard

The pre-registered economic tests need, for every stock in the model
universe, a daily return series that is: **survivorship-free** (fire-sale
stocks delist — dropping them deletes exactly the HOSTAGE worst cases),
**corporate-action-true** (a 1:5 split prints −80% on the tape that no
investor experienced), and **identity-continuous** (§9). The only available
source is the NSE bhavcopy — raw daily closes, 2011–2025, including delisted
names (`data/VALIDATION_DATA/bhavcopy_parquets/`).

The central hazard, stated numerically: on known split/bonus ex-days, the
raw tape's median absolute "return" is **0.508** — the signature of factor-2
events (the modal case: a price halving that is bookkeeping, not economics).
A single unadjusted split inside a 20-day measurement window fabricates a
crash larger than any effect we are testing for by two orders of magnitude
(50,000 bp of artifact against ~50 bp of hypothesized signal). Most of
Module 5's effort was neutralizing this hazard *verifiably*.

### 8.2 Panel assembly and the audit-driven repairs

`src/fii/data_prep/module5a_price_panel.py` (stage `price_panel`) assembles
5.95M stock-days from 15 yearly parquets. Each repair was triggered by a
specific finding of the raw-data audit
(`src/fii/validation/audits/module4c_data_audit.py`):

- **R1 — the year-0020 bug.** 1,703 rows in `prices_2020` carried literal
  year 0020 (a two-digit-year parse upstream). Poisonous precisely because
  date joins *silently drop* such rows — no error, just missing data.
  Repair: offset +2000 years.
- **R2 — the 2011 null-ISIN hole.** All early-2011 rows (169,866) had null
  ISINs. Repair: backfill each symbol's ISIN from its next observed value;
  1,341 unfillable rows dropped. This repair carried known risks (symbol
  recycling, CA-timing) that were *not* waved away: it was validated only
  later and independently, when the canonical v3 panel showed 2011 coverage
  of 98.45% (§9.4) — had the backfill been wrong, that number would have
  cratered.
- **R3 — series dedupe.** One (isin, date) can appear under multiple NSE
  series; keep EQ > BE > BZ.
- **Macro timezone repair.** yfinance NIFTY/VIX dates landed one day early
  (20% on weekends — a tz artifact): +1 day, after which 3,491/3,492 dates
  matched the trading calendar. The S&P 500 enters lagged one day (US
  closes after India — a lookahead otherwise).

Two gates certified the assembly. **G1 (survivorship):** RCOM, HDIL, DHFL,
JETAIRWAYS and peers present with correct delisting-era end dates — the
panel is survivorship-free. **G2 (join coverage):** model stock-days match
price rows at 90.7%, and — the part that matters for inference —
**uniformly across archetypes (89.8–91.6%)**: no archetype-selective
attrition, so measurement error is not correlated with the groups being
compared. (Where the missing 9.3% went, and why a crosswalk was *not* the
answer, is §9.2.)

### 8.3 Failure #1: the assumption that died on a gate

The original design assumed NSE's `prev_close` column is restated on
ex-dates — i.e. that `close/prev_close − 1` is already CA-adjusted. This
assumption was **tested rather than trusted** (Gate 0, in the retired
`legacy/colab_modules/module5b_forward_returns.py`): on 672 known
split/bonus ex-days, median |ret| from prev_close = 0.505 ≈ median |ret_cc|
= 0.506 from close-to-close. Had prev_close been restated, the two medians
would diverge wildly on exactly those days; their identity proved
**prev_close is raw**, voided every downstream number computed to that
point, and forced the adjustment to be *built*, not assumed. The episode
also produced a process correction that shaped the whole repository: the
retired script had bundled the data gate and the model-facing analysis in
one step, and the user's correction — *one verifiable step per script* —
became the stepwise, gated architecture the pipeline still has.

### 8.4 The corporate-action mathematics

For a face-value split $f_{old} \to f_{new}$ and a bonus of $a{:}b$ ($a$ new
shares per $b$ held), the ex-date price divisors are

$$F_{split} = \frac{f_{old}}{f_{new}}, \qquad
F_{bonus} = \frac{a+b}{b}, \qquad
F_{same-day} = F_{split}\cdot F_{bonus},$$

(e.g. TITAN 2011: a 10→1 split plus 1:1 bonus ⇒ $F = 10 \times 2 = 20$),
and the adjusted return over an ex-date is

$$r^{adj}_t = (1 + r^{cc}_t)\,F - 1.$$

Worked example from the log: a raw −94.7% print on a ×20 day becomes
$0.053 \times 20 - 1 = +6.6\%$ — the stock actually *rose* that day.

**Parsing and tape-verification**
(`src/fii/data_prep/module5b1_ca_factors.py`, stage `ca_factors`). Factors
are parsed from the NSE corporate-action file's free-text PURPOSE field —
and every parsed factor is then **verified against the tape**: the observed
ex-day ratio $P_{t-1}/P_t$ must equal the factor within a ×/÷1.3 band (the
band absorbs the genuine same-day market move). This verification is what
caught **Failure #2, the combined-event parser bug**: on rows like *"Bonus
1:1 And Face Value Split Rs.10 To Re.1"*, the v1 split regex grabbed the
first two numbers in the string — the bonus ratio — silently dropping the
split component on TITAN, ONGC, NALCO and peers. The tell was diagnostic,
not lucky: the disagreement table clustered at obs/factor ≈ 5 and ≈ 10,
exactly the missing split magnitudes. The v2 parser anchors split numbers
to text *after* the split keyword and deduplicates to at most one split
plus one bonus per symbol-day. Result: 570/570 splits parsed, **99.1% of
1,103 factors tape-confirmed** (splits 99.5%, bonuses 98.8%), and each of
the 7 residual disagreements individually explained and *correct to fail* —
e.g. DRREDDY/NTPC "bonuses" that were **debenture** issues with no equity
price effect (applying them would fabricate −50%/−86% crashes), and
postponed ex-dates. Policy locked: apply confirmed or unverifiable factors;
null (never adjust) the ex-day return of confirmed-wrong events.

### 8.5 Failure #3: the silent clipboard corruption — and the process fix

The application step (`src/fii/data_prep/module5b2_apply_adjustment.py`,
stage `apply_adjustment`) initially produced internally impossible output:
median |ret_adj| = 2.04, zero ex-days small, and TITAN's ex-day `ret_cc`
printed +0.053 on the same row where step-1 had measured an 18.75× ratio.
Numbers that contradict each other within one printout mean the *executed
code* differs from the file. A read-only diagnostic
(`src/fii/validation/audits/module5b2d_diagnose.py` — deliberately
diagnostic-before-fix) found the smoking gun: the stored `ret_cc` equalled
a fresh recompute **plus exactly 1.0 on all 5.94M rows**. The hand-pasted
Colab cell had lost the "−1" from the return formula — clipboard corruption
that, unlike incident #1 (a loud SyntaxError), ran silently. The file's
formula was correct all along.

The process fix outlived the incident: **scripts became the only executable
artifact** — stored in Drive, run byte-exact via `exec(open(path))`, never
pasted — which is the direct ancestor of this repository's runner
(`src/fii/runner.py`) executing preserved stage files. The same diagnostic
run also vindicated the factor table independently (applying factors to the
*fresh* series: median |adj| = 0.037) and surfaced two structural findings:
splits **mint new ISINs** (breaking per-ISIN return chains at exactly the
biggest corporate events — the motivation for §9), and **symbol
migrations** (Failure #4: UNOMINDA's split occurred under its old symbol
MINDAIND, so the CA row pointed at a price chain not containing the event —
even correct code would have fabricated a +2860% return). The migration
class was closed with an **application guard**: any factor-day still
showing |ret_adj| > 50% *after* adjustment is nulled and logged. The guard
caught 45 days — on inspection, all renames (ZYDUSLIFE ex-Cadila,
TATACONSUM ex-Tata Global, CYIENT ex-Infotech…).

### 8.6 The certification gates

The clean, guarded, exec-run application passed both pre-registered gates
(`outputs/logs/*_apply_adjustment.log`):

- **Gate A:** median |ret_adj| across confirmed ex-days **0.508 → 0.037**
  (later 0.038 on the v3 panel's 803 ex-days), with 99.3% of ex-day returns
  under 20% — corporate-action days became ordinary trading days.
- **Gate B:** extreme-day counts at the expected few-hundred-per-year scale,
  adjusted ≤ raw everywhere, largest survivors identifiable as
  suspension-relist artifacts (handled downstream by the ±50% clip), no
  split debris.

`returns_panel_v2.parquet` was thereby certified — and then superseded by
one more identity-driven rebuild.

## 9. Identity: the ISIN closure problem

### 9.1 Why identity is a research problem, not a housekeeping detail

An ISIN is not a company. Splits and face-value changes mint new ISINs;
mergers and demergers kill them; partly-paid and DVR lines coexist with the
main line. Two failure modes threaten this project specifically:
**fragmentation** (one company's history split across ISINs — Alok
Industries' insolvency-and-relisting, a quintessential HOSTAGE name, breaks
into two unrelated stubs exactly at its distress event) and **false
merging** (mapping a merger target onto its acquirer manufactures a fake
continuous history). The v2 panel, keyed on raw ISINs, dropped 7.8% of
model stock-days (pre-split history stranded under retired ISINs) and broke
return chains at every face-value event.

### 9.2 First, the dog that didn't bark: the attrition diagnostic

Before building any closure, the 9.3% G2 join miss was decomposed
read-only (`src/fii/validation/audits/module5d_attrition_diagnostic.py`):
was it identity churn (fixable by a crosswalk) or genuine non-trading? The
answer: **99.5% of unmatched stock-days were Cause B** — the cisin *is* in
the tape but has no price that date (NSDL FII data is exchange-agnostic;
BSE-primary days, pre-listing, post-delisting and suspension days produce
an FII trade with no NSE close — correctly unmeasurable). A crosswalk would
have recovered 0.05% of rows and was **not built** at that point. Equally
important for inference: START- and END-anchor match rates were identical
per archetype (HOSTAGE 95.9/95.9 in TEST), so reversal anchors were not
differentially lost. The lesson worth generalizing: *diagnose attrition
before engineering against it — most "missing data" problems are facts
about the world, not defects in the pipeline.*

### 9.3 The closure rule (user-specified, audit-verified)

The eventual canonicalization — motivated not by the join miss but by the
stranded pre-split history — is **issuer-bounded and
corporate-action-type-conditional** (design record:
`docs/research_log/FII_Module5_validation_log.md` §3e; memory of the design
debate in the project log):

1. **The entity key is inside the ISIN.** Characters 4–7 of an Indian ISIN
   are the issuer code (ONGC "213A" is stable even across an IN8→INE
   prefix change). Legitimate identity closure lives *within* one issuer
   code; any lookup link crossing issuer codes is a merger/acquisition
   candidate and must mean identity **death**, not mapping.
2. **Event type decides.** Value-preserving events (split, bonus,
   face-value change, rename) map old → the issuer's terminal
   (latest-trading) ISIN. Value-changing events (merger, demerger,
   amalgamation) terminate the old identity.
3. **Co-existence guard.** If two ISINs of one issuer *traded
   simultaneously for ≥180 days*, they are distinct lines (DVR,
   partly-paid) and are never collapsed.

The rule was not trusted on paper; it was audited against the data before
being applied. `module5k_isin_accounting.py`: all 752 lookup links are
same-issuer — zero cross-issuer links, so the entity-boundary condition
already held; issuer-code chains decompose into 1,117 clean
single-active chains (auto-closable), 168 fully-dead, and only 8 with >1
active line (the DVR/dual-class set — correctly not collapsed).
`module5l_noisin_probe.py` closed the degenerate-key branch: fabricated
`NOISIN<digits>` placeholders and null-ISIN records — a worry because they
could either pollute the universe or fragment real companies — turned out
to be 0.03% of raw trades, overwhelmingly *mutual-fund* records (non-equity,
correctly out of scope), touching **zero** of the 946 model names.
`module5m_universe_integrity.py` answered "is 946 even the right count":
946 cisins = 939 distinct issuer codes; six of the seven doubled issuers
were genuine fragmentation (Tata Steel, Alok, Ruchi/Patanjali, Bajaj
Finance, Cholamandalam, Vaibhav Global), one a genuine co-existence
(Bharti Airtel's partly-paid line). All audits are re-runnable:
`python pipeline.py --phase audit`.

### 9.4 The v3 build and what it bought

`src/fii/data_prep/module5j_canonical_panel.py` (stage `canonical_panel`)
applies the closure to **both sides** — the price tape *and* the model's
state history (dual-side, so event windows and states agree about who is
who; day-labels themselves stay frozen). Outcomes
(`outputs/logs/*_canonical_panel.log`):

- **9 model-side merges** (7 fragmented pairs + 2 terminal fixes):
  946 cisins → **939 companies**.
- **Gate A re-passed on v3**: median |ret_adj| = 0.038 over **803**
  confirmed ex-days — 130 *more* verifiable ex-days than v2, because
  face-value splits that used to be null chain-breaks are now measurable
  across the canonical chain.
- **Coverage 90.4% → 98.49%**, uniform across archetypes (HOSTAGE 98.25%);
  TEST era 99.85%.
- **The 2011 backfill (R2) independently validated**: 98.45% coverage in
  2011 — the recycling/timing risks flagged at repair time did not
  materialize.

Outputs: `returns_panel_v3.parquet` + `states_v3.parquet` (804,591
stock-days, 939 companies) — the certified basis for every result in
Part IV.

**Value at this stage, round five.** Nothing in §§8–9 is glamorous, and all
of it is load-bearing: the headline economic effect (~50 bp/20d) is three
orders of magnitude smaller than one unadjusted split. The transferable
assets: the Gate-0 pattern (*test the vendor's adjustment convention before
using it* — NSE's prev_close is raw, contrary to common belief); the
tape-verification loop for CA factors (parse → predict the ex-day ratio →
confirm on the tape → null what fails); the application guard against
symbol migrations; and the issuer-bounded, event-conditional,
co-existence-guarded ISIN closure — to our knowledge not documented
elsewhere for Indian data — plus the audit suite that certifies it. Any
Indian-equities research group can lift Part III wholesale.

*— End of Part III. Part IV opens the economics: the event-study machinery
and its inference correction (§10), the volume-signed mechanism (§11), the
panel regression (§12), robustness (§13), the INNOV control (§14), the
LightGBM challenger (§15), PIN (§16), the honest nulls (§17), and the
backtests (§18).*

---
---

# Part IV — Economic validation

*Everything in Parts I–III was construction. This part is the confrontation
with prices — designed as a gauntlet the hypothesis had to survive, and
which, twice, it did not: the pre-registered story died and a different,
better-supported one replaced it. Stages: `event_study` through
`bt_style_switch`; every log in `outputs/validation/`.*

## 10. Event studies: mathematics and the inference correction

### 10.1 The measurement object

The unit is the **episode** — a maximal run of consecutive same-archetype
days in one stock. Two anchors, testing different economics: **START**
(first day; tests drift while the behaviour unfolds) and **END** (last day;
tests what happens *after the flow stops* — the correct anchor for any
price-pressure story, because pressure relaxation begins when the pressure
ends). For stock $i$ with market-adjusted, CA-adjusted daily abnormal
return $ar_{it} = r^{adj}_{it} - r^{NIFTY}_t$, clipped at ±50%:

$$CAR_i(k) = \sum_{t=\tau_i+1}^{\tau_i+k} ar_{it}, \qquad k \in \{1,5,10,20\},$$

starting at $t+1$, **never** at the anchor day itself: day-0's return is
mechanically correlated with the flow that defined the label (same-day
impact), so it is reported as context and excluded from evidence.

Three integrity provisions, each answering a specific bias
(definitions: `docs/research_log/FII_Module5_validation_log.md` §1):

- **Delisting truncation.** Events whose stock stops trading inside the
  window are truncated at the last price and **kept** — dropping them would
  delete exactly the fire-sale-to-bankruptcy outcomes the HOSTAGE test
  needs (survivorship bias where it hurts most). Truncation is slightly
  optimistic (post-delisting value ≈ 0), so trunc% is printed per archetype
  to keep the bias visible.
- **±50% daily clip.** Kills suspension-relist artifacts (₹0.15 → ₹8.55
  prints +5600%) while leaving genuine circuit-limit days untouched.
- **Date-clustered bootstrap** ($N=1000$). Events are not independent: one
  crisis day launches dozens of episodes that then share a market path. An
  i.i.d. t-test fakes precision by counting each as independent evidence.
  Instead, calendar anchor-days are resampled with replacement — each draw
  takes *all* events of that day — and the 95% interval and two-sided p
  come from the resampled distribution of mean CAR. The clustering unit
  (day) is the conservative choice.

### 10.2 The inference flaw we caught in our own first pass

The first model-facing run (`legacy/colab_modules/module5b3_car_start.py`;
post-mortem in log §3c) tested CARs **against zero** — and everything was
"significant," including the placebo. Diagnosis: the TEST-era ALL_LABELED
baseline — the mean forward CAR over *every* labeled stock-day — was
**+52 bp/20d versus NIFTY**, a classic benchmark mismatch (beta-1
adjustment of mid-caps against a large-cap index in a 2021–25 mid-cap
bull). Every archetype inherits that shift; only
$\text{archetype} - \text{baseline}$ is meaningful. The correction
(`src/fii/validation/module5b4_car_diff.py`, stage `event_study`): report
**excess CAR** with the date-clustered bootstrap run on the *difference*.
This is a difference-in-differences: date-level shocks common to all
archetypes cancel. The flawed vs-zero p-values were voided, and the
episode is retained as a caution: *the most dangerous inference errors
produce significance, not absurdity.*

### 10.3 The verdict that inverted the hypothesis (log §3f)

On the certified v3 panel, END-anchor excess CARs
(`outputs/validation/event_study.log`):

| Archetype | Prediction | TRAIN excess CAR20 | TEST excess CAR20 |
|---|---|---|---|
| HOSTAGE (dispersed sell) | **reversal** (the flagship) | −15 bp, p=0.23 | +10 bp, p=0.60 |
| SHARK_DIST (concentrated sell) | continued decline | **+68 bp, p=0.000** | **+33 bp, p=0.042** |
| SHARK_ACC (concentrated buy) | continued rise | **−50 bp, p=0.000** | −13 bp, ns |
| ROBOT (placebo) | ≈ 0 | ≈0, p≫0.05 | ≈0, p≫0.05 |

The placebo behaving proves the method; the hypothesis rows falsify the
hypothesis. The flagship HOSTAGE reversal is **absent** — null in both
eras. And the "informed" Shark rows are *wrong-signed, significant, and
replicating*: concentrated selling is followed by price **recovery**;
concentrated buying by **give-back**. The data-driven reframing: **the
concentration axis marks temporary price pressure** — the Coval–Stafford
reversal mechanism is real, but it lives on the *concentrated* episodes,
not the dispersed ones. This was inversion number one. It demanded a
mechanism check before being believed.

## 11. The mechanism: a liquidity shock signed by volume

### 11.1 The corroboration that failed, informatively (log §3g)

If concentrated FII selling is block-like liquidity demand, it should
coincide with exchange-disclosed block/bulk deals. Tested
(`src/fii/validation/module6_deal_corroboration.py`): SHARK_DIST days carry
*fewer* sell-side deal disclosures than HOSTAGE days (0.73% vs 0.96%,
ratio 0.76×, z = −4.2) — opposite to prediction — and within every
archetype, buy-deal and sell-deal rates are nearly equal, revealing that
disclosed-deal incidence is a two-sided *activity/attention* proxy, not a
directional-pressure measure. Three structural confounds explain the null:
disclosure thresholds (0.5% of shares, single orders) miss institutional
orders *worked* across a day; the disclosure universe is all participants
(promoters, domestic funds) while our concentration is FII-only; and
HOSTAGE days are simply eventful. At this point two economic
interpretations were dead (informed/forced from §10; block-visibility from
here), and a decision rule was imposed to prevent interpretation-shopping:
**one final mechanism test with a pre-committed stopping rule** — if it
failed, the reversal would be published as an unexplained regularity, full
stop.

### 11.2 The pre-committed test, and its confirmation (log §3h)

If concentrated episodes are liquidity shocks, they must carry the
*unmaskable* signature: price pressure during the episode **and elevated
trading volume at the climax**, followed by recovery. The bar, written
before the run: SHARK_DIST must show episode CAR < 0 *and* day-0 relative
volume > 1 *in both eras*, or mechanism-hunting stops. The full arc
(`src/fii/validation/module6b_liquidity_shock_profile.py`;
figure `outputs/figures/F1_mechanism_arc.png`,
table `outputs/tables/T3_mechanism_arc.csv`):

- **SHARK_DIST:** ≈ −270/−226 bp excess decline in the month *before* the
  episode → episode-leg pressure (epCAR −17/−5 bp, medians −18/−12) with
  **day-0 volume above baseline** → **+81/+32 bp excess recovery** after
  the END — roughly one-third of the preceding decline retraced.
- **SHARK_ACC mirrors with flipped signs:** +305/+359 bp run-up, +70/+48 bp
  episode pressure, elevated volume, −49/−15 bp give-back.
- **HOSTAGE — the decisive contrast:** *no* episode pressure, volume
  **below** baseline (0.95–0.98×), and no recovery, ever. Dispersed selling
  is *quiet* distribution the market absorbs without a liquidity premium —
  and its price marks are permanent.
- **ROBOT:** flat arc, ≈1.0× volume. The volume ordering SHARK_DIST >
  SHARK_ACC > baseline > HOSTAGE replicates across eras.

The pre-committed bar was met in both eras. The final economic reading —
the sentence the whole project compresses to — became:

> **The concentration of FII participation separates transitory from
> permanent price impact.** Concentrated flow is liquidity-demanding:
> volume-marked, price-pressuring, partially reverting. Dispersed flow is
> information-consistent: quiet, and permanent. The archetype names
> inverted; the mechanism is coherent, replicating, placebo-clean, and
> established from internal data alone.

It also retro-explains the block-deal null: disclosure data measures a
*visibility threshold*; volume — which cannot be masked — carries the
actual footprint.

## 12. Panel regression: fixed effects, clustering, and the bad control

### 12.1 Why a regression after event studies

Event-study differences control for *era-level* drift but not for the
possibility that archetype episodes happen to sit on particular *stocks*
(cheap, illiquid, high-beta) or particular *dates*, or that "reversal after
concentrated selling" is just the generic reversal of any recent loser. The
formal machinery (`src/fii/validation/module7_panel_regression.py`, stage
`panel_regression`; estimator: `linearmodels.PanelOLS` — a deliberate
choice after a user correction that statistical models must be *visible
library calls*, auditable against documentation, not hand-rolled algebra):

$$y_{ie} = \alpha_i + \delta_{t(e)} + \sum_{a} \beta_a D^a_{ie}
+ \boldsymbol{\gamma}' \mathbf{z}_{ie} + \varepsilon_{ie},$$

where $e$ indexes episode-END events, $y$ is the forward 20-day abnormal
return (bp), $D^a$ are archetype dummies (omitted category: UNTAGGED — the
"directional but unremarkable" days, the natural control), $\alpha_i$ are
**stock fixed effects** (absorbing every time-invariant stock attribute:
sector, average liquidity, governance, listing venue), and $\delta_t$ are
**date fixed effects** (absorbing every market-wide date shock: index
moves, macro news, VIX level — which is *why* VIX appears nowhere until
§17: it is implicitly controlled here). Estimation is the within
transformation — demeaning $y$ and regressors by entity and time — so
$\beta_a$ is identified purely from *within-stock, within-date* variation:
"on the same date, did the stock whose concentrated-sell episode just ended
outperform another stock of its own average behaviour?"

### 12.2 Inference: two-way clustered standard errors

Residuals correlate both within a stock over time (persistent stock-level
shocks) and across stocks within a period (common shocks the date FE don't
fully absorb, plus the mechanical overlap of 20-day windows). Standard
errors are therefore **clustered two ways, by stock and by calendar
month**, via the cluster-robust sandwich estimator
$\widehat{V} = (X'X)^{-1}\big(\sum_g X_g' \hat{u}_g \hat{u}_g' X_g\big)
(X'X)^{-1}$ summed over clusters $g$, combined across the two dimensions by
inclusion–exclusion (Cameron–Gelbach–Miller). Clustering by *month* rather
than day for the time dimension is the conservative choice given 20-day
overlapping outcomes.

### 12.3 The specification ladder and the bad control, reasoned

- **R0:** fixed effects only.
- **R1:** + characteristics measured strictly before the anchor: rolling
  120-day beta, momentum (t−126…−21), Amihud illiquidity, log turnover,
  20-day volatility, relative volume, log price, log episode length.
- **R2:** R1 + **the pre-episode 20-day return (pre20)** — included knowing
  it is a *bad control* in the causal sense: the episode itself produces
  price pressure (§11), so pre20 is partly a **mediator**, and controlling
  for it absorbs part of the true effect. That is the point: R2 is a
  deliberately conservative lower bound that mechanically soaks up any
  "losers bounce" channel. If the archetype dummy survives R2, the generic
  loser-reversal objection is dead by construction. R1-vs-R2 brackets the
  effect.

One econometric note we state rather than bury: archetype labels are
**estimated regressors** (outputs of a fitted model). Classification error
in a dummy attenuates its coefficient toward zero — so reported effects are,
if anything, understated.

### 12.4 Results and the two flags (logs §3i–3j; `outputs/tables/T1_panel_regression_main.csv`, full coefficients `outputs/regression_outputs/`)

The headline (R2, restored full sample — see flag 2):

| END-event dummy | TRAIN | TEST (OOS) |
|---|---|---|
| SHARK_DIST | **+65.4***, t=5.35 | **+48.6***, t≈2.9 |
| SHARK_ACC | **−87.9***, t=−6.23 | **−47.6***, t=−2.84 |
| HOSTAGE | −3.6, ns | −8.1, ns |
| ROBOT | −7.7, ns | −38.4*** *(see §13)* |

with n = 63,259 / 35,572 episode-ends. (Bounce-free companions, windows
t+3..t+22: SHARK_DIST +58.0*** / +36.7** — the first-two-day bid-ask
component is 11%/24% of these coefficients; measured in §13.5b and quoted
alongside the headline wherever it matters.) Controls behave sensibly (log
price strongly negative — the microcap-bounce effect; Amihud positive in
TEST — an illiquidity premium), and pre20 enters significantly in TRAIN
while the SHARK_DIST dummy *barely moves* from R1 to R2 — generic
loser-reversal exists in this market and the archetype effect is not it. Two honest flags
were raised in the first pass and resolved in §13: the TEST-era ROBOT
placebo failure, and a TRAIN sample collapse (66k→27k) in R1/R2 caused by
sparse index-return nulls poisoning 120-day rolling betas (one null kills a
window; zero-filling 0.6% of index gaps restored n with coefficients
essentially unchanged — proof the collapse was never selection).

## 13. Robustness: overlap, horizons, dose–response, and the placebo that wasn't

All in `src/fii/validation/module7b_robustness.py` (stage `robustness`,
log `outputs/validation/robustness.log`).

**(A) The ROBOT "placebo failure" is structural, and the honest fix is a
better control.** Decomposing ROBOT-END events by *what follows them*: a
ROBOT (neutral) episode ends **by transitioning into a directional
regime**, so its post-window mechanically contains the successor regime's
flow. ROBOT post20 splits +94 bp after BUY-transitions vs −13 bp after
SELL-transitions; the TEST-era negative aggregate is transition mechanics
plus delisting truncation (327 truncated events at −129 bp), not a leak in
the machinery. Conclusion: ROBOT-END is structurally invalid as a placebo
*at the END anchor*. The valid null is **HOSTAGE** — a selling episode
differing from SHARK_DIST only in concentration — which is clean
everywhere. We report this rather than hide it: a reader should see that
one pre-registered control failed *for a reason the data explains*, and
that the mechanism-relevant control is stronger than the failed one.

**(B) Non-overlapping episodes.** Within stock, greedily keep episodes ≥28
calendar days apart (39.6k of 102.9k kept): SHARK_DIST **strengthens**
(+76***/+108***). The overlap objection — that clustered episodes
double-count one price path — is dead; if anything, overlap *diluted* the
effect.

**(C) Horizons.** R2 dummies at 10/20/30/60 days
(`outputs/tables/T4_horizons.csv`, figure `F2_horizons.png`): SHARK_DIST
builds +39 → +65 → +89 → +109 bp (TRAIN) and +24 → +49 → +51 → +73 (TEST);
HOSTAGE is ≈0 at every horizon. A transitory-impact signature must build
and persist — a momentum artifact would decay, a microstructure bounce
would flip. This panel is the single strongest exhibit in the project.

**(D) Continuous dose–response.** Replace dummies entirely: within
sell-regime episodes, regress post20 on the episode-mean concentration
$\overline{F}^{entity\_s}_e$ (no thresholds, no labels — sidestepping
estimated-regressor attenuation). All four era×side cells are
right-signed; TRAIN-SELL is cleanly significant (+27.2 bp per unit,
p<0.01); TEST-SELL is +15 bp, p=0.16. Read honestly: the *linear* slope is
diluted where the *tail* dummy is strong — the reversal is a
tail-of-concentration phenomenon, consistent with block-like liquidity
demand being an extreme event, and TEST is underpowered for the continuum
version. We claim the tail effect, not a universal linear law.

### 13.5 The referee tests (Module 13)

An external review conceded the methodology and challenged the economics
on three testable grounds. Each became a pre-registered experiment
(scripts `module13a/b/c`, stages `backbone_ablation`,
`incremental_value`, `recon_exclusion`; log §3p):

**(i) "Is the HMM central or decoration?"** Answered by the ablation in
§5.6: a census-matched persistence rule reproduces Table 1 within band
(κ = 0.891 label agreement). Verdict V1 — the HMM is not necessary; the
contribution is the measure. Adopted as the paper's framing.

**(ii) "Show the measure adds information beyond existing predictors."**
Two instruments. At the *episode* level, adding the obvious competitors —
episode-mean flow imbalance, log flow size, scaled net flow — to the R2
spec moves D_SHARK_DIST by only −13%/−11% (+65.4→+57.1*** TRAIN,
+48.5→+43.4*** TEST): **composition is not flow magnitude in disguise.**
At the *daily* level, a LightGBM fitted with versus without the
composition block ({F_entity, F_entity_buy, F_breadth}; 788,935
stock-days) gains ΔIC = +0.0060 with paired daily t = +3.09 — a **28%
relative improvement** (0.0211 → 0.0272) that cleared both pre-registered
numeric bars — while the *quintile spread* did not improve, and the
composition-only model showed the inverse profile: the lowest IC
(+0.0043) but the largest and only significant spread (+73.2 bp,
non-overlap t = 2.84). Read together: composition ranks the broad
cross-section weakly but its extremes are potent — a third independent
confirmation (with the dose-response and the dummy-vs-continuous
contrast) that the signal's information is **tail-concentrated, exactly
where the paper claims it**.

**(iii) "Could it be ETF/index rebalancing?"** Excluding every episode
ending within ±7 calendar days of any of 150 MSCI/FTSE/NIFTY review
dates — discarding 41% of episodes — leaves Table 1 intact (SHARK_DIST
+61.5*** TRAIN / +50.9** TEST; at ±3 days, +72.2***/+59.1***, if anything
stronger). **Passive index mechanics are excluded** as the driver.

**The boundary we concede rather than test.** The reviewer's sharpest
sentence — *"your variable measures how trading is distributed, but the
interpretation extends to why trading occurred"* — is correct as a
boundary statement, and this document's language is drawn at exactly that
line. What is measured: transitory versus permanent price impact, sorted
by participation concentration. What is supported but not proven: the
information interpretation of the permanent side ("information-
*consistent*", backed by the count-based PIN loading, §16). What cannot
be distinguished with this data: informed dispersed selling versus
non-informational permanent-impact channels such as ESG or mandate-driven
exits under downward-sloping demand — both produce quiet, permanent
declines, and stock-level recon membership is likewise untestable with a
market-wide date file. These live in §19's ledger and the limitations,
not in the claims.

### 13.5b The skeptic tests (Module 15): four holes we found in our own work, two claims repriced

After every external round closed, a deliberately hostile re-read of the
research (not the paper) surfaced four first-order objections **no
reviewer had raised**: (i) bid-ask bounce — episode ends after
concentrated selling close at the bid, so mid-reversion mechanically
fakes a "+reversal"; (ii) the nearest prior art is not the block-trade
literature but volume-conditioned reversal (Conrad–Hameed–Niden 1994),
never confronted head-to-head; (iii) the paper's headline contrast
("SD reverts, HOSTAGE doesn't") was supported by "significant vs not
significant," which is not itself a test; (iv) the SUB_ACC robustness
promised in Module 1 was never executed. The first three were tested
(`module15_skeptic_tests.py`, stage `skeptic_tests`, pre-registered;
log §3r); the fourth is stated as an open item.

**Results, with the honest repricing.** (i) Bounce-free windows
(t+3..t+22): SHARK_DIST +58.0*** / **+36.7*** vs the standard
+65.4/+48.5 — a bounce component of **11% (train) and 24% (test)**.
Bounce is not the driver (the effect survives and continues building),
but headline magnitudes are henceforth quoted with their bounce-free
companions, and the execution-savings translation scales down
accordingly. Interpretive note: the spread concession *is* the
liquidity-demand mechanism in miniature, so this component is part of
the story, not an artifact of it — it is simply not novel. (ii) A
public volume-conditioned-reversal dummy (bottom-quintile loser with
relvol > 1.1 — computable from free data) enters weak and insignificant
(+26/+21 bp) while D_SHARK_DIST is unchanged to the first decimal
beside it; and the GBT ladder gives the definitive incremental-value
hierarchy: PUBLIC 0.0160 → +composition 0.0226 (paired t = 5.20);
PUBLIC+conventional-flow 0.0289 → +composition 0.0330 (t = 3.23).
**Composition adds significant information beyond free public data and
beyond conventional private flow.** (iii) The direct contrast
β_SD − β_HO is significant in the standard window in both eras (+69.0,
t = 4.49; +56.6, p = 0.016) — but in the bounce-free test-era window it
is +30.1 with t = 1.31: **directionally right, underpowered.** The
sharpest version of the headline sentence is therefore stated with that
qualification wherever it appears, and the permanence claim carries its
power bound (a HOSTAGE reversal smaller than ~40 bp would be
undetectable here).

### 13.6 The tail probe (Module 14): the reversal is a liquid-market phenomenon

A second external critique argued the opposite of caution: that the *real*
mispricing lives in the illiquid 75% we excluded, where dislocations are
hundreds of basis points. This was testable — and the 13A rule backbone
made it feasible (no HMM, no sequence floors). Three pre-registered steps
(`module14a/b/c`; log §3q): a feasibility census (the *measurable* tail is
the attributable tail — 137,774 stock-days across 1,618 names; gates
passed), a method-frozen label build (backbone cuts re-derived on
tail-TRAIN landed at −0.545/+0.556, strikingly near the liquid universe's;
episode clustering 2.25×/2.86×, p = 0.005 — the temporal structure exists
out there at full strength; one implementation failure, #12 in the ledger,
caught by its own impossible printout), and the economics.

**The verdict, pre-registered: NO TAIL EFFECT.** Tail concentrated-selling
episodes show *bigger* dislocations (pre-episode −455/−225 bp; episode leg
−196 bp) and a *stronger* volume climax (day-0 volume 3.42×/1.58×
trailing) — the mechanism's fingerprint transfers emphatically. The
recovery does not: post-END excess CAR20 is +3 bp (TRAIN, p = 0.93) and
+36 bp (TEST, CI [−51,+128], p = 0.37), with even the optimistic point
estimate at one-third of the tail's own friction bar (Roll round-trip
spread ~98–115 bp, against 156 bp median *daily* moves).

The economic reading upgrades the paper's story: **the post-episode
reversal is itself liquidity provision** — it exists only where arbitrage
capital operates. In the liquid 946, someone steps in within a month and
corrects the overshoot; in the tail, the dent stays. Transitory price
impact is a liquid-market phenomenon, and limits-to-arbitrage binds at
*both* ends of the spectrum: in liquid names the signal is smaller than
the costs; in illiquid names the reversal does not exist and the costs
triple. The long-standing scope caveat ("fire-sale effects may be
*stronger* in the excluded tail; our estimates are a floor") is hereby
**revised by measurement**: for the reversal side it was wrong in the
hedged direction. One suggestive cell is reported without being claimed:
tail HOSTAGE episodes in the test era show *continuation* (−95 bp,
CI [−167,−20]) — dispersed selling in illiquid names keeps falling —
directionally consistent with the information story, but a single
significant cell in a coverage-confounded era.

## 14. Alternative explanations and the flow-surprise control (INNOV)

The remaining alternative after §§12–13: maybe "concentration" is just
**flow-surprise magnitude** in disguise — big unexpected flow both looks
concentrated and mean-reverts. Closing this required a flow-expectation
model (`src/fii/validation/module9_net_innov.py`, stage `flow_innovation`).

**Construction.** Per stock, an AR(5) on scaled net flow
$n_{it} = NET_{it}/\overline{GROSS}_i$:
$$n_{it} = c_i + \sum_{k=1}^{5}\phi_{ik} n_{i,t-k} + u_{it}, \qquad
INNOV_{it} \equiv \hat{u}_{it},$$
fitted per stock by least squares over the **full sample — with deliberate
look-ahead**, and the rationale is on record because it was demanded
(log §3l): (1) INNOV is a *yardstick*, not a claim — full-sample
coefficients give the stablest expectation and the cleanest decomposition;
an expanding-window version injects estimation noise into a control;
(2) the bias direction is **conservative for us** — INNOV is the competing
explanation, and hindsight makes it the best-case competitor, so the
archetype surviving against an information-advantaged control is the
*stronger* result; (3) the asymmetry is the design: the archetypes (the
claim) are strictly real-time; INNOV (the instrument) is allowed hindsight.
Sanity: INNOV correlates with same-day returns at IC +0.33/+0.28 (99–100%
of days positive) — it measures what it should.

**The key test (T3).** Add episode-mean INNOV to the R2 regression:
**D_SHARK_DIST is unchanged to the decimal in TRAIN (+65.4***) and remains
+43.6*** in TEST**, while mean-INNOV itself enters at −25/−32*** —
surprise-magnitude reversion is real *and separate*. Concentration is not
flow-surprise in disguise. A non-parametric backstop (T4): SHARK_DIST
out-reverts HOSTAGE within every TRAIN INNOV-tercile and 2 of 3 TEST
terciles (one exact tie, reported).

**The bonus regularity (T2), with its caveat welded on.** INNOV predicts
*negative* forward returns in both eras (daily IC −0.023/−0.017,
t = −3.2/−2.6): unexpected FII flow reverts. Because the expectation model
is in-sample, this is a **decomposition result** — FII flow contains a
transitory component whose price effect reverts — and *not* an
implementable signal. It is the project's second regularity, always stated
with that caveat.

## 15. The machine-learning challenger: why LightGBM won the battle and lost the war

### 15.1 The question the challenger answers

Is the HMM the *bottleneck*? The regime pipeline uses four features and a
particular model class; perhaps a flexible learner on all ten features
finds forward-return structure the regimes miss — in which case the
right next step is better machine learning, not more economics. The
design (`src/fii/validation/module8_gbt_shap.py`, stage `gbt_challenger`)
pits LightGBM against a **regime baseline** (each archetype's TRAIN-mean
outcome as the prediction — the information content of the labels alone),
with **pre-registered success bars**: test-era IC gap > 0.01 over the
baseline *and* top-minus-bottom quintile spread with non-overlap t > 2.

### 15.2 The mathematics being deployed

**Gradient boosting** fits an additive model $\hat{y}^{(M)}(\mathbf{x}) =
\sum_{m=1}^{M} f_m(\mathbf{x})$, each $f_m$ a regression tree chosen to
minimize the second-order Taylor approximation of the loss around the
current prediction: with $g_i = \partial_{\hat{y}} \ell(y_i,\hat{y}_i)$ and
$h_i = \partial^2_{\hat{y}} \ell(y_i,\hat{y}_i)$, the optimal leaf weight
is $w_L^* = -\sum_{i\in L} g_i \big/ (\sum_{i \in L} h_i + \lambda)$ and
the split gain is the induced reduction in
$-\tfrac{1}{2}\sum_L (\sum_{i\in L} g_i)^2/(\sum_{i\in L} h_i + \lambda)$.
LightGBM grows trees **leaf-wise** (always splitting the highest-gain
leaf) on **histogram-binned** features — fast, and natively tolerant of
missing values, which matters because the feature store's nulls are
honest (§3.5) and must not be imputed. Configuration: 400 trees, learning
rate 0.05, 63 leaves; target: forward 20-day abnormal CAR; same universe,
same frozen split.

**SHAP attribution.** Shapley values from cooperative game theory: feature
$j$'s contribution is its marginal effect averaged over all orderings of
feature inclusion,
$\phi_j = \sum_{S \subseteq F\setminus\{j\}}
\frac{|S|!(|F|-|S|-1)!}{|F|!}[v(S\cup\{j\}) - v(S)]$ — the unique
attribution satisfying local accuracy, consistency and symmetry. For trees
this is computed *exactly* (TreeSHAP / `pred_contrib=True`), so the
attributions are not approximations.

### 15.3 What happened — including a bug we own

The challenger *beat the baseline*: TEST IC +0.0279 vs +0.0117 (gap
+0.016 > 0.01 bar), monotone quintiles, Q5−Q1 = +74.5 bp/20d with
non-overlap t = 3.22 (> 2 bar); TRAIN IC 0.218 — heavy memorization,
expected of 400 trees, which is precisely why only TEST counts. SHAP said
the edge was *not* mainly the concentration tail: top features were
`F_entity_buy` (ride the pressure while flow persists — an *arc-timing*
signal a day-label cannot express), the contrarian `F_imbal`/`F_streak`
(features the HMM never used), and `F_breadth`. One artifact was caught
and is owned: the module's single-feature IC table lacked a per-day dropna,
producing a spurious −0.135 "IC" for F_breadth (recomputed correctly:
+0.012). The GBT itself was unaffected (LightGBM handles NaN natively);
the correction is recorded in log §3k.

### 15.4 The decomposition that settles it (`module8b_demeaning_check.py`)

Before concluding "sequence models next," a gate: how much of the GBT edge
is *dynamic* (which-day information) versus *cross-sectional* (which-stock
information — characteristics in disguise)? Method: within-stock demeaning
using **TRAIN-era stock means** (test-safe), then (a) recompute feature
ICs — essentially unchanged, so no feature's signal is purely a stock
identity — and (b) refit the GBT on demeaned features, isolating dynamics.
Result: dynamics-only IC +0.0161, spread +53.4 bp, monotone — but
**non-overlap t = 1.48 < 2: the pre-registered bar for the dynamic
increment is not met.** Also measured: `F_breadth` is the most
characteristic-like feature (55% between-stock variance, 0.77 TRAIN→TEST
rank stability), explaining part of the full-GBT edge as stock selection.

**So why did LightGBM not deliver a big advantage?** Because its edge
decomposes into (i) the regime information itself, (ii) static stock
characteristics that fixed effects already handle in the economics, and
(iii) a *sub-significance* timing component. Held to the pre-commitment:
**the HMM is not the bottleneck; the features are the ceiling at this
data's statistical power; LSTMs/sequence models are not justified.** The
one identified lever for future work — the top dynamic SHAP feature —
is within-stock *breadth change* (FII participation surges), which the
current regime rules never use.

## 16. External validation: the PIN model

### 16.1 Why PIN, and why it is genuinely independent

Every test so far shares one input: returns. The Easley–O'Hara PIN
(probability of informed trading) is estimated from **order counts only** —
it has never seen a price, a return, or an archetype definition — so its
agreement is external corroboration in the strict sense. Our variant uses
FII buy/sell trade counts per stock-day ("FII-PIN"; stated limitation:
classical PIN uses all market orders).

### 16.2 The model and its estimation (`src/fii/validation/module11_pin.py`)

Each day is no-news (prob $1-\alpha$), bad-news ($\alpha\delta$) or
good-news ($\alpha(1-\delta)$); buys and sells arrive Poisson with
uninformed rates $\varepsilon_b, \varepsilon_s$ and an informed rate $\mu$
added to the news side. The day's likelihood is the three-component
mixture; the stock-year log-likelihood is maximized by L-BFGS-B with
Lin–Ke stabilization (log-sum-exp over components; the classical EHO
likelihood overflows on high-count days) and logit/log parameter
transforms. Then
$$PIN = \frac{\alpha\mu}{\alpha\mu + \varepsilon_b + \varepsilon_s},$$
the share of order flow attributable to informed trading. Estimated per
stock-year (≥60 days), 4,456 stock-years, 826 stocks; output
`data/VALIDATION_DATA/fii_pin_stockyear.parquet`.

**Sanity before use:** PIN levels median 0.19, IQR 0.14–0.25 (the
literature band), and correlation with log turnover −0.117 — informed
trading probability higher in smaller names, the canonical sign.

### 16.3 The result (T2; `outputs/tables/T5_pin_loadings.csv`, figure `F4`)

Regressing stock-year PIN on the year's archetype shares (turnover
controlled, stock-clustered SE):

| | TRAIN | TEST |
|---|---|---|
| HOSTAGE share | **+0.231***, t=7.9 | **+0.149***, t=4.5 |
| SHARK_DIST share | +0.077***, t=2.9 | +0.065**, t=2.5 |

Dispersed-selling exposure carries **~3× the informed-trading loading** of
concentrated-selling exposure, in both eras — an independent construct
arriving at the same transitory/permanent assignment as the price tests.
Two disciplined nuances: the pre-registration said SHARK_DIST ≈ 0 and it
is small-but-significant, so the claim is **relative** (concentrated
selling carries *less* information, not none); and T3 confirms the
liquidity reading from the other side — the SHARK_DIST reversal is *not*
concentrated in high-PIN names (it is larger where PIN is low), exactly as
a liquidity effect should be, while HOSTAGE is ≈0 in both PIN halves
(permanence is not hiding a conditional bounce).

## 17. State dependence: the honest nulls (VIX, Kyle's λ)

Late in the project the user noticed India VIX had "never been used." The
first half of the answer is econometric: **VIX levels were controlled all
along** — date fixed effects absorb every market-wide variable, VIX
included. What FE cannot absorb is an *interaction*: is the reversal
bigger in high-VIX states? Tested (`src/fii/validation/module10_vix_lambda.py`):
D_SHARK_DIST × high-VIX is +83*** in TEST — the entire TEST reversal
living in high-VIX halves, a textbook liquidity signature — but −27 (ns)
in TRAIN. By this project's replication standard, **non-replicating ⇒
suggestive, not claimable**, and that is how it is reported.

The second candidate: a FII-flow **Kyle λ** per stock — trailing 120
flow-day $\mathrm{cov}(ar_t, \tilde{f}_t)/\mathrm{var}(\tilde{f}_t)$ with
scaled net flow $\tilde f$, past-only (median +0.0043, sane) — does the
reversal amplify in high-impact names? The interaction is **null in both
eras**. Raw terciles *do* show 2.5× amplification in high-λ names, but the
main effect exposes it: high-λ names bounce more after *all* episode types
— an illiquidity characteristic, not an archetype mechanism — and the FE
machinery is what prevented that false claim from being made. Both nulls
are reported at full length precisely because a validation battery that
only ever confirms is not a validation battery.

## 18. Economic significance: backtests and limits to arbitrage

### 18.1 Why backtest a research finding at all

Statistical significance at 50 bp/20d says nothing about economic
harvestability. The user's question — build strategies with and without
the model and compare — is the correct final test, and it was run under
the same discipline as everything else: a gated engine, frozen baselines,
pre-registered verdicts (`src/fii/backtest/`; full detail
`docs/05_backtesting.md`; log §3o).

**The engine** (`engine.py`) is a daily cross-sectional long/short
machine: alpha → within-day demean → unit gross → 5% position cap;
signals formed at close $t$ trade at close $t{+}1$ and earn day $t{+}2$
(one-day lag); costs 15 bps one-way on turnover. Its own correctness is
gated (stage `engine_gates`): vectorized-equals-naive-loop exactness; a
**cheat-alpha alignment test** (tomorrow's return as today's signal:
Sharpe +211 at lag 0, collapsing to −0.5 at lag 1 — proving both the
alignment and that the lag genuinely delays); and exact cost/turnover
accounting on a constructed flip book. Episode ENDs are detected
real-time-legally: an end is knowable only at the close of the first day
*after* the run.

### 18.2 Strategy pairs and verdicts (tables `T6_backtest_metrics.csv`, figure `F3`)

Four pairs, each a no-model baseline (frozen first) against an
HMM-conditioned twin; verdict rule: TEST-era net-of-cost paired ΔSharpe
with a 20-day-block bootstrap 95% CI. Headline numbers, TEST era:

| Book | Gross Sharpe | Breakeven cost (bps, one-way) | Net Sharpe @15 bps |
|---|---|---|---|
| S3 mechanical concentration proxies | **+1.44** | 7.2 | −1.54 |
| S2H flow-following ex-concentrated-days | +1.46 | 4.3 | −3.62 |
| S4H style-switch on regimes | +0.54 | 1.7 | −4.08 |
| (every book) | — | **2–8** | **negative** |

Three findings, in decreasing order of confidence. **First**, the model's
information is real at the portfolio level and appears exactly where the
research says it should: removing concentrated-regime days improves a
flow-following book's *gross* Sharpe in **both eras** (ΔSharpe CI
[+0.03,+0.22] train, [+0.07,+0.49] test — the pre-registered "signal real,
costs bind" verdict), and the regime-arc style-switch book (trend within
concentrated episodes, reversion after their ends, flat elsewhere) is the
only strategy gross-positive in both eras while its mechanical twin is
gross-negative. **Second**, honest verdict accounting: of the formal
net-of-cost verdicts, one strategy was inconclusive, two read "HMM
subtracts" (one of them, S3H, had a construction flaw — unequal leg
magnitudes made it effectively a HOSTAGE-short churn book — owned in the
log), and S4H's formal "HMM adds value" is driven mostly by its twin's
cost bleed; the defensible claim is the gross one. **Third**, the wall:
breakeven costs of 2–8 bps one-way against realistic 15 bps mean **nothing
here is a trading strategy at daily rebalance** — the reversal is
~50 bp/20d on ~7% of stock-days.

### 18.3 The limits-to-arbitrage reading

This is not a disappointment to be minimized; it is the *explanation of
the regularity's persistence*. An anomaly that survives fourteen years in
a heavily-arbitraged market must have a reason arbitrageurs leave it
alone: harvesting it costs more than it pays. That also identifies who
*can* use the signal — anyone already obligated to trade (execution desks
scheduling around concentrated-flow episodes; PMs deciding whether a
decline is §11's quiet-permanent kind or the volume-marked-reverting
kind), for whom the transaction costs are already sunk. Short legs face
India's securities-lending constraints on top — stated, as everything
else, rather than assumed away.

*— End of Part IV. Part V closes the ledger: every assumption in one
table, the value audit by audience, the complete failure log, and the
conclusion.*

---
---

# Part VI — Phase II: from validated states to prediction

## 23. Causal estimation, calibrated forecasting, and the decision layer

Phase I established the states as measurement; Phase II (charter:
`docs/PHASE2_PLAN.md`, pre-registered; log `FII_Phase2_log.md`) asked the
engineering question: *using only information through t, estimate the
current regime, forecast its evolution, and convert forecasts into
decisions — under walk-forward discipline.* Data horizon: the sample ends
2025-03-28; all folds and claims respect it. Four stages, four verdicts,
two of them failures worth their weight.

**16A — the causal foundation (PASS).** Phase-I labels were smoothed
(full-sequence Viterbi: the state at t sees the future). The filtered
posterior P(S_t | x_{1:t}) from the frozen parameters recognizes new
regimes with a median lag of one day (p90: 3), agrees with the smoothed
labels 91.3%, and — the gate — Table 1 on fully causal labels is intact
(SHARK_DIST +60.2***/+50.3***; test marginally *stronger*). With 13A,
this closes the look-ahead question from two independent directions.

**16B — calibration (original gate FAIL, on the record; amended gate
PASS).** Our first persistence baseline conditioned on yesterday's
*smoothed* label — unknowable at time t and quasi-circular with the
smoothed target; the filtered nowcast lost to it on Brier (won on
log-loss) and the pre-registered gate failed as written. The amendment
(documented before rerunning) replaced it with the information-set-fair
baseline — persistence of yesterday's *filtered* state — which the
nowcast beats decisively (Brier 0.121 vs 0.262; log-loss 0.182 vs 0.532).
Calibration is excellent where 79% of the probability mass lives
(conf 0.99 → hit 0.994) and overconfident mid-range (0.70 → 0.55) — a
measured input caveat for anything consuming these posteriors.

**16C — the hazard model (PASS, decisively).** The economically relevant
forecast is the episode END (the reversal starts when concentrated flow
stops). A walk-forward gradient-boosted discrete hazard (yearly refits
2014–2025Q1; 14,273 causal episodes; right-censoring handled; runs broken
on >21-day calendar gaps) forecasts "today is the last day" at **AUC
0.797 against the age-only baseline's 0.569** — out-of-sample era
*stronger* than in-sample (0.809 vs 0.786) — with log-loss improvements
at paired t = +42 (k=1) and +34 (k=3). Episode ends are genuinely
forecastable beyond duration alone.

**16D — the decision layer (FAIL on its pre-registered bar — the
instructive result).** Entering on the model's end-forecast instead of
waiting for confirmation adds **+23 bp/episode out-of-sample
(CI [+7,+40]), cost-neutral** (same single entry, earlier). But an
age-only anticipator — "enter once the episode is a few days old,"
three lines, no model — earns +18 bp, and paired on common episodes the
model is significantly *worse* (−15 bp, t = −2.96). Diagnostics: the
threshold selection was fold-unstable, and roughly half the anticipation
gain is the first post-end day, which §13.5b already identified as
partly spread concession.

**Phase II conclusion.** The framework works end-to-end — causal
estimation, calibrated probabilities, real forecast skill — and its
decision value is capped by exactly the ceilings Phase I measured (8B's
sub-significant dynamic increment; Module 12's cost wall). The honest
sentence: *episode ends are forecastable but not monetizable beyond a
trivial age rule; the practical ~20 bp/episode of anticipation value
requires no machine learning at all.* For the execution-overlay story,
that simplifies the advice: do not wait for confirmed episode ends — act
once a concentrated episode is a few days old. The hazard model earns
its keep as monitoring (calibrated end-probabilities), not as timing
alpha. Both failed gates in this phase are reported as failed; both
taught more than the passes.

---
---

# Part V — The ledger

## 19. Assumptions, in one place

Every result in Parts II–IV rests on assumptions. Scattering them across
sections is how papers hide fragility; here they are collected, each with
its status: **tested** (a check exists in the pipeline), **bounded** (the
direction of the induced bias is known and stated), or **maintained**
(assumed, with the argument for why).

| # | Assumption | Where it bites | Status |
|---|---|---|---|
| A1 | Masked entity IDs are internally consistent *within* a month | Axis 3 / all concentration measures | **Tested** — per-month counts sane (~150 brokers, ~3,000 FIIs); cross-month identity separately *disproved* (§2.2) and never used |
| A2 | Value-weighted ID coverage ≥50% ⇒ concentration measurable | late-sample (2024–25) HOSTAGE rows | **Tested + bounded** — coverage holds value-weighted; late-era HOSTAGE *prevalence* still flagged uninterpretable (log §4, Module 3) |
| A3 | Within-day rank-probit preserves the economically relevant signal | every feature | **Maintained + bounded** — buys stationarity; forfeits market-wide level information by construction (census shares pinned) |
| A4 | First-order Markov states, diagonal-Gaussian emissions | HMM backbone | **Tested indirectly** — OOS replication of signatures/census/transitions (§7.2); diagonal justified by near-orthogonal features (§4.2) |
| A5 | 20-day windows / 5-day HHI smoothing are the right time scales | features, overlays | **Maintained** — dwell times (13–17d) and episode medians (2–3d) are outputs, not inputs, and came out economically sensible; window sweep not exhaustively run |
| A6 | Train ≤2021-04 generalizes to test ≥2021-07 | everything frozen | **Tested** — the entire Part IV replication structure |
| A7 | NSE bhavcopy close-to-close per canonical ISIN = investor return | all CARs | **Tested** — Gate A (0.508→0.038); **bounded** residuals: dividends unadjusted (small at daily scale), rights unadjusted (333 events, counted), relist artifacts clipped at ±50% |
| A8 | Beta-1 market adjustment (NIFTY subtraction) | CARs, arc | **Bounded** — the +52bp TEST drift it creates is neutralized by baselines/date FE; rolling β₁₂₀ enters the regressions as a control |
| A9 | Truncated CARs (delistings valued at last price) | HOSTAGE permanence | **Bounded, direction stated** — optimistic for HOSTAGE; trunc% printed per archetype, so permanence is *understated*, strengthening that finding |
| A10 | Two-way (stock × month) clustering captures the error dependence | all panel t-stats | **Maintained** — conservative relative to day-clustering; overlap additionally attacked head-on by the non-overlap subsample (§13B) |
| A11 | Estimated archetype labels ⇒ attenuation, not inflation | dummy coefficients | **Maintained** — classical errors-in-dummies logic; effects understated |
| A12 | INNOV's full-sample fit is a conservative competitor | §14 T3 | **Argued + bounded** — hindsight strengthens the *alternative*, so survival is the stronger claim; INNOV's own T2 finding carries the decomposition-only caveat |
| A13 | FII-slice PIN and FII-flow λ proxy their all-market versions | §16–17 | **Maintained, stated limitation** — the endorsement is relative loadings within one consistent slice |
| A14 | 15 bps one-way is the relevant cost benchmark | §18 verdicts | **Bounded** — the full 0/5/10/15/30 grid is reported (`outputs/metrics/backtest_metrics.csv`); conclusions stated as breakevens, not point verdicts |
| A15 | The liquid-946 universe represents FII price impact | external validity | **Measured (Module 14)** — the reversal does *not* generalize to the attributable tail (no-effect verdict, §13.6); tail dislocations are larger and unrecovered. The reversal claim is a liquid-market claim by measurement, not by scope restriction |

## 20. Why this research is valuable, audience by audience

The running "value" thread, consolidated — with the honest calibration
first: **the theory here is not new.** Temporary-versus-permanent impact of
liquidity versus informed trading dates to Kraus–Stoll (1972) and runs
through the entire block-trade and fire-sale literature. Pitched as a
conceptual discovery, this work would be rightly rejected. Its
contributions are measurement, evidence, and method — and after the
Module-13 referee tests, they rank in this order:

1. **The identifier audit** (§2.2) — a standalone methodology for any
   masked-panel dataset, and the part an external reviewer independently
   singled out as most original.
2. **The participant-composition measure**, with its incremental
   information content now quantified: +28% relative IC (t = 3.09) over
   the standard feature set, robust to direct flow-magnitude controls,
   tail-concentrated by three independent tests (§13.5).
3. **The transitory/permanent evidence** — model-class-independent
   (§5.6), passive-mechanics-excluded, PIN-corroborated.
4. **The HMM**, demoted by our own ablation to what it is: a taxonomy
   convenience that buys episode smoothness.

By audience:

**For empirical microstructure researchers.** The first (to our knowledge)
daily-frequency, entity-composition-based identification of the
transitory/permanent split in an emerging market — built from a data class
(masked depository records) that normally never leaves regulators. Plus
two reusable negative results: single-day concentration snapshots are
structurally invisible to HMMs (§5.4, with the autocorrelation diagnosis
and audit-legal smoothing repair), and disclosed block-deal data measures
visibility thresholds, not institutional concentration (§11.1).

**For anyone who must trade Indian equities.** The backtests locate the
value precisely: not a standalone strategy (breakevens 2–8 bps), but free
information for agents whose costs are already sunk. Concretely: a desk
liquidating a position into a volume-marked concentrated-selling episode is
selling into the one configuration that reverts +49–65 bp within a month;
a PM watching a holding decline *quietly* on dispersed FII selling is
watching the configuration that never comes back, and that an independent
PIN estimator flags as ~3× more informed.

*In money*: a desk instructed to sell ₹100 crore of a mid-cap that is
currently inside a concentrated-selling episode gives up roughly the
episode-end reversal — ~37–49 bp out-of-sample (bounce-free to standard window, §13.5b) — by
executing immediately rather than after the concentrated flow stops:
**≈ ₹37–49 lakh on that one order**, at zero incremental cost, from a screening variable computable
each evening from depository data. Statistical readers remember t = 2.95;
desks remember fifty lakh. The permanent side prices identically in
reverse: treating a quiet, dispersed-selling decline as "noise that will
come back" is, on this evidence, a decision to hold the one configuration
that does not.

**For risk and policy.** The composition lens inverts the usual panic
ordering: headline-grabbing concentrated FII dumps are the self-correcting
kind; quiet, broad, low-volume FII exits are the informative kind. A
regulator monitoring FII outflows gets a screening variable that separates
the two in real time from depository data it already possesses.

**For methodologists and students.** The project is a worked example of
pre-registered, gate-driven empirical research on messy financial data:
every claim has a falsification test that could have fired, five of them
did fire (§21), and the two headline "discoveries" are both *survivals of
attack* rather than products of search. The repository reproduces
everything from raw data in ~8 minutes (`python pipeline.py --all`).

**For the ML-in-finance conversation.** A documented case where the
fashionable escalation (GBT → LSTM) was evaluated under pre-registered
bars and correctly *declined*: the flexible learner's edge decomposed into
regime information plus static characteristics plus a sub-significance
timing residual (§15.4). Knowing when the model class is not the
bottleneck is worth more than another model.

## 21. The failure log: everything that broke and how we knew

The project's quality argument is not that nothing broke — thirteen things
broke — but that every break was caught by a gate, diagnosed read-only,
fixed at the root, and recorded. Full narrative:
`docs/research_log/FII_Module5_validation_log.md`; summary:

| # | Failure | Class | Caught by | Fix |
|---|---|---|---|---|
| 1 | Assumed NSE `prev_close` is CA-restated (it is raw) | false assumption | Gate 0: ex-day medians identical raw vs prev-close | built the CA factor pipeline (§8.4) |
| 2 | Combined split+bonus rows parsed as bonus-only (TITAN, ONGC…) | parser bug | tape-verification disagreements clustering at ×5/×10 | keyword-anchored split regex + dedupe guard |
| 3 | Pasted Colab cell silently lost "−1" in the return formula (all 5.94M rows) | execution ≠ file | internally contradictory printout → read-only diagnostic found ret_cc = fresh+1.0 exactly | byte-exact script execution, ancestor of this repo's runner |
| 4 | Symbol migrations put CA events on chains without the crash (+2860% fabrications) | data-class trap | application guard: post-adjustment \|ret\|>50% | guard nulls & logs (45 catches, all renames) |
| 5 | v1 fit-cap used most-recent-400 rows (recency-biased emissions) | subtle sampling bias | design review in 3A | random contiguous blocks |
| 6 | GMM threshold calibration produced a 49/51 coin-flip split (boundary +0.155) | model failure | its own pre-registered stability test (Δ=0.665) + k-means & test-GMM disagreement | quantile rule; three-way convergence at −0.513 |
| 7 | Event-study bootstrap tested vs zero under a +52bp drifting baseline (placebo "significant") | inference flaw | ROBOT row significance | difference-in-differences vs ALL_LABELED baseline |
| 8 | TRAIN regression n collapsed 66k→27k (index-return nulls poisoning rolling betas) | silent attrition | n printed per spec | zero-fill 0.6% index gaps; coefficients unchanged — never selection |
| 9 | Single-feature IC table lacked per-day dropna (F_breadth "−0.135" artifact) | diagnostic bug | implausibility check on one number | recomputed (+0.012); GBT itself unaffected |
| 10 | ROBOT placebo failed in TEST | control failure | flagged, then decomposed | structural (END = directional transition); HOSTAGE promoted as the valid null |
| 11 | Re-run calibration silently regenerated the rejected GMM thresholds (the falsification was manual in Colab) | provenance gap | artifact fingerprint diff on the census | falsification rule made binding in code (§6.4) |
| 12 | Tail-label build: probit NaNs passed `is_not_null` (polars NaN ≠ null), poisoning the quantile cuts — backbone printed an impossible `+nan` cut, BUY regime silently empty, census deformed into a false gate breach | library semantics trap | internally contradictory printout (a probit q₍29₎ cannot be +0.29) | `fill_nan(None)` at the transform + assert that cuts are finite and straddle zero (§13.6) |
| 13 | §7.2's "transition diagonal replicated at 0.95" in TEST was circular — a Viterbi decode using the frozen ~0.95-sticky prior looks ~0.95 almost regardless of TEST's true persistence | circular replication metric | reviewer probe: does the row survive removing the transition-matrix prior? | per-day classification on frozen emissions alone (no transition matrix, no look-ahead) isolates genuine TEST persistence at 0.888, not 0.95; a tempting EM-on-TEST refit was also tried and rejected (look-ahead + Baum-Welch's own stickiness bias); caveat added to §7.2 |

Two wrong economic narratives also died in the open: informed-Shark /
forced-Hostage (§10.3) and block-visibility (§11.1). The narrative that
survived did so against a pre-committed stopping rule.

## 22. Conclusion

Fourteen years of masked FII transaction data, one audited measurement
(within-day participation concentration), one deliberately blind regime
model, and a validation gauntlet that killed its authors' hypothesis twice,
yield two regularities:

1. **Concentration separates transitory from permanent FII price impact.**
   Concentrated episodes are liquidity events — volume-marked at the
   climax, reverting +49–65 bp over the following month (+37–58 bp with
   the first two days excluded, §13.5b), robust to fixed
   effects, two-way clustering, loser-reversal controls, overlap, horizon,
   dose–response, and a hindsight-advantaged flow-surprise control.
   Dispersed selling is the opposite in every measured respect — quiet,
   permanent, and carrying ~3× the informed-trading probability by an
   estimator that never saw a price.
2. **The surprise component of FII flow reverts** out of sample — a
   decomposition-level regularity, stated with its look-ahead caveat
   permanently attached.

The regularities are economically thin by design of the world: breakeven
trading costs of 2–8 bps explain why they persist un-arbitraged, and locate
their use with agents whose trading costs are already paid. The model
itself was shown *not* to be the binding constraint — a flexible learner
with more features could not clear a pre-registered dynamics bar — so the
honest frontier is more data (the illiquid tail, where the permanent-impact
mechanism should be strongest), not more model.

What we would defend most, if allowed one claim: not the 65 basis points,
but the *procedure* — pre-registered predictions, one gated test per step,
read-only diagnosis before any fix, falsification rules that bind in code,
and a repository where every number in this document regenerates from raw
data in eight minutes. The finding inverted its own hypothesis and
survived. That is what the procedure was for.

*— End.*
