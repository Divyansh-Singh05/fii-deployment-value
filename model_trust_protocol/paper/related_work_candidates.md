# Candidate references for the related-work section

Assembled 2026-08-25 by web search. **Nothing here has been read in full** — the
overwhelming majority are paywalled. Each entry records what the search
evidence actually supports, so you can decide which to obtain.

## How to read the status column

| | Meaning |
|---|---|
| **CONFIRMED** | Title, authors, journal and year all appeared together in a publisher or indexing result |
| **PARTIAL** | Title and venue confirmed; one or more of author list, year, volume or pages is inferred and must be checked before citing |
| **NEED ACCESS** | Full text not retrievable. Almost everything below is in this state — the flag marks papers whose *content* we need, not just their existence |

**Window:** published 2023 or later, per your constraint. Three entries from
2022 are included and marked, because they are the direct antecedents of papers
inside the window; drop them if the rule is strict.

**Excluded on principle:** arXiv and SSRN preprints, working-paper series, blog
posts and vendor research. Search returned many; none is listed.

---

## A · Replication and credibility — frames §1 and §5.3

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| A1 | Jensen, Kelly & Pedersen, *Is There a Replication Crisis in Finance?* | **Journal of Finance**, 2023 | CONFIRMED · NEED ACCESS | The direct counterweight to Hou–Xue–Zhang: argues most factors do replicate. Our §5.3 claim — that we study the survivors of the statistical filter — is only defensible if we engage with *both* sides of this dispute. **Highest priority to obtain.** |
| A2 | Bowles, Reed, Ringgenberg & Thornock, *Anomaly Time* | **Journal of Finance**, 2024 | CONFIRMED · NEED ACCESS | Concerns when an anomaly's returns actually occur relative to its discovery. Bears on our availability level (L2) from a different direction |
| A3 | *Replication crisis in finance? A unified modeling framework for reconciling pessimism and optimism* | **Finance Research Letters**, 2025 | PARTIAL (authors not captured) · NEED ACCESS | Attempts to reconcile A1 with Hou–Xue–Zhang. If it succeeds, our positioning paragraph should cite it rather than presenting the dispute as open |
| A4 | Harvey, *Editorial: Replication in Financial Economics* | **Critical Finance Review** | PARTIAL (year not captured) · NEED ACCESS | Editorial framing for the whole replication programme |
| A5 | *Understanding researchers' perceptions and experiences in finance research replication studies: a pre-registered report* | **Pacific-Basin Finance Journal** (Elsevier), 2024 | CONFIRMED · NEED ACCESS | Survey evidence on how common failed replication is among practising finance researchers. Would support §5.3 with something other than our own case |
| A6 | *The Null Result Penalty* | **The Economic Journal** (Oxford), 134(657), 2024 | CONFIRMED · NEED ACCESS | Quantifies the publication penalty on null findings. Directly relevant to why a paper like ours is uncommon |
| A7 | Pre-registration and pre-analysis plans, evidence from ~16,000 test statistics | **Journal of Political Economy: Microeconomics**, 2024 | PARTIAL (authors and exact title not captured) · NEED ACCESS | Empirical evidence on whether pre-registration reduces p-hacking. Our §3.4 leans on pre-registration; this is the paper that says how much it buys |
| A8 | Hou, Xue & Zhang, *Replicating Anomalies* | **Review of Financial Studies**, 33(5), 2019–2133, 2020 | CONFIRMED · figures verified | Outside the window but load-bearing. Already cited in `related_work.md` |

## B · Costs, capacity and implementability — frames §4.5 and §5.1

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| B1 | Muravyev, Pearson & Pollet, *Anomalies and Their Short-Sale Costs* | **Journal of Finance**, 80(6), 3639–3694, 2025 | CONFIRMED · figures verified | Our closest comparator in spirit and sharpest contrast in shape. Already cited |
| B2 | Penasse, *Understanding Alpha Decay* | **Management Science** | PARTIAL (year uncertain — possibly 2022) · NEED ACCESS | Mechanism for why an edge shrinks after discovery. Adjacent to but distinct from our execution level |
| B3 | *The race to exploit anomalies and the cost of slow trading* | **Journal of Financial Markets** (Elsevier), 2022 | CONFIRMED · **outside window** | Speed and cost as the binding constraint. Include only if the window is soft |
| B4 | *Exploring the factor zoo with a machine-learning portfolio* | **International Review of Financial Analysis** (Elsevier), 2024 | CONFIRMED · NEED ACCESS | Factor-zoo pruning under a portfolio criterion rather than a t-statistic — the same "does it convert" question at portfolio level |

## C · Backtest methodology and selection accounting — the gap §6.1 names

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| C1 | *Backtest overfitting in the machine learning era: a comparison of out-of-sample testing methods in a synthetic controlled environment* | **Knowledge-Based Systems** (Elsevier), 2024 | CONFIRMED · NEED ACCESS | Compares purged/combinatorial CV against conventional splits in a controlled setting where truth is known. **This is the paper our §6.1 selection-accounting gap should cite**, and it is the method we would use to close it |
| C2 | *Data snooping bias in tests of the relative performance of multiple forecasting models* | **Journal of Banking & Finance** (Elsevier), 2021 | CONFIRMED · **outside window** | Shows false-discovery rates rise sharply as the alternative set is trimmed. Directly relevant to our eight-strategy comparison. Include only if the window is soft |

## D · Machine learning and design choices — frames §5.4

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| D1 | Bagnara, *Asset Pricing and Machine Learning: a critical review* | **Journal of Economic Surveys** (Wiley), 2024 | CONFIRMED · NEED ACCESS | Survey; useful for one sentence positioning our instruments as instruments |
| D2 | *Empirical Asset Pricing via Machine Learning: the Role of Research Design Choices* | venue not captured, 2025 | PARTIAL — **venue must be established before citing** | The closest published work to our thesis that design choices dominate. If it is in a reputable venue this is a high-priority obtain; if it is a preprint, exclude |
| D3 | *Stock market anomalies and machine learning across the globe* | **Journal of Asset Management** (Springer), 2023 | CONFIRMED · NEED ACCESS | International, includes emerging and frontier markets. Relevant to our external-validity limitation |

## E · Alternative data — frames §5.6

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| E1 | Coyle et al., *What is the value of data? A review of empirical methods* | **Journal of Economic Surveys** (Wiley), 2024 | CONFIRMED · NEED ACCESS | A survey of methods for valuing data. Our paper is an instance of exactly this question answered for one dataset; this is where we position §5.6 |
| E2 | *Alternative data in finance and business: emerging applications and theory analysis* | **Financial Innovation** (Springer, open access), 2024 | CONFIRMED · **likely retrievable** | Review of the alternative-data literature. Open access, so obtainable without a subscription |

## F · Forecast evaluation and scoring — frames §3.2, §4.4 and the §6.1 scoring gap

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| F1 | Allen et al., *Tail calibration of probabilistic forecasts* | **Journal of the American Statistical Association**, 2025 | CONFIRMED · NEED ACCESS | **The single most relevant methods paper found.** Our §6.1 concedes that CRPS is a whole-distribution score used for a tail claim; this is the current treatment of exactly that problem. Highest methodological priority |
| F2 | *Proper Scoring Rules for Estimation and Forecast Evaluation* | **Annual Review of Statistics and Its Application**, 2025 | CONFIRMED · NEED ACCESS | Current survey. Supplies the "compare under multiple scoring rules" point our Murphy-diagram gap rests on |
| F3 | Arnold et al., *Decompositions of the mean continuous ranked probability score* | **Electronic Journal of Statistics**, 2024 | CONFIRMED · **open access** | CRPS decomposition into calibration and discrimination components. Would let §4.4 say *which* part of the score the ablation moves rather than only how much |
| F4 | *Bootstrapping out-of-sample predictability tests with real-time data* | **Journal of Econometrics** (Elsevier), 2024/25 | CONFIRMED (year to check) · NEED ACCESS | Predictive-ability testing under data revision. Our vintage architecture is the real-time discipline; this is its inferential counterpart |

## G · Econometric identification — frames the §6.1 staggered-DiD gap

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| G1 | *A comparative analysis of two-way fixed effects estimators in staggered treatment designs* | **Journal of Econometrics** (Elsevier), 2025 | CONFIRMED · NEED ACCESS | **The paper §6.1's first gap should cite.** A comparison of the available corrections, which is what we need in order to say which one our design would require |
| G2 | de Chaisemartin & d'Haultfœuille, *Difference-in-Differences Estimators of Intertemporal Treatment Effects* | **Review of Economics and Statistics** (MIT), 108(4), 863–, 2026 | PARTIAL (authorship inferred) · NEED ACCESS | Current statement of the intertemporal estimator |

## H · Risk-model validation — frames §3.2 and §6.1

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| H1 | *Evaluation of backtesting techniques on risk models with different horizons* | **Journal of Risk Model Validation** (Risk.net) | PARTIAL (year not captured) · NEED ACCESS | Horizon-dependence of VaR backtesting power. Directly relevant: our engine serves h=1 and h=5 and retires h=20 |
| H2 | *Backtesting value-at-risk: a comparison between filtered bootstrap and historical simulation* | **Journal of Risk Model Validation** (Risk.net) | PARTIAL (year not captured) · NEED ACCESS | Filtered historical simulation is what our stock-day engine is. This is a like-for-like comparison paper |

## I · Regime models — instruments, not subjects

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| I1 | *Forecasting realised volatility using regime-switching models* | **International Review of Economics & Finance** (Elsevier), 2025 | CONFIRMED (journal inferred from DOI prefix — verify) · NEED ACCESS | Regime switching for volatility across eight markets. One citation establishing our regime instrument as standard rather than novel |
| I2 | *Markov-switching threshold stochastic volatility models with regime changes* | **AIMS Mathematics**, 2024 | CONFIRMED · open access | Lower-tier venue. Include only if I1 is unobtainable |

## J · Microstructure, informed trading and flows — frames §2 and §4

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| J1 | *Estimation of the probability of informed trading models via an expectation-conditional maximization algorithm* | **Financial Innovation** (Springer, open access), 2024 | CONFIRMED · **likely retrievable** | Current PIN estimation practice and its known biases. Our programme used a PIN estimator and withdrew the claim built on it; this is the modern treatment |
| J2 | Ghachem & Ersan, *PINstimation: an R package for estimating probability of informed trading models* | **The R Journal**, 2023 | CONFIRMED · **open access** | The reference implementation. Useful for stating precisely which PIN variant was estimated |
| J3 | *Policy information uncertainty and foreign institutional investors' trading behaviour: evidence from India* | **Review of Quantitative Finance and Accounting** (Springer), 2025 | CONFIRMED · NEED ACCESS | The most recent journal work on FII behaviour in our exact market. Needed so §2 is not the only place India appears |
| J4 | *Order flow and cryptocurrency returns* | **Journal of Financial Markets** (Elsevier), 2026 | CONFIRMED · NEED ACCESS | Permanent versus transitory decomposition of order flow, recent. Our transitory/permanent framing needs one current citation |

## K · International and emerging-market replication — frames §6.2

| # | Reference | Venue | Status | Why it matters to us |
|---|---|---|---|---|
| K1 | *Replicating and Digesting Anomalies in the Chinese A-Share Market* | **Management Science**, 2023 | CONFIRMED · NEED ACCESS | Replication in a large emerging market. The closest published analogue to our external-validity position |
| K2 | *The world of anomalies: smaller than we think?* | **Journal of International Money and Finance** (Elsevier), 2022 | CONFIRMED · **outside window** | International out-of-sample anomaly tests. Include only if the window is soft |

---

## What I would obtain first

Six, in order. Each closes a specific hole rather than adding coverage.

1. **A1 · Jensen, Kelly & Pedersen (JF 2023)** — our §5.3 arithmetic assumes the pessimistic side of the replication dispute. If this paper's optimistic finding stands, that paragraph needs rewriting, not softening.
2. **F1 · Allen et al. (JASA 2025)** — the current treatment of tail calibration, which is the gap §6.1 concedes most directly.
3. **G1 · Journal of Econometrics (2025)** — needed to state *which* staggered-DiD correction our design requires, rather than listing four.
4. **C1 · Knowledge-Based Systems (2024)** — the selection-accounting method we would use to close the second gap.
5. **E1 · Coyle et al. (JES 2024)** — establishes that "what is this dataset worth" is a recognised question with existing methods, which is where §5.6 sits.
6. **A7 · JPE Micro (2024)** — how much pre-registration actually buys. §3.4 leans on it.

Three are open access and I can probably retrieve them without help: **E2**, **F3**, **J1/J2**.

## Two cautions

**Nothing here is a substitute for reading.** Every entry rests on a search
result. Venue and year are the fields most often wrong in such results, and a
mis-attributed citation in a paper about provenance would be an unusually poor
look. Each should be checked against the publisher's own record before it
enters the manuscript.

**D2 is the one to be careful about.** *Empirical Asset Pricing via Machine
Learning: the Role of Research Design Choices* is the closest thing found to our
own thesis, which is exactly the condition under which a citation is most
tempting and least scrutinised. Its venue was not established. If it turns out
to be a preprint, it does not go in — and if it is published in a strong venue,
our contribution claim needs re-reading against it.
