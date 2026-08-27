# FII Regime-Detection — Module 1 Findings & Decisions
*Research log. Last updated: 2026-07-08.*

## 0. Objective

Build an HMM that classifies each **stock-day of FII trading activity** into three behavioural archetypes:

| Archetype | Economic story | Flow signature |
|---|---|---|
| **Robot** | Passive / index-rebalance flow riding the broad tide | Transient, low persistence, small uniform (VWAP) slices |
| **Shark** | Informed conviction accumulation | Persistent one-way, large blocks, **concentrated** book |
| **Hostage** | Forced fire-sale (redemptions / liquidation) | Persistent one-way **sell**, fragmented, **dispersed** book (Coval–Stafford) |

No single feature separates all three; separation comes from where a stock-day lands on three axes at once.

---

## 1. Data

- **Source:** NSDL FII transaction feed, one `.parquet` per year (~2010–2025), in the `ISIN_MAPPING/` Drive folder.
- **Grain:** one row per reported FII transaction. Key fields: `FII`, `SUB_ACC`, `BRKER` (masked entity IDs), `ISIN`, `TR_DATE`, `RFDE_RPT_DT`, `TR_TYPE` (i8), `RATE`, `QUANTITY`, `VALUE_INR`, `RFDE_INSTR_TYPE`.
- **Real-trade scope** (used throughout): `TR_TYPE ∈ {1 (buy), 4 (sell)}` **AND** `RATE > 0` **AND** `RFDE_INSTR_TYPE == "REG_DL_INSTR_EQ"`. This strips corporate-action legs (types 7/15/16/17, all 100% zero-rate) and non-equity instruments. ≈ **28 M** equity buy/sell rows across the sample.
- **Related workstream (input):** ISIN lineage tables (`active_isins.csv`, `inactive_isins.csv`) already built — every inactive ISIN maps to its canonical ACTIVE successor or is flagged terminal. **Integration point:** the stock-day panel should key on the *canonical* ISIN so a company whose ISIN changed (temp-ISIN / restructuring) isn't split across reincarnations.

---

## 2. Architecture decisions (fixed)

### 2.1 FII_NET_INNOV is validation-only
`FII_NET_INNOV` (flow-surprise residual) is a **Module 4 validation object**, built the AAK way *with* look-ahead, and **never enters the HMM**. Reason: it is the independent sort used to *check* the decoded states; if the same construct also fed the model, validation would be marking its own homework. A leakage-safe flow-surprise would be a *different, differently-named* object — and using it as a feature forfeits it as the validation sort. Decision: keep INNOV pure-validation.

### 2.2 Base panel
From scoped trades, aggregate to **(canonical ISIN, day)**: `buy_value, sell_value, buy_count, sell_count, total_quantity`. Derive `NET = buy − sell`, `GROSS = buy + sell`, `N = total trades`, `mean_trade_size`. Every feature is built from these.

### 2.3 The three axes
1. **Persistence (splits Robot).** Trailing ~20-day mean of `sign(NET)`, scaled by intensity. Low ⇒ Robot (transient); high ⇒ Shark/Hostage (sustained one-way). Flow-only, strictly backward, trivially leakage-safe. Strongest axis.
2. **Blockiness (isolates Shark).** Mean trade size, ranked. Large ⇒ Shark; small/fragmented ⇒ Robot/Hostage. Moderate.
3. **Entity concentration (splits Shark vs Hostage — the hard one).** Per stock-day, participation-weighted **entity-book HHI**: are the entities trading this name concentrated across their book (Shark) or dispersed (Hostage)? **Gated on the entity audit (Section 3).**
- Optional 4th (only if 3 don't separate): contemporaneous `NET/GROSS` directional imbalance.

### 2.4 Normalisation rule
Every feature → **within-day cross-sectional percentile rank**. This single step neutralises the ~254% breadth growth and the 2021 structural break (a "high" day means high relative to *that day's* market). Plus: all trailing windows strictly backward; the **May–Jun 2021** window masked before any window touches it; **no VIX/price/macro** anywhere in the features.

### 2.5 Model
3-state Gaussian HMM, diagonal covariance (Hostage is rare — diagonal keeps its emission estimable), Viterbi decode, **label states post-hoc from feature-mean signatures** (never assume state order). Clustering is *not* proof — proof is post-hoc characterisation + the forward-return check (Shark drifts, Hostage reverses, Robot is transient).

---

## 3. Entity audit — the make-or-break test for Axis 3

**Question:** do the masked entity IDs carry a **stable identity across time**, so we can measure an entity's book concentration — and can we track it across months (fire-sales play out over weeks)?

The audit ran in three iterations because the first assumption about the ID format was wrong.

### 3.1 ID format is heterogeneous (not the uniform scheme first assumed)
The initial sample looked like `F + 10 digits + YYYYMM` (17 chars). Profiling (shape = letters→A, digits→9) showed that pattern is **only ~15% of rows**. Actual `FII` families:

| Shape | Example | Chars | Rows | Suffix = report-month? |
|---|---|---|---|---|
| `A999999999999` | `F387862664940` | 13 | 8.38 M | no |
| `A9999999999999` | `F1182835927334` | 14 | 5.44 M | no |
| `A9999999999999999` | `F5944222638201101` | 17 | 4.29 M | **yes (~0.91)** |
| `A999999999999999999` | `F233433053393201101` | 19 | 2.16 M | ~0.77 |
| `(null)` (literal string) | `(null)` | 6 | 1.73 M | — (missing) |
| `null` | — | — | 1.96 M | — (missing) |

### 3.2 The scheme changed over time (two eras)
`length × year` and `suffix==month × year` show a clean regime change:

- **Early era (~2011–2015):** 17/19-char IDs = `PREFIX + core + YYYYMM`. Suffix matches the report month ~0.93. → identity = strip the trailing 6 chars.
- **Late era (~2021–2025):** 13/14-char IDs, **no month suffix** (`suffix==month = 0.0`). → identity = the raw ID.
- **Transition** ~2016–2020.

**Missingness** (`(null)` literal + true null) ≈ 13% overall but **year-skewed**: FII null-rate 0.0 in 2011 → **0.30 in 2024, 0.39 in 2025**. `SUB_ACC` is missing less than `FII`.

### 3.3 Retention, measured per-era with correct per-format identity
Month-to-month retention (share of an era's month-*m* entities that reappear in *m+1*), and `distinct-overall / summed-per-month` (≈1 ⇒ every ID unique to its month):

| Era | Level | median retention | distinct/summed | entities/mo |
|---|---|---|---|---|
| Early (17/19) | FII | 0.317 | 0.503 | 2,271 |
| Early | SUB_ACC | 0.245 | 0.600 | 3,512 |
| Early | **BRKER (control)** | **0.372** | 0.474 | 319 |
| Late (13/14) | FII | 0.242 | 0.775 | 3,303 |
| Late | SUB_ACC | 0.283 | 0.732 | 2,822 |
| Late | **BRKER (control)** | **0.385** | 0.628 | 150 |

The **broker control failed**: only ~150–320 brokers trade in a month (a near-fixed real-world set), so stable IDs would retain ≈ 1.0. Retention ~0.37 — nearly identical across two totally different formats — was the first red flag.

### 3.4 Decisive test — months-per-entity distribution
Retention conflates "didn't trade next month" with "got re-masked." The number of *distinct months each ID appears in* separates them cleanly:

| Era (span) | Level | distinct IDs | median mo | max mo | ≥12-month |
|---|---|---|---|---|---|
| Early (84 mo) | FII | 95,987 | 2 | 10 | **0.0%** |
| Early | SUB_ACC | 177,093 | 1 | 10 | **0.0%** |
| Early | **BRKER** | **12,700** | 2 | **10** | **0.0%** |
| Late (98 mo) | FII | 250,949 | 1 | 2 | **0.0%** |
| Late | SUB_ACC | 202,334 | 1 | 2 | **0.0%** |
| Late | **BRKER** | **9,221** | 2 | **2** | **0.0%** |

**Verdict — unambiguous:** `≥12-month = 0.0%` everywhere, over 84–98-month spans, *including brokers*. A ~300-broker real set fragments into **12,700 (early) / 9,221 (late)** distinct IDs; no broker ever appears in more than **10 (early) / 2 (late)** months.

> **The masked IDs are re-minted on a short cycle (~monthly). There is NO stable cross-month entity identity, at any level, in any era.**

---

## 4. What the audit means for the model

- **Cross-month entity tracking is dead.** We cannot follow a fund's liquidation across a quarter.
- **Within-month / single-day identity is intact** (per-month counts are sane — ~150 brokers, ~3,000 FIIs — so IDs are internally consistent inside a month).
- **Axes 1 & 2 are unaffected** — they use no entity IDs.

### Decision
- **Three-axis model proceeds**, with **Axis 3 restricted to a single-day (within-month) entity-book HHI.** The Hostage is identified by **dispersed sell-side book concentration within the day**, not by cross-month persistence. Fire sales cluster on redemption days, so this is a valid (noisier) proxy — the Hostage remains the **fragile state**, now for a measured reason.
- **Entity level = `FII`** (umbrella best captures manager-level dispersal), **`SUB_ACC` as robustness check.**
- **Add a per-stock-day ID-coverage floor** (skip stock-days where too much gross value has a null/`(null)` ID). Coverage is thinnest in 2024–25 (30–40% missing FII) — a stated limitation.

### Caveat on preliminary numbers
Book-breadth figures seen earlier (FII median ~2 names/day, p90 9; ~54% multi-name/day; brokers far broader) came from the **first, corrupted-core** audit and must be **recomputed with the correct per-format identity** before being trusted. Directionally they indicate real within-day dispersion variance exists.

---

## 5. Construction choices pending (defaults proposed)

| # | Choice | Proposed default |
|---|---|---|
| 1 | Axis 3 form | Within-day, sell-side entity-book HHI, participation-weighted to the stock-day, coverage-floored |
| 2 | Blockiness | **Shares** per trade (`QUANTITY`/N), not rupees/trade (avoids price-level confound under within-day ranking) |
| 3 | Persistence | Feed both signed and \|·\| of 20-day mean sign(NET); intensity scaler = trailing \|NET\|/GROSS |
| 4 | Rank → Gaussian | Probit-transform within-day ranks so diagonal-Gaussian emissions are honest |
| 5 | Liquidity floor | Drop stock-days below a min trade count / gross before ranking |

---

## 6. Next step

Build the **base panel + three probit-ranked features** (Axis 3 = FII within-day book-HHI), ending with an **ID-coverage + feature-correlation diagnostic** so we see immediately how much of the panel Axis 3 covers (esp. 2024–25). First real modelling artifact.

## 7. Explicitly deferred
Axis B/C beyond the minimal vector, HMM state-count/BIC (Module 2), and VIX/EPFR/rebalance validation (Module 3). None block Section 6.
