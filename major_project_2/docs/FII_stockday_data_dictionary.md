# FII Stock-Day Panel — Data Dictionary, Validation & Design Rationale
*Companion to `FII_Module1_findings.md`. Describes `stockday_features.parquet`. Updated 2026-07-08.*

## 0. What this table is

- **Grain:** one row per **(canonical ISIN, trading day)** — a *stock-day*.
- **Scope:** ~2.42 M stock-days, 3,815 stocks, 2011-01-03 → 2025-03-29 (2025 partial, data ends March).
- **Source:** NSDL FII equity buy/sell trades (`TR_TYPE ∈ {1,4}`, `RATE > 0`, `RFDE_INSTR_TYPE = REG_DL_INSTR_EQ`), ~24 M rows.
- **Purpose:** substrate for a 3-state HMM that labels each stock-day as **Robot / Shark / Hostage**. Only three columns (`F_persist`, `F_block`, `F_entity`) feed the model; the rest are the raw material, intermediates, and audit trail behind them.

---

## 1. How a row is produced (pipeline stages)

1. **Scope filter** — keep equity buy(1)/sell(4) with `RATE > 0` (drops corporate-action legs, all zero-rate) and `RFDE_INSTR_TYPE = REG_DL_INSTR_EQ` (drops non-equity).
2. **Drop unattributable trades** — `ISIN` null/empty (278 of 24 M, 0.001%).
3. **Canonical ISIN** — join the ISIN-lineage map so a company whose ISIN changed (temp-ISIN / restructuring) isn't split across reincarnations; `cisin = coalesce(mapped_canonical, ISIN)`.
4. **Entity ID (per-format)** — masked `FII` decoded by era: early (17/18/19-char) strip trailing `YYYYMM`; late (13/14-char) use raw; drop null / literal `"(null)"`. *(See findings doc §3 — IDs are re-minted ~monthly, so entity identity is within-month only.)*
5. **Mask the 2021 structural break** — remove May–Jun 2021 before any trailing window touches it.
6. **Aggregate to stock-day** — sum buy/sell value, counts, quantity → derive NET/GROSS/N/mean-trade-size.
7. **Axis 3 (within-day entity book-HHI)** — computed on the sell side, participation-weighted, coverage-gated.
8. **Axes 1 & 2 (trailing, past-only)** — 20-day backward windows, shifted by 1 so today is never in its own baseline.
9. **Liquidity floor** — `eligible = N ≥ 5`.
10. **Normalise** — each raw feature → within-day cross-sectional percentile rank → probit (inverse-normal).

---

## 2. Column dictionary

### 2.1 Keys
| Column | Type | Formula / meaning | Utility |
|---|---|---|---|
| `cisin` | str | canonical ISIN (lineage-mapped) | stock identity; groups a company across ISIN changes |
| `TR_DATE` | date | trade date | time index; sequence order for the HMM |

### 2.2 Base aggregates (the substrate; not model inputs)
| Column | Type | Formula | Utility |
|---|---|---|---|
| `buy_value` | f64 | Σ `VALUE_INR` where `TR_TYPE=1` | gross FII buying (₹) |
| `sell_value` | f64 | Σ `VALUE_INR` where `TR_TYPE=4` | gross FII selling (₹) |
| `n_buys` | u32 | count of buy trades | trade-count buy leg |
| `n_sells` | u32 | count of sell trades | trade-count sell leg |
| `total_qty` | f64 | Σ `QUANTITY` | shares transacted (volume proxy) |
| `NET` | f64 | `buy_value − sell_value` | net directional flow (sign = pressure direction) |
| `GROSS` | f64 | `buy_value + sell_value` | total activity / turnover |
| `N` | u32 | `n_buys + n_sells` | total trades; drives the liquidity floor |
| `mean_trade_size` | f64 | `GROSS / N` | ₹ per trade — the raw blockiness signal |

### 2.3 Axis-3 support (entity concentration)
| Column | Type | Formula | Utility |
|---|---|---|---|
| `valid_sell` | f64 | Σ sell value with a valid entity ID | numerator of Axis-3 coverage |
| `a3_coverage` | f64 | `valid_sell / sell_value` | share of sell value that is entity-attributable |
| `entity_hhi_raw` | f64 | participation-weighted sell-book HHI (below); null if `a3_coverage < 0.5` | raw Axis-3; concentrated vs dispersed sellers |

> **RENAMED AND REDEFINED (2026-08-23, audit item 1.)** `entity_hhi_raw` is now
> `entity_conc_raw` → `F_entity_conc`, and it is the Herfindahl of participation
> shares **within the stock-day**: `conc(s,d) = Σ_e (v(e,d,s) / Σ_e' v(e',d,s))²`.
> The book-focus quantity documented below is retained as `bookfocus_raw` →
> `F_bookfocus` and is **not** fitted by the HMM. The two correlate r = −0.2460.
> The description below documents the pre-audit column.

**`entity_hhi_raw` formula (PRE-AUDIT — superseded).** For each entity *e* and day *d*, let *v(e,d,s)* = *e*'s sell value in stock *s*. Entity book-HHI:
`HHI(e,d) = Σ_s ( v(e,d,s) / Σ_s v(e,d,s) )²` — 1 if *e* sold one name, →0 if spread over many.
Stock-day value: `entity_hhi_raw(s,d) = Σ_e v(e,d,s)·HHI(e,d) / Σ_e v(e,d,s)` — participation-weighted over the entities selling *s* that day.
*High = concentrated sellers (Shark distributing). Low = dispersed sellers (Hostage fire-sale).*

### 2.4 Trailing intermediates (past-only; scaffolding, not model inputs)
| Column | Type | Formula | Utility |
|---|---|---|---|
| `pers_signed` | f64 | `mean(sign(NET), 20d).shift(1)` per stock | directional consistency of flow, [-1,1] |
| `intensity` | f64 | `mean(\|NET\|/GROSS, 20d).shift(1)` per stock | how one-sided the flow has been |
| `mts_base` | f64 | `mean(mean_trade_size, 20d).shift(1)` per stock | the stock's own normal trade size (price-neutral baseline) |

### 2.5 Raw features (pre-normalisation)
| Column | Type | Formula | Axis / utility |
|---|---|---|---|
| `persistence_raw` | f64 | `pers_signed × intensity` | **Axis 1** — sustained directional pressure; separates Robot (≈0) from Shark(+)/Hostage(−) |
| `blockiness_raw` | f64 | `mean_trade_size / mts_base` | **Axis 2** — blockiness *surprise* vs the stock's own norm; price-neutral (self-normalised) |
| `imbalance` | f64 | `NET / GROSS` | **standby Axis 4** — contemporaneous directional imbalance; used only if the three don't separate |

### 2.6 Flag + model inputs
| Column | Type | Formula | Utility |
|---|---|---|---|
| `eligible` | bool | `N ≥ 5` | liquidity floor; only eligible days are ranked/modelled |
| `F_persist` | f64 | probit(within-day rank of `persistence_raw`, eligible only) | **HMM input — Axis 1** |
| `F_block` | f64 | probit(within-day rank of `blockiness_raw`, eligible only) | **HMM input — Axis 2** |
| `F_entity` | f64 | probit(within-day rank of `entity_hhi_raw`, eligible only) | **HMM input — Axis 3** |

**Normalisation formula (all three F_*):** within each day *d*, over eligible stock-days,
`p = rank(x) / (n_valid + 1)` (percentile in (0,1)) → `F = Φ⁻¹(p)` (probit). Non-eligible / warm-up / uncovered rows are null.

**Null semantics:** a `F_*` is null when the day is illiquid (`N<5`), in a stock's ~15-day warm-up (no backward window), or (for `F_entity`) below Axis-3 coverage. ~1.5 M of 2.42 M rows are null on this basis — expected, and ignored by the model.

---

## 3. The leakage discipline (why the windows are shaped this way)
- Every trailing statistic is `.shift(1)` — today is never in its own baseline.
- Row-based windows over surviving trading days **skip** the masked 2021 gap instead of spanning it.
- The **only** contemporaneous signal allowed is the within-day cross-sectional rank (market-relative "today" is known today).
- No price / VIX / macro anywhere in the features.

---

## 4. Validation metrics (how we earned trust in the data)

### 4.1 Entity audit (gate for Axis 3) — see findings doc §3 for full detail
- **Verdict:** masked IDs re-mint ~monthly; **no stable cross-month identity.** Proof: `≥12-month persistence = 0.0%` over 84–98-month spans, *including brokers* (a ~300-entity real set fragmented into 12,700 early / 9,221 late IDs; broker `max_months` = 10 early, 2 late).
- **Consequence:** Axis 3 is a **single-day / within-month** feature. Within-month IDs are internally consistent (per-month counts sane: ~150 brokers, ~3,000 FIIs), so the book-HHI is computable; cross-month fund tracking is not.
- **ID scheme:** two eras (early `PREFIX+core+YYYYMM`; late plain 13/14-char). FII missing ~13% overall, 30–40% in 2024–25.

### 4.2 Axis-3 coverage
- **`axis3_coverage = 1.0` every year, incl. 2024–25.** Coverage is **value-weighted**, so even with 30–40% of FII *rows* missing an ID, valid-ID sell *value* clears the 50% floor on essentially every stock-day. Axis 3 survives the late years.

### 4.3 Feature validation (post-probit, complete cases = 929,195)
- **Distribution ~N(0,1):** means ≈ 0 (−0.042, 0.023, −0.001), std ≈ 0.97, quartiles ≈ −0.67 / 0 / +0.67 → probit transform correct; no ties artefact.
- **Independence (correlation, want ≪ 0.6):**
  | | persist | block | entity |
  |---|---|---|---|
  | persist | 1.00 | 0.015 | −0.142 |
  | block | 0.015 | 1.00 | 0.195 |
  | entity | −0.142 | 0.195 | 1.00 |
  Near-orthogonal; the two non-zero terms point the economically correct way (concentrated players also take bigger blocks → +0.19; dispersed sellers skew sell-persistent → −0.14).
- **Sample:** 929,195 complete-case stock-days = 80.7% of 1,151,531 eligible (the ~19% gap = warm-up + coverage). Liquidity floor removes ~52% of all stock-days (`N<5`).
- **Hygiene fixes applied:** NaN→null after probit (else warm-up masqueraded as present, poisoning `corr`/`describe`); 278 null-ISIN phantom rows dropped.

### 4.4 Config (dials most worth sweeping)
`BASE_WIN=20`, `MIN_SAMPLES=15`, `MIN_TRADES=5`, `COVERAGE_MIN=0.5`, 2021 mask = May–Jun.

---

## 5. Design question: is a per-ISIN, per-day feature the right unit? (deep dive)

**Short answer: yes — it is both necessary and the correct common denominator.** The reasoning, and the tensions:

**Why per-stock-day is right.**
1. **The label is a time-varying property of a stock, not a static one.** A stock can be a Shark target this quarter and Robot-driven the next. The archetype describes *what is happening to a stock on a day*, so the unit of decoding must be the stock-day. A per-stock static label or a per-entity label can't express "Reliance on 2020-03-23 looks like a fire-sale."
2. **The HMM's power is temporal.** Regimes persist (accumulation campaigns, redemption episodes) and then resolve — which is exactly what the later forward-return check tests (Shark *drifts*, Hostage *reverses*, Robot is *transient*). That onset→duration→reversal structure only exists in a per-stock **daily sequence**; coarsening to weekly/monthly would blur the block-deal spike and the fire-sale day the features are built to catch.
3. **Daily matches the event granularity.** Block deals and redemption dumps are day-level events. The trades are day-dated. Daily is the natural resolution of the phenomenon.
4. **It is the common denominator that unifies stock-centric and entity-centric archetypes.** Robot and Shark are stock-centric ("flow into this name"). The **Hostage is entity-centric** (a fund dumping its whole book) — but it *manifests* as dispersed selling pressure spread across many stock-days. Projecting the entity's book-HHI onto the stock-day (participation-weighting) is what lets all three archetypes live in one feature space. Without the stock-day projection, the Hostage couldn't share an axis system with the other two.

**The tension we accepted.** Axis 3 is intrinsically an *entity* process reduced to a stock-day snapshot. Because IDs don't survive cross-month (§4.1), that snapshot is genuinely single-day — we cannot watch a fund liquidate over a quarter, only see that *today* this stock's sellers are dispersed. That's why the Hostage is the fragile state: it's identified from a noisier, lower-information projection than the other two. This is a real limitation, not a modelling error — and it's the honest ceiling on Hostage precision.

**Why "daily" isn't naïvely noisy.** Two of the three features are *not* single-day quantities: persistence is a 20-day trailing state updated daily, and blockiness is today's size against a 20-day own-baseline. Only Axis 3 is purely contemporaneous. So the panel is a hybrid — **daily output grain, but features that embed memory** — which is what keeps a single noisy day from dominating a stock-day's coordinates.

**The one caveat this creates for the HMM step.** Pooling 929 k stock-days into shared emissions is **observation-weighted**: hyper-active large-caps contribute far more rows than thin mid-caps, so the fitted emissions can tilt toward large-cap behaviour. Within-day ranking mitigates it (each day's cross-section is self-normalised) but doesn't fully remove it. Worth considering per-stock weighting or a row cap when we fit — flagged for Module 2, not a panel-level fix.

**What we deliberately did *not* build.** Per-entity daily features (an entity's own trajectory) are off the table — the audit killed cross-month entity identity, so an entity time series can't be assembled. The entity signal enters *only* through its within-day projection onto the stock-day. That is the correct and only defensible use of the entity data given the masking.

---

## 6. Known limits
- Hostage identifiable only within-day (no cross-month fund tracking) → the fragile state.
- Axis-3 attribution thinnest in 2024–25 (heaviest ID missingness), though value-weighted coverage holds.
- Pooled HMM emissions are observation-weighted toward active names (Module 2 consideration).
- 2025 is a partial year (through March); trailing windows near the edge are shorter.
- Features are validated *statistically* (distribution + independence), not yet *economically* — the state characterisation + forward-return check earn that later.
