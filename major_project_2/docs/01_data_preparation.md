# Data Preparation

Phase: `data_prep` (+ `canonical`). Stages: `price_panel`, `ca_factors`,
`apply_adjustment`, `canonical_panel`. Audits: `data_audit`,
`adjustment_diagnose`, `data_lineage`, `attrition`, `universe_audit`,
`isin_provenance`, `cisin_validation`, `tape_canonicalization`,
`isin_accounting`, `noisin_probe`, `universe_integrity`.

## Purpose

Build a **certified** daily price panel: survivorship-free, corporate-action
adjusted, canonically keyed, with every repair verified by a gate. No feature
engineering happens here.

## Inputs

| Dataset | Location | Content |
|---|---|---|
| NSE bhavcopy | `VALIDATION_DATA/bhavcopy_parquets/prices_YYYY.parquet` | EOD OHLCV per (symbol, series, date), incl. delisted names |
| Corporate actions | `nse_corporate_actions.csv`, `bse_corporate_actions.csv`, `ca_raw.parquet` | ex-dates + purpose strings ("Bonus 1:1", "Face Value Split…") |
| Macro | `nifty50.parquet`, `sp500.parquet`, `usdinr.parquet`, `india_vix.parquet` | market/currency/vol context |
| ISIN tables | `ISIN_MAPPING/isin_lookup.parquet`, `active/inactive_isins*.csv` | identity change records |

## Outputs

`returns_panel.parquet` → `returns_panel_v2.parquet` (CA-adjusted) →
`returns_panel_v3.parquet` + `states_v3.parquet` (canonical keys).

## Methodology

### 1. Tape repair (`price_panel`)

Three verified defects, three repairs:
- **R1** `prices_2020` carries year-0020 dates → `offset_by("2000y")`.
- **R2** `prices_2011` ISINs are all null → symbol→ISIN backfill (validated
  later: 98.45% coverage in v3, no rework needed).
- **R3** multi-series duplicates → keep EQ > BE > BZ.
- Macro series from yfinance arrive tz-shifted (start on a Sunday) → +1 day.

Gate **G1/G2**: row-count and model-join coverage printed and bounded.

### 2. Corporate-action adjustment (`ca_factors`, `apply_adjustment`)

**The false assumption this replaced** (Gate 0 failure, logged): NSE bhavcopy
`prev_close` is *not* restated on ex-dates — `close/prev_close − 1` is a raw
return. Adjustment had to be built, not assumed.

For a split of face value $f_{old} \to f_{new}$ and a bonus of $a{:}b$
(a new shares per b held), the price divisors on the ex-date are:

$$F_{split} = \frac{f_{old}}{f_{new}}, \qquad F_{bonus} = \frac{a+b}{b}$$

The adjusted close-to-close return over an ex-date is

$$r^{adj}_t = (1 + r^{cc}_t)\cdot F - 1,\qquad r^{cc}_t = \frac{P_t}{P_{t-1}} - 1 .$$

**Every factor is tape-verified**: the observed ex-day ratio
$P_{t-1}/P_t$ must fall within $[F/1.3,\, 1.3F]$. 99.1% of applied factors
confirm. Combined events ("Bonus 1:1 AND Face Value Split") required a
keyword-anchored parser (the first-numbers regex silently dropped the split
component — caught by the disagreement clustering at ratios ≈5 and ≈10).

**Application guard**: if $|r^{adj}| > 0.5$ where a factor was applied, the
return is nulled — this caught 45 symbol-migration artifacts where a factor
would otherwise fabricate returns like +2860% across a rename.

Gate **A** (pre-registered): median $|r^{adj}|$ across all confirmed ex-days
must be < 0.05. Result: **0.508 raw → 0.038 adjusted** over 803 ex-days.

### 3. Canonical identity (`canonical_panel`)

ISINs mutate; 5,960 raw FII-side ISINs resolve to 3,812 canonical identities.
The closure rule is **CA-type-conditional and issuer-bounded**:

- ISIN characters 4–7 are the **issuer code** — the entity key. Legitimate
  closure lives *within* one issuer code; links crossing issuer codes are
  merger/acquisition candidates and mean **identity death**, not mapping.
- Value-preserving events (split, bonus, FV change) map old → the issuer's
  latest-trading ISIN. Value-changing events (merger, demerger) terminate.
- A **180-day trading-overlap guard** prevents collapsing genuinely distinct
  co-existing lines (DVR / partly-paid, e.g. Bharti Airtel IN9397D01014).

The closure is applied to **both** the price tape and the model's state
history (dual-side), merging 9 fragmented identities (Tata Steel, Alok,
Ruchi/Patanjali, …). Coverage of model stock-days: **90.4% → 98.49%**,
uniform across archetypes. Model universe: 946 cisins = **939 companies**.

Degenerate keys (fabricated `NOISIN<digits>`, null ISINs) were audited
(`noisin_probe`): 0 of 946 model cisins affected; the ~0.03% invalid raw
records are mutual-fund/non-equity instruments, correctly out of scope.

## Validation: why these gates exist

| Gate | Failure it detects | Interpretation of PASS |
|---|---|---|
| G1/G2 (panel) | silent row loss, model-tape key mismatch | joins are keyed correctly |
| ratio band (factors) | mis-parsed CA ratios | factor ≈ what the tape did |
| Gate A (ex-day median) | unadjusted or double-adjusted returns | returns are economically real across CA events |
| application guard | factor applied to wrong symbol-chain | no fabricated returns |
| dual-side closure checks | tape and model disagreeing on identity | event windows span identity changes |

A worked example of why this matters: without Gate A, a 1:10 split day enters
the event study as a −90% "return" on a HOSTAGE episode — fatal for a
reversal test. Without the closure, Alok Industries' insolvency-and-relisting
(a HOSTAGE archetype) splits into two unrelated stubs and drops out of the
permanence test.
