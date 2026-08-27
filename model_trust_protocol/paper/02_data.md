# 2 · The record

<!-- provenance -->
*Every numeral in this section is traced. `convgap facts` verifies each against
`outputs/replication/descriptives.csv`, which `convgap replicate` regenerates
from source. 12/12 claims traced.*
<!-- /provenance -->

---

## 2.1 What the data is

The object of study is a transaction-level record of foreign institutional
investor activity in Indian equities, sourced from depository settlement data.
It covers **1 January 2011 to 31 March 2025** and **25,155,785 trades** across
5,960 distinct instrument identifiers. Each record carries a date, an instrument identifier, a masked
participant identifier, a side, and a traded value.

Three properties make the record unusual enough to be worth the exercise.

**It is transaction-level, not aggregate.** Published FII data in India is a
daily market-wide net figure. This record resolves to the instrument and the
day, and within that to the individual participant, which permits statistics
about the *composition* of flow — how concentrated or dispersed the
participation behind a day's activity is — that aggregate reporting cannot
support.

**It is entity-attributed.** Participant identifiers allow a day's activity in
one instrument to be decomposed across the entities responsible for it.

**It is of a class that normally does not leave regulators.** Depository
settlement records are not a commercial data product. Whatever the record turns
out to be worth, that figure is informative precisely because this is the
strongest version of the underlying data anyone is likely to obtain.

## 2.2 The constraint that governs every measurement built on it

Masked participant identifiers appear persistent and are not. They are
**re-minted at monthly boundaries.**

This is a measured fact rather than a stated caveat, and the way it was measured
determines how much weight it carries. Month-to-month reappearance rates are
uninformative, because they conflate "this entity did not trade" with "this
entity was re-masked." The decisive test runs the persistence statistic —
distinct months per identifier over the full span — on a **control population
whose true persistence is known independently**: brokers and clearing members, a
near-fixed real-world set numbering in the hundreds.

The control fragments.

| population | distinct identifiers | median months per identifier | mean |
|---|---:|---:|---:|
| masked participant | 363,663 | 1.0 | 1.209 |
| masked sub-account | 394,398 | 1.0 | 1.210 |
| **broker (control)** | **22,262** | **1.0** | **1.397** |

Over a 171-month span, a population of a few hundred real brokers presents as
22,262 identifiers with a median lifetime of one month. Since the control cannot
possibly have that turnover, the identifiers are re-minted, and the masked
populations — which behave identically — must be treated the same way.

Every entity statistic in this paper is therefore **within-day**, and at most
within-month. No measurement tracks an entity across a monthly boundary, and
no result depends on doing so. This is stated here rather than in a limitations
section because it is not a limitation of the analysis; it is a property of the
data that bounds what the analysis can be.

## 2.3 Instrument universe and the price panel

Returns come from NSE daily settlement prices, assembled into a panel spanning
the flow record's period. Two constructions are load-bearing.

**Corporate-action adjustment.** Split, bonus and similar factors are parsed
from exchange filings and verified against observed price ratios before
application, with a guard that nulls and logs any post-adjustment daily return
exceeding 50% in magnitude.

**Instrument identity across corporate events.** Identifier changes strand
history at exactly the events that matter most. A point-in-time identifier map
with issuer-bounded closure resolves each instrument to a canonical entity as
of each date.

The modelling universe is the liquid segment in which participation
concentration is measurable at all: **1,028 instruments** over **802,806
instrument-days**, spanning 24 January 2011 to 28 March 2025.

*(Project documents report 946 instruments. That figure is traced to a
provenance log written before the audit rebuilt the state object, and does not
describe the object analysed here. A companion claim of 939 issuers could not
be traced to any artifact and is not made.)*

A separately reported probe establishes that the
programme's main price result does *not* extend to the illiquid remainder,
which is a scope statement obtained by measurement rather than by restriction.

## 2.4 The frozen split

All parameters, thresholds and specifications are fixed on data through
**2021-04-30**. The test era begins **2021-07-01**. The intervening two months
are **masked at source** — they are absent from the flow feed — and the gap
doubles as an embargo buffer across a genuine structural break.

The two eras are not a repeat sample. The test era covers **851 instruments**
across **294,243 instrument-days**; the training era covers **618** across
**508,563**. Only **441 instruments appear in both**, a minority of each. Replication across the split is
therefore replication on a substantially different cross-section, which
strengthens what it establishes and weakens direct comparability of magnitudes. We report both eras for every result
without exception.

## 2.5 Missing data

Five months are absent from the flow feed and are **excluded, never imputed**:

- **2021-05 and 2021-06** — masked at source, coincident with the frozen split;
- **2023-06, 2023-09 and 2023-11** — present in the feed but carrying a
  wholly null direction flag, comprising **3.004% of trades** and falling
  entirely within the test era.

The concentration of the second group in the test era is stated as a known
asymmetry rather than adjusted for.

## 2.6 Derived samples

Different levels of the friction ladder require different evaluation samples.
Each is reported with its own scope rather than collapsed into a single figure.

| Application layer | Scored sample | Period |
|---|---|---|
| Price panel | 5,945,010 rows, 3,514 trading days | 2011 – 2025 |
| Modelling universe (states) | 802,806 instrument-days, 1,028 instruments | 2011-01-24 – 2025-03-28 |
| Cross-sectional volatility regression | 577,245 instrument-days | full |
| Market-level density, EWMA base | 2,739 scored days | full |
| Market-level density, GJR base | 2,235 scored days | full |
| Weekly density | 2,210 scored weeks | full |

The two market-level engines score different numbers of days because the
GJR base requires a longer burn-in. That difference is not incidental: it is
part of the mechanism by which the two reach opposite verdicts, and it is
reported wherever they are compared.

## 2.7 Data availability

The underlying transaction records are proprietary and cannot be
redistributed. The derived artifacts from which every number in this paper is
drawn — parameter vintages, scored panels, metric tables, run logs — are
retained, and each reported numeral is bound to a specific artifact, a locator
within it, and a SHA-256 digest recorded at verification. A reader with access
to the source trees can check any figure without reading the analysis code; the
digest tells them immediately whether they hold the same artifact we did.

Code implementing the verification protocol is available; see §3.5.
