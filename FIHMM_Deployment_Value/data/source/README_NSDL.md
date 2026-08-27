# The source record, and why it is not here

## What it is

A transaction-level record of foreign institutional investor activity in Indian
equities, sourced from depository settlement data. 1 January 2011 to 31 March
2025; **25,155,785 trades** across 5,960 instrument identifiers. Each record
carries a trade date, an instrument identifier, a masked participant
identifier, a side, a traded value, and - decisively for this paper - **the
date the custodian reported it**.

Three properties make it worth the exercise: it is transaction-level rather
than aggregate; it is entity-attributed; and it is of a class that normally
does not leave regulators. Whatever it turns out to be worth is therefore
close to an upper bound on what a commercial equivalent would be worth on the
same market.

## Why it is not distributed

It is not a commercial data product and it cannot be redistributed. There is
no endpoint to call, which is why `src/data/download_nsdl.py` downloads
nothing and says so rather than failing at runtime.

**Access basis.** Obtained for academic research under an institutional
data-access arrangement, with participant identifiers masked at source by the
provider before release.

## The constraint every reader must know

Masked participant identifiers **appear persistent and are not**. They are
re-minted at monthly boundaries.

This was established rather than assumed, and the way it was established is
what makes it load-bearing. Month-to-month reappearance rates are
uninformative, because they conflate "this entity did not trade" with "this
entity was re-masked". The decisive test runs the persistence statistic on a
**control population whose true persistence is known independently**: brokers
and clearing members, a near-fixed real-world set numbering in the hundreds.

| population | identifiers | median months | mean |
|---|---:|---:|---:|
| masked participant | 363,663 | 1.0 | 1.209 |
| masked sub-account | 394,398 | 1.0 | 1.210 |
| **broker (control)** | **22,262** | **1.0** | **1.397** |

Over a 171-month span, a few hundred real brokers present as 22,262
identifiers with a median lifetime of one month. The control cannot possibly
have that turnover, so the identifiers are re-minted - and the masked
populations, which behave identically, must be treated the same way.

Consequently **every entity statistic in this programme is within-day**, and at
most within-month. No measurement tracks an entity across a monthly boundary
and no result depends on doing so. `src/data/schema.py` encodes this as
`NON_PERSISTENT_IDENTIFIERS`; treat any analysis that violates it as invalid.

## What you can reproduce without the raw record

| Reproducible from derived artifacts | Requires the raw record |
|---|---|
| Table 2 (controlled pair) | Table 1 and the L0-L2 regressions |
| Table 4 (ablation) | The reporting-lag curve |
| Survivor boundary conditions | The identifier-persistence test |
| Level 5 breakeven and cost grid | Corporate-action adjustment |
| All of `analysis/`, against fixtures | |

`reproduction/verify_results.py` reports `SKIP` rather than failing when a
source tree is unreachable, so a partial reproduction still returns a usable
report.

## Missing months, excluded and never imputed

- **2021-05, 2021-06** - masked at source. These coincide with the frozen
  split, so the embargo and the outage are the same two months.
- **2023-06, 2023-09, 2023-11** - present in the feed but carrying a wholly
  null direction flag; 3.004% of trades. All three fall inside the **test**
  era. That asymmetry is reported, not adjusted for.
