# Masked Identifiers Are Not Identities
## An audit protocol for anonymized transaction panels, with a case study on Indian depository data

*Standalone methodological note extracted from the FII flow-regimes
project (repository: this codebase; long-form treatment:
`FII_thesis.md` §2.2; primary record:
`../research_log/FII_Module1_findings.md` §3).*

### Abstract

Regulators and depositories increasingly release transaction-level data
with masked participant identifiers. Researchers routinely treat these
masks as stable pseudonyms — computing participant persistence, tracking
portfolios through time, measuring entry and exit. We show, on fourteen
years of NSDL records of foreign-institutional trading in Indian equities
(~28 million transactions), that this assumption can fail totally and
*invisibly*: the masked IDs are re-minted on approximately a monthly
cycle, so that a real population of ~300 brokers appears as 12,700
distinct identities, and **no identifier of any kind, at any level, in
any era, appears in twelve or more months of an 84–98-month span**. Naive
use of such IDs does not add noise to entity-level statistics; it
*fabricates* them. We present the three-step audit protocol that
established this — format profiling, the insufficiency of retention
rates, and a months-per-identity test against a control population of
known persistence — and the design consequences for research when the
audit fails. The protocol runs in minutes on any masked panel and should,
we argue, be a mandatory first step before any cross-period entity
statistic is computed.

---

### 1. The problem

Anonymized microdata is the growth area of empirical finance:
exchange-provided order data, depository transaction feeds, regulatory
holdings snapshots. Anonymization is nearly always implemented as a
*mask*: the true participant code is replaced by an opaque token. The
researcher's implicit assumption is that the mapping is **stable** — one
entity, one token, forever. Nothing in the data advertises whether that
is true. The token *looks* the same on every row it appears; its
reappearance across months looks like entity persistence; its
disappearance looks like exit.

The failure mode is silent because every downstream statistic still
computes. Persistence rates, survival curves, entry cohorts,
concentration dynamics, "smart money" tracking — all produce plausible
numbers from re-minted masks. The numbers describe the masking schedule,
not the market.

### 2. The protocol

#### Step 1 — Profile the identifier formats

Map every character: letters→`A`, digits→`9`. Tabulate the resulting
shapes by row count and by time. In our case study, what a sample
suggested was one uniform 17-character scheme (`F` + 10 digits + YYYYMM)
turned out to be **six coexisting families** — 13-, 14-, 17- and
19-character forms plus two null conventions (a literal `"(null)"`
string and true nulls) — and the *mixture changed over time*: an early
era (~2011–2015) of long IDs with an embedded report-month suffix
(match rate ~0.91 against the file's report date), and a late era
(~2021–2025) of short IDs with no suffix. Format profiling alone
already yields two warnings: embedded dates inside "identifiers" are a
mask fingerprint, and a scheme change mid-sample means no single
identity rule can span the panel.

#### Step 2 — Understand why retention rates cannot answer the question

The natural test — what fraction of month-$m$ entities reappear in month
$m{+}1$? — is uninformative on its own, because low retention conflates
two hypotheses that demand opposite research designs:

- $H_A$: the entity did not trade next month (real dynamics);
- $H_B$: the entity traded, under a freshly minted mask (artifact).

Our measured retentions (0.24–0.39 across levels and eras) are exactly
the kind of "plausible" number that gets reported as institutional
turnover. They are unidentifiable as stated.

#### Step 3 — The decisive test: months-per-identity, against a control of known persistence

Two ingredients separate $H_A$ from $H_B$:

**(i) The statistic.** For each distinct ID, count the number of distinct
calendar months in which it ever appears, over the full span. Under
stable masks, the distribution's upper tail is populated by persistent
entities; under re-minting, the distribution is truncated at the minting
period regardless of true persistence.

**(ii) The control population.** Choose a sub-population whose *true*
persistence is externally known to be high and whose *true* cardinality
is known to be small. In brokered markets this is the broker/member
field: a near-fixed set of a few hundred real firms, active every month
by construction. Stable masks ⇒ broker IDs span essentially the whole
sample; re-minting ⇒ the broker field fragments like everything else.

Case-study result:

| Era (span) | Level | distinct IDs | median months | max months | ≥12 months |
|---|---|---|---|---|---|
| Early (84 mo) | FII | 95,987 | 2 | 10 | **0.0%** |
| Early | **Broker (control)** | **12,700** | 2 | **10** | **0.0%** |
| Late (98 mo) | FII | 250,949 | 1 | 2 | **0.0%** |
| Late | **Broker (control)** | **9,221** | 2 | **2** | **0.0%** |

A ~300-firm real population appearing as 9,000–13,000 identities, with a
hard ceiling of 10 (early) and 2 (late) months per identity, admits one
explanation: **the masks are re-minted on a short cycle**. The
truncation ceiling itself estimates the minting period. Note the late
era is *more* aggressively masked — anonymization regimes tighten, and
a panel that was usable one way in 2013 may not be in 2023.

### 3. What to do when the audit fails

A failed audit does not necessarily kill a research design; it *scopes*
it. The rules we derived:

1. **Cross-period entity statistics are off the table.** No persistence,
   no survival, no cross-month portfolio tracking, full stop.
2. **Within-mint-period identity survives.** If per-period ID counts are
   sane against known population sizes (ours: ~150 brokers, ~3,000 FIIs
   per month — both plausible), IDs are internally consistent inside the
   period. Same-day and same-month structure — participation counts,
   within-day book concentration, participation-weighted projections —
   remain measurable.
3. **Smoothing across days is legal; linking across days is not.** A
   trailing average of daily *measurements* (e.g., a 5-day mean of daily
   concentration snapshots) requires no cross-day identity and can
   restore the temporal coherence that single-day snapshots lack.
4. **Track missingness by period and weight by value.** Our ID
   missingness rose from 0% (2011) to 39% (2025) — any level-based use
   of the entity field is confounded late-sample. Value-weighted
   coverage with an explicit floor (we used 50%) converts attribution
   decay into honest missing data instead of quiet mismeasurement.
5. **Re-run the audit per era** whenever format profiling shows a scheme
   change; every rule above is era-specific.

### 4. Why this matters beyond one dataset

The conditions producing this failure are generic: privacy regulation
pushes data providers toward aggressive masking; masking schedules are
undocumented (documenting them would weaken the anonymization); and
every statistic a researcher computes from re-minted masks still
*runs*. We suspect published results already exist whose
"entity dynamics" are masking schedules. The audit is cheap — three
queries, minutes of compute — and its control-population design gives it
power that no amount of inspection of the focal population can: you
cannot tell re-minting from turnover by staring at the entities you
don't know; you can by staring at the ones you do.

### 5. Relation to the parent project

In the parent study, the failed audit forced the central measurement
(participant concentration) to be redesigned as a within-day,
participation-weighted, coverage-gated projection — and that redesigned
measure went on to separate transitory from permanent institutional
price impact out of sample. The audit did not merely prevent an error;
it produced the constraint under which the eventual contribution was
built. Full construction: `../ADOPTION_RECIPE.md`; validation:
`FII_thesis.md` Parts II–IV.
