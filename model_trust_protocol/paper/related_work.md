# Related work

**Verified citations only.** Every entry below was checked against the
published record. See the quarantine at the foot of this file.

---

## The replication tradition

**Hou, K., Xue, C. & Zhang, L. (2020). Replicating Anomalies. *Review of
Financial Studies* 33(5), 2019–2133.** Re-tests 452 published anomalies.
With microcaps mitigated via NYSE breakpoints and value-weighted returns, 65%
fail to clear |t| ≥ 1.96; at a multiple-testing hurdle of 2.78 the failure rate
is 82.1%; in the trading-frictions category, 96% fail. *Our relation:* they
establish that most published predictors do not clear the statistical hurdle.
Every application in this paper does clear it. We study the population at which
their filter stops.

**Harvey, C. R. (2017). The Scientific Outlook in Financial Economics.
*Journal of Finance* 72(4)** (presidential address). Argues that incentives for
significant results, unreported tests and multiple testing make widespread
non-replication predictable, and that conventional thresholds are inadequate.
*Our relation:* the problem statement our design responds to.

## Friction-first evaluation

**Muravyev, D., Pearson, N. D. & Pollet, J. M. (2025). Anomalies and Their
Short-Sale Costs. *Journal of Finance* 80(6), 3639–3694.** Across 162
anomalies, the average long-short portfolio return is +0.14% per month before
short-sale costs and −0.01% after borrow fees. Anomalies are unprofitable even
before fees once the highest-fee observations — 12% of stock-dates — are
excluded. *Our relation:* the closest design in spirit, and the sharpest
contrast in shape. They price **one** friction across **many** predictors; we
price **six**, ordered, across **one** dataset. Their exclusion result is also
an average/extreme dissociation, which is one of the mechanisms we document.

## Institutional flow and its effects

**Lakonishok, J., Shleifer, A. & Vishny, R. W. (1992). The impact of
institutional trading on stock prices. *Journal of Financial Economics* 32(1).**
Finds no evidence of substantial herding or positive-feedback trading by pension
fund managers except among small stocks, and no strong cross-sectional relation
between changes in institutional holdings and abnormal returns. *Our relation:*
an early demonstration that the absence of a predicted institutional-flow effect
is itself reportable evidence.

---

## The positioning, in one table

| | Predictors | Frictions priced | Shape |
|---|---:|---:|---|
| Hou, Xue & Zhang (2020) | 452 | 1 | breadth |
| Muravyev, Pearson & Pollet (2025) | 162 | 1 | breadth |
| **This paper** | 1 dataset, 7 applications | **6, ordered** | **depth** |

Breadth and depth answer different questions. Breadth establishes how often a
friction is decisive across a population of predictors. Depth establishes
*which* friction binds for a given use — a question breadth cannot address,
because it holds the friction fixed.

---

## To verify before drafting

- Chordia, Goyal & Saretto, *Anomalies and False Rejections* (RFS 2020) —
  likely the intended reference for a mis-attributed citation encountered
  during literature assembly. Not yet checked.
- International/out-of-sample anomaly replication (Australian and other
  non-US samples) — a claimed failure to replicate Heston & Sadka (2008) and
  Keloharju et al. (2016) has not been traced to a specific paper.
- Finance-specific reproducibility surveys.

## Quarantine — do not cite

The following were supplied during literature assembly and **could not be
found in any search of the published record.** They are recorded here so they
cannot drift back into the manuscript.

| Claimed citation | Status |
|---|---|
| Yang Lu (2026), *Null Results Under Pre-specified Retail Implementation Regimes* | **No trace.** Searched by exact title and by author plus keywords. Almost certainly fabricated |
| "WASTE — Economics, Econometrics & Finance failure-mode index", cataloguing 199+ negative results | **No trace.** No such resource surfaced |
| "Chordia, Goyal, Shanken & Shivakumar" on predictive-return replication | **Probable conflation** of distinct author groups |

Both untraceable items were described in terms that fitted this project's
situation unusually closely. That is characteristic of fabricated references,
and is the reason every citation above was checked individually.
