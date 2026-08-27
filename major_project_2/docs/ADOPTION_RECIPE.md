# Computing Participant-Composition Concentration on Your Own Data
### A one-page implementation recipe

*People adopt methods that are easier to implement than to reinvent. This
page is the complete recipe for the participation-concentration measure
validated in this repository, portable to any transaction-level dataset
with (even masked) participant identifiers. Reference implementation:
`src/fii/features/module1_feature_store_v2.py`.*

## What you need

Transaction records with: date, instrument, participant ID (masking is
fine — see step 0), side (buy/sell), and traded value. Daily granularity
or finer.

## Step 0 — Audit the identifiers first (do not skip)

Masked IDs often *look* persistent and are not. Before anything else run
the three-part audit (full protocol: `docs/paper/identifier_audit_note.md`):

1. **Profile ID formats** (map letters→A, digits→9): mixed families and
   embedded dates reveal masking schemes and scheme *changes*.
2. **Do not trust retention** (month-to-month reappearance): it conflates
   "didn't trade" with "was re-masked."
3. **The decisive test:** months-per-distinct-ID over the full span,
   run on a **control population whose true persistence is known**
   (brokers, members, custodians — any near-fixed real-world set). If the
   control fragments, your IDs are re-minted, and every cross-period
   entity statistic you compute will be fiction.

The audit's verdict scopes everything below: with monthly re-minting,
concentration is a **within-day (at most within-month)** measurement.

## Step 1 — Entity-day book concentration

For entity $e$ on day $d$, with $v(e,d,s)$ = $e$'s sell value in
instrument $s$:

$$HHI(e,d) = \sum_s \left(\frac{v(e,d,s)}{\sum_{s'} v(e,d,s')}\right)^2$$

= 1 if the entity sold one name; → 0 as its selling disperses. Compute
buy-side separately.

## Step 2 — Project onto the instrument-day (participation-weighted)

$$C(s,d) = \frac{\sum_e v(e,d,s)\cdot HHI(e,d)}{\sum_e v(e,d,s)}$$

High $C$: today's sellers of $s$ are *focused* on it. Low $C$: $s$ is
incidental to broad selling programs.

## Step 3 — Three guards, all load-bearing

- **Coverage gate:** null the measure whenever attributable value < 50%
  of the day's total. Missing attribution must produce missing data,
  never mismeasurement. Check coverage *value-weighted* — row-level ID
  missingness overstates the problem.
- **Liquidity floor:** require ≥5 trades; concentration from two prints
  is noise.
- **Leakage rule:** any smoothing/trailing statistic ends *yesterday*
  (`shift(1)`); mask known structural breaks before windows touch them.

## Step 4 — Smooth, then normalize

- **Smooth** the daily snapshots with a ~5-day trailing mean (this
  requires *no* cross-day entity identity — it averages measurements,
  not entities). Raw single-day concentration has lag-1 autocorrelation
  ~0.33 and is invisible to regime models; smoothed ~0.77.
- **Normalize** within each day across instruments:
  $F = \Phi^{-1}\big(\text{rank}/(n+1)\big)$. This makes the measure
  invariant to participation growth and reporting-regime changes, and
  gives exact N(0,1) marginals.

## Step 5 — Use it

The validated regularity (Indian equities, 2011–2025, out-of-sample):
during **persistent selling**, top-quartile concentration
($F > q_{75}$) marks *transitory* impact — volume-marked pressure that
reverts ~50 bp over 20 days after the flow stops; bottom-quartile
($F < q_{25}$) marks *permanent*, quiet, information-consistent decline.
Derive the quantile cuts on a training era and freeze them. The signal's
information is **tail-concentrated**: expect it in episode extremes, not
as a broad cross-sectional alpha (ΔIC ≈ +0.006 when added to standard
flow features; the extremes carry ~70 bp/20d spreads).

## Pitfalls observed the hard way

Unadjusted corporate actions fabricate returns 100× the effect size —
verify your price adjustment against the tape before any event study.
Identity churn (ISIN/ticker changes) strands history exactly at the
corporate events that matter. Test your benchmark: "significant vs zero"
is meaningless under a drifting baseline; difference against the labeled
universe. Full failure catalogue: `docs/paper/FII_thesis.md` §21.
