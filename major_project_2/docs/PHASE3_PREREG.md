# Phase III — Pre-Registration

Written BEFORE any result was inspected. Two independent threads.
Bars are binding: a thread that misses its bar is reported as a negative
result, not re-specified.

---

## Thread A — Module 18: DISPERSED_ACC (the missing 2x2 cell)

### Motivation

The archetype overlay is asymmetric. Frozen thresholds match the train-era
quantiles of the concentration feature *within each directional state*:

| cut         | value  | equals                          |
|-------------|--------|---------------------------------|
> **Thresholds superseded (2026-08-23).** These were derived on the book-focus feature. On the corrected participation measure the vintage medians are hostage −0.1546, shark_dist +0.8124, dispersed_acc −0.1233, shark_acc +0.8540. The two measures are not commensurable, so the old and new cut-points cannot be compared.

| HOSTAGE     | -0.513 | SELL `F_entity_s` **q25**       |
| SHARK_DIST  | +0.877 | SELL `F_entity_s` **q75**       |
| SHARK_ACC   | +0.795 | BUY  `F_entity_buy_s` **q75**   |
| *(absent)*  | -0.593 | BUY  `F_entity_buy_s` **q25**   |

The sell side used both tails; the buy side used only the upper tail. The
missing cut is implied by the same frozen rule — no new discretion.

The unlabelled population is large: UNTAGGED_DIRECTIONAL = 287,014 stock-days,
of which 168,598 (59%) are BUY_REGIME. It currently sits inside the **omitted
category** of Table 1, so if it carries directional drift every published
coefficient is measured against a contaminated reference.

The paper's central claim is stated direction-agnostically ("the concentration
axis separates transitory from permanent FII price impact") but was tested on
three of four cells.

### Method (frozen, no new discretion)

- Threshold = TRAIN-era q25 of `F_entity_buy_s` within Viterbi BUY_REGIME,
  derived in-script by the same procedure that produced the published cuts.
- `DISPERSED_ACC = BUY_REGIME & F_entity_buy_s < q25_train`.
- Episodes, controls, and estimator identical to Module 7: post-END 20d
  market-adjusted CAR (bp), PanelOLS with stock + date FE, SE two-way
  clustered (stock x month), UNTAGGED omitted, specs R0/R1/R2.

### Pre-registered bars

- **P1 (primary — permanence).** Framework predicts dispersed buying is
  PERMANENT, mirroring HOSTAGE: `D_DISPERSED_ACC` shows |t| < 2 in BOTH eras
  AND |coef| < 25 bp. -> "2x2 COMPLETED, symmetric".
- **P2 (horizons).** No monotone build across CAR10/20/30/60 (the transitory
  signature). Flat, like HOSTAGE.
- **P3 (contamination of published results).** In the 5-archetype spec,
  `D_SHARK_DIST` and `D_SHARK_ACC` keep sign, keep |t| >= 2 in both eras, and
  stay within +/-30% of the published 4-archetype values.
- **P4 (census).** DISPERSED_ACC share is of comparable magnitude to the other
  active archetypes (~5-9% of stock-days).

### Failure conditions (reported, not re-specified)

- `D_DISPERSED_ACC` significant with |coef| > 40 bp and REPLICATING across
  eras -> dispersed buying is not permanent; the concentration ->
  transitory/permanent mapping is **sell-side specific** and the paper's
  central claim must be narrowed to the sell side.
- P3 breached -> published Table 1 has a contaminated reference group and the
  headline numbers must be restated, not merely supplemented.

### Explicitly disclaimed priors

Earlier speculation that this cell "should be weakest" (short-sale
constraints, thinner buy-side entity attribution) is UNTESTED opinion. It does
not gate the test and is not a prediction of record.

---

## Thread B — Module 19 gate: TVTP viability

> [!CAUTION]
> **WITHDRAWN (2026-08-23) — audit item 3.** This thread was pre-registered to
> address a regime-persistence deficit. That deficit was an artifact.
>
> The transition matrix was estimated with one transition per *observed row*
> (hmmlearn has no notion of elapsed time) and then deployed in C2 as though it
> were a *per-trading-day* matrix, propagating belief with `A^k`. Those are
> different units. Measured on the existing state labels:
>
> | transitions used | mean diag | λ₂ | half-life |
> |---|---:|---:|---:|
> | pooled (what the fit saw) | 0.9491 | 0.9416 | 11.5 d |
> | k = 1 only (a true daily A) | 0.9648 | 0.9665 | 20.4 d |
> | **gap-aware estimator (current)** | **0.9796** | **0.9793** | **33.8 d** |
>
> A 77% error in mixing half-life, concentrated in thin stocks. Under the
> corrected estimator the chain is *more* persistent than even the k=1
> benchmark, so the motivating deficit does not exist and the macro channel
> this thread proposed is not needed to explain it.
>
> Thread B is closed as **void on its premise** — not as a negative result.
> Re-opening it requires a persistence deficit re-established against the
> gap-aware fit.

### Motivation

Macro cannot enter this model as a level: date fixed effects absorb every
market-wide variable (Module 10 found exactly this), and within-day
cross-sectional ranking holds the archetype census constant by construction.

The one admissible channel is the **transition matrix** — macro sets the
persistence of flow regimes, not their cross-sectional targets. The static
fitted matrix has lambda2 = 0.9235, a mixing half-life of **8.7 days**, which
is why h=20 forward state prediction retains little and h=60 retains nothing.

Firewall (binding): emissions stay 100% flow. Under joint Baum-Welch the
E-step posterior would depend on macro and leak into mu/Sigma, so **emissions
are FROZEN** at Phase-I values and only transition parameters are estimated.
This keeps state definitions numerically identical to what Modules 5-13
validated and leaves the overlay thresholds valid.

### Gate G1 (run BEFORE building anything)

Split trading days into India-VIX terciles. Compute the empirical transition
matrix of the existing state sequence within each tercile.

- **PASS**: SELL-state self-transition rate is monotone in VIX tercile AND
  (high - low) >= 0.01 in BOTH eras, permutation p < 0.05.
- **FAIL**: diagonals flat -> macro does not modulate persistence in this
  data; TVTP thread CLOSES and is reported as a negative result.

Rationale for the bar: a 20-day half-life (the working hypothesis) requires
lambda2 ~ 0.966 vs 0.9235 today. A detectable tercile spread is a necessary
condition; failing it means the softmax model has nothing to fit.

### Known power constraint

`z_t` is identical across all ~235 stocks on a given day, so the effective
sample for macro coefficients is the number of DAYS (~2,400 train), not
stock-days — and fewer independent macro episodes than that. Any inference
uses date-clustered SE or a time-block bootstrap. Module 10's VIX interaction
already failed to replicate (+83*** TEST, -27 ns TRAIN), which is the relevant
warning shot.

Pilot uses India VIX only (already on disk). DXY / US10Y / Brent / USDINR
collection is deferred until G1 passes.

### Downstream (only if G1 passes)

TVTP -> state-dependent mixing -> h-step predictive state distribution ->
outcome mixture density scored by **CRPS / PIT** against climatology and a
rolling-vol Gaussian, both eras, time-block bootstrap. Discrete state
posteriors continue to be scored by log/Brier (Module 16B machinery).

---

## Thread B — AMENDMENT 1 (2026-08-03): macro vector slate

### Why an amendment rather than a quiet re-run

G1 tested India VIX and failed. That is retained as a result, not withdrawn.
This amendment tests a DIFFERENT hypothesis, motivated a priori: an FII
prices risk in USD, so its hurdle rate is driven by US rates, Treasury
volatility and FX, none of which India VIX contains. The theory predicted
India VIX would be the wrong instrument BEFORE that result was seen, which is
what distinguishes this from variable-hunting.

### Power is established FIRST (this is what makes a null interpretable)

Per-vector simulation on real panel geometry, real tercile series, permutation
null at date level. Minimum detectable effect at 80% power, alpha=0.05:

| vector   | TRAIN p@0.01 | TEST p@0.01 | note                          |
|----------|--------------|-------------|-------------------------------|
| move     | 1.00         | 1.00        |                               |
| d_us10y  | 1.00         | 1.00        |                               |
| dxy_mom  | 1.00         | 1.00        |                               |
| inr_mom  | 1.00         | 1.00        |                               |
| spread   | 1.00         | 0.99        | TEST terciles skewed 57/274/529 |

Benchmark: doubling mean dwell on the empirical Viterbi self-transition scale
(base 0.966) requires delta ~ 0.017. All five vectors resolve this at
power ~1.00. Therefore a null on any vector is EVIDENCE OF ABSENCE, not
absence of evidence.

NOTE (scale correction, on record): an earlier statement that the hypothesis
"needs delta ~ 0.04" conflated the fitted transmat diagonal (0.938, lambda2
0.9235, half-life 8.7d) with the empirical Viterbi self-transition (0.966).
These are different scales. On the measurement scale used by the gate, the
correct benchmark is ~0.017.

### The slate (locked; additions require a further amendment)

1. `move`     — MOVE index, US Treasury volatility (the liquidity-shock channel)
2. `d_us10y`  — 20d change in US 10Y yield (the hurdle-rate channel)
3. `dxy_mom`  — 20d DXY momentum (the dollar channel)
4. `inr_mom`  — 20d USD/INR momentum (the realised FX-penalty channel)
5. `spread`   — US 10Y minus India 10Y (FRED INDIRLTLT01STM, MONTHLY,
                ffilled; the structural cost-of-capital channel)

All rolling-standardised over 250 trading days (causal), lagged one day beyond
the timezone argument (US close on Indian date t is known before India's open
on t+1; we lag a further day regardless).

EXCLUDED: `curve` (10Y-3M) — lag-1 autocorr 0.991, n_eff = 14, and 0.384
correlated with d_us10y.
NOT TESTED: INR/USD forward premiums — unobtainable, and equal to the rate
differential under covered interest parity, hence not an independent channel.

### Bar (identical to G1, Bonferroni-corrected)

PASS requires, for a given vector:
  - SELL self-transition monotone in tercile, AND
  - |hi - lo| >= 0.01, AND
  - permutation p < 0.01  (alpha 0.05 / 5 vectors), in BOTH eras.

Any vector passing -> proceed to fit a full TVTP on THAT VECTOR ALONE
(6 transition parameters vs n_eff 55-96; a multi-vector softmax at 18
parameters is NOT supported by this sample and will not be fitted).

All five vectors are reported regardless of outcome. No vector is dropped
post hoc.

### Failure condition

All five fail -> reported as a positive finding: FII regime persistence is not
macro-modulated in this data at a magnitude this design can detect, which
supports the reading that flow topology is autonomous rather than macro-driven.
This closes the TVTP thread permanently rather than provisionally.

---

## AMENDMENT 2 — C6 evaluation strata, and a data-coverage disclosure

Filed after C2 (daily filter) and BEFORE any outcome density (C4), predictive
mixture (C5), or scoring (C6) has been computed. No forward return has been
touched at the time of filing. The stratification below is therefore a
pre-specification, not a post-hoc slice.

### Motivating measurement (C2 output, 595,736 stock-days)

The filtered posterior P(S_t | x_{1:t}) is far more peaked than the Phase III
design assumed:

| statistic                        | value            |
|----------------------------------|------------------|
| median p_max                     | 0.993            |
| share p_max > 0.99               | 55.6%            |
| share p_max > 0.90               | 82.3%            |
| share p_max < 0.70               | 6.47% (38,542)   |
| mean entropy                     | 0.146 of 1.099 nats |

Cause: four well-separated emission features (min adjacent F_persist gap 1.043)
combined with self-transitions above 0.93. A single day's observation nearly
determines the state.

### Consequence for the central hypothesis

The Phase III claim under test is that SOFT probability labels beat HARD labels
inside a risk engine. On 93.5% of stock-days the soft label IS the hard label to
within 0.3, so the predictive mixture is arithmetically near-identical to a
hard-label lookup there. A full-panel test would dilute any real effect across a
majority of rows on which the two methods CANNOT differ. A null from such a test
would mean "no room to differ", not "no effect" — an uninterpretable result of
exactly the kind this pre-registration exists to prevent.

### Locked strata for C6

PRIMARY   — uncertain days, p_max < 0.70, n = 38,542.
            The regime-transition population; the only rows where the soft/hard
            distinction is arithmetically capable of mattering.
SECONDARY — full panel, n = 595,736. Reported always, for dilution context.
TERTIARY  — p_max in [0.70, 0.90], n ~ 35,600, as a monotonicity check: any
            genuine effect should be intermediate between the two above.

The 0.70 cut is fixed now and will not be tuned. All three strata are reported
regardless of outcome. A PASS on PRIMARY with a null on SECONDARY is a valid
and expected result and will be reported as such, with the dilution arithmetic
shown explicitly.

### Standing limitation (not a gate, a disclosure)

The state posterior itself cannot be calibrated directly: the regime is latent
and there is no ground-truth state to score against. Agreement with Phase-I
Viterbi labels (91.12%) is an agreement statistic between two estimators, not a
calibration measurement. Calibration becomes measurable only at C6, against
OBSERVED returns, via CRPS and PIT. No claim of "calibrated regime
probabilities" will be made anywhere on the basis of C2 alone.

### Data-coverage disclosure (found during C2 pre-flight)

states_v3.parquet is missing five calendar months. Two distinct causes:

  - 2021-05, 2021-06 — absent from the raw FII source (2021.parquet has no
    rows). Deliberately masked at module1_feature_store_v2.py:34,62 so that
    rolling features step over the window. Documented in code, correct.

  - 2023-06, 2023-09, 2023-11 — PRESENT and healthy in the raw source
    (231,625 / 283,834 / 240,283 trades; valid ISIN, RATE, QUANTITY,
    VALUE_INR, RFDE_INSTR_TYPE) but TR_TYPE, the buy/sell direction flag,
    is 100% NULL in all three. The feature store filters on
    TR_TYPE.is_in([1,4]), so every row is dropped before any downstream
    stage sees it. This is a CSV-to-parquet parse failure, not missing data.

Extent, measured across all 169 months (no other month affected):
    755,742 trades = 3.00% of all trades
    INR 13.53 trillion = 2.99% of all traded value
    Entirely inside the TEST era.

Recoverability: the source CSVs (2023_06.csv etc., named in `source_file`) are
no longer on disk; only the year parquets remain. No other column encodes
direction — RFDE_RPT_TYPE is a single constant, and RFDE_STLD_CODE,
RFDE_REASON_DLAY and RFDE_AMDMNT_REASON have indistinguishable distributions in
affected and unaffected months. Direction is therefore unrecoverable locally.

Decision: the three months remain EXCLUDED. Imputing trade direction would
fabricate the dependent construct of the entire model. The exclusion is correct;
what was wrong was that it happened silently. It is recorded here and must
appear in the thesis data section. If the original source is re-obtained, the
fix is a re-parse and a full rerun of Phase I onward.

Downstream effect already handled: C2 propagates with A^k over elapsed TRADING
days, so the resulting month-long holes decay belief toward the stationary
distribution instead of carrying stale state across them. Parameter age reaches
91 days at the 2021-04-30 -> 2021-07-30 vintage boundary for the same reason and
is carried as a column (`param_age_days`) in the C2 output.

---

## AMENDMENT 3 — primary stratum moves from p_max to pa_max

Filed after C3 (soft-threshold archetype overlay) and BEFORE C4, C5 or C6.
No outcome density, no forward return, and no predictive score has been
computed at the time of filing. This amendment is motivated by the STRUCTURE
of the estimator, not by any result.

### Change

PRIMARY stratum for C6 becomes  pa_max < 0.70,  n = 45,296  (7.60%).
Supersedes the  p_max < 0.70,  n = 38,542  stratum locked in Amendment 2.

The p_max stratum is RETAINED and reported in full as a co-stratum. Nothing
is dropped; the reporting set only grows.

### Justification

The risk engine consumes ARCHETYPES, not states. The soft-versus-hard question
is therefore decided on archetype uncertainty, and pa_max is the quantity that
measures it. p_max measures uncertainty about an intermediate latent variable.

C3 showed the two are not the same. Archetype uncertainty has two sources:

  - regime ambiguity: 38,542 rows with p_max < 0.70. All of these are
    necessarily archetype-uncertain — state uncertainty propagates through the
    overlay.
  - boundary proximity: 5,499 rows that are state-CERTAIN (p_max > 0.90) but
    archetype-uncertain (pa_max < 0.70). The regime is known; the day sits near
    a concentration threshold, so the archetype is genuinely ambiguous.

The second group is invisible to p_max and is exactly the information hard
labelling destroys. Excluding it would have tested the hypothesis on a
population chosen by the wrong criterion.

### Locked strata for C6 (superseding Amendment 2)

PRIMARY   — pa_max < 0.70,          n = 45,296   (7.60%)
CO        — p_max  < 0.70,          n = 38,542   (6.47%)
SECONDARY — full panel,             n = 595,736
TERTIARY  — pa_max in [0.70, 0.90], as a monotonicity check

The 0.70 cut is unchanged and will not be tuned. All strata are reported
regardless of outcome.

### Why this is not post-hoc

The reasoning above uses only (i) which variable the downstream engine
consumes, and (ii) the joint distribution of p_max and pa_max — both known
without reference to any outcome. C4 has not been run. If a reader judges the
second amendment in a single session to weaken the pre-registration, the
Amendment 2 stratum stands unaltered in the record above and its result will be
reported alongside, so the comparison can be made directly.

---

## AMENDMENT 3a — restatement of stratum sizes after a calendar defect

Filed immediately after Amendment 3, still BEFORE C5/C6. Corrects arithmetic,
not policy. The p_max/pa_max cut of 0.70 and the primary/co/secondary structure
of Amendment 3 are UNCHANGED.

### The defect

C4 pre-flight found that the Phase III trading calendar had been built as the
UNION of price dates and FII dates. The custodian feed carries rows stamped on
non-trading days — 8,256 on Saturdays and Sundays plus 910 on market holidays,
97% of them in 2016-17, 9,166 rows in all (1.14% of states_v3). Admitting those
dates as calendar columns gave them no prices, which invalidated EVERY
multi-day outcome window spanning one. Whole months carried zero usable 20-day
outcomes (2016-03, 2016-04, 2017-02, 2017-03, 2017-04).

Fix: the calendar is now defined by prices alone (dates on which at least 50
of the modelled stocks have a finite adjusted return). Rows on non-trading
dates are dropped at the head of C2, so C3 and C4 inherit a calendar on which
every day prices. A vintage asof landing on a non-trading day (1 of 106) is
snapped back to the last trading day at or before it, which can only tighten
the embargo.

Effect: h=20 outcome coverage rose from 77.6% to 98.0% of scorable rows, and
the C4 warm-up frontier fell from 11 vintages to 2.

### Restated sizes (superseding the counts quoted in Amendment 3)

| stratum                     | Amendment 3 | restated |
|-----------------------------|-------------|----------|
| PRIMARY   pa_max < 0.70     | 45,296      | 44,795   |
| CO        p_max  < 0.70     | 38,542      | 38,148   |
| SECONDARY full panel        | 595,736     | 587,128  |
| state-certain but arch-uncertain | 5,499  | 5,395    |

The qualitative claim motivating Amendment 3 is unaffected: 5,395 rows remain
state-certain (p_max > 0.90) yet archetype-uncertain (pa_max < 0.70), and
pa_max still identifies a 17.4% larger primary population than p_max.

### Additional exclusion now on record

474 stock-day returns (0.0200% of the modelled price panel) carry
|ret_adj| > 0.5 or are non-finite and are excluded from all outcome
computation, with invalidity propagated to every forward window containing
one. These are corporate-action adjustment failures, not market moves —
e.g. CHEMPLASTS 2021-08-24, close 535.6 against prev_close 541.0, a -1.0% day
recorded as +3507%. Retaining them would let six data errors dominate the tail
estimates that C5 and C6 exist to measure.

### C4 warm-up frontier (locked)

Scoring may not begin before the vintage at which every archetype clears an
effective sample size of 100 and never falls back below it:

    h = 1   vintage 1  (2016-02-29)
    h = 5   vintage 1  (2016-02-29)
    h = 20  vintage 2  (2016-03-31)

Earlier vintages are unusable by construction and are excluded from C6 rather
than scored with a thin density.

---

## AMENDMENT 3b — restated counts after re-estimating the vintages

Filed after C1 was rebuilt from a reconstructed, version-controlled harness and
C2-C7 rerun. This restates ARITHMETIC only. Every rule locked in Amendments 2,
3 and 3a is unchanged: the primary stratum is still pa_max < 0.70, the cut is
still 0.70, and it was not tuned.

### Why the counts moved

The original harness ran from a scratch directory that was cleared between
sessions; only its output survived. It could not be recovered from that output
(see the provenance note in module_c1_refit_harness.py), so the vintages were
re-estimated with a reconstructed harness whose data selection reproduces the
original EXACTLY - window (asof - 1826 days, asof], >= 60 observations per
stock, each stock capped at its last 400 - but whose restart scheme differs.
The reconstruction finds a higher-likelihood optimum in 68 of 106 vintages
(median +2,644 log-likelihood). Different optima give slightly different
thresholds, which shift pa_max, which shifts the stratum sizes.

The original output is preserved as outputs/phase3/vintages_original.parquet.

### Restated sizes

| stratum                          | Amendment 3a | restated |
|----------------------------------|--------------|----------|
| PRIMARY   pa_max < 0.70          | 44,795       | 39,932   |
| CO        p_max  < 0.70          | 38,148       | 38,355   |
| TERTIARY  pa_max in [0.70, 0.90] | 72,876       | 67,332   |
| SECONDARY full panel             | 587,128      | 587,128  |
| state-certain but arch-uncertain | 5,395        | 1,283    |

### One motivating figure weakened, and it is reported as such

Amendment 3 moved the primary stratum from p_max to pa_max because 5,395 rows
were state-certain (p_max > 0.90) yet archetype-uncertain (pa_max < 0.70) -
days near a concentration threshold that p_max cannot see. Under the
re-estimated vintages that group is 1,283 rows, not 5,395.

The qualitative argument stands: the risk engine consumes archetypes, so
archetype uncertainty remains the right criterion, and a group of 1,283 rows
invisible to p_max still exists. But it is roughly a quarter of the size that
motivated the amendment, and the primary stratum is now SMALLER than the
p_max co-stratum (39,932 against 38,355) rather than 17% larger. Readers should
weigh the amendment accordingly. Both strata are reported in full, as they
have been since Amendment 3.

### Threshold shifts driving the change

| threshold        | original | new     | shift   | shift / bootstrap SE |
|------------------|----------|---------|---------|----------------------|
| th_hostage       | -0.5100  | -0.5740 | -0.0639 | -2.33                |
| th_shark_dist    | +0.8853  | +0.9180 | +0.0327 | +1.28                |
| th_dispersed_acc | -0.5526  | -0.6401 | -0.0875 | -3.01                |
| th_shark_acc     | +0.7864  | +0.7949 | +0.0085 | +0.32                |

Two of four shifts exceed two bootstrap standard errors. This is a genuine
finding about the estimator rather than a nuisance: the archetype cut-points
are NOT invariant to which EM optimum the fit lands in, even though the
emission means move by less than 0.12 and the self-transitions are unchanged
to three decimals. It belongs in the thesis as a stated limitation.

### Effect on the pre-registered conclusions

The three headline C6 findings survive re-estimation:

  - conditioning adds nothing: soft_roll vs clim_roll remains null on the
    PRIMARY stratum at every horizon (p 0.12, 0.14, 0.86)
  - the apparatus beats an EWMA-Normal at h=1: p < 0.0001, unchanged
  - soft vs hard is real but negligible: still 1 of 9 primary tests clears
    Bonferroni (h=20, p=0.0005), and the h=1 sign now flips to slightly
    negative (-0.000014, p=0.55), which strengthens rather than weakens the
    reading that the effect is economically nil

C7's comparative results are materially unchanged (ES coverage 1.018 against
the Gaussian's 1.237 at h=1; per-stock Kupiec pass rates 91.6% against 60.1%).
