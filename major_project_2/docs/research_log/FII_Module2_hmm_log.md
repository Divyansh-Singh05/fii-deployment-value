# FII Regime Detection — Module 2 Log: HMM Fitting, the Hostage Test & the Hybrid Architecture

> [!NOTE]
> **Historical record — read with `docs/AUDIT_RETRACTIONS.md`.** This log is a
> dated account of what was believed while the work was being done, and is
> deliberately **not** rewritten. Several conclusions below were overturned by
> the 2026-08 engineering audit; in particular the concentration feature these
> entries reason about measured entity book focus, not participation
> concentration (r = −0.2460 between the two).


*Companion to `FII_Module1_findings.md` and `FII_stockday_data_dictionary.md`. Covers 2026-07-08 → 2026-07-09.*

## 0. Objective & inputs

Fit a Gaussian HMM on the validated feature store to decode each stock-day into the three archetypes — **Robot** (transient herd flow), **Shark** (informed concentrated positioning), **Hostage** (forced dispersed fire-sale) — and *earn* the labels via signature checks rather than assume them.

- **Input:** `stockday_features_v2.parquet` (Module 1; 10 probit features, ~N(0,1), leakage-safe).
- **HMM feature subset:** `F_persist`, `F_block`, `F_entity`, `F_entity_buy` — one direct signature per archetype. Excluded: `F_activity` (r=0.88 with block), `F_streak` (r=0.56 with persist), rest held in reserve.
- **Model class:** GaussianHMM, diagonal covariance (rare-state estimability), Viterbi decode, multiple EM inits, post-hoc labeling from state feature-means.

**Pre-registered success criterion (set before fitting):** the Hostage exists as a regime iff some state shows `F_persist clearly < 0` **and** `F_entity clearly < 0` (persistent + dispersed selling). Pre-registering this prevented rationalising whatever clusters appeared.

---

## 1. Phase 0 — Pre-fit triage of three critiques

Before fitting, three externally-raised risks were assessed:

1. **Axis-3 coverage cliff.** VALID — and it exposed that our earlier claim "value-weighted coverage = 1.0 every year" was an artifact of the NaN-contamination bug (pre-`fill_nan` diagnostics counted NaN as present). Corrected numbers (from the cliff check, below): usable `F_entity` falls from 0.96 (2011) to **0.53 (2025)**; mass in the 0.4–0.6 near-cliff band rises from ~0 to **0.13**. Consequence adopted: 2024–25 state *frequencies* are coverage-confounded — a standing caveat for any backtest.
2. **Observation-weighting bias.** VALID. Within-day ranking fixes cross-sectional *position*, not row *counts* — hyperactive large caps contribute a valid row nearly every day and would dominate EM. Mitigation adopted: **fit-cap** — each stock contributes at most its most recent 400 complete rows to the *fitting* sample; the fitted model then decodes **all** full sequences. (Fit balanced, decode everything.)
3. **Liquidity-floor survivorship** ("liquidation happens when liquidity evaporates"). PARTIALLY VALID with a saving nuance: `N` counts *FII trades* and fire-sales are FII-activity events; moreover the entity book-HHI is computed over the entity's whole day-book, so an entity dumping 40 names contributes a *dispersed* signal to every name it touches, even at 1 trade in that name. Kept `N≥5` for Module 2; queued "re-examine Hostage-adjacent days at N≥2" as a Module-3 robustness item.

---

## 2. Phase 1 — First 3-state fit (script: *MODULE 2 · 3-STATE GAUSSIAN HMM*)

**What the code did and why:**
- **Cliff check** (first block): honest re-measurement of Axis-3 usability by year — replacing the invalidated 1.0 claim with real numbers.
- **Sequence construction:** complete cases on the 4 features → per-stock, time-ordered sequences; stocks with <60 complete days dropped (too short to carry regime structure). Universe: **776,068 stock-days, 984 stocks**.
- **Fit:** cap 400 rows/stock → fitting sample **278,462 rows**; 5 random EM inits, keep best log-likelihood. Two inits converged to the same optimum (−1,424,681) — reassuring stability.
- **Post-hoc labeling rule:** Robot = state with |mean persist| smallest; of the rest, Hostage = lower sell-HHI. **With built-in contradiction warnings** — the script refuses to bless labels whose signatures contradict the archetypes.

**Results & interpretation:**

| state (labeled) | F_persist | F_block | F_entity | F_entity_buy | verdict |
|---|---|---|---|---|---|
| "HOSTAGE" | **+1.167** | 0.034 | −0.222 | 0.048 | ⚠ mislabeled — this is persistent **buying** |
| ROBOT | 0.006 | 0.062 | 0.074 | 0.081 | clean Robot (all ≈ 0) |
| "SHARK" | **−1.140** | −0.014 | +0.172 | −0.191 | ⚠ mislabeled — persistent **selling**, *concentrated* sellers |

- Both warnings fired → labels rejected. The model had found a **directional split** (buy / neutral / sell), not the archetype split: persistence means ±1.15 dwarf entity means ±0.2, so likelihood was dominated by one axis.
- Transitions: dwell ≈ 11d (neutral), 13.5d, 14.1d — persistent regimes, plausible time scales.
- Occupancy 43/28/28; **year-shares nearly constant across 15 years** — recognised as *by construction*: within-day ranking removes the market-wide component, so "everyone is a Hostage" market days cannot exist in this feature space. Market-wide episodes must appear in validation (Module 3/4), not in state prevalence.
- Two economically real reads extracted from the "failure":
  (a) the persistent-sell state has **concentrated** sellers → it likely *mixes* Hostages (dispersed) with Sharks distributing (concentrated);
  (b) **dispersed selling co-occurs with persistent buying** (−0.22 in the buy state) — broad-book entities are the liquidity *providers* during accumulation. Dispersion alone ≠ fire-sale; the Hostage requires dispersion **and** net-sell pressure jointly.

---

## 3. Phase 2 — Dissection + state-count sweep (script: *MODULE 2b*)

**Why:** two hypotheses from Phase 1 needed evidence: is the sell state a mixture (bimodal in `F_entity`)? and does a higher k give the Hostage its own state (the deferred BIC question)?

**Results:**
- **Dissection of the persistent-sell state (220,481 days):** `F_entity` quantiles q5 = −1.51 … q95 = +1.74; **25.8% dispersed (<−0.5), 37.4% mid, 36.8% concentrated (>+0.5)**. The mixture material exists — plenty of dispersed-sell days — the model just wasn't separating them.
- **BIC sweep k=2..6:** monotone decline (big-n artifact); Δ(k=5→6) ≈ 330 on a 2.7M scale → true elbow **k=5**. But the k=5/6 solutions are a **persistence ladder** (−1.52, −0.66, ~0, ~0, +0.66, +1.54) with two 1-day transient noise states — again gradations of one axis. **No state with persist<0 AND entity<0 at any k.**

---

## 4. Phase 3 — Mechanism diagnosis & the smoothing fix (script: *MODULE 2c*)

**Hypothesis:** an HMM state must be *temporally coherent*; EM carves states along smooth features. We fed it one ultra-smooth feature and three jittery ones.

**Test — median per-stock lag-1 autocorrelation:**

| feature | lag-1 AC | note |
|---|---|---|
| `F_persist` | **0.926** | 20d rolling mean — smooth by construction |
| `F_block` | 0.290 | same-day surprise |
| `F_entity` | 0.334 | single-day snapshot (audit-forced) |
| `F_entity_buy` | 0.295 | single-day snapshot |

**Mechanism confirmed.** The audit's single-day constraint didn't just weaken Axis 3 — it made it *structurally invisible to an HMM*.

**Fix:** `F_entity_s`, `F_entity_buy_s` = **5-day trailing rolling mean of the daily HHI snapshots**, re-ranked within day, probit. Audit-legal: this averages *stock-level dispersion measurements* across days — it requires **no** cross-day entity identity. Result: lag-1 AC → **0.767 / 0.763**. The entity axis became HMM-visible.

**Refit sweep (k=3..6) on `[F_persist, F_block, F_entity_s, F_entity_buy_s]`:**
- k=3: directional triad again (sell state entity_s **+0.24** — still concentrated).
- **k=4 — the most informative fit of the module:** kept the two directional poles and split the *neutral* middle by **participant structure**: a concentrated-both-sides state (entity_s +0.56, entity_buy_s +0.78, dwell 14.0d) vs a dispersed-both-sides state (−0.45, −0.67, dwell 9.5d). → **The regime space is 2-D: flow direction × participant concentration.** The occupied cells: sell/concentrated, neutral/concentrated, neutral/dispersed, buy/mildly-dispersed. The **missing corners are exactly sell/dispersed (Hostage) and buy/concentrated (accumulating Shark)** — rare corners never earn a state because EM allocates states by probability mass.
- k=5: persistence ladder returns; still no Hostage corner.

**Module-2 headline finding (pre-registered criterion fired at every k, both raw and smoothed):**
> **Persistent selling in Indian FII flows is concentrated-seller selling. The Coval-Stafford Hostage (persistent + dispersed selling) does not exist as a *temporal regime* at the stock-day level — Hostage days exist (~26% of persistent-sell days) but are episodic, not regime-forming.**

---

## 5. Phase 4 — Final hybrid architecture (script: *MODULE 2 · FINAL*)

**Design (the pre-registered fork):** HMM for what it *can* see — temporal flow regimes — plus **overlay rules** on the entity coordinates for what is episodic:

| Label | Rule (TH = 0.5 probit ≈ 69th pct) |
|---|---|
| HOSTAGE | SELL_REGIME ∧ `F_entity_s < −0.5` |
| SHARK_DIST | SELL_REGIME ∧ `F_entity_s > +0.5` |
| SHARK_ACC | BUY_REGIME ∧ `F_entity_buy_s > +0.5` |
| ROBOT | NEUTRAL state |
| UNTAGGED_DIRECTIONAL | directional regime, unremarkable entity structure |

Backbone = best k=3 model (SELL / NEUTRAL / BUY). Economically this is arguably *truer* than the original design: fire sales are **episodes within selling regimes**, not regimes themselves.

**Census (776,068 decoded stock-days):** ROBOT 42.4% | SHARK_DIST 11.4% | SHARK_ACC 9.7% | **HOSTAGE 6.9%** | UNTAGGED 29.6%. The Hostage is the rarest label — matching the framework's "fragile, rare state" prediction *empirically* rather than by assumption. The 30% UNTAGGED is the honest "directional but unremarkable" bucket — no force-fitting.

**Episode-structure test (the make-or-break for the overlay):** 55,530 Hostage days form **15,495 episodes** → mean 3.6d, median 2, p90 8, **max 53**. Null model: within sell-regimes Hostage days are 23.7% (55,530/234,501); i.i.d. tagging at that rate would give mean run **1.31d**, median 1. Observed clustering ≈ **2.7× the noise baseline** → Hostage days chain into genuine multi-day episodes. PASS.

**By-year shares:** Hostage drifts 7.7% (2011) → 5.3% (2025). NOT interpreted — the decline tracks the ID-missingness rise (coverage confound, §1.1). Standing caveat.

---

## 6. Phase 5 — Face validity: the longest Hostage episodes vs known events

ISINs mapped via NSE `EQUITY_L.csv`:

| len | Company | Window | Known event |
|---|---|---|---|
| 53d | **JSW Steel** | 2013-05-29 → 08-13 | **Taper tantrum** — starts 7 days after Bernanke's May 22 speech; INR collapse, FII metals exodus |
| 42d | **Vedanta** | 2015-10 → 12 | **Global commodity crash** / Glencore panic |
| 40d | **HDIL** | 2014-09 → 11 | Leveraged-realty distress (later PMC-fraud infamous; now NSE series BZ) |
| 39d | **Reliance Communications** | 2016-07 → 09 | Debt spiral into **Jio launch** (Sep 2016); later insolvent; now series BE |
| 38d | **Vedanta (Sesa)** | 2013-01 → 03 | Iron-ore mining bans + CAD crisis |
| 28d | (delisted ISIN) | 2016-11-04 → 12-13 | Begins **4 days before demonetization** |
| 27d | **Ambuja Cements** | 2016-12 → 2017-01 | **Demonetization** aftermath — cement most cash-exposed |
| 27d | **Embassy Dev. (ex-Indiabulls RE)** | 2014-09 → 11 | Same window as HDIL — sector-wide realty fire-sale caught in two names independently |

Reads:
1. Episodes land on the canonical macro stress windows **despite** within-day ranking removing the market-wide mean → these are the names sold hardest *relative to an already-stressed market* — exactly Coval-Stafford's prediction that fire sales concentrate in commonly-held names.
2. **Two of the top four names subsequently went bankrupt** (RCom, HDIL) — the label finds real distress.
3. Even the unmapped ISINs are absent from the current NSE list (delisted) — misses consistent with the story. **Face validity: PASS.**

---

## 7. Module 2 status & what's next

**Done:** directional HMM backbone (k=3, dwell 13–17d) + archetype overlays; Hostage validated by episode-clustering (2.7× noise) and face validity (taper tantrum / commodity crash / demonetization / pre-bankruptcy names). Output: `stockday_states_final.parquet`.

**Methodological principles that carried the module:** pre-registered success criteria; labeling code that *warns on contradiction* instead of blessing clusters; fit balanced / decode everything; diagnose mechanism (autocorrelation) before re-engineering; audit-compliant smoothing; accept the pre-registered fork (hybrid) when the regime hypothesis failed.

**Open items → Module 3/4:**
- **Forward-return validation** (Shark-acc drifts, Hostage reverses, Robot transient) — needs external EOD prices (NSE bhavcopy); not yet sourced.
- **INNOV validation** — build `FII_NET_INNOV` the AAK way (look-ahead allowed, validation-only) and test that states sort it.
- Robustness: overlay threshold sweep (TH 0.3/0.5/0.7); liquidity floor N≥2 sensitivity for Hostage-adjacent days; 2024–25 coverage confound handling.
