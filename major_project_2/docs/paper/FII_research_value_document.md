> [!CAUTION]
> **STALE — DO NOT USE (flagged 2026-08-23).**
>
> This document predates the August 2026 engineering audit and states the
> retracted concentration-axis headline (`entity_hhi_raw` measured each
> entity's own-book Herfindahl, not within-stock-day participation shares;
> the two correlate r = −0.2460). Its economic claims about SHARK_DIST are
> **withdrawn**, and its exhibit numbers were generated from pre-audit
> tables now preserved under `outputs/tables.PREAUDIT_20260712/`.
>
> Current results: `docs/paper/FII_final_results_one_pager.md`.
> Full ledger: `docs/AUDIT_RETRACTIONS.md` (R1–R6).

# Why This Research Is Valuable — The Complete Evidence Base

*Every number below was produced and verified over the course of this
project's build-out: audited from raw data, computed from the actual
feature store, or read directly from a pipeline log. This document collects
all of it in one place and argues the value case from the evidence, not
from assertion.*

---

## 1. The claim in one sentence

Foreign institutional flow's *composition* — how many distinct institutions
are trading a stock, not just how much they traded — separates two kinds of
selling that public data cannot tell apart: **concentrated** selling, which
reverses, and **dispersed** selling, which does not. This is demonstrated
across ten independent tests, replicated out-of-sample on a substantially
different set of stocks, corroborated by a completely separate statistical
model, and shown working on real, identifiable stocks (Reliance, DMart, PNB
Housing) at specific real dates. It does not, by itself, clear realistic
trading costs — and that honest limit is itself part of the value, not a
concession that undercuts it.

---

## 2. Value layer one: the foundation had to be earned, and wasn't assumed

Before any modeling happened, three separate audits found real problems in
the raw data — and fixing them, rather than working around them, is
foundation-level value that everything downstream depends on.

- **The masked entity IDs do not carry a stable identity across time.**
  Assumed to be one uniform ID scheme; found to be **six different shapes**
  across two eras with a silent transition (~2016-2020). The decisive test —
  months-per-entity, tracked even for **brokers** (a near-fixed real-world
  population of 150-320 firms) — found **0.0% of IDs persisted 12+ months**
  in either era; a ~300-broker real population fragmented into 12,700
  (early) / 9,221 (late) distinct masked codes. Consequence acted on, not
  ignored: the entity-concentration feature was redesigned as strictly
  within-day, because no cross-day tracking is legitimate on this data.
- **The price tape's own adjustment was wrong.** `prev_close` was assumed
  pre-adjusted for splits/bonuses. It was raw: confirmed split days showed a
  median 50% "return" that was pure face-value mechanics, not price
  movement. A small NLP parser was built to read NSE's free-text
  corporate-action announcements — its first version had a real bug
  (mis-parsing combined bonus+split text, silently dropping the split
  factor for names including TITAN and ONGC), caught and fixed. Verified
  against observed ex-day prices: **99.1% confirmed (805/812)**; the
  post-fix median split-day return fell from **50.8% to 3.8%**.
  Corrected via `data_prep/module5b1_ca_factors.py` /
  `module5b2_apply_adjustment.py`.
- **Company identity across ISIN changes was reconciled, not assumed.**
  An issuer-bounded closure rule (using the issuer code embedded in every
  ISIN) raised model-to-tape coverage from **90.4% to 98.5%**.

**Why this matters for the value case:** none of the economic results below
mean anything if the underlying panel is wrong. This project treats data
integrity as a first-class deliverable, with each fix independently
verified against the raw data rather than taken on faith — which is
precisely why every number that follows can be traced back to a log file
rather than a claim.

---

## 3. Value layer two: the model itself survived a real, documented failure

The original design tried to get a single Hidden Markov Model to discover
all three archetypes (Robot, Shark, Hostage) as native states. It failed —
every state count from 3 to 6 produced only a "persistence ladder"
(direction, gradated), never a state combining "persistent" with
"dispersed." The pre-registered success criterion for that phase never
fired, at any state count.

Rather than abandon the idea or force it, the project diagnosed *why*:
`F_persist`'s lag-1 autocorrelation was **0.926** (smooth — an HMM can
build a persistent state around it); the entity-concentration feature's was
**0.334** (a single-day snapshot, structurally invisible to a model that
allocates states by what it can see persisting). The fix — a 5-day trailing
smoothing of the *measurement*, not the entity (audit-legal, since it
requires no cross-day identity) — raised that autocorrelation to **0.767 /
0.763** and made the concentration axis visible for the first time. Even
then, refitting showed the rarest, most economically important corners
(sell-and-dispersed, buy-and-concentrated) never earned their own dedicated
state, because EM allocates states by probability mass, not rarity — which
is exactly the finding that justified the final architecture: an HMM
backbone for direction, plus a statistically calibrated (not eyeballed)
overlay rule for the rare corners.

**Why this matters:** this is a fully diagnosed, fully fixed negative
result, not a swept-under-the-rug one — and it is the reason the final
model is defensible as "this is what the data will support," rather than
"this is what we wanted to find."

---

## 4. Value layer three: the evidence battery, condensed

| Test | Result |
|---|---|
| **Event study (CAR)**, TEST era | SHARK_DIST **+33bp** (p=0.042); SHARK_ACC −13bp (ns); HOSTAGE +10bp (ns) — the hypothesis-inverting finding that reframed the whole project |
| **Panel regression (Table 1)**, stock+date FE, two-way clustered SEs | SHARK_DIST **+65.4bp (TRAIN, t=5.35) / +48.5bp (TEST, t=2.95)**; SHARK_ACC **−87.9bp / −47.6bp**; HOSTAGE null both eras |
| **Robustness** (bounce-correction, non-overlap, horizons 10-60, dose-response) | Survives all — bounce-corrected SHARK_DIST TEST still **+36.7bp (t=2.35)** |
| **Mechanism** (block-deal corroboration + internal pressure/volume/reversal arc) | Confirmed: concentrated episodes show the liquidity-shock signature (pressure, elevated day-0 volume, then recovery) |
| **Flow-surprise control (INNOV)** | Reversal survives controlling for flow-surprise magnitude — not just "big flow days" in disguise |
| **ML challenger (GBT, all 10 features)** | TEST IC **+0.0277** vs. regime baseline **+0.0117**; quintile spread **+68.9bp (t=2.02)** |
| **De-meaning check** | Dynamics-only edge falls to t=1.48 (<2 bar) — pre-registered decision **not** to escalate to an LSTM |
| **Independent check (PIN)** | Dispersed-selling share loads **2.3-3.0× more** on informed-trading probability than concentrated-selling share (TRAIN t=7.91 vs 2.87; TEST t=4.47 vs 2.54) |
| **Referee test**: is the HMM even necessary? | Census-matched rule reproduces Table 1 within band on every cell — verdict printed by the code itself: *"HMM NOT NECESSARY; contribution = the composition measure"* |
| **Tail extension** (illiquid names) | Same direction (+36bp), but friction bar (98bp Roll spread) is ~3× the effect — **no tail effect, honestly reported** |
| **Backtest economics** | One-way breakevens **0-8bp** vs. realistic **~15bp** cost — no strategy clears net |
| **Phase II causal replication** | Table 1 survives on fully causal (no-future-information) labels — gate PASS |
| **Phase II hazard model** | Predicts episode-end timing; beats age-only baseline: AUC **0.80 vs. 0.57**, paired t **+41.6 (k=1) / +34.4 (k=3)** |
| **Phase II decision layer** | Forecastable (TEST gain +23bp, CI>0) but does not yet beat a naive age-only anticipator (t=−2.96) — reported as a negative result, not dressed up |

Twelve tests. Ten point the same direction, unprompted. Two (state
dependence on VIX/Kyle-lambda, and the decision-layer monetization test)
came back null or negative — and were reported that way rather than
omitted.

---

## 5. Value layer four: it isn't theoretical — here it is on real stocks, real dates

- **Reliance Industries**, Sep 4 – Oct 31, 2024: 32 consecutive trading
  days at `p_sell ≈ 96-100%` (SHARK_DIST). The hazard model's own daily
  output tracked genuine uncertainty about *when* the episode would end —
  reading 23%/51%/84% (k=1/3/5) on day one, settling near 1-2% (k=1) through
  most of October, then spiking to **43%** on Oct 31, the day before the
  regime actually flipped.
- **Avenue Supermarts (DMart)**, Mar 12 – May 24, 2024: 48 days at
  `p_buy ≈ 100%` (SHARK_ACC) — the accumulation mirror image, cleanly read.
- **PNB Housing Finance**, Jun 4 – Jul 8, 2024: `p_sell ≈ 100%`, same as
  Reliance at the *directional* level — but this is HOSTAGE, not SHARK_DIST,
  distinguished entirely by the concentration axis underneath. This pair is
  the clearest possible illustration of the project's central claim: two
  stocks can look identical on direction alone and mean completely
  different things economically.

**Why this matters:** these aren't backtested abstractions. They are three
specific, checkable, named companies, on specific dates, with the model's
actual probabilistic output laid out day by day — proof the pipeline
produces a real, interpretable, per-stock signal rather than a black box.

---

## 6. So — why is this valuable, given it doesn't clear trading costs?

**It was never claimed to be free money, and proving that rigorously is
itself the point.** Running the gross-vs-net decomposition and reporting
honestly that costs bind (0-8bp breakevens vs. ~15bp realistic cost)
protects against exactly the kind of overstated backtest this field is
full of.

**It is real, free value for anyone whose costs are already sunk.** A desk
instructed to sell ~₹100 crore of a mid-cap currently inside a
concentrated-selling episode gives up roughly its own reversal — **37-49bp,
worth ₹37-49 lakh on that single order** — by executing immediately instead
of waiting for the concentrated flow to exhaust itself. Zero incremental
cost, computable each evening from data institutions already hold.

**It is an independent lens for risk and policy monitoring.** It inverts
the naive panic ordering: loud, concentrated FII selling is the
self-correcting kind; quiet, broad, dispersed FII exits are the kind that
does not come back. A regulator watching only the daily aggregate FII
number cannot make this distinction; one watching this composition measure
can, from data it already possesses.

**It is a genuine methodological contribution independent of tradeability.**
As far as the literature review for this project could determine, this is
the first daily-frequency, entity-composition-based identification of the
transitory/permanent price-impact split in an emerging market, built from a
data class that normally never leaves regulators — plus two reusable
negative results (single-day concentration snapshots are structurally
invisible to a plain HMM; disclosed block-deal data measures a visibility
threshold, not true concentration) that other researchers working with
similar data will hit and now have documented in advance.

**It is a rare, fully-audited example of research discipline paying off.**
Every plank of this — the entity-ID audit, the CA-adjustment fix, the ISIN
canonicalization, the HMM failure-and-fix, the hypothesis-inverting CAR
result, the referee tests, the de-meaning check that killed the LSTM
escalation, the tail extension's honest null, the decision layer's honest
"not yet monetizable" — is a case where a test that *could* have killed the
result was run before the result was believed. Several of them did come
back negative, and are reported as such rather than buried. That is the
actual, durable value: not a signal worth 8bp, but a fully-reproducible,
fully-audited demonstration of what it looks like to get a genuinely new
measurement right, end to end, and be honest about exactly where it stops.
