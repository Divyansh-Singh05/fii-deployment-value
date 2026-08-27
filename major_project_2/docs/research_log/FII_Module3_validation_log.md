# FII Regime Detection — Module 3 Log: Out-of-Sample Validation, Threshold Calibration & Statistics

> [!NOTE]
> **Historical record — read with `docs/AUDIT_RETRACTIONS.md`.** This log is a
> dated account of what was believed while the work was being done, and is
> deliberately **not** rewritten. Several conclusions below were overturned by
> the 2026-08 engineering audit; in particular the concentration feature these
> entries reason about measured entity book focus, not participation
> concentration (r = −0.2460 between the two).


*Companion to `FII_Module2_hmm_log.md`. Covers 2026-07-09. Scripts: `module3a_model_split_oos.py`, `module3b_threshold_calibration.py` (+ quantile patch), `module3c_descriptive_stats.py`.*

## 0. What Module 3 had to establish

Module 2 ended with a hybrid architecture (directional HMM + archetype overlays) validated only **in-sample**. Module 3 had three jobs:
- **3A** — a genuine temporal train/test protocol (never done before this module; also fixing v1's recency-biased fit-cap).
- **3B** — replace the arbitrary ±0.5 overlay threshold with a statistically derived one, with falsification built in.
- **3C** — a formal statistics battery, effect-sizes-first (n ≈ 800k ⇒ p-values alone are decoration).

**Protocol:** train = ≤ 2021-04-30, test = ≥ 2021-07-01 (the masked May–Jun 2021 gap is the natural break). Everything — HMM parameters, state identities, thresholds — derived on TRAIN only, frozen, then applied to TEST.

---

## 1. 3A — The backbone replicates out-of-sample

Setup: 509,185 train stock-days (572 stocks) vs 295,773 test stock-days (830 stocks — a substantially different universe). Fit-cap fixed: **random contiguous block ≤400 days/stock** (v1 used most-recent-400, quietly biasing emissions toward late years).

| Check | TRAIN | TEST (frozen model) |
|---|---|---|
| SELL signature (persist / entity_s) | −1.04 / +0.18 | −1.07 / +0.30 |
| BUY signature (persist / entity_s) | +1.08 / −0.299 | +1.13 / −0.304 |
| Census (sell/neutral/buy) | 29.1 / 42.9 / 28.0 | 28.5 / 43.6 / 27.9 |
| Transition diagonal | 0.95 / 0.95 / 0.95 | 0.95 / 0.95 / 0.95 |

**Verdict: PASS.** Signatures, occupancy and transition structure replicate almost digit-for-digit on unseen years and a ~45% larger stock universe. The directional HMM is a stable structure, not curve-fit.

---

## 2. 3B — The GMM calibration FAILED its own falsification test (and that's the story)

**Design:** BIC-selected GMM (1/2/3 comp) on `F_entity_s` within the TRAIN sell regime; threshold = posterior-0.5 boundary of the dispersed component; k-means cross-check; stability check by re-deriving on TEST.

**What happened:**
- Train BIC picked k=2 — but the components were **49%/51% at means −0.50/+0.84**: a coin-flip split of the sell regime, not a rare-tail-vs-majority structure. Its boundary (**+0.155**) would have tagged mildly *concentrated* days as Hostage, doubled the Hostage census to 14%, and eliminated the "unremarkable middle" (everything in the sell regime became either HOSTAGE or SHARK_DIST).
- **Every independent check disagreed with it and agreed with each other:** k-means boundary **−0.441**; test-era GMM (which found a proper rare tail: weight 0.24, mean −0.84) boundary **−0.510**; stability Δ = 0.665 → tripped the pre-registered UNSTABLE flag.
- Fallback engaged per protocol: **quantile rule** (train q25 of sell-regime `F_entity_s`) = **−0.513**.

**Three-way convergence:** k-means −0.441 · test-GMM −0.510 · train-q25 **−0.513**. Three methods with different assumptions land within 0.07 — and on the original eyeballed −0.5, now *derived* rather than assumed.

**Final frozen thresholds (train-derived, applied to both eras):**
| Overlay | Rule | Stability (train vs test quantile) |
|---|---|---|
| HOSTAGE | `F_entity_s < −0.513` | −0.513 vs −0.331 (Δ 0.18, PASS; drift direction consistent with the 2024–25 coverage confound) |
| SHARK_DIST | `F_entity_s > +0.877` | +0.877 vs +0.966 (Δ 0.09) |
| SHARK_ACC | `F_entity_buy_s > +0.795` | +0.795 vs +0.798 (Δ 0.003) |

**Calibrated census:** ROBOT 42.9/43.6 · UNTAGGED 35.5/35.9 (middle restored) · SHARK_DIST 7.3/8.0 · SHARK_ACC 7.0/7.0 · HOSTAGE 7.3/5.6 (train/test %; Hostage test dip = coverage confound, not economics).

**Methodological note worth preserving:** the calibration had a falsification test built in; it fired; the correction came from three independent methods converging. This episode is the project's strongest credential — the machinery was designed to catch its own failure and did.

---

## 3. 3C — Final statistics battery (calibrated labels, per era)

**Caveat first:** in the HOSTAGE vs SHARK_DIST contrast, `F_entity_s` shows d≈−5.7, KS D=1.000 — **tautological** (the two labels are defined by a cut on that variable). The informative rows are all the *others*.

### Effect sizes (Cohen's d), TRAIN → TEST
| Contrast | F_persist | F_block | F_entity_s | F_entity_buy_s |
|---|---|---|---|---|
| HOSTAGE vs SHARK_DIST | +0.22 → +0.26 | −0.19 → −0.38 | (tautological) | −0.46 → −0.55 |
| SHARK_ACC vs ROBOT | **+3.19 → +3.20** | ~0 | ~0 | **+1.52 → +1.42** |
| HOSTAGE vs ROBOT | **−2.98 → −3.01** | −0.19 → −0.34 | **−1.40 → −1.43** | −0.74 → −0.71 |

Reads:
1. **Replication is near-exact across eras** — the non-tautological effect sizes move by ≤0.15 between disjoint time periods. The label structure generalises.
2. **Orthogonality confirmed:** within the sell regime, dispersed vs concentrated sellers are near-identical on persistence and blockiness — the entity axis carries genuinely independent information (as designed).
3. **A small, correctly-signed bonus:** HOSTAGE trades *smaller* than SHARK_DIST (block d −0.19 train, −0.38 test) — the original framework predicted fire-sale *fragmentation* vs Shark *blocks*; the sign shows up exactly as theory says, just as a small effect.
4. Blockiness is otherwise a weak separator everywhere (the "Sharks take size" intuition does not survive within-regime).

### Episode-clustering permutation test (200 within-stock shuffles)
| Era | Observed mean run | Null | Ratio | p |
|---|---|---|---|---|
| TRAIN | 3.64d | 1.47 ± 0.00d | **2.48×** | 0.005 |
| TEST | 3.39d | 1.42 ± 0.01d | **2.38×** | 0.005 |

Hostage days cluster into genuine multi-day episodes at ~2.4–2.5× the shuffled-null rate, **replicating out-of-sample**. (Also held at ~2.6× under the rejected +0.155 threshold — the phenomenon is robust to the cut.)

### Episode lengths (final labels)
| Archetype | median | p90 | max (train/test) |
|---|---|---|---|
| HOSTAGE | 2d | 8/7d | 53 / 31 |
| SHARK_ACC | 3d | 9/10d | 58 / 56 |
| SHARK_DIST | 3d | 10/10d | 59 / 61 |
| ROBOT | 11d | 39–43d | 854 / 287 |
| UNTAGGED | 4d | 14–15d | 106 / 129 |

Economically sensible: Robot dwells for weeks; the active archetypes come in ~3-day-median bursts with heavy tails; the 53d max Hostage episode is the JSW Steel taper-tantrum event (train era).

---

## 4. Standing limitations
- 2024–25 Hostage frequency is coverage-confounded (FII-ID missingness 30–40%); the train→test Hostage dip (7.3→5.6%) and the q25 drift (−0.513→−0.331) both carry this fingerprint. Do not interpret late-era Hostage *prevalence* economically.
- HOSTAGE-vs-SHARK_DIST separation on `F_entity_s` is by construction; their *economic* difference must be established by forward returns (Module 4).
- Labels remain statistically validated, not yet economically: the forward-return check (Shark-acc drifts, Hostage reverses, Robot transient) requires external EOD prices — not yet sourced. INNOV-based validation (AAK-style, look-ahead allowed, validation-only) can run on flow data alone.

## 5. Status
Module 3 **complete**: backbone OOS-validated, thresholds calibrated with documented convergent evidence, statistics battery on record for both eras. Current model output: `stockday_states_calibrated.parquet` (804,958 stock-days, era + state + archetype). Next: Module 4 — economic validation (price data decision pending).
