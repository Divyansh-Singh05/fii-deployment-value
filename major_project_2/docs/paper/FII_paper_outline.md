# PAPER OUTLINE — FII Flow Regimes and the Transitory/Permanent Decomposition of Institutional Price Impact

*Working scaffold. Maps the validated evidence (Modules 1–11, logs §3a–3n) into paper sections. Numbers are final results; prose to be written into each block. Two headline contributions flagged **[C1]/[C2]**.*

---

## Working title
"When Do Foreign Institutions Move Prices? Trade Concentration and the Transitory–Permanent Decomposition of FII Flow." (alt: "Concentrated vs Dispersed: Separating Liquidity from Information in Foreign Institutional Trading.")

## Abstract (draft skeleton)
- Object: daily masked NSDL FII trade records, ~940 liquid Indian stocks, 2011–2025.
- Method: unsupervised HMM regime detection on within-day cross-sectional flow features → directional backbone + concentration overlays (Robot / Shark-dist / Shark-acc / Hostage).

> [!CAUTION]
> **RETRACTION (2026-08-23) — the headline finding below is withdrawn.**
>
> The concentration feature the model was fitted on did not implement the
> formula this document states. `entity_hhi_raw` computed a
> participation-weighted mean of each *entity's own book* Herfindahl — how
> focused a seller is across stocks — not the Herfindahl of participation
> shares *within the stock-day*. The two measures correlate **r = −0.2460**
> on the rebuilt panel: they are close to opposites, not variants.
>
> Re-running the full Phase I battery on the corrected feature moves the
> result from one archetype to the other:
>
> | Test (published spec) | Published (book measure) | Corrected (participation measure) |
> |---|---|---|
> | Panel R2, TRAIN | SHARK_DIST **+54.0** (t=3.5) | SHARK_DIST +21.1 (t=1.37, ns) |
> | Panel R2, TEST | SHARK_DIST **+47.5** (t=2.9) | SHARK_DIST +18.2 (t=1.26, ns) |
> | Panel R2, TRAIN | HOSTAGE ≈ 0 (p≥0.62) | HOSTAGE **+78.6** (t=4.47) |
> | Panel R2, TEST | HOSTAGE ≈ 0 (p≥0.62) | HOSTAGE **+42.4** (t=2.73) |
> | Event study, END | SHARK_DIST +68 / +33 bp | SHARK_DIST −1 (p=0.94) / +13 (p=0.55) |
> | Block-deal rate | SHARK_DIST 0.73% < HOSTAGE 0.96% | SHARK_DIST **1.08%** > HOSTAGE 0.68% |
> | Day-0 volume climax | SHARK_DIST 1.12× / 1.13× | SHARK_DIST 1.01× / **0.93×** (gate needs >1) |
>
> The reversal is robust to ~11 alternative specifications — but it now sits
> on **dispersed** selling, which is the reading this project's own
> pre-registration started from and claimed to have overturned.
>
> **No replacement headline is asserted here, because the corrected evidence
> does not yet support one.** Two problems are open:
> 1. *Internal conflict.* The panel regression says dispersed selling
>    **reverts** (transitory). The PIN estimator still loads on dispersed
>    selling (`sh_host` 0.1334/0.1369 vs `sh_sd` 0.1007/0.0394 ns), which
>    reads as **informed** (permanent). Both cannot be right.
> 2. *The pre-registered skeptic gates now fail.* T1 — "the reversal is
>    materially bid-ask bounce" (it passed pre-audit with a reprice).
>    T2 — "SHARK_DIST and HOSTAGE are not distinguishable". T3b — "a free
>    public volume signal absorbs it".
>
> Until those are resolved, treat every concentration-based economic claim in
> this document as **unsupported**. The engineering audit and the risk engine
> are unaffected; see `docs/AUDIT_RETRACTIONS.md`.

- **[C1]** The **concentration** of FII flow (few vs many participating entities) separates **transitory** from **permanent** price impact: concentrated selling is followed by a +48–65 bp/20d reversal (concentrated buying mirrors, −), while dispersed selling produces *permanent* declines (no reversal). Survives OOS, full FE panel regression, and a battery of alternatives.
- **[C2]** Unexpected FII flow (AR innovation) reverts out-of-sample — a distinct transitory channel.
- Independent Easley–O'Hara PIN (built from counts, blind to returns) endorses the reading: dispersed selling carries ~3× the informed-trading loading of concentrated selling.
- Framing: **surprise reversal of the original hypothesis** — dispersed ("Hostage/fire-sale") flow behaves *informed*; concentrated ("Shark/informed") flow behaves *liquidity-demanding*. Honest and earned.

---

## 1. Introduction
- Motivation: FII flows and EM price impact; the informed-vs-liquidity question; why *concentration* (composition) is under-studied vs *magnitude*.
- Preview of [C1]/[C2] and the independent PIN endorsement.
- Contribution vs literature: Coval-Stafford (fire sales), Easley-O'Hara (PIN), Kyle (impact), FII-flow literature (AAK-style). Our twist: identify the transitory/permanent split from an *unsupervised flow taxonomy*, then validate economically.
- **Honest framing paragraph**: pre-registered hypotheses were refuted; the data delivered the inverse; every alternative was tested not argued. This *is* the paper's credibility.

## 2. Data (Modules 4–5)
- NSDL masked FII trades 2011–2025 (TR_TYPE buy/sell, RATE>0).
- NSE bhavcopy EOD prices (survivorship-free, incl. delisted).
- **Data-engineering appendix material** (shows rigor): corporate-action adjustment built from parsed split/bonus factors (99% tape-confirmed); issuer-code canonical ISIN closure (recovered +7.8% coverage, 90.4→98.5%); 2011 ISIN backfill validation. → these become a "Data construction and validation" appendix; cite the gates (Gate A ex-day median |ret| 0.508→0.037).
- Macro: NIFTY50, S&P500(lag), USDINR, India VIX.
- **Scope caveat (state early & often)**: model universe = 939 liquid large/mid-cap FII names (~25% of FII-traded canonical names). HOSTAGE fire-sale effects may be *stronger* in the excluded illiquid tail → results are a conservative floor for the permanent-decline names, and the reversal generalizes to liquid names.

## 3. The regime model (Modules 1–3)
- Feature store: 10 within-day cross-sectional probit-ranked flow features; leakage discipline (past-only windows, masked May–Jun 2021, no price/VIX in features).
- 3-state directional HMM backbone (SELL/NEUTRAL/BUY) + overlay rules for concentration archetypes (why hybrid: Module-2 finding that HMM can't natively form concentration states — persistence dominates).
- Threshold calibration with built-in falsification (the GMM that failed its own stability test → quantile rule; three-way convergence). **This is a methods-credibility exhibit.**
- Temporal OOS protocol: train ≤ Apr-2021, test ≥ Jul-2021, everything frozen. Signatures/census/transitions replicate near-exactly.
- **Exhibit**: archetype signature table + census + OOS replication.

## 4. Empirical strategy (Modules 5–7)
- Event-time abnormal returns (market-adjusted; baseline-relative to handle the +52 bp TEST small-cap drift).
- Anchors: START (drift), END (reversal — the correct anchor, flow has stopped).
- Panel regression: PanelOLS, stock + date FE, SE two-way clustered (stock×month), UNTAGGED omitted category. Specs R0/R1/R2 (R2 adds pre20 as a *conservative bad-control lower bound*).
- **Language discipline**: "predicts conditional on controls," never causal; labels estimated → attenuation toward zero (conservative).

## 5. Main result — the transitory/permanent decomposition **[C1]** (Modules 5B, 7, 7b)
- **Headline table** (paper Table 1) — R2 restored full sample (Module 7b-B):
  | Archetype | TRAIN | TEST |
  |---|---|---|
  | SHARK_DIST (concentrated sell) | +65.4*** | +48.6*** |
  | SHARK_ACC (concentrated buy) | −87.9*** | −47.6*** |
  | HOSTAGE (dispersed sell) | ≈0 (ns) | ≈0 (ns) |
- Interpretation: concentration → reversal (transitory); dispersion → permanent.
- Loser-reversal objection killed: SHARK_DIST survives pre20 control (pre20 itself significant → generic reversal exists, dummy barely moves).
- HOSTAGE flat-zero = the mechanism-relevant null (same direction, differs only in concentration).

## 6. Mechanism — liquidity shock, from internal data (Module 6B)
- Full event arc: pre-episode decline (~−270 bp backdrop) → concentrated climax with **elevated volume (1.12–1.13× own trailing avg)** → +80/+32 bp recovery after flow stops (~⅓ retrace).
- **Exhibit**: arc figure (pre20/day0/epCAR/post20 + relvol) per archetype; SHARK_ACC mirror; HOSTAGE flat + BELOW-baseline volume (quiet distribution).
- Why block-deal data FAILED and volume succeeded (Module 6): block reporting = visibility threshold (0.5%, all-participant), misses worked FII orders; volume is unmaskable. → one honest paragraph, turns a null into a methods point.

## 7. Robustness (Module 7b)
- Non-overlapping episodes: SHARK_DIST *strengthens* (+76/+108***).
- Horizons CAR10/20/30/60: reversal **builds and persists** (TRAIN +39→+65→+89→+109), not a momentum artifact. HOSTAGE ≈0 all horizons.
- Continuous dose-response (F_entity_s, no thresholds): right-signed all cells; TRAIN-SELL +27*** clean, TEST weaker → **tail phenomenon** (effect in extreme-concentration episodes; honest).
- ROBOT placebo caveat: contaminated at END anchor (its end = a directional transition; Module 7b-A explains mechanically) → HOSTAGE is the valid null.

## 8. Alternative explanations ruled out
- Past-loser reversal → pre20 control (§5).
- Market drift / size → date FE + baseline-relative + logto/Amihud/beta controls.
- Characteristics-in-disguise → within-stock demeaning (Module 8b): demeaned ICs ≈ raw.
- **Flow-surprise magnitude [key]** → NET_INNOV control (Module 9-T3): SHARK_DIST unchanged (+65.4 TRAIN to the decimal) with episode-mean INNOV in the model; concentration ≠ surprise magnitude. INNOV built with deliberate look-ahead (yardstick, conservative competitor) — footnote the rationale.
- State-dependence (VIX/λ, Module 10): mixed/null, **not claimed** — reported as one honest paragraph (TEST-only VIX interaction; λ amplification is a characteristic effect, not archetype-specific).

## 9. External validation — PIN endorses the reading **[C1 support]** (Module 11)
- FII-PIN (Easley–O'Hara MLE on buy/sell counts, blind to prices/returns).
- **Exhibit**: PIN ~ archetype shares. sh_host +0.231***/+0.149*** >> sh_sd +0.077***/+0.065** → dispersed selling ~3× informed loading. Independent construct endorses HOSTAGE=information/permanent.
- Nuance stated: concentrated selling carries *less* information, not *none* (relative claim).
- T3: reversal not concentrated in high-PIN names (liquidity, not info).
- Limitation: FII-slice PIN, not all-market.

## 10. The second regularity — flow-surprise reversion **[C2]** (Module 9-T2)
- INNOV forward IC −0.023/−0.017 (t −3.2/−2.6), both eras.
- **Decomposition result, not a tradable signal** (in-sample expectation model) — stated plainly.

## 11. Is the regime model the right tool? (Modules 8, 8b)
- GBT challenger on all 10 features vs regime baseline: modest edge, but the *dynamic* increment (demeaned) fails the pre-registered spread t>2 bar (t=1.48).
- Conclusion: HMM not the bottleneck; features are the ceiling at this power; regimes = the right dynamic description. Deeper/sequence models not justified.
- Future-work hook: within-stock breadth changes (FII participation surges) — top dynamic SHAP feature, unused by the HMM; first lever with more data (illiquid tail, post-2025).

## 12. Limitations (consolidate the running caveats)
- Liquid-universe scope (fire-sale tail excluded).
- Masked FII IDs → within-month concentration only, no cross-month entity tracking (Module-1 entity audit).
- No fundamentals (no value/size/sector factors; stock FE absorb permanent components only).
- FII-flow λ and FII-PIN are FII-slice, not total-market.
- 2024–25 coverage confound (ID missingness 30–40%).
- Non-causal throughout (predictive, conditional).

## 13. Conclusion
- Two regularities; the inverted-hypothesis narrative; concentration as the axis that separates liquidity from information in institutional flow.
- Methodological contribution: a pre-registered, gate-driven validation battery that killed two silent code bugs, one false data assumption, and two wrong economic narratives before publication.

---

## Exhibit list (build order)
1. T1 archetype signatures + OOS replication (§3).
2. **Table 1** main panel regression R0/R1/R2 both eras (§5). ← the paper's centerpiece.
3. Mechanism arc figure (§6).
4. Robustness panel: non-overlap / horizons / dose-response (§7).
5. Alternatives table incl. INNOV control (§8).
6. PIN ~ shares table (§9).
7. GBT decision box (§11).

## Open drafting decisions (need author calls)
- [ ] Journal target → sets length, notation, whether the data-construction saga is appendix or online-only.
- [ ] Lead with [C1] alone, or [C1]+[C2] as twin contributions? (Recommend [C1] lead, [C2] as a section.)
- [ ] How much of the "failures caught" methodology to foreground — it's a genuine credibility asset but some referees read it as over-sharing. (Recommend a tight methods box, full detail in appendix/replication log.)
- [ ] PIN robustness: add all-market PIN if BSE/NSE full order flow can be sourced? (currently FII-slice only — stated limitation.)
