# 04 · Methodology

## Design

A methods paper with one deep demonstration. Three legs:

1. **The protocol** (`01_PROTOCOL.md`) — nine questions, fixed order, each with
   a void scope, stated normatively and impersonally.
2. **The register** (`03_FAILURE_MODES.md`) — 38 failure modes, each a property
   of the evaluation design, each with a measured magnitude.
3. **The demonstrations** — the three places where the claim is proved rather
   than argued:
   - **D1 · Test power.** Synthetic PIT samples with known ground truth; a
     resampled-observed null rejects none of three degenerate cases, a
     null-centred construction rejects all three at 5.0% size. Establishes Q5.
   - **D2 · Evaluation-set look-ahead.** Identical forecasts, two
     stratifications, breach-rate spread 0.30 versus −0.04. Establishes Q3.5.
   - **D3 · Truncation audit.** Whole-pipeline re-execution in a truncated
     world; exact zeros with availability equality across 12 risk and 24 scoring
     columns. Establishes Q3.2–Q3.3.

D1–D3 are what make the paper falsifiable. Everything else is argument.

## The demonstration engine

| | |
|---|---|
| Object | Non-parametric flow-conditioned predictive density; VaR, ES, CRPS, PIT |
| Benchmark | Closed-form EWMA-Normal, identical volatility model, identical evaluation dates |
| Scored | 578,481 stock-days, 562 names with ≥250 days, 2016-02-02 → 2025-03-28 |
| Horizons | 1, 5 served; 20 retired on measured failure |
| Design | Walk-forward, 106 immutable monthly parameter vintages, `s + h ≤ asof` embargo |
| Rebuild | `./run_all.sh` in `~/Desktop/fii_risk_engine`, ~7 minutes from the sealed vintages |

The engine is used as an instrument, not as a result. Its substantive finding
(fat tails defeat the Gaussian; the conditioning mechanism contributes nothing)
appears because the protocol produces it, not because it is the paper's claim.

## Evidence status

| Item | Status |
|---|---|
| Protocol, 9 questions, ~50 checks | **Drafted** |
| Failure-mode register, 38 modes with magnitudes | **Drafted** from `LIMITATIONS.md`, `AUDIT_RETRACTIONS.md`, `C7_RISK_REPORT.md` |
| Test catalogue, 50 instruments | **Drafted**; 5-item build list |
| D1 numbers | Exist (`LIMITATIONS.md` R2-1). **Script must be rewritten standalone and model-agnostic** |
| D2 numbers | Exist (`panel_inference.expanding_quintiles`, s11 output). **Needs a minimal standalone reproduction** |
| D3 | Working (`s13_lookahead_audit.py`, ~91 s). **Needs generalising past this pipeline's stage names** |
| Direction classification (FLATTER/INVERT/PENALISE/VOID) | **Assigned by reasoning, not measured.** Defensible for most, but the counts should be presented as a classification, not a statistic |

## Work required before any prose

In order:

1. **Extract D1 as a standalone, model-agnostic harness.** Input: a test
   statistic, a sampler for H₀, a set of labelled alternatives. Output: size and
   power. This is simultaneously the paper's strongest exhibit and the library's
   first module — build it once.
2. **Extract D2 on the smallest slice that reproduces 0.30 vs −0.04.** Ideally
   on simulated data as well, so the effect is shown to be a property of the
   stratification rather than of this panel.
3. **Generalise D3's interface.** The truncation harness currently names this
   pipeline's stages. A model-agnostic version takes a callable chain and a
   truncation date.
4. **Re-verify every magnitude in the register against a fresh run.** The
   register is assembled from documents. Before publication each number must be
   traced to a current output file, because the engine was re-run after the
   evidence floors and volatility correction were changed and some figures in
   older documents predate that.
5. **Decide the ablation's presentation** (see `05_OPEN_QUESTIONS.md` Q2). F6.1
   is the register's largest single finding and it is a *negative* result about
   the demonstration engine. How prominently it sits determines what kind of
   paper this is.

## Reproduction contract

Every number claimed regenerates from a command:

| Artifact | Command | Time |
|---|---|---|
| Engine chain s06→s13 | `./run_all.sh` (`~/Desktop/fii_risk_engine`) | ~7 min |
| Look-ahead audit alone | `python -m s13_lookahead_audit` | ~91 s |
| Full research object, 62 stages | `python pipeline.py --all` (`~/Desktop/Major Project 2`) | ~8 min |

Both source trees are read-only for this project.

## Explicitly not done

- Data preparation, identifier resolution and corporate-action handling are
  premises, not content. Stated once in the scope paragraph.
- No catalogue of implementation defects, and no discussion of why errors escape
  notice. Every mode in the register is reproducible by correct code following
  standard practice.
- No attempt to rescue any withdrawn substantive finding. The engine is an
  instrument.
