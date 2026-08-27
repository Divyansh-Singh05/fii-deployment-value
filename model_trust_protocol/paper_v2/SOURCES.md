# Which source is authoritative, and what is stale

Written 2026-08-26, before drafting paper_v2. Two manuscript versions exist and
NEITHER dominates. Build v2 by taking numbers from one and method/apparatus
from the other, per the table below.

## Rule of thumb

| Ask for | Go to |
|---|---|
| Any NUMBER | `submission/ijf/manuscript.md` |
| The name/definition of a test | `paper/04_results.md`, `paper/03_methodology.md` |
| Citations | `paper/related_work.md` (verified + quarantined) |
| What is still unfixed | `paper/06_limitations.md` 6.1 |
| Round-3 re-derivations | `paper/referee_response/RESULTS.log` |
| Round-4 re-derivations | `paper_v2/rederivations/RESULTS.log` |

## STALE — do not copy from `paper/*.md`

Round 3 (`paper/referee_response/`) corrected these; the corrections landed in
the IJF manuscript on 25 Aug 23:35 and were NEVER back-ported to `paper/*.md`
(frozen 24 Aug 23:59). All five are numbers the project's own scripts disproved.

| Location | Stale value | Correct value | Killed by |
|---|---|---|---|
| `00_abstract.md:34` | t-2 row `-0.32` | `-0.43` | exp5, exp7 |
| `00_abstract.md:37` | EWMA pair `-2.35` | `-2.22` | exp1 |
| `00_abstract.md:38` | GJR pair `-0.95` | `+0.28` | exp1 |
| `04_results.md:45` | `-0.32 / -0.30 / -0.13` | `-0.43 / -0.09 / -0.44` | exp5, exp7 |
| `04_results.md:81` | `-2.35 PASS / -0.95 FAIL` | `-2.22 pass / +0.28 fail` | exp1 |
| `03_methodology.md:31` | SEs "two-way clustered on instrument and calendar month" | clustered on DATE; two-way + bootstrap reported alongside | round-3 code read |

`04_results.md:130` (AUC 0.797, t=+41.6, +23bp) is NOT stale — it is correctly
quoted there AS the superseded pre-audit figure being corrected. Keep it.

Why -2.35/-0.95 is wrong and not merely old: that pair varied the baseline AND
the feature set (2 vs 5 features) AND the scored day count. It was never a
controlled comparison. exp1 rebuilt it on an identical tilt and identical 2,235
days: `-2.22` vs `+0.28`. The GJR arm changes SIGN, not just magnitude.

## STALE — research tree artifacts

Never read these; they predate the audit that rebuilt the state object.

- `outputs/tables.PREAUDIT_20260712/` (whole directory)
- `outputs/phase3/*.PREAUDIT.*` — c6_results, C7_RISK_REPORT,
  stock_risk_profiles, vintages
- `outputs/phase3/*.BASELINE.*` — c6_results, C7_RISK_REPORT, predictive
- `data/ISIN_MAPPING/*.PREAUDIT.parquet` — stockday_features_v2,
  stockday_states_calibrated, stockday_states_split
- `data/VALIDATION_DATA/*.PREAUDIT.parquet` — returns_panel_v3, states_v3

Current equivalents are the same paths WITHOUT the infix.

## STALE — code docstring, never cited

`src/fii/validation/module19_institutional_share.py:33` claims custodian
reporting is "1.5% of value same-day, 67% by T+1". Measured values are 0.13%
and 94.30% (exp4b). Both docstring figures are wrong and neither has a
producing stage. Not cited in either manuscript; fix or delete the docstring.

## What `paper/*.md` still OWNS — do not discard

The IJF rewrite compressed these out. v2 needs them back.

1. **The gate's name.** "Diebold-Mariano" appears twice in `04_results.md`,
   ZERO times in the IJF manuscript, which says only "a score differential of
   -2.0". This is what critics2 correctly flagged as an undefined statistic.
   The name was there and got stripped.
2. **`related_work.md`** — 5 citations, each verified against the published
   record, plus a QUARANTINE list of 3 fabricated/conflated references. The IJF
   manuscript instead carries ~30 unverified citations. The quarantine
   discipline is the paper's own thesis applied to itself; keep it.
3. **`06_limitations.md` 6.1** — the list of instruments the framework says the
   programme should have used and did not. This IS the round-4 fix roadmap
   (Clark-West, DSR, ES backtests, tail-weighted CRPS, PIT independence,
   staggered DiD, purged CV). No counterpart in the IJF version.
4. **`03_methodology.md` 3.5** — the four verification gates, stated in full.

## What `submission/ijf/manuscript.md` OWNS

- All corrected numbers (above).
- The multiplicity accounting at line 244: p=0.0095 -> 0.038 Holm, 0.052 on the
  common window, plus the -1.29 bare-baseline miss and the sequential-search
  caveat. critics2's "not formally corrected" charge is wrong about this file.
- The measured reporting-lag curve (0.13 / 94.30 / 97.51%).
- Equations 1-4 as numbered display objects.

`manuscript_v1_backup.md` (25 Aug 22:02) is superseded by `manuscript.md`
(23:35). Reference only.

## Open, and NOT settled by either version

Round 4 in progress. `exp10` is done and logged.

- Nested-model validity of the survivor's gate. exp10: DM reproduces at -2.223
  but the block bootstrap gives p=0.033 and Clark-West on the variance forecast
  gives t=+1.27, p=0.10 (fail). Estimation is EXPANDING-window, so
  Giacomini-White's finite-window escape does not apply. exp11 (rolling window)
  will settle it.
- A5 instrument and cost stack. `src/fii/config.py` carries only a generic
  `tcost_bps_oneway` on turnover. No instrument is modelled anywhere in the
  codebase, so neither manuscript can state one.
- Hazard calibration; deflated Sharpe on the 8 books; ES backtest;
  seven-applications x binding-constraint table.
