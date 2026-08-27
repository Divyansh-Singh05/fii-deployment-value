# FII Flow Regimes
### Trade concentration and the transitory–permanent decomposition of institutional price impact

A fully reproducible quantitative research pipeline. Fourteen years of masked
NSDL FII trade records (India, 2011–2025), an unsupervised flow-regime model,
and a pre-registered, gate-driven validation battery whose headline finding
**inverted the project's own starting hypothesis**:


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

> **Concentrated** FII selling (few participating institutions) is
> liquidity-demanding — prices fall with a volume climax, then revert
> +49–65 bp/20d once the flow stops. **Dispersed** selling (many
> institutions, quiet volume) is information — the decline is permanent.
> An independent Easley–O'Hara PIN estimator, blind to returns, endorses the
> split (~3× informed-trading loading on dispersed-selling exposure).

Full results: [`docs/00_project_overview.md`](docs/00_project_overview.md) ·
working paper: [`docs/paper/`](docs/paper/) · unabridged evidence trail:
[`docs/research_log/`](docs/research_log/).

---

## Quickstart

```bash
# 1. environment (uv)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# macOS + LightGBM: brew install libomp

# 2. data (extracts the research data export into data/)
./scripts/setup_data.sh          # or: SOURCE_DIR=/path/to/zips ./scripts/setup_data.sh

# 3. the whole research chain, one command
python pipeline.py --all
```

Useful invocations:

```bash
python pipeline.py --list                      # stage manifest
python pipeline.py --phase validation          # one phase
python pipeline.py --stage panel_regression    # one stage
python pipeline.py --from-stage canonical_panel  # resume mid-chain
python pipeline.py --phase audit               # read-only diagnostics
python pipeline.py --model hmm_regime          # train/evaluate a model
```

Every run writes a timestamped log per stage to `outputs/logs/`; seeds are
fixed from `config/config.yaml`.

## Repository layout

```
pipeline.py                  unified entrypoint
config/config.yaml           paths, frozen split dates, frozen thresholds, costs
src/fii/
  paths.py, config.py        config-driven paths (FII_DATA_ROOT overridable)
  runner.py                  stage executor: seeding, logging, halt-on-failure
  stages/registry.py         ordered stage manifest (the dependency graph)
  data_prep/                 tape repair → CA adjustment → canonical panel
  features/                  the 10-feature stock-day flow store
  models/                    BaseModel interface + auto-discovery registry
    hmm_stages/              HMM build/calibration stages (the main model)
    hmm_regime.py            main model behind the common interface
    lightgbm_gbt.py          GBT challenger
    _template.py             copy this to add a new model (one file, nothing else)
  validation/                the economic-validation battery (+ audits/)
  backtest/                  engine.py + gates + strategy pairs
legacy/colab_modules/        every original Colab script, byte-for-byte
docs/                        thesis-grade methodology docs (see map below)
outputs/                     logs, tables, figures, metrics… (generated)
data/                        research data (generated by setup, gitignored)
```

## Documentation map

| | |
|---|---|
| [`docs/00_project_overview.md`](docs/00_project_overview.md) | findings, pipeline diagram, design philosophy |
| [`docs/01_data_preparation.md`](docs/01_data_preparation.md) | CA-adjustment math, ISIN canonicalization, the gates |
| [`docs/02_feature_engineering.md`](docs/02_feature_engineering.md) | feature definitions, probit ranking, leakage rules |
| [`docs/03_models.md`](docs/03_models.md) | HMM foundations (Baum–Welch, Viterbi), hybrid design, LightGBM, extension contract |
| [`docs/04_validation_framework.md`](docs/04_validation_framework.md) | event study, PanelOLS spec, INNOV, PIN likelihood |
| [`docs/05_backtesting.md`](docs/05_backtesting.md) | engine mechanics, metric definitions, honest results |
| [`docs/STAGES.md`](docs/STAGES.md) | per-stage purpose / inputs / outputs / gates |
| [`docs/paper/FII_thesis.md`](docs/paper/FII_thesis.md) | the long-form account (22 sections; also as PDF/DOCX) — every decision with its math, reality check, and file citation |
| [`docs/paper/identifier_audit_note.md`](docs/paper/identifier_audit_note.md) | standalone note: the masked-identifier audit protocol |
| [`docs/ADOPTION_RECIPE.md`](docs/ADOPTION_RECIPE.md) | one-page recipe: compute the composition measure on your own data |

## Design decisions worth knowing

**Stages are the original research scripts, preserved.** This research was
built as sequentially-verified scripts, each with pre-registered PASS/FAIL
gates in its printed output; the validation log cites those exact artifacts.
Migration applied exactly two mechanical substitutions (Colab paths →
`fii.paths`; backtest session-coupling → an explicit import), auditable in
`scripts/migrate_colab_modules.py`. Originals live untouched in `legacy/`.
Changing stage *logic* requires re-running the affected gates — that is the
contract.

**The temporal protocol is frozen.** Train ≤ 2021-04-30, test ≥ 2021-07-01,
May–June 2021 masked everywhere. These dates live in config for
transparency, not for tuning.

**Failures are part of the record.** The battery caught two silent code
corruptions, a false exchange-data assumption, a CA-parser bug, and two
wrong economic narratives — all documented in the research log with their
fixes. This is a feature of the methodology, not an embarrassment.

**Extending with a new model** = copy `src/fii/models/_template.py`, set a
name, implement four methods. The registry auto-discovers it; the frozen
split, feature-store contract, and validation battery apply unchanged.

## Data note

The pipeline expects the two research data exports (NSDL FII trade parquets
+ NSE price/CA/deal data). They are **not** redistributed in this repository;
`scripts/setup_data.sh` extracts them from local zips (default:
`~/Desktop/Major Project 1/Data`, override with `SOURCE_DIR`). NSDL FII
records are licensed data — see `docs/01_data_preparation.md` for schema.

## Requirements

Python ≥ 3.11 · numpy, pandas, polars, pyarrow, scipy, statsmodels,
linearmodels, hmmlearn, lightgbm, scikit-learn, pyyaml, matplotlib
(`requirements.txt` / `pyproject.toml`). macOS LightGBM needs
`brew install libomp`.
