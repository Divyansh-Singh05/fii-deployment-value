# FIHMM Deployment Value

Reproduction package for **"From Statistical Evidence to Decision Value: Six
Deployment Constraints on an Institutional Flow Dataset"** (*IEEE Access*).

The study takes one dataset — a fourteen-year, 25.16-million-trade depository
settlement record of foreign institutional flow in Indian equities — applies it
at **seven points in a modelling stack**, and subjects each application to
**six ascending deployment constraints**. Every application clears conventional
significance and out-of-sample persistence. **Six of the seven then fail to
reach a use.** One survives, under five stated boundary conditions.

## Quick start

```bash
pip install -r environment/requirements.txt

python reproduction/verify_results.py     # recompute vs the manuscript
python experiments/run_all.py             # the full ladder, L0 -> L5
python reproduction/reproduce_all.py      # everything, then verify

python reproduction/reproduce_figures.py --list   # the ten figures
python reproduction/reproduce_figures.py 4        # regenerate just Figure 4
```

**No configuration, no data download, no external path.** The dataset ships in
`data/` and every module resolves its inputs through `config/paths.yaml`, which
contains only package-relative paths. Clone it, install five libraries, run it.

## The ladder

| Level | Constraint | The question | Lost |
|---|---|---|---:|
| L0 | Existence | Is the effect present under conventional inference? | 0 |
| L1 | Persistence | Does it hold on an era the design never touched? | 0 |
| L2 | Availability | Is it knowable when the decision must be made? | 1 |
| L3 | Competition | Does it beat the baseline it faces in use? | 2 + control |
| L4 | Detectability | Can the consuming metric resolve it? | 2 |
| L5 | Execution | Does the edge exceed the cost of capturing it? | 1 |

The ordering is by **institutional reality**, not statistical severity, and it
is ordered so attrition is cumulative: an application that dies at L5 has
survived four filters that killed others.

**Nothing is lost at L0 or L1.** All attrition happens under deployment
constraint, and it is spread across four different levels — so no single
friction explains the pattern. That is the paper's central observation, and it
is not visible to a design that holds the friction fixed and varies the
predictor.

## What each directory is for

| Path | Purpose |
|---|---|
| `config/` | Every constant, transcribed from its producing stage |
| `data/` | The dataset. Self-contained; nothing is fetched at runtime |
| `src/data/` | Schema, validation, and the access statement |
| `src/features/` | Flow, return and volatility construction |
| `src/models/` | The instruments: OLS, GARCH, HMM, hazard, GBDT, density, portfolio |
| `src/validation/` | The six constraints, one module each |
| `experiments/` | One runner per level, plus `run_all.py` |
| `analysis/` | Statistical tests, multiplicity, robustness, the summary |
| `reproduction/` | Regenerate tables and figures, then verify |
| `analysis/figures.py` | All ten figures, one function each |
| `data/expected/` | Manuscript values, used as regression fixtures |

## Three things a reader should know before running anything

**The models are instruments, not contributions.** The dataset is the object of
study. When the ablation reports that a regime model's conditioning contributes
nothing to a risk density, that is a measurement about the flow data — not an
invitation to a better architecture.

**Identifiers are re-minted monthly.** Masked participant identifiers look
persistent and are not; a broker control population of a few hundred real firms
presents as 22,262 identifiers with a median lifetime of one month. Every
entity statistic is therefore **within-day**. See `data/source/README_NSDL.md`.

**Two expected results deliberately record a failure.** The survivor's
Clark–West adjustment (t = +1.27, p = 0.10) and its common-window multiplicity
p-value (0.052) do not clear their bars, and the manuscript reports both. A run
that turned them into passes would be wrong.

## Self-containment

Exactly one module in this package reads a path outside it: `src/data/ingest.py`,
the one-time ingestion that built `data/` from the upstream research trees, and
which names those trees in `config/ingestion.yaml`. It is not part of the
reproduction path and does not run again. Every other module - every model,
every level runner, every reproduction script - resolves inputs through
`dataset()`, which accepts only package-relative keys declared in
`config/paths.yaml`. There is no environment override and no escape hatch.

The upstream trees can be deleted without affecting a single number below.

## Verification

Every numeral in the manuscript clears four gates: **fresh** (artifact not
older than the code that writes it), **primary** (the computation, not a
document describing it), **traced** (SHA-256 matches a recorded digest), and
**extracted** (a locator reads the value back out and compares it).

On a `MISMATCH`, the declaration is corrected — never the tolerance.

`config/paths.yaml` quarantines artifacts carrying `PREAUDIT`, `BASELINE`,
`_smoke`, `_trunc` or `_unit`, and `artifact()` refuses to open them. That
guard exists because the failure it prevents actually occurred three times in
this programme's own source material.

## Licence and data

Code is MIT. **The underlying records are proprietary, are not included, and
are not licensed for redistribution.**
