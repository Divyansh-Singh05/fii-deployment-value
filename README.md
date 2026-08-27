# FII Deployment-Value Research

Private monorepo for the institutional-flow deployment-constraint programme:
the research tree, the standalone risk engine, the publication tree, and the
self-contained reproduction package.

**The data is proprietary.** The underlying depository settlement records are
not a commercial product and are not redistributable outside this repository
and its named collaborators.

---

## First run, after cloning

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
./unpack_data.sh
```

`unpack_data.sh` reassembles `data_archive/` and unpacks it in place. The
dataset is stored as a **split tar archive** at 95 MB per part because GitHub
rejects any single file over 100 MB; splitting keeps the repository free of Git
LFS. Parquet is already compressed, so the archive is stored uncompressed —
gzipping it measured *larger*, not smaller.

After unpacking you will have `major_project_2/data/`,
`major_project_2/outputs/` and `fii_risk_engine/outputs/`. Those paths are
gitignored, so the unpacked copies are never committed back.

---

## What is in here

| Directory | Role |
|---|---|
| `major_project_2/` | **The research tree.** Authoritative for every empirical result. All validation modules, the phase-3 density stack, backtests. |
| `fii_risk_engine/` | **Standalone engine.** Flat, runnable copy of the stock-day HMM/density pipeline (`s01`–`s13`). |
| `model_trust_protocol/` | **Publication tree.** Manuscripts, LaTeX/Word builds, figure generation, the `convgap` provenance tooling. |
| `FIHMM_Deployment_Value/` | **Reproduction package.** Fully self-contained: ships its own 24 MB dataset and recomputes every headline number with no external paths. |
| `data_archive/` | The split dataset. Restored by `unpack_data.sh`. |
| `VENUE_TARGETING.md` | Which repo to use for what, the redundancy traps, and how to target a journal. **Read this first.** |

---

## Read this before writing code

`VENUE_TARGETING.md` is not optional background. It records three findings that
are not obvious from the file tree and that have already caused wrong numbers:

1. **`major_project_2` holds the newest engine code, not `fii_risk_engine`.**
   Verified stage by stage. The two trees hold the same nine stages under
   different names; two have diverged scientifically, and only
   `major_project_2` has the EVT tail benchmark and the institutional-share
   volatility correction.
2. **Modification time is not a version signal.** Several `fii_risk_engine`
   files are newer by timestamp and older by content.
3. **Numbered variants are not revisions.** `module21`–`module24` are four
   *different* engines, not four versions of one.

Confirm which module is the intended source before building on it. A wrong
choice does not fail loudly — it produces a plausible number that is quietly
wrong.

## What is deliberately absent

Artifacts matching `PREAUDIT`, `BASELINE`, `_smoke`, `_trunc`, `_unit` are
excluded from the archive and gitignored. They predate the audit that rebuilt
the state object and must never enter a reported number. `legacy/` and the
superseded `stockday_features.parquet` (zero code references) are also out.

## Verifying you have a working tree

```bash
cd FIHMM_Deployment_Value
pip install -r environment/requirements.txt
python reproduction/verify_results.py
```

That recomputes the manuscript's headline numbers from its own bundled data and
compares them against the published values. It needs nothing from the archive.

## Superseded repositories

`Divyansh-Singh05/major-project-code` and `Divyansh-Singh05/fii-flow-regimes`
predate this monorepo. They are left in place as a fallback; treat this
repository as the live one.
