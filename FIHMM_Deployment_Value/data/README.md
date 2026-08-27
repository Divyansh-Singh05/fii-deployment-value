# Data layer

Five directories, in pipeline order. Nothing here is generated at import time;
every stage writes explicitly and logs a digest.

| Directory | Contents | In git? |
|---|---|---|
| `source/` | Documentation only - see `README_NSDL.md` | yes |
| `corporate_actions/` | Parsed filings, match report, price-ratio validation | yes |
| `instrument_mapping/` | Point-in-time identity map and its validation | yes |
| `derived/` | Ingested and adjusted panels | no - regenerated |
| `expected/` | Manuscript values, used as regression fixtures | yes |

## The quarantine

`config/paths.yaml` declares `quarantined_infixes`, and
`src/common/utilities.artifact()` refuses any path matching one. The infixes
are `PREAUDIT`, `BASELINE`, `_smoke`, `_trunc`, `_unit`.

This exists because the failure it prevents actually happened. The programme's
own verification protocol found three instances of stale material reaching a
reported number: two pre-registered gating regressions with no producing
stage, a metrics table quoted from a copy six weeks behind its origin, and a
stage whose reported effect size was a third larger than its current object
supports because it was never re-executed after an audit rebuilt that object.
None changed a verdict. All changed a number that had already been written
down.

A digest match on a stale copy still passes. That is why `fresh` is a separate
gate from `traced`.

## The four admission gates

Every numeral admitted to the manuscript clears all four:

1. **Fresh** - the artifact is not older than the code that writes it, and no
   upstream stage has been re-run since.
2. **Primary** - the source is the computation, not a document transcribing it
   and not a copy of its output.
3. **Traced** - the artifact's SHA-256 matches a recorded digest; a changed
   artifact reports `STALE` rather than being silently accepted.
4. **Extracted** - a locator identifying exactly one number reads the value
   back out and compares it. A disagreement reports `MISMATCH`, and **the
   declaration is corrected, never the tolerance**.

## What is actually in these directories

Built by `python -m src.data.build_data_layer`, which is re-runnable and writes
`MANIFEST.csv` with a SHA-256 for every file it produces.

| File | Rows | Class | Derived from |
|---|---:|---|---|
| `corporate_actions/corporate_actions.csv` | 1,103 | PUBLIC | `ca_adjustment_factors.parquet` |
| `corporate_actions/ca_matches.csv` | 2 | PUBLIC | NSE + BSE filings |
| `corporate_actions/ca_validation.csv` | 1,103 | PUBLIC | `ca_adjustment_factors.parquet` |
| `instrument_mapping/instrument_map.csv` | 1,784 | PUBLIC | `isin_pit_map.parquet` |
| `instrument_mapping/mapping_validation.csv` | 14 | PUBLIC | `isin_mapping_final.csv` |
| `derived/daily_features.parquet` | 3,490 | DERIVED | `module21.build_frame()` |
| `derived/analysis_dataset.parquet` | 2,739 | DERIVED | both market-engine score sets |

### The three disclosure classes

**PUBLIC** - corporate actions and the instrument map. Derived from exchange
filings that are already public. Redistributable.

**DERIVED** - market-level aggregates and scored panels. No participant
identifier, no instrument-level position, nothing from which a trade could be
reconstructed. Redistributable.

**RESTRICTED** - the transaction record itself. **Not written.**
`derived/adjusted_transactions.RESTRICTED.json` carries its schema, its row
count and the SHA-256 of the upstream artifact instead, so the package is
complete and honest rather than complete and wrong.

### The package is self-contained

Every number in the manuscript now recomputes from this directory alone.
`derived/panel_analysis.parquet` carries the full 577,245-row regression panel,
so **Table 1 and Levels 0-2 no longer need any external tree** - and neither
does anything else.

Exactly one module ever reads outside the package: `src/data/ingest.py`, the
one-time ingestion that built this directory and which names the upstream trees
in `config/ingestion.yaml`. It is not part of the reproduction path. Every
other module resolves inputs through `dataset()`, which accepts only keys
declared in `config/paths.yaml`, all package-relative. There is no environment
override.

The upstream research trees can be deleted without affecting a single result.

### Corporate-action validation: 7 mismatches, reported not suppressed

Each parsed factor is checked against the **observed** pre/post price ratio
rather than applied on trust:

| Status | Count | Meaning |
|---|---:|---|
| CONFIRMED | 805 | observed ratio matches the parsed factor |
| UNVERIFIABLE | 291 | no clean ratio observable (illiquid or suspended around the ex-date) |
| MISMATCH | 7 | observed ratio contradicts the filing |

The seven mismatches share a signature worth stating. Six are **bonus** issues
whose observed ratio is close to 1.0 against a parsed factor of 1.5, 2.0 or
7.0 - the price did not move as a bonus would require. That is consistent with
the upstream price series having already been adjusted at source, in which case
applying the factor again would *introduce* the error rather than remove it.
The seventh (DPSCLTD, 2011-12-15) is a lone outlier at 209.88 against a factor
of 10.

They are flagged rather than fixed, because a diagnostic is read-only and a
data repair is never bundled with the analysis that motivated it. The 50%
daily-return guard in `src/corporate_actions/adjust.py` is what catches the
downstream consequence if any of them is applied wrongly.
