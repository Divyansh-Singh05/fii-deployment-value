# convgap

**Measuring the conversion of statistical significance into forecasting value.**

Research code for a study of one question: when a predictor in a financial model
is established as statistically significant by conventional means, how much of
that finding survives into a forecast or a decision?

The package pairs, for each predictor, the significance result and the value
result **measured on the same object and the same data**, and reports the ratio
of observed improvement to the upper bound implied by the measured effect size.

---

## The design

Each *channel* is one route from an established statistical result to a
decision-relevant one, together with the mechanism by which the two differ.
Channels are stratified by the modelling layer at which significance was
established, so that the finding cannot be an artifact of a single methodology.

| | Channel | Layer | Mechanism |
|---|---|---|---|
| C1 | Institutional share of turnover → volatility | conditional-mean regression | Information decays before the lag a decision can use |
| C2 | Flow tilt on a GJR-GARCH base | density forecast | A better-specified baseline already absorbs the predictor |
| C3 | S&P 500 → Nifty weekly spillover | density forecast | Era-stability is evidence of reality, not of incremental content |
| C4 | Episode-end hazard model | classification / hazard | Real skill, non-incremental to a three-line rule |
| C5 | Best long/short book | portfolio | Execution cost consumes the signal |
| C6 | Participant-composition block | feature attribution | Significant in the extremes, absent on average |
| C7 | Flow-regime conditioning in a risk density | density forecast | Ablation shows the mechanism contributes nothing |
| **C8** | **Aggregate flow → market density** | **density forecast** | **Control: the predictor converts** |

C8 is load-bearing. Without a converting channel the study asserts absence
rather than measuring a rate, and the test suite enforces its presence.

## Provenance

The study is about the difference between a number and the evidence for it, so
the package holds itself to the same standard. Every value carries the artifact
it came from, a locator precise enough for a third party to follow by hand, and
a digest of that artifact recorded at verification. Two conditions must both
hold before a value may appear in a submitted exhibit:

- **traced** — the artifact exists and its digest matches the recorded one;
  a changed artifact is reported `STALE` rather than silently accepted;
- **primary** — the source is the computation that produced the number, not a
  project document transcribing it. A digest match on prose establishes only
  that nobody edited the prose.

`convgap verify` reports the two separately, because promoting a row from a
document to its originating artifact is a distinct piece of work from checking
that the document is unchanged.

## Install

```bash
make install          # uv venv + editable install with dev extras
```

## Use

```bash
make channels         # registered channels and layer coverage
make verify           # trace every number to its source
make table1           # build the conversion exhibit
make all              # lint, typecheck, test  (no source trees needed)
```

`channels`, `all`, `lint`, `typecheck` and `test` run without any research data.
`verify` and `table1` need the source trees below.

## Data availability

The study draws on two read-only project trees. Neither is modified by this
package, and the underlying transaction records are proprietary and cannot be
redistributed.

| Key | Contents | Default location |
|---|---|---|
| `research` | 62-stage research object: panel econometrics, backtests, market-level density engines, state forecasting | `~/Desktop/Major Project 2` |
| `engine` | Stock-day distributional risk engine, stages s01–s13 | `~/Desktop/fii_risk_engine` |

Override either with an environment variable:

```bash
export CONVGAP_RESEARCH_ROOT="/path/to/research"
export CONVGAP_ENGINE_ROOT="/path/to/engine"
```

Paths, digests and locators are recorded for every exhibit entry, so a reviewer
with access to the trees can check any number without reading this code.

## Layout

```
src/convgap/
  provenance.py      source trees, digests, verification status, source tier
  evidence.py        a number with its statistic, era, sample size and origin
  channel.py         the Channel abstraction and its resolved result
  registry.py        registration, discovery, layer-coverage checks
  benchmark/         implied-versus-observed improvement accounting
  channels/          one module per channel
  exhibits/          Table 1
  cli.py             convgap channels | verify | table1
tests/               unit tests, including invariants on the study design
config/sources.yaml  source-tree resolution
docs/                framing, protocol, instrument families, paper spine
```

Design invariants are enforced as tests rather than documented as intentions:
every layer must be covered, at least one channel must report conversion, every
channel must state a mechanism, and every number must carry an actionable
locator.

## Status

Phase 0 complete: the channel set is specified and the exhibit builds.
Phase 1 in progress: promoting all eight channels from project documents to the
artifacts that produced them, and recording digests. `convgap verify` is the
work list.

## Licence

MIT. See `LICENSE`.
