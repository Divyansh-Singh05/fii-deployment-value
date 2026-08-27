# FII Flow-Conditioned Risk Engine

A standalone forecasting engine that conditions next-day, next-week and
next-month return distributions on *who* is trading a stock — the
concentration and persistence of foreign institutional flow — rather than on
price history alone.

It serves VaR, Expected Shortfall, CRPS and PIT for any stock and window, and
is scored head-to-head against a closed-form EWMA-Normal benchmark.

**This project is self-contained.** Every module it imports lives in this
directory and everything it writes goes to `outputs/` here. The one external
dependency is the read-only market and flow data, which is too large to
duplicate — see [DATA.md](DATA.md).

---

## 1 · Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2 · Point it at the data

The engine looks for a directory containing `VALIDATION_DATA/` and
`ISIN_MAPPING/`, in this order:

1. `$FII_DATA_ROOT`
2. `./data` (a local copy or symlink)
3. `~/Desktop/Major Project 2/data`

Any one of these works:

```bash
export FII_DATA_ROOT="/path/to/Major Project 2/data"
# or
ln -s "/path/to/Major Project 2/data" ./data
```

It fails loudly at import if none of them resolve. Only two files are needed
to run the serve path:

```
VALIDATION_DATA/states_v3.parquet          19 MB
VALIDATION_DATA/returns_panel_v3.parquet  205 MB
```

## 3 · Run it

`outputs/phase3/vintages.parquet` ships with the project — that is the fitted
model, 106 sealed monthly parameter snapshots. Everything downstream rebuilds
from it in about seven minutes:

```bash
./run_all.sh
```

Per-stage logs and a timing table land in `outputs/logs/`. To use a specific
interpreter: `PYTHON=/path/to/python ./run_all.sh`.

Then query the engine:

```bash
python -m s12_risk_engine --stock RELIANCE --start 2023-01-01 --end 2023-12-29 --horizon 1
```

`--horizon` accepts 1 and 5. **h=20 is retired from the public interface** and
is reachable only with `allow_internal=True` in code — it breaches 5.67% at
the 5% level and is not calibrated (see §5).

The panel only carries stock-days on which FII flow actually occurred, so a
thinly-traded name or a short window can legitimately return *"no scored days
in this window"*. Liquid names with near-full coverage include RELIANCE, TCS,
ICICIBANK, LT, MARUTI and NTPC (~2,050 scored days each).

A worked interpretation of a full run is in **[temp_result.md](temp_result.md)**.

### Rebuilding the model itself

```bash
python -m s05_refit_harness --rebuild --jobs 2     # ~25 min
```

Vintages are independent, but the fit is memory-bandwidth bound: on an 8 GB
machine `--jobs 2` is the optimum and `--jobs 3` is *slower*. Verify a rebuild
against the shipped file with `--verify 5 40 90`.

`s01`–`s04` rebuild the feature store, the backbone HMM and the calibrated
thresholds from the raw transaction record. They are included for completeness
and are not needed to run the engine.

---

## 4 · Files

### Support modules

| file | description |
|---|---|
| `paths.py` | Locates the data, keeps all writes inside this project. Fails loudly if the data is missing. |
| `gap_aware_hmm.py` | Gaussian HMM whose transition matrix is expressed **per trading day**: observations `k` days apart are linked by `A^k`, not `A`. Means, covariances and startprob keep closed forms; `A` is maximised numerically because `k > 1` has none. The E-step steps every stock sequence forward together, which is what makes a full refit ~25 min instead of 17 h. |
| `trading_calendar.py` | The single price-defined trading calendar. Both the estimator and the filter measure elapsed time against it, so they cannot disagree. |
| `panel_inference.py` | Date-block bootstrap and conditional-calibration strata. Replaces the textbook χ²(9) and fractional-count Kupiec, neither of which is valid under panel dependence. |
| `lineage.py` | Stamps every artifact with input hashes and the commit that produced it. |
| `isin_pit_map.py` | Point-in-time identity intervals, so a stock's ISIN history is resolved as of the scoring date and never with hindsight. |

### Pipeline

| file | description |
|---|---|
| `s01_feature_store.py` | The four flow features per stock-day: participation concentration on each side, block intensity, persistence. Liquidity floor → within-day rank → probit, all windows lagged and gap-skipping. |
| `s02_hmm_backbone.py` | Three-state Gaussian HMM over those features — SELL / NEUTRAL / BUY as separated modes on the persistence axis. |
| `s03_threshold_calibration.py` | Archetype cut-points from BIC-selected 1-D Gaussian mixtures, threshold at the posterior-0.5 boundary, k-means cross-check. |
| `s04_causal_filter_frozen.py` | Forward-only filter under frozen parameters; confirms filtered labels reproduce the economics of full-sequence labels. |
| `s05_refit_harness.py` | **Creates the vintages.** Re-fits monthly on a trailing 5-year window, 106 times, sealing each. Six restarts with a structural gate. Thresholds are posterior-weighted quantiles with stock-clustered bootstrap intervals. |
| `s06_daily_filter.py` | **Extracts the probabilities.** Forward recursion giving `P(S_t \| x_1..t)`, propagated across gaps with `A^k` over elapsed trading days. Hand-written because a library `predict_proba` sees the whole sequence. |
| `s07_archetype_probs.py` | Three state probabilities → seven archetype **weights**. Threshold uncertainty enters as `Φ((θ̂ − F)/s)`. |
| `s08_outcome_densities.py` | Past-only outcome CDFs per vintage × archetype × horizon on volatility-standardised returns, under an `s + h ≤ asof` embargo. |
| `s09_predictive_mixture.py` | Mixes those densities by the archetype weights; reads off VaR, ES, CRPS, PIT. A VaR is withheld unless the tail actually carries enough raw observations to support the level. |
| `s10_validation.py` | Pre-registered tests: Diebold-Mariano with Newey-West, Kupiec with block-bootstrap intervals, Christoffersen at h=1, PIT against its own bootstrap null. |
| `s11_stock_risk_profiles.py` | Head-to-head against EWMA-Normal, panel-wide and per stock, plus mandatory conditional-calibration strata. |
| `s12_risk_engine.py` | **The query engine.** Verifies on startup that it reproduces the stored pipeline output. |
| `s13_lookahead_audit.py` | **The proof of no look-ahead.** Truncates all inputs at a date, re-runs s06→s09, and requires every earlier row to be identical *and equally available* — a value flipping finite→NaN fails rather than being skipped. |

---

## 5 · What the engine will and will not claim

Honest scope, from the validation output rather than the design intent:

- **h=1 and h=5 are calibrated.** Breach rates 5.12% / 1.04% against nominal
  5% / 1%, PIT not rejected against its own bootstrap null, ES coverage 1.02.
  The Gaussian benchmark fails decisively at the same horizons.
- **h=20 is not.** It over-breaches and is retired from the public interface.
  Consecutive 20-day windows overlap by 19 days, so the independent
  information is roughly a twentieth of what the row count implies.
- **Calibration is conditional, not uniform.** Breach rates run ~1.23×
  nominal in the lowest EWMA-volatility quintile and ~0.76× in the highest.
  A pooled headline hides this, so `s11` reports the strata as mandatory
  output.
- **Tails are genuinely fat.** ES/VaR of 1.38–1.43 against a Gaussian 1.146;
  `z < −5` occurs 691 times where a Gaussian expects 0.17.
- The engine forecasts **distributions, not direction.** It is not a
  trading signal, and the economic interpretation of the flow archetypes is
  a separate and currently unsettled question.

---

## 6 · Not included

ISIN mapping and corporate-action reconciliation, price-panel assembly, and
the Phase I/II economic validation suite. Those are upstream plumbing or a
different research question; this tree is the risk engine only.
