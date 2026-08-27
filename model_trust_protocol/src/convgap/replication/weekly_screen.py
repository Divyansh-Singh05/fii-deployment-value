"""Re-derivation of the weekly risk screen (pre-registered cells W1-W3).

The same gap as the aggregate-flow screen: `docs/PREREG_WEEKLY_ENGINE.md`
fixes three regressions before any result was seen and records their outcome
in an appended Result section, but no pipeline stage computes them. The engine
they gate has a stage; the gate does not. This file re-derives all three.

**The specification, transcribed and not varied.**

Target::

    RV5(t) = sqrt( sum_{k=1..5} r(t+k)^2 )

next-week realised volatility of the index. Controls present in every cell:
``RV5(t-5)`` (own trailing week, non-overlapping), the square root of an EWMA
variance with ``lambda = 0.94``, and ``india_vix(t)``. Inference is Newey-West
with 10 lags, the overlap in the target being the reason.

Three cells, one predictor each::

    W1  NF5(t-2)  sum of scaled net flow over t-6 .. t-2   (flow persistence)
    W2  inr5(t)   5-day USDINR log return ending t
    W3  spx5      5-day S&P 500 log return ending at the last US close
                  strictly before t

The availability alignment on W3 is load-bearing and is taken from the engine
module: the US close is dated forward one day before being joined backward-asof
to the Indian trading date, so no cell uses a US return published after the
Indian close it is asked to predict from.

**The bar**, identical in form to the daily screen: full-sample ``|t| >= 2.50``
and the same sign with ``|t| >= 1.5`` in both eras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from convgap.replication.aggregate_flow import (
    BAR_ERA_T,
    BAR_FULL_T,
    GROSS_WINDOW,
    NW_LAGS,
    TEST_START,
    TRAIN_END,
    ScreenCell,
    ScreenResult,
    _ols_hac,
    build_flow,
)

HORIZON: Final = 5
EWMA_LAMBDA: Final = 0.94


def _ewma_sigma(ret: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Causal EWMA volatility; value at t uses returns through t."""
    var = np.full(len(ret), np.nan)
    running = float(np.nanvar(ret[:60])) if len(ret) >= 60 else float(np.nanvar(ret))
    for i, r in enumerate(ret):
        if np.isfinite(r):
            running = lam * running + (1.0 - lam) * r * r
        var[i] = running
    return np.sqrt(var)


def build_frame(validation_data: Path, isin_mapping: Path) -> pl.DataFrame:
    """Index, macro and flow series aligned to Indian trading dates."""
    market = (
        pl.scan_parquet(validation_data / "returns_panel_v3.parquet")
        .select(["date", "nifty50_ret", "india_vix"])
        .unique(subset=["date"])
        .sort("date")
        .collect()
        .with_columns(pl.col("india_vix").forward_fill())
        .filter(pl.col("nifty50_ret").is_not_null() & pl.col("india_vix").is_not_null())
    )

    def _foreign(path: Path, col: str, out: str) -> pl.DataFrame:
        return (
            pl.read_parquet(path)
            .sort("date")
            .with_columns(lr=(pl.col(col) / pl.col(col).shift(1)).log() * 100)
            .with_columns(
                **{out: pl.col("lr").rolling_sum(HORIZON, min_samples=HORIZON)},
                avail=pl.col("date").dt.offset_by("1d"),
            )
            .select("avail", out)
            .drop_nulls()
            .sort("avail")
        )

    spx = _foreign(validation_data / "sp500.parquet", "sp500", "spx5")
    df = market.join_asof(spx, left_on="date", right_on="avail", strategy="backward")
    df = df.drop([c for c in df.columns if c.startswith("avail")])

    inr = _foreign(validation_data / "usdinr.parquet", "usdinr", "inr5")
    df = df.join_asof(inr, left_on="date", right_on="avail", strategy="backward")
    df = df.drop([c for c in df.columns if c.startswith("avail")])

    flow = build_flow(isin_mapping)
    df = df.join(flow, on="date", how="left").sort("date")
    df = df.with_columns(
        gmean=pl.col("gross").rolling_mean(GROSS_WINDOW, min_samples=GROSS_WINDOW // 2).shift(1)
    )
    df = df.with_columns(
        nf=pl.when(pl.col("net").is_not_null() & pl.col("gmean").is_not_null())
        .then(pl.col("net") / pl.col("gmean"))
        .otherwise(0.0)
    )

    ret = df["nifty50_ret"].to_numpy().astype(float)
    n = len(ret)
    rv5 = np.full(n, np.nan)
    for t in range(n - HORIZON):
        rv5[t] = np.sqrt(np.sum(ret[t + 1 : t + 1 + HORIZON] ** 2))

    # NaN is a *value* in polars, not a null: `drop_nulls` will not remove it.
    # The forward window leaves NaN in the final rows of rv5, and a single NaN
    # makes the whole regression return NaN rather than dropping the row.
    df = df.with_columns(
        rv5=pl.Series(rv5).fill_nan(None),
        sig_ewma=pl.Series(_ewma_sigma(ret)).fill_nan(None),
    ).with_columns(
        rv5_lag=pl.col("rv5").shift(HORIZON),
        # W1: scaled net flow summed over t-6 .. t-2
        nf5_t2=pl.col("nf").rolling_sum(HORIZON, min_samples=HORIZON).shift(2),
    )
    return df


_CELLS: Final = {
    "W1": ("flow persistence, NF5(t-2)", "nf5_t2"),
    "W2": ("USDINR 5-day return", "inr5"),
    "W3": ("S&P 500 5-day return", "spx5"),
}
_CONTROLS: Final = ["rv5_lag", "sig_ewma", "india_vix"]


def run_screen(validation_data: Path, isin_mapping: Path) -> list[ScreenResult]:
    """Estimate all three pre-registered weekly cells on FULL, TRAIN and TEST."""
    df = build_frame(validation_data, isin_mapping)
    eras = {
        "FULL": df,
        "TRAIN": df.filter(pl.col("date") <= TRAIN_END),
        "TEST": df.filter(pl.col("date") >= TEST_START),
    }

    results: list[ScreenResult] = []
    for cell, (desc, predictor) in _CELLS.items():
        by_era: dict[str, ScreenCell] = {}
        for era, frame in eras.items():
            cols = ["rv5", predictor, *_CONTROLS]
            sub = frame.select(cols).drop_nulls()
            y = sub["rv5"].to_numpy()
            if not np.isfinite(y).all():
                raise ValueError(f"{cell}/{era}: non-finite target survived drop_nulls")
            x = sub.select([predictor, *_CONTROLS]).to_numpy()
            coef, t_stat, p = _ols_hac(y, x, focus=0)
            by_era[era] = ScreenCell(cell, era, sub.height, coef, t_stat, p)
        results.append(ScreenResult(cell, desc, by_era["FULL"], by_era["TRAIN"], by_era["TEST"]))
    return results


def render(results: list[ScreenResult]) -> str:
    lines = [
        "weekly risk screen - pre-registered cells W1-W3",
        f"  target RV5(t) | Newey-West {NW_LAGS} | controls RV5(t-5), EWMA sigma, VIX",
        f"  bar: |t|>={BAR_FULL_T} full AND same sign with |t|>={BAR_ERA_T} both eras",
        "",
        f"{'cell':<5}{'predictor':<30}{'FULL t':>9}{'TRAIN t':>9}{'TEST t':>9}"
        f"{'(a)':>6}{'(b)':>6}{'verdict':>9}",
    ]
    for r in results:
        lines.append(
            f"{r.cell:<5}{r.description:<30}{r.full.t_stat:>9.2f}"
            f"{r.train.t_stat:>9.2f}{r.test.t_stat:>9.2f}"
            f"{'PASS' if r.leg_a else 'FAIL':>6}{'PASS' if r.leg_b else 'FAIL':>6}"
            f"{r.verdict:>9}"
        )
    lines.append("")
    lines.append("inherited, from PREREG_WEEKLY_ENGINE.md Result section:")
    lines.append(f"{'W3':<5}{'S&P 500 5-day return':<30}{-3.50:>9.2f}{-2.72:>9.2f}{-4.39:>9.2f}")
    return "\n".join(lines)
