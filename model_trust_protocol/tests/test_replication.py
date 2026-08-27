from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from convgap.replication import aggregate_flow as af


def _panel(n: int = 900, seed: int = 3) -> pl.DataFrame:
    """A synthetic frame with a known flow effect on next-day |return|."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    nf = rng.normal(size=n)
    vix = 15 + rng.normal(size=n)
    # The regression is |r(t+1)| ~ nf(t-2), so |r(k)| must depend on nf(k-3).
    # Plant a negative effect: more net inflow, calmer market three days later.
    scale = np.full(n, 0.01)
    scale[3:] = 0.01 * np.exp(-0.6 * nf[:-3])
    ret = rng.normal(size=n) * scale
    return (
        pl.DataFrame({"date": dates, "nifty50_ret": ret, "india_vix": vix, "nf": nf})
        .with_columns(
            y_dir=pl.col("nifty50_ret").shift(-1),
            y_vol=pl.col("nifty50_ret").shift(-1).abs(),
            nf_t0=pl.col("nf"),
            nf_t2=pl.col("nf").shift(2),
        )
        .with_columns(
            neg_t0=pl.min_horizontal(pl.col("nf_t0"), pl.lit(0.0)),
            neg_t2=pl.min_horizontal(pl.col("nf_t2"), pl.lit(0.0)),
            **{f"vol_lag{k}": pl.col("nifty50_ret").abs().shift(k) for k in range(5)},
            **{f"dir_lag{k}": pl.col("nifty50_ret").shift(k) for k in range(5)},
        )
    )


def test_specification_constants_match_the_prereg() -> None:
    assert af.NW_LAGS == 10
    assert af.GROSS_WINDOW == 250
    assert af.N_TARGET_LAGS == 5
    assert af.BAR_FULL_T == 2.50
    assert af.BAR_ERA_T == 1.50
    assert dt.date(2021, 4, 30) == af.TRAIN_END
    assert dt.date(2021, 7, 1) == af.TEST_START


def test_recovers_a_planted_effect() -> None:
    cell = af._estimate(
        _panel(),
        "y_vol",
        ["nf_t2", "neg_t2", *[f"vol_lag{k}" for k in range(5)], "india_vix"],
        "exclude",
    )
    assert cell is not None
    assert cell.coef < 0, "planted effect is negative"
    assert abs(cell.t_stat) > 2, "planted effect should be detectable"


def test_finds_nothing_when_flow_is_noise() -> None:
    panel = (
        _panel()
        .with_columns(nf_t2=pl.Series(np.random.default_rng(99).normal(size=_panel().height)))
        .with_columns(neg_t2=pl.min_horizontal(pl.col("nf_t2"), pl.lit(0.0)))
    )
    cell = af._estimate(
        panel,
        "y_vol",
        ["nf_t2", "neg_t2", *[f"vol_lag{k}" for k in range(5)], "india_vix"],
        "exclude",
    )
    assert cell is not None
    assert abs(cell.t_stat) < af.BAR_FULL_T, "a pure-noise regressor must not clear the bar"


def test_zero_flow_reading_must_be_declared() -> None:
    with pytest.raises(ValueError, match="zero_flow"):
        af._estimate(_panel(), "y_vol", ["nf_t2", "vol_lag0"], "whatever")


def test_exclude_reading_drops_unobserved_days() -> None:
    panel = _panel().with_columns(
        nf_t2=pl.when(pl.arange(0, pl.len()) % 5 == 0).then(pl.lit(0.0)).otherwise(pl.col("nf_t2"))
    )
    cols = ["nf_t2", "neg_t2", *[f"vol_lag{k}" for k in range(5)], "india_vix"]
    kept = af._estimate(panel, "y_vol", cols, "as_zero")
    dropped = af._estimate(panel, "y_vol", cols, "exclude")
    assert kept is not None and dropped is not None
    assert dropped.n < kept.n


def test_verdict_requires_both_legs() -> None:
    def cell(t: float, coef: float) -> af.ScreenCell:
        return af.ScreenCell("C4", "x", 100, coef, t, 0.0)

    strong = af.ScreenResult("C4", "d", cell(-4.0, -1), cell(-2.0, -1), cell(-2.0, -1))
    assert strong.leg_a and strong.leg_b and strong.passes

    weak_full = af.ScreenResult("C4", "d", cell(-1.0, -1), cell(-2.0, -1), cell(-2.0, -1))
    assert not weak_full.leg_a and not weak_full.passes

    sign_flip = af.ScreenResult("C4", "d", cell(-4.0, -1), cell(-2.0, -1), cell(2.0, +1))
    assert sign_flip.leg_a and not sign_flip.leg_b, "a sign flip fails regardless of t"
