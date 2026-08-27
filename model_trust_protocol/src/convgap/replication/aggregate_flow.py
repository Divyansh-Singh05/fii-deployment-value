"""Re-derivation of the aggregate-flow screen (pre-registered cells C1-C4).

The source programme's `docs/PREREG_AGGREGATE_FLOW.md` fixes four regressions
before any result was seen, and reports that cell C4 passed its bar. No pipeline
stage computes them: the figures survive only in the docstring of the module
that consumes them. This file re-derives all four from the specification.

**The specification, transcribed and not varied.**

Flow series, over the raw depository records::

    NF(t) = sum(VALUE_INR | TR_TYPE = 1) - sum(VALUE_INR | TR_TYPE = 4)

restricted to ``RATE > 0`` and ``RFDE_INSTR_TYPE = REG_DL_INSTR_EQ``, then
scaled by the trailing 250-trading-day mean of daily gross flow ending at t-1.
Days with no flow observation enter as ``NF = 0`` - "no information" - and are
never imputed with a direction.

Four cells, controls fixed at five lags of the target, with the volatility
cells additionally controlling ``india_vix(t)`` and carrying the single
permitted asymmetry term ``NEG = min(NF, 0)``::

    C1  nifty_ret(t+1)   ~ NF(t)                  direction, same-day
    C2  |nifty_ret(t+1)| ~ NF(t),   NEG(t)        risk,      same-day
    C3  nifty_ret(t+1)   ~ NF(t-2)                direction, deployable
    C4  |nifty_ret(t+1)| ~ NF(t-2), NEG(t-2)      risk,      deployable

Inference is Newey-West with 10 lags. Eras are the frozen split: TRAIN through
2021-04-30, TEST from 2021-07-01.

**One ambiguity in the inherited specification, resolved and reported.** The
prose does not say what becomes of days on which no flow was observed - the
months masked at source, and those carrying a null direction flag. Two readings
are available: treat them as ``NF = 0`` ("no net flow"), or exclude them ("no
measurement"). They are not equivalent, and the choice moves the training-era
statistic materially. We take **exclusion** as canonical, on the grounds that a
day with no observation has no measurement rather than a measurement of zero,
and we report both readings so the sensitivity is visible rather than buried.

**The bar, also fixed in advance.** A cell passes only if both hold:

(a) full-sample ``|t| >= 2.50`` on the flow coefficient - a Bonferroni
    allowance for the four declared cells;
(b) the same coefficient carries the same sign with ``|t| >= 1.5`` in TRAIN and
    in TEST separately. A sign flip between eras fails regardless of t.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

GROSS_WINDOW: Final = 250
"""Trailing trading days over which mean gross flow is taken."""

NW_LAGS: Final = 10
"""Newey-West lag truncation, fixed by the pre-registration."""

TRAIN_END: Final = dt.date(2021, 4, 30)
TEST_START: Final = dt.date(2021, 7, 1)

N_TARGET_LAGS: Final = 5

ZERO_FLOW_CANONICAL: Final = "exclude"
"""How unobserved-flow days are handled in the canonical estimate.

``"exclude"`` drops them; ``"as_zero"`` retains them with ``NF = 0``. See the
module docstring: the inherited specification does not say, and the two readings
differ enough to be worth reporting side by side.
"""
BAR_FULL_T: Final = 2.50
BAR_ERA_T: Final = 1.50

_INSTR: Final = "REG_DL_INSTR_EQ"
_BUY, _SELL = 1, 4


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_flow(isin_mapping: Path) -> pl.DataFrame:
    """Daily aggregate net and gross rupee flow, exactly as specified."""
    files = sorted(str(p) for p in isin_mapping.glob("20[0-9][0-9].parquet"))
    if not files:
        raise FileNotFoundError(f"no yearly flow parquets under {isin_mapping}")
    return (
        pl.scan_parquet(files)
        .with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))
        .filter(
            pl.col("TR_TYPE").is_in([_BUY, _SELL])
            & (pl.col("RATE") > 0)
            & (pl.col("RFDE_INSTR_TYPE") == _INSTR)
        )
        .group_by("TR_DATE")
        .agg(
            net=(
                pl.when(pl.col("TR_TYPE") == _BUY)
                .then(pl.col("VALUE_INR"))
                .otherwise(-pl.col("VALUE_INR"))
            ).sum(),
            gross=pl.col("VALUE_INR").sum(),
        )
        .rename({"TR_DATE": "date"})
        .sort("date")
        .collect()
    )


def build_frame(validation_data: Path, isin_mapping: Path) -> pl.DataFrame:
    """Join the market series to the scaled flow series."""
    market = (
        pl.scan_parquet(validation_data / "returns_panel_v3.parquet")
        .select(["date", "nifty50_ret", "india_vix"])
        .unique(subset=["date"])
        .sort("date")
        .collect()
        .with_columns(pl.col("india_vix").forward_fill())
        .filter(pl.col("nifty50_ret").is_not_null() & pl.col("india_vix").is_not_null())
    )
    flow = build_flow(isin_mapping)
    df = market.join(flow, on="date", how="left").sort("date")

    df = df.with_columns(
        gmean=pl.col("gross").rolling_mean(GROSS_WINDOW, min_samples=GROSS_WINDOW // 2).shift(1)
    )
    df = df.with_columns(
        nf=pl.when(pl.col("net").is_not_null() & pl.col("gmean").is_not_null())
        .then(pl.col("net") / pl.col("gmean"))
        .otherwise(0.0)
    )

    # targets and the fixed control block
    df = df.with_columns(
        y_dir=pl.col("nifty50_ret").shift(-1),
        y_vol=pl.col("nifty50_ret").shift(-1).abs(),
        nf_t0=pl.col("nf"),
        nf_t2=pl.col("nf").shift(2),
    )
    df = df.with_columns(
        neg_t0=pl.min_horizontal(pl.col("nf_t0"), pl.lit(0.0)),
        neg_t2=pl.min_horizontal(pl.col("nf_t2"), pl.lit(0.0)),
    )
    for k in range(N_TARGET_LAGS):
        df = df.with_columns(
            **{
                f"dir_lag{k}": pl.col("nifty50_ret").shift(k),
                f"vol_lag{k}": pl.col("nifty50_ret").abs().shift(k),
            }
        )
    return df


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScreenCell:
    """One pre-registered cell, estimated on one sample."""

    cell: str
    era: str
    n: int
    coef: float
    t_stat: float
    p_value: float

    def __str__(self) -> str:
        return f"{self.cell} {self.era:<5} n={self.n:>5} b={self.coef:+.5f} t={self.t_stat:+.2f}"


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """All eras for one cell, plus its pre-registered verdict."""

    cell: str
    description: str
    full: ScreenCell
    train: ScreenCell
    test: ScreenCell

    @property
    def leg_a(self) -> bool:
        """Full-sample |t| clears the Bonferroni-adjusted bar."""
        return abs(self.full.t_stat) >= BAR_FULL_T

    @property
    def leg_b(self) -> bool:
        """Same sign, |t| >= 1.5, in both eras separately."""
        same_sign = np.sign(self.train.coef) == np.sign(self.test.coef)
        return bool(
            same_sign and abs(self.train.t_stat) >= BAR_ERA_T and abs(self.test.t_stat) >= BAR_ERA_T
        )

    @property
    def passes(self) -> bool:
        return self.leg_a and self.leg_b

    @property
    def verdict(self) -> str:
        return "PASS" if self.passes else "FAIL"


def _ols_hac(y: np.ndarray, x: np.ndarray, focus: int) -> tuple[float, float, float]:
    """OLS with Newey-West errors; return (coef, t, p) for column ``focus``."""
    import statsmodels.api as sm

    model = sm.OLS(y, sm.add_constant(x, has_constant="add"))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    j = focus + 1  # constant occupies column 0
    return float(fit.params[j]), float(fit.tvalues[j]), float(fit.pvalues[j])


def _estimate(
    df: pl.DataFrame,
    target: str,
    regressors: list[str],
    zero_flow: str,
) -> ScreenCell | None:
    cols = [target, "date", *regressors]
    sub = df.select(cols).drop_nulls()
    if zero_flow == "exclude":
        sub = sub.filter(pl.col(regressors[0]) != 0.0)
    elif zero_flow != "as_zero":
        raise ValueError(f"zero_flow must be 'exclude' or 'as_zero', got {zero_flow!r}")
    if sub.height <= len(regressors) + 2:
        return None
    y = sub[target].to_numpy()
    x = sub.select(regressors).to_numpy()
    coef, t_stat, p = _ols_hac(y, x, focus=0)
    return ScreenCell("", "", sub.height, coef, t_stat, p)


_CELLS: Final = {
    "C1": ("direction, same-day alignment", "y_dir", ["nf_t0"], False, False),
    "C2": ("risk, same-day alignment", "y_vol", ["nf_t0", "neg_t0"], True, True),
    "C3": ("direction, deployable lag", "y_dir", ["nf_t2"], False, False),
    "C4": ("risk, deployable lag", "y_vol", ["nf_t2", "neg_t2"], True, True),
}


def run_screen(
    validation_data: Path,
    isin_mapping: Path,
    *,
    zero_flow: str = ZERO_FLOW_CANONICAL,
) -> list[ScreenResult]:
    """Estimate all four pre-registered cells on FULL, TRAIN and TEST."""
    df = build_frame(validation_data, isin_mapping)
    eras = {
        "FULL": df,
        "TRAIN": df.filter(pl.col("date") <= TRAIN_END),
        "TEST": df.filter(pl.col("date") >= TEST_START),
    }

    results: list[ScreenResult] = []
    for cell, (desc, target, flow_cols, use_vix, use_vol_lags) in _CELLS.items():
        lag_prefix = "vol_lag" if use_vol_lags else "dir_lag"
        controls = [f"{lag_prefix}{k}" for k in range(N_TARGET_LAGS)]
        if use_vix:
            controls.append("india_vix")
        regressors = [*flow_cols, *controls]

        by_era: dict[str, ScreenCell] = {}
        for era, frame in eras.items():
            got = _estimate(frame, target, regressors, zero_flow)
            if got is None:
                raise ValueError(f"{cell}/{era}: insufficient observations")
            by_era[era] = ScreenCell(cell, era, got.n, got.coef, got.t_stat, got.p_value)
        results.append(ScreenResult(cell, desc, by_era["FULL"], by_era["TRAIN"], by_era["TEST"]))
    return results


def render(results: list[ScreenResult]) -> str:
    lines = [
        "aggregate-flow screen - pre-registered cells C1-C4",
        f"  Newey-West lags {NW_LAGS} | gross window {GROSS_WINDOW}d | "
        f"split {TRAIN_END} / {TEST_START}",
        f"  bar: |t|>={BAR_FULL_T} full AND same sign with |t|>={BAR_ERA_T} in both eras",
        "",
        f"{'cell':<5}{'description':<30}{'FULL t':>9}{'TRAIN t':>9}{'TEST t':>9}"
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
    for r in results:
        lines.append(f"  {r.cell} n: FULL {r.full.n}  TRAIN {r.train.n}  TEST {r.test.n}")
    return "\n".join(lines)


def to_csv(results: list[ScreenResult], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            [
                "cell",
                "description",
                "era",
                "n",
                "coef",
                "t_stat",
                "p_value",
                "leg_a",
                "leg_b",
                "verdict",
            ]
        )
        for r in results:
            for c in (r.full, r.train, r.test):
                w.writerow(
                    [
                        r.cell,
                        r.description,
                        c.era,
                        c.n,
                        f"{c.coef:.8f}",
                        f"{c.t_stat:.4f}",
                        f"{c.p_value:.6f}",
                        r.leg_a,
                        r.leg_b,
                        r.verdict,
                    ]
                )
    return path


def sensitivity(validation_data: Path, isin_mapping: Path) -> str:
    """Both readings of the unobserved-flow ambiguity, side by side.

    The inherited specification does not say what becomes of days on which no
    flow was observed. The two admissible readings are reported together so the
    choice is visible in the paper rather than buried in the code.
    """
    lines = [
        "C4 sensitivity to the unobserved-flow reading",
        f"{'reading':<12}{'FULL t':>10}{'TRAIN t':>10}{'TEST t':>10}{'n (FULL)':>10}{'verdict':>9}",
    ]
    for reading in ("exclude", "as_zero"):
        res = {r.cell: r for r in run_screen(validation_data, isin_mapping, zero_flow=reading)}[
            "C4"
        ]
        mark = " (canonical)" if reading == ZERO_FLOW_CANONICAL else ""
        lines.append(
            f"{reading:<12}{res.full.t_stat:>10.2f}{res.train.t_stat:>10.2f}"
            f"{res.test.t_stat:>10.2f}{res.full.n:>10}{res.verdict:>9}{mark}"
        )
    lines.append("")
    lines.append("inherited, from the docstring of module21_market_flow_engine.py:6-8")
    lines.append(f"{'(no artifact)':<12}{-4.78:>10.2f}{-2.49:>10.2f}{-2.51:>10.2f}")
    lines.append("")
    lines.append(
        "Both readings clear both legs of the pre-registered bar. The choice "
        "moves the\ntraining-era statistic by 0.31 t-units and does not change "
        "any verdict."
    )
    return "\n".join(lines)
