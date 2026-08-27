"""A5 - a real cross-sectional signal that execution cost consumes.

Source history: see ``docs/verification_log.md`` entries V1, V1a and V1b. This
application was promoted twice. The first promotion pointed at
``outputs/metrics/backtest_metrics.csv``, which is a *copy* written by a
reporting step and had fallen six weeks behind its origin; both the digest check
and the value check passed on it. The freshness gate exists because of that.
"""

from __future__ import annotations

from convgap.application import Application, Layer, evidence_set
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.extract import Extractor, csv_cell
from convgap.friction import FrictionLevel
from convgap.provenance import Source
from convgap.recipe import Recipe, RecipeKind
from convgap.registry import register
from convgap.role import Role

#: The exhibit table written directly by the reporting stage. The near-identical
#: file at ``outputs/metrics/backtest_metrics.csv`` is a copy of this one and is
#: deliberately not used.
_T6 = "outputs/tables/T6_backtest_metrics.csv"
_PRODUCER = "src/fii/reporting/make_exhibits.py"

#: Best out-of-sample gross book on the current run. Named rather than inferred:
#: "best" is a selection and must be declared.
_BOOK = "S3_PROXY"


def _src(locator: str) -> Source:
    return Source(tree="research", artifact=_T6, locator=locator, producer=_PRODUCER)


def _cell(era: str, cost: str, column: str) -> Extractor:
    return csv_cell({"strategy": _BOOK, "era": era, "cost_bps": cost}, column)


@register
class CostWall(Application):
    key = "A5"
    title = "Cross-sectional concentration book"
    layer = Layer.PORTFOLIO
    role = Role.APPLICATION
    fails_at = FrictionLevel.EXECUTION
    mechanism = (
        "The record's information survives every earlier level and is consumed by the "
        "cost of acting on it. The book carries no fitted model at all - it is a "
        "mechanical proxy for the concentration measure - so what is being priced is "
        "the data itself rather than an architecture. The informative quantity is the "
        "breakeven one-way cost, which is a property of the strategy; a net Sharpe at "
        "an assumed cost is a property of the assumption, and a reader whose costs "
        "differ from ours can use the first and cannot use the second."
    )

    def recipes(self) -> tuple[Recipe, ...]:
        return (
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="research",
                argv=["{python}", "pipeline.py", "--phase", "backtest"],
                produces=["outputs/tables/T6_backtest_metrics.csv"],
                est_seconds=65,
                note="Backtest phase writes bt12_*.parquet; the exhibits phase\n"
                "then rebuilds T6 from them, so both must run in that order.",
            ),
        )

    def significance(self) -> EvidenceSet:
        return evidence_set(
            f"gross Sharpe, {_BOOK}, zero cost",
            Evidence(
                1.36,
                Statistic.SHARPE,
                _src(f"row strategy={_BOOK}, era=TRAIN, cost_bps=0.0, column sharpe"),
                era=Era.TRAIN,
                n=2530,
                extractor=_cell("TRAIN", "0.0", "sharpe"),
            ),
            Evidence(
                1.51,
                Statistic.SHARPE,
                _src(f"row strategy={_BOOK}, era=TEST, cost_bps=0.0, column sharpe"),
                era=Era.TEST,
                n=920,
                extractor=_cell("TEST", "0.0", "sharpe"),
                note="Highest gross Sharpe of the eight books in the test era.",
            ),
        )

    def value(self) -> EvidenceSet:
        return evidence_set(
            "breakeven one-way cost, and net Sharpe at 15 bp",
            Evidence(
                7.33,
                Statistic.COST_BPS,
                _src(
                    f"row strategy={_BOOK}, era=TRAIN, cost_bps=0.0, column margin_bps "
                    "(gross margin per unit traded = the one-way cost at which net "
                    "profit is zero)"
                ),
                era=Era.TRAIN,
                extractor=_cell("TRAIN", "0.0", "margin_bps"),
            ),
            Evidence(
                7.41,
                Statistic.COST_BPS,
                _src(f"row strategy={_BOOK}, era=TEST, cost_bps=0.0, column margin_bps"),
                era=Era.TEST,
                extractor=_cell("TEST", "0.0", "margin_bps"),
                note="Roughly half the 15 bp institutional one-way cost assumed.",
            ),
            Evidence(
                -1.42,
                Statistic.SHARPE,
                _src(f"row strategy={_BOOK}, era=TRAIN, cost_bps=15.0, column sharpe"),
                era=Era.TRAIN,
                extractor=_cell("TRAIN", "15.0", "sharpe"),
            ),
            Evidence(
                -1.53,
                Statistic.SHARPE,
                _src(f"row strategy={_BOOK}, era=TEST, cost_bps=15.0, column sharpe"),
                era=Era.TEST,
                extractor=_cell("TEST", "15.0", "sharpe"),
            ),
        )
