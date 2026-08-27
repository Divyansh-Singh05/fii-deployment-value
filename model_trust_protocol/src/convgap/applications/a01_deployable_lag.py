"""A1 - institutional share of turnover: information that decays before it can be used."""

from __future__ import annotations

from convgap.application import Application, Layer, evidence_set
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.extract import Extractor, log_regex
from convgap.friction import FrictionLevel
from convgap.provenance import Source
from convgap.recipe import Recipe, RecipeKind
from convgap.registry import register
from convgap.role import Role

_LOG = "outputs/logs/20260824_231214_institutional_share.log"
_PRODUCER = "src/fii/validation/module19_institutional_share.py"


def _src(locator: str) -> Source:
    return Source(tree="research", artifact=_LOG, locator=locator, producer=_PRODUCER)


def _row(label: str, era: str) -> Extractor:
    """Second numeric column of a results row is the t-statistic."""
    return log_regex(rf"{label}\s+\[{era}\]\s+[-+\d.]+\s+([-+\d.]+)")


@register
class DeployableLag(Application):
    key = "A1"
    title = "Institutional share of turnover -> next-day volatility"
    layer = Layer.REGRESSION
    role = Role.APPLICATION
    fails_at = FrictionLevel.AVAILABILITY
    mechanism = (
        "The effect is measured contemporaneously and is absent at the lag a decision "
        "could act on. Significance is established on an information set that does not "
        "exist when the forecast must be issued. Nothing else changes between the two "
        "specifications - same panel, same fixed effects, same controls, same "
        "clustering - so the collapse locates the finding as economics rather than as "
        "a signal, which is the reading the source module itself states."
    )

    def recipes(self) -> tuple[Recipe, ...]:
        return (
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="research",
                argv=["{python}", "pipeline.py", "--stage", "institutional_share"],
                produces=[_LOG],
                est_seconds=72,
                note="Panel regressions of z^2 on institutional share of\n"
                "turnover, 577,245 stock-days.",
            ),
        )

    def significance(self) -> EvidenceSet:
        loc = "section 1, row 'stock + date FE', t column"
        return evidence_set(
            "z^2 on institutional share, stock + date FE, log(turnover) controlled",
            Evidence(
                -5.53,
                Statistic.T_STAT,
                _src(loc),
                era=Era.FULL,
                n=577_245,
                extractor=_row(r"stock \+ date FE", "FULL"),
            ),
            Evidence(
                -4.42,
                Statistic.T_STAT,
                _src(loc),
                era=Era.TRAIN,
                n=288_489,
                extractor=_row(r"stock \+ date FE", "TRAIN"),
            ),
            Evidence(
                -3.25,
                Statistic.T_STAT,
                _src(loc),
                era=Era.TEST,
                n=288_756,
                extractor=_row(r"stock \+ date FE", "TEST"),
            ),
        )

    def value(self) -> EvidenceSet:
        loc = "section 1, DEPLOYABILITY block, row 'share at t-2', t column"
        return evidence_set(
            "identical specification, share known only at t-2",
            Evidence(
                -0.32,
                Statistic.T_STAT,
                _src(loc),
                era=Era.FULL,
                n=575_304,
                extractor=_row("share at t-2", "FULL"),
            ),
            Evidence(
                -0.30,
                Statistic.T_STAT,
                _src(loc),
                era=Era.TRAIN,
                n=287_365,
                extractor=_row("share at t-2", "TRAIN"),
            ),
            Evidence(
                -0.13,
                Statistic.T_STAT,
                _src(loc),
                era=Era.TEST,
                n=287_939,
                extractor=_row("share at t-2", "TEST"),
            ),
        )
