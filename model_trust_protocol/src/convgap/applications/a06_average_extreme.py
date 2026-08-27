"""A6 - significant in the extremes, absent on average."""

from __future__ import annotations

from convgap.application import Application, Layer, evidence_set
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.extract import log_regex
from convgap.friction import FrictionLevel
from convgap.provenance import Source
from convgap.recipe import Recipe, RecipeKind
from convgap.registry import register
from convgap.role import Role

_LOG = "outputs/logs/20260824_231827_incremental_value.log"
_PRODUCER = "src/fii/validation/module13b_incremental_value.py"


def _src(locator: str) -> Source:
    return Source(tree="research", artifact=_LOG, locator=locator, producer=_PRODUCER)


@register
class AverageExtremeDissociation(Application):
    key = "A6"
    title = "Participant-composition feature block"
    layer = Layer.ATTRIBUTION
    role = Role.APPLICATION
    fails_at = FrictionLevel.DETECTABILITY
    mechanism = (
        "Average and extreme carry different answers, and the metric chosen decides "
        "which one is reported. The block's top-minus-bottom quintile spread is "
        "significant and stable on non-overlapping episodes, while its incremental "
        "information coefficient over conventional flow features is indistinguishable "
        "from zero. Both are computed on the same block, the same panel and the same "
        "run. A single summary statistic reports whichever half it happens to weight, "
        "and the incremental coefficient - the quantity a user of the data would "
        "consult - weights the half that is empty."
    )

    def recipes(self) -> tuple[Recipe, ...]:
        return (
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="research",
                argv=["{python}", "pipeline.py", "--stage", "incremental_value"],
                produces=[_LOG],
                est_seconds=112,
                note="Episode-level flow-magnitude controls plus a gradient-boosted "
                "challenger fitted with and without the composition block.",
            ),
        )

    def significance(self) -> EvidenceSet:
        return evidence_set(
            "composition-only model, Q5-Q1 forward-return spread",
            Evidence(
                74.8,
                Statistic.BASIS_POINTS,
                _src("row 'COMP only', Q5-Q1 spread in bp"),
                era=Era.FULL,
                extractor=log_regex(r"COMP only\s*:.*?Q5-Q1 \+([\d.]+)bp"),
            ),
            Evidence(
                2.86,
                Statistic.T_STAT,
                _src("row 'COMP only', non-overlapping-episode t on the spread"),
                era=Era.FULL,
                extractor=log_regex(r"COMP only\s*:.*?non-overlap t=\+([\d.]+)"),
            ),
        )

    def value(self) -> EvidenceSet:
        return evidence_set(
            "incremental information coefficient over conventional flow features",
            Evidence(
                0.0012,
                Statistic.IC,
                _src("line 'dIC (full - std)', point estimate"),
                era=Era.FULL,
                extractor=log_regex(r"dIC \(full - std\) = \+([\d.]+)"),
                note="Pre-registered bar 0.005.",
            ),
            Evidence(
                0.67,
                Statistic.T_STAT,
                _src("line 'dIC (full - std)', paired daily t"),
                era=Era.FULL,
                extractor=log_regex(r"dIC \(full - std\).*?paired daily t = \+([\d.]+)"),
                note="Pre-registered bar t >= 2. Stage verdict: NOT ESTABLISHED.",
            ),
        )
