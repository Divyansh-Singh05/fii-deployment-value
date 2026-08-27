"""A4 - real forecast skill that a three-line rule already supplies."""

from __future__ import annotations

from convgap.application import Application, Layer, evidence_set
from convgap.benchmark import Detectability
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.extract import log_regex
from convgap.friction import FrictionLevel
from convgap.provenance import Source
from convgap.recipe import Recipe, RecipeKind
from convgap.registry import register
from convgap.role import Role

_HAZARD = "outputs/logs/20260824_232029_phase2_hazard.log"
_DECISION = "outputs/logs/20260824_232201_phase2_decision.log"
_P_HAZ = "src/fii/phase2/module16c_hazard.py"
_P_DEC = "src/fii/phase2/module16d_decision.py"


@register
class TrivialRuleDominance(Application):
    key = "A4"
    title = "Episode-end hazard model"
    layer = Layer.CLASSIFICATION
    role = Role.APPLICATION
    fails_at = FrictionLevel.COMPETITION
    mechanism = (
        "Skill is real and non-incremental. The walk-forward hazard beats an age-only "
        "Kaplan-Meier baseline on out-of-window log loss at a large paired t, and "
        "discriminates better out of sample than in, which rules out memorisation. It "
        "then loses head-to-head to a three-line age rule on the episodes both act "
        "upon. Both rules beat zero; only the paired comparison on shared events "
        "reveals the ordering, and a design that compared each against zero separately "
        "would have reported the model as the better of two positives."
    )

    def recipes(self) -> tuple[Recipe, ...]:
        return (
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="research",
                argv=["{python}", "pipeline.py", "--phase", "phase2"],
                produces=[_HAZARD, _DECISION],
                est_seconds=260,
                note="16C walk-forward discrete-time hazard and 16D decision layer. "
                "Both must run on the same rebuilt state object: the figures in the "
                "project documents predate the audit and are materially different.",
            ),
        )

    def significance(self) -> EvidenceSet:
        def _s(loc: str) -> Source:
            return Source(tree="research", artifact=_HAZARD, locator=loc, producer=_P_HAZ)

        stale = (
            "Project documents report AUC 0.797 with paired t = +41.6. Those come "
            "from a 2026-07-13 run on the pre-audit state object; Phase II was never "
            "re-executed after the audit rebuilt the states. On the current object "
            "the skill is materially smaller and the verdict is unchanged."
        )
        return evidence_set(
            "k=1 episode-end hazard vs age-only Kaplan-Meier, out-of-window",
            Evidence(
                0.647,
                Statistic.AUC,
                _s("k=1 pooled line, AUC"),
                era=Era.FULL,
                extractor=log_regex(r"k=1:.*?AUC ([\d.]+) \(KM"),
                note=stale,
            ),
            Evidence(
                0.569,
                Statistic.AUC,
                _s("k=1 pooled line, KM baseline AUC"),
                era=Era.FULL,
                extractor=log_regex(r"k=1:.*?AUC [\d.]+ \(KM ([\d.]+)\)"),
            ),
            Evidence(
                0.637,
                Statistic.AUC,
                _s("k=1 TRAIN row, AUC"),
                era=Era.TRAIN,
                extractor=log_regex(r"k=1:[^\n]*\n\s*TRAIN[^\n]*AUC ([\d.]+)"),
            ),
            Evidence(
                0.657,
                Statistic.AUC,
                _s("k=1 TEST row, AUC"),
                era=Era.TEST,
                extractor=log_regex(r"k=1:[^\n]*\n[^\n]*\n\s*TEST[^\n]*AUC ([\d.]+)"),
                note="Out of sample above in sample: not memorisation.",
            ),
            Evidence(
                14.42,
                Statistic.T_STAT,
                _s("k=1 pooled line, paired t vs KM"),
                era=Era.FULL,
                extractor=log_regex(r"k=1:.*?paired t \(vs KM\) \+([\d.]+)"),
            ),
        )

    def value(self) -> EvidenceSet:
        def _d(loc: str) -> Source:
            return Source(tree="research", artifact=_DECISION, locator=loc, producer=_P_DEC)

        return evidence_set(
            "16D decision layer, paired against an age-only anticipator",
            Evidence(
                -18.0,
                Statistic.BASIS_POINTS,
                _d("TEST block, MODEL gain"),
                era=Era.TEST,
                extractor=log_regex(r"TEST:\s*MODEL n=\d+\s+gain ([-+\d]+)"),
                note="Per episode; 95% CI [-52, +19], covering zero.",
            ),
            Evidence(
                16.0,
                Statistic.BASIS_POINTS,
                _d("TEST block, KM rule gain"),
                era=Era.TEST,
                extractor=log_regex(r"TEST:.*?KM\s+n=\d+\s+gain \+([\d]+)", dotall=True),
            ),
            Evidence(
                -31.0,
                Statistic.BASIS_POINTS,
                _d("TEST block, paired model-KM difference on common episodes"),
                era=Era.TEST,
                n=402,
                extractor=log_regex(
                    r"TEST:.*?paired model-KM on \d+ common episodes: "
                    r"d=([-+\d]+)bp",
                    dotall=True,
                ),
                note="Stage verdict: 16D does not clear its bar.",
            ),
            Evidence(
                -1.85,
                Statistic.T_STAT,
                _d("TEST block, paired model-KM t-statistic"),
                era=Era.TEST,
                extractor=log_regex(
                    r"TEST:.*?paired model-KM on \d+ common episodes: "
                    r"d=[-+\d]+bp, t=([-+\d.]+)",
                    dotall=True,
                ),
            ),
        )

    def detectability(self) -> Detectability:
        return Detectability(
            implied=16.0,
            observed=-18.0,
            unit="bp per episode",
            method=(
                "Implied = the gain the trivial age rule actually delivers in the "
                "same era on the same episodes (+16 bp), taken as the value "
                "available without the dataset at all. Observed = the model's own "
                "gain (-18 bp). The ratio is negative because the model does not "
                "merely fail to add to the trivial rule; it underperforms it."
            ),
        )
