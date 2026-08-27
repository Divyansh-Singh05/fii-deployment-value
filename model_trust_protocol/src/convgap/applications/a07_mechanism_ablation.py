"""A7 - a statistically real conditioning variable that the density cannot use."""

from __future__ import annotations

from convgap.application import Application, Layer, evidence_set
from convgap.benchmark import Detectability
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.extract import log_regex, parquet_cell
from convgap.friction import FrictionLevel
from convgap.provenance import Source
from convgap.recipe import Recipe, RecipeKind
from convgap.registry import register
from convgap.role import Role

_PERM = "outputs/logs/20260824_232504_model_descriptives.log"
_P_PERM = "src/fii/models/hmm_stages/module3c_descriptive_stats.py"
_C6 = "outputs/phase3/c6_results.parquet"

#: The pre-registered primary stratum: stock-days on which archetype identity is
#: genuinely uncertain. If the conditioning carries information anywhere, it is
#: here, which is why the pre-registration nominated it.
_PRIMARY = "PRIMARY  pa_max<0.70"
_FULL = "SECONDARY full panel"


def _ablation(stratum: str, horizon: int, column: str) -> object:
    return parquet_cell(
        {"test": "DM", "a": "soft_roll", "b": "clim_roll", "stratum": stratum, "horizon": horizon},
        column,
    )


@register
class MechanismAblation(Application):
    key = "A7"
    title = "Flow-regime conditioning in the stock-day predictive density"
    layer = Layer.DENSITY
    role = Role.APPLICATION
    fails_at = FrictionLevel.DETECTABILITY
    mechanism = (
        "The conditioning variable is established as real by a test that never sees a "
        "price - the labels chain into episodes far beyond a within-unit shuffled "
        "null, in both eras - and the density built on it is indistinguishable from "
        "the identical machinery with the conditioning removed, in the very stratum "
        "the pre-registration nominated as the place it should matter most. On the "
        "full panel the contrast is statistically significant and economically "
        "vacant: about five parts in a hundred thousand of a score whose level is "
        "around 0.55. Comparison against an external benchmark establishes that a "
        "system is better; only ablation establishes why, and the answer here is the "
        "empirical shape of the distribution rather than the flow record."
    )

    def recipes(self) -> tuple[Recipe, ...]:
        return (
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="engine",
                argv=["./run_all.sh"],
                produces=[_C6],
                est_seconds=430,
                note="Rebuilds the stock-day engine chain s06-s13 from the sealed "
                "vintages, including the C6 pre-registered validation that carries "
                "the ablation.",
            ),
            Recipe(
                kind=RecipeKind.PIPELINE_STAGE,
                tree="research",
                argv=["{python}", "pipeline.py", "--stage", "model_descriptives"],
                produces=[_PERM],
                est_seconds=90,
                note="Supplies the significance half: the episode-clustering "
                "permutation test, which never sees a price.",
            ),
        )

    def significance(self) -> EvidenceSet:
        def _p(loc: str) -> Source:
            return Source(tree="research", artifact=_PERM, locator=loc, producer=_P_PERM)

        return evidence_set(
            "episode clustering vs a within-unit shuffled null (ratio of mean run)",
            Evidence(
                2.58,
                Statistic.RATIO,
                _p("permutation test block, TRAIN ratio"),
                era=Era.TRAIN,
                p_value=0.005,
                extractor=log_regex(r"TRAIN: observed mean run.*?ratio = ([\d.]+)x"),
                note="The null preserves each unit's label count and destroys "
                "only temporal adjacency, so it isolates clustering from "
                "prevalence.",
            ),
            Evidence(
                2.64,
                Statistic.RATIO,
                _p("permutation test block, TEST ratio"),
                era=Era.TEST,
                p_value=0.005,
                extractor=log_regex(r"TEST: observed mean run.*?ratio = ([\d.]+)x"),
            ),
            Evidence(
                3.61,
                Statistic.RATIO,
                _p("permutation block, TRAIN observed mean run length, days"),
                era=Era.TRAIN,
                extractor=log_regex(r"TRAIN: observed mean run = ([\d.]+)d"),
            ),
            Evidence(
                1.40,
                Statistic.RATIO,
                _p("permutation block, TRAIN shuffled-null mean run, days"),
                era=Era.TRAIN,
                extractor=log_regex(r"TRAIN: observed mean run.*?null = ([\d.]+) "),
            ),
            Evidence(
                4.05,
                Statistic.RATIO,
                _p("permutation block, TEST observed mean run length, days"),
                era=Era.TEST,
                extractor=log_regex(r"TEST: observed mean run = ([\d.]+)d"),
            ),
            Evidence(
                1.53,
                Statistic.RATIO,
                _p("permutation block, TEST shuffled-null mean run, days"),
                era=Era.TEST,
                extractor=log_regex(r"TEST: observed mean run.*?null = ([\d.]+) "),
            ),
        )

    def value(self) -> EvidenceSet:
        def _c(loc: str) -> Source:
            return Source(tree="engine", artifact=_C6, locator=loc)

        return evidence_set(
            "C6 ablation: archetype-conditional mixture vs conditioning removed",
            Evidence(
                -0.821,
                Statistic.DM_STAT,
                _c(f"test=DM, a=soft_roll, b=clim_roll, stratum='{_PRIMARY}', h=1"),
                era=Era.FULL,
                n=1984,
                extractor=_ablation(_PRIMARY, 1, "stat"),  # type: ignore[arg-type]
                p_value=0.4116,
                note="Pre-registered primary stratum, p = 0.41: indistinguishable "
                "from zero where the mechanism should bind hardest.",
            ),
            Evidence(
                2.903,
                Statistic.DM_STAT,
                _c(f"test=DM, a=soft_roll, b=clim_roll, stratum='{_FULL}', h=1"),
                era=Era.FULL,
                n=1989,
                extractor=_ablation(_FULL, 1, "stat"),  # type: ignore[arg-type]
                p_value=0.0037,
                note="Full panel, p = 0.0037. Statistically favours conditioning.",
            ),
            Evidence(
                5.0e-5,
                Statistic.CRPS,
                _c(f"test=DM, a=soft_roll, b=clim_roll, stratum='{_FULL}', h=1, column effect"),
                era=Era.FULL,
                extractor=_ablation(_FULL, 1, "effect"),  # type: ignore[arg-type]
                rtol=2e-2,
                note="The magnitude behind that significance: 5e-5 of a score "
                "whose level is about 0.55.",
            ),
            Evidence(
                -4.2e-5,
                Statistic.CRPS,
                _c(f"test=DM, a=soft_roll, b=clim_roll, stratum='{_PRIMARY}', h=1, column effect"),
                era=Era.FULL,
                extractor=_ablation(_PRIMARY, 1, "effect"),  # type: ignore[arg-type]
                rtol=5e-2,
                note="Primary stratum magnitude, reported beside its statistic so "
                "the table does not carry an undeclared figure.",
            ),
        )

    def detectability(self) -> Detectability:
        return Detectability(
            implied=0.0055,
            observed=5.0e-5,
            unit="CRPS",
            method=(
                "Implied = 1% of the baseline mean CRPS of roughly 0.55, taken as a "
                "deliberately modest threshold for an improvement a user of a risk "
                "density would notice. Observed = the ablation's own point estimate "
                "on the full panel. The ratio states how much of even that modest "
                "threshold the conditioning delivers."
            ),
        )
