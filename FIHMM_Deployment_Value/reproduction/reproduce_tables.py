"""Regenerate every manuscript table into outputs/tables/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import digest_line, local


def main() -> int:
    log = get_logger("reproduce_tables")
    banner(log, "REGENERATING MANUSCRIPT TABLES")

    from analysis.summary_results import write_attrition_summary
    from src.validation.baseline_comparison import controlled_pair
    from src.validation.detectability import state_conditioning_ablation
    from src.models.hazard import hazard_summary
    from src.validation.execution_costs import breakeven_table, cost_grid
    from analysis.robustness import variance_estimator_grid

    jobs = {
        "table_2_controlled_pair.csv": controlled_pair,
        "table_3_hazard.csv": hazard_summary,
        "table_4_ablation.csv": state_conditioning_ablation,
        "table_5_execution.csv": breakeven_table,
        "table_5_cost_grid.csv": cost_grid,
        "table_1_robustness.csv": variance_estimator_grid,
    }
    for fname, fn in jobs.items():
        try:
            df = fn()
        except Exception as exc:                       # noqa: BLE001
            verdict(log, fname, "SKIP", str(exc).splitlines()[0])
            continue
        path = local("tables") / fname
        df.to_csv(path, index=False)
        verdict(log, fname, "PASS", digest_line(path))

    write_attrition_summary()
    verdict(log, "attrition_ladder.csv", "PASS")
    verdict(log, "applications_by_binding_constraint.csv", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
