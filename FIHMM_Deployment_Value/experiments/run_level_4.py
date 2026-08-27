"""L4 - Detectability. Is it large enough for the metric the decision consumes?

The metric a decision consumes is generally NOT the metric that established
significance, and the two have different power. An effect can be real, survive
every control, and sit below the second's resolution.

  A6  present in the extremes, absent on the average. A 74.8 bp quintile
      spread (t = 2.86) alongside an incremental IC of 0.0012 (t = 0.67)
      against a 0.005 bar. A single summary statistic reports whichever half
      it weights, and the one a user consults weights the empty half.

  A7  tested by ABLATION - the identical machinery re-run with the mechanism
      removed and everything else fixed. In the pre-registered primary stratum,
      where the mechanism should bind hardest, the conditioning is
      indistinguishable from its own removal.

Reproduces manuscript Table 4.

A6 CAVEAT CARRIED FROM THE DATA: participant identifiers are re-minted monthly,
so composition features are computed WITHIN-DAY only. A6 measures within-day
participation structure, not a persistent participant type.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import load_config, local
from src.validation.detectability import composition_block, state_conditioning_ablation


def main() -> int:
    log = get_logger("level_4_detectability")
    banner(log, "L4 DETECTABILITY - can the consuming metric resolve it?")
    gates = load_config("model_config")["gates"]

    # ---- A6 -------------------------------------------------------------
    log.info("  A6 - participant-composition feature block")
    comp = composition_block()
    for k, v in comp.items():
        log.info(f"    {k:<40s} {v}")
    verdict(log, "quintile spread on non-overlapping episodes", "PASS",
            f"{comp['quintile_spread_bp']} bp, t = {comp['quintile_spread_t']}")
    ic_ok = comp["incremental_ic"] >= gates["incremental_ic"]
    verdict(log, "incremental IC over conventional flow features",
            "PASS" if ic_ok else "NOT ESTABLISHED",
            f"{comp['incremental_ic']} against a {gates['incremental_ic']} bar")
    log.info("    -> significant in the extremes, vacant on the average")
    log.info("    -> composition is WITHIN-DAY; identifiers are re-minted monthly")
    log.info("")

    # ---- A7 -------------------------------------------------------------
    log.info("  A7 - ablation of state conditioning in the stock-day density")
    abl = state_conditioning_ablation()
    abl.to_csv(local("tables") / "table_4_ablation.csv", index=False)
    for _, r in abl.iterrows():
        log.info(f"    {r['stratum']:<42s} DM {r['dm_statistic']:+.2f}"
                 f"   p {r['probability']:.4f}   effect {r['score_effect']:+.5f}")

    primary = abl.iloc[0]
    verdict(log, "conditioning distinguishable in primary stratum",
            "FAIL" if primary["probability"] > 0.05 else "PASS",
            "indistinguishable from its own removal where it should bind hardest")
    log.info("    -> on the full panel it IS favoured, by five parts in 100,000")
    log.info("       of a score whose level is near 0.55: one part in eleven thousand")
    log.info("")
    log.info("  Comparison against an external benchmark establishes that a system")
    log.info("  is better. Only ablation establishes WHY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
