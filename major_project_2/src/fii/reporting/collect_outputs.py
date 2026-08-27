# ============================================================================
# COLLECT OUTPUTS — populate every outputs/ bucket with its real content.
#
# The research stages print their evidence (captured per-run in
# outputs/logs) and write data artifacts into data/. This stage gathers
# everything a reader of the repo expects to find under outputs/:
#
#   predictions/             model outputs (calibrated states, canonical v3)
#   trained_models/          backbone params + overlay thresholds (JSON)
#   validation/              latest log per validation/backtest stage,
#                            under a stable name (the human-readable report)
#   descriptive_statistics/  census + signatures + mechanism arc
#   metrics/                 backtest metrics + machine-readable run summary
#   regression_outputs/      full PanelOLS coefficient tables (exhibits)
# ============================================================================
import json
import re
import shutil

import polars as pl

from fii.paths import (DESCRIPTIVES, ISIN_MAPPING, LOGS, METRICS,
                       PREDICTIONS, TABLES, TRAINED_MODELS,
                       VALIDATION_DATA, VALIDATION_OUT,
                       ensure_output_tree)
from fii.stages.registry import STAGES

ensure_output_tree()
copied = []


def cp(src, dst_dir, name=None):
    if src.exists():
        dst = dst_dir / (name or src.name)
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(dst_dir.parent.parent)))
    else:
        print(f"  !! missing {src}")


# ---- predictions: the model's stock-day output ------------------------------
cp(ISIN_MAPPING / "stockday_states_calibrated.parquet", PREDICTIONS)
cp(VALIDATION_DATA / "states_v3.parquet", PREDICTIONS)

# ---- metrics: backtest metrics + run summary ---------------------------------
cp(TABLES / "T6_backtest_metrics.csv", METRICS, "backtest_metrics.csv")
runs: dict[str, dict] = {}
pat = re.compile(r"^(\d{8}_\d{6})_(.+)\.log$")
for f in sorted(LOGS.glob("*.log")):
    m = pat.match(f.name)
    if m:
        ts, stage = m.groups()
        runs[stage] = {"last_run": ts, "log": f.name,
                       "bytes": f.stat().st_size}
(METRICS / "run_summary.json").write_text(json.dumps(
    {"stages_run": runs, "n_stages_defined": len(STAGES)}, indent=2))
copied.append("metrics/run_summary.json")

# ---- validation: latest log per evidence stage, stable names -----------------
EVIDENCE_PHASES = {"validation", "backtest", "model", "audit"}
for s in STAGES:
    if s.phase in EVIDENCE_PHASES and s.name in runs:
        cp(LOGS / runs[s.name]["log"], VALIDATION_OUT, f"{s.name}.log")

# ---- descriptive statistics ---------------------------------------------------
cp(TABLES / "T2_archetype_census.csv", DESCRIPTIVES,
   "archetype_census.csv")
cp(TABLES / "T3_mechanism_arc.csv", DESCRIPTIVES,
   "episode_arc_by_archetype.csv")
st = pl.read_parquet(ISIN_MAPPING / "stockday_states_calibrated.parquet")
fcols = [c for c in st.columns if c.startswith("F_")]
if fcols:
    (st.group_by("era", "archetype")
       .agg([pl.len().alias("n")]
            + [pl.col(c).mean().round(3) for c in fcols])
       .sort(["archetype", "era"])
       .write_csv(DESCRIPTIVES / "archetype_feature_signatures.csv"))
    copied.append("descriptive_statistics/archetype_feature_signatures.csv")

print("collected:")
for c in copied:
    print("  ", c)
missing_models = [f for f in ("hmm_backbone_params.json",
                              "overlay_thresholds.json")
                  if not (TRAINED_MODELS / f).exists()]
if missing_models:
    print("NOTE: trained_models missing", missing_models,
          "— rerun hmm_train_oos / threshold_calibration once")
print("DONE — outputs/ buckets populated.")
