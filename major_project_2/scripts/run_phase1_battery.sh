#!/usr/bin/env bash
# Re-run the Phase I economic-validation battery and the backtests against the
# REBUILT research object (corrected concentration axis + point-in-time
# identity).
#
# DOES NOT HALT ON FAILURE, unlike fii.runner. That is deliberate and is the
# whole point of this run: these stages carry gates that were pre-registered
# against the pre-audit object, so a failure here is a RESULT ("the finding
# does not reproduce on the corrected axis"), not a broken pipeline. Halting on
# the first one would throw away every downstream answer. Each stage's exit
# status and full log are recorded and the run continues.
#
# Stages are ordered by real data dependency, so a downstream failure caused by
# an upstream one is visible as such.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/outputs/logs/phase1_battery_${STAMP}.log"
RESULTS="$ROOT/outputs/logs/phase1_battery_${STAMP}_results.tsv"
mkdir -p "$ROOT/outputs/logs"
printf 'stage\tstatus\tseconds\tlog\n' > "$RESULTS"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

STAGES=(
  event_study
  deal_corroboration
  liquidity_shock
  panel_regression
  robustness
  gbt_challenger
  demeaning_check
  flow_innovation
  vix_lambda
  pin_model
  backbone_ablation
  incremental_value
  recon_exclusion
  skeptic_tests
  engine_gates
  bt_baselines
  bt_hmm_twins
  bt_gross_diagnosis
  bt_style_switch
)

say "Phase I battery on the rebuilt object — ${#STAGES[@]} stages, no halt on failure"
say "results table: $(basename "$RESULTS")"

pass=0; fail=0
for st in "${STAGES[@]}"; do
  slog="$ROOT/outputs/logs/${STAMP}_p1_${st}.log"
  t0=$(date +%s)
  say "=== $st ==="
  if "$PY" pipeline.py --stage "$st" > "$slog" 2>&1; then
    status=PASS; pass=$((pass+1))
  else
    status=FAIL; fail=$((fail+1))
  fi
  t1=$(date +%s); secs=$((t1-t0))
  printf '%s\t%s\t%s\t%s\n' "$st" "$status" "$secs" "$(basename "$slog")" >> "$RESULTS"
  say "    $status  ${secs}s  -> $(basename "$slog")"
  if [ "$status" = FAIL ]; then
    grep -n "Traceback" -A 8 "$slog" | head -12 | sed 's/^/      /' | tee -a "$LOG" > /dev/null
  fi
done

say "BATTERY COMPLETE — $pass passed, $fail failed of ${#STAGES[@]}"
say "full log: $LOG"
column -t "$RESULTS" | tee -a "$LOG"
