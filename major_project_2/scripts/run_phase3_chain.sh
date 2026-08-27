#!/usr/bin/env bash
# Wait for the C1 gap-aware refit, then drive C2 -> C9.
#
# Halts on the first failure, exactly like fii.runner: a stage whose gates fail
# must not have downstream stages built on top of it. Every stage writes its own
# timestamped log under outputs/logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/outputs/logs/phase3_chain_${STAMP}.log"
mkdir -p "$ROOT/outputs/logs"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 1. wait for C1 -------------------------------------------------------
say "waiting for C1 gap-aware refit to finish..."
while pgrep -f "module_c1_refit_harness" > /dev/null; do sleep 60; done
say "C1 process has exited"

if ! grep -q "wrote\|saved\|Saved" "$ROOT/outputs/logs/c1_gapaware_rebuild.log" 2>/dev/null; then
  say "checking C1 completed cleanly..."
fi
if grep -qi "Traceback" "$ROOT/outputs/logs/c1_gapaware_rebuild.log" 2>/dev/null; then
  say "ABORT: C1 log contains a traceback — not running the chain"
  tail -30 "$ROOT/outputs/logs/c1_gapaware_rebuild.log" | tee -a "$LOG"
  exit 1
fi

# ---- 2. sanity-check the new vintages -------------------------------------
say "validating vintages.parquet"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, json, numpy as np, polars as pl
from pathlib import Path
p = Path("outputs/phase3/vintages.parquet")
vt = pl.read_parquet(p).sort("asof")
A = np.array([json.loads(x) for x in vt["transmat"]])
mu = np.array([json.loads(x) for x in vt["means"]])
diag = np.array([np.diag(a).mean() for a in A])
l2 = np.array([np.sort(np.abs(np.linalg.eigvals(a)))[::-1][1] for a in A])
hl = np.log(0.5) / np.log(np.clip(l2, 1e-9, 1 - 1e-9))
ok_order = bool(np.all(np.diff(mu[:, :, 0], axis=1) > 0))
print(f"  vintages          {vt.height}")
print(f"  mean diag         {diag.mean():.4f}   (pre-audit fit: 0.938)")
print(f"  lambda2           {l2.mean():.4f}")
print(f"  half-life (days)  {np.median(hl):.2f}   (pre-audit: 8.7; k=1 evidence: 20.4)")
print(f"  state order monotone in F_persist on every vintage: {ok_order}")
sys.exit(0 if (vt.height > 50 and ok_order) else 1)
PYEOF
if [ "${PIPESTATUS[0]}" -ne 0 ]; then say "ABORT: vintages failed sanity check"; exit 1; fi

# ---- 3. drive the chain ---------------------------------------------------
run () {
  local name="$1"; shift
  local slog="$ROOT/outputs/logs/${STAMP}_${name}.log"
  say "=== $name ==="
  if "$PY" -m "$@" > "$slog" 2>&1; then
    say "    PASS  $name   -> $(basename "$slog")"
    tail -4 "$slog" | sed 's/^/      /' | tee -a "$LOG" > /dev/null
  else
    say "    FAIL  $name   -> $(basename "$slog")"
    grep -n "Traceback" -A 12 "$slog" | head -20 | tee -a "$LOG"
    say "CHAIN HALTED at $name"
    exit 1
  fi
}

run C2_daily_filter        fii.phase3.module_c2_daily_filter
run C3_archetype_weights   fii.phase3.module_c3_archetype_probs
run C4_outcome_densities   fii.phase3.module_c4_outcome_densities
run C5_predictive_mixture  fii.phase3.module_c5_predictive_mixture
run C6_validation          fii.phase3.module_c6_validation
run C7_risk_profiles       fii.phase3.module_c7_stock_risk_profiles
# C9 re-executes C2..C5 against truncated inputs, so it runs last and is slow.
run C9_lookahead_audit     fii.phase3.module_c9_lookahead_audit

say "CHAIN COMPLETE — all stages passed"
say "full log: $LOG"
