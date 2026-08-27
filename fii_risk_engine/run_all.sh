#!/usr/bin/env bash
# Serve-path rebuild: everything downstream of the fitted vintages.
#
# The vintages (outputs/phase3/vintages.parquet) ship with this project --
# they are the fitted model. Re-fitting them from the flow panel is s05 and
# takes ~45 min; see README. Everything below rebuilds in about seven minutes
# and is what actually produces the risk numbers.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

# Preflight. `python` is not on PATH on a stock macOS, and a bare `python3`
# usually has none of the dependencies — both of which used to surface as an
# obscure failure partway through a stage rather than as a clear message here.
if ! command -v "$PY" > /dev/null 2>&1; then
  echo "ERROR: interpreter '$PY' not found."
  echo "  Set one explicitly:  PYTHON=/path/to/python ./run_all.sh"
  exit 1
fi
# Real imports, not importlib.util.find_spec — `import importlib` does not
# expose `.util`, so a find_spec probe dies silently and the check passes
# vacuously. Only the modules THIS chain imports are required; linearmodels is
# used by s04 alone, which is not part of the serve path.
MISSING=$("$PY" - <<'PYCHECK' 2>/dev/null
bad = []
for m in ("numpy", "polars", "scipy"):
    try:
        __import__(m)
    except Exception:
        bad.append(m)
print(" ".join(bad))
PYCHECK
) || MISSING="(could not run $PY)"
if [ -n "${MISSING:-}" ]; then
  echo "ERROR: '$PY' is missing required packages: $MISSING"
  echo "  Either install them:   $PY -m pip install -r requirements.txt"
  echo "  or point at a venv:    PYTHON=/path/to/venv/bin/python ./run_all.sh"
  exit 1
fi
if [ ! -f outputs/phase3/vintages.parquet ]; then
  echo "ERROR: outputs/phase3/vintages.parquet is missing — that is the fitted"
  echo "  model this script builds on. Either restore it, or refit with:"
  echo "    $PY -m s05_refit_harness --rebuild --jobs 2    # ~25 min"
  exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/logs
RESULTS="outputs/logs/run_${STAMP}.tsv"
printf 'stage\tstatus\tseconds\n' > "$RESULTS"

STAGES=(
  s06_daily_filter
  s07_archetype_probs
  s08_outcome_densities
  s09_predictive_mixture
  s10_validation
  s11_stock_risk_profiles
  s13_lookahead_audit
)

fail=0
for st in "${STAGES[@]}"; do
  log="outputs/logs/${STAMP}_${st}.log"
  echo "[$(date +%H:%M:%S)] === $st ==="
  t0=$(date +%s)
  if "$PY" -m "$st" > "$log" 2>&1; then
    status=PASS
  else
    status=FAIL; fail=$((fail+1))
    echo "    FAILED — last lines:"; tail -15 "$log" | sed 's/^/      /'
  fi
  t1=$(date +%s)
  printf '%s\t%s\t%s\n' "$st" "$status" "$((t1-t0))" >> "$RESULTS"
  echo "    $status  $((t1-t0))s  -> $log"
  # A gate failure downstream of a broken stage is noise; stop.
  [ "$status" = FAIL ] && { echo "HALTED at $st"; exit 1; }
done

echo
echo "COMPLETE — $((${#STAGES[@]}-fail))/${#STAGES[@]} stages passed"
column -t "$RESULTS"
