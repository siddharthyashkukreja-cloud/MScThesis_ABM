#!/usr/bin/env bash
# Overnight contagion-experiment campaign — H1 (tiered vs direct), H2 (margin
# regimes), E6 (gaps vs shock), SV-MJD robustness overlay, open-disorder sensitivity.
# Runs a PRE-FLIGHT smoke test (must be error-free) then the full 40-seed matrix.
# The driver (run_experiments.py) checkpoints to output/experiments/rows.csv and is
# RESUMABLE — just re-run this script to continue after an interruption. Aggregation
# + SUMMARY.md run automatically at the end. Stdin from /dev/null; launch ONCE,
# ideally in tmux. Typical wall time ~0.5-3h (scales with cores). Env: NPROC, SEEDS.
cd "$(dirname "$0")" || exit 1
OUT=output/experiments
mkdir -p "$OUT/logs"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "A run is already in progress ($OUT/.lock) — aborting. If stale: rmdir $OUT/.lock" >&2
  exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT
RUNLOG="$OUT/run.log"; ts(){ date +%H:%M:%S; }; say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

say "=== PRE-FLIGHT smoke (2 seeds x 2 c, every family + both fundamentals) ==="
rm -f "$OUT/rows_smoke.csv" 2>/dev/null
if ! python3 run_experiments.py smoke < /dev/null >> "$RUNLOG" 2>&1; then
  say "SMOKE FAILED — see $RUNLOG. Aborting before the full run."; exit 1
fi
ERRS=$(python3 - <<'PY'
import pandas as pd
d = pd.read_csv("output/experiments/rows_smoke.csv")
print(int((~((d["error"].isna()) | (d["error"].astype(str) == ""))).sum()))
PY
)
if [ "$ERRS" != "0" ]; then
  say "SMOKE produced $ERRS errored rows — aborting. Inspect $OUT/rows_smoke.csv."; exit 1
fi
say "smoke clean (0 errors)."

say "=== FULL matrix (SEEDS=${SEEDS:-42:82}, NPROC=${NPROC:-auto}) ==="
T0=$(date +%s)
python3 run_experiments.py all < /dev/null >> "$RUNLOG" 2>&1
say "=== done in $(( ($(date +%s)-T0)/60 )) min ==="
say "READ output/experiments/SUMMARY.md + summary_*.csv. Re-run this script to resume if interrupted."
