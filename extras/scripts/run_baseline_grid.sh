#!/usr/bin/env bash
# Independent GRID-search baseline (no surrogate) — the easy-to-defend headline and a
# cross-check of the surrogate optimum, with no surrogate drift. Evaluates the TRUE loss on
# a regular grid per regime: calm 3-d (7/dim = 343 nodes), stressed 4-d (5/dim = 625 nodes),
# 20-day, 3 seeds/node. ~3.5-4h. No sklearn needed (true loss, not a surrogate). Stdin from
# /dev/null so it survives a closed terminal. LAUNCH ONCE (use tmux).
cd "$(dirname "$0")" || exit 1
OUT=output/baseline_grid
mkdir -p "$OUT/logs"
# Single-instance lock (atomic mkdir) — refuses a second launch instead of clobbering the
# shared caches, the bug that corrupted earlier double-launched runs.
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "A baseline-grid run is already in progress ($OUT/.lock exists) — aborting." >&2
  echo "If it's stale (no calibrate.py running): rmdir $OUT/.lock" >&2
  exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT
RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

run(){ local reg="$1" nper="$2" s
  say "GRID $reg  n_per_dim=$nper  (n_days=20, n_runs=3)"
  rm -f output/calibrated_params_grid.json
  s=$(date +%s)
  if python3 calibrate.py grid "$reg" "$nper" 20 3 < /dev/null > "$OUT/logs/grid_${reg}.log" 2>&1; then
    cp output/calibrated_params_grid.json "$OUT/grid_${reg}.json"
    say "GRID $reg  DONE in $(( ($(date +%s)-s)/60 )) min  -> $OUT/grid_${reg}.json"
  else
    say "GRID $reg  FAILED (see $OUT/logs/grid_${reg}.log)"
  fi
}

say "=== baseline grid start (true-loss grid; calm 7^3, stressed 5^4, 3 seeds) ==="
T0=$(date +%s)
run calm 7
run stressed 5
say "=== baseline grid done in $(( ($(date +%s)-T0)/60 )) min ==="
say "Results: output/baseline_grid/grid_calm.json , grid_stressed.json"
