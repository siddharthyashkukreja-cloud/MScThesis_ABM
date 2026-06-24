#!/usr/bin/env bash
# D65 RE-LOCK — the final Kalman-baseline calibration: zi_mu re-included in
# both regimes, evidence-tightened boxes (the D60 stressed floors were binding
# — see the D65 entry in AGENT.md). Recipe mirrors D60/D64: exhaustive grid
# headline (calm 4-d x 5/dim = 625 nodes; stressed 5-d x 4/dim = 1024 nodes;
# 20d, 3 seeds) + fresh-common-seed top-5 validation + high-res surrogate
# cross-check (160 LHS, 20d, 6 seeds, 2 refine). Results -> output/relock/.
# Root result files (the D60 record) are stashed and restored at exit. Stale
# LHS caches are removed up front (the param set changed: zi_mu column added).
# Stdin from /dev/null; LAUNCH ONCE. ~11-13h.
#
# To repeat for the SV-MJD scenario generator with the SAME D65 loop, run
# afterwards:  FV_GBM=1 OUT_TAG=relock_gbm ./scripts/run_relock.sh
cd "$(dirname "$0")" || exit 1
OUT=output/${OUT_TAG:-relock}
mkdir -p "$OUT/logs"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "A re-lock run is already in progress ($OUT/.lock exists) — aborting." >&2
  echo "If stale (no calibrate.py running): rmdir $OUT/.lock" >&2
  exit 1
fi

STASH="calibrated_params_grid.json calibrated_params.json calibration_grid_calm.csv calibration_grid_stressed.csv calibration_stage2_calm.csv calibration_stage2_stressed.csv"
for f in $STASH; do [ -f "output/$f" ] && cp "output/$f" "$OUT/.stash_$f"; done
restore(){ for f in $STASH; do [ -f "$OUT/.stash_$f" ] && mv "$OUT/.stash_$f" "output/$f"; done; }
trap 'restore; rmdir "$OUT/.lock" 2>/dev/null' EXIT

rm -f output/calibration_lhs_calm.csv output/calibration_lhs_stressed.csv \
      output/calibration_lhs_calm_gbm.csv output/calibration_lhs_stressed_gbm.csv

RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

DAYS="${CAL_DAYS:-75}"   # matched calm/stressed window length (set CAL_DAYS=40 etc. for a faster run)
RUNS="${CAL_RUNS:-3}"    # identical seed count for grid AND surrogate (D comparable, audit #4)
grid(){ local reg="$1" nper="$2" s
  say "GRID $reg  n_per_dim=$nper  (n_days=$DAYS, n_runs=$RUNS)"
  rm -f output/calibrated_params_grid.json
  s=$(date +%s)
  if python3 scripts/calibrate.py grid "$reg" "$nper" "$DAYS" "$RUNS" < /dev/null > "$OUT/logs/grid_${reg}.log" 2>&1; then
    cp output/calibrated_params_grid.json "$OUT/grid_${reg}.json"
    cp "output/calibration_grid_${reg}.csv" "$OUT/grid_surface_${reg}.csv" 2>/dev/null
    say "GRID $reg  DONE in $(( ($(date +%s)-s)/60 )) min -> $OUT/grid_${reg}.json"
  else
    say "GRID $reg  FAILED (see $OUT/logs/grid_${reg}.log)"
  fi
}

fresh(){ local reg="$1" s
  say "FRESH-SEED VALIDATION $reg (top-5 nodes, 2 fresh common seed sets)"
  s=$(date +%s)
  if python3 scripts/validate_grid_fresh.py "$reg" 5 "$DAYS" "$RUNS" < /dev/null > "$OUT/logs/fresh_${reg}.log" 2>&1; then
    for f in output/grid_freshseed_${reg}*.json; do [ -f "$f" ] && mv "$f" "$OUT/"; done
    say "FRESH $reg  DONE in $(( ($(date +%s)-s)/60 )) min"
  else
    say "FRESH $reg  FAILED (see $OUT/logs/fresh_${reg}.log)"
  fi
}

sur(){ local reg="$1" s
  say "SURROGATE $reg  (160 LHS, ${DAYS}d, $RUNS seeds, 2 refine, 32/round, 32 stage-2)"
  rm -f output/calibrated_params.json
  s=$(date +%s)
  if python3 scripts/calibrate.py run "$reg" 160 "$DAYS" "$RUNS" 2 32 32 < /dev/null > "$OUT/logs/sur_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "$OUT/sur_${reg}.json"
    say "SURROGATE $reg  DONE in $(( ($(date +%s)-s)/60 )) min -> $OUT/sur_${reg}.json"
  else
    say "SURROGATE $reg  FAILED (see $OUT/logs/sur_${reg}.log)"
  fi
}

say "=== D65 re-lock start (zi_mu in loop; evidence-tightened boxes; FV_GBM=${FV_GBM:-0}) ==="
T0=$(date +%s)
grid calm 5
grid stressed 4
fresh calm
fresh stressed
sur calm
sur stressed
say "=== done in $(( ($(date +%s)-T0)/60 )) min — results in $OUT/ ==="
say "Review grid vs surrogate vs fresh-seed agreement, then copy the chosen"
say "optimum into globals.CALIBRATED (the D60 record stays in output/baseline_grid/)."
