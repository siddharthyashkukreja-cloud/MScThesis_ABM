#!/usr/bin/env bash
# Thesis-final BASELINE calibration (E0 — no campaign flags) at high resolution.
# The current globals.CALIBRATED came from a low-budget (36-LHS) pre-campaign run;
# this locks a trustworthy baseline theta to report. Baseline is low-dim (calm 3
# params, stressed 4), so the surrogate is easy — the budget here favours SEEDS
# (6) to drive down validated-D noise, the thing that made E5 unreproducible.
# Pure execution, no env flags. ~3.5h.
#
#   resolution: 160 LHS, 20 days, 6 seeds, 2 refine x32, 24 stage-2
#   args: <regime> <n_lhs> <n_days> <n_runs> <n_refine> <n_per_refine> <n_stage2>

cd "$(dirname "$0")" || exit 1
OUT=output/baseline_hires
mkdir -p "$OUT/logs"
RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

run(){ local reg="$1" s
  say "E0 $reg  flags=[none/baseline]  args=[160 20 6 2 32 24]"
  rm -f "output/calibration_lhs_${reg}.csv" output/calibrated_params.json
  s=$(date +%s)
  if python3 calibrate.py run "$reg" 160 20 6 2 32 24 < /dev/null > "$OUT/logs/E0_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "$OUT/E0_${reg}.json"
    say "E0 $reg  DONE in $(( ($(date +%s)-s)/60 )) min  -> $OUT/E0_${reg}.json"
  else
    say "E0 $reg  FAILED (see $OUT/logs/E0_${reg}.log)"
  fi
}

say "=== high-res baseline start (E0, 160 LHS / 6 seeds, no flags) ==="
T0=$(date +%s)
run calm
run stressed
say "=== high-res baseline done in $(( ($(date +%s)-T0)/60 )) min ==="
say "Results: output/baseline_hires/E0_calm.json , E0_stressed.json"
