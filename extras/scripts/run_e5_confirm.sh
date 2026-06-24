#!/usr/bin/env bash
# E5 confirmation run — re-run FT/MT activation + cancellation (FTMT_GATES) at higher
# resolution to confirm the stressed win (D 13.84 -> 9.19 at 192 LHS / 3 seeds) and
# re-check calm, before deciding to adopt. More LHS AND more seeds than campaign_v2
# (the validated-D noise that flipped E5 between v1 and v2 is seed-driven, so 5 seeds).
# Pure execution — no Claude Code. Stressed first (the result we're confirming).
#
#   resolution: 256 LHS, 20 days, 5 seeds, 2 refine rounds x40, 32 stage-2  (~4-5h total)
#   args to calibrate.py run: <regime> <n_lhs> <n_days> <n_runs> <n_refine> <n_per_refine> <n_stage2>

cd "$(dirname "$0")" || exit 1
OUT=output/e5_confirm
mkdir -p "$OUT/logs"
RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

run(){ local reg="$1" s
  say "E5 $reg  flags=[FTMT_GATES=1]  args=[256 20 5 2 40 32]"
  rm -f "output/calibration_lhs_${reg}.csv" output/calibrated_params.json
  s=$(date +%s)
  if FTMT_GATES=1 python3 calibrate.py run "$reg" 256 20 5 2 40 32 > "$OUT/logs/E5_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "$OUT/E5_${reg}.json"
    say "E5 $reg  DONE in $(( ($(date +%s)-s)/60 )) min  -> $OUT/E5_${reg}.json"
  else
    say "E5 $reg  FAILED (see $OUT/logs/E5_${reg}.log)"
  fi
}

say "=== E5 confirmation start (FTMT_GATES, 256 LHS / 5 seeds) ==="
T0=$(date +%s)
run stressed
run calm
say "=== E5 confirmation done in $(( ($(date +%s)-T0)/60 )) min ==="
say "Results: output/e5_confirm/E5_stressed.json , E5_calm.json"
