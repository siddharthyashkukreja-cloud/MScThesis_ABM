#!/usr/bin/env bash
# Campaign v2 — re-run the two promising configs and their combination at longer
# "lock-in" resolution (more LHS / seeds / refinement than the v1 screens).
# Pure execution: calibrate.py run + the campaign env flags already implemented on
# this branch. No Claude Code needed — just: bash run_campaign_v2.sh
#
#   E2     mt_lambda in the loop                  (MT_LAMBDA_IN_LOOP)
#   E5     FT/MT activation prob + cancellation    (FTMT_GATES)
#   COMBO  both at once                            (MT_LAMBDA_IN_LOOP + FTMT_GATES)
#
# Each run clears the LHS cache + calibrated_params.json first (the calibrated
# parameter set changes per config, so a stale cache would corrupt the surrogate),
# then copies its result to output/campaign_v2/<EID>_<regime>.json. Errors are
# logged and skipped, never fatal. ~6h total — leave it running.
#
# Args to `calibrate.py run`: <regime> <n_lhs> <n_days> <n_runs> <n_refine> <n_per_refine> <n_stage2>

cd "$(dirname "$0")" || exit 1
OUT=output/campaign_v2
mkdir -p "$OUT/logs"
RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

run(){ # eid  flags  regime  args
  local eid="$1" flags="$2" reg="$3" args="$4" s
  say "$eid $reg  flags=[$flags]  args=[$args]"
  rm -f "output/calibration_lhs_${reg}.csv" output/calibrated_params.json
  s=$(date +%s)
  if env $flags python3 calibrate.py run "$reg" $args > "$OUT/logs/${eid}_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "$OUT/${eid}_${reg}.json"
    say "$eid $reg  DONE in $(( ($(date +%s)-s)/60 )) min  -> $OUT/${eid}_${reg}.json"
  else
    say "$eid $reg  FAILED (see $OUT/logs/${eid}_${reg}.log) — continuing"
  fi
}

say "=== campaign v2 start (E2, E5, COMBO at lock-in resolution) ==="
T0=$(date +%s)
for reg in calm stressed; do run E2    "MT_LAMBDA_IN_LOOP=1"               "$reg" "128 20 3 2 32 24"; done
for reg in calm stressed; do run E5    "FTMT_GATES=1"                      "$reg" "192 20 3 2 40 32"; done
for reg in calm stressed; do run COMBO "MT_LAMBDA_IN_LOOP=1 FTMT_GATES=1"  "$reg" "224 20 3 2 40 32"; done
say "=== campaign v2 done in $(( ($(date +%s)-T0)/60 )) min ==="
say "Results: output/campaign_v2/<EID>_<regime>.json ; logs in output/campaign_v2/logs/"
