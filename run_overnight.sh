#!/bin/bash
# Thesis-level calibration — runs both regimes, ~6-10h on a MacBook Air.
# Doubles N_RUNS to 16 (vs the 8 in the standard "full" command) to kill
# the iteration-mode noise we kept seeing; bumps N_LHS to 200 for cleaner
# 5-d surrogate coverage; everything else at the documented thesis-final
# scale. Logs to output/calibration_overnight.log.
#
# Run with:  nohup ./run_overnight.sh > /dev/null 2>&1 &
# (or just  ./run_overnight.sh  if you'll keep the terminal open)

set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "Overnight calibration started: $(date)"
echo "Theta: 5-d (depth_mean, depth_sigma, zi_alpha, zi_mu, zi_delta)"
echo "Loss:  4-component (Hill + V + ACF1 + ACF2 short-lag, D45)"
echo "Roster: 10 FT + 10 BCM + 10 MT + 20 ZI + 5 NBCM + 1 CCP"
echo "================================================================"

# Always start from a clean cache (the LHS/stage-2 CSVs cache previous-run
# samples — reusing them after model changes corrupts the surrogate).
rm -f output/calibration_lhs_*.csv output/calibration_stage2_*.csv

# Args: N_LHS=200 N_DAYS=30 N_RUNS=16 N_REFINE=3 N_PER_REFINE=30 N_STAGE2=80
python3 calibrate.py run 200 30 16 3 30 80 2>&1 | tee output/calibration_overnight.log

echo "================================================================"
echo "Finished: $(date)"
echo "Calibrated theta written to output/calibrated_params.json"
echo "Validation table + grouped Δ contributions in output/calibration_overnight.log"
echo "================================================================"
