#!/usr/bin/env bash
# ============================================================================
# Launch the THREE overnight jobs IN PARALLEL. They write to separate output
# dirs and run in separate processes (own globals), so they never collide —
# use on a multi-core machine (needs ~3-4 cores free).
#
#   [1] CORE  descriptive (main model)   -> output/results/descriptive/
#   [2] H1    margin procyclicality arms -> output/thesis_final/experiments/
#   [3] SYNTH joint log-vol calibration + stressed ensemble -> output/overnight_joint_logvol/
#
# Core + H1 now also emit, per run: client_defaults.csv (type / carrying CM / position-at-
# default / IM / loss / timing), waterfall_events.csv (level + L1-L5 + mutualised),
# client_freezes.csv (own-distress vs CM-contagion, kappa, timing), porting_events.csv
# (n_ported / n_unported on a member default); core also emits member_balance_band.csv
# (cash / maintenance-margin / DF-contribution bands by member type).
#
# Sized for a ~15-16h window (core + H1 finish in ~5-6h; the 2f calibration is the long pole).
# Launch:   nohup bash scripts/run_parallel.sh > output/parallel.log 2>&1 &
# Monitor:  tail -f output/logs/core.log output/logs/h1.log output/logs/synth.log
# Override: DESC_SEEDS=100 HYP_SEEDS=150 TWOF_HOURS=12 bash scripts/run_parallel.sh
# ============================================================================
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
mkdir -p output/logs
DESC_SEEDS=${DESC_SEEDS:-100}
HYP_SEEDS=${HYP_SEEDS:-150}
TWOF_HOURS=${TWOF_HOURS:-12}
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "=== PARALLEL OVERNIGHT  $(ts) ==="
echo "core N=$DESC_SEEDS | H1 N=$HYP_SEEDS | synth ${TWOF_HOURS}h (stressed)"

N_SEEDS=$DESC_SEEDS python3 scripts/run_model_descriptive.py > output/logs/core.log 2>&1 &
P1=$!; echo "  [1] core  PID $P1  -> output/results/descriptive/"

N_SEEDS=$HYP_SEEDS python3 scripts/run_thesis_experiments.py > output/logs/h1.log 2>&1 &
P2=$!; echo "  [2] H1    PID $P2  -> output/thesis_final/experiments/"

python3 scripts/overnight_joint.py --regime stressed --log-vol --max-hours "$TWOF_HOURS" --calib-frac 0.85 > output/logs/synth.log 2>&1 &
P3=$!; echo "  [3] synth PID $P3  -> output/overnight_joint_logvol/"

echo "monitor: tail -f output/logs/{core,h1,synth}.log"
FAIL=0
wait $P1 && echo "[1] core DONE  $(ts)"  || { echo "[1] core FAILED"; FAIL=1; }
wait $P2 && echo "[2] H1 DONE  $(ts)"    || { echo "[2] H1 FAILED"; FAIL=1; }
wait $P3 && echo "[3] synth DONE  $(ts)" || { echo "[3] synth FAILED"; FAIL=1; }
echo "=== ALL DONE  $(ts)  (fail=$FAIL) ==="
echo "--- core summary ---";  tail -n 4  output/logs/core.log  2>/dev/null
echo "--- H1 summary ---";    tail -n 7  output/logs/h1.log    2>/dev/null
echo "--- synth summary ---"; tail -n 6  output/logs/synth.log 2>/dev/null
