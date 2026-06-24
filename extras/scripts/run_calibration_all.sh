#!/usr/bin/env bash
# run_calibration_all.sh — full thesis-level calibration in one shot: BOTH the
# surrogate-assisted SMM (XGB-Chiarella) and the exhaustive grid search (Gao 2023),
# for BOTH regimes (calm, stressed), recording EVERY moment (target vs achieved)
# alongside the loss and per-component deltas.
#
# This is long-running (hours — thesis-final resolution). Run it detached:
#     tmux new -s calib            # start a persistent session
#     ./run_calibration_all.sh     # (or: NRUNS=4 GRID_STR=6 ./run_calibration_all.sh to go lighter)
#     # detach with Ctrl-b then d ; reattach later with:  tmux attach -t calib
# or:  nohup ./run_calibration_all.sh > output/calib_all.log 2>&1 &
#
# Outputs (output/):
#   calibrated_params.json          surrogate optima + per-moment target vs validated (both regimes)
#   calibrated_params_grid.json     grid optima + per-moment values at the optimum (both regimes)
#   calibration_lhs_{regime}.csv    surrogate design — every moment per sampled theta
#   calibration_grid_{regime}.csv   full loss surface — D + every component delta + every moment per node
#   calibration_stage2_{regime}.csv surrogate stage-2 refinement on the true simulator
# After it finishes, copy the chosen optimum (grid is the headline) into
# model/globals.py CALIBRATED, then regenerate the figures.

set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

LOG="output/calib_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
echo "Logs -> $LOG"

# ── Resolution (override any via env) ────────────────────────────────────────
# Surrogate-assisted SMM:
NLHS=${NLHS:-256}        # Latin-hypercube design size
NDAYS=${NDAYS:-20}       # calm subsample length (stressed auto-caps to its ~29-day window)
NRUNS=${NRUNS:-8}        # seeds averaged per evaluation (variance reduction)
NREFINE=${NREFINE:-4}    # active-learning rounds
NPERREF=${NPERREF:-48}   # new sims per active-learning round
NSTAGE2=${NSTAGE2:-64}   # stage-2 true-simulator refinement size
# Grid search (true simulator at every node): calm is 3-d, stressed is 4-d.
GRID_CALM=${GRID_CALM:-9}   #  9^3 =   729 nodes
GRID_STR=${GRID_STR:-7}     #  7^4 = 2,401 nodes

# D56/D57 — the Kalman fundamental retains the overnight gaps and writes REAL per-bar
# timestamps (so the true ≈405-bar RTH-session boundaries are recoverable downstream),
# so regenerate both V_t paths first (calm + stressed); the stressed episode then carries
# its true ~-33% drawdown. (~6.5s — the per-day MLE is vectorised across days, D57.)
echo "== regenerate fundamentals (Kalman, overnight gaps + real session ts — D56/D57) =="
$PY data/v_kalman.py generate-all

# The fundamental changed (D56 gaps), the trader population was doubled, and ACF1 gained
# a 5-min lag, so the cached LHS design is stale — delete it so the surrogate regenerates
# from scratch on the new fundamentals.
rm -f output/calibration_lhs_calm.csv output/calibration_lhs_stressed.csv

echo "== empirical moment targets =="
$PY calibrate.py targets | tee "$LOG/targets.txt"

# ── Surrogate-assisted SMM, per regime (results MERGE into one JSON) ──────────
for R in calm stressed; do
  echo "== surrogate-SMM: $R =="
  $PY calibrate.py run "$R" "$NLHS" "$NDAYS" "$NRUNS" "$NREFINE" "$NPERREF" "$NSTAGE2" \
      2>&1 | tee "$LOG/surrogate_$R.log"
done

# ── Exhaustive grid search, per regime (regime-appropriate resolution) ───────
echo "== grid search: calm =="
$PY calibrate.py grid calm "$GRID_CALM" "$NDAYS" "$NRUNS" 2>&1 | tee "$LOG/grid_calm.log"
echo "== grid search: stressed =="
$PY calibrate.py grid stressed "$GRID_STR" "$NDAYS" "$NRUNS" 2>&1 | tee "$LOG/grid_stressed.log"

echo
echo "==================  DONE  =================="
echo "Surrogate optima:  output/calibrated_params.json"
echo "Grid optima:       output/calibrated_params_grid.json"
echo "Per-moment detail: output/calibration_{lhs,grid,stage2}_*.csv"
echo "Logs:              $LOG/"
echo
echo "Cross-method comparison (theta* should agree on the identified levers):"
$PY - <<'PYEOF'
import json
for f, label in [("output/calibrated_params.json", "SURROGATE"),
                 ("output/calibrated_params_grid.json", "GRID")]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  {label}: (not found) {e}"); continue
    print(f"\n== {label} ==")
    for r, res in d.get("results", {}).items():
        th = res.get("theta_stage2") or res.get("theta_grid") or {}
        D = res.get("true_loss_validated", res.get("D_grid"))
        ds = "  ".join(f"{k}={v:.3f}" for k, v in th.items())
        print(f"  {r:9s} D={D:.3f}  {ds}")
PYEOF
