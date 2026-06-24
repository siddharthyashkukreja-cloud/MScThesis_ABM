#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Overnight calibration + validation, both regimes, one command.
#   1. XGBoost surrogate calibration on the D68 widened bounds (calm, stressed)
#   2. Franke MCR + J specification p-value on each located optimum
#   3. Auto-wire the optimum into globals.CALIBRATED (backup + verify)
#   4. Print a summary you can paste numbers from in the morning
#
# Run from the repo root, sleep-proofed and detached:
#   nohup caffeinate -i ./scripts/run_calib_overnight.sh > output/overnight.log 2>&1 &
#   tail -f output/overnight.log
# Knobs (env): LHS DAYS RUNS  (surrogate);  MCR_M MCR_B  (validation).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.."                       # repo root
mkdir -p output/relock/logs
# Bigger budget for the 6-d (calm) / 7-d (stressed) search (mt_lambda + mt_gamma added).
LHS=${LHS:-256}; DAYS=${DAYS:-75}; RUNS=${RUNS:-3}
REFINE=${REFINE:-3}; PER_REFINE=${PER_REFINE:-48}; STAGE2=${STAGE2:-48}
MCR_M=${MCR_M:-50}; MCR_B=${MCR_B:-2000}
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START  surrogate(LHS=$LHS DAYS=$DAYS RUNS=$RUNS REFINE=$REFINE/$PER_REFINE STAGE2=$STAGE2)  +  MCR(M=$MCR_M B=$MCR_B)"

for reg in calm stressed; do
  echo "[$(ts)] SURROGATE $reg ..."
  if python3 scripts/calibrate.py run "$reg" "$LHS" "$DAYS" "$RUNS" "$REFINE" "$PER_REFINE" "$STAGE2" \
       < /dev/null > "output/relock/logs/sur_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "output/relock/sur_${reg}.json"
    echo "[$(ts)] SURROGATE $reg DONE -> output/relock/sur_${reg}.json"
  else
    echo "[$(ts)] SURROGATE $reg FAILED -> output/relock/logs/sur_${reg}.log"; continue
  fi

  echo "[$(ts)] VALIDATION $reg (MCR + J) ..."
  if python3 scripts/franke_diagnostics.py "$reg" "$MCR_M" "$MCR_B" "output/relock/sur_${reg}.json" \
       < /dev/null > "output/relock/logs/franke_${reg}.log" 2>&1; then
    echo "[$(ts)] VALIDATION $reg DONE -> output/relock/franke_${reg}.json"
  else
    echo "[$(ts)] VALIDATION $reg FAILED -> output/relock/logs/franke_${reg}.log"
  fi
done

echo "[$(ts)] WIRING globals.CALIBRATED ..."
python3 scripts/wire_lock.py < /dev/null || echo "[$(ts)] wiring skipped (see error above)"

echo ""
echo "================  SUMMARY  ================"
python3 - <<'PY'
import json, os
for reg in ("calm", "stressed"):
    sp, fp = f"output/relock/sur_{reg}.json", f"output/relock/franke_{reg}.json"
    print(f"\n[{reg}]")
    if os.path.exists(sp):
        d = json.load(open(sp))["results"][reg]
        th = {k: round(v, 4) for k, v in d.get("theta_stage2", {}).items()}
        cd = {k: round(v, 2) for k, v in d.get("component_deltas", {}).items()}
        r2 = d.get("surrogate_d_accuracy", {}).get("r2")
        print("  theta :", th)
        print(f"  D={d.get('true_loss_validated'):.2f}  components={cd}  surrogate R2={r2:.2f}")
    if os.path.exists(fp):
        f = json.load(open(fp)); pm = f["per_moment_mcr"]
        print(f"  MCR: mean per-moment {100*sum(pm)/len(pm):.0f}%  joint {100*f['joint_mcr']:.0f}% "
              f"(ceiling {100*f['joint_ceiling']:.0f}%)  J p-value {f['p_value']:.3f}")
print("\nParams: output/relock/sur_*.json | Validation: output/relock/franke_*.json | globals: wired")
PY
echo "[$(ts)] ALL DONE"
