#!/usr/bin/env bash
# SV-MJD recalibration (D62c) — refit the behavioural theta on the synthetic
# Stein-Stein/Heston + Merton fundamental (FV_GBM=1; data/fv_gbm_*.csv) so the
# SV-MJD scenario-generator ensembles run on their own calibrated optimum
# (ablation-C1 design: same empirical targets/weights, independent fundamental).
# Recipe mirrors the D60 lock: grid headline (calm 7^3, stressed 5^4, 20d,
# 3 seeds) + high-res surrogate cross-check (160 LHS, 20d, 6 seeds, 2 refine).
# Results -> output/baseline_gbm/. The root result files (the LOCKED Kalman
# record: calibrated_params*.json, calibration_grid_*.csv, calibration_
# stage2_*.csv) are stashed first and restored at exit, so the lock survives
# even on failure. Stdin from /dev/null (survives a closed terminal); LAUNCH
# ONCE (use tmux). ~6-8h total.
cd "$(dirname "$0")" || exit 1
OUT=output/baseline_gbm
mkdir -p "$OUT/logs"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "A gbm recalibration is already in progress ($OUT/.lock exists) — aborting." >&2
  echo "If stale (no calibrate.py running): rmdir $OUT/.lock" >&2
  exit 1
fi

STASH="calibrated_params_grid.json calibrated_params.json calibration_grid_calm.csv calibration_grid_stressed.csv calibration_stage2_calm.csv calibration_stage2_stressed.csv"
for f in $STASH; do [ -f "output/$f" ] && cp "output/$f" "$OUT/.stash_$f"; done
restore(){ for f in $STASH; do [ -f "$OUT/.stash_$f" ] && mv "$OUT/.stash_$f" "output/$f"; done; }
trap 'restore; rmdir "$OUT/.lock" 2>/dev/null' EXIT

export FV_GBM=1
RUNLOG="$OUT/run.log"
ts(){ date +%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }

grid(){ local reg="$1" nper="$2" s
  say "GBM GRID $reg  n_per_dim=$nper  (n_days=20, n_runs=3)"
  rm -f output/calibrated_params_grid.json
  s=$(date +%s)
  if python3 calibrate.py grid "$reg" "$nper" 20 3 < /dev/null > "$OUT/logs/grid_${reg}.log" 2>&1; then
    cp output/calibrated_params_grid.json "$OUT/grid_${reg}.json"
    cp "output/calibration_grid_${reg}.csv" "$OUT/grid_surface_${reg}.csv" 2>/dev/null
    say "GBM GRID $reg  DONE in $(( ($(date +%s)-s)/60 )) min -> $OUT/grid_${reg}.json"
  else
    say "GBM GRID $reg  FAILED (see $OUT/logs/grid_${reg}.log)"
  fi
}

sur(){ local reg="$1" s
  say "GBM SURROGATE $reg  (160 LHS, 20d, 6 seeds, 2 refine, 32/round, 32 stage-2)"
  rm -f output/calibrated_params.json
  s=$(date +%s)
  if python3 calibrate.py run "$reg" 160 20 6 2 32 32 < /dev/null > "$OUT/logs/sur_${reg}.log" 2>&1; then
    cp output/calibrated_params.json "$OUT/sur_${reg}.json"
    say "GBM SURROGATE $reg  DONE in $(( ($(date +%s)-s)/60 )) min -> $OUT/sur_${reg}.json"
  else
    say "GBM SURROGATE $reg  FAILED (see $OUT/logs/sur_${reg}.log)"
  fi
}

fresh(){ local reg="$1" tag="$2" envprefix="$3" s
  say "FRESH-SEED VALIDATION $tag $reg (top-5 grid nodes, 2 fresh common seed sets)"
  s=$(date +%s)
  if $envprefix python3 validate_grid_fresh.py "$reg" 5 20 3 < /dev/null \
        > "$OUT/logs/fresh_${tag}_${reg}.log" 2>&1; then
    say "FRESH $tag $reg DONE in $(( ($(date +%s)-s)/60 )) min (log: $OUT/logs/fresh_${tag}_${reg}.log)"
  else
    say "FRESH $tag $reg FAILED (see $OUT/logs/fresh_${tag}_${reg}.log)"
  fi
}

say "=== market-layer completion run start ==="
T0=$(date +%s)

# Stage 0 — fresh-seed validation of the LOCKED Kalman grid optimum (D60),
# against the original loss surfaces (still untouched at this point). Closes
# the winner's-curse audit item on the headline theta. Runs WITHOUT FV_GBM.
fresh calm    kalman "env -u FV_GBM"
fresh stressed kalman "env -u FV_GBM"
for reg in calm stressed; do
  [ -f "output/grid_freshseed_${reg}.json" ] && cp "output/grid_freshseed_${reg}.json" output/baseline_grid/
done

# Stage 1 — SV-MJD grid headline (overwrites the root surfaces; restored at exit).
grid calm 7
grid stressed 5

# Stage 2 — fresh-seed validation of the SV-MJD grid optima (FV_GBM is exported).
fresh calm    gbm ""
fresh stressed gbm ""
for reg in calm stressed; do
  [ -f "output/grid_freshseed_${reg}_gbm.json" ] && mv "output/grid_freshseed_${reg}_gbm.json" "$OUT/"
done

# Stage 3 — SV-MJD surrogate cross-check (XGBoost SMM at D60-lock resolution).
sur calm
sur stressed

say "=== done in $(( ($(date +%s)-T0)/60 )) min — results in $OUT/ ==="
say "Kalman fresh-seed validation: output/baseline_grid/grid_freshseed_{calm,stressed}.json"
say "SV-MJD theta (grid + surrogate + fresh-seed): $OUT/"
say "NB: do NOT copy the gbm theta into globals.CALIBRATED — it is the SV-MJD"
say "ensemble theta; CALIBRATED stays the locked Kalman baseline (D60)."
