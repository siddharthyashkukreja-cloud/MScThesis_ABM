#!/usr/bin/env bash
# ============================================================================
# Overnight FINAL run — everything needed for a complete set of thesis results.
# Stages (toggle with env flags; defaults give the full remaining suite):
#
#   RERUN_CORE=1   re-run descriptive + H1 + netting   (default OFF — already done at 60/40)
#   ROBUST=1       severity + close-out + mechanism     (default ON  — the new robustness suite)
#   TWOFACTOR=1    two-factor synthetic calibration + 30-path ensemble + clearing (default ON)
#
# Order: [core] -> robustness (fast, ~1.5h) -> two-factor (long pole, budgeted). Running the
# fast robustness first means those results are on disk early even if the 2f run is still going.
# Coexists with a single-factor calibrate_svmjd.py already running (separate output dir + CSV).
#
# Launch before bed (keeps running after you close the terminal):
#     nohup bash scripts/run_overnight.sh > output/overnight.log 2>&1 &
# Watch:   tail -f output/overnight.log
# Examples:
#     RERUN_CORE=1 TWOF_HOURS=6 nohup bash scripts/run_overnight.sh > output/overnight.log 2>&1 &
#     TWOFACTOR=0 nohup bash scripts/run_overnight.sh > output/overnight.log 2>&1 &   # robustness only
# ============================================================================
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
mkdir -p output/logs

DESC_SEEDS=${DESC_SEEDS:-60}
HYP_SEEDS=${HYP_SEEDS:-40}
SEV_SEEDS=${SEV_SEEDS:-20}
CO_SEEDS=${CO_SEEDS:-20}
MECH_SEEDS=${MECH_SEEDS:-40}
RERUN_CORE=${RERUN_CORE:-0}
ROBUST=${ROBUST:-1}
TWOFACTOR=${TWOFACTOR:-1}
TWOF_HOURS=${TWOF_HOURS:-6}
REGIME=${REGIME:-stressed}
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "================ OVERNIGHT RUN  $(ts) ================"
echo "core=$RERUN_CORE  robust=$ROBUST  twofactor=$TWOFACTOR (${TWOF_HOURS}h)  regime=$REGIME"
echo "seeds: desc=$DESC_SEEDS hyp=$HYP_SEEDS sev=$SEV_SEEDS co=$CO_SEEDS mech=$MECH_SEEDS"

# ── [optional] core: descriptive + hypotheses (only if you want them refreshed) ──
if [ "$RERUN_CORE" = "1" ]; then
  echo; echo "[core 1/2] MAIN MODEL — descriptive  ($(ts))"
  N_SEEDS=$DESC_SEEDS python3 scripts/run_model_descriptive.py \
      > "output/logs/descriptive_${DESC_SEEDS}.log" 2>&1 \
      && echo "  descriptive OK -> output/results/descriptive/*.csv" \
      || echo "  descriptive FAILED — see output/logs/descriptive_${DESC_SEEDS}.log"
  echo; echo "[core 2/2] HYPOTHESES — margin arms + netting  ($(ts))"
  N_SEEDS=$HYP_SEEDS python3 scripts/run_thesis_experiments.py \
      > "output/logs/hypotheses_${HYP_SEEDS}.log" 2>&1 \
      && echo "  hypotheses OK -> output/thesis_final/experiments/*.csv" \
      || echo "  hypotheses FAILED — see output/logs/hypotheses_${HYP_SEEDS}.log"
fi

# ── robustness: severity sweep + close-out sweep + tail mechanism ──
if [ "$ROBUST" = "1" ]; then
  echo; echo "[robust] SEVERITY + CLOSE-OUT + MECHANISM  ($(ts))"
  SEV_SEEDS=$SEV_SEEDS CO_SEEDS=$CO_SEEDS MECH_SEEDS=$MECH_SEEDS \
    python3 scripts/run_robustness.py --max-hours 3 \
      > "output/logs/robustness.log" 2>&1 \
      && echo "  robustness OK -> output/robustness/{severity,closeout,mechanism}_summary.csv" \
      || echo "  robustness FAILED — see output/logs/robustness.log"
fi

# ── two-factor synthetic: long final calibration + 30-path ensemble + clearing ──
if [ "$TWOFACTOR" = "1" ]; then
  echo; echo "[2factor] CALIBRATION + 30-PATH ENSEMBLE + CLEARING  (${TWOF_HOURS}h, $(ts))"
  python3 scripts/overnight_2f.py --regime "$REGIME" --max-hours "$TWOF_HOURS" \
      > "output/logs/overnight_2f.log" 2>&1 \
      && echo "  two-factor OK -> output/overnight_2f/{calibration.json,validation.csv,clearing.csv,SUMMARY.md}" \
      || echo "  two-factor FAILED — see output/logs/overnight_2f.log"
fi

echo; echo "================ DONE  $(ts) ================"
echo "--- robustness summary ---";  cat  output/robustness/SUMMARY.md      2>/dev/null
echo "--- two-factor summary ---";  cat  output/overnight_2f/SUMMARY.md    2>/dev/null
echo
echo "Morning: open results_figures.ipynb (core figs) and add panels for"
echo "  output/robustness/*  and  output/overnight_2f/*  (severity curve, closeout bars, |r|-ACF repair)."
