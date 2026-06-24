#!/usr/bin/env bash
# =============================================================================
# THESIS-FINAL overnight pipeline. One command, set-and-forget. ~13-18h.
#
#   1. pre-flight (deps + the live model state: V_t form, transfer close-out, gross margining)
#   2. RELOCK     -> ./scripts/run_relock.sh : grid + fresh-seed + surrogate, both regimes (output/relock/)
#   3. verify     : compare the relock grid optima to globals.CALIBRATED (warn if they drift)
#   4. MCR        -> calibrate.py mcr : Franke/HFABM Moment Coverage Ratio on the locked theta
#   5. EXPERIMENTS-> run_thesis_experiments.py : H1 tiered/direct, H2 margin regimes, gross/net,
#                    calm + stressed at ACTUAL severity (reverse-stress + fire-sale deferred)
#   6. collect all artifacts under output/thesis_final/
#
# LAUNCH (detached):  nohup ./scripts/run_thesis_final.sh > output/thesis_final_console.log 2>&1 &
# MONITOR:            tail -f output/thesis_final/run.log
# SV-MJD robustness (separate night): FV_GBM=1 OUT_TAG=relock_gbm nohup ./scripts/run_relock.sh &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1
OUT=output/thesis_final
mkdir -p "$OUT"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "A thesis-final run is already in progress ($OUT/.lock). If stale: rmdir $OUT/.lock" >&2
  exit 1
fi
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT
RUNLOG="$OUT/run.log"
ts(){ date +%F_%H:%M:%S; }
say(){ echo "[$(ts)] $*" | tee -a "$RUNLOG"; }
N_SEEDS="${N_SEEDS:-40}"
MCR_M="${MCR_M:-80}"

# ---- 1. pre-flight -----------------------------------------------------------
say "=== THESIS-FINAL pipeline start (N_SEEDS=$N_SEEDS, MCR_M=$MCR_M) ==="
python3 - <<'PY' >> "$RUNLOG" 2>&1 || { say "PRE-FLIGHT FAILED — see run.log (nothing started)"; exit 1; }
import importlib, inspect
for m in ("numpy", "pandas", "scipy", "xgboost"):
    importlib.import_module(m)
import model.agents as a, model.globals as g, calibrate  # noqa: F401
assert "ctx.v * (1.0 + self.z_score * params.ft_sigma_c * sigma_t)" in \
    inspect.getsource(a.FundamentalTrader.submit_orders), "V_t reservation not active"
# final clearing model: physical IM escrow, net-position capital ratio, static position limit on
# CLIENT-CLEARING BCMs only (house-only BCMs follow the ODD 8% capital-adequacy floor; sigma-VaR
# cap + deleverage buffer retired), member open-market client close-out, CCP auction for
# member/direct-client defaults.
assert g.IM_ESCROW is True, "IM_ESCROW must be on for the final model"
assert g.POSITION_LIMIT_X > 0 and g.POSITION_LIMIT_CLIENTS_ONLY, "static cap on client-clearing BCMs only (house-only BCMs follow the 8% CAR floor)"
assert g.CLIENT_CLOSEOUT == "firesale", "client default = member open-market liquidation"
assert g.CLOSEOUT_MODE == "transfer", "member/direct-client default = CCP auction (transfer)"
assert g.CLIENT_MARGIN_NETTING == "gross", "CLIENT_MARGIN_NETTING must default 'gross'"
print("pre-flight OK:", g.CALIBRATED)
print("clearing:", dict(IM_ESCROW=g.IM_ESCROW, POSITION_LIMIT_X=g.POSITION_LIMIT_X,
                        CLIENT_CLOSEOUT=g.CLIENT_CLOSEOUT, CLOSEOUT_MODE=g.CLOSEOUT_MODE))
PY
say "pre-flight OK"
cp model/globals.py "$OUT/globals_snapshot.py"
{ git rev-parse HEAD; git status -s; } >> "$RUNLOG" 2>/dev/null || true

T0=$(date +%s)
# ---- 2-3. RELOCK + VERIFY (re-fit theta on the data windows). With EXPERIMENTS_ONLY=1 these
#          are skipped and the theta already in globals.CALIBRATED is used. Final-run flow:
#          (1) full run re-fits theta; (2) copy the relock optimum into globals.CALIBRATED;
#          (3) EXPERIMENTS_ONLY=1 ./scripts/run_thesis_final.sh -> MCR/J-test + experiments on it.
if [ "${EXPERIMENTS_ONLY:-0}" != "1" ]; then
# ---- 2. RELOCK (the proven calibration; output/relock/) -----------------------
say "RELOCK -> ./scripts/run_relock.sh (grid + fresh-seed + surrogate, both regimes)"
if ./scripts/run_relock.sh < /dev/null >> "$OUT/logs_relock.log" 2>&1; then
  say "RELOCK done in $(( ($(date +%s)-T0)/60 )) min"
  cp -r output/relock "$OUT/relock" 2>/dev/null || true
else
  say "RELOCK FAILED (see $OUT/logs_relock.log) — aborting before MCR/experiments"; exit 1
fi

# ---- 3. verify the relock optima still match globals.CALIBRATED ---------------
say "VERIFY relock grid optima vs globals.CALIBRATED"
python3 - <<'PY' 2>&1 | tee -a "$RUNLOG"
import json
from model.globals import CALIBRATED
ok = True
for r in ("calm", "stressed"):
    try:
        g = json.load(open(f"output/relock/grid_{r}.json"))["results"][r]["theta_grid"]
    except Exception as e:
        print(f"  {r}: could not read relock grid json ({e})"); ok = False; continue
    c = CALIBRATED[r]
    diff = {k: (round(g.get(k, 0.0), 5), round(c.get(k, 0.0), 5))
            for k in set(g) | set(c) if abs(g.get(k, 0.0) - c.get(k, 0.0)) > 1e-6}
    print(f"  {r}: {'MATCH' if not diff else 'DIFFERS ' + str(diff)}")
    ok = ok and not diff
print("  -> globals.CALIBRATED is current" if ok else
      "  -> WARNING: relock optimum DIFFERS from globals.CALIBRATED. Review output/relock/, "
      "re-copy the chosen optimum into globals.CALIBRATED, then re-run steps 4-5 "
      "(calibrate.py mcr + run_thesis_experiments.py).")
PY

else
  say "EXPERIMENTS_ONLY=1 — skipped relock/verify; using globals.CALIBRATED for MCR + experiments"
fi

# ---- 4. MCR + Franke J-test (HFABM) on globals.CALIBRATED — ALWAYS runs -------
say "MCR + J-test — calm + stressed, M_long=$MCR_M"
python3 scripts/calibrate.py mcr calm     "$MCR_M" 20 < /dev/null >> "$OUT/logs_mcr.log" 2>&1 && say "MCR calm done" || say "MCR calm FAILED"
python3 scripts/calibrate.py mcr stressed "$MCR_M" 20 < /dev/null >> "$OUT/logs_mcr.log" 2>&1 && say "MCR stressed done" || say "MCR stressed FAILED"
cp output/mcr_calm.json output/mcr_stressed.json "$OUT/" 2>/dev/null || true

# ---- 5. contagion experiments (H1 / H2 / gross-vs-net; calm + stressed) ------
say "EXPERIMENTS -> run_thesis_experiments.py (N_SEEDS=$N_SEEDS)"
T1=$(date +%s)
if N_SEEDS="$N_SEEDS" python3 scripts/run_thesis_experiments.py < /dev/null >> "$OUT/logs_experiments.log" 2>&1; then
  say "EXPERIMENTS done in $(( ($(date +%s)-T1)/60 )) min -> $OUT/experiments/"
else
  say "EXPERIMENTS FAILED (see $OUT/logs_experiments.log)"
fi

# ---- 6. done -----------------------------------------------------------------
say "=== THESIS-FINAL pipeline done in $(( ($(date +%s)-T0)/60 )) min ==="
say "Artifacts in $OUT/: relock/ (calibration), mcr_{calm,stressed}.json (validation),"
say "experiments/ (H1/H2/NET summaries), logs_*.log. Review the VERIFY note above before writing."
