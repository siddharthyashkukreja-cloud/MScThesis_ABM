#!/usr/bin/env bash
# =============================================================================
# Overnight re-calibration under the NEW V_t-relative FT reservation form
#   R = V_t * (1 + z * ft_sigma_c * sigma_t)        (was V_t + z*ft_sigma_c*sigma_t*v0)
# sigma_t half-life stays at 30 min (deferred decision — change later if wanted).
#
# This is a thin SAFETY WRAPPER around run_relock.sh: it pre-flights the
# environment (so a 12h run does not die at minute 1 on a missing dependency or
# the wrong reservation form being live), then hands off to the proven relock.
#
# Kalman (primary) path. For the SV-MJD robustness relock, run AFTERWARDS on a
# separate night:
#   FV_GBM=1 OUT_TAG=relock_gbm nohup ./scripts/run_relock.sh > output/relock_gbm_console.log 2>&1 &
#
# LAUNCH (detached, survives logout):
#   nohup ./run_recalib_vt.sh > output/recalib_vt_console.log 2>&1 &
# MONITOR:
#   tail -f output/relock/run.log
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "=== PRE-FLIGHT $(date) ==="
python3 - <<'PY' || { echo "PRE-FLIGHT FAILED — fix the above before launching (nothing was started)."; exit 1; }
import importlib, inspect
# 1. calibration dependencies present (calibrate.py needs these; not in every env)
for m in ("numpy", "pandas", "scipy", "xgboost"):
    importlib.import_module(m)
    print(f"  dep ok: {m}")
# 2. the NEW V_t reservation form is the one that will be calibrated
import model.agents as a
src = inspect.getsource(a.FundamentalTrader.submit_orders)
assert "ctx.v * (1.0 + self.z_score * params.ft_sigma_c * sigma_t)" in src, \
    "V_t reservation form is NOT active in model/agents.py — aborting."
print("  V_t reservation form: ACTIVE")
# 3. full calibration stack imports cleanly (catches any syntax error in the edits)
import calibrate  # noqa: F401
print("  calibration stack imports cleanly")
PY

echo "=== PRE-FLIGHT PASSED — launching full relock $(date) ==="
echo "    Expect ~11-15h. Results land in output/relock/.  Monitor: tail -f output/relock/run.log"
exec ./scripts/run_relock.sh
