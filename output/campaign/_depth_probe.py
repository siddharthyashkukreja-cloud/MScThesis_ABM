"""Book-depth sanity spot-check for one experiment regime (campaign acceptance check).
Reconstructs the validated theta under the SAME env as the calibration run (so PARAM_KEYS
and the population/flags match), runs one capped sim, and prints {depth_mean, depth_max}.
Acceptance: bid_depth+ask_depth must stay bounded (no runaway book)."""
import sys, os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import calibrate as C
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier

regime, jpath = sys.argv[1], sys.argv[2]
C._activate_regime(regime)
res = json.load(open(jpath))["results"][regime]
theta = np.array([float(res["theta_stage2"][k]) for k in C.PARAM_KEYS])
p = C._theta_to_params(theta, regime)
t = build_traders(p, seed=999)
ccp = build_clearing_tier(t, p, seed=999) if os.environ.get("CLEARING_IN_LOOP") else None
n = min(C._sim_steps(regime, 5), 2000)          # stationary — 2000 steps suffices
h = Simulation(p, t, seed=999, ccp=ccp).run(n)
tot = np.array(h["bid_depth"], float) + np.array(h["ask_depth"], float)
tot = tot[np.isfinite(tot)]
print(json.dumps({"depth_mean": float(tot.mean()) if len(tot) else float("nan"),
                  "depth_max": float(tot.max()) if len(tot) else float("nan")}))
