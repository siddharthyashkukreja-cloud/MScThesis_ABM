import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CONTRACT_USD
from model.run_simulation import build_traders, build_clearing_tier
from model.simulation import Simulation

"""Measure whether clearing members actually breach their capital ratios (and therefore deleverage /
stop out) in the committed tiered/direct runs. rows.csv records member DEFAULTS but not capital ratios
or deleverage events, so this re-runs the experiment arms with both ratios instrumented:

  - Basel leverage ratio   cash / exposure        >= LR_FLOOR_BCM (4.25%)  -> BCM deleverages own book
  - CFTC Reg 1.17          cash / IM(client book) >= REG117_FLOOR_NBCM (8%) -> stop-out (both tiers)

A breach of either is exactly the deleverage/stop-out trigger in simulation.py (~L737-763).
Usage:  N=40 PYTHONPATH=. python3 agent_context/probe_capital_ratios.py   (N tiered+direct stressed, calm)"""

LR, REG = G.LR_FLOOR_BCM, G.REG117_FLOOR_NBCM
N = int(os.environ.get("N", "40"))

def window(regime):
    fv = pd.read_csv(f"data/fv_{regime}.csv")
    V, SIG = fv["V_smooth"].to_numpy(float), fv["sigma_t"].to_numpy(float)
    st = list(day_start_steps(regime))
    s, e = (st[10], st[30]) if regime == "stressed" else (0, st[20])
    return V, SIG, s, e - s

def run(regime, seed, direct):
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"; G.DF_DECOUPLE_IM = False
    V, SIG, s, n = window(regime)
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[regime], stressed=(regime == "stressed"))
    tr = build_traders(p, seed=seed)
    ccp = build_clearing_tier(tr, p, seed=seed, direct=direct)
    sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=s)
    sim.v_array = V[s:s+n]; sim.sigma_t_array = SIG[s:s+n]
    sim.run(n)
    ch = pd.DataFrame(sim.clearing_history); ncl = len(pd.DataFrame(sim.client_history))
    mids = pd.Series(sim.history["mid_price"], index=sim.history["t"])
    ch = ch[ch["kind"].isin(["BCM", "NBCM"])].copy()
    ch["mid"] = ch["t"].map(mids)
    ch["expo"] = ch["own_position"].abs() * p.volume_lot * CONTRACT_USD * ch["mid"] + ch["client_notional"]
    ch["basel"] = np.where(ch["expo"] > 0, ch["cash"] / ch["expo"], np.inf)
    ch["reg117"] = ch["capital_ratio"]
    ch["breach"] = (ch["reg117"] <= REG) | ((ch["kind"] == "BCM") & (ch["basel"] <= LR))
    g = ch.groupby(["agent_id", "kind"]).agg(min_reg117=("reg117", "min"),
        min_basel=("basel", "min"), ever_breach=("breach", "any")).reset_index()
    g["seed"] = seed; g["cl_def"] = ncl
    return g

def summarize(regime, direct, nseed):
    A = pd.concat([run(regime, sd, direct) for sd in range(42, 42 + nseed)], ignore_index=True)
    arm = "DIRECT" if direct else "TIERED"
    cd = A.groupby("seed").cl_def.first()
    print(f"\n===== {regime} / {arm} ({nseed} seeds) =====")
    print(f"  client defaults {cd.mean():.2f}/seed | member DEFAULTS recorded separately in rows.csv (0 tiered)")
    for k in ["BCM", "NBCM"]:
        s = A[A.kind == k]
        if not len(s): continue
        mr = s.min_reg117.replace([np.inf, -np.inf], np.nan); mb = s.min_basel.replace([np.inf, -np.inf], np.nan)
        print(f"  {k}: floor-breaches {int(s.ever_breach.sum())}/{len(s)} CM-runs | "
              f"min cash/IM(Reg1.17)={mr.min():.3f}(fl {REG}) min cash/expo(Basel)={mb.min():.3f}(fl {LR})")
    print(f"  -> deleverage/stop-out fires in {int(A.groupby('seed').ever_breach.any().sum())}/{nseed} seeds")

if __name__ == "__main__":
    summarize("stressed", False, N)
    summarize("stressed", True, min(N, 10))
    summarize("calm", False, min(N, 10))
    print(f"\nfloors: Basel LR (BCM cash/exposure) {LR} ; Reg 1.17 (cash/IM) {REG} ; deleverage target {G.DELEVERAGE_TARGET_BCM}")
