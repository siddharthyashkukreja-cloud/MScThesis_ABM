"""Leverage-floor sweep: does the calm-clean / stress-breach-but-survive behaviour hold as the
8% cash/exposure floor is lowered toward the accurate Basel/eSLR G-SIB range (~4%)?
Uniform floor (only client-carrying CMs ever breach, so uniform ~ differentiated for the breach).
Resumable -> outputs/floor_sweep.csv."""
import os, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps
import model.run_simulation as RS
from model.simulation import Simulation

PATH = os.environ.get("FPATH", "/sessions/ecstatic-cool-planck/mnt/outputs/floor_sweep.csv")
NSESS = int(os.environ.get("NSESS", "12"))
SEED = int(os.environ.get("SEED", "42"))
FLOORS = [float(x) for x in os.environ.get("FLOORS", "0.03,0.04,0.05,0.055,0.06,0.08").split(",")]

def run(rg, floor, target):
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"
    G.CCP_CALIBRATION["cap_ratio_floor"] = floor
    G.CAP_DELEVERAGE_TARGET = target
    FV = pd.read_csv(f"data/fv_{rg}.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps(rg)); s = st[10] if rg == "stressed" else 0
    e = st[min(len(st)-1, (10+NSESS if rg == "stressed" else NSESS))]; n = e - s
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[rg], stressed=(rg == "stressed"))
    tr = RS.build_traders(p, seed=SEED); ccp = RS.build_clearing_tier(tr, p, seed=SEED, direct=False)
    sim = Simulation(p, tr, seed=SEED, ccp=ccp, v_start=s); sim.v_array = V[s:s+n]; sim.sigma_t_array = SIG[s:s+n]
    sim.run(n)
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    cm_rows = ch[ch.kind.isin(["BCM", "NBCM"])] if len(ch) else ch
    bcmr = cm_rows[cm_rows.kind == "BCM"] if len(cm_rows) else cm_rows
    nbcmr = cm_rows[cm_rows.kind == "NBCM"] if len(cm_rows) else cm_rows
    breach = int((cm_rows.capital_ratio <= floor).sum()) if len(cm_rows) else 0
    return dict(regime=rg, floor=floor, target=target,
                breach_cycles=breach,
                bcm_min_k=round(float(bcmr.capital_ratio.min()),3) if len(bcmr) else float("nan"),
                nbcm_min_k=round(float(nbcmr.capital_ratio.min()),3) if len(nbcmr) else float("nan"),
                client_def=len(cl),
                bcm_def=int(bcmr[bcmr.has_defaulted].agent_id.nunique()) if len(bcmr) else 0,
                nbcm_def=int(nbcmr[nbcmr.has_defaulted].agent_id.nunique()) if len(nbcmr) else 0,
                max_wf=int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0)

if __name__ == "__main__":
    done = set()
    if os.path.exists(PATH):
        done = {(r.regime, round(float(r.floor),3)) for r in pd.read_csv(PATH).itertuples()}
    todo = [(rg, f) for rg in ("calm", "stressed") for f in FLOORS if (rg, round(f,3)) not in done]
    budget = float(os.environ.get("BUDGET", "40")); t0 = time.time(); k = 0
    for (rg, f) in todo:
        if time.time()-t0 > budget:
            print(f"budget hit; wrote {k}; {len(todo)-k} left — rerun"); break
        target = round(f + 0.025, 3)   # management buffer above the floor (Sid's 5.5->8 spirit)
        r = run(rg, f, target)
        pd.DataFrame([r]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False); k += 1
        print(f"[{rg:8s} floor={f:.3f} tgt={target:.3f}] breach_cyc={r['breach_cycles']:4d} "
              f"bcm_min_k={r['bcm_min_k']} cl_def={r['client_def']:3d} bcm_def={r['bcm_def']} "
              f"nbcm_def={r['nbcm_def']} wf={r['max_wf']} | {time.time()-t0:.1f}s")
    if not todo:
        df = pd.read_csv(PATH).sort_values(["regime","floor"])
        print(df.to_string(index=False))
