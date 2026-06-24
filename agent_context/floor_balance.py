"""Who defaults under deep stress? Compare floor/capital schemes for BCM-vs-NBCM default balance.
  A_current     : Option A (BCM Basel-LR cash/exposure incl. house; NBCM Reg1.17 cash/IM); NBCM $50M-1B
  B_nbcm_big    : same floors; NBCM capital raised to $0.5-3B (realistic big non-bank FCMs)
  C_unified     : de-confounded — BCM client-clearing also on Reg1.17 cash/IM (house separate); NBCM $50M-1B
  D_unified_big : de-confounded + NBCM $0.5-3B
Deep reverse-stress (c) on the COVID path. Resumable."""
import os, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CCP_CASH
import model.run_simulation as RS
from model.simulation import Simulation

PATH = os.environ.get("FBPATH", "/sessions/ecstatic-cool-planck/mnt/outputs/floor_balance.csv")
SCHEMES = {
    "A_current":     dict(nbcm=(5e7, 1e9),  unified=False),
    "B_nbcm_big":    dict(nbcm=(5e8, 3e9),  unified=False),
    "C_unified":     dict(nbcm=(5e7, 1e9),  unified=True),
    "D_unified_big": dict(nbcm=(5e8, 3e9),  unified=True),
}
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,7,123").split(",")]
C = float(os.environ.get("C", "2.2"))

def run(scheme, seed):
    s = SCHEMES[scheme]
    G.DIFFERENTIATED_FLOORS = True; G.UNIFIED_CLIENT_FLOOR = s["unified"]
    RS.NBCM_CASH_RANGE = s["nbcm"]; G.NBCM_CASH_RANGE = s["nbcm"]
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"
    G.CLOSEOUT_MODE = "transfer"; G.CLIENT_CLOSEOUT = "firesale"; G.CLOSEOUT_RECOVERY = 0.80
    FV = pd.read_csv("data/fv_stressed.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps("stressed")); a = st[10]; b = st[30]; n = b - a
    V = V[a:a+n].copy(); SIG = SIG[a:a+n].copy()
    r = np.diff(np.log(V)); V = V[0]*np.exp(np.concatenate([[0.0], np.cumsum(r*C)])); SIG = SIG*C
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
                    v0=float(V[0]), tick_size=0.25, dt_minutes=1.0, **CALIBRATED["stressed"], stressed=True)
    tr = RS.build_traders(p, seed=seed); ccp = RS.build_clearing_tier(tr, p, seed=seed, direct=False)
    sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=a); sim.v_array = V; sim.sigma_t_array = SIG
    sim.run(n)
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    cmr = ch[ch.kind.isin(["BCM", "NBCM"])] if len(ch) else ch
    bd = int(cmr[cmr.has_defaulted & (cmr.kind == "BCM")].agent_id.nunique()) if len(cmr) else 0
    nd = int(cmr[cmr.has_defaulted & (cmr.kind == "NBCM")].agent_id.nunique()) if len(cmr) else 0
    wf = int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0
    return dict(scheme=scheme, seed=seed, c=C, client_def=len(cl), bcm_def=bd, nbcm_def=nd, deepest_wf=wf)

if __name__ == "__main__":
    done = set()
    if os.path.exists(PATH):
        done = {(r.scheme, int(r.seed)) for r in pd.read_csv(PATH).itertuples()}
    todo = [(s, sd) for s in SCHEMES for sd in SEEDS if (s, sd) not in done]
    budget = float(os.environ.get("BUDGET", "40")); t0 = time.time(); k = 0
    for (s, sd) in todo:
        if time.time()-t0 > budget:
            print(f"budget hit; wrote {k}; {len(todo)-k} left — rerun"); break
        row = run(s, sd)
        pd.DataFrame([row]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False); k += 1
        print(f"[{s:14s} s{sd}] cl_def={row['client_def']:3d} BCM_def={row['bcm_def']} NBCM_def={row['nbcm_def']} wf={row['deepest_wf']} | {time.time()-t0:.0f}s")
    if not todo:
        df = pd.read_csv(PATH)
        g = df.groupby("scheme").agg(bcm_def=("bcm_def","sum"), nbcm_def=("nbcm_def","sum"),
                                     client_def=("client_def","mean"), wf=("deepest_wf","max"),
                                     seeds=("seed","count")).round(1)
        print(f"\n=== summed over {len(SEEDS)} seeds at c={C} ===\n", g.to_string())
