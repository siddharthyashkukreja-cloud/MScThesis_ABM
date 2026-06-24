"""Option A end-to-end: (1) calm + actual-COVID should be BENIGN (accurate floors); (2) deeper
stress (reverse-stress amplified COVID, c>1) should bring contagion, with tiered < direct.
Deep-stress runs use firesale close-out (conservation-safe; same for both arms -> apples-to-apples).
Resumable -> outputs/optA.csv."""
import os, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CCP_CASH, CONTRACT_USD
import model.run_simulation as RS
from model.simulation import Simulation

PATH = os.environ.get("APATH", "/sessions/ecstatic-cool-planck/mnt/outputs/optA.csv")

def all_inv(sim):
    s = sum(int(getattr(t, "inventory", 0)) for t in sim.traders)
    for m in sim.ccp.members.values():
        if not any(m is x for x in sim.traders): s += int(getattr(m, "inventory", 0))
    return s + int(getattr(sim.ccp, "inventory", 0))

def run(regime, c, direct):
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"
    G.CLOSEOUT_MODE = "transfer" if c == 1.0 and not direct else "firesale"  # conserve under defaults
    G.CLIENT_CLOSEOUT = "firesale"; G.CLOSEOUT_RECOVERY = 0.80
    FV = pd.read_csv(f"data/fv_{regime}.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps(regime)); s = st[10] if regime == "stressed" else 0
    e = st[min(len(st)-1, (30 if regime == "stressed" else 20))]; n = e - s
    V = V[s:s+n].copy(); SIG = SIG[s:s+n].copy()
    if c != 1.0:                                  # reverse-stress amplification of the path
        r = np.diff(np.log(V)); V = V[0]*np.exp(np.concatenate([[0.0], np.cumsum(r*c)])); SIG = SIG*c
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[0]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[regime], stressed=(regime == "stressed"))
    tr = RS.build_traders(p, seed=42); ccp = RS.build_clearing_tier(tr, p, seed=42, direct=direct)
    sim = Simulation(p, tr, seed=42, ccp=ccp, v_start=s); sim.v_array = V; sim.sigma_t_array = SIG
    inv_absmax = 0
    for t in range(n):
        sim.step()
        if t % 120 == 0: inv_absmax = max(inv_absmax, abs(all_inv(sim)))
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    cm_rows = ch[ch.kind.isin(["BCM","NBCM"])] if len(ch) else ch
    return dict(regime=regime, c=c, arm=("direct" if direct else "tiered"),
                dd_pct=round(100*(mid.min()/mid[0]-1),1) if len(mid) else float("nan"),
                client_def=len(cl),
                bcm_def=int(cm_rows[cm_rows.has_defaulted & (cm_rows.kind=="BCM")].agent_id.nunique()) if len(cm_rows) else 0,
                nbcm_def=int(cm_rows[cm_rows.has_defaulted & (cm_rows.kind=="NBCM")].agent_id.nunique()) if len(cm_rows) else 0,
                deepest_wf=int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0,
                ccp_cash_used_B=round((CCP_CASH-ccp.cash)/1e9,3), inv_absmax=inv_absmax)

CONFIGS = [("calm",1.0,False), ("stressed",1.0,False),
           ("stressed",2.0,False), ("stressed",2.0,True),
           ("stressed",2.5,False), ("stressed",2.5,True)]

if __name__ == "__main__":
    print("floors:", G.DIFFERENTIATED_FLOORS, "BCM", G.LR_FLOOR_BCM, "NBCM(cash/IM)", G.REG117_FLOOR_NBCM,
          "client", G.CLIENT_FREEZE_FLOOR)
    done = set()
    if os.path.exists(PATH):
        done = {(r.regime, float(r.c), r.arm) for r in pd.read_csv(PATH).itertuples()}
    budget = float(os.environ.get("BUDGET","40")); t0=time.time()
    for (rg,c,d) in CONFIGS:
        if (rg, c, "direct" if d else "tiered") in done: continue
        if time.time()-t0 > budget: print("budget hit; rerun to continue"); break
        r = run(rg,c,d)
        pd.DataFrame([r]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False)
        print(f"[{rg:8s} c={c} {r['arm']:6s}] dd={r['dd_pct']:6}% cl_def={r['client_def']:3d} "
              f"bcm_def={r['bcm_def']} nbcm_def={r['nbcm_def']} wf={r['deepest_wf']} "
              f"ccp_used={r['ccp_cash_used_B']}B inv|max|={r['inv_absmax']} | {time.time()-t0:.1f}s")
