import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Probe WHICH design choices make the CCP so resilient, by re-running the deepest GBM stressed
episodes under each candidate lever and measuring defaults + how deep the waterfall draws.

Waterfall levels: 0 = seized IM only (L0); 1 = defaulter DF; 2 = SITG; 3 = pooled (mutualised)
DF; 4 = survivor cash; 5 = CCP cash. Goal: a config giving >=3-4 defaults and deepest >= 2-3
(SITG / pooled DF tapped).

Arms (vs the reactive/tiered/0.80-recovery baseline):
  flat05 / flat03  -> NON-procyclical flat IM (5% / 3%): kills the rising-IM coverage
  direct           -> direct clearing: removes the member-capital buffer ahead of the DF
  flat05_dir       -> both
  lowrec05         -> member close-out recovery 0.80 -> 0.50 (bigger default loss)

Usage:  BUDGET=34 GSEED=110 ARMS=baseline,flat05,flat03,direct,flat05_dir,lowrec05 python3 gbm_lever.py
"""
import os, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, CCP_CASH
import model.run_simulation as RS
from model.simulation import Simulation

PATH = os.environ.get("QPATH", "output/verify_rebalance/gbm_lever.csv")
ASEED = int(os.environ.get("ASEED", "42"))
GSEEDS = [int(x) for x in os.environ.get("GSEEDS", os.environ.get("GSEED", "110")).split(",")]
ARMS = {
    "baseline":    dict(mode="reactive", flat=None, direct=False, rec=0.80),
    "flat05":      dict(mode="flat",     flat=0.05, direct=False, rec=0.80),
    "flat03":      dict(mode="flat",     flat=0.03, direct=False, rec=0.80),
    "direct":      dict(mode="reactive", flat=None, direct=True,  rec=0.80),
    "flat05_dir":  dict(mode="flat",     flat=0.05, direct=True,  rec=0.80),
    "lowrec05":    dict(mode="reactive", flat=None, direct=False, rec=0.50),
    "combo":       dict(mode="flat",     flat=0.03, direct=True,  rec=0.50),
    # CCP open-market Almgren-Chriss disposal of the defaulter's book (CLOSEOUT_MODE='firesale'):
    # endogenous loss (book-walk in a thin crash LOB) + price-impact feedback onto survivors.
    "ccp_fs":      dict(mode="reactive", flat=None, direct=False, rec=0.80, closeout="firesale"),
    "ccp_fs_dir":  dict(mode="reactive", flat=None, direct=True,  rec=0.80, closeout="firesale"),
    "rec06":       dict(mode="reactive", flat=None, direct=False, rec=0.60),
    "rec06_dir":   dict(mode="reactive", flat=None, direct=True,  rec=0.60),
}


def run(arm, gseed):
    a = ARMS[arm]
    fv_path = f"data/fv_gbm_stressed_s{gseed}.csv"
    G.FV_CSV["stressed"] = fv_path
    try: G.day_start_steps.cache_clear()
    except Exception: pass
    G.IM_MODE = a["mode"]
    if a["flat"] is not None: G.IM_FLAT_FRAC = a["flat"]
    G.CLIENT_MARGIN_NETTING = "gross"
    G.CLOSEOUT_RECOVERY = a["rec"]
    G.CLOSEOUT_MODE = a.get("closeout", "transfer")
    V = pd.read_csv(fv_path)["V_smooth"].to_numpy(float)
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[0]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED["stressed"], stressed=True)
    p.fv_csv = fv_path
    tr = RS.build_traders(p, seed=ASEED)
    ccp = RS.build_clearing_tier(tr, p, seed=ASEED, direct=a["direct"])
    sim = Simulation(p, tr, seed=ASEED, ccp=ccp, v_start=0)
    sim.run(len(V))
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    wf = int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0
    cm_def = int(ch[ch.has_defaulted].agent_id.nunique()) if len(ch) else 0
    nbcm_def = int(ch[ch.has_defaulted & (ch.kind == "NBCM")].agent_id.nunique()) if len(ch) else 0
    lvl_hist = ch.waterfall_level.dropna().astype(int) if len(ch) else pd.Series([], dtype=int)
    return dict(arm=arm, gseed=gseed,
                client_def=len(cl), bcm_def=cm_def - nbcm_def, nbcm_def=nbcm_def,
                deepest_wf=wf, n_cycles_wf_ge1=int((lvl_hist >= 1).sum()),
                n_cycles_sitg_ge2=int((lvl_hist >= 2).sum()),
                ccp_cash_used_B=round((CCP_CASH - ccp.cash) / 1e9, 3))


def main():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    done = set()
    if os.path.exists(PATH):
        done = {(r.arm, int(r.gseed)) for r in pd.read_csv(PATH).itertuples()}
    arms = os.environ.get("ARMS", ",".join(ARMS)).split(",")
    todo = [(a, g) for g in GSEEDS for a in arms if (a, g) not in done]
    budget = float(os.environ.get("BUDGET", "34")); t0 = time.time(); k = 0
    for (a, g) in todo:
        if time.time() - t0 > budget:
            print(f"budget hit; wrote {k}; {len(todo)-k} left — rerun"); return
        pd.DataFrame([run(a, g)]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False); k += 1
    df = pd.read_csv(PATH).sort_values(["gseed", "deepest_wf"])
    pd.set_option("display.width", 220)
    print("\nLEVER TEST on GBM deep episode(s) (deepest_wf: 0=IM,1=defDF,2=SITG,3=pooledDF,4=survivor,5=CCP):\n",
          df.to_string(index=False))


if __name__ == "__main__":
    main()
