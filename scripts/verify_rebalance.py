import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Verify the coordinated balance-sheet rebalance (large clients + 2x BCM prop, reading the
committed globals). Checks, per regime, over a RAMPED window (books filled), multi-seed:
  (a) client IM share ~ CME ~82% direction (target material majority),
  (b) calm is CLEAN  (no clearer breaches, no defaults),
  (c) stressed produces the CLIENT-DEFAULT contagion + the BCM breaches via its client book.
Resumable: writes per (regime, seed) to output/verify_rebalance/rows.csv; rerun until table prints.

Usage:  BUDGET=37 NSESS=12 python3 verify_rebalance.py
"""
import os, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps
import model.run_simulation as RS
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from model.simulation import Simulation

PATH = os.environ.get("VPATH", "output/verify_rebalance/rows.csv")
NSESS = int(os.environ.get("NSESS", "12"))
SCALE = float(os.environ.get("SCALE", "1.0"))   # multiplies the committed (6x) client cash
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,7,123,2024").split(",")]


def typ(a):
    if isinstance(a, BankingClearingMember): return "BCM"
    if isinstance(a, NonBankingClearingMember): return "NBCM"
    if isinstance(a, MomentumTrader): return "MT"
    if isinstance(a, ZeroIntelligenceTrader): return "ZI"
    if isinstance(a, FundamentalTrader): return "FT"
    return "x"


def run(rg, seed, nsess=NSESS):
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"
    RS.FT_CLIENT_CASH = (G.FT_CLIENT_CASH[0]*SCALE, G.FT_CLIENT_CASH[1]*SCALE)
    RS.MT_CLIENT_CASH = (G.MT_CLIENT_CASH[0]*SCALE, G.MT_CLIENT_CASH[1]*SCALE)
    RS.ZI_CLIENT_CASH = (G.ZI_CLIENT_CASH[0]*SCALE, G.ZI_CLIENT_CASH[1]*SCALE)
    _hm = float(os.environ.get("HOUSE_MARGIN", "0"))   # client leverage cap = 1/house_margin
    if _hm > 0: G.CCP_CALIBRATION["im_percent"] = _hm  # 0.20 -> 5x, 0.15 -> 6.67x
    FV = pd.read_csv(f"data/fv_{rg}.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps(rg)); s = st[10] if rg == "stressed" else 0
    e = st[min(len(st)-1, (10+nsess if rg == "stressed" else nsess))]; n = e - s
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[rg], stressed=(rg == "stressed"))
    tr = RS.build_traders(p, seed=seed); ccp = RS.build_clearing_tier(tr, p, seed=seed, direct=False)
    sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=s); sim.v_array = V[s:s+n]; sim.sigma_t_array = SIG[s:s+n]
    tymap = {a.agent_id: typ(a) for a in tr}
    for m in ccp.members.values(): tymap.setdefault(m.agent_id, typ(m))
    cli, bcm, ncyc, ramp = 0.0, 0.0, 0, max(1, n//2)   # average IM over 2nd half (post-ramp)
    for t in range(n):
        sim.step()
        if t % 60 == 0 and t >= ramp:
            ncyc += 1
            for a in tr:
                if tymap[a.agent_id] in ("FT", "MT", "ZI"): cli += getattr(a, "_posted_im", 0.0)
            for m in ccp.members.values():
                if tymap[m.agent_id] == "BCM": bcm += getattr(m, "_posted_im", 0.0)
    cli /= max(ncyc, 1); bcm /= max(ncyc, 1)
    share = 100*cli/(cli+bcm) if (cli+bcm) > 0 else 0
    ch = pd.DataFrame(sim.clearing_history)
    cm_rows = ch[ch.kind.isin(["BCM", "NBCM"])] if len(ch) else ch
    bcmr = cm_rows[cm_rows.kind == "BCM"] if len(cm_rows) else cm_rows
    nbcmr = cm_rows[cm_rows.kind == "NBCM"] if len(cm_rows) else cm_rows
    breach = int((cm_rows.capital_ratio <= 0.08).sum()) if len(cm_rows) else 0
    bcm_k = float(bcmr.capital_ratio.min()) if len(bcmr) else float("nan")
    nbcm_k = float(nbcmr.capital_ratio.min()) if len(nbcmr) else float("nan")
    bcm_mk = float(bcmr.capital_ratio.mean()) if len(bcmr) else float("nan")
    nbcm_mk = float(nbcmr.capital_ratio.mean()) if len(nbcmr) else float("nan")
    bcm_def = int(bcmr[bcmr.has_defaulted].agent_id.nunique()) if len(bcmr) else 0
    nbcm_def = int(nbcmr[nbcmr.has_defaulted].agent_id.nunique()) if len(nbcmr) else 0
    wf = int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0
    return dict(regime=rg, seed=seed, sessions=nsess,
                client_im_B=round(cli/1e9, 2), bcm_im_B=round(bcm/1e9, 2),
                total_im_B=round((cli+bcm)/1e9, 2), client_share_pct=round(share, 1),
                breaches=breach, bcm_min_k=round(bcm_k, 3), bcm_mean_k=round(bcm_mk, 3),
                nbcm_min_k=round(nbcm_k, 3), nbcm_mean_k=round(nbcm_mk, 3),
                client_def=len(pd.DataFrame(sim.client_history)), bcm_def=bcm_def, nbcm_def=nbcm_def,
                max_wf=wf)


def main():
    os.makedirs("output/verify_rebalance", exist_ok=True)
    done = set()
    if os.path.exists(PATH):
        old = pd.read_csv(PATH); done = {(r.regime, r.seed) for r in old.itertuples()}
    todo = [(rg, sd) for rg in ("calm", "stressed") for sd in SEEDS if (rg, sd) not in done]
    budget = float(os.environ.get("BUDGET", "37")); t0 = time.time(); k = 0
    for (rg, sd) in todo:
        if time.time()-t0 > budget:
            print(f"budget hit; wrote {k}; {len(todo)-k} left — rerun"); return
        pd.DataFrame([run(rg, sd)]).to_csv(PATH, mode="a", header=not os.path.exists(PATH), index=False); k += 1
    df = pd.read_csv(PATH).sort_values(["regime", "seed"])
    g = df.groupby("regime").agg(share_pct=("client_share_pct", "mean"),
                                 total_im_B=("total_im_B", "mean"),
                                 breaches=("breaches", "sum"),
                                 bcm_min_k=("bcm_min_k", "min"), bcm_mean_k=("bcm_mean_k", "mean"),
                                 nbcm_min_k=("nbcm_min_k", "min"),
                                 client_def=("client_def", "sum"), bcm_def=("bcm_def", "sum"),
                                 nbcm_def=("nbcm_def", "sum"), max_wf=("max_wf", "max")).round(2)
    if {"calm", "stressed"}.issubset(set(g.index)):
        g.loc["procyc x", "total_im_B"] = round(g.loc["stressed", "total_im_B"] / max(g.loc["calm", "total_im_B"], 1e-9), 2)
    pd.set_option("display.width", 240)
    print("\nPER (regime, seed):\n", df.to_string(index=False))
    print("\nBY REGIME (share/IM/kappa = mean; defaults & breaches summed; 'procyc x' = stress/calm IM):\n", g.to_string())
    print(f"\nglobals in effect: POSITION_LIMIT_X={G.POSITION_LIMIT_X}  "
          f"FT={RS.FT_CLIENT_CASH}  NBCM={RS.NBCM_CASH_RANGE}")


if __name__ == "__main__":
    main()
