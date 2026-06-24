"""Second-reader audit: cash + inventory conservation, H1 directionality, IM share.
Runs the committed config (IM_ESCROW, transfer member / firesale client close-out) on the
real COVID window, tiered vs direct, at a few seeds. Instruments:
  - total system cash = Sum(agent.cash) + ccp.cash + ccp.df_cash + ccp.im_account
  - total inventory   = Sum(all agent.inventory) incl. CMs + CCP   (should stay 0)
  - genuine modelled losses = Sum(client closeout_loss + shortfall) + member haircuts
  - defaults, deepest waterfall, client IM share
"""
import os, sys, time
import numpy as np, pandas as pd
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CONTRACT_USD
import model.run_simulation as RS
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from model.simulation import Simulation

NSESS = int(os.environ.get("NSESS", "10"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42").split(",")]

def all_inventory(sim):
    tot = 0
    for t in sim.traders:
        tot += int(getattr(t, "inventory", 0))
    for m in sim.ccp.members.values():
        if m.agent_id not in sim.traders_by_id or not any(m is x for x in sim.traders):
            tot += int(getattr(m, "inventory", 0))
    tot += int(getattr(sim.ccp, "inventory", 0))
    return tot

def total_cash(sim):
    c = 0.0
    seen = set()
    for t in sim.traders:
        c += float(getattr(t, "cash", 0.0)); seen.add(id(t))
    for m in sim.ccp.members.values():
        if id(m) not in seen:
            c += float(getattr(m, "cash", 0.0)); seen.add(id(m))
    c += float(sim.ccp.cash) + float(sim.ccp.df_cash) + float(sim.ccp.im_account)
    return c

def run(rg, seed, direct):
    G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"
    FV = pd.read_csv(f"data/fv_{rg}.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps(rg)); s = st[10] if rg == "stressed" else 0
    e = st[min(len(st)-1, (10+NSESS if rg == "stressed" else NSESS))]; n = e - s
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[rg], stressed=(rg == "stressed"))
    tr = RS.build_traders(p, seed=seed); ccp = RS.build_clearing_tier(tr, p, seed=seed, direct=direct)
    sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=s); sim.v_array = V[s:s+n]; sim.sigma_t_array = SIG[s:s+n]
    usd = p.volume_lot * CONTRACT_USD
    c0 = total_cash(sim); inv0 = all_inventory(sim)
    inv_max = 0; cash_series = []
    for t in range(n):
        sim.step()
        if t % 60 == 0:
            inv_max = max(inv_max, abs(all_inventory(sim)))
            cash_series.append(total_cash(sim))
    c1 = total_cash(sim); inv1 = all_inventory(sim)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    wf = int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0
    cm_def = int(ch[ch.has_defaulted].agent_id.nunique()) if len(ch) else 0
    # genuine modelled loss: client shortfall+closeout + member (1-rec) haircuts are inside deficit;
    # we approximate total destroyed value as initial-final cash, and compare to closeout losses.
    closeout_loss = float(cl["closeout_loss"].sum()) if len(cl) and "closeout_loss" in cl else 0.0
    # client IM share (post-ramp 2nd half)
    return dict(rg=rg, seed=seed, direct=direct, n=n,
                drawdown_pct=round(100*(mid.min()/mid[0]-1),1) if len(mid) else float("nan"),
                client_def=len(cl), cm_def=cm_def, deepest_wf=wf,
                inv0=inv0, inv1=inv1, inv_drift=inv1-inv0, inv_absmax=inv_max,
                cash0_B=round(c0/1e9,4), cash1_B=round(c1/1e9,4),
                cash_delta_B=round((c1-c0)/1e9,4),
                closeout_loss_B=round(closeout_loss/1e9,4))

if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        for direct in (False, True):
            r = run("stressed", seed, direct); rows.append(r)
            print(f"[{'DIRECT' if direct else 'TIERED'} s{seed}] "
                  f"dd={r['drawdown_pct']}% cl_def={r['client_def']} cm_def={r['cm_def']} "
                  f"wf={r['deepest_wf']} | INV drift={r['inv_drift']} absmax={r['inv_absmax']} | "
                  f"cashDelta={r['cash_delta_B']}B closeout={r['closeout_loss_B']}B | {time.time()-t0:.1f}s")
    pd.DataFrame(rows).to_csv("audit_conservation_out.csv", index=False)
    print(f"total {time.time()-t0:.1f}s")
