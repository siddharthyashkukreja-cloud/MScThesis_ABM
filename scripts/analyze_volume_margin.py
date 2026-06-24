import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Volume-by-trader-type, sim-vs-empirical volume, and posted-margin-by-type analysis + graphs.
Outputs -> output/volume_margin/ : volume_by_type.png, volume_sim_vs_empirical.png,
margin_by_type.png, margin_timeseries_stressed.png, summary.csv.

Volume is attributed to BOTH sides of each fill (participation volume). Margin is the escrowed
initial margin posted by each agent type (BCM own book; FT/MT/ZI clients post their own).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CONTRACT_USD
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from model.run_simulation import build_traders, build_clearing_tier
from model.simulation import Simulation

EMP_VOL = {"calm": 2000, "stressed": 3500}     # empirical RTH front-month ES per-minute volume
COL = {"FT": "#1f77b4", "MT": "#ff7f0e", "ZI": "#2ca02c", "BCM": "#d62728",
       "NBCM": "#9467bd", "CCP": "#7f7f7f"}
os.makedirs("output/volume_margin", exist_ok=True)
G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"


def typ(a):
    if isinstance(a, BankingClearingMember): return "BCM"
    if isinstance(a, NonBankingClearingMember): return "NBCM"
    if isinstance(a, MomentumTrader): return "MT"
    if isinstance(a, ZeroIntelligenceTrader): return "ZI"
    if isinstance(a, FundamentalTrader): return "FT"
    return "other"


def run(rg, n_days=10):
    FV = pd.read_csv(f"data/fv_{rg}.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps(rg))
    s = st[10] if rg == "stressed" else 0
    e = st[min(len(st) - 1, (20 if rg == "stressed" else n_days))]
    n = e - s
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[rg], stressed=(rg == "stressed"))
    lot = p.volume_lot
    tr = build_traders(p, seed=42); ccp = build_clearing_tier(tr, p, seed=42, direct=False)
    sim = Simulation(p, tr, seed=42, ccp=ccp, v_start=s); sim.v_array = V[s:s + n]; sim.sigma_t_array = SIG[s:s + n]
    tymap = {a.agent_id: typ(a) for a in tr}
    for m in ccp.members.values(): tymap.setdefault(m.agent_id, typ(m))
    tymap[ccp.ccp_id] = "CCP"
    vol = {}; step_vol = []
    im_keys = ["BCM", "FT", "MT", "ZI"]; im_ts = {k: [] for k in im_keys}
    for t in range(n):
        sim.step()
        sv = 0
        for f in sim.lob.step_fills:
            q = f.qty; sv += q
            for sid in (f.buyer_id, f.seller_id):
                ty = tymap.get(sid, "other"); vol[ty] = vol.get(ty, 0) + q
        step_vol.append(sv)
        if t % 60 == 0:
            agg = {k: 0.0 for k in im_keys}
            for a in tr:
                ty = tymap[a.agent_id]
                if ty in agg: agg[ty] += getattr(a, "_posted_im", 0.0)
            for m in ccp.members.values():
                if tymap[m.agent_id] == "BCM": agg["BCM"] += getattr(m, "_posted_im", 0.0)
            for k in im_keys: im_ts[k].append(agg[k])
    permin = float(np.mean(step_vol) * lot)
    im_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in im_ts.items()}
    return dict(rg=rg, vol=vol, permin_sim=permin, permin_emp=EMP_VOL[rg],
                im_ts=im_ts, im_mean=im_mean, lot=lot, n=n)


R = {rg: run(rg) for rg in ("calm", "stressed")}
order = ["ZI", "FT", "MT", "BCM"]

# ---- print + csv ----
rows = []
for rg, d in R.items():
    tot = sum(d["vol"].values())
    print(f"\n[{rg}] sim per-min contract volume {d['permin_sim']:.0f}  vs empirical {d['permin_emp']}  "
          f"(ratio {d['permin_sim']/d['permin_emp']:.2f})")
    for k in order:
        sh = 100 * d["vol"].get(k, 0) / tot if tot else 0
        print(f"    {k:4s} volume share {sh:5.1f}%   mean posted IM ${d['im_mean'].get(k,0)/1e9:5.2f}B")
        rows.append(dict(regime=rg, type=k, vol_share_pct=round(sh, 2),
                         mean_posted_im_B=round(d["im_mean"].get(k, 0) / 1e9, 3)))
pd.DataFrame(rows).to_csv("output/volume_margin/summary.csv", index=False)

# ---- Fig 1: volume share by type ----
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(order)); w = 0.38
for i, rg in enumerate(("calm", "stressed")):
    tot = sum(R[rg]["vol"].values())
    sh = [100 * R[rg]["vol"].get(k, 0) / tot for k in order]
    ax.bar(x + (i - 0.5) * w, sh, w, label=rg, color=("#4c9be8" if i == 0 else "#e8734c"))
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("share of trading volume (%)")
ax.set_title("Trading-volume share by agent type"); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("output/volume_margin/volume_by_type.png", dpi=130); plt.close(fig)

# ---- Fig 2: sim vs empirical per-min volume ----
fig, ax = plt.subplots(figsize=(6, 4.2))
x = np.arange(2)
ax.bar(x - 0.2, [R["calm"]["permin_sim"], R["stressed"]["permin_sim"]], 0.4, label="simulated", color="#4c9be8")
ax.bar(x + 0.2, [R["calm"]["permin_emp"], R["stressed"]["permin_emp"]], 0.4, label="empirical ES", color="#888")
ax.set_xticks(x); ax.set_xticklabels(["calm", "stressed"]); ax.set_ylabel("contracts / minute")
ax.set_title("Per-minute contract volume: simulated vs empirical"); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("output/volume_margin/volume_sim_vs_empirical.png", dpi=130); plt.close(fig)

# ---- Fig 3: mean posted margin by type ----
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(order)); w = 0.38
for i, rg in enumerate(("calm", "stressed")):
    vals = [R[rg]["im_mean"].get(k, 0) / 1e9 for k in order]
    ax.bar(x + (i - 0.5) * w, vals, w, label=rg, color=("#4c9be8" if i == 0 else "#e8734c"))
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("mean posted initial margin ($bn)")
ax.set_title("Initial margin posted at the CCP, by agent type"); ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("output/volume_margin/margin_by_type.png", dpi=130); plt.close(fig)

# ---- Fig 4: posted-margin time series (stressed), stacked ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
d = R["stressed"]; T = np.arange(len(d["im_ts"]["BCM"])) * 60 / 390.0   # sessions
ax.stackplot(T, [np.array(d["im_ts"][k]) / 1e9 for k in order],
             labels=order, colors=[COL[k] for k in order], alpha=.85)
ax.set_xlabel("session (stressed / COVID window)"); ax.set_ylabel("posted initial margin ($bn)")
ax.set_title("Posted initial margin through the stress, by agent type"); ax.legend(loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("output/volume_margin/margin_timeseries_stressed.png", dpi=130); plt.close(fig)
print("\nsaved 4 figures + summary.csv -> output/volume_margin/")
