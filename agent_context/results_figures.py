"""Results-section figures (clearing / contagion). Committed config. Produces:
  fig_res_margin.png   — H2: posted IM by agent type through the COVID stress (procyclicality)
  fig_res_h1.png       — H1: deepest waterfall level vs drawdown, tiered vs direct (severity sweep)
  fig_res_share_net.png— client IM share by type; gross-vs-net total IM (NET)
Modest defaults (scale C_GRID / SEEDS for the thesis). Pure numpy/pandas/mpl."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CONTRACT_USD, CCP_CASH
import model.run_simulation as RS
from model.simulation import Simulation
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)

OUT = "output/figs"; os.makedirs(OUT, exist_ok=True)
C_GRID = [float(x) for x in os.environ.get("C_GRID", "1.0,1.5,2.0,2.5").split(",")]
SEED = int(os.environ.get("SEED", "42"))
COL = {"FT": "#229954", "MT": "#7D3C98", "ZI": "#909497", "BCM": "#1A5276"}

def _typ(a):
    if isinstance(a, BankingClearingMember): return "BCM"
    if isinstance(a, NonBankingClearingMember): return "NBCM"
    if isinstance(a, MomentumTrader): return "MT"
    if isinstance(a, ZeroIntelligenceTrader): return "ZI"
    return "FT"

def run(c=1.0, direct=False, netting="gross", margin="reactive", collect_im=False):
    G.IM_MODE = "flat" if margin == "flat" else "reactive"
    if margin == "flat": G.IM_FLAT_FRAC = 0.05
    G.CLIENT_MARGIN_NETTING = netting
    G.CLOSEOUT_MODE = "transfer"; G.CLIENT_CLOSEOUT = "firesale"; G.CLOSEOUT_RECOVERY = 0.80
    FV = pd.read_csv("data/fv_stressed.csv"); V = FV["V_smooth"].to_numpy(float); SIG = FV["sigma_t"].to_numpy(float)
    st = list(day_start_steps("stressed")); s = st[10]
    e = st[min(len(st)-1, int(os.environ.get("WIN", "30")))]; n = e - s
    V = V[s:s+n].copy(); SIG = SIG[s:s+n].copy()
    if c != 1.0:
        r = np.diff(np.log(V)); V = V[0]*np.exp(np.concatenate([[0.0], np.cumsum(r*c)])); SIG = SIG*c
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
                    v0=float(V[0]), tick_size=0.25, dt_minutes=1.0, **CALIBRATED["stressed"], stressed=True)
    tr = RS.build_traders(p, seed=SEED); ccp = RS.build_clearing_tier(tr, p, seed=SEED, direct=direct)
    sim = Simulation(p, tr, seed=SEED, ccp=ccp, v_start=s); sim.v_array = V; sim.sigma_t_array = SIG
    tymap = {a.agent_id: _typ(a) for a in tr}; [tymap.setdefault(m.agent_id, _typ(m)) for m in ccp.members.values()]
    keys = ["FT", "MT", "ZI", "BCM"]; im_ts = {k: [] for k in keys}
    for t in range(n):
        sim.step()
        if collect_im and t % 60 == 0:
            agg = {k: 0.0 for k in keys}
            for a in tr:
                if tymap[a.agent_id] in agg: agg[tymap[a.agent_id]] += getattr(a, "_posted_im", 0.0)
            for m in ccp.members.values():
                if tymap[m.agent_id] == "BCM": agg["BCM"] += getattr(m, "_posted_im", 0.0)
            for k in keys: im_ts[k].append(agg[k])
    ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    cmr = ch[ch.kind.isin(["BCM", "NBCM"])] if len(ch) else ch
    out = dict(c=c, direct=direct, drawdown=100*(mid.min()/mid[0]-1) if len(mid) else np.nan,
               client_def=len(cl),
               bcm_def=int(cmr[cmr.has_defaulted & (cmr.kind == "BCM")].agent_id.nunique()) if len(cmr) else 0,
               nbcm_def=int(cmr[cmr.has_defaulted & (cmr.kind == "NBCM")].agent_id.nunique()) if len(cmr) else 0,
               deepest_wf=int(ch.waterfall_level.max()) if len(ch) and ch.waterfall_level.notna().any() else 0,
               total_im=float(ch.groupby("t").initial_margin.sum().mean()) if len(ch) else 0.0,
               im_ts={k: np.array(v) for k, v in im_ts.items()})
    return out

# ---- Fig 1 (H2): posted IM by type through the COVID stress (stacked) ----
covid = run(c=1.0, collect_im=True)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
order = ["ZI", "MT", "FT", "BCM"]; lab = {"ZI": "Noise clients", "MT": "Momentum clients",
        "FT": "Fundamental clients", "BCM": "Bank members (house)"}
T = np.arange(len(covid["im_ts"]["FT"])) * 60 / 390.0
ax.stackplot(T, [covid["im_ts"][k]/1e9 for k in order], labels=[lab[k] for k in order],
             colors=[COL[k] for k in order], alpha=.9)
ax.set_xlabel("session (COVID window)"); ax.set_ylabel("posted initial margin ($bn)")
ax.set_title("H2 — Posted initial margin by participant type through the stress"); ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT}/fig_res_margin.png", dpi=130); plt.close(fig)

# ---- Fig 2 (H1): deepest waterfall vs drawdown, tiered vs direct ----
sweep = {arm: [run(c=c, direct=(arm == "direct")) for c in C_GRID] for arm in ("tiered", "direct")}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for arm, col, mk in (("tiered", "#1A5276", "o"), ("direct", "#C0392B", "s")):
    dd = [r["drawdown"] for r in sweep[arm]]; wf = [r["deepest_wf"] for r in sweep[arm]]
    ax.plot(dd, wf, mk + "-", color=col, lw=2, ms=9, label=arm)
ax.set_xlabel("drawdown (%)"); ax.set_ylabel("deepest waterfall level reached")
ax.set_yticks(range(0, 6)); ax.set_yticklabels(["0 IM", "1 def-DF", "2 SITG", "3 pooled-DF", "4 survivor", "5 CCP"])
ax.invert_xaxis()
ax.set_title("H1 — Mutualisation threshold: tiered localises, direct reaches the fund")
ax.legend(frameon=False); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{OUT}/fig_res_h1.png", dpi=130); plt.close(fig)

# ---- Fig 3: client IM share by type + gross-vs-net total IM ----
share = {k: covid["im_ts"][k][len(covid["im_ts"][k])//2:].mean() for k in ["FT", "MT", "ZI", "BCM"]}
net = run(c=1.0, netting="net")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
tot = sum(share.values())
a1.bar([lab[k] for k in order], [100*share[k]/tot for k in order], color=[COL[k] for k in order])
a1.set_ylabel("share of posted IM (%)"); a1.set_title(f"IM share by type (clients ≈ {100*(tot-share['BCM'])/tot:.0f}%)")
a1.tick_params(axis="x", rotation=20); a1.grid(axis="y", alpha=.3)
a2.bar(["gross\n(US/CME)", "net\n(EU omnibus)"], [covid["total_im"]/1e9, net["total_im"]/1e9],
       color=["#1A5276", "#e8734c"]); a2.set_ylabel("mean system IM ($bn)")
a2.set_title("NET — gross vs net client margining"); a2.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_res_share_net.png", dpi=130); plt.close(fig)

print("H1 sweep:")
for arm in ("tiered", "direct"):
    print(" ", arm, [(round(r["drawdown"]), r["deepest_wf"], f"B{r['bcm_def']}/N{r['nbcm_def']}") for r in sweep[arm]])
print(f"saved 3 figs -> {OUT}/  | client IM share {100*(tot-share['BCM'])/tot:.0f}% | gross {covid['total_im']/1e9:.1f}B net {net['total_im']/1e9:.1f}B")
