import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model.globals import ModelParams, V0, CALIBRATED
from model import globals as G
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from model.run_simulation import build_traders, build_clearing_tier

# Real tiered structure (seed 42).
p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                n_bcm_with_clients=5, v0=V0["stressed"], tick_size=0.25, dt_minutes=1.0,
                **CALIBRATED["stressed"], stressed=True)
traders = build_traders(p, seed=42)
ccp = build_clearing_tier(traders, p, seed=42)
by_id = {t.agent_id: t for t in traders}

C = {"CCP": "#C0392B", "BCMc": "#1A5276", "BCMo": "#A9CCE3", "NBCM": "#E67E22",
     "FT": "#229954", "MT": "#7D3C98", "ZI": "#909497"}
bcms  = [m for m in ccp.members.values() if isinstance(m, BankingClearingMember)]
nbcms = [m for m in ccp.members.values() if isinstance(m, NonBankingClearingMember)]
own  = [m for m in bcms if not m.client_ids]
cli  = [m for m in bcms if m.client_ids]
carriers = [m for pair in zip(cli, nbcms) for m in pair]   # alternate BCMc, NBCM (10)
_ordered, _oi = [], 0
for _slot in range(len(own) + len(carriers)):
    if _slot % 3 == 2 and _oi < len(own):
        _ordered.append(own[_oi]); _oi += 1
    elif carriers:
        _ordered.append(carriers.pop(0))
    else:
        _ordered.append(own[_oi]); _oi += 1
ordered = _ordered                                # own-account woven in every 3rd slot
def mcol(m):
    if isinstance(m, NonBankingClearingMember): return C["NBCM"]
    return C["BCMc"] if m.client_ids else C["BCMo"]
def ctype(c):
    return "MT" if isinstance(c, MomentumTrader) else ("ZI" if isinstance(c, ZeroIntelligenceTrader) else "FT")

cmax = max(m.cash for m in ccp.members.values())
def msz(c): return 300 + 3200 * (c / cmax)        # member node area ~ capital
# Client node area ~ its initial balance (cash): FT asset-managers largest, ZI noise smallest.
ccmax = max((by_id[cid].cash for m in cli + nbcms for cid in m.client_ids), default=1.0)
def csz(cash): return 14 + 95 * (cash / ccmax)

N = len(ordered); R1, R2 = 1.0, 2.0
ang = {m.agent_id: 2 * np.pi * i / N for i, m in enumerate(ordered)}
mpos = {m.agent_id: (R1 * np.cos(ang[m.agent_id]), R1 * np.sin(ang[m.agent_id])) for m in ordered}

fig, ax = plt.subplots(figsize=(12.5, 12.5)); ax.axis("off"); ax.set_aspect("equal")
fig.patch.set_facecolor("white"); lim = R2 + 0.55; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

for m in ordered:                                  # member -> CCP spokes
    x, y = mpos[m.agent_id]; ax.plot([x, 0], [y, 0], color="#D5D8DC", lw=1.0, zorder=1)

wedge = (2 * np.pi / N) * 0.82
for m in cli + nbcms:                              # clients fanned in the member's wedge, 2 rows
    a0 = ang[m.agent_id]; cids = m.client_ids; k = len(cids)
    offs = np.linspace(-wedge / 2, wedge / 2, k) if k > 1 else [0.0]
    mx, my = mpos[m.agent_id]
    for j, cid in enumerate(cids):
        a = a0 + offs[j]; r = R2 + (0.17 if j % 2 else 0.0)
        cx, cy = r * np.cos(a), r * np.sin(a)
        ax.plot([mx, cx], [my, cy], color=C[ctype(by_id[cid])], lw=0.5, alpha=0.4, zorder=1)
        ax.scatter([cx], [cy], s=csz(by_id[cid].cash), color=C[ctype(by_id[cid])],
                   edgecolors="white", linewidths=0.5, zorder=3)

for m in ordered:                                  # member nodes (size ~ capital)
    x, y = mpos[m.agent_id]
    ax.scatter([x], [y], s=msz(m.cash), color=mcol(m), edgecolors="#34495E", linewidths=1.1, zorder=4)
ax.scatter([0], [0], s=3400, color=C["CCP"], edgecolors="#34495E", linewidths=1.6, zorder=5)
ax.text(0, 0, "CCP", ha="center", va="center", color="white", fontsize=15, fontweight="bold", zorder=6)

# Legend — node types + the solvency rule each is held to. Both client-clearing tiers (banking and
# non-banking) share the FCM rule (CFTC Reg 1.17, cash/IM on the client book); a BCM's house book is
# bounded separately by the static gross-leverage cap.
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor=C["CCP"],markersize=18,label='CCP'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["BCMc"],markersize=16,
              label=f'Banking CM — Clients + House  (Reg 1.17 cash/IM $\\geq$ {G.REG117_FLOOR_NBCM:.0%} + bank Basel-LR cash/exp $\\geq$ {G.LR_FLOOR_BCM:.2%})'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["BCMo"],markersize=16,
              label=f'Banking CM — House Only  (Basel-LR; own book $\\leq${G.POSITION_LIMIT_X:.0f}$\\times$cash)'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["NBCM"],markersize=14,
              label=f'Non-Banking CM  (Reg 1.17 cash/IM $\\geq$ {G.REG117_FLOOR_NBCM:.0%}; no Basel-LR)'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["FT"],markersize=11,label='Fundamental Client'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["MT"],markersize=11,label='Momentum Client'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor=C["ZI"],markersize=11,
              label=f'Noise Client     (clients freeze at cash/exp $\\leq$ {G.CLIENT_FREEZE_FLOOR:.0%})')]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2,
          fontsize=12, frameon=False, handletextpad=0.4, columnspacing=1.4)
ax.set_title("Tiered Clearing Network   (member node $\\propto$ capital, client node $\\propto$ balance)\n"
             "10 banking + 5 non-banking CMs  ·  90 clients  ·  leverage-balanced assignment",
             fontsize=13, fontweight="bold", pad=8)
fig.savefig("clearing_topology.png", dpi=130, facecolor="white")
print("wrote clearing_topology.png | BCMc=%d BCMo=%d NBCM=%d clients=%d | NBCM $%.1f-%.1fB unified=%s"
      % (len(cli), len(own), len(nbcms), sum(len(m.client_ids) for m in cli+nbcms),
         G.NBCM_CASH_RANGE[0]/1e9, G.NBCM_CASH_RANGE[1]/1e9, G.UNIFIED_CLIENT_FLOOR))
