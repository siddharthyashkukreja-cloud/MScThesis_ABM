#!/usr/bin/env python3
"""
Mechanism & robustness figures for Results §5.3, CSV-driven (fast, no model re-run).

Reads the H1 experiment logs (150 seeds, stressed):
    output/thesis_final/experiments/client_defaults.csv  (kind, assumed_pos, ... , margin)
    output/thesis_final/experiments/waterfall_events.csv  (level, L1..L5, deficit, margin)
    output/thesis_final/experiments/client_freezes.csv    (reason, margin)
and writes three figures to output/results/ (also copies to FIG_DEST if set):
    mech_position.png        fewer-but-fatter-tailed defaulted positions (the freeze-trap)
    mech_loss_freeze.png     where the loss is absorbed (waterfall layers) + why clients freeze (contagion)
    mech_defaults_by_type.png who bears the defaults (FT / MT / ZI) -- relegated to the appendix

Run:  PYTHONPATH=. python3 scripts/mechanism_figures.py
      FIG_DEST=/path/to/MSc_Thesis/Figures PYTHONPATH=. python3 scripts/mechanism_figures.py
"""
import os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Thesis figures carry a LaTeX caption: suppress in-figure headers (suptitle) + boxed notes.
import matplotlib.figure as _mfig, matplotlib.axes as _maxes
_mfig.Figure.suptitle = lambda *a, **k: None
_maxes.Axes.text = (lambda _o: (lambda self, *a, **k: None if "bbox" in k else _o(self, *a, **k)))(_maxes.Axes.text)

E   = "output/thesis_final/experiments"
OUT = "output/results"; os.makedirs(OUT, exist_ok=True)
FIG_DEST = os.environ.get("FIG_DEST", "")
REGIME, N_SEEDS = "stressed", 150
ARMS  = ["flat04", "flat08", "flat12", "reactive"]
LABEL = {"flat04": "flat-4%", "flat08": "flat-8%", "flat12": "flat-12%", "reactive": "reactive"}
FLAT, REACT, GREY = "#1A5276", "#E07A3C", "#9aa5b1"
BARCOL = [FLAT, FLAT, FLAT, REACT]
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.size": 11})

cd = pd.read_csv(f"{E}/client_defaults.csv"); cd = cd[cd.regime == REGIME].copy()
cd["abspos"] = cd["assumed_pos"].abs()
wf = pd.read_csv(f"{E}/waterfall_events.csv"); wf = wf[wf.regime == REGIME].copy()
fz = pd.read_csv(f"{E}/client_freezes.csv"); fz = fz[fz.regime == REGIME].copy()

def _save(fig, name):
    p = f"{OUT}/{name}"; fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); print("saved", p)
    if FIG_DEST and os.path.isdir(FIG_DEST):
        shutil.copy(p, os.path.join(FIG_DEST, name)); print("  -> copied to", FIG_DEST)

def _sep(ax):  # dashed divider before the reactive arm
    ax.axvline(2.5, color=GREY, ls="--", lw=1.0, zorder=1)

# ============================================================ M1: position-at-default (freeze-trap)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
data = [cd.loc[cd.margin == a, "abspos"].to_numpy() for a in ARMS]
bp = ax[0].boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="white", lw=1.5), whis=(5, 95))
for patch, c in zip(bp["boxes"], BARCOL):
    patch.set_facecolor(c); patch.set_alpha(0.85)
for a, x in zip(ARMS, range(4)):
    n = (cd.margin == a).sum() / N_SEEDS
    ax[0].text(x, ax[0].get_ylim()[1]*0.96, f"n={n:.1f}", ha="center", va="top", fontsize=8.5, color="#333")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([LABEL[a] for a in ARMS])
ax[0].set_ylabel("|position| at client default (contracts)")
ax[0].set_title("Defaulted-position distribution (box: 5–95%)"); _sep(ax[0])

p50 = [np.median(cd.loc[cd.margin == a, "abspos"]) for a in ARMS]
p99 = [np.quantile(cd.loc[cd.margin == a, "abspos"], 0.99) for a in ARMS]
x = np.arange(4)
ax[1].bar(x - 0.2, p50, 0.4, color=GREY, label="median")
ax[1].bar(x + 0.2, p99, 0.4, color=[*[FLAT]*3, REACT], label="99th percentile")
ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[a] for a in ARMS])
ax[1].set_ylabel("|position| at default (contracts)")
ax[1].set_title("Median falls, tail fattens (the freeze-trap)"); ax[1].legend(fontsize=9); _sep(ax[1])
fig.suptitle("As flat margin rises, client defaults get fewer and typically smaller — but the tail fattens",
             fontsize=12.5, fontweight="bold", y=1.02)
_save(fig, "mech_position.png")

# ===================================== M2: where the loss is absorbed (layers) + why clients freeze (channel)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
# left — deficit by waterfall layer: loss climbs L1->L2->L3; L4 (survivor cash) and L5 (CCP cash) never engage
layers = ["L1_own_df", "L2_sitg", "L3_pooled_df", "L4_survivor_cash", "L5_ccp_cash"]
lnames = ["L1 own DF", "L2 SITG", "L3 pooled DF", "L4 survivor cash", "L5 CCP cash"]
lcols  = ["#1A5276", "#7FB3D5", "#E07A3C", "#c0c5cc", "#6b7280"]
tot = {a: (wf.loc[wf.margin == a, layers].sum() / N_SEEDS / 1e9) for a in ARMS}  # $B per seed
bottom = np.zeros(4)
for lay, nm, c in zip(layers, lnames, lcols):
    vals = np.array([tot[a][lay] for a in ARMS])
    ax[0].bar(range(4), vals, bottom=bottom, color=c, label=nm, edgecolor="white", lw=0.5)
    bottom += vals
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([LABEL[a] for a in ARMS])
ax[0].set_ylabel("loss absorbed per seed (\\$B)"); ax[0].set_title("Where the loss is absorbed")
ax[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
ax[0].text(0.97, 0.97, "L4–L5 never engage", transform=ax[0].transAxes, ha="right", va="top",
           fontsize=8.3, color="#555", style="italic"); _sep(ax[0])
mut = [(wf[(wf.margin == a) & (wf.level >= 3)].shape[0]) / N_SEEDS for a in ARMS]  # kept for the diagnostic print
# right — freeze channel: member contagion vs own distress (~98% contagion across all arms)
ct = fz.groupby(["margin", "reason"]).size().unstack(fill_value=0).reindex(ARMS)
share = ct.div(ct.sum(axis=1), axis=0) * 100
cont = share.get("cm_contagion", pd.Series(0, index=ARMS)).to_numpy()
dist = share.get("own_distress", pd.Series(0, index=ARMS)).to_numpy()
ax[1].bar(range(4), cont, color="#1A5276", label="member contagion (CM withdrew clearing)", edgecolor="white")
ax[1].bar(range(4), dist, bottom=cont, color="#E07A3C", label="own distress (client hit 4% floor)", edgecolor="white")
for x, c in enumerate(cont):
    ax[1].text(x, c / 2, f"{c:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([LABEL[a] for a in ARMS])
ax[1].set_ylabel("share of client-freeze time (%)"); ax[1].set_ylim(0, 108)
ax[1].set_title("Why clients freeze"); ax[1].legend(fontsize=7.8, loc="upper center", framealpha=0.95); _sep(ax[1])
fig.suptitle("Loss climbs the first three waterfall layers; client freezes are member-contagion, not own distress",
             fontsize=12.5, fontweight="bold", y=1.02)
_save(fig, "mech_loss_freeze.png")

# ============================================================ M3: defaults by client type
fig, ax = plt.subplots(figsize=(7.6, 4.4))
KINDS = ["FundamentalTrader", "MomentumTrader", "ZeroIntelligenceTrader"]
KLAB  = ["Fundamental", "Momentum", "Noise (ZI)"]
KCOL  = ["#1A5276", "#7FB3D5", "#E07A3C"]
by = cd.groupby(["margin", "kind"]).size().unstack(fill_value=0).reindex(ARMS) / N_SEEDS
bottom = np.zeros(4)
for k, lab, c in zip(KINDS, KLAB, KCOL):
    vals = by[k].to_numpy() if k in by else np.zeros(4)
    ax.bar(range(4), vals, bottom=bottom, color=c, label=lab, edgecolor="white", lw=0.5)
    bottom += vals
ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[a] for a in ARMS])
ax.set_ylabel("client defaults per seed"); ax.axvline(2.5, color=GREY, ls="--", lw=1.0)
ax.set_title("Who bears the defaults, by client type"); ax.legend(fontsize=9)
fig.suptitle("Fundamental and momentum clients carry the defaults; noise traders least",
             fontsize=12, fontweight="bold", y=1.00)
_save(fig, "mech_defaults_by_type.png")

print("\nrelocation check (per seed): client defaults & p99 position by arm")
for a in ARMS:
    sub = cd[cd.margin == a]
    print(f"  {LABEL[a]:9s}  n={len(sub)/N_SEEDS:5.2f}  median={np.median(sub.abspos):6.0f}  "
          f"p99={np.quantile(sub.abspos,0.99):6.0f}  mut/seed={mut[ARMS.index(a)]:.2f}")
