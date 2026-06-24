#!/usr/bin/env python3
"""
B_capital_ratio.png — clearing-member capital ratio, redesigned (clean, 2-panel).

Panel A: median capital ratio kappa by member type over the stressed window (IQR band), with the
8% floor — the banks (client-clearing + house-only) ride the floor, the NBCMs sit higher.
Panel B: per-seed floor breaches and defaults by member type — banks breach but survive (deleverage),
NBCMs breach less but are the ONLY tier that defaults (they cannot deleverage). The juxtaposition is the
point: fragility is structural (no deleveraging), not a matter of leverage level.

Reads output/results/descriptive/{capital_ratio_band.csv, member_clientclearing.csv, summary_scalars.csv,
member_kappa_path.csv}. Writes output/results/B_capital_ratio.png (+ FIG_DEST copy).
Run: PYTHONPATH=. python3 scripts/fig_capital_ratio.py
"""
import os, shutil
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Thesis figures carry a LaTeX caption: suppress in-figure headers (suptitle) + boxed notes.
import matplotlib.figure as _mfig, matplotlib.axes as _maxes
_mfig.Figure.suptitle = lambda *a, **k: None
_maxes.Axes.text = (lambda _o: (lambda self, *a, **k: None if "bbox" in k else _o(self, *a, **k)))(_maxes.Axes.text)
from model.globals import LR_FLOOR_BCM

D = "output/results/descriptive"; OUT = "output/results"; os.makedirs(OUT, exist_ok=True)
FIG_DEST = os.environ.get("FIG_DEST", "")
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"figure.facecolor":"white","savefig.facecolor":"white",
                     "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"font.size":11})
TYPES = [("BCMc","BCM — client-clearing","#1A5276"),
         ("BCMo","BCM — house-only","#7FB3D5"),
         ("NBCM","NBCM","#E07A3C")]
FLOOR_C = "#9aa5b1"

band = pd.read_csv(f"{D}/capital_ratio_band.csv")
mc   = pd.read_csv(f"{D}/member_clientclearing.csv"); mcs = mc[mc.regime=="stressed"]
sm   = pd.read_csv(f"{D}/summary_scalars.csv"); sms = sm[sm.regime=="stressed"]
kp   = pd.read_csv(f"{D}/member_kappa_path.csv"); kps = kp[kp.regime=="stressed"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A — median kappa by type over the stressed window (floor zone)
bs = band[band.regime=="stressed"]
for mt, lab, col in TYPES:
    s = bs[bs.mtype==mt].sort_values("t")
    if s.empty: continue
    x = s["t"].to_numpy()/390.0
    ax[0].fill_between(x, s["p25"], s["p75"], color=col, alpha=0.18, lw=0)
    ax[0].plot(x, s["p50"], color=col, lw=2.1, label=lab)
ax[0].axhline(LR_FLOOR_BCM, color=FLOOR_C, ls="--", lw=1.5)
ax[0].text(ax[0].get_xlim()[1], LR_FLOOR_BCM, " 8% floor", color="#6b7280", fontsize=8.5, va="center")
ax[0].set_ylim(0.0, 0.28)
ax[0].set_xlabel("trading session"); ax[0].set_ylabel(r"capital ratio $\kappa=\mathrm{cash}/\mathrm{exposure}$")
ax[0].set_title("Capital ratio by member type (stressed; median, IQR)")
ax[0].legend(loc="upper right", fontsize=8.5, framealpha=0.92)

# Panel B — per-seed breaches + defaults by member type
bcmc_br = mcs[mcs.agent_id.between(30,34)].groupby("seed")["breached"].sum().mean()
bcmo_br = mcs[mcs.agent_id.between(35,39)].groupby("seed")["breached"].sum().mean()
nbcm_br = (kps[kps.mtype=="NBCM"].groupby("agent_id")["capital_ratio"].min() < LR_FLOOR_BCM).sum()  # rep seed
nbcm_def = sms.nbcm_defaults.mean(); bcm_def = sms.cm_defaults.mean()
breach = [bcmc_br, bcmo_br, float(nbcm_br)]
deflt  = [bcm_def, bcm_def, nbcm_def]
x = np.arange(3); cols = [c for _,_,c in TYPES]
ax[1].bar(x, breach, 0.62, color=cols, alpha=0.9, label="members breaching 8% / seed")
for xi, (b, d) in enumerate(zip(breach, deflt)):
    tag = f"{d:.2f} defaults/seed" if d > 0 else "0 defaults"
    ax[1].text(xi, b + 0.12, tag, ha="center", va="bottom", fontsize=8.5,
               color=("#b03a2e" if d > 0 else "#6b7280"),
               fontweight=("bold" if d > 0 else "normal"))
ax[1].set_xticks(x); ax[1].set_xticklabels(["BCM\nclient-clearing","BCM\nhouse-only","NBCM"])
ax[1].set_ylabel("members breaching the floor / seed (of 5)")
ax[1].set_ylim(0, 5.0)
ax[1].set_title("Banks breach but survive; only NBCMs default")

fig.suptitle("Members ride the capital floor under stress — but only the non-banks, which cannot deleverage, fail",
             fontweight="bold", fontsize=12.5, y=1.02)
fig.tight_layout()
p = f"{OUT}/B_capital_ratio.png"; fig.savefig(p, bbox_inches="tight"); print("saved", p)
if FIG_DEST and os.path.isdir(FIG_DEST):
    shutil.copy(p, os.path.join(FIG_DEST, "B_capital_ratio.png")); print("  -> copied to", FIG_DEST)
print(f"breach/seed: BCMc {bcmc_br:.2f}  BCMo {bcmo_br:.2f}  NBCM(rep) {nbcm_br}  | NBCM def/seed {nbcm_def:.2f}")
