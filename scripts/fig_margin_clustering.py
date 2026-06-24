#!/usr/bin/env python3
"""
F_margin_call_clustering.png — margin calls per hour over the window, with BURSTS shaded.

Simple and direct: the per-hour margin-call count (calm top, stressed bottom); wherever it exceeds
3 calls/hour the background is shaded. Clustering shows up as the WIDTH of the shaded bands — bursts
form wide contiguous shading, a memoryless process would give only thin scattered marks. Stress is
shaded almost throughout (sustained bursts); calm only spikes occasionally.

Reads output/results/descriptive/margin_calls.csv (regime, hour, p50, p25, p75).
Run: PYTHONPATH=. python3 scripts/fig_margin_clustering.py
"""
import os, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mfig, matplotlib.axes as _maxes
_mfig.Figure.suptitle = lambda *a, **k: None
_maxes.Axes.text = (lambda _o: (lambda self, *a, **k: None if "bbox" in k else _o(self, *a, **k)))(_maxes.Axes.text)

D = "output/results/descriptive"; OUT = "output/results"; os.makedirs(OUT, exist_ok=True)
FIG_DEST = os.environ.get("FIG_DEST", "")
DARK, ORANGE, GREY = "#1A5276", "#E07A3C", "#9aa5b1"
THR = 3
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"figure.facecolor":"white","savefig.facecolor":"white",
                     "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"font.size":11})

mh = pd.read_csv(f"{D}/margin_calls.csv")
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for ax, (reg, lab, col) in zip(axes, [("calm", "Calm", DARK), ("stressed", "Stressed", ORANGE)]):
    g = mh[mh.regime == reg].sort_values("hour")
    x, y = g["hour"].to_numpy(float), g["p50"].to_numpy(float)
    ymax = max(y.max(), THR) + 1
    ax.fill_between(x, 0, ymax, where=(y > THR), color=col, alpha=0.22, step="mid", lw=0)  # shaded bursts
    ax.plot(x, y, color=col, lw=1.2)
    ax.axhline(THR, color=GREY, ls="--", lw=1.1)
    ax.set_ylim(0, ymax); ax.set_xlim(x.min(), x.max())
    ax.set_ylabel(f"{lab}\nmargin calls / hour")
    frac = float((y > THR).mean())
    ax.text(0.995, 0.90, f"{100*frac:.0f}% of hours in a burst ($>{THR}$/hr)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=col)
axes[1].set_xlabel("trading hour over the window")
fig.tight_layout()
p = f"{OUT}/F_margin_call_clustering.png"; fig.savefig(p, bbox_inches="tight"); print("saved", p)
if FIG_DEST and os.path.isdir(FIG_DEST):
    shutil.copy(p, os.path.join(FIG_DEST, "F_margin_call_clustering.png")); print("  -> copied to", FIG_DEST)
for reg, lab, _ in [("calm","Calm",0),("stressed","Stressed",0)]:
    y = mh[mh.regime == reg]["p50"].to_numpy(float)
    print(f"  {lab}: {100*(y>THR).mean():.0f}% of hours > {THR} calls/hr (mean {y.mean():.1f}/hr)")
