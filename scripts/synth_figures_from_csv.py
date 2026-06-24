#!/usr/bin/env python3
"""Synthetic-ensemble figures, CSV-driven (fast — no model re-run).

Reads the saved synthetic-ensemble results at the locked joint optimum:
    output/synth_results/mids/ret_s*.csv     per-path intraday log-returns   -> stylised facts
    output/synth_results/summary.csv          per-path clearing scalars
    output/synth_results/im_path.csv          total IM per (path, t)
    output/overnight_joint/paths/fv_joint_s*.csv   the FV price paths (V_t)
and the empirical ES returns (scripts.calibrate), and writes three figures
(also copied to the thesis Figures/ dir if FIG_DEST is set):
    synth_paths.png      the synthetic stress-path ensemble (V_t)
    synth_facts.png      stylised facts: standardised-return CCDF (tails) + |r|-ACF (clustering)
    synth_clearing.png   clearing outcomes: client defaults vs path vol + posted-IM band

Run:  PYTHONPATH=. python3 scripts/synth_figures_from_csv.py
      FIG_DEST=/path/to/MSc_Thesis/Figures PYTHONPATH=. python3 scripts/synth_figures_from_csv.py
"""
import os, glob, re, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Thesis figures carry a LaTeX caption: suppress in-figure headers (suptitle) + boxed notes.
import matplotlib.figure as _mfig, matplotlib.axes as _maxes
_mfig.Figure.suptitle = lambda *a, **k: None
_maxes.Axes.text = (lambda _o: (lambda self, *a, **k: None if "bbox" in k else _o(self, *a, **k)))(_maxes.Axes.text)
from scripts.calibrate import _empirical_returns, _acf

REGIME = "stressed"
SR   = "output/synth_results"
PATHS = "output/overnight_joint/paths"
OUT  = "output/results"; os.makedirs(OUT, exist_ok=True)
FIG_DEST = os.environ.get("FIG_DEST", "")           # optional: copy PNGs into the thesis Figures/ dir
MAXLAG = 30
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.size": 11})
STEEL, NAVY, BLACK, GREY = "#7FB3D5", "#1A5276", "#1A5276", "#9aa5b1"

def _num(f):  # numeric sort key for ..._s12.csv
    m = re.search(r"_s(\d+)\.csv$", f); return int(m.group(1)) if m else 0

ret_files  = sorted(glob.glob(f"{SR}/mids_noclr/ret_s*.csv"), key=_num)   # NO-clearing returns -> stylised facts (bare market = the calibration target; clearing damps clustering)
clr_files  = sorted(glob.glob(f"{SR}/mids/ret_s*.csv"), key=_num)         # with-clearing returns -> path realised vol for the clearing scatter
path_files = sorted(glob.glob(f"{PATHS}/fv_joint_s*.csv"), key=_num)
rets = [pd.read_csv(f)["ret"].to_numpy() for f in ret_files]
rets = [r[np.isfinite(r)] for r in rets if len(r) > 100]
emp  = _empirical_returns(REGIME); emp = emp[np.isfinite(emp)]
N = len(rets)
print(f"loaded {N} return paths, {len(path_files)} price paths")

def _ccdf(x):
    z = np.sort(np.abs(np.asarray(x)) / np.std(x))[::-1]
    return z, np.arange(1, len(z) + 1) / len(z)

def _save(fig, name):
    p = f"{OUT}/{name}"; fig.savefig(p, bbox_inches="tight"); print("saved", p)
    if FIG_DEST and os.path.isdir(FIG_DEST):
        shutil.copy(p, os.path.join(FIG_DEST, name)); print("  -> copied to", FIG_DEST)

# ============================================================ FIG 1: price-path ensemble
Vs = [pd.read_csv(f)["V_smooth"].to_numpy() for f in path_files]
L = min(len(v) for v in Vs); Vs = np.array([v[:L] for v in Vs])
idx = np.arange(L) / 390.0                                    # trading days
norm = 100.0 * Vs / Vs[:, [0]]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for row in norm:
    ax.plot(idx, row, color=STEEL, alpha=0.12, lw=0.7)
ax.plot(idx, norm.mean(0), color=NAVY, lw=2.6, label="ensemble mean")
ax.axhline(100, color=GREY, lw=0.8, ls=":")
ax.set_xlabel("trading day"); ax.set_ylabel("fundamental level (start $=100$)")
ax.set_title(f"Synthetic ES stress-path ensemble ($N={len(Vs)}$)")
ax.legend(frameon=False, loc="lower left")
ax.text(0.97, 0.95, f"mean end {norm.mean(0)[-1]:.0f}\n(${{\\sim}}{100-norm.mean(0)[-1]:.0f}\\%$ decline)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
fig.tight_layout(); _save(fig, "synth_paths.png")

# ============================================================ FIG 2: stylised facts (tails + clustering)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
# (a) CCDF of standardised |returns| (log-log): heavy tail vs Gaussian
for r in rets:
    zx, zy = _ccdf(r); ax[0].loglog(zx, zy, color=STEEL, alpha=0.10, lw=0.7)
zx, zy = _ccdf(np.concatenate(rets)); ax[0].loglog(zx, zy, color=STEEL, lw=2.4, label="synthetic (pooled)")
ex, ey = _ccdf(emp); ax[0].loglog(ex, ey, color=BLACK, lw=2.4, label="empirical ES")
gx, gy = _ccdf(np.random.default_rng(0).normal(size=300000)); ax[0].loglog(gx, gy, color=GREY, ls=":", lw=1.6, label="Gaussian")
ax[0].set_xlim(0.3, 30); ax[0].set_ylim(1e-5, 1.2)
ax[0].set_xlabel("standardised $|r|$"); ax[0].set_ylabel(r"$P(|r| > x)$")
ax[0].set_title("Heavy tails"); ax[0].legend(frameon=False, fontsize=9, loc="lower left")
# (b) |r|-ACF (clustering): synthetic mean (+ spread) vs empirical
lags = np.arange(1, MAXLAG + 1)
acfs = np.array([[_acf(np.abs(r), k) for k in lags] for r in rets])
for row in acfs:
    ax[1].plot(lags, row, color=STEEL, alpha=0.12, lw=0.7)
ax[1].plot(lags, acfs.mean(0), color=STEEL, lw=2.6, label=f"synthetic (mean, $N={N}$)")
ax[1].plot(lags, [_acf(np.abs(emp), k) for k in lags], color=BLACK, lw=2.6, label="empirical ES")
ax[1].axhline(0, color=GREY, lw=0.6); ax[1].set_xlim(1, MAXLAG); ax[1].set_ylim(bottom=0)
ax[1].set_xlabel("lag (minutes)"); ax[1].set_ylabel(r"ACF of $|r|$")
ax[1].set_title("Volatility clustering"); ax[1].legend(frameon=False, fontsize=9)
fig.suptitle("Synthetic two-factor log-volatility ensemble reproduces the ES stylised facts",
             fontsize=12.5, fontweight="bold", y=1.01)
fig.tight_layout(); _save(fig, "synth_facts.png")

# ============================================================ FIG 3: clearing robustness + tail
summ = pd.read_csv(f"{SR}/summary.csv")
vol = {_num(f): np.std(pd.read_csv(f)["ret"]) * 1e4 for f in clr_files}   # path realised vol (cleared market), bps/min
summ["vol_bps"] = summ["path"].map(vol)
nmut = int((summ["deepest_wf"] >= 3).sum())   # mutualised default fund drawn (L3+)
n4   = int((summ["deepest_wf"] >= 4).sum())   # surviving-member pro-rata loss-sharing (L4)
n5   = int((summ["deepest_wf"] >= 5).sum())   # CCP cash (L5)
RED = "#c0392b"

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) clearing severity vs path volatility, coloured by deepest waterfall level (benign drawn first, tail on top)
order = summ.sort_values("deepest_wf").index
sc = ax[0].scatter(summ.loc[order, "vol_bps"], summ.loc[order, "client_def"],
                   c=summ.loc[order, "deepest_wf"], cmap="YlOrRd", vmin=0, vmax=5,
                   s=44 + 11 * summ.loc[order, "cm_def"].clip(0, 8), edgecolor="k", linewidth=0.4, zorder=3)
cb = fig.colorbar(sc, ax=ax[0], ticks=range(6)); cb.set_label("deepest waterfall level")
ax[0].set_xlabel("path realised volatility (bps/min)")
ax[0].set_ylabel("client defaults per path")
ax[0].set_title("Clearing stress scales with volatility")
ax[0].text(0.04, 0.96,
           f"{nmut}/{len(summ)} paths reach the mutualised\n"
           f"default fund (L$\\geq$3); {n4} reach member\n"
           f"loss-sharing (L4), {n5} reach CCP cash (L5).\n"
           f"{int(summ['cm_def'].sum())} member defaults in total\n"
           f"(marker size $\\propto$ member defaults).",
           transform=ax[0].transAxes, ha="left", va="top", fontsize=8.2,
           bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.92))

# (b) procyclical margin demand across the ensemble: posted IM level, median + 10-90% band.
#     Margin-CALL clustering is a WITHIN-path phenomenon (each path's vol clusters at different
#     times, so a cross-path average washes it out); it is reported as a within-path statistic.
im = pd.read_csv(f"{SR}/im_path.csv")
g = im.groupby("t")["im_total"]
med, lo, hi = g.median() / 1e9, g.quantile(0.10) / 1e9, g.quantile(0.90) / 1e9
t_days = med.index.to_numpy() / 390.0
shares = []                                       # share of a path's IM-calls in its own busiest 10% of hours
for p, grp in im.groupby("path"):
    s = grp.sort_values("t"); d = np.clip(np.diff(s["im_total"].to_numpy()), 0, None)
    d = d[s["t"].to_numpy()[1:] >= 390]           # drop the day-0 portfolio build-up (not a stress call)
    if d.sum() > 0:
        ds = np.sort(d)[::-1]; shares.append(ds[:max(1, len(ds)//10)].sum() / d.sum())
clust = 100 * float(np.mean(shares))

ax[1].fill_between(t_days, lo, hi, color=STEEL, alpha=0.20, label="10--90\\% of paths")
ax[1].plot(t_days, med, color=NAVY, lw=2.2, label="ensemble median")
ax[1].set_xlabel("trading day"); ax[1].set_ylabel("posted initial margin (\\$B)")
ax[1].set_title("Procyclical margin demand, robust across paths")
ax[1].set_ylim(bottom=0)
ax[1].text(0.5, 0.05, f"margin calls cluster in time: within a path the\n"
           f"busiest $10\\%$ of hours carry ${clust:.0f}\\%$ of all margin called",
           transform=ax[1].transAxes, ha="center", va="bottom", fontsize=8.2,
           bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.92))
ax[1].legend(frameon=False, fontsize=8.6, loc="upper left")

fig.suptitle("Clearing on the synthetic stressed ensemble: robust on the median path, with a volatility-driven tail",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout(); _save(fig, "synth_clearing.png")

print(f"\nensemble: {N} paths | client_def median {summ['client_def'].median():.0f} "
      f"mean {summ['client_def'].mean():.1f} max {summ['client_def'].max():.0f} | "
      f"member defaults {int(summ['cm_def'].sum())} on {int((summ['cm_def']>0).sum())} paths | "
      f"reach L>=3: {nmut}/{len(summ)} (L4: {n4}, L5: {n5}) | within-path busiest-10%-of-hours carry {clust:.0f}% of calls")
