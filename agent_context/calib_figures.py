"""Calibration-section figures (bare-market layer vs empirical ES). Lineage visual set
(HFABM / XGB-Chiarella / Franke-Westerhoff): fat-tail CCDF, ACF of r and |r| with bands, an
empirical-vs-simulated price overlay, and a calibrated-vs-empirical moment table. Pure numpy/pandas/mpl.
Outputs -> output/figs/ : fig_calib_tails.png, fig_calib_acf.png, fig_calib_price.png, calib_moments.csv."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model.globals import ModelParams, CALIBRATED, day_start_steps
from model.run_simulation import build_traders
from model.simulation import Simulation

OUT = "output/figs"; os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,123").split(",")]
EMP_C, SIM_C = "#34495E", "#e8734c"           # empirical / simulated colours
REG = {"calm": "calm (2019)", "stressed": "stressed (COVID 2020)"}

def _intraday_logret(V, opens):
    r = np.diff(np.log(V)); mask = np.ones(len(r), bool)
    for o in opens:
        if 0 < o <= len(r): mask[o-1] = False        # drop the overnight-gap return
    return r[mask]

def emp_returns(regime):
    df = pd.read_csv(f"data/fv_{regime}.csv")
    V = df["V_smooth"].to_numpy(float)
    if "ts" in df.columns:
        d = pd.to_datetime(df["ts"]).dt.normalize().to_numpy()
        opens = list(np.flatnonzero(np.r_[True, d[1:] != d[:-1]]))
    else:
        opens = list(range(0, len(V), 390))
    return _intraday_logret(V, opens)

def sim_returns(regime, n_days=int(os.environ.get("NDAYS", "18"))):
    df = pd.read_csv(f"data/fv_{regime}.csv"); V = df["V_smooth"].to_numpy(float)
    SIG = df["sigma_t"].to_numpy(float) if "sigma_t" in df.columns else None
    st = list(day_start_steps(regime)); n = st[min(len(st)-1, n_days)] if len(st) > n_days else len(V)
    out, mids = [], None
    for seed in SEEDS:
        p = ModelParams(n_fundamental=40, n_momentum=20, n_zi=40, n_bcm=0, n_nbcm=0,
                        n_bcm_with_clients=0, v0=float(V[0]), tick_size=0.25, dt_minutes=1.0,
                        **CALIBRATED[regime], stressed=(regime == "stressed"))
        tr = build_traders(p, seed=seed)
        sim = Simulation(p, tr, seed=seed, ccp=None); sim.v_array = V[:n]
        if SIG is not None: sim.sigma_t_array = SIG[:n]
        sim.run(n)
        m = np.array([x for x in sim.history["mid_price"]], float)
        opens = [i for i in range(len(m)) if i in sim._day_start_rows]
        out.append(_intraday_logret(m[~np.isnan(m)], opens))
        if mids is None: mids = m
    return np.concatenate(out), mids, V[:n]

def acf(x, lags):
    x = x - x.mean(); v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in lags])

def hill(x, frac=0.05):
    a = np.sort(np.abs(x))[::-1]; k = max(int(frac * len(a)), 20)
    return 1.0 / np.mean(np.log(a[:k] / a[k]))

DATA = {rg: dict(emp=emp_returns(rg)) for rg in ("calm", "stressed")}
for rg in ("calm", "stressed"):
    sr, mids, Vw = sim_returns(rg); DATA[rg].update(sim=sr, mid=mids, V=Vw)

# ---- Fig 1: fat-tail CCDF of |standardised return| (log-log), empirical vs sim ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, rg in zip(axes, ("calm", "stressed")):
    for lab, col, x in (("empirical", EMP_C, DATA[rg]["emp"]), ("simulated", SIM_C, DATA[rg]["sim"])):
        z = np.sort(np.abs(x) / x.std())[::-1]; ccdf = np.arange(1, len(z)+1) / len(z)
        ax.loglog(z, ccdf, color=col, lw=1.6, label=lab)
    ax.set_title(f"Return tail — {REG[rg]}"); ax.set_xlabel("|standardised return|")
    ax.set_ylabel("CCDF  P(|r| > x)"); ax.legend(frameon=False); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_calib_tails.png", dpi=130); plt.close(fig)

# ---- Fig 2: ACF of r and |r| with +/-1.96/sqrt(N) band, empirical vs sim (2x2) ----
lags = np.arange(1, 21)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, rg in enumerate(("calm", "stressed")):
    for j, (name, tx) in enumerate((("returns r", lambda v: v), ("abs returns |r|", np.abs))):
        ax = axes[i, j]
        for lab, col, off, x in (("empirical", EMP_C, -0.18, DATA[rg]["emp"]),
                                 ("simulated", SIM_C, 0.18, DATA[rg]["sim"])):
            ax.stem(lags + off, acf(tx(x), lags), linefmt=col, markerfmt="o", basefmt=" ")
            ax.plot([], [], color=col, label=lab)
        b = 1.96 / np.sqrt(len(DATA[rg]["emp"]))
        ax.axhspan(-b, b, color="grey", alpha=.18)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"ACF of {name} — {REG[rg]}", fontsize=10)
        if i == 1: ax.set_xlabel("lag (min)")
        if j == 0: ax.set_ylabel("ACF")
        ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_calib_acf.png", dpi=130); plt.close(fig)

# ---- Fig 3: empirical mid vs one simulated path (stressed = the informative regime) ----
fig, ax = plt.subplots(figsize=(9, 4))
rg = "stressed"; V = DATA[rg]["V"]; m = DATA[rg]["mid"]
t = np.arange(len(V)) / 390.0
ax.plot(t, V, color=EMP_C, lw=1.4, label="empirical efficient mid $V_t$")
ax.plot(t[:len(m)], m, color=SIM_C, lw=0.8, alpha=.8, label="simulated mid (seed 42)")
ax.set_xlabel("session"); ax.set_ylabel("ES index"); ax.set_title("Price path — stressed (COVID 2020)")
ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_calib_price.png", dpi=130); plt.close(fig)

# ---- Table: calibrated-sim vs empirical moments ----
def moments(x):
    return dict(std=x.std(), kurtosis=pd.Series(x).kurt() + 3, hill=hill(x),
                acf_r1=acf(x, [1])[0], acf_absr1=acf(np.abs(x), [1])[0],
                acf_absr10=acf(np.abs(x), [10])[0])
rows = []
for rg in ("calm", "stressed"):
    me, ms = moments(DATA[rg]["emp"]), moments(DATA[rg]["sim"])
    for k in me:
        rows.append(dict(regime=rg, moment=k, empirical=round(me[k], 5), simulated=round(ms[k], 5)))
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/calib_moments.csv", index=False)
print(tab.to_string(index=False))
print(f"\nsaved 3 figs + calib_moments.csv -> {OUT}/   (emp n: calm {len(DATA['calm']['emp'])}, stressed {len(DATA['stressed']['emp'])})")
