#!/usr/bin/env python3
"""Ch.4 validation figure: |r|-ACF (volatility clustering), synthetic vs empirical ES.

The overnight run did NOT cache the per-path series (output/.../paths/ is empty), so
this re-draws a representative ensemble at the LOCKED joint optimum
(output/overnight_joint_logvol/calibration.json) and plots the absolute-return
autocorrelation of the simulated mid against empirical ES. Reproducible: the optimum
(sv/jump/mu overrides + agent overrides) is fixed; only the path RNG seeds vary.

Outputs:
  output/overnight_joint_logvol/synth_acf.png   (the figure)
  output/overnight_joint_logvol/synth_acf.csv   (lag, empirical, synthetic_mean)

Env: ACF_PATHS (default 16), ACF_DAYS (default 30), ACF_MAXLAG (default 30).
"""
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import v_gbm                                   # noqa: E402
import scripts.overnight_joint as oj                     # noqa: E402
from scripts.calibrate import _empirical_returns, _acf   # noqa: E402

REGIME  = "stressed"
N_PATHS = int(os.environ.get("ACF_PATHS", 16))
N_DAYS  = int(os.environ.get("ACF_DAYS", 30))
MAXLAG  = int(os.environ.get("ACF_MAXLAG", 30))
OUT     = oj.REPO / "output" / "overnight_joint_logvol"
FVJ     = "data/fv_gbm_joint_logvol_acf.csv"             # own scratch path, no collision

def absacf(r, lags):
    a = np.abs(np.asarray(r, float)); a = a[np.isfinite(a)]
    return np.array([_acf(a, k) for k in lags])

CACHE = OUT / "synth_acf_paths.csv"     # one row per path (cols lag1..lagMAXLAG); RESUMABLE across calls

def main():
    t0 = time.time()
    base = json.loads((oj.REPO / "output" / "v_gbm_params.json").read_text())
    ro = json.load(open(OUT / "calibration.json"))["reverse_optimum"]
    sv, jmp, agent, mu = ro["sv_override"], ro["jump_override"], ro["agent_overrides"], ro["mu_override"]
    lags = np.arange(1, MAXLAG + 1)
    cols = [f"lag{k}" for k in lags]

    emp = absacf(_empirical_returns(REGIME), lags)
    print(f"empirical |r|-ACF: lag1={emp[0]:.3f}  lag20={emp[19]:.3f}", flush=True)

    # resume from cache: add ACF_PATHS MORE paths each call (seeds offset so no duplicates)
    syn = pd.read_csv(CACHE)[cols].values.tolist() if CACHE.exists() else []
    n0 = len(syn)
    for i in range(1, N_PATHS + 1):
        seed = 100 + n0 + i
        v_gbm.generate(REGIME, seed=seed, n_days=N_DAYS, out_path=oj.REPO / FVJ,
                       sv_override=sv, jump_override=jmp, mu_override=mu)
        r, _, _ = oj.run_market(REGIME, base, FVJ, N_DAYS * oj.BARS, 1, 0, agent)
        if len(r):
            syn.append(list(absacf(r, lags)))
            pd.DataFrame(syn, columns=cols).to_csv(CACHE, index=False)   # save after EVERY path
        print(f"  path {n0+i} (total {len(syn)})  ({time.time()-t0:.0f}s)", flush=True)
    syn = np.array(syn); m = syn.mean(0)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for row in syn:
        ax.plot(lags, row, color="#1f77b4", alpha=0.14, lw=0.8)
    ax.plot(lags, m,   color="#1f77b4", lw=2.4, label=f"synthetic mid (ensemble mean, $N={len(syn)}$)")
    ax.plot(lags, emp, color="black",   lw=2.4, label="empirical ES")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("lag (minutes)"); ax.set_ylabel(r"ACF of $|r|$")
    ax.set_xlim(1, MAXLAG); ax.set_ylim(bottom=min(0.0, float(np.nanmin([m.min(), emp.min()])) - 0.02))
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "synth_acf.png", dpi=160)
    pd.DataFrame({"lag": lags, "empirical": emp, "synthetic_mean": m}).to_csv(OUT / "synth_acf.csv", index=False)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT/'synth_acf.png'}", flush=True)

if __name__ == "__main__":
    main()
