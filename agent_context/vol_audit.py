"""Audit the IM/DF volatility estimator vs real CCP practice, on the REAL ES path.
Checks: (1) EWMA half-life (model 2d vs RiskMetrics ~11d / lambda 0.94); (2) Gaussian z=2.326 vs the
EMPIRICAL 99% quantile of EWMA-standardized returns (the FHS quantile real CCPs use); (3) overnight
gaps excluded vs close-to-close. Pure pandas/numpy."""
import numpy as np, pandas as pd
from model import globals as G

Z, MPOR, FLOOR, CAP = G.IM_CONF_Z, G.IM_MPOR_DAYS, G.IM_FLOOR, G.IM_CAP
BARS = G.TRADING_MINUTES_PER_DAY  # 390

def im_frac(sig_min):  # model formula: clip(z * sig_day * sqrt(MPOR), floor, cap)
    sd = sig_min * np.sqrt(BARS)
    return np.clip(Z * sd * np.sqrt(MPOR), FLOOR, CAP)

def ewma_sigma(r, halflife_bars, seed):
    lam = 1.0 - 0.5 ** (1.0 / halflife_bars)
    var = seed ** 2; out = np.empty(len(r))
    for i, x in enumerate(r):
        var = (1 - lam) * var + lam * x * x
        out[i] = np.sqrt(var)
    return out

for rg in ("calm", "stressed"):
    df = pd.read_csv(f"data/fv_{rg}.csv", parse_dates=["ts"])
    V = df["V_smooth"].to_numpy(float)
    day = df["ts"].dt.normalize().to_numpy()
    is_open = np.r_[True, day[1:] != day[:-1]]        # first bar of each session
    logr = np.diff(np.log(V))
    open_bar = is_open[1:]                              # aligns to logr (return INTO this bar)
    r_intraday = logr.copy(); r_intraday[open_bar] = np.nan   # drop overnight-gap returns
    ri = r_intraday[~np.isnan(r_intraday)]
    seed = G.SIGMA_V[rg]
    # EWMA at the model's 2-day half-life vs RiskMetrics ~11-day (lambda_daily 0.94)
    s2 = ewma_sigma(ri, 2 * BARS, seed)
    s11 = ewma_sigma(ri, 11 * BARS, seed)
    im2, im11 = im_frac(s2), im_frac(s11)
    # fat tails: empirical 99% one-tailed quantile of EWMA-standardized returns vs Gaussian 2.326
    z_std = ri[1:] / s2[:-1]                            # standardize by prior-bar sigma (FHS-style)
    q99 = np.nanpercentile(np.abs(z_std), 99)
    # overnight gap magnitude (close-to-close vs intraday)
    gap = logr[open_bar]
    print(f"\n=== {rg} ===")
    print(f"  IM half-life 2d : mean {100*im2.mean():.2f}%  peak {100*im2.max():.2f}%")
    print(f"  IM half-life 11d: mean {100*im11.mean():.2f}%  peak {100*im11.max():.2f}%  (RiskMetrics lambda~0.94)")
    print(f"  reactivity (peak/mean): 2d = {im2.max()/im2.mean():.2f}x   11d = {im11.max()/im11.mean():.2f}x")
    print(f"  empirical 99% |std-return| = {q99:.2f}  vs Gaussian z = {Z:.3f}  -> understatement {100*(q99/Z-1):.0f}%")
    print(f"  overnight gaps: n={len(gap)} std={100*np.nanstd(gap):.2f}%  worst={100*np.nanmin(gap):.2f}%  "
          f"vs intraday 1-min std {100*np.nanstd(ri):.3f}%")
    # share of close-to-close daily variance that is overnight
    dvar_intra = np.nanvar(ri) * BARS
    dvar_gap = np.nanvar(gap)
    print(f"  overnight share of daily variance ~ {100*dvar_gap/(dvar_gap+dvar_intra):.0f}%  "
          f"(excluded from the IM estimator)")
