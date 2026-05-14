"""
data/v_jd.py — Merton (1976) jump-diffusion calibration and path generation
for the exogenous V_t fundamental.

Calibration: Lee & Mykland (2008) jump detection on 5-min log-returns of ES
front-month close (rolled by data/roll.py). Open-bar (cross-day) returns
dropped. Diffusion sigma estimated from non-jump-bar variance (BNS 2004);
jump mean / std from flagged jump bars. Lambda fixed at the ODD-anchored
value 3 jumps per RTH day = 3/78 per step.

Generator: forward-simulates a Merton-JD path with the calibrated (sigma, m, s)
per regime. Output schema matches data/fv_{regime}.csv (ts, V_smooth) so the
existing Simulation.load path picks it up transparently.

Usage:
    python data/v_jd.py calibrate
    python data/v_jd.py generate <regime> <seed>          # writes data/fv_{regime}.csv
    python data/v_jd.py generate <regime> <seed> <path>   # writes to a custom path
    python data/v_jd.py generate-all <seed>               # writes both fv_{calm,stressed}.csv
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR.parent / "output"

REGIMES = ("calm", "stressed")
LAMBDA_PER_STEP = 3.0 / 78.0          # ODD §Stochasticity: 3 jumps per RTH day
LEE_MYKLAND_K = 78                    # local BV rolling window (= 1 RTH day)
LEE_MYKLAND_ALPHA = 0.01              # 1% significance


# ── data loading ─────────────────────────────────────────────────────────────

def load_returns(regime: str) -> dict:
    """Returns dict with ts (DatetimeIndex), close, returns, is_open_bar (bool).
    Open-bar returns are those crossing a date boundary (overnight gap)."""
    df = pd.read_csv(PROC_DIR / f"ES_front_{regime}_5m.csv",
                     index_col=0, parse_dates=True)
    close = df["close"].to_numpy(dtype=float)
    log_p = np.log(close)
    returns = np.concatenate([[np.nan], np.diff(log_p)])
    dates = pd.DatetimeIndex(df.index).date
    is_open_bar = np.zeros(len(df), dtype=bool)
    is_open_bar[0] = True
    is_open_bar[1:] = (dates[1:] != dates[:-1])
    return {"ts": df.index, "close": close,
            "returns": returns, "is_open_bar": is_open_bar}


# ── Lee-Mykland (2008) jump detection ────────────────────────────────────────

def lee_mykland(returns: np.ndarray, valid: np.ndarray,
                K: int = LEE_MYKLAND_K, alpha: float = LEE_MYKLAND_ALPHA):
    """Vectorised Lee-Mykland statistic.

    Local volatility via bipower variation over K-1 pairs ending at bar i:
        BV_i = (pi/2) · (1/(K-1)) · Σ |r_j| · |r_{j-1}|

    Threshold scales with total valid sample size n:
        c_n = (2 log n)^(-1/2)
        C_n = (2 log n)^(1/2) − (log π + log log n) / (2 (2 log n)^(1/2))
        β*  = −log(−log(1 − α))
        reject if |L_i| > β* · c_n + C_n   ⇔   jump bar

    Returns (flags, sigma_local, threshold).
    """
    n = len(returns)
    abs_r = np.abs(returns)
    abs_r_masked = np.where(valid, abs_r, np.nan)

    # pair[j-1] = |r_{j-1}| · |r_j|  for j in 1..n-1
    pair = abs_r_masked[1:] * abs_r_masked[:-1]
    pair_series = pd.Series(pair)
    bv = pair_series.rolling(window=K - 1, min_periods=K // 2).mean() * (np.pi / 2.0)

    sigma_local = np.full(n, np.nan)
    sigma_local[1:] = np.sqrt(np.maximum(bv.to_numpy(), 1e-24))

    n_valid = max(int(valid.sum()), 2)
    log_n = np.log(n_valid)
    c_n = (2.0 * log_n) ** -0.5
    C_n = (2.0 * log_n) ** 0.5 - (np.log(np.pi) + np.log(log_n)) / (2.0 * (2.0 * log_n) ** 0.5)
    beta_star = -np.log(-np.log(1.0 - alpha))
    threshold = beta_star * c_n + C_n

    with np.errstate(invalid="ignore", divide="ignore"):
        L = returns / sigma_local
    flags = valid & np.isfinite(L) & (np.abs(L) > threshold)
    return flags, sigma_local, threshold


# ── JD parameter estimation ──────────────────────────────────────────────────

def estimate_jd_params(returns: np.ndarray, valid: np.ndarray,
                       flags: np.ndarray) -> dict:
    jump_r = returns[valid & flags]
    nonjump_r = returns[valid & ~flags]
    sigma = float(np.std(nonjump_r, ddof=1)) if len(nonjump_r) > 1 else 1e-6
    m = float(np.mean(jump_r)) if len(jump_r) > 0 else 0.0
    s = float(np.std(jump_r, ddof=1)) if len(jump_r) > 1 else max(sigma * 5.0, 1e-3)
    n_valid = int(valid.sum())
    n_jumps = int(flags.sum())
    lambda_emp = n_jumps / n_valid if n_valid > 0 else 0.0
    return {
        "sigma": sigma,
        "jump_mean": m,
        "jump_std": s,
        "n_valid": n_valid,
        "n_jumps": n_jumps,
        "lambda_emp_per_step": lambda_emp,
    }


# ── calibration entry point ──────────────────────────────────────────────────

def calibrate(verbose: bool = True) -> dict:
    results = {"lambda_per_step": LAMBDA_PER_STEP,
               "lee_mykland_K": LEE_MYKLAND_K,
               "lee_mykland_alpha": LEE_MYKLAND_ALPHA}
    for regime in REGIMES:
        data = load_returns(regime)
        valid = (~data["is_open_bar"]) & np.isfinite(data["returns"])
        flags, _sigma_local, thresh = lee_mykland(
            data["returns"], valid, K=LEE_MYKLAND_K, alpha=LEE_MYKLAND_ALPHA)
        p = estimate_jd_params(data["returns"], valid, flags)
        p["v0"] = float(data["close"][0])
        p["threshold"] = float(thresh)
        results[regime] = p
        if verbose:
            print(f"[{regime}] n_valid={p['n_valid']}, n_jumps={p['n_jumps']}, "
                  f"lambda_emp={p['lambda_emp_per_step']:.5f}/step "
                  f"({p['lambda_emp_per_step']*78:.2f}/day), "
                  f"sigma={p['sigma']:.6f}, "
                  f"jump_mean={p['jump_mean']:.5f}, jump_std={p['jump_std']:.5f}, "
                  f"v0={p['v0']:.2f}, thresh={p['threshold']:.3f}")
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "v_jd_params.json"
    out_path.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"Saved {out_path}")
    return results


# ── generator: Merton JD path ────────────────────────────────────────────────

def generate(regime: str, seed: int, out_path: Path | None = None) -> np.ndarray:
    params_path = OUT_DIR / "v_jd_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Run calibration first: {params_path}")
    all_params = json.loads(params_path.read_text())
    p = all_params[regime]
    sigma = p["sigma"]
    m = p["jump_mean"]
    s = p["jump_std"]
    v0 = p["v0"]
    lam = all_params["lambda_per_step"]

    df = pd.read_csv(PROC_DIR / f"ES_front_{regime}_5m.csv",
                     index_col=0, parse_dates=True)
    n_steps = len(df)
    timestamps = df.index

    rng = np.random.default_rng(seed)
    kappa = np.exp(m + 0.5 * s * s) - 1.0
    drift = -0.5 * sigma * sigma - lam * kappa

    Z = rng.standard_normal(n_steps)
    n_jumps = rng.poisson(lam, size=n_steps)
    log_increments = drift + sigma * Z
    for t in range(n_steps):
        nj = int(n_jumps[t])
        if nj > 0:
            log_increments[t] += rng.normal(m, s, size=nj).sum()
    log_increments[0] = 0.0
    log_v = np.log(v0) + np.cumsum(log_increments)
    V = np.exp(log_v)

    if out_path is None:
        out_path = DATA_DIR / f"fv_{regime}.csv"
    out_path = Path(out_path)
    out_df = pd.DataFrame({"ts": timestamps, "V_smooth": V})
    out_df.to_csv(out_path, index=False)
    print(f"[{regime}] seed={seed} n={n_steps}, V0={V[0]:.2f}, "
          f"V_end={V[-1]:.2f}, range=[{V.min():.2f}, {V.max():.2f}], "
          f"empirical_jumps={int(n_jumps.sum())} → {out_path}")
    return V


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calibrate()
    elif cmd == "generate":
        if len(sys.argv) < 4:
            print("Usage: generate <regime> <seed> [out_path]"); sys.exit(1)
        regime = sys.argv[2]
        seed = int(sys.argv[3])
        out = Path(sys.argv[4]) if len(sys.argv) >= 5 else None
        generate(regime, seed, out)
    elif cmd == "generate-all":
        if len(sys.argv) < 3:
            print("Usage: generate-all <seed>"); sys.exit(1)
        seed = int(sys.argv[2])
        for r in REGIMES:
            generate(r, seed)
    else:
        print(f"Unknown command: {cmd}\n"); print(__doc__); sys.exit(1)


if __name__ == "__main__":
    _main()
