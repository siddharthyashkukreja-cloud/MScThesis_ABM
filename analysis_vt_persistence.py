"""
Sweep the Vasicek-OU mean-reversion `α` (V_t stochastic-vol persistence)
and measure how it propagates to mid-level clustering. Tests the
hypothesis: "the long-lag |r| ACF gap is upstream — fix V_t persistence,
the mid follows."

For each α-multiplier in the grid, the script:
  1. Reads the data-calibrated Merton + Vasicek-OU params from
     `output/v_gbm_params.json`.
  2. Generates a V_t path with α scaled, σ_vol rescaled so the stationary
     variance `σ_vol²/(2α)` stays constant (i.e., only the persistence
     changes, not the vol level).
  3. Writes a TEMP fv csv (not data/fv_*.csv — does not interfere with
     calibration).
  4. Runs the D44-simplified ABM for `n_days` at the current θ.
  5. Computes 1-min and daily |r| ACFs + Hill + kurt.
  6. Prints a side-by-side table.

CLI:
    python3 analysis_vt_persistence.py [regime] [n_days]
        regime  ∈ {calm, stressed}    default: calm
        n_days                          default: 50

Recommended workflow:
    # after overnight calibration completes:
    python3 analysis_vt_persistence.py calm 50
    # → identifies the α-multiplier that best lifts daily/long-lag ACF.
    # If a clear winner emerges, edit data/v_gbm.py to lock in the new α,
    # regenerate fv_*.csv, re-run calibration.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.globals import ModelParams, V0, CALIBRATED
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier, BARS_PER_DAY

ROOT = Path(__file__).parent
BARS = BARS_PER_DAY


def _acf(x, k):
    if len(x) <= k: return float('nan')
    x = x - x.mean(); v = float((x*x).sum())
    return 0.0 if v == 0.0 else float((x[:-k]*x[k:]).sum() / v)


def _acf_smooth(x, c):
    lags = (1, 2) if c <= 1 else (c-1, c, c+1)
    vals = [v for v in (_acf(x, l) for l in lags) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _hill(r, frac=0.05):
    a = np.abs(np.asarray(r, float)); a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 100: return float('nan')
    k = max(int(frac*n), 20)
    if k >= n: return float('nan')
    s = np.sort(a)[::-1]; thr = s[k]
    if thr <= 0: return float('nan')
    xi = float(np.mean(np.log(s[:k]) - np.log(thr)))
    return 1.0/xi if xi > 0 else float('nan')


def generate_vt(regime: str, alpha_mult: float, n_days: int,
                seed: int, out_path: Path) -> None:
    """Regenerate V_t with α rescaled. Keeps `E[σ_t²]` constant by setting
    `σ_vol' = σ_vol · √alpha_mult` (stationary variance σ_vol²/(2α) stays
    constant). Same Merton-jump diffusion, same drift, same v0."""
    p = json.loads((ROOT/"output"/"v_gbm_params.json").read_text())[regime]
    sigma_d = p["sigma_d"]
    v0 = p["v0"]
    mu = float(p.get("mean_return", 0.0))
    delta = p["jump_delta"]
    lam_step = p["jump_lambda_day"] / BARS
    sv = p["sv"]
    alpha = sv["alpha"] * alpha_mult
    sigma_vol = sv["sigma_vol"] * (alpha_mult ** 0.5)   # keep stationary var const
    sv_var = sigma_vol**2 / (2.0 * alpha)
    theta_eff = (sigma_d**2 - sv_var) ** 0.5 if sigma_d**2 > sv_var else sv["theta"]
    n_steps = n_days * BARS
    rng = np.random.default_rng(seed)
    e_a = float(np.exp(-alpha))
    ou_sd = float(sigma_vol * np.sqrt((1.0 - np.exp(-2.0*alpha))/(2.0*alpha)))
    eps_s = rng.standard_normal(n_steps)
    sigma_path = np.empty(n_steps)
    sigma_path[0] = theta_eff
    for t in range(n_steps - 1):
        sigma_path[t+1] = theta_eff + e_a*(sigma_path[t]-theta_eff) + ou_sd*eps_s[t]
    sigma_path = np.maximum(sigma_path, 1e-10)
    eps_r = rng.standard_normal(n_steps)
    jump_mask = rng.random(n_steps) < lam_step
    jumps = jump_mask * rng.normal(0.0, delta, n_steps)
    drift = mu - 0.5 * sigma_path**2
    log_inc = drift + sigma_path*eps_r + jumps
    log_inc[0] = 0.0
    V = np.exp(np.log(v0) + np.cumsum(log_inc))
    ts = pd.date_range(start="2019-01-02", periods=n_steps, freq="min")
    pd.DataFrame({"ts": ts, "V_smooth": V, "sigma_t": sigma_path}).to_csv(
        out_path, index=False)


def run_one(regime: str, alpha_mult: float, n_days: int, seed: int = 42):
    tmp = ROOT/"output"/f"_tmp_fv_{regime}_x{alpha_mult:.3f}.csv"
    generate_vt(regime, alpha_mult, n_days, seed, tmp)
    theta = dict(CALIBRATED[regime])
    params = ModelParams(
        n_fundamental=10, n_momentum=10, n_momentum_long=0,
        n_mm=0, n_zi=20, n_vt=0, n_ct=0,
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        stressed=(regime == "stressed"),
        fv_csv=str(tmp), **theta,
    )
    traders = build_traders(params, seed=seed)
    ccp = build_clearing_tier(traders, params, seed=seed)
    hist = Simulation(params, traders, seed=seed, ccp=ccp).run(n_days * BARS)
    mid = pd.Series(hist["mid_price"]).ffill().bfill().to_numpy()
    mid = mid[mid > 0]
    r_1m = np.diff(np.log(mid))
    daily_mid = mid[BARS-1::BARS]
    r_d = np.diff(np.log(daily_mid))
    a_1m = np.abs(r_1m); a_d = np.abs(r_d)
    tmp.unlink(missing_ok=True)
    return {
        "alpha_mult": alpha_mult,
        "1m_ret_std": r_1m.std(),
        "1m_kurt": (((r_1m-r_1m.mean())/r_1m.std())**4).mean()-3,
        "1m_hill": _hill(r_1m),
        "1m_acf_a_1":  _acf_smooth(a_1m, 1),
        "1m_acf_a_10": _acf_smooth(a_1m, 10),
        "1m_acf_a_30": _acf_smooth(a_1m, 30),
        "1m_acf_a_60": _acf_smooth(a_1m, 60),
        "1m_acf_a_90": _acf_smooth(a_1m, 90),
        "d_n":     len(r_d),
        "d_acf_a_1": _acf_smooth(a_d, 1),
        "d_acf_a_5": _acf_smooth(a_d, 5),
    }


def empirical(regime: str):
    df = pd.read_csv(ROOT/"data"/"processed"/f"ES_front_{regime}_1m.csv",
                     index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    r_1m = np.diff(np.log(mid))
    dates = pd.DatetimeIndex(df.index).date
    r_1m = r_1m[dates[1:] == dates[:-1]]
    df_idx = pd.DatetimeIndex(df.index)
    last_per_day = df["mid"].groupby(df_idx.date).last().to_numpy()
    r_d = np.diff(np.log(last_per_day))
    a_1m, a_d = np.abs(r_1m), np.abs(r_d)
    return {
        "alpha_mult": "EMP",
        "1m_ret_std": r_1m.std(),
        "1m_kurt": (((r_1m-r_1m.mean())/r_1m.std())**4).mean()-3,
        "1m_hill": _hill(r_1m),
        "1m_acf_a_1":  _acf_smooth(a_1m, 1),
        "1m_acf_a_10": _acf_smooth(a_1m, 10),
        "1m_acf_a_30": _acf_smooth(a_1m, 30),
        "1m_acf_a_60": _acf_smooth(a_1m, 60),
        "1m_acf_a_90": _acf_smooth(a_1m, 90),
        "d_n":     len(r_d),
        "d_acf_a_1": _acf_smooth(a_d, 1),
        "d_acf_a_5": _acf_smooth(a_d, 5),
    }


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "calm"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    grid = [1.0, 0.5, 0.25, 0.1, 0.05]
    print(f"α-persistence sweep: regime={regime}, n_days={n_days}")
    print(f"(α_mult = 1.0 reproduces current calibration; smaller = slower")
    print(f" mean-reversion = more long-memory σ_t)\n")
    rows = [run_one(regime, m, n_days) for m in grid]
    rows.append(empirical(regime))
    df = pd.DataFrame(rows)
    cols = ["alpha_mult", "1m_ret_std", "1m_kurt", "1m_hill",
            "1m_acf_a_1", "1m_acf_a_10", "1m_acf_a_30",
            "1m_acf_a_60", "1m_acf_a_90",
            "d_n", "d_acf_a_1", "d_acf_a_5"]
    print(df[cols].to_string(index=False,
        formatters={c: (lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v))
                    for c in cols}))


if __name__ == "__main__":
    main()
