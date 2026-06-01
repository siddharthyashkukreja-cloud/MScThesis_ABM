"""
data/v_gbm.py — Stein-Stein-style stochastic-volatility Merton
jump-diffusion calibration and path generation for the exogenous V_t
fundamental (D33; supersedes the D25 constant-σ Merton jump-diffusion).

V_t evolves as a GBM with a stochastic, mean-reverting volatility σ_t
(Vasicek / Ornstein-Uhlenbeck) and Poisson Merton jumps. The SV channel was
prompted by a Deloitte CCP Risk Model webinar (the white paper has no
equations and the ODD is incomplete on the fundamental); the formal
OU-on-σ model is Stein-Stein (1991), the jumps are Merton (1976):

    dσ_t  = α (θ − σ_t) dt + σ_vol dW^σ      (Vasicek-OU, 1-min cadence)
    Δlog V = (μ − ½σ_t²) + σ_t · Z + J,       Z ~ N(0,1)
    J = jump: with prob λ_step a draw from N(0, δ²), else 0

A Stein-Stein-like SV process: σ_t mean-reverts to long-run level θ with
speed α and per-step shock σ_vol. The persistent σ_t produces volatility
clustering — high |r| follows high |r|, naturally giving positive `|r|`
ACF at multi-step lags (the moment Deloitte calibrate via `acf_absr`).

**Calibration** is decoupled into three layers:

  1. Jump-diffusion (existing D25 logic) — `(σ_d, λ, δ)` from the
     variance/kurtosis decomposition of 1-min returns. `λ` is fixed at
     `JUMP_LAMBDA_DAY = 3/day` (ODD); `σ_d` and `δ` follow.

  2. Stochastic-vol (D33, NEW) — `(α, θ, σ_vol)` from method-of-moments
     on the realised-volatility time series. Non-overlapping 30-min
     windows of 1-min returns give RV_w = √(Σ r_i²); divide by √30 for a
     per-minute σ estimate. Then:
         θ          = mean(σ_per_min)
         Var(σ_t)   = Var(σ_per_min)  (stationary OU variance = σ_vol²/(2α))
         ρ_1 (30m)  = exp(−α · 30)    (OU autocorrelation at 30-min lag)
     solving for α (1/min), σ_vol (per √min), θ (per-minute σ).

  3. θ is then RE-ANCHORED so that `E[σ_t²] = σ_d²` — keeping total
     return variance equal to the jump-diffusion-only calibration. This
     ensures the SV layer adds *clustering* without inflating the model's
     volatility level.

`σ_t` is generated alongside V_t via the exact OU discretisation and
floored at a small positive value. The simulator reads only `V_smooth`
from `fv_{regime}.csv`, but the path is also written with a `sigma_t`
column for diagnostics.

Why this matters: D24-D27 established that long-memory volatility
clustering (`acf_absr` ACF) is structurally unreachable in a market layer
where σ is constant and clustering can only come from FT herding around
jumps. A Vasicek-OU σ_t injects an exogenous persistent vol channel that
the mid-anchored MM can transmit to the mid — the SV-demand mechanism of the
Gao-Chiarella-Heston literature (Gao et al. 2023), prompted by the Deloitte
CCP Risk Model webinar.

Caveat (Hill-gap cause): the jump/SV variance split is NOT jointly identified.
`calibrate()` assigns ALL empirical excess kurtosis to jumps assuming a
constant σ (the f below), THEN adds the OU layer and only re-anchors θ to
preserve VARIANCE — not kurtosis. So the SV layer's own kurtosis contribution
is double-counted into δ, biasing the jump size high → the validated 1-min
Hill comes out too heavy (~1.1 vs empirical ~3.25). Candidate fix: cap the
jump-variance share (f ≤ 0.5) or lower λ so the SV layer carries more of the
kurtosis.

Usage:
    python data/v_gbm.py calibrate                        # joint MJD + SV calibration
    python data/v_gbm.py generate <regime> <seed> [n_days]   # writes data/fv_{regime}.csv
    python data/v_gbm.py generate-all <seed> [n_days]        # writes both fv_{calm,stressed}.csv

n_days defaults to 120 — the V_t path length is decoupled from the
empirical sample length, so the stressed regime (~29 empirical days) can
still drive calibration runs of any horizon.
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

BARS_PER_DAY = 390   # 6.5-hour RTH day at the 1-min cadence
JUMP_LAMBDA_DAY = 3.0   # Merton jump intensity — jumps per RTH day (ODD §Calibration)
RV_WINDOW_MIN = 30   # Realised-vol window for SV calibration — non-overlapping (D33)


# ── data loading ─────────────────────────────────────────────────────────────

def load_returns(regime: str) -> dict:
    """Returns dict with ts (DatetimeIndex), mid, returns, is_open_bar (bool).
    `mid` is the end-of-minute (best_bid+best_ask)/2 — the same price series
    the agent calibration matches against (calibrate.py). Open-bar returns
    are those crossing a date boundary (overnight gap)."""
    df = pd.read_csv(PROC_DIR / f"ES_front_{regime}_1m.csv",
                     index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    log_p = np.log(mid)
    returns = np.concatenate([[np.nan], np.diff(log_p)])
    dates = pd.DatetimeIndex(df.index).date
    is_open_bar = np.zeros(len(df), dtype=bool)
    is_open_bar[0] = True
    is_open_bar[1:] = (dates[1:] != dates[:-1])
    return {"ts": df.index, "mid": mid,
            "returns": returns, "is_open_bar": is_open_bar}


# ── calibration entry point ──────────────────────────────────────────────────

def calibrate_sv(data: dict, window_min: int = RV_WINDOW_MIN) -> dict:
    """Vasicek-OU stochastic-vol calibration via method of moments on the
    realised-volatility time series (D33). RV is computed in
    non-overlapping `window_min`-bar windows; per-minute σ proxy is
    `RV / √window_min`. From the σ time series:
        θ        = mean(σ)
        Var(σ)   = σ_vol² / (2α)          (stationary OU variance)
        ρ_lag1   = exp(−α · window_min)   (OU autocorr at the RV cadence)
    Returns (α, θ, σ_vol) in 1-min units."""
    r = data["returns"].copy()
    valid = (~data["is_open_bar"]) & np.isfinite(r)
    r = np.where(valid, r, 0.0)
    n = len(r)
    n_wins = n // window_min
    r_chunks = r[: n_wins * window_min].reshape(n_wins, window_min)
    valid_chunks = (valid[: n_wins * window_min]
                    .reshape(n_wins, window_min).all(axis=1))
    rv = np.sqrt(np.sum(r_chunks ** 2, axis=1))           # window σ
    sigma_per_min = rv[valid_chunks] / np.sqrt(window_min)
    sigma_per_min = sigma_per_min[sigma_per_min > 0]
    theta = float(sigma_per_min.mean())
    var_s = float(np.var(sigma_per_min, ddof=1))
    # OU autocorr at lag 1 (window_min minutes between samples).
    corr_lag1 = float(np.corrcoef(sigma_per_min[:-1], sigma_per_min[1:])[0, 1])
    corr_lag1 = float(np.clip(corr_lag1, 1e-3, 0.999))
    alpha = -np.log(corr_lag1) / window_min               # per-minute
    sigma_vol = float(np.sqrt(2.0 * alpha * var_s))       # per √min
    return {
        "alpha": float(alpha), "theta": theta, "sigma_vol": sigma_vol,
        "rv_var": var_s, "rv_corr_lag1": corr_lag1,
        "window_min": int(window_min),
        "n_windows": int(valid_chunks.sum()),
    }


def calibrate(verbose: bool = True) -> dict:
    """Joint calibration per regime — Merton jump-diffusion (D25 logic)
    plus Vasicek-OU stochastic-vol (D33). See module docstring."""
    results = {}
    lam_step = JUMP_LAMBDA_DAY / BARS_PER_DAY
    for regime in REGIMES:
        data = load_returns(regime)
        valid = (~data["is_open_bar"]) & np.isfinite(data["returns"])
        r = data["returns"][valid]
        var = float(np.var(r, ddof=1))
        sigma_total = var ** 0.5
        mu = float(np.mean(r))
        z = (r - r.mean()) / r.std()
        ex_kurt = float((z ** 4).mean() - 3.0)
        # Jump-variance share (same algebra as D25 — assumes constant σ for
        # the diffusion piece; the SV layer (below) is added without
        # re-decomposing the kurtosis so that the calibration remains
        # closed-form per layer).
        f = (lam_step * max(ex_kurt, 0.0) / 3.0) ** 0.5
        f = min(f, 0.95)
        delta = (f * var / lam_step) ** 0.5
        sigma_d = ((1.0 - f) * var) ** 0.5
        sv = calibrate_sv(data)
        p = {
            "sigma_d": sigma_d,
            "jump_lambda_day": JUMP_LAMBDA_DAY,
            "jump_delta": delta,
            "mean_return": mu,
            "v0": float(data["mid"][0]),
            "sigma_total": sigma_total,
            "excess_kurtosis": ex_kurt,
            "jump_var_frac": float(f),
            "n_valid": int(valid.sum()),
            "sv": sv,
        }
        results[regime] = p
        if verbose:
            print(f"[{regime}] n={p['n_valid']}  sigma_total={sigma_total:.3e}  "
                  f"ex_kurt={ex_kurt:.1f}  ->  sigma_d={sigma_d:.3e}  "
                  f"lambda={JUMP_LAMBDA_DAY:.1f}/day  delta={delta:.3e}  "
                  f"(jumps carry {f*100:.0f}% of variance)")
            print(f"          SV: alpha={sv['alpha']:.4f}/min  theta={sv['theta']:.3e}  "
                  f"sigma_vol={sv['sigma_vol']:.3e}  "
                  f"(RV n={sv['n_windows']} @ {sv['window_min']}m, "
                  f"ρ_1={sv['rv_corr_lag1']:.3f})")
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "v_gbm_params.json"
    out_path.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"Saved {out_path}")
    return results


# ── generator: jump-diffusion path ───────────────────────────────────────────

REGIME_START = {"calm": "2019-01-02", "stressed": "2020-02-24"}
DEFAULT_N_DAYS = 120   # ~6 months — comfortably covers any calibration horizon
                       # (the path length is decoupled from the empirical
                       # sample length, so the stressed regime is not capped)


def generate(regime: str, seed: int, n_days: int = DEFAULT_N_DAYS,
             out_path: Path | None = None) -> np.ndarray:
    """Forward-simulate a Stein-Stein-like SV jump-diffusion V_t path of
    `n_days` RTH days (D33). σ_t is an OU process around θ_eff (chosen so
    that `E[σ_t²] = σ_d²`, preserving the D25 total return variance);
    returns are `(μ − ½σ_t²) + σ_t·Z + J`. Path length is a free parameter
    — NOT capped at the empirical sample length, so the stressed regime
    (~29 empirical days) can still drive long runs."""
    params_path = OUT_DIR / "v_gbm_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Run calibration first: {params_path}")
    p = json.loads(params_path.read_text())[regime]
    sigma_d, v0 = p["sigma_d"], p["v0"]
    mu = float(p.get("mean_return", 0.0))      # data-calibrated per-step drift
    delta = p["jump_delta"]
    lam_step = p["jump_lambda_day"] / BARS_PER_DAY
    sv = p["sv"]
    alpha, sigma_vol = sv["alpha"], sv["sigma_vol"]
    sv_var = sigma_vol * sigma_vol / (2.0 * alpha)        # stationary Var(σ_t)
    # Re-anchor θ so E[σ_t²] = θ² + Var(σ_t) = σ_d² (D33). If the
    # RV-implied σ_vol²/(2α) already exceeds σ_d², use the RV mean and
    # accept the variance overshoot — the agent calibration absorbs it.
    if sigma_d * sigma_d > sv_var:
        theta_eff = float((sigma_d * sigma_d - sv_var) ** 0.5)
    else:
        theta_eff = float(sv["theta"])
    n_steps = n_days * BARS_PER_DAY

    rng = np.random.default_rng(seed)

    # OU exact discretisation, Δt = 1 min: σ_{t+1} = θ + e^{-α}(σ_t − θ) + s·Z
    e_alpha = float(np.exp(-alpha))
    ou_sd = float(sigma_vol * np.sqrt((1.0 - np.exp(-2.0 * alpha)) / (2.0 * alpha)))
    eps_s = rng.standard_normal(n_steps)
    sigma_path = np.empty(n_steps)
    sigma_path[0] = theta_eff
    for t in range(n_steps - 1):
        sigma_path[t + 1] = (theta_eff + e_alpha * (sigma_path[t] - theta_eff)
                             + ou_sd * eps_s[t])
    sigma_path = np.maximum(sigma_path, 1e-10)            # floor: no negative vol

    # Returns: stochastic-σ Gaussian + Merton jumps.
    eps_r = rng.standard_normal(n_steps)
    jump_mask = rng.random(n_steps) < lam_step
    jumps = jump_mask * rng.normal(0.0, delta, n_steps)
    drift = mu - 0.5 * sigma_path * sigma_path
    log_increments = drift + sigma_path * eps_r + jumps
    log_increments[0] = 0.0
    V = np.exp(np.log(v0) + np.cumsum(log_increments))

    # Synthesised RTH timestamps (390 bars/day, business days). The
    # simulator only reads the V_smooth column; ts is cosmetic.
    days = pd.bdate_range(start=REGIME_START[regime], periods=n_days)
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=30)
          + pd.Timedelta(minutes=k) for d in days for k in range(BARS_PER_DAY)]
    ts = pd.DatetimeIndex(ts[:n_steps])

    if out_path is None:
        out_path = DATA_DIR / f"fv_{regime}.csv"
    out_path = Path(out_path)
    pd.DataFrame({"ts": ts, "V_smooth": V,
                  "sigma_t": sigma_path}).to_csv(out_path, index=False)
    print(f"[{regime}] seed={seed} n_days={n_days} n={n_steps}, mu={mu:.2e}, "
          f"{int(jump_mask.sum())} jumps, V0={V[0]:.2f}, V_end={V[-1]:.2f}, "
          f"range=[{V.min():.2f}, {V.max():.2f}]; "
          f"σ_t mean={sigma_path.mean():.3e}, std={sigma_path.std():.3e} "
          f"(θ_eff={theta_eff:.3e}) -> {out_path}")
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
            print("Usage: generate <regime> <seed> [n_days]"); sys.exit(1)
        regime = sys.argv[2]
        seed = int(sys.argv[3])
        n_days = int(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_N_DAYS
        generate(regime, seed, n_days)
    elif cmd == "generate-all":
        if len(sys.argv) < 3:
            print("Usage: generate-all <seed> [n_days]"); sys.exit(1)
        seed = int(sys.argv[2])
        n_days = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_N_DAYS
        for r in REGIMES:
            generate(r, seed, n_days)
    else:
        print(f"Unknown command: {cmd}\n"); print(__doc__); sys.exit(1)


if __name__ == "__main__":
    _main()
