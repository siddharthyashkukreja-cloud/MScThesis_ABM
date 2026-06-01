"""
data/v_kalman.py — Kalman-filter calibration of V_t (efficient price + noise).

A state-space alternative to data/v_gbm.py. The plain GBM fit uses σ_v = std
of observed 1-min log-returns, which conflates EFFICIENT-PRICE σ_v with
MICROSTRUCTURE NOISE σ_ε (bid-ask bounce, discreteness, transient impact).
Since the ABM is meant to MANUFACTURE that microstructure noise endogenously,
baking it into the exogenous V_t double-counts.

State-space form (one independent sequence per RTH day, so the overnight
gap is never bridged):

    State        log V_t (latent efficient log-price)
    Transition   log V_{t+1} = log V_t + μ + σ_v · Z,    Z ~ N(0,1)
    Observation  log p_t     = log V_t + ε_t,             ε_t ~ N(0, σ_ε²)

MLE via the Kalman log-likelihood (prediction-error decomposition), L-BFGS-B
on (μ, log σ_v, log σ_ε), pooled across days.

Sanity check: Roll (1984) closed-form, valid under the same i.i.d.-noise model
(returns are then MA(1)):
        σ_ε² = −γ_1   (lag-1 autocovariance of returns)
        σ_v² =  γ_0 + 2 γ_1

This is a DIAGNOSTIC — it does not write fv CSVs. It reports σ_v_KF, σ_ε, and
the percent change vs the GBM σ_v in output/v_gbm_params.json. If the change
is material (≥ 10–20%) adopt the KF σ_v: copy it into v_gbm_params.json and
regenerate the paths with `data/v_gbm.py generate-all`. Otherwise the direct
fit is fine and you've shown it.

Caveat: real microstructure noise is autocorrelated (bid-ask bounce → richer
than pure MA(1)). The close-price series (last trade in the 1-min bar) is the
most bounce-prone — the BBO mid would be cleaner. The KF σ_ε absorbs the
noise variance correctly; the SHAPE assumption (i.i.d.) affects σ_v slightly.
A later refinement could extend to a colored-noise observation model.

Usage:
    python data/v_kalman.py calibrate
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
REGIME_START = {"calm": "2019-01-02", "stressed": "2020-02-24"}   # cosmetic fv ts
BARS_PER_DAY = 390


# ── data loading: per-day intra-RTH log-price sequences ─────────────────────

def load_per_day_log_prices(regime: str, column: str = "close") -> list:
    """Return a list of np.ndarray, one per RTH date, of log-prices. Splitting
    by date ensures the overnight gap is never crossed inside the filter (the
    same convention as data/v_gbm.py's open-bar exclusion). `column` is "close"
    for the diagnostic and "mid" for the fundamental generator (the mid is the
    calibration target and is less bounce-prone)."""
    df = pd.read_csv(PROC_DIR / f"ES_front_{regime}_1m.csv",
                     index_col=0, parse_dates=True)
    px = df[column].to_numpy(dtype=float)
    log_p = np.log(px)
    dates = pd.DatetimeIndex(df.index).date
    days, cur_start = [], 0
    for i in range(1, len(dates)):
        if dates[i] != dates[i - 1]:
            seq = log_p[cur_start:i]
            if len(seq) >= 10:
                days.append(seq)
            cur_start = i
    seq = log_p[cur_start:]
    if len(seq) >= 10:
        days.append(seq)
    return days


# ── Kalman log-likelihood (scalar state) ────────────────────────────────────

def kalman_neg_loglik(theta: np.ndarray, days: list) -> float:
    """Negative Kalman log-likelihood under the local-level state-space
    (μ, σ_v, σ_ε) parameterised as (μ, log σ_v, log σ_ε)."""
    mu, lsv, lse = theta
    q = float(np.exp(2.0 * lsv))   # σ_v²
    r = float(np.exp(2.0 * lse))   # σ_ε²
    acc = 0.0
    for y in days:
        # initial state: anchor on the first obs, with one-period obs uncertainty
        x = float(y[0])
        P = r
        for t in range(1, len(y)):
            x_pred = x + mu
            P_pred = P + q
            e = float(y[t]) - x_pred
            S = P_pred + r
            acc += np.log(S) + e * e / S
            K = P_pred / S
            x = x_pred + K * e
            P = (1.0 - K) * P_pred
    n_obs = sum(len(y) - 1 for y in days)
    return 0.5 * (acc + n_obs * np.log(2.0 * np.pi))


# ── Roll (1984) closed-form ─────────────────────────────────────────────────

def roll_estimator(days: list) -> dict:
    """Roll's bid-ask-noise decomposition (σ_v, σ_ε) from γ_0 and γ_1 of the
    pooled intra-day returns. Only identified when γ_1 < 0 (bounce-style
    negative autocorrelation at lag 1)."""
    rets = np.concatenate([np.diff(y) for y in days if len(y) > 1])
    if len(rets) < 2:
        return {}
    gamma_0 = float(np.var(rets, ddof=1))
    mean = float(rets.mean())
    gamma_1 = float(np.mean(rets[1:] * rets[:-1]) - mean * mean)
    if gamma_1 >= 0:
        return {
            "gamma_0": gamma_0, "gamma_1": gamma_1, "valid": False,
            "note": "γ_1 ≥ 0 — Roll's i.i.d.-noise identification fails; "
                    "use KF MLE instead.",
        }
    sigma_eps = float(np.sqrt(-gamma_1))
    sigma_v_sq = gamma_0 + 2.0 * gamma_1
    return {
        "gamma_0": gamma_0, "gamma_1": gamma_1, "valid": True,
        "sigma_v": float(np.sqrt(max(sigma_v_sq, 0.0))),
        "sigma_eps": sigma_eps,
    }


# ── Kalman smoother + fundamental-path generator (XGB-Chiarella §2.5.2) ──────

def _kalman_smooth(y: np.ndarray, mu: float, q: float, r: float) -> np.ndarray:
    """RTS smoother for the local-level model (state = latent log V_t, obs =
    log p_t; q = σ_v², r = σ_ε²). Returns E[log V_t | all obs] — the latent
    EFFICIENT log-price with microstructure noise removed. Pure numpy."""
    n = len(y)
    xf = np.empty(n); Pf = np.empty(n); xp = np.empty(n); Pp = np.empty(n)
    x = float(y[0]); P = r
    xf[0] = xp[0] = x; Pf[0] = Pp[0] = P
    for t in range(1, n):
        xpr = x + mu; Ppr = P + q
        xp[t] = xpr; Pp[t] = Ppr
        S = Ppr + r; K = Ppr / S
        x = xpr + K * (float(y[t]) - xpr); P = (1.0 - K) * Ppr
        xf[t] = x; Pf[t] = P
    xs = xf.copy()
    for t in range(n - 2, -1, -1):
        C = Pf[t] / Pp[t + 1]
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
    return xs


def _fit_params(days: list) -> tuple:
    """(μ, σ_v², σ_ε²) for the local-level model — Kalman MLE (scipy) when
    available, else the Roll (1984) closed form, else an equal variance split.
    The numpy fallbacks keep the generator runnable without scipy."""
    rets = np.concatenate([np.diff(y) for y in days])
    raw = float(rets.std(ddof=1)); mu0 = float(rets.mean())
    try:
        from scipy.optimize import minimize
        x0 = np.array([mu0, np.log(raw / np.sqrt(2.0)), np.log(raw / np.sqrt(2.0))])
        res = minimize(kalman_neg_loglik, x0, args=(days,), method="L-BFGS-B",
                       options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 200})
        mu, lsv, lse = res.x
        return float(mu), float(np.exp(2.0 * lsv)), float(np.exp(2.0 * lse))
    except Exception:
        roll = roll_estimator(days)
        if roll.get("valid"):
            return mu0, float(roll["sigma_v"] ** 2), float(roll["sigma_eps"] ** 2)
        v = float(np.var(rets, ddof=1))
        return mu0, 0.5 * v, 0.5 * v


def _local_vol(rets: np.ndarray, halflife: float = 30.0) -> np.ndarray:
    """Per-minute local volatility = sqrt(EWMA[r²]), half-life `halflife` min.
    This σ_t carries the REAL volatility clustering (the multi-timescale
    persistence the synthetic single-OU SV of D33 could not) and feeds the FT
    belief width (D34: σ_fund_t = √390·σ_t·v0)."""
    if len(rets) == 0:
        return np.array([0.0])
    lam = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = np.empty(len(rets)); v = float(rets[0] ** 2)
    for t in range(len(rets)):
        v = (1.0 - lam) * v + lam * float(rets[t] ** 2); out[t] = v
    return np.sqrt(out)


def generate(regime: str, out_path=None) -> np.ndarray:
    """Write data/fv_{regime}.csv with a Kalman-SMOOTHED real fundamental
    (V_smooth) plus the real local volatility (sigma_t). The latent efficient
    price is RTS-smoothed per RTH day, then days are spliced CONTINUOUSLY
    (overnight gaps removed) so the simulator sees no spurious day-boundary
    jumps. This is the XGB-Chiarella §2.5.2 data-derived fundamental — an
    alternative to the synthetic SV-MJD of data/v_gbm.py, and ODD-faithful (the
    ODD §Mech #9 fundamental signal is itself a historical data series). It
    carries the REAL return tails and REAL volatility clustering, which the
    agent layer cannot manufacture (see the calibration residuals)."""
    days = load_per_day_log_prices(regime, column="mid")
    mu, q, r = _fit_params(days)
    logV_parts, sig_parts = [], []
    prev_end = None
    for y in days:
        S = _kalman_smooth(y, mu, q, r)
        logV = S.copy() if prev_end is None else prev_end + (S - S[0])
        prev_end = float(logV[-1])
        logV_parts.append(logV)
        sig = _local_vol(np.diff(y))
        sig_parts.append(np.concatenate([[sig[0]], sig]))   # length == len(y)
    V_smooth = np.exp(np.concatenate(logV_parts))
    sigma_t = np.maximum(np.concatenate(sig_parts), 1e-10)
    n = len(V_smooth)
    n_days = int(np.ceil(n / BARS_PER_DAY))
    bdays = pd.bdate_range(start=REGIME_START[regime], periods=n_days)
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=30) + pd.Timedelta(minutes=k)
          for d in bdays for k in range(BARS_PER_DAY)][:n]
    out_path = Path(out_path) if out_path else (DATA_DIR / f"fv_{regime}.csv")
    pd.DataFrame({"ts": ts, "V_smooth": V_smooth,
                  "sigma_t": sigma_t}).to_csv(out_path, index=False)
    print(f"[{regime}] Kalman fundamental -> {out_path}  n={n} (~{n_days}d)  "
          f"σ_v={np.sqrt(q):.3e} σ_ε={np.sqrt(r):.3e}  "
          f"V0={V_smooth[0]:.2f} Vend={V_smooth[-1]:.2f}  σ_t mean={sigma_t.mean():.3e}")
    return V_smooth


# ── end-to-end calibration ──────────────────────────────────────────────────

def calibrate(verbose: bool = True) -> dict:
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise RuntimeError("scipy required for the KF MLE. "
                           "Install with `pip install scipy`.")
    OUT_DIR.mkdir(exist_ok=True)
    direct = {}
    gbm_path = OUT_DIR / "v_gbm_params.json"
    if gbm_path.exists():
        direct = json.loads(gbm_path.read_text())

    results = {}
    for regime in REGIMES:
        days = load_per_day_log_prices(regime)
        if not days:
            print(f"[{regime}] no data")
            continue
        rets = np.concatenate([np.diff(y) for y in days])
        raw_std = float(rets.std(ddof=1))
        mu0 = float(rets.mean())
        # init: split return variance equally between state and noise.
        x0 = np.array([mu0,
                       np.log(raw_std / np.sqrt(2.0)),
                       np.log(raw_std / np.sqrt(2.0))])
        res = minimize(kalman_neg_loglik, x0, args=(days,),
                       method="L-BFGS-B",
                       options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 200})
        mu, lsv, lse = res.x
        sigma_v_kf = float(np.exp(lsv))
        sigma_eps_kf = float(np.exp(lse))
        roll = roll_estimator(days)
        gbm_sigma = direct.get(regime, {}).get("sigma")
        gap_pct = ((sigma_v_kf - gbm_sigma) / gbm_sigma * 100.0
                   if gbm_sigma else None)
        results[regime] = {
            "kf": {
                "mu": float(mu),
                "sigma_v": sigma_v_kf,
                "sigma_eps": sigma_eps_kf,
                "neg_loglik": float(res.fun),
                "converged": bool(res.success),
                "n_days": len(days),
                "n_obs": int(sum(len(y) for y in days)),
            },
            "roll_1984": roll,
            "gbm_direct_sigma": gbm_sigma,
            "kf_vs_gbm_pct_change": gap_pct,
            "raw_return_std": raw_std,
        }
        if verbose:
            _print(regime, results[regime])

    out = OUT_DIR / "v_kalman_params.json"
    out.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"\nSaved {out}")
        print("→ DIAGNOSTIC only: this does not write fv CSVs.")
        print("→ If |kf_vs_gbm_pct_change| is material (≥ 10-20%), copy the KF")
        print("  σ_v into output/v_gbm_params.json (sigma field) and regenerate")
        print("  the paths with `data/v_gbm.py generate-all 42 100`.")
    return results


def _print(regime: str, r: dict):
    kf = r["kf"]
    roll = r["roll_1984"] or {}
    conv = "✓" if kf["converged"] else "✗"
    print(f"\n[{regime}]  n_days={kf['n_days']}  n_obs={kf['n_obs']}  conv={conv}")
    print(f"  KF MLE:    μ={kf['mu']:+.2e}/min   "
          f"σ_v={kf['sigma_v']:.6f}   σ_ε={kf['sigma_eps']:.6f}")
    if roll.get("valid"):
        print(f"  Roll '84:  σ_v={roll['sigma_v']:.6f}   "
              f"σ_ε={roll['sigma_eps']:.6f}   "
              f"(γ_0={roll['gamma_0']:.2e}, γ_1={roll['gamma_1']:.2e})")
    elif roll:
        print(f"  Roll '84:  INVALID — γ_1={roll['gamma_1']:.2e} ≥ 0  "
              f"(no bounce-style negative autocorr; use KF MLE)")
    if r["gbm_direct_sigma"] is not None:
        print(f"  GBM σ_v = {r['gbm_direct_sigma']:.6f}   "
              f"→ KF vs GBM change: {r['kf_vs_gbm_pct_change']:+.1f}%")
    print(f"  raw return std = {r['raw_return_std']:.6f}  "
          f"(should ≈ √(σ_v² + 2σ_ε²) = "
          f"{np.sqrt(kf['sigma_v']**2 + 2*kf['sigma_eps']**2):.6f})")


def _main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calibrate()
    elif cmd == "generate":
        generate(sys.argv[2] if len(sys.argv) > 2 else "calm")
    elif cmd == "generate-all":
        for r in REGIMES:
            generate(r)
    else:
        print(f"Unknown command: {cmd}\n"); print(__doc__); sys.exit(1)


if __name__ == "__main__":
    _main()
