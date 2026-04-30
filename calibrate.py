# scripts/calibrate.py
"""
UKF-based calibration of the extended Chiarella model.
Calibrate once on stressed period (e.g. GFC 2008-2009) and
once on non-stressed (e.g. 2012-2019).

Usage:
    python scripts/calibrate.py --data data/spy_5min.csv --regime calm
    python scripts/calibrate.py --data data/spy_5min.csv --regime stressed
"""

import argparse
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from model.globals import ModelParams
from model.simulation import Simulation
from model.agents import FundamentalTrader, MomentumTrader, NoiseTrader


# ── 1. State transition function (model equations) ─────────────────────────

def fx(x, dt, params: ModelParams, rng: np.random.Generator):
    """
    Propagate hidden state one step:
      x = [price, momentum, fundamental]
    This mirrors what Market.step() and GlobalState.step_fundamental() do.
    """
    p, m, v = x

    # Mispricing
    mispricing = v - p

    # Fundamental trader demand (threshold + Poisson mean)
    if abs(mispricing) > params.delta:
        lam = params.kappa * abs(mispricing)
        qf  = np.sign(mispricing) * rng.poisson(lam)
    else:
        qf = 0.0

    # Momentum trader demand
    qm = np.sign(m) * params.beta * np.tanh(params.gamma * abs(m))

    # Noise trader demand (expected value = 0, variance = sigma_n^2)
    qn = params.sigma_n * rng.normal()

    # Excess demand and price impact
    D     = qf + qm + qn
    p_new = p + params.lambda_ * D

    # Momentum update (EWMA)
    m_new = params.alpha * (p_new - p) + (1 - params.alpha) * m

    # Fundamental value step
    v_new = v + params.mu_v + params.sigma_v * rng.normal()

    return np.array([p_new, m_new, v_new])


def hx(x):
    """Observation: we observe only the price."""
    return np.array([x[0]])


def _make_pd(P: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Force a matrix to be symmetric positive-definite."""
    P = (P + P.T) / 2                          # symmetrise
    eigvals = np.linalg.eigvalsh(P)
    if eigvals.min() <= 0:
        P += (-eigvals.min() + jitter) * np.eye(P.shape[0])
    return P

# ── 2. Negative log-likelihood via UKF ─────────────────────────────────────

def ukf_nll(theta_vec, obs: np.ndarray, regime: str) -> float:
    (lambda_, kappa, delta, alpha, beta, gamma,
     sigma_v, sigma_n, mu_v) = theta_vec

    if any(v <= 0 for v in [lambda_, kappa, alpha, beta, gamma, sigma_v, sigma_n]):
        return 1e10
    if not (0 < alpha < 1):
        return 1e10

    rng = np.random.default_rng(42)

    params = ModelParams(
        n_noise=50, n_fundamental=25, n_momentum=25,
        lambda_=lambda_, lambda_tran=lambda_*2, rho_tran=0.5,
        v0=obs[0], m0=0.0, price_distortion=0.0,
        mu_v=mu_v, sigma_v=sigma_v,
        kappa=kappa, delta=delta,
        alpha=alpha, beta=beta, gamma=gamma,
        sigma_n=sigma_n, p_noise=0.2,
        noise_size_dist="geometric", noise_size_param=0.3,
    )

    points = MerweScaledSigmaPoints(n=3, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=3, dim_z=1,
        dt=1.0,
        fx=lambda x, dt: fx(x, dt, params, rng),
        hx=hx,
        points=points,
    )

    ukf.x = np.array([obs[0], 0.0, obs[0]])
    ukf.P = np.diag([1.0, 0.1, 1.0])

    # FIX 1 — Q scaled to log-return magnitudes, not raw price
    ukf.Q = np.diag([sigma_n**2, (alpha * sigma_v)**2, sigma_v**2]) * 1e-4
    ukf.R = np.array([[1e-4]])

    nll = 0.0
    for y in obs[1:]:
        try:
            ukf.predict()

            # FIX 2 — symmetrise P after predict before update
            ukf.P = _make_pd(ukf.P)

            ukf.update(np.array([y]))

            # FIX 3 — use slogdet instead of det (numerically stable),
            #          and pinv instead of inv for safety
            innov = ukf.y
            S     = ukf.S
            S     = _make_pd(S, jitter=1e-10)   # ensure S is PD too
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:                        # S went singular — skip step
                continue
            nll += 0.5 * (float(innov.T @ np.linalg.pinv(S) @ innov) + logdet)

        except np.linalg.LinAlgError:
            return 1e10   # this parameter set is numerically infeasible

    return nll


# ── 3. Optimisation ─────────────────────────────────────────────────────────

def calibrate(obs: np.ndarray, regime: str) -> dict:
    # Initial guess: [lambda_, kappa, delta, alpha, beta, gamma, sigma_v, sigma_n, mu_v]
    theta0 = np.array([0.01, 1.0, 0.5, 0.1, 1.0, 1.0, 0.5, 0.5, 0.0])

    bounds = [
        (1e-5, 1.0),   # lambda_
        (1e-3, 10.0),  # kappa
        (0.0,  5.0),   # delta
        (0.01, 0.99),  # alpha
        (0.1,  10.0),  # beta
        (0.1,  10.0),  # gamma
        (1e-4, 5.0),   # sigma_v
        (1e-4, 5.0),   # sigma_n
        (-0.1, 0.1),   # mu_v
    ]

    print(f"\nCalibrating [{regime}] over {len(obs)} observations...")
    result = minimize(
        ukf_nll,
        theta0,
        args=(obs, regime),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    names = ["lambda_", "kappa", "delta", "alpha", "beta", "gamma",
             "sigma_v", "sigma_n", "mu_v"]
    params_out = dict(zip(names, result.x))
    params_out["regime"]  = regime
    params_out["nll"]     = result.fun
    params_out["success"] = result.success

    print(f"  Converged: {result.success}  NLL: {result.fun:.4f}")
    for k, v in params_out.items():
        if k not in ("regime", "success"):
            print(f"    {k:12s} = {v:.6f}")

    return params_out


# ── 4. Data loading & regime splitting ──────────────────────────────────────

def load_and_split(calm_path: str, stressed_path: str):
    """
    Load pre-split calm and stressed CSV files.
    Uses DPrice (daily closing price) as the observed price series.
    Returns log-price arrays.
    """
    calm_df    = pd.read_csv(calm_path,    parse_dates=["date"]).sort_values("date").dropna(subset=["DPrice"])
    stressed_df = pd.read_csv(stressed_path, parse_dates=["date"]).sort_values("date").dropna(subset=["DPrice"])

    calm_obs    = np.log(calm_df["DPrice"].values)
    stressed_obs = np.log(stressed_df["DPrice"].values)

    return stressed_obs, calm_obs

# ── 5. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calm",     default="data/thesis_data_calm.csv")
    parser.add_argument("--stressed", default="data/thesis_data_stressed.csv")
    parser.add_argument("--regime",   default="both", choices=["stressed", "calm", "both"])
    parser.add_argument("--out",      default="output/calibrated_params.csv")
    args = parser.parse_args()

    stressed_obs, calm_obs = load_and_split(args.calm, args.stressed)

    results = []
    if args.regime in ("stressed", "both"):
        results.append(calibrate(stressed_obs, "stressed"))
    if args.regime in ("calm", "both"):
        results.append(calibrate(calm_obs, "calm"))

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")