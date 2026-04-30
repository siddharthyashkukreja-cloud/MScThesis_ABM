"""
calibrate.py  —  KF + EM calibration of the Extended Chiarella model
                  following Majewski, Ciliberti & Bouchaud (2020).

State-space model (linear in hidden fundamental v_t):

    v_{t+1} = v_t + g + sigma_v * eps_t        (fundamental process)
    p_{t+1} = p_t + gamma*(v_t - p_t) + beta*u_t + sigma_n * eta_t

where u_t = tanh(m_t) is the observable momentum control,
m_t is the EWMA momentum signal computed from prices.

Hidden state:  v_t  (scalar)
Observation:   p_t  (scalar, log-price)

Parameters estimated via EM:
    theta = (lambda_, gamma, beta, g, sigma_v, sigma_n)

Fixed parameters (set analytically as per Majewski):
    alpha  = 1 / (1 + horizon)  where horizon = 5 days (one week)

Outputs:
    output/calibrated_params.csv   — estimated parameters per regime
    output/fundamental_value.csv   — filtered & smoothed V_t path per regime
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixed parameter (Majewski eq. 3.7 footnote 5)
# ---------------------------------------------------------------------------
ALPHA = 1.0 / (1.0 + 5.0)   # momentum EWMA decay, horizon = 5 trading days


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_regime(path: str) -> np.ndarray:
    """Load a regime CSV and return log-price array (using DPrice column)."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.dropna(subset=["DPrice"])
    return np.log(df["DPrice"].values)


# ---------------------------------------------------------------------------
# Momentum signal  m_t  (observable, computed from prices before filtering)
# ---------------------------------------------------------------------------

def compute_momentum(log_prices: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Discrete EWMA of log-returns:  m_{t+1} = (1 - alpha)*m_t + alpha*r_t
    Returns array of same length as log_prices (m[0] = 0).
    """
    T = len(log_prices)
    m = np.zeros(T)
    for t in range(1, T):
        r = log_prices[t] - log_prices[t - 1]
        m[t] = (1.0 - alpha) * m[t - 1] + alpha * r
    return m


# ---------------------------------------------------------------------------
# Kalman Filter  (E-step forward pass)
# ---------------------------------------------------------------------------

@dataclass
class KFResult:
    v_filt:   np.ndarray   # E[v_t | p_{1:t}]          filtered mean
    P_filt:   np.ndarray   # Var[v_t | p_{1:t}]        filtered variance
    v_pred:   np.ndarray   # E[v_t | p_{1:t-1}]        predicted mean
    P_pred:   np.ndarray   # Var[v_t | p_{1:t-1}]      predicted variance
    innov:    np.ndarray   # p_t - E[p_t | p_{1:t-1}]  innovation
    S:        np.ndarray   # Var of innovation
    K:        np.ndarray   # Kalman gain
    log_lik:  float        # predictive log-likelihood


def kalman_filter(
    log_prices: np.ndarray,
    u: np.ndarray,
    gamma: float,
    beta: float,
    g: float,
    sigma_v: float,
    sigma_n: float,
    v0: float,
    P0: float = 1.0,
) -> KFResult:
    """
    Run the Kalman filter for the linear state-space model.

    Transition:   v_{t+1} = v_t + g               + sigma_v * eps
    Observation:  p_{t+1} = (1-gamma)*p_t + gamma*v_t + beta*u_t + sigma_n * eta

    Re-arranged as observation equation at time t+1:
        p_{t+1} = gamma * v_t + c_t + sigma_n * eta
    where c_t = (1 - gamma)*p_t + beta*u_t   (known constant given past)

    Hidden state:  v_t  (scalar)
    """
    T = len(log_prices)

    v_pred = np.zeros(T)
    P_pred = np.zeros(T)
    v_filt = np.zeros(T)
    P_filt = np.zeros(T)
    innov  = np.zeros(T)
    S_arr  = np.zeros(T)
    K_arr  = np.zeros(T)

    # Initialise with first observation as prior for v
    v_filt[0] = v0
    P_filt[0] = P0

    log_lik = 0.0
    Q = sigma_v ** 2   # process noise variance
    R = sigma_n ** 2   # observation noise variance
    H = gamma          # observation matrix (scalar)

    for t in range(T - 1):
        # --- Predict ---
        v_pred[t + 1] = v_filt[t] + g
        P_pred[t + 1] = P_filt[t] + Q

        # --- Innovation ---
        c_t = (1.0 - gamma) * log_prices[t] + beta * u[t]
        y_hat = H * v_pred[t + 1] + c_t
        e = log_prices[t + 1] - y_hat
        S = H * P_pred[t + 1] * H + R

        innov[t + 1]  = e
        S_arr[t + 1]  = S

        # --- Update ---
        K = P_pred[t + 1] * H / S
        v_filt[t + 1] = v_pred[t + 1] + K * e
        P_filt[t + 1] = (1.0 - K * H) * P_pred[t + 1]
        K_arr[t + 1]  = K

        # --- Log-likelihood contribution ---
        log_lik += -0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)

    return KFResult(
        v_filt=v_filt, P_filt=P_filt,
        v_pred=v_pred, P_pred=P_pred,
        innov=innov, S=S_arr, K=K_arr,
        log_lik=log_lik,
    )


# ---------------------------------------------------------------------------
# Kalman Smoother  (Rauch-Tung-Striebel, needed for M-step expectations)
# ---------------------------------------------------------------------------

def kalman_smoother(kf: KFResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RTS smoother.  Returns:
        v_smooth[t]  = E[v_t | p_{1:T}]
        P_smooth[t]  = Var[v_t | p_{1:T}]
        C_smooth[t]  = Cov[v_{t+1}, v_t | p_{1:T}]   (cross-covariance, t=0..T-2)
    """
    T = len(kf.v_filt)
    v_s = np.copy(kf.v_filt)
    P_s = np.copy(kf.P_filt)
    C_s = np.zeros(T - 1)   # C_s[t] = Cov(v_{t+1}, v_t | Y)

    for t in range(T - 2, -1, -1):
        if kf.P_pred[t + 1] < 1e-15:
            J = 0.0
        else:
            J = kf.P_filt[t] / kf.P_pred[t + 1]   # smoother gain
        v_s[t] = kf.v_filt[t] + J * (v_s[t + 1] - kf.v_pred[t + 1])
        P_s[t] = kf.P_filt[t] + J * (P_s[t + 1] - kf.P_pred[t + 1]) * J
        C_s[t] = J * P_s[t + 1]

    return v_s, P_s, C_s


# ---------------------------------------------------------------------------
# M-step  (closed-form parameter updates, Majewski Appendix B)
# ---------------------------------------------------------------------------

def m_step(
    log_prices: np.ndarray,
    u: np.ndarray,
    v_s: np.ndarray,
    P_s: np.ndarray,
    C_s: np.ndarray,
) -> dict:
    """
    Closed-form M-step updates for (gamma, beta, g, sigma_v, sigma_n, v0).

    Observation equation at t=1..T-1:
        p_{t+1} = gamma * v_t + (1-gamma)*p_t + beta*u_t + sigma_n * eta

    So residual:  r_t = p_{t+1} - (1-gamma)*p_t - gamma*E[v_t] - beta*u_t
    We solve via OLS on the joint sufficient statistics.
    """
    T = len(log_prices)

    # Sufficient statistics for observation equation (t = 0 .. T-2)
    # regressors: [v_t, p_t, u_t]  ->  target: p_{t+1}
    p_next = log_prices[1:]        # T-1
    p_curr = log_prices[:-1]       # T-1
    v_curr = v_s[:-1]              # E[v_t]
    V_curr = P_s[:-1]              # Var[v_t]
    u_curr = u[:-1]
    n = T - 1

    # Build normal equations:  [gamma, (1-gamma), beta] via joint regression
    # Treat (1-gamma) as a free coefficient a, so p_{t+1} = gamma*v_t + a*p_t + beta*u_t
    # Then enforce a + gamma = 1  via post-processing.
    # For simplicity use unconstrained OLS then project.
    X = np.column_stack([v_curr, p_curr, u_curr])   # (n, 3)
    y = p_next

    # E[X^T X] — need to correct for uncertainty in v_t
    XtX = X.T @ X
    XtX[0, 0] += np.sum(V_curr)   # E[v_t^2] = (E[v_t])^2 + Var[v_t]

    XtX = XtX + 1e-8 * np.eye(3)  # ridge for stability
    XtY = X.T @ y
    # Correction: E[v_t * p_{t+1}] uses v_s only (p_{t+1} is observed)
    coeffs = np.linalg.solve(XtX, XtY)

    gamma_hat = float(np.clip(coeffs[0], 1e-4, 0.9999))
    # Enforce a + gamma = 1
    beta_hat  = float(coeffs[2])

    # Residual variance -> sigma_n
    resid = p_next - gamma_hat * v_curr - (1.0 - gamma_hat) * p_curr - beta_hat * u_curr
    # E[resid^2] needs variance correction
    sigma_n2 = float(np.mean(resid ** 2) + gamma_hat ** 2 * np.mean(V_curr))
    sigma_n_hat = float(np.sqrt(max(sigma_n2, 1e-10)))

    # Sufficient statistics for transition equation (t = 0 .. T-2)
    Ev_next  = v_s[1:]            # E[v_{t+1}]
    Ev_curr  = v_s[:-1]
    Ev2_next = P_s[1:]  + v_s[1:] ** 2
    Ev2_curr = P_s[:-1] + v_s[:-1] ** 2
    Ecross   = C_s[:n]  + v_s[1:] * v_s[:-1]

    # g:  E[v_{t+1} - v_t] averaged
    g_hat = float(np.mean(Ev_next - Ev_curr))

    # sigma_v^2 = E[(v_{t+1} - v_t - g)^2]
    sigma_v2 = float(np.mean(
        Ev2_next - 2.0 * Ecross + Ev2_curr
        - 2.0 * g_hat * (Ev_next - Ev_curr)
        + g_hat ** 2
    ))
    sigma_v_hat = float(np.sqrt(max(sigma_v2, 1e-10)))

    # Initial fundamental value
    v0_hat = float(v_s[0])

    return dict(
        gamma=gamma_hat,
        beta=beta_hat,
        g=g_hat,
        sigma_v=sigma_v_hat,
        sigma_n=sigma_n_hat,
        v0=v0_hat,
    )


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------

def em_calibrate(
    log_prices: np.ndarray,
    regime: str,
    max_iter: int = 10000000,
    tol: float = 1e-6,
) -> dict:
    """
    Run EM to convergence.  Returns dict of estimated parameters
    plus the smoothed fundamental value path.
    """
    T = len(log_prices)
    print(f"\nCalibrating [{regime}] over {T} observations...")

    # Pre-compute observable momentum control  u_t = tanh(m_t)
    m = compute_momentum(log_prices, alpha=ALPHA)
    u = np.tanh(m)

    # --- Initialise parameters ---
    daily_ret_std = float(np.std(np.diff(log_prices)))
    params = dict(
        gamma   = 0.05,
        beta    = 0.10,
        g       = float(np.mean(np.diff(log_prices))),
        sigma_v = daily_ret_std * 0.5,
        sigma_n = daily_ret_std,
        v0      = log_prices[0],
    )

    prev_ll = -np.inf

    for i in range(max_iter):
        # --- E-step: Kalman filter + smoother ---
        kf = kalman_filter(
            log_prices, u,
            gamma   = params["gamma"],
            beta    = params["beta"],
            g       = params["g"],
            sigma_v = params["sigma_v"],
            sigma_n = params["sigma_n"],
            v0      = params["v0"],
        )
        v_s, P_s, C_s = kalman_smoother(kf)

        # --- M-step: closed-form updates ---
        new_params = m_step(log_prices, u, v_s, P_s, C_s)
        params.update(new_params)

        # --- Convergence check ---
        ll = kf.log_lik
        delta = abs(ll - prev_ll)
        if i > 0 and delta < tol:
            print(f"  Converged at iteration {i+1}  (ΔLL={delta:.2e})  LL={ll:.4f}")
            break
        prev_ll = ll

        if (i + 1) % 20 == 0:
            print(f"  iter {i+1:3d}  LL={ll:.4f}")
    else:
        print(f"  Max iterations reached.  LL={prev_ll:.4f}")

    # --- Final filter pass for output ---
    kf = kalman_filter(
        log_prices, u,
        gamma   = params["gamma"],
        beta    = params["beta"],
        g       = params["g"],
        sigma_v = params["sigma_v"],
        sigma_n = params["sigma_n"],
        v0      = params["v0"],
    )
    v_s, P_s, _ = kalman_smoother(kf)

    # Map to ModelParams naming convention
    result = {
        "regime"  : regime,
        "lambda_" : float(np.nan),   # estimated separately from order flow data
        "gamma"   : params["gamma"],
        "beta"    : params["beta"],
        "alpha"   : ALPHA,
        "g"       : params["g"],
        "sigma_v" : params["sigma_v"],
        "sigma_n" : params["sigma_n"],
        "v0"      : params["v0"],
        "log_lik" : kf.log_lik,
        "n_obs"   : T,
        "v_filtered": v_s,          # full path — saved separately
        "v_stderr"  : np.sqrt(P_s), # posterior std of V_t
    }

    print(f"    gamma    = {params['gamma']:.6f}")
    print(f"    beta     = {params['beta']:.6f}")
    print(f"    alpha    = {ALPHA:.6f}  (fixed)")
    print(f"    g        = {params['g']:.6f}")
    print(f"    sigma_v  = {params['sigma_v']:.6f}")
    print(f"    sigma_n  = {params['sigma_n']:.6f}")
    print(f"    v0       = {params['v0']:.6f}")
    print(f"    log_lik  = {kf.log_lik:.4f}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate Extended Chiarella model via KF + EM"
    )
    parser.add_argument("--calm",     default="data/thesis_data_calm.csv")
    parser.add_argument("--stressed", default="data/thesis_data_stressed.csv")
    parser.add_argument("--regime",   default="both",
                        choices=["stressed", "calm", "both"])
    parser.add_argument("--out",      default="output/calibrated_params.csv")
    parser.add_argument("--max_iter", default=200, type=int)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    results = []
    paths = []

    if args.regime in ("stressed", "both"):
        obs_s = load_regime(args.stressed)
        res_s = em_calibrate(obs_s, "stressed", max_iter=args.max_iter)
        results.append({k: v for k, v in res_s.items()
                        if k not in ("v_filtered", "v_stderr")})
        paths.append(("stressed", obs_s, res_s["v_filtered"], res_s["v_stderr"]))

    if args.regime in ("calm", "both"):
        obs_c = load_regime(args.calm)
        res_c = em_calibrate(obs_c, "calm", max_iter=args.max_iter)
        results.append({k: v for k, v in res_c.items()
                        if k not in ("v_filtered", "v_stderr")})
        paths.append(("calm", obs_c, res_c["v_filtered"], res_c["v_stderr"]))

    # Save parameter table
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nSaved parameters to {args.out}")

    # Save fundamental value paths
    for regime_name, obs, v_s, v_std in paths:
        out_v = f"output/fundamental_value_{regime_name}.csv"
        pd.DataFrame({
            "t"          : np.arange(len(obs)),
            "log_price"  : obs,
            "v_smoothed" : v_s,
            "v_stderr"   : v_std,
        }).to_csv(out_v, index=False)
        print(f"Saved fundamental value path to {out_v}")
