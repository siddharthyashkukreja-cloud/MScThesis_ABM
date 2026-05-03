# calibrate.py

import pandas as pd
import numpy as np
from pathlib import Path


def load_emini(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def estimate_zi_params(df: pd.DataFrame, step_seconds: float = 60.0) -> dict:
    """
    Estimate ZI parameters alpha, mu, delta from E-mini tick data.

    Method: Vytelingum et al. (2025) Section 3.3.1 — treat each rate as
    a Poisson process; MLE estimate is empirical frequency per simulation step.

    Parameters
    ----------
    df : DataFrame with columns [timestamp, event_type]
         event_type in {"limit", "market", "cancel"}
    step_seconds : simulation step length in real seconds

    Returns
    -------
    dict with keys: alpha, mu, delta, n_steps
    """
    duration_sec = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    n_steps = max(1, duration_sec / step_seconds)

    counts = df["event_type"].value_counts()
    n_limit  = counts.get("limit",  0)
    n_market = counts.get("market", 0)
    n_cancel = counts.get("cancel", 0)

    # rates per step (Bernoulli probability approximation for low rates)
    alpha = n_limit  / n_steps
    mu    = n_market / n_steps
    delta = n_cancel / n_steps

    # clip to valid probability range
    alpha = float(np.clip(alpha, 1e-4, 0.99))
    mu    = float(np.clip(mu,    1e-4, 0.99))
    delta = float(np.clip(delta, 1e-4, 0.99))

    return {"alpha": alpha, "mu": mu, "delta": delta, "n_steps": int(n_steps)}


def estimate_fundamental_params(df: pd.DataFrame, price_col: str = "mid_price") -> dict:
    """
    Estimate GBM fundamental parameters (mu_f, sigma_f) from mid-price series.
    Uses log-return MLE — consistent with ODD data-driven fundamental signal.
    """
    prices = df[price_col].dropna().values
    log_returns = np.diff(np.log(prices))
    mu_f    = float(np.mean(log_returns))
    sigma_f = float(np.std(log_returns))
    return {"mu_f": mu_f, "sigma_f": sigma_f, "initial_price": float(prices[0])}


def calibrate(data_path: str, output_path: str = "output/calibrated_params.csv",
              step_seconds: float = 60.0):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = load_emini(data_path)

    zi = estimate_zi_params(df, step_seconds=step_seconds)
    fund = estimate_fundamental_params(df)

    results = {**zi, **fund}
    pd.DataFrame([results]).to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    params = calibrate("data/emini_ticks.csv")
    for k, v in params.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")