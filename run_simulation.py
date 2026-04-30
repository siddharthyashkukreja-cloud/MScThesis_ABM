"""
run_simulation.py  —  Run ABM with calibrated parameters for each regime.

Loads calibrated parameters from output/calibrated_params.csv,
constructs ModelParams, runs N_RUNS simulations per regime,
and saves aggregated output for analysis.ipynb comparison.

Outputs (per regime):
    output/sim_{regime}_runs.csv      — all runs stacked (t, run_id, price, fundamental)
    output/sim_{regime}_median.csv    — median + IQR path across runs
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from model.globals import ModelParams, GlobalState
from model.market import Market
from model.agents import FundamentalTrader, MomentumTrader, NoiseTrader
from model.simulation import Simulation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_RUNS   = 50    # Monte Carlo runs per regime
SEED_BASE = 0    # seeds: SEED_BASE + run_id

REGIME_STEPS = {
    "calm"    : 503,
    "stressed": 252,
}

# Fixed agent counts
N_FUNDAMENTAL = 30
N_MOMENTUM    = 10
N_NOISE       = 10


# ---------------------------------------------------------------------------
# Build ModelParams from calibrated row
# ---------------------------------------------------------------------------

def build_params(row: pd.Series, n_steps: int) -> ModelParams:
    """
    Map KF-EM estimates to ModelParams.

    Calibrated fields:  gamma, beta, alpha (fixed), g, sigma_v, sigma_n, v0
    The remaining fields (lambda_, kappa, delta, etc.) use sensible defaults
    consistent with the Majewski linear demand formulation.
    """
    # Fundamental value initial level from calibration
    v0 = float(row["v0"])

    # lambda_: Majewski do not estimate this from price data alone.
    # Use a small positive value so price impact is present but weak.
    # Can be updated once order-flow data is available.
    lambda_     = 1
    lambda_tran = 0.005
    rho_tran    = 0.9

    # kappa: fundamentalist demand scale.  In the linear KF model,
    # kappa is absorbed into gamma (gamma = lambda * kappa in Majewski eq 3.4).
    # Recover: kappa = gamma / lambda_
    gamma = float(row["gamma"])
    kappa = gamma / lambda_

    # delta: dead-band.  Set to 1 std of mispricing ~ sigma_n.
    delta = float(row["sigma_n"]) * 0.5

    return ModelParams(
        n_noise        = N_NOISE,
        n_fundamental  = N_FUNDAMENTAL,
        n_momentum     = N_MOMENTUM,
        lambda_        = lambda_,
        lambda_tran    = lambda_tran,
        rho_tran       = rho_tran,
        v0             = v0,
        m0             = 0.0,
        price_distortion = 0.0,    # start at fair value
        mu_v           = float(row["g"]),
        sigma_v        = float(row["sigma_v"]),
        kappa          = kappa,
        delta          = delta,
        alpha          = float(row["alpha"]),
        beta           = float(row["beta"]),
        gamma          = float(row["gamma"]),
        sigma_n        = float(row["sigma_n"]),
        p_noise        = 1.0,          # noise traders active every step
        noise_size_dist  = "fixed",
        noise_size_param = 1.0,
    )


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def single_run(params: ModelParams, n_steps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    traders = [
        FundamentalTrader(params, rng),
        MomentumTrader(params, rng),
        NoiseTrader(params, rng),
    ]

    sim = Simulation(params, traders, seed=seed)
    history = sim.run(n_steps)

    df = pd.DataFrame({
        "t"          : history["t"],
        "price"      : history["price"],
        "fundamental": history["fundamental"],
        "momentum"   : history["momentum"],
    })
    # Compute log versions to match calibration output
    df["log_price"]       = np.log(np.clip(df["price"],       1e-10, None))
    df["log_fundamental"] = np.log(np.clip(df["fundamental"], 1e-10, None))
    df["mispricing"]      = df["log_price"] - df["log_fundamental"]
    df["log_return"]      = df["log_price"].diff()
    return df


# ---------------------------------------------------------------------------
# Monte Carlo over N_RUNS
# ---------------------------------------------------------------------------

def run_regime(regime: str, row: pd.Series) -> None:
    n_steps = REGIME_STEPS[regime]
    params  = build_params(row, n_steps)

    print(f"\nSimulating [{regime}]  {N_RUNS} runs x {n_steps} steps...")
    print(f"  gamma={params.gamma:.4f}  beta={params.beta:.4f}  "
          f"sigma_v={params.sigma_v:.5f}  sigma_n={params.sigma_n:.5f}  "
          f"g={params.mu_v:.6f}  v0={params.v0:.4f}")

    all_runs = []
    for run_id in range(N_RUNS):
        df = single_run(params, n_steps, seed=SEED_BASE + run_id)
        df["run_id"] = run_id
        all_runs.append(df)

    stacked = pd.concat(all_runs, ignore_index=True)
    out_runs = f"output/sim_{regime}_runs.csv"
    stacked.to_csv(out_runs, index=False)
    print(f"  Saved {len(stacked)} rows to {out_runs}")

    # Median + IQR per timestep
    grp = stacked.groupby("t")
    median = pd.DataFrame({
        "t"             : grp["t"].first(),
        "log_price_med" : grp["log_price"].median(),
        "log_price_p25" : grp["log_price"].quantile(0.25),
        "log_price_p75" : grp["log_price"].quantile(0.75),
        "log_fund_med"  : grp["log_fundamental"].median(),
        "mispricing_med": grp["mispricing"].median(),
        "mispricing_p25": grp["mispricing"].quantile(0.25),
        "mispricing_p75": grp["mispricing"].quantile(0.75),
        "log_ret_med"   : grp["log_return"].median(),
        "log_ret_std"   : grp["log_return"].std(),
    }).reset_index(drop=True)

    out_med = f"output/sim_{regime}_median.csv"
    median.to_csv(out_med, index=False)
    print(f"  Saved median path to {out_med}")

    # Print quick diagnostics
    all_rets = stacked["log_return"].dropna()
    from scipy import stats as sp
    kurt = sp.kurtosis(all_rets)
    skew = sp.skew(all_rets)
    mis_std = stacked["mispricing"].std()
    print(f"  Return skew={skew:.3f}  excess_kurt={kurt:.3f}  "
          f"mispricing_std={mis_std:.5f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    params_df = pd.read_csv("output/calibrated_params.csv")

    for regime in ["calm", "stressed"]:
        row = params_df[params_df.regime == regime].iloc[0]
        run_regime(regime, row)

    print("\nDone. Now run analysis.ipynb to compare simulated vs empirical.")
