"""
run_simulation.py  --  Monte Carlo with KF-calibrated parameters.

Uses log_price_mode=True (Option A): agents output log-price increments
directly, exactly matching the KF state equation.  lambda_ is unused.

Outputs saved to output/:
    sim_{regime}_runs.csv     -- all runs stacked
    sim_{regime}_median.csv   -- per-timestep median + IQR
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats as sp

from model.globals import ModelParams
from model.agents import FundamentalTrader, MomentumTrader, NoiseTrader
from model.simulation import Simulation

# ---------------------------------------------------------------------------
N_RUNS    = 50
SEED_BASE = 0
REGIME_STEPS = {"calm": 503, "stressed": 252}
N_FUNDAMENTAL = 30
N_MOMENTUM    = 10
N_NOISE       = 10


# ---------------------------------------------------------------------------
def build_params(row: pd.Series) -> ModelParams:
    """
    Map KF-EM calibration row -> ModelParams for log_price_mode.

    KF coefficients used directly:
      gamma   -> FundamentalTrader: dp = gamma * log(v/p)
      beta    -> MomentumTrader:    dp = beta  * tanh(m)
      sigma_n -> NoiseTrader:       dp ~ N(0, sigma_n)
      sigma_v -> fundamental GRW log-vol
      g       -> fundamental GRW log-drift

    lambda_, kappa, lambda_tran, rho_tran retained for volume-mode
    backward compatibility but not used here.
    """
    return ModelParams(
        n_noise          = N_NOISE,
        n_fundamental    = N_FUNDAMENTAL,
        n_momentum       = N_MOMENTUM,
        lambda_          = 1.0,          # unused in log mode
        lambda_tran      = 0.0,
        rho_tran         = 0.0,
        v0               = float(row["v0"]),
        m0               = 0.0,
        price_distortion = 0.0,
        mu_v             = float(row["g"]),
        sigma_v          = float(row["sigma_v"]),
        kappa            = 1.0,          # unused in log mode
        delta            = float(row["sigma_n"]) * 0.5,
        alpha            = float(row["alpha"]),
        beta             = float(row["beta"]),
        gamma            = float(row["gamma"]),
        sigma_n          = float(row["sigma_n"]),
        p_noise          = 1.0,
        noise_size_dist  = "fixed",
        noise_size_param = 1.0,
        log_price_mode   = True,
    )


# ---------------------------------------------------------------------------
def single_run(params: ModelParams, n_steps: int, seed: int) -> pd.DataFrame:
    rng     = np.random.default_rng(seed)
    traders = [
        FundamentalTrader(params, rng),
        MomentumTrader(params, rng),
        NoiseTrader(params, rng),
    ]
    sim     = Simulation(params, traders, seed=seed)
    history = sim.run(n_steps)

    df = pd.DataFrame({
        "t"          : history["t"],
        "price"      : history["price"],
        "fundamental": history["fundamental"],
        "momentum"   : history["momentum"],
        "vol"        : history["vol"],
    })
    df["log_price"]       = np.log(np.clip(df["price"],       1e-10, None))
    df["log_fundamental"] = np.log(np.clip(df["fundamental"], 1e-10, None))
    df["mispricing"]      = df["log_price"] - df["log_fundamental"]
    df["log_return"]      = df["log_price"].diff()
    return df


# ---------------------------------------------------------------------------
def run_regime(regime: str, row: pd.Series) -> None:
    n_steps = REGIME_STEPS[regime]
    params  = build_params(row)

    print(f"\nSimulating [{regime}]  {N_RUNS} runs x {n_steps} steps  [log_price_mode]")
    print(f"  gamma={params.gamma:.4f}  beta={params.beta:.4f}  "
          f"sigma_v={params.sigma_v:.5f}  sigma_n={params.sigma_n:.5f}  "
          f"g={params.mu_v:.6f}  v0={params.v0:.4f}")

    all_runs = []
    for run_id in range(N_RUNS):
        df = single_run(params, n_steps, seed=SEED_BASE + run_id)
        df["run_id"] = run_id
        all_runs.append(df)

    stacked = pd.concat(all_runs, ignore_index=True)
    stacked.to_csv(f"output/sim_{regime}_runs.csv", index=False)

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
    median.to_csv(f"output/sim_{regime}_median.csv", index=False)

    # Diagnostics vs empirical target
    all_rets = stacked["log_return"].dropna()
    target   = float(row["sigma_n"])
    sim_vol  = all_rets.std()
    print(f"  Target sigma_n={target:.5f}  Sim vol={sim_vol:.5f}  ratio={sim_vol/max(target,1e-12):.2f}")
    print(f"  Skew={sp.skew(all_rets):.3f}  "
          f"ExKurt={sp.kurtosis(all_rets):.3f}  "
          f"MispricingStd={stacked['mispricing'].std():.5f}")
    print(f"  Saved {len(stacked)} rows -> output/sim_{regime}_runs.csv")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    params_df = pd.read_csv("output/calibrated_params.csv")
    for regime in ["calm", "stressed"]:
        row = params_df[params_df.regime == regime].iloc[0]
        run_regime(regime, row)
    print("\nDone. Open analysis.ipynb to compare simulated vs empirical.")
