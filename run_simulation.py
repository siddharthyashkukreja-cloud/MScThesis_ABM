"""
run_simulation.py  --  Monte Carlo with KF-calibrated parameters.

Uses log_price_mode=True (Option A): agents output log-price increments
directly, exactly matching the KF state equation.

Outputs saved to output/:
    sim_{regime}_runs.csv        -- per-step market data, all runs stacked
    sim_{regime}_trades.csv      -- per-agent trade log, all runs stacked
    sim_{regime}_margin.csv      -- per-step margin/balance-sheet aggregates
    sim_{regime}_median.csv      -- per-timestep median + IQR (market data)
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
N_RUNS         = 50
SEED_BASE      = 0
REGIME_STEPS   = {"calm": 503, "stressed": 252}
N_FUNDAMENTAL  = 30
N_MOMENTUM     = 10
N_NOISE        = 10


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
    """
    return ModelParams(
        n_noise          = N_NOISE,
        n_fundamental    = N_FUNDAMENTAL,
        n_momentum       = N_MOMENTUM,
        lambda_          = 1.0,
        lambda_tran      = 0.0,
        rho_tran         = 0.0,
        v0               = float(row["v0"]),
        m0               = 0.0,
        price_distortion = 0.0,
        mu_v             = float(row["g"]),
        sigma_v          = float(row["sigma_v"]),
        kappa            = 1.0,
        delta            = float(row["sigma_n"]) * 0.5,
        alpha            = float(row["alpha"]),
        beta             = float(row["beta"]),
        gamma            = float(row["gamma"]),
        sigma_n          = float(row["sigma_n"]),
        p_noise          = 1.0,
        noise_size_dist  = "fixed",
        noise_size_param = 1.0,
        log_price_mode   = True,
        initial_wealth   = 1_000.0,  # normalised wealth units
        im_rate          = 0.10,     # 10% IM (EMIR RTS 2016/2251 floor)
    )


# ---------------------------------------------------------------------------
def build_traders(params: ModelParams, rng: np.random.Generator):
    """Build labelled trader list with sequential agent IDs."""
    traders, aid = [], 0
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(params, rng, agent_id=aid)); aid += 1
    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(params,    rng, agent_id=aid)); aid += 1
    for _ in range(params.n_noise):
        traders.append(NoiseTrader(params,       rng, agent_id=aid)); aid += 1
    return traders


# ---------------------------------------------------------------------------
def single_run(params: ModelParams, n_steps: int, seed: int,
               run_id: int):
    rng     = np.random.default_rng(seed)
    traders = build_traders(params, rng)
    sim     = Simulation(params, traders, seed=seed)
    history = sim.run(n_steps)

    market_df = pd.DataFrame({
        "run_id"     : run_id,
        "t"          : history["t"],
        "price"      : history["price"],
        "fundamental": history["fundamental"],
        "momentum"   : history["momentum"],
        "vol"        : history["vol"],
    })
    market_df["log_price"]  = np.log(np.clip(market_df["price"],       1e-10, None))
    market_df["log_fund"]   = np.log(np.clip(market_df["fundamental"], 1e-10, None))
    market_df["mispricing"] = market_df["log_price"] - market_df["log_fund"]
    market_df["log_return"] = market_df["log_price"].diff()

    trades_df = pd.DataFrame(sim.trade_log)
    if not trades_df.empty:
        trades_df.insert(0, "run_id", run_id)

    margin_df = pd.DataFrame({
        "run_id"        : run_id,
        "t"             : history["t"],
        "n_margin_calls": history["n_margin_calls"],
        "n_defaults"    : history["n_defaults"],
        "n_new_defaults": history["n_new_defaults"],
        "system_equity" : history["system_equity"],
        "total_margin"  : history["total_margin"],
    })

    return market_df, trades_df, margin_df


# ---------------------------------------------------------------------------
def run_regime(regime: str, row: pd.Series) -> None:
    n_steps = REGIME_STEPS[regime]
    params  = build_params(row)
    n_agents = params.n_fundamental + params.n_momentum + params.n_noise

    print(f"\nSimulating [{regime}]  {N_RUNS} runs x {n_steps} steps  "
          f"({n_agents} agents, log_price_mode)")
    print(f"  gamma={params.gamma:.4f}  beta={params.beta:.4f}  "
          f"sigma_v={params.sigma_v:.5f}  sigma_n={params.sigma_n:.5f}  "
          f"g={params.mu_v:.6f}  v0={params.v0:.4f}  im_rate={params.im_rate:.0%}")

    all_market, all_trades, all_margin = [], [], []
    for run_id in range(N_RUNS):
        m, t, mg = single_run(params, n_steps, seed=SEED_BASE + run_id,
                               run_id=run_id)
        all_market.append(m)
        all_trades.append(t)
        all_margin.append(mg)

    market_stack = pd.concat(all_market, ignore_index=True)
    trade_stack  = pd.concat(all_trades, ignore_index=True)
    margin_stack = pd.concat(all_margin, ignore_index=True)

    market_stack.to_csv(f"output/sim_{regime}_runs.csv",   index=False)
    trade_stack.to_csv( f"output/sim_{regime}_trades.csv", index=False)
    margin_stack.to_csv(f"output/sim_{regime}_margin.csv", index=False)

    grp = market_stack.groupby("t")
    pd.DataFrame({
        "t"             : grp["t"].first(),
        "log_price_med" : grp["log_price"].median(),
        "log_price_p25" : grp["log_price"].quantile(0.25),
        "log_price_p75" : grp["log_price"].quantile(0.75),
        "log_fund_med"  : grp["log_fund"].median(),
        "mispricing_med": grp["mispricing"].median(),
        "mispricing_p25": grp["mispricing"].quantile(0.25),
        "mispricing_p75": grp["mispricing"].quantile(0.75),
        "log_ret_std"   : grp["log_return"].std(),
    }).reset_index(drop=True).to_csv(f"output/sim_{regime}_median.csv", index=False)

    all_rets = market_stack["log_return"].dropna()
    target   = float(row["sigma_n"])
    print(f"  Target sigma_n={target:.5f}  Sim vol={all_rets.std():.5f}  "
          f"ratio={all_rets.std()/max(target, 1e-12):.2f}")
    print(f"  Skew={sp.skew(all_rets):.3f}  ExKurt={sp.kurtosis(all_rets):.3f}")

    mg_last = margin_stack.groupby("run_id").last()
    print(f"  Margin calls/step (avg): {margin_stack['n_margin_calls'].mean():.2f}")
    print(f"  Total defaults (avg/run): {mg_last['n_defaults'].mean():.1f}/{n_agents}")
    print(f"  Saved {len(market_stack):,} market | "
          f"{len(trade_stack):,} trade | "
          f"{len(margin_stack):,} margin rows")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    params_df = pd.read_csv("output/calibrated_params.csv")
    for regime in ["calm", "stressed"]:
        row = params_df[params_df.regime == regime].iloc[0]
        run_regime(regime, row)
    print("\nDone.")
