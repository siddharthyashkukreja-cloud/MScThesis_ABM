# scripts/run_trading_core.py
import os
import csv
import numpy as np

from model.globals import ModelParams
from model.simulation import Simulation
from model.agents import FundamentalTrader, MomentumTrader, NoiseTrader

def build_traders(params: ModelParams, seed: int = 123):
    rng = np.random.default_rng(seed)
    traders = []

    # Noise traders — participation now driven by p_noise, not 1/N
    for _ in range(params.n_noise):
        traders.append(NoiseTrader(params, rng))

    # Fundamental traders
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(params, rng))

    # Momentum traders
    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(params, rng))

    return traders


def main():
    params = ModelParams(
        # Population
        n_noise        = 20,
        n_fundamental  = 30,
        n_momentum     = 10,

        # Price impact
        lambda_        = 1e-3,       # permanent impact
        lambda_tran    = 5e-3,       # transitory impact (5× permanent)
        rho_tran       = 0.8,        # 80% of transient decays each step

        # Fundamental process
        v0             = 100.0,
        m0             = 0.0,
        price_distortion = 2.0,      # start 2 points below fundamental
        mu_v           = 0.0,
        sigma_v        = 0.05,       # per-step fundamental vol

        # Fundamental trader
        kappa          = 0.05,
        delta          = 2,        # dead-band half-width around fundamental

        # Momentum trader
        alpha          = 0.1,
        beta           = 0.5,
        gamma          = 10.0,

        # Noise trader
        sigma_n        = 1.0,
        p_noise        = 0.20,           # 20% chance to trade per step
        noise_size_dist  = "geometric",
        noise_size_param = 0.3,          # mean size ≈ 1/0.3 ≈ 3.3 units × sigma_n
    )

    traders = build_traders(params, seed=123)
    sim     = Simulation(params, traders, seed=42)
    history = sim.run(n_steps=50_000)

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "trading_core_history.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for i in range(len(history["t"])):
            writer.writerow([history[k][i] for k in history.keys()])

    print(f"Saved history to {out_path}")


if __name__ == "__main__":
    main()