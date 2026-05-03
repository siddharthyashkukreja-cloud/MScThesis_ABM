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

    # Noise traders
    p_participate = 1.0 / params.n_noise if params.n_noise > 0 else 0.0
    for _ in range(params.n_noise):
        traders.append(
            NoiseTrader(
                params,
                rng,
                participation_prob=p_participate,
                vol_scale=1.0,
            )
        )

    # Fundamental traders
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(params, rng))

    # Momentum traders
    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(params, rng))

    return traders


def main():
    params = ModelParams(
        n_noise=50,
        n_fundamental=0,
        n_momentum=0,
        lambda_=1e-4,
        v0=100.0,
        m0=0.0,
        price_distortion=0.0,
        mu_v=0.0,
        sigma_v=0.1,
        kappa=0.1,
        alpha=0.1,
        beta=0.5,
        gamma=10.0,
        sigma_n=1.0,
    )

    traders = build_traders(params, seed=123)
    sim = Simulation(params, traders, seed=42)
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
  