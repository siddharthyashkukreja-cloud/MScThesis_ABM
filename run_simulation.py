"""
Entry point for Stage 1: LOB + ZI traders.

Parameters are provisional; calibration from data will follow once
Cont-Stoikov estimation on the daily order-flow data is complete.
"""

import numpy as np
import pandas as pd
from model.globals import ModelParams
from model.agents import ZeroIntelligenceTrader
from model.simulation import Simulation


def build_zi_traders(params: ModelParams, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [ZeroIntelligenceTrader(params, rng, agent_id=i) for i in range(n)]


def main():
    params = ModelParams(
        n_zi=100,
        n_fundamental=0,
        n_momentum=0,
        v0=4500.0,          # approximate E-mini level (calibrate later)
        tick_size=0.25,     # E-mini tick
        dt_minutes=5.0,
        order_ttl=2,        # 2 x 5-min steps ~ ODD's 10 x 1-min steps
        # ZI rates (provisional; to be estimated from BuyVol/SellVol data)
        lambda_lo=0.15,
        mu_mo=0.025,
        delta_co=0.025,
        depth_k=1.0,
        depth_alpha=0.5,
        zi_qty_min=1,
        zi_qty_max=10,
        sigma_v=0.0,        # fundamental frozen in Stage 1
    )

    traders = build_zi_traders(params, params.n_zi, seed=42)
    sim = Simulation(params, traders, seed=42)
    history = sim.run(n_steps=510)

    import os; os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv("output/stage1_run.csv", index=False)
    print(df[["t", "mid_price", "spread", "volume"]].tail(10))


if __name__ == "__main__":
    main()
