"""
Entry point for Stage 1: LOB + ZI traders only.

Parameters are provisional; calibration from E-mini order-flow data
(Cont-Stoikov estimation) will replace zi_alpha, zi_mu, zi_delta once
calibrate.py is run against the daily BuyVol/SellVol data.
"""

import numpy as np
import pandas as pd
import os

from model.globals import ModelParams
from model.agents import ZeroIntelligenceTrader
from model.simulation import Simulation


def build_zi_traders(params: ModelParams, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [
        ZeroIntelligenceTrader(agent_id=i, cash=float(rng.uniform(1e5, 1e6)))
        for i in range(n)
    ]


def main():
    params = ModelParams(
        n_zi=10,
        n_fundamental=0,
        n_momentum=0,
        v0=450.0,           # approximate SPY level (calibrate later from data)
        tick_size=0.01,     # SPY $0.01 tick
        dt_minutes=5.0,
        order_ttl=2,        # 2 x 5-min steps ~ ODD 10 x 1-min steps
        # ZI rates (ODD defaults; Cont-Stoikov 2008 / Farmer-Daniels 2003)
        zi_alpha=0.15,
        zi_mu=0.025,
        zi_delta=0.025,
        zi_qty_min=1,
        zi_qty_max=10,
        zi_offset_max=5,
        sigma_v=0.0,        # fundamental frozen in Stage 1
    )

    traders = build_zi_traders(params, params.n_zi, seed=42)
    sim = Simulation(params, traders, seed=42)
    history = sim.run(n_steps=78)   # 1 trading day = 78 x 5-min steps (NYSE)

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv("output/stage1_run.csv", index=False)


if __name__ == "__main__":
    main()
