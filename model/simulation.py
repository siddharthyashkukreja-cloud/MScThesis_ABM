# mscthesis_abm/model/simulation.py
from typing import List, Dict, Any
import numpy as np

from .globals import ModelParams, GlobalState
from .market import Market, MarketState
from .agents import BaseTrader


class Simulation:
    """
    Top-level simulation driver for the Extended Chiarella ABM.
    """

    def __init__(
        self,
        params: ModelParams,
        traders: List[BaseTrader],
        seed: int = 42,
    ):
        self.params  = params
        self.globals = GlobalState(params, seed=seed)
        self.market  = Market(params, self.globals)
        self.traders = traders

        self.history: Dict[str, List[float]] = {
            "t"            : [],
            "price"        : [],
            "fundamental"  : [],
            "momentum"     : [],
            "excess_demand": [],
        }

    def step(self) -> None:
        market_state = self.market.get_state()

        # 1. Traders observe
        for trader in self.traders:
            trader.observe(market_state)

        # 2. Traders decide orders
        orders       = []
        active_orders = []
        for trader in self.traders:
            side, volume = trader.decide(market_state)
            orders.append((side, volume))
            if side != 0 and volume > 0:
                active_orders.append((side, volume))

        # 3. Market clears — called ONCE on active orders
        stats = self.market.step(active_orders)
        new_price = self.market.price

        # 4. Update all traders' inventory/PnL
        for trader, (side, vol) in zip(self.traders, orders):
            trader.inventory += side * vol
            trader.cash      -= side * vol * new_price
            trader.mtm_pnl    = trader.inventory * (new_price - trader.entry_price)

        # 5. Record
        self.history["t"].append(self.globals.t)
        self.history["price"].append(self.market.price)
        self.history["fundamental"].append(self.globals.v)
        self.history["momentum"].append(self.market.momentum)
        self.history["excess_demand"].append(self.market.excess_demand)

        # 6. Advance fundamental process
        self.globals.step_fundamental()

    def run(self, n_steps: int) -> Dict[str, List[float]]:
        for _ in range(n_steps):
            self.step()
        return self.history
