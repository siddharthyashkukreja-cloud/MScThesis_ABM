# mscthesis_abm/model/simulation.py
from typing import List, Dict, Any
import numpy as np

from .globals import ModelParams, GlobalState
from .market import Market, MarketState
from .agents import BaseTrader


class Simulation:
    """
    Top-level simulation driver for the trading model.
    """

    def __init__(
        self,
        params: ModelParams,
        traders: List[BaseTrader],
        seed: int = 42,
    ):
        self.params = params
        self.globals = GlobalState(params, seed=seed)
        self.market = Market(params, self.globals)
        self.traders = traders

        # Simple containers for recording; you can replace with pandas later
        self.history: Dict[str, List[float]] = {
            "t": [],
            "price": [],
            "fundamental": [],
            "momentum": [],
            "excess_demand": [],
        }

    def step(self) -> None:
        """
        Run one simulation step.
        """
        market_state = self.market.get_state()

        # Traders observe current market
        for trader in self.traders:
            trader.observe(market_state)

       # Traders decide orders
        orders = []          # full list, one entry per trader (includes (0, 0.0) for non-traders)
        active_orders = []   # only non-zero orders passed to market

        for trader in self.traders:
            side, volume = trader.decide(market_state)
            orders.append((side, volume))
            if side != 0 and volume > 0:
                active_orders.append((side, volume))

        # Market clears on active orders only
        stats = self.market.step(active_orders)
        new_price = self.market.price

        # Update ALL traders' inventory/PnL using full aligned list
        for trader, (side, vol) in zip(self.traders, orders):
            trader.inventory += side * vol
            trader.cash      -= side * vol * new_price
            trader.mtm_pnl    = trader.inventory * (new_price - trader.entry_price)


        # Market clears
        stats = self.market.step(orders)

        # Record
        self.history["t"].append(self.globals.t)
        self.history["price"].append(self.market.price)
        self.history["fundamental"].append(self.globals.v)
        self.history["momentum"].append(self.market.momentum)
        self.history["excess_demand"].append(self.market.excess_demand)

        # Advance fundamental process
        self.globals.step_fundamental()

    def run(self, n_steps: int) -> Dict[str, List[float]]:
        """
        Run the simulation for n_steps and return the recorded history.
        """
        for _ in range(n_steps):
            self.step()
        return self.history