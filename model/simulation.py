# mscthesis_abm/model/simulation.py
from typing import List, Dict
import numpy as np

from .globals import ModelParams, GlobalState
from .market import Market, MarketState
from .agents import BaseTrader


class Simulation:
    """
    Top-level simulation driver.

    Dispatches to market.step_log() or market.step() based on
    params.log_price_mode.  All outputs (price, inventory, mtm_pnl,
    excess_demand) are structurally identical so the downstream
    margin-call / CCP stress framework needs no changes.
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

        self.history: Dict[str, list] = {
            "t"            : [],
            "price"        : [],
            "fundamental"  : [],
            "momentum"     : [],
            "excess_demand": [],
            "vol"          : [],
        }

    def step(self) -> None:
        market_state = self.market.get_state()

        for trader in self.traders:
            trader.observe(market_state)

        decisions = [trader.decide(market_state) for trader in self.traders]

        if self.params.log_price_mode:
            stats = self.market.step_log(decisions)
        else:
            active = [(s, v) for s, v in decisions if s != 0 and v > 0]
            stats  = self.market.step(active)

        new_price = self.market.price

        # Inventory / cash / MtM -- identical formula in both modes.
        # In log mode, magnitude is a dimensionless log-increment used
        # as a stylised position size for margin-call accounting.
        for trader, (side, mag) in zip(self.traders, decisions):
            trader.inventory += side * mag
            trader.cash      -= side * mag * new_price
            trader.mtm_pnl    = trader.inventory * (new_price - trader.entry_price)

        self.history["t"].append(self.globals.t)
        self.history["price"].append(self.market.price)
        self.history["fundamental"].append(self.globals.v)
        self.history["momentum"].append(self.market.momentum)
        self.history["excess_demand"].append(self.market.excess_demand)
        self.history["vol"].append(self.market.current_vol())

        self.globals.step_fundamental()

    def run(self, n_steps: int) -> Dict[str, list]:
        for _ in range(n_steps):
            self.step()
        return self.history
