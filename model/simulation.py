# mscthesis_abm/model/simulation.py
from typing import List, Dict
import numpy as np

from .globals import ModelParams, GlobalState
from .market import Market, MarketState
from .agents import BaseTrader
from .margin import MarginEngine 


class Simulation:
    """
    Top-level simulation driver.

    Each step:
      1. Agents observe market state
      2. Agents submit decisions (side, magnitude)
      3. Market clears -> new price
      4. Inventory, cash, entry-price updated per agent (VWAP)
      5. MarginEngine runs -> margin calls, defaults recorded

    self.history       -- per-step market + margin aggregates
    self.trade_log     -- per-agent per-step trade records
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
        self.margin_engine = MarginEngine(traders, params)

        self.history: Dict[str, list] = {
            "t"             : [],
            "price"         : [],
            "fundamental"   : [],
            "momentum"      : [],
            "excess_demand" : [],
            "vol"           : [],
            "n_margin_calls": [],
            "n_defaults"    : [],
            "n_new_defaults": [],
            "system_equity" : [],
            "total_margin"  : [],
        }
        self.trade_log: List[dict] = []

    def step(self) -> None:
        t            = self.globals.t
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

        # ── Inventory / cash / VWAP entry-price update ─────────────────
        for trader, (side, mag) in zip(self.traders, decisions):
            if side == 0 or mag <= 0.0 or trader.defaulted:
                self.trade_log.append({
                    "t"         : t,
                    "agent_id"  : trader.agent_id,
                    "agent_type": trader.agent_type,
                    "side"      : 0,
                    "magnitude" : 0.0,
                    "price"     : new_price,
                    "inventory" : trader.inventory,
                })
                continue

            old_inv = trader.inventory
            new_inv = old_inv + side * mag

            # VWAP entry-price update
            if abs(new_inv) < 1e-12:
                trader.entry_price = new_price          # flat: reset
            elif (old_inv >= 0 and side > 0) or (old_inv <= 0 and side < 0):
                # Adding to existing direction: update VWAP
                trader.entry_price = (
                    old_inv * trader.entry_price + side * mag * new_price
                ) / new_inv
            # Reducing position: entry price unchanged (realised PnL on close)

            trader.inventory  = new_inv
            trader.cash      -= side * mag * new_price

            self.trade_log.append({
                "t"         : t,
                "agent_id"  : trader.agent_id,
                "agent_type": trader.agent_type,
                "side"      : side,
                "magnitude" : mag,
                "price"     : new_price,
                "inventory" : trader.inventory,
            })

        # ── Margin engine ─────────────────────────────────────────────
        margin_result = self.margin_engine.step(new_price)

        # ── Record history ────────────────────────────────────────────
        self.history["t"].append(t)
        self.history["price"].append(new_price)
        self.history["fundamental"].append(self.globals.v)
        self.history["momentum"].append(self.market.momentum)
        self.history["excess_demand"].append(self.market.excess_demand)
        self.history["vol"].append(self.market.current_vol())
        self.history["n_margin_calls"].append(margin_result.n_calls)
        self.history["n_defaults"].append(margin_result.n_defaults)
        self.history["n_new_defaults"].append(margin_result.n_new_default)
        self.history["system_equity"].append(margin_result.system_equity)
        self.history["total_margin"].append(margin_result.total_margin)

        self.globals.step_fundamental()

    def run(self, n_steps: int) -> Dict[str, list]:
        for _ in range(n_steps):
            self.step()
        return self.history
