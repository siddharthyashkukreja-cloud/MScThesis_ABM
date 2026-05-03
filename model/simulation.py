"""
Simulation driver -- Stage 1 (LOB + ZI only).

Step sequence mirrors the Simudyne ODD:
  1. ZI traders submit limit orders (depth-rate Poisson draws).
  2. ZI traders submit market orders.
  3. Call auction: LOB.match() clears at best available prices.
  4. LOB ages surviving orders (TTL decrement / expiry).
  5. Record snapshot.
  6. GlobalState advances fundamental (no-op in Stage 1, active Stage 2+).
"""

from typing import List, Dict
import numpy as np

from .globals import ModelParams, GlobalState
from .lob import LOB
from .agents import ZeroIntelligenceTrader


class Simulation:
    def __init__(self, params: ModelParams, traders: List[ZeroIntelligenceTrader],
                 seed: int = 42):
        self.params = params
        self.gs = GlobalState(params, seed=seed)
        self.lob = LOB(params.tick_size, params.order_ttl)
        self.traders = traders

        # Seed the LOB with a valid mid-price so tick 0 has a price reference
        self.lob.mid_price = params.v0
        self.lob.best_bid = params.v0 - params.tick_size
        self.lob.best_ask = params.v0 + params.tick_size

        self.history: Dict[str, List] = {
            "t": [], "mid_price": [], "spread": [],
            "bid_depth": [], "ask_depth": [], "volume": [],
            "fundamental": [],
        }

    def step(self) -> dict:
        for trader in self.traders:
            trader.submit_orders(
                lob=self.lob,
                params=self.params,
                fundamental=self.gs.v,
                tick=self.gs.t,
            )

        fills = self.lob.match()
        self.lob.age_orders()
        snap = self.lob.snapshot()

        self.history["t"].append(self.gs.t)
        self.history["mid_price"].append(snap["mid_price"])
        self.history["spread"].append(snap["spread"])
        self.history["bid_depth"].append(snap["bid_depth"])
        self.history["ask_depth"].append(snap["ask_depth"])
        self.history["volume"].append(snap["volume"])
        self.history["fundamental"].append(self.gs.v)

        self.gs.step()
        return snap

    def run(self, n_steps: int) -> Dict[str, List]:
        for _ in range(n_steps):
            self.step()
        return self.history
