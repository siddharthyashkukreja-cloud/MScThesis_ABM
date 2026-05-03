"""
Simulation driver -- Stage 1 (LOB + ZI only).

Step sequence follows the Simudyne ODD §Step Sequence (Stage 1 subset):
  1. ZI traders cancel resting orders + submit new limit / market orders.
  2. MatchingEngine: call-auction LOB.match() clears crossed quotes.
  3. LOB.age_orders(): TTL decrement / expiry.
  4. Record snapshot.
  5. GlobalState advances fundamental (frozen in Stage 1: sigma_v=0).
"""

from typing import List, Dict

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
                rng=self.gs.rng,
            )

        self.lob.match()
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
