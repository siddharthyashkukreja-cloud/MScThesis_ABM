"""
Agent definitions.

Stage 1: ZeroIntelligenceTrader only.
Stage 2+ hooks: BaseTrader balance-sheet fields and abstract observe/decide
interface remain unchanged so FundamentalTrader and MomentumTrader can be
added without touching the LOB or simulation wiring.

ZI parametrisation follows Cont-Stoikov (2008):
  lambda_lo  : limit order arrival rate at best-quote distance 1
  depth(i)   : lambda_lo * depth_k * i^(-depth_alpha)   i >= 1
  mu_mo      : market order rate per side per step
  delta_co   : cancellation prob per resting order per step

The vol_proxy slot on ZeroIntelligenceTrader is intentionally left for the
Gao (2023) extension where ZI order size scales with stochastic variance.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .globals import ModelParams
from .lob import LOB


@dataclass
class BalanceSheet:
    cash: float
    inventory: int       # signed position in asset units
    margin_posted: float = 0.0
    margin_called: bool = False
    defaulted: bool = False

    def mark_to_market(self, price: float) -> float:
        return self.cash + self.inventory * price


class BaseTrader:
    def __init__(self, params: ModelParams, rng: np.random.Generator,
                 agent_id: int, agent_type: str):
        self.params = params
        self.rng = rng
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.bs = BalanceSheet(cash=0.0, inventory=0)

    def submit_orders(self, lob: LOB, mid_price: float,
                      fundamental: float, momentum: float,
                      vol_proxy: float) -> None:
        raise NotImplementedError


class ZeroIntelligenceTrader(BaseTrader):
    """
    Generates stochastic limit and market orders independent of fundamentals.
    Order arrival follows Cont-Stoikov Poisson rates; sizes are uniform [min, max].

    vol_proxy hook: when a Gao-style stochastic variance is available, pass
    it as vol_proxy and the order size will scale proportionally. Currently
    vol_proxy=1.0 keeps behaviour stable until Stage 2.
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator,
                 agent_id: int):
        super().__init__(params, rng, agent_id, "zi")
        self._depth_rates_cache: Optional[np.ndarray] = None

    def _depth_rates(self, n_levels: int = 10) -> np.ndarray:
        if self._depth_rates_cache is None:
            i = np.arange(1, n_levels + 1, dtype=float)
            self._depth_rates_cache = (
                self.params.lambda_lo * self.params.depth_k * i ** (-self.params.depth_alpha)
            )
        return self._depth_rates_cache

    def submit_orders(self, lob: LOB, mid_price: float,
                      fundamental: float, momentum: float,
                      vol_proxy: float = 1.0) -> None:
        p = self.params
        rng = self.rng

        # Limit orders: Poisson rate per level, power-law decay with depth
        depth_rates = self._depth_rates()
        for side in (1, -1):
            for i, rate in enumerate(depth_rates, start=1):
                if rng.random() < rate:
                    qty = max(1, round(int(rng.integers(p.zi_qty_min, p.zi_qty_max + 1)) * vol_proxy))
                    if side == 1:
                        ref = lob.best_ask if not np.isnan(lob.best_ask) else mid_price
                        price = ref - i * p.tick_size
                    else:
                        ref = lob.best_bid if not np.isnan(lob.best_bid) else mid_price
                        price = ref + i * p.tick_size
                    lob.add_limit(self.agent_id, side, price, qty)

        # Market orders
        for side in (1, -1):
            if rng.random() < p.mu_mo:
                qty = max(1, round(int(rng.integers(p.zi_qty_min, p.zi_qty_max + 1)) * vol_proxy))
                lob.add_market(self.agent_id, side, qty)
