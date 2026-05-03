import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .globals import ModelParams
from .lob import LOB, Order

OrderSide = str  # "buy" or "sell"


@dataclass
class BaseTrader:
    agent_id: int
    cash: float
    inventory: int = 0
    pnl: float = 0.0
    margin_posted: float = 0.0
    defaulted: bool = False

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price

    @property
    def equity(self) -> float:
        return self.cash + self.pnl - self.margin_posted


@dataclass
class ZeroIntelligenceTrader(BaseTrader):
    """
    Zero-intelligence noise trader.

    Order-event rates (alpha, mu, delta) per simulation step follow
    Cont & Stoikov (2008) §3.1; per-resting-order cancellation interpretation
    matches Farmer-Daniels (2003).

    Limit-price placement: depth k ticks from the opposite best quote
    (Cont-Stoikov 2008, depth-from-opposite anchor), with k drawn uniformly
    over {1, ..., zi_offset_max} as a Farmer-Daniels (2003) simplification
    of the empirical depth profile. Empty-book fallback: anchor on
    fundamental. Market orders execute at the prevailing best opposite
    quote inside LOB.match() / LOB.add_market(); price arg unused.
    """
    _queued_order_ids: List[int] = field(default_factory=list)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      fundamental: float, tick: int,
                      rng: np.random.Generator):
        # cancellations: each resting order independently with prob delta
        surviving = []
        for oid in self._queued_order_ids:
            if rng.random() < params.zi_delta:
                lob.cancel(oid)
            else:
                surviving.append(oid)
        self._queued_order_ids = surviving

        side = 1 if rng.random() < 0.5 else -1
        qty = int(rng.integers(params.zi_qty_min, params.zi_qty_max + 1))
        k = int(rng.integers(1, params.zi_offset_max + 1))

        if rng.random() < params.zi_alpha:
            if side == 1:
                anchor = lob.best_ask if not np.isnan(lob.best_ask) else fundamental
                price = anchor - k * params.tick_size
            else:
                anchor = lob.best_bid if not np.isnan(lob.best_bid) else fundamental
                price = anchor + k * params.tick_size
            price = max(price, params.tick_size)
            oid = lob.add_limit(self.agent_id, side, price, qty)
            self._queued_order_ids.append(oid)

        if rng.random() < params.zi_mu:
            lob.add_market(self.agent_id, side, qty)


@dataclass
class FundamentalTrader(BaseTrader):
    """
    Demand proportional to (fundamental - mid_price), scaled by kappa.
    Limit price offset by a fixed private z_score * sigma from fundamental.
    z_score drawn once from N(0,1) at initialisation -- per-agent heterogeneity.
    Parameters kappa and sigma live in ModelParams.ft_kappa / ft_sigma.
    """
    z_score: float = 0.0    # per-agent private valuation offset; set at init

    def submit_orders(self, lob: LOB, params: ModelParams,
                      fundamental: float, mid_price: float, tick: int):
        demand = params.ft_kappa * (fundamental - mid_price)
        if abs(demand) < 1e-6:
            return
        side = 1 if demand > 0 else -1
        limit_price = fundamental + self.z_score * params.ft_sigma
        limit_price = max(limit_price, params.tick_size)
        qty = max(1, int(abs(demand)))
        lob.add_limit(self.agent_id, side, limit_price, qty)


@dataclass
class MomentumTrader(BaseTrader):
    """
    Demand = beta * tanh(momentum), where momentum is the EWMA price trend.
    Limit price leans in the direction of the signal by z_score * sigma.
    z_score drawn once from N(0,1) at initialisation -- per-agent heterogeneity.
    Parameters beta and sigma live in ModelParams.mt_beta / mt_sigma.
    """
    z_score: float = 0.0    # per-agent private valuation offset; set at init

    def submit_orders(self, lob: LOB, params: ModelParams,
                      mid_price: float, momentum: float, tick: int):
        demand = params.mt_beta * np.tanh(momentum)
        if abs(demand) < 1e-6:
            return
        side = 1 if demand > 0 else -1
        limit_price = mid_price + self.z_score * params.mt_sigma * side
        limit_price = max(limit_price, params.tick_size)
        qty = max(1, int(abs(demand)))
        lob.add_limit(self.agent_id, side, limit_price, qty)
