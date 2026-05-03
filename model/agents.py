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
    ZI trader following Vytelingum et al. (2025) eq. (4) for limit price.
    Parameters (alpha, mu, delta) are read from ModelParams -- not stored here.
    Only z_score is per-agent (drawn once at init from N(0,1)).

    Limit price (Vytelingum et al. eq. 4):
        buy  limit = ask * (1 + delta)   [fallback: fundamental * (1 + delta)]
        sell limit = bid * (1 - delta)   [fallback: fundamental * (1 - delta)]
    Market orders are priced aggressively (5% through best quote) to guarantee fill.
    """
    _queued_order_ids: List[int] = field(default_factory=list)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      fundamental: float, tick: int):
        """
        Resolve cancellations, then submit limit and/or market orders
        directly into the LOB.
        """
        # cancellations: each resting order cancelled independently with prob delta
        surviving = []
        for oid in self._queued_order_ids:
            if np.random.random() < params.zi_delta:
                lob.cancel(oid)
            else:
                surviving.append(oid)
        self._queued_order_ids = surviving

        # reference prices -- fall back to fundamental if LOB is empty
        ref_ask = lob.best_ask if not np.isnan(lob.best_ask) else fundamental
        ref_bid = lob.best_bid if not np.isnan(lob.best_bid) else fundamental

        side = 1 if np.random.random() < 0.5 else -1
        qty = np.random.randint(params.zi_qty_min, params.zi_qty_max + 1)

        if np.random.random() < params.zi_alpha:
            price = ref_ask * (1 + params.zi_delta) if side == 1 else ref_bid * (1 - params.zi_delta)
            price = max(price, params.tick_size)
            oid = lob.add_limit(self.agent_id, side, price, qty)
            self._queued_order_ids.append(oid)

        if np.random.random() < params.zi_mu:
            price = ref_ask * 1.05 if side == 1 else ref_bid * 0.95
            price = max(price, params.tick_size)
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
