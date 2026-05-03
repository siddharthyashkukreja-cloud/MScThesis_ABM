# agents.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

OrderSide = str  # "buy" or "sell"

@dataclass
class Order:
    side: OrderSide
    price: float
    quantity: int
    agent_id: int
    expiry: int  # tick at which order expires

@dataclass
class BaseTrader:
    agent_id: int
    cash: float
    inventory: int = 0
    pnl: float = 0.0
    margin_posted: float = 0.0
    im_rate: float = 0.10
    defaulted: bool = False

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price

    @property
    def equity(self) -> float:
        return self.cash + self.pnl - self.margin_posted


@dataclass
class FundamentalTrader(BaseTrader):
    kappa: float = 0.1       # demand sensitivity to mispricing
    z_score: float = 0.0     # private valuation offset (drawn at init)
    sigma: float = 1.0       # price volatility estimate

    def orders(self, mid_price: float, fundamental: float, tick: int) -> List[Order]:
        demand = self.kappa * (fundamental - mid_price)
        if abs(demand) < 1e-6:
            return []
        side = "buy" if demand > 0 else "sell"
        # limit price offset by private z_score * sigma from fundamental
        limit_price = fundamental + self.z_score * self.sigma
        limit_price = max(limit_price, 1e-4)
        qty = max(1, int(abs(demand)))
        return [Order(side=side, price=limit_price, quantity=qty,
                      agent_id=self.agent_id, expiry=tick + 10)]


@dataclass
class MomentumTrader(BaseTrader):
    beta: float = 0.05
    z_score: float = 0.0
    sigma: float = 1.0

    def orders(self, mid_price: float, fundamental: float,
               momentum: float, tick: int) -> List[Order]:
        demand = self.beta * np.tanh(momentum)
        if abs(demand) < 1e-6:
            return []
        side = "buy" if demand > 0 else "sell"
        limit_price = mid_price + self.z_score * self.sigma * np.sign(demand)
        limit_price = max(limit_price, 1e-4)
        qty = max(1, int(abs(demand)))
        return [Order(side=side, price=limit_price, quantity=qty,
                      agent_id=self.agent_id, expiry=tick + 10)]


@dataclass
class NoiseTrader(BaseTrader):
    """
    ZI trader following Vytelingum et al. (2025) eq. (4) for limit price,
    parameterised by Cont-Stoikov alpha/mu/delta (ODD Section Calibration).

    alpha: probability of submitting a limit order each tick
    mu:    probability of submitting a market order each tick
    delta: relative depth offset for limit price AND cancellation rate proxy
    """
    alpha: float = 0.15
    mu: float = 0.025
    delta: float = 0.025
    _queued_order_ids: List[int] = field(default_factory=list)

    def orders(self, best_bid: Optional[float], best_ask: Optional[float],
               fundamental: float, tick: int) -> Tuple[List[Order], List[int]]:
        """
        Returns (new_orders, cancel_ids).
        Uses best_bid/best_ask when available; falls back to fundamental.
        Equation (4) from Vytelingum et al.:
            buy  limit price = ask * (1 + delta)   [or fundamental * (1+delta)]
            sell limit price = bid * (1 - delta)   [or fundamental * (1-delta)]
        """
        new_orders: List[Order] = []
        cancel_ids: List[int] = []

        # cancellations: each queued order cancelled independently with prob delta
        surviving = []
        for oid in self._queued_order_ids:
            if np.random.random() < self.delta:
                cancel_ids.append(oid)
            else:
                surviving.append(oid)
        self._queued_order_ids = surviving

        # reference prices — fall back to fundamental if LOB empty
        ref_ask = best_ask if best_ask is not None else fundamental
        ref_bid = best_bid if best_bid is not None else fundamental

        side = "buy" if np.random.random() < 0.5 else "sell"
        qty = np.random.randint(1, 11)

        # limit order submission
        if np.random.random() < self.alpha:
            if side == "buy":
                price = ref_ask * (1 + self.delta)
            else:
                price = ref_bid * (1 - self.delta)
            price = max(price, 1e-4)
            order = Order(side=side, price=price, quantity=qty,
                          agent_id=self.agent_id, expiry=tick + 10)
            new_orders.append(order)
            self._queued_order_ids.append(id(order))

        # market order submission (separate draw, can co-occur)
        if np.random.random() < self.mu:
            # market orders: priced aggressively to guarantee fill
            price = ref_ask * 1.05 if side == "buy" else ref_bid * 0.95
            price = max(price, 1e-4)
            new_orders.append(Order(side=side, price=price, quantity=qty,
                                    agent_id=self.agent_id, expiry=tick + 1))

        return new_orders, cancel_ids