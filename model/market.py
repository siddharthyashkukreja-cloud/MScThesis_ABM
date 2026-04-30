from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

from .globals import GlobalState, ModelParams


@dataclass
class MarketState:
    price: float
    fundamental: float
    momentum: float
    vol: float
    t: int


class Market:
    """
    Centralised market/auctioneer implementing linear price impact,
    EWMA momentum, and EWMA volatility.
    """

    def __init__(self, params: ModelParams, globals_: GlobalState):
        self.params = params
        self.globals = globals_

        self.price = params.v0 - params.price_distortion
        self.momentum = params.m0
        self.last_price = self.price

        # Demand stats
        self.excess_demand = 0.0
        self.cum_abs_demand = 0.0
        self.cum_excess_demand = 0.0

        # Volatility EWMA
        self.vol_ewma = 0.0
        self.vol_alpha = 0.05  # EWMA weight

    def current_vol(self) -> float:
        # sqrt of EWMA variance; small floor to avoid zero
        return max(self.vol_ewma ** 0.5, 1e-8)

    def get_state(self) -> MarketState:
        return MarketState(
            price=self.price,
            fundamental=self.globals.v,
            momentum=self.momentum,
            vol=self.current_vol(),
            t=self.globals.t,
        )

    def step(self, orders: Iterable[Tuple[int, float]]) -> Dict[str, float]:
        """
        Perform one market clearing step.

        orders: iterable of (side, volume)
          side   in {+1 (buy), -1 (sell)}
          volume >= 0
        """
        demand = 0.0
        supply = 0.0

        for side, volume in orders:
            if volume <= 0.0:
                continue
            if side > 0:
                demand += volume
            elif side < 0:
                supply += volume

        excess = demand - supply
        self.excess_demand = excess
        self.cum_abs_demand += demand + supply
        self.cum_excess_demand += excess

        # Price impact
        self.last_price = self.price
        delta_p = self.params.lambda_ * excess
        self.price = max(0.0, self.price + delta_p)

        # Momentum update (EWMA of price changes)
        dp = self.price - self.last_price
        a = self.params.alpha
        self.momentum = a * dp + (1.0 - a) * self.momentum

        # Volatility update (EWMA of squared returns)
        self.vol_ewma = self.vol_alpha * (dp ** 2) + (1.0 - self.vol_alpha) * self.vol_ewma

        return {
            "demand": demand,
            "supply": supply,
            "excess_demand": excess,
            "price": self.price,
            "momentum": self.momentum,
            "vol": self.current_vol(),
        }