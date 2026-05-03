from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import numpy as np

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
        self.excess_demand      = 0.0
        self.cum_abs_demand     = 0.0
        self.cum_excess_demand  = 0.0

        self.vol_ewma  = params.sigma_v ** 2
        self.vol_alpha = 0.05

        # Demand stats
        self.excess_demand = 0.0
        self.cum_abs_demand = 0.0

    def current_vol(self) -> float:
        return max(self.vol_ewma ** 0.5, 1e-8)

    def get_state(self) -> MarketState:
        return MarketState(
            price=self.price,
            fundamental=self.globals.v,
            momentum=self.momentum,
            vol=self.current_vol(),
            t=self.globals.t,
        )

    # ------------------------------------------------------------------
    # LOG-PRICE MODE
    # ------------------------------------------------------------------

    def step_log(self, increments: List[Tuple[int, float]]) -> Dict[str, float]:
        """
        Log-price clearing.
        increments: list of (side, magnitude), side in {-1, 0, +1}, magnitude >= 0.
        Net log-price change = sum of side*magnitude across all agents.
        """
        net_log_dp = 0.0
        abs_flow   = 0.0
        buy_flow   = 0.0
        sell_flow  = 0.0

        for side, mag in increments:
            if mag <= 0.0:
                continue
            signed      = side * mag
            net_log_dp += signed
            abs_flow   += mag
            if side > 0:
                buy_flow  += mag
            elif side < 0:
                sell_flow += mag

        self.last_price  = self.price
        self.log_price  += net_log_dp
        self.price       = np.exp(self.log_price)

        # excess_demand in log-increment units (usable by CCP stress layer)
        self.excess_demand      = net_log_dp
        self.cum_abs_demand    += abs_flow
        self.cum_excess_demand += net_log_dp

        # Momentum: EWMA of log-returns
        a = self.params.alpha
        self.momentum = a * net_log_dp + (1.0 - a) * self.momentum

        # Volatility EWMA
        self.vol_ewma = (
            self.vol_alpha * net_log_dp ** 2
            + (1.0 - self.vol_alpha) * self.vol_ewma
        )

        return {
            "demand"        : buy_flow,
            "supply"        : sell_flow,
            "excess_demand" : net_log_dp,
            "price"         : self.price,
            "momentum"      : self.momentum,
            "vol"           : self.current_vol(),
        }

    # ------------------------------------------------------------------
    # VOLUME MODE  (legacy, unchanged)
    # ------------------------------------------------------------------

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

        self.perm_price = max(0.0, self.perm_price + self.params.lambda_ * excess)
        self.transient  = (
            self.params.rho_tran * self.transient
            + self.params.lambda_tran * excess
        )

        self.last_price = self.price
        self.price      = max(0.0, self.perm_price + self.transient)
        self.log_price  = np.log(max(self.price, 1e-10))

        # Price impact
        self.last_price = self.price
        delta_p = self.params.lambda_ * excess
        self.price = max(0.0, self.price + delta_p)

        # Momentum update (EWMA of price changes)

        dp = self.price - self.last_price
        a = self.params.alpha
        self.momentum = a * dp + (1.0 - a) * self.momentum

        self.vol_ewma = (
            self.vol_alpha * dp ** 2
            + (1.0 - self.vol_alpha) * self.vol_ewma
        )

        return {
            "demand"        : demand,
            "supply"        : supply,
            "excess_demand" : excess,
            "price"         : self.price,
            "momentum"      : self.momentum,
            "vol"           : self.current_vol(),
        }

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

