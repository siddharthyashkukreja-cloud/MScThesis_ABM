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
    Centralised market / auctioneer.

    Two clearing modes controlled by params.log_price_mode:

    LOG-PRICE MODE (True)  -- KF-consistent
        Agents submit (side, magnitude) where magnitude is a
        log-price increment.  Net log-price change:
            log(p_{t+1}) = log(p_t) + sum_i  side_i * magnitude_i
        This directly implements the Majewski/Chiarella-Iori
        KF state equation without a lambda middleman.

    VOLUME MODE (False)  -- legacy, unchanged
        dp = lambda_ * net_order_flow + transitory component.

    CCP / MARGIN COMPATIBILITY
        Both modes expose identical outputs:
        price, momentum, vol, excess_demand.
        Trader inventory/mtm_pnl updated in simulation.py
        via the same (side, magnitude) convention.
    """

    def __init__(self, params: ModelParams, globals_: GlobalState):
        self.params  = params
        self.globals = globals_

        self.log_price  = np.log(max(params.v0 - params.price_distortion, 1e-10))
        self.perm_price = params.v0 - params.price_distortion
        self.transient  = 0.0
        self.price      = np.exp(self.log_price)
        self.last_price = self.price
        self.momentum   = params.m0

        self.excess_demand      = 0.0
        self.cum_abs_demand     = 0.0
        self.cum_excess_demand  = 0.0

        self.vol_ewma  = params.sigma_v ** 2
        self.vol_alpha = 0.05

    def current_vol(self) -> float:
        return max(self.vol_ewma ** 0.5, 1e-8)

    def get_state(self) -> MarketState:
        return MarketState(
            price       = self.price,
            fundamental = self.globals.v,
            momentum    = self.momentum,
            vol         = self.current_vol(),
            t           = self.globals.t,
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
        self.excess_demand      = excess
        self.cum_abs_demand    += demand + supply
        self.cum_excess_demand += excess

        self.perm_price = max(0.0, self.perm_price + self.params.lambda_ * excess)
        self.transient  = (
            self.params.rho_tran * self.transient
            + self.params.lambda_tran * excess
        )

        self.last_price = self.price
        self.price      = max(0.0, self.perm_price + self.transient)
        self.log_price  = np.log(max(self.price, 1e-10))

        dp = self.price - self.last_price
        a  = self.params.alpha
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
