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
    Centralised market/auctioneer.

    Price impact is split into:
      - permanent component  (lambda_)     : persistent shift in fair value
      - transitory component (lambda_tran) : short-lived microstructure impact,
                                             mean-reverts at rate rho_tran per step.
    This mimics the depth-of-book effect of a LOB without explicitly maintaining
    one, and is a standard decomposition in the empirical market-impact literature
    (e.g. Almgren-Chriss, Gatheral).
    """

    def __init__(self, params: ModelParams, globals_: GlobalState):
        self.params  = params
        self.globals = globals_

        self.perm_price = params.v0 - params.price_distortion
        self.transient  = 0.0
        self.price      = self.perm_price
        self.last_price = self.price
        self.momentum   = params.m0

        # Demand accumulators
        self.excess_demand      = 0.0
        self.cum_abs_demand     = 0.0
        self.cum_excess_demand  = 0.0

        # Volatility EWMA
        self.vol_ewma  = params.sigma_v ** 2   # warm-start instead of zero
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

        # --- Permanent impact ---
        self.perm_price = max(0.0, self.perm_price + self.params.lambda_ * excess)

        # --- Transitory impact (mean-reverts each step) ---
        self.transient = (
            self.params.rho_tran * self.transient        # decay
            + self.params.lambda_tran * excess           # new impulse
        )

        # --- Observed price = permanent + transient ---
        self.last_price = self.price
        self.price      = max(0.0, self.perm_price + self.transient)

        # --- Momentum update (EWMA of price changes) ---
        dp = self.price - self.last_price
        a  = self.params.alpha
        self.momentum = a * dp + (1.0 - a) * self.momentum

        # --- Volatility update (EWMA of squared returns) ---
        self.vol_ewma = (
            self.vol_alpha * dp ** 2
            + (1.0 - self.vol_alpha) * self.vol_ewma
        )

        return {
            "demand"       : demand,
            "supply"       : supply,
            "excess_demand": excess,
            "price"        : self.price,
            "momentum"     : self.momentum,
            "vol"          : self.current_vol(),
        }