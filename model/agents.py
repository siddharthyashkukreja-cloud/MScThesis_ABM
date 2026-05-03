# mscthesis_abm/model/agents.py
"""
Agent definitions for the Extended Chiarella-Iori ABM.

All agents return (side: int, magnitude: float) regardless of mode.

  LOG-PRICE MODE (params.log_price_mode=True):
    magnitude = absolute log-price increment this agent contributes.
    FundamentalTrader: gamma * |log(v/p)|
    MomentumTrader:    beta  * tanh(|m|)
    NoiseTrader:       |sigma_n * N(0,1)|

  VOLUME MODE (params.log_price_mode=False):
    magnitude = order volume (units of stock, legacy behaviour).

Inventory accounting in simulation.py uses  inventory += side * magnitude
in both modes.  In log mode, magnitude acts as a stylised position size
proportional to price impact, sufficient for margin/CCP accounting.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from model.market import MarketState
from model.globals import ModelParams


@dataclass
class TraderState:
    pass


class BaseTrader:
    def __init__(self, params: ModelParams, rng: np.random.Generator):
        self.params      = params
        self.rng         = rng
        self.state       = TraderState()
        self.inventory   = 0.0
        self.cash        = 0.0
        self.entry_price = params.v0
        self.mtm_pnl     = 0.0

    def observe(self, market_state: MarketState) -> None:
        raise NotImplementedError

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        raise NotImplementedError


class FundamentalTrader(BaseTrader):
    """
    LOG MODE:  dp = gamma * log(v/p)  (direct KF mean-reversion term)
    VOL MODE:  Poisson volume proportional to |v - p| (legacy)
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        super().__init__(params, rng)
        self.log_mispricing = 0.0
        self.mispricing     = 0.0

    def observe(self, market_state: MarketState) -> None:
        p = max(market_state.price, 1e-10)
        v = max(market_state.fundamental, 1e-10)
        self.log_mispricing = np.log(v) - np.log(p)
        self.mispricing     = v - p

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        if self.params.log_price_mode:
            x = self.log_mispricing
            if abs(x) <= self.params.delta:
                return 0, 0.0
            mag  = self.params.gamma * abs(x)
            side = +1 if x > 0 else -1
            return side, mag
        else:
            x = self.mispricing
            if abs(x) <= self.params.delta:
                return 0, 0.0
            lam    = self.params.kappa * abs(x)
            volume = float(self.rng.poisson(lam))
            if volume == 0:
                return 0, 0.0
            side = +1 if x > 0 else -1
            return side, volume


class MomentumTrader(BaseTrader):
    """
    LOG MODE:  dp = beta * tanh(momentum)  (direct KF momentum term)
    VOL MODE:  same formula treated as order volume (legacy)
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        super().__init__(params, rng)
        self.momentum = 0.0

    def observe(self, market_state: MarketState) -> None:
        self.momentum = market_state.momentum

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        m = self.momentum
        if m == 0.0:
            return 0, 0.0
        mag = self.params.beta * np.tanh(abs(m))
        if mag <= 0.0:
            return 0, 0.0
        side = +1 if m > 0 else -1
        return side, mag


class NoiseTrader(BaseTrader):
    """
    LOG MODE:  dp ~ N(0, sigma_n)  (direct KF noise term)
    VOL MODE:  discrete volume from geometric/Poisson/fixed (legacy)
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        super().__init__(params, rng)

    def observe(self, market_state: MarketState) -> None:
        pass

    def _draw_size_legacy(self) -> float:
        dist = self.params.noise_size_dist
        p    = self.params.noise_size_param
        if dist == "geometric":
            n = self.rng.geometric(p)
        elif dist == "poisson":
            n = max(self.rng.poisson(p), 1)
        else:
            n = 1
        return float(n) * self.params.sigma_n

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        if self.params.log_price_mode:
            if self.rng.random() > self.params.p_noise:
                return 0, 0.0
            shock = self.rng.normal() * self.params.sigma_n
            if shock == 0.0:
                return 0, 0.0
            return (+1 if shock > 0 else -1), abs(shock)
        else:
            if self.rng.random() > self.params.p_noise:
                return 0, 0.0
            volume = self._draw_size_legacy()
            if volume <= 0.0:
                return 0, 0.0
            side = +1 if self.rng.random() < 0.5 else -1
            return side, volume
