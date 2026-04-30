# mscthesis_abm/agents.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from model.market import MarketState
from model.globals import ModelParams


@dataclass
class TraderState:
    # Placeholder for inventory, cash etc. if you want later
    pass


class BaseTrader:
    """
    Base interface for all traders.
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.state = TraderState()

    def observe(self, market_state: MarketState) -> None:
        raise NotImplementedError

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        """
        Returns (side, volume):
          side   in {+1 (buy), -1 (sell), 0 (wait)}
          volume >= 0
        """
        raise NotImplementedError


class FundamentalTrader(BaseTrader):
    """
    Fundamental-value trader: mean-reversion towards v_t with
    linear and cubic demand in mispricing (kappa, kappa_3), only linear for now.
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        super().__init__(params, rng)
        self.mispricing = 0.0

    def observe(self, market_state: MarketState) -> None:
        # v_t - p_t
        self.mispricing = market_state.fundamental - market_state.price

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        x = self.mispricing
        # Linear demand: q_f = kappa * (v_t - p_t)
        q = self.params.kappa * x

        if q == 0.0:
            return 0, 0.0

        side = +1 if q > 0 else -1
        volume = abs(q)
        return side, volume


class MomentumTrader(BaseTrader):
    """
    Trend-following trader: trades on EWMA price-momentum m_t,
    with demand β * tanh(γ |m_t|).
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

        mag = self.params.beta * np.tanh(self.params.gamma * abs(m))
        if mag <= 0.0:
            return 0, 0.0

        side = +1 if m > 0 else -1
        return side, mag


class NoiseTrader(BaseTrader):
    """
    Zero-intelligence / noise trader.

    - Trades with small participation probability each step.
    - Order size scales with current volatility: σ_n * f(vol_t).

    Here we take vol_t from market_state if available, otherwise
    fall back to the fundamental vol parameter sigma_v.
    """

    def __init__(
        self,
        params: ModelParams,
        rng: np.random.Generator,
        participation_prob: float,
        vol_scale: float = 1.0,
    ):
        super().__init__(params, rng)
        self.participation_prob = participation_prob
        self.vol_scale = vol_scale

    def observe(self, market_state: MarketState) -> None:
        # Noise trader ignores information for direction, but we may
        # use volatility proxy in decide()
        self._last_market_state = market_state

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        # Participation
        if self.rng.random() > self.participation_prob:
            return 0, 0.0

        # Volatility proxy:
        # - if MarketState has attribute 'vol', use it
        # - else fall back to sigma_v from params
        vol = getattr(market_state, "vol", self.params.sigma_v)
        vol = max(vol, 1e-8)  # avoid zero

        # Scale noise order with volatility: σ_n * vol_scale * vol
        base = self.params.sigma_n
        volume = base * self.vol_scale * vol

        if volume <= 0.0:
            return 0, 0.0

        # Random side
        side = +1 if self.rng.random() < 0.5 else -1
        return side, volume