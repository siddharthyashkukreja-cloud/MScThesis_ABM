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
        self.inventory   = 0.0
        self.cash        = 0.0
        self.entry_price = params.v0   # or p0 — reference price for MtM
        self.mtm_pnl     = 0.0

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
    Fundamental-value trader with:
      - Dead-band threshold δ: only trade if |v_t - p_t| > delta
      - Discrete Poisson volume: n ~ Poisson(kappa * |v_t - p_t|)
        so order size is a random integer, mean proportional to mispricing,
        but near-zero mispricings produce near-zero *expected* orders AND
        are suppressed entirely below the threshold.
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator):
        super().__init__(params, rng)
        self.mispricing = 0.0

    def observe(self, market_state: MarketState) -> None:
        self.mispricing = market_state.fundamental - market_state.price

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        x = self.mispricing

        # --- Dead band: do nothing if mispricing is small ---
        if abs(x) <= self.params.delta:
            return 0, 0.0

        # --- Discrete volume draw: Poisson with mean kappa * |mispricing| ---
        lam = self.params.kappa * abs(x)
        volume = float(self.rng.poisson(lam))

        if volume == 0:
            return 0, 0.0

        side = +1 if x > 0 else -1
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

    - Participation probability p_noise per step (from ModelParams, 
      not 1/N so participation is independent of pool size).
    - Order size drawn from a discrete distribution (geometric, Poisson, or fixed)
      scaled by sigma_n. Discrete sizes give a heavier-tailed volume distribution
      even before adding an LOB.
    - Side is iid uniform ±1.
    """

    def __init__(
        self,
        params: ModelParams,
        rng: np.random.Generator,
    ):
        super().__init__(params, rng)

    def observe(self, market_state: MarketState) -> None:
        pass  # noise trader ignores all signals

    def _draw_size(self) -> float:
        dist = self.params.noise_size_dist
        p    = self.params.noise_size_param

        if dist == "geometric":
            # geometric(p): mean = 1/p; heavy right tail
            n = self.rng.geometric(p)
        elif dist == "poisson":
            # poisson(lam): mean = lam
            n = max(self.rng.poisson(p), 1)
        else:
            # fixed unit size
            n = 1

        return float(n) * self.params.sigma_n

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        # Participation gate
        if self.rng.random() > self.params.p_noise:
            return 0, 0.0

        volume = self._draw_size()
        if volume <= 0.0:
            return 0, 0.0

        side = +1 if self.rng.random() < 0.5 else -1
        return side, volume