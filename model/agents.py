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

Balance sheet per agent
-----------------------
  equity        = initial_wealth + cumulative_pnl
  margin_posted = collateral locked at CCP (IM + VM calls)
  margin_called : bool flag set each step when a margin shortfall exists
  defaulted     : bool flag -- equity < 0, position force-closed by CCP

All balance-sheet state is updated by Simulation.step(), not here,
so agents remain stateless w.r.t. market clearing (clean separation).
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
    def __init__(self, params: ModelParams, rng: np.random.Generator,
                 agent_id: int = 0, agent_type: str = "base"):
        self.params      = params
        self.rng         = rng
        self.state       = TraderState()
        self.agent_id    = agent_id
        self.agent_type  = agent_type    # "fundamental" | "momentum" | "noise"

        # ── Position accounting ───────────────────────────────────────────
        self.inventory    = 0.0          # net position (log-increment units)
        self.cash         = params.initial_wealth  # starts fully in cash
        self.entry_price  = params.v0   # VWAP entry price (updated on trades)
        self.realised_pnl = 0.0         # closed P&L
        self.mtm_pnl      = 0.0         # open mark-to-market P&L

        # ── Balance sheet ─────────────────────────────────────────────────
        self.initial_wealth  = params.initial_wealth
        self.equity          = params.initial_wealth   # cash + unrealised PnL
        self.im_rate         = params.im_rate          # initial margin rate
        self.margin_posted   = 0.0      # collateral currently posted at CCP
        self.margin_called   = False    # received a margin call this step?
        self.defaulted       = False    # equity < 0 => CCP closed position

    def observe(self, market_state: MarketState) -> None:
        raise NotImplementedError

    def decide(self, market_state: MarketState) -> Tuple[int, float]:
        raise NotImplementedError


class FundamentalTrader(BaseTrader):
    """
    Fundamental-value trader: mean-reversion towards v_t with
    linear and cubic demand in mispricing (kappa, kappa_3), only linear for now.
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator,
                 agent_id: int = 0):
        super().__init__(params, rng, agent_id=agent_id,
                         agent_type="fundamental")
        self.log_mispricing = 0.0
        self.mispricing     = 0.0

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
    LOG MODE:  dp = beta * tanh(momentum)  (direct KF momentum term)
    VOL MODE:  same formula treated as order volume (legacy)
    """

    def __init__(self, params: ModelParams, rng: np.random.Generator,
                 agent_id: int = 0):
        super().__init__(params, rng, agent_id=agent_id,
                         agent_type="momentum")
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
