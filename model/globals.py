from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    # ── Population ───────────────────────────────────────────────────────
    n_zi: int               # ZI (noise) traders
    n_fundamental: int      # to be added Stage 2+
    n_momentum: int         # to be added Stage 2+

    # ── Market / asset ────────────────────────────────────────────────────
    v0: float               # initial fundamental price
    tick_size: float        # minimum price increment (e.g. 0.25 for E-mini)
    dt_minutes: float       # simulation step in real minutes (e.g. 5)

    # ── LOB / order expiry ──────────────────────────────────────────────
    order_ttl: int          # steps before unmatched limit order expires (ODD: 1-10)

    # ── ZI Cont-Stoikov parameters ────────────────────────────────────────────
    # Calibrated from E-mini order-flow data (Vytelingum et al. 2025, s.3.3.1).
    # alpha: probability a ZI trader submits a limit order each tick
    # mu:    probability a ZI trader submits a market order each tick
    # delta: relative depth offset for limit price; also used as per-order cancellation prob
    zi_alpha: float         # limit order submission probability   (ODD default: 0.15)
    zi_mu: float            # market order submission probability  (ODD default: 0.025)
    zi_delta: float         # cancellation prob / depth offset     (ODD default: 0.025)

    # ── ZI order sizing ───────────────────────────────────────────────────
    zi_qty_min: int         # minimum order quantity (units)
    zi_qty_max: int         # maximum order quantity (ODD: Discrete[1, 10])

    # ── Fundamental trader parameters (Stage 2+) ─────────────────────────
    # kappa: demand sensitivity to price-fundamental mispricing
    # sigma: price volatility estimate used to scale private valuation offset
    ft_kappa: float = 0.1   # mean-reversion strength toward fundamental
    ft_sigma: float = 1.0   # volatility scaling for private z-score offset

    # ── Momentum trader parameters (Stage 2+) ────────────────────────────
    # beta:  demand scaling for tanh(momentum) signal
    # sigma: same role as ft_sigma but for momentum agents
    mt_beta: float = 0.05   # momentum demand scale
    mt_sigma: float = 1.0   # volatility scaling for private z-score offset

    # ── Fundamental process (GBM; replaced by data signal Stage 2) ───────
    mu_v: float = 0.0       # drift per step
    sigma_v: float = 0.01   # vol per step (calibrated later)

    # ── Placeholder slots for Gao stochastic vol extension (Stage 2+) ────
    kappa_v: float = 0.0    # vol mean-reversion speed
    theta_v: float = 0.0    # long-run variance
    xi_v: float = 0.0       # vol-of-vol


class GlobalState:
    """
    Tracks simulation time and the fundamental value process.
    Fundamental follows GBM per step; all agents read self.v each tick
    so the same fundamental is shared without re-drawing.
    """

    def __init__(self, params: ModelParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.v = params.v0
        self.t = 0

    def step(self):
        shock = self.params.mu_v + self.params.sigma_v * self.rng.standard_normal()
        self.v *= np.exp(shock)
        self.t += 1
