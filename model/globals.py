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
    # Calibrated from daily BuyVol / SellVol imbalance data.
    # lambda_lo: limit order arrival rate (per side, per step)
    # mu_mo:     market order arrival rate (per side, per step)
    # delta_co:  cancellation rate (per step per resting order)
    # depth_k, depth_alpha: power-law depth: lambda(i) = depth_k * i^(-depth_alpha)
    lambda_lo: float        # limit order rate (best quote distance 1)
    mu_mo: float            # market order arrival rate
    delta_co: float         # cancellation rate per resting order per step
    depth_k: float          # power-law scale for depth
    depth_alpha: float      # power-law exponent for depth

    # ── ZI order sizing ───────────────────────────────────────────────────
    zi_qty_min: int         # minimum order quantity (units)
    zi_qty_max: int         # maximum order quantity (ODD: Discrete[1, 10])

    # ── Fundamental process (GBM; used from Stage 2) ────────────────────────
    mu_v: float = 0.0       # drift per step
    sigma_v: float = 0.01   # vol per step (calibrated later)

    # ── Placeholder slots for Gao volatility extension (Stage 2+) ───────────────
    # These are read by agents once the stochastic vol layer is added.
    kappa_v: float = 0.0    # vol mean-reversion speed
    theta_v: float = 0.0    # long-run variance
    xi_v: float = 0.0       # vol-of-vol


class GlobalState:
    """
    Tracks simulation time and the fundamental value process.
    Fundamental follows GBM; per-step shock is drawn here so every
    agent sees the same v_t without re-drawing.
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
