from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    # ── Population ───────────────────────────────────────────────────────
    n_zi: int               # ZI (noise) traders
    n_fundamental: int      # FT (Stage 2+, Chiarella fundamental traders)
    n_momentum: int         # MT (Stage 2+, Chiarella momentum traders)

    # ── Market / asset ────────────────────────────────────────────────────
    v0: float               # initial fundamental price
    tick_size: float        # minimum price increment (e.g. 0.01 for SPY)
    dt_minutes: float       # simulation step in real minutes (e.g. 5)

    # ── LOB / order expiry ──────────────────────────────────────────────
    order_ttl: int          # steps before unmatched limit order expires (ODD: 1-10)

    # ── ZI Cont-Stoikov / Farmer parameters ──────────────────────────────
    # Order-event rates per simulation step (Cont & Stoikov 2008 §3.1 style;
    # ODD §Calibration values; Farmer-Daniels 2003 ZI placement geometry).
    # alpha: probability each ZI submits a limit order this step
    # mu:    probability each ZI submits a market order this step
    # delta: per-resting-order cancellation probability this step
    zi_alpha: float         # limit submission probability         (ODD default: 0.15)
    zi_mu: float            # market submission probability        (ODD default: 0.025)
    zi_delta: float         # per-order cancellation probability   (ODD default: 0.025)

    # ── ZI / FT / MT order sizing ────────────────────────────────────────
    # ODD §Stochasticity: order quantity ~ Discrete[1, 10] each step;
    # shared by all submitting agents (ZI, FT, MT) for simplicity.
    zi_qty_min: int         # minimum order quantity (units)
    zi_qty_max: int         # maximum order quantity (ODD: Discrete[1, 10])

    # ── ZI limit-price placement ─────────────────────────────────────────
    # Buy limits placed k ticks below best ask; sell limits k ticks above
    # best bid. Cont & Stoikov 2008 use an empirical depth profile; we use
    # a Farmer-Daniels uniform draw k ~ U{1, zi_offset_max} as a baseline
    # simplification, to be replaced by an empirically fitted profile when
    # tick-level order-flow data is available. Empty-book fallback: anchor
    # on fundamental.
    zi_offset_max: int = 5  # max placement depth from opposite best (ticks)

    # ── Fundamental trader parameters (Stage 2+) ─────────────────────────
    # ODD §Prediction: limit_price = V + z * sigma_F, with z ~ N(0,1) drawn
    # once per agent at init (heterogeneous private valuation; ODD Mech #1).
    # Side = sign(reservation - mid_price); qty ~ Uniform{1, zi_qty_max}.
    ft_sigma: float = 0.5   # std-dev (price units) of private valuation offset

    # ── Momentum trader parameters (Stage 2+) ────────────────────────────
    # Direction from EWMA log-return signal (Majewski et al. 2018):
    #   M_t = lambda_ewma * M_{t-1} + (1 - lambda_ewma) * (log P_t - log P_{t-1})
    # Placement uses ODD §Prediction scheme with V anchor + private z*sigma_M
    # offset. This grafts ODD's heterogeneity formula onto a Chiarella-style
    # direction signal -- explicit Stage 2 extension (ODD has no MT).
    mt_sigma: float = 0.5         # std-dev of MT private valuation offset
    mt_lambda_ewma: float = 0.95  # EWMA decay (Majewski 2018; tune at Stage 8)
    mt_threshold: float = 1e-4    # min |M_t| to act (avoid zero-crossing noise)

    # ── Fundamental process: Merton (1976) jump-diffusion ────────────────
    # log V_{t+1} = log V_t + (mu_v - sigma_v^2/2) + sigma_v * eps + Sum J_i
    # where Sum J_i is over n_jumps ~ Poisson(jump_lambda), each J ~ N(jump_mean, jump_std).
    # Defaults are SPY-typical per-step values (annualised ~9% vol, no drift)
    # and ODD §Stochasticity jump magnitudes; jump_lambda=0 in Stage 2 baseline,
    # turn on for stress runs (ODD: lambda=3 per 510-tick day = 0.038 per 5-min).
    mu_v: float = 0.0
    sigma_v: float = 0.001     # ~9% annualised at 5-min cadence (252 * 78 steps/yr)
    jump_lambda: float = 0.0   # Poisson rate per step (ODD stressed: ~0.038)
    jump_mean: float = 0.0     # ODD §Stochasticity
    jump_std: float = 0.01     # ODD §Stochasticity

    # ── Placeholder slots for Gao (2023) stochastic-vol extension ────────
    kappa_v: float = 0.0    # CIR mean-reversion speed
    theta_v: float = 0.0    # CIR long-run variance
    xi_v: float = 0.0       # vol-of-vol


class GlobalState:
    """
    Simulation clock + Merton (1976) jump-diffusion fundamental value V_t.

    All agents read self.v each tick; momentum signal lives in Simulation
    so it can persist across steps and be exposed to MTs.
    """

    def __init__(self, params: ModelParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.v = params.v0
        self.t = 0

    def step(self):
        p = self.params
        diffusion = (p.mu_v - 0.5 * p.sigma_v ** 2) + p.sigma_v * self.rng.standard_normal()
        if p.jump_lambda > 0.0:
            n_jumps = int(self.rng.poisson(p.jump_lambda))
            jump = (p.jump_mean * n_jumps
                    + p.jump_std * float(self.rng.standard_normal(n_jumps).sum())
                    if n_jumps > 0 else 0.0)
        else:
            jump = 0.0
        self.v *= float(np.exp(diffusion + jump))
        self.t += 1
