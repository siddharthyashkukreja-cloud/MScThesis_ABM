from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    # ── Population: direct (non-cleared) market participants ────────────
    n_zi: int               # ZI (noise) traders, direct
    n_fundamental: int      # FT direct (zero in real-bank-tier setup)
    n_momentum: int         # MT direct (zero in real-bank-tier setup)

    # ── Market / asset ────────────────────────────────────────────────────
    v0: float               # initial fundamental price
    tick_size: float        # minimum price increment (e.g. 0.01 for SPY)
    dt_minutes: float       # simulation step in real minutes (e.g. 5)

    # ── LOB / order expiry ──────────────────────────────────────────────
    # order_ttl=1 means limit orders DO NOT REST: they participate in the
    # current step's match, unfilled portion expires next step. Avoids the
    # complexity of multi-step cancellation logic and matches the user's
    # design intent (MM is the sole liquidity provider; other agents
    # submit fresh limits each step that get matched or expire).
    order_ttl: int          # default 1 in run_simulation.py

    # ── ZI Cont-Stoikov / Farmer parameters (Bernoulli per step) ─────────
    # Per-step Bernoulli probabilities (ODD §Calibration interpretation).
    # alpha: prob each ZI submits a limit order this step
    # mu:    prob each ZI submits a market order this step
    # No cancellation rate — with order_ttl=1, orders expire automatically.
    zi_alpha: float         # limit submission prob per step    (ODD: 0.15)
    zi_mu: float            # market submission prob per step   (ODD: 0.025)

    # ── ZI order sizing (Bernoulli, ODD-faithful per-step) ───────────────
    zi_qty_min: int         # ODD §Stochasticity: U[1, 10]
    zi_qty_max: int

    # ── Directional-agent order sizing (FT, MT, BCM-FT, FT/MT clients) ──
    # Deterministic 1-order-per-step submitters. To preserve ODD's
    # per-minute volume target (calibrated at 1-min), qty scales by
    # dt_minutes: at dt=5, U[5, 50].
    dir_qty_min: int = 5
    dir_qty_max: int = 50

    # ── Momentum signal (Majewski 2018 EWMA decay) ──────────────────────
    mt_lambda_ewma: float = 0.95   # EWMA decay (per step)
    mt_threshold: float = 1e-4     # min |M_t| for MT to act

    # ── Fundamental V process: Heston (correlated) jump-diffusion ───────
    # log V step:  d log V = (mu_v - nu/2) dt + sqrt(nu) dW1 + jumps
    # Variance step (CIR): d nu = kappa(theta - nu) dt + xi sqrt(nu) dW2
    # corr(dW1, dW2) = rho  (negative for equity index leverage effect)
    #
    # ρ implementation: Z2 = rho*Z1 + sqrt(1-rho²)*Z_indep
    # where Z1 drives diffusion and Z2 drives variance.
    #
    # Calibration:
    #   mu_v, sigma_v (= sqrt(theta_v)): direct from daily WRDS aggregates
    #   theta_v: per-regime, set from sigma_v^2
    #   kappa_v, xi_v, rho_v: estimate from 5-min realised-vol time series
    #   jump_*: Lee-Mykland 2008 filter on daily returns
    mu_v: float = 0.0
    sigma_v: float = 0.001     # per-step diffusion (CIR-off fallback)
    kappa_v: float = 0.0       # CIR mean-reversion speed (per step)
    theta_v: float = 0.0       # CIR long-run variance (per step)
    xi_v: float = 0.0          # CIR vol-of-vol (per step); CIR active iff > 0
    rho_v: float = -0.7        # leverage correlation (Heston typical)
    jump_lambda: float = 0.0385  # Poisson rate of jumps per step (ODD: 3/78)
    jump_mean: float = 0.0       # jump mean (log-V units)
    jump_std: float = 0.01       # jump std (ODD §Stochasticity)

    # ── Clearing Members (Stage 3+) ──────────────────────────────────────
    # Banking CMs trade on own account in two modes:
    #   - market_maker (MM): HFABM ladder of bids+asks at fixed tick offsets
    #   - fundamental: V·(1 + z·sqrt(nu_t)) reservation, market or limit submission
    # The first n_bcm_mm of n_bcm BCMs run as MMs; remaining are FT-prop.
    # The first n_bcm_with_clients of n_bcm BCMs carry a client book.
    # NBCMs never trade on own account; they clear for clients only.
    n_bcm: int = 0
    n_bcm_mm: int = 0
    n_bcm_with_clients: int = 0
    n_nbcm: int = 0
    clients_per_book: int = 6
    client_book_ft: int = 2
    client_book_mt: int = 2
    client_book_zi: int = 2

    # ── Market-maker ladder (HFABM Cao 2024 §3.2) ───────────────────────
    # Each step, MM cancels prior quotes and posts:
    #   bid limits at  mid - k * tick  for k = 1..mm_n_levels
    #   ask limits at  mid + k * tick  for k = 1..mm_n_levels
    # Each level has size mm_qty. mm_n_levels=4 follows HFABM E-mini.
    # Inventory limit: when |inv| > mm_inventory_limit, MM submits market
    # orders to liquidate (independent of cap-ratio Basel breach).
    mm_n_levels: int = 4              # HFABM E-mini default
    mm_qty: int = 50                  # quote size per side per level
    mm_inventory_limit: int = 1000    # |inv| trigger for forced liquidation


@dataclass
class SimContext:
    """Per-step state passed to every trader's submit_orders."""
    v: float
    mid_price: float
    momentum: float
    tick: int
    traders_by_id: Optional[dict] = None
    v_var: float = 0.0   # current CIR variance state (also = sigma_v^2 if CIR off)


class GlobalState:
    """
    Heston correlated jump-diffusion fundamental V_t.

    Per step:
      1. Draw correlated Brownians via Z2 = rho * Z1 + sqrt(1 - rho^2) * Z_indep
      2. CIR variance update (only if xi_v > 0)
      3. Diffusion: d log V = (mu - nu/2) + sqrt(nu) * Z1 + jumps
    """

    def __init__(self, params: ModelParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.v = params.v0
        self.t = 0
        if params.xi_v > 0.0:
            self.v_var = params.theta_v if params.theta_v > 0.0 else params.sigma_v ** 2
        else:
            self.v_var = params.sigma_v ** 2

    def step(self):
        p = self.params
        z1 = self.rng.standard_normal()
        # Correlated Brownians (Heston leverage)
        if p.xi_v > 0.0:
            z_indep = self.rng.standard_normal()
            z2 = p.rho_v * z1 + float(np.sqrt(max(0.0, 1.0 - p.rho_v ** 2))) * z_indep
            new_var = (self.v_var
                       + p.kappa_v * (p.theta_v - self.v_var)
                       + p.xi_v * float(np.sqrt(max(0.0, self.v_var))) * z2)
            self.v_var = max(0.0, new_var)

        sigma_t = float(np.sqrt(self.v_var))
        diffusion = (p.mu_v - 0.5 * sigma_t ** 2) + sigma_t * z1
        if p.jump_lambda > 0.0:
            n_jumps = int(self.rng.poisson(p.jump_lambda))
            jump = (p.jump_mean * n_jumps
                    + p.jump_std * float(self.rng.standard_normal(n_jumps).sum())
                    if n_jumps > 0 else 0.0)
        else:
            jump = 0.0
        self.v *= float(np.exp(diffusion + jump))
        self.t += 1
