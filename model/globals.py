from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    # ── Population: direct (non-cleared) market participants ────────────
    n_zi: int               # ZI (noise) traders, direct
    n_fundamental: int      # FT direct (zero in Stage 3 baseline; use BCM)
    n_momentum: int         # MT direct (zero in Stage 3 baseline)

    # ── Market / asset ────────────────────────────────────────────────────
    v0: float               # initial fundamental price
    tick_size: float        # minimum price increment (e.g. 0.01 for SPY)
    dt_minutes: float       # simulation step in real minutes (e.g. 5)

    # ── LOB / order expiry ──────────────────────────────────────────────
    order_ttl: int          # steps before unmatched limit order expires (ODD: 1-10)

    # ── ZI Cont-Stoikov / Farmer parameters ──────────────────────────────
    # Rates are per minute (Cont & Stoikov 2008 continuous-time semantics).
    # ZI.submit_orders scales by dt_minutes: arrivals via Poisson(rate*dt),
    # cancellations via per-resting Bernoulli with prob 1-exp(-delta*dt).
    # Defaults match ODD §Calibration (originally given as Bernoulli prob
    # per 1-min step; numerically identical when dt_minutes=1 and rates are
    # small).
    zi_alpha: float         # limit-arrival rate / minute / ZI    (ODD: 0.15)
    zi_mu: float            # market-arrival rate / minute / ZI   (ODD: 0.025)
    zi_delta: float         # cancel rate / minute / resting-order (ODD: 0.025)

    # ── ZI order sizing (Poisson-arrival agents) ─────────────────────────
    # ZI uses Poisson arrivals scaled by dt_minutes, so order COUNT already
    # tracks cadence. Per-order qty stays at ODD's Discrete[1, 10] so
    # per-minute volume is cadence-invariant.
    zi_qty_min: int         # minimum order quantity (units)
    zi_qty_max: int         # maximum order quantity (ODD: Discrete[1, 10])

    # ── Directional-agent order sizing (FT, MT, BCM-FT, clients) ────────
    # These agents submit at most ONE order per step (deterministic when
    # triggered), so per-minute volume scales as 1/dt_minutes. To preserve
    # ODD's per-minute volume target (calibrated at 1-min cadence with
    # U[1,10] qty), we scale qty by dt_minutes: at dt=5, U[5,50].
    # Used by FT.submit_orders, MT.submit_orders, BankingClearingMember
    # FT-prop submission, BCM fire-sale lots, and all client traders that
    # are FT/MT (not the ZI clients, which keep zi_qty_*).
    dir_qty_min: int = 5         # minimum directional-agent qty
    dir_qty_max: int = 50        # maximum (ODD U[1,10] x dt_minutes=5)

    # ── ZI limit-price placement: exponential (Geometric) depth ──────────
    # k ~ Geometric(zi_offset_p) capped at zi_offset_max ticks. Cont &
    # Stoikov 2008 fit an exponentially-decaying empirical depth profile
    # lambda(i); Geometric(p) is its discrete-tick analog.
    #   default p=0.5 -> P(k=1)=.5, P(k=2)=.25, mean=2 ticks, p99 ~7 ticks
    # Empty-book fallback: anchor on fundamental V instead of best-opposite.
    zi_offset_p: float = 0.5     # Geometric parameter (decay); higher = tighter
    zi_offset_max: int = 20      # cap on Geometric tail (ticks)

    # ── Fundamental trader parameters (Stage 2+) ─────────────────────────
    # ODD §Prediction-style placement, but with V-RELATIVE scaling so the
    # parameter is regime-invariant:
    #   reservation = V * (1 + z * ft_sigma_rel)
    # ft_sigma_rel is the per-agent reservation spread as a FRACTION of V
    # (e.g., 0.005 = 0.5%). z ~ N(0,1) fixed at agent init (ODD Mech #1).
    # Side = sign(reservation - mid); qty ~ Uniform{1, zi_qty_max}.
    ft_sigma_rel: float = 0.005   # fraction of V (= 50 bps default)

    # ── Momentum trader parameters (Stage 2+) ────────────────────────────
    # Direction from EWMA log-return signal (Majewski et al. 2018):
    #   M_t = lambda_ewma * M_{t-1} + (1 - lambda_ewma) * (log P_t - log P_{t-1})
    # Placement uses V-relative offset (Stage 8 calibration upgrade):
    #   price = V * (1 + z * mt_sigma_rel)
    # Direction comes from sign(M_t); placement spread is z*sigma_rel of V.
    mt_sigma_rel: float = 0.005   # fraction of V
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

    # ── CIR stochastic volatility (Gao 2023 / Heston-style) ─────────────
    # Variance v_var follows discrete CIR per step:
    #   v_var_{t+1} = max(0, v_var_t + kappa_v * (theta_v - v_var_t)
    #                          + xi_v * sqrt(v_var_t) * Z_2)
    # Price diffusion uses sigma_t = sqrt(v_var_t) instead of constant
    # sigma_v. xi_v=0 disables CIR and falls back to constant sigma_v
    # (variance stays at sigma_v^2). Initial variance: theta_v if
    # active, else sigma_v^2. Z_2 is independent of the price-diffusion
    # Brownian increment (no leverage effect; rho=0 simplification).
    # Calibration target: kappa_v sets mean-reversion speed of vol bursts;
    # theta_v anchors long-run vol; xi_v controls vol-of-vol amplitude.
    kappa_v: float = 0.0    # CIR mean-reversion speed (per step)
    theta_v: float = 0.0    # CIR long-run variance (per step)
    xi_v: float = 0.0       # vol-of-vol (per step); 0 disables CIR

    # ── Clearing Members (Stage 3+) ──────────────────────────────────────
    # Banking CMs trade on own account in two flavours:
    #   - market_maker (MM): post symmetric bid+ask quotes, provide liquidity
    #   - fundamental (FT-style): trade on V+z*sigma reservation
    # The first n_bcm_mm of n_bcm BCMs run as MMs; remaining are FT-prop.
    # The first n_bcm_with_clients of n_bcm BCMs carry a client book. These
    # two flags are independent: MMs CAN have clients (matches real exchange
    # bank-MM-prime-broker tier where Goldman/JPM both market-make AND clear
    # for clients). Non-Banking CMs are passive; they clear for clients only
    # and never trade. Client book composition is configurable via
    # clients_per_book and client_type_mix.
    n_bcm: int = 0                 # Banking CMs total
    n_bcm_mm: int = 0              # of those, how many run as market makers
    n_bcm_with_clients: int = 0    # of the n_bcm BCMs, how many carry clients
    n_nbcm: int = 0                # Non-Banking CMs (all carry clients)
    clients_per_book: int = 6      # clients per CM-with-clients
    # Client type composition: (n_FT, n_MT, n_ZI) per book; must sum to clients_per_book
    client_book_ft: int = 2
    client_book_mt: int = 2
    client_book_zi: int = 2

    # ── Market-maker quoting: Stoikov 2008 with V-relative scales ────────
    # MM cancels its previous quotes each step and posts fresh symmetric
    # quotes around mid, skewed by inventory. All scales are basis points
    # of mid so they're regime-invariant:
    #   shift  = -mm_inventory_skew_bps * inventory * 1e-4
    #   bid    = mid * (1 + shift - mm_half_spread_bps * 1e-4)
    #   ask    = mid * (1 + shift + mm_half_spread_bps * 1e-4)
    # Quote size is mm_qty per side. Long inventory shifts both quotes
    # down (encourages selling); short shifts up.
    mm_half_spread_bps: float = 30.0       # basis points above/below mid
    mm_qty: int = 50                       # quote size per side (units)
    mm_inventory_skew_bps: float = 0.5     # bps of price skew per unit inventory


@dataclass
class SimContext:
    """Per-step state passed to every trader's submit_orders.

    Carries the fundamental V_t, prev-step post-match mid (NaN before
    first clear), EWMA log-return momentum signal, traders_by_id for
    CMs to look up client inventory, and the current CIR variance state
    v_var (used by ZI vol scaling per Gao 2023 noise-demand spec).
    """
    v: float
    mid_price: float
    momentum: float
    tick: int
    traders_by_id: Optional[dict] = None
    v_var: float = 0.0   # current CIR variance state; 0 means use sigma_v^2 fallback


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
        # Variance state: anchor at theta_v if CIR active, else sigma_v^2
        if params.xi_v > 0.0:
            self.v_var = params.theta_v if params.theta_v > 0.0 else params.sigma_v ** 2
        else:
            self.v_var = params.sigma_v ** 2

    def step(self):
        p = self.params
        # CIR variance update (only if xi_v > 0)
        if p.xi_v > 0.0:
            z2 = self.rng.standard_normal()
            new_var = (self.v_var
                       + p.kappa_v * (p.theta_v - self.v_var)
                       + p.xi_v * float(np.sqrt(max(0.0, self.v_var))) * z2)
            self.v_var = max(0.0, new_var)

        sigma_t = float(np.sqrt(self.v_var))
        diffusion = (p.mu_v - 0.5 * sigma_t ** 2) + sigma_t * self.rng.standard_normal()
        if p.jump_lambda > 0.0:
            n_jumps = int(self.rng.poisson(p.jump_lambda))
            jump = (p.jump_mean * n_jumps
                    + p.jump_std * float(self.rng.standard_normal(n_jumps).sum())
                    if n_jumps > 0 else 0.0)
        else:
            jump = 0.0
        self.v *= float(np.exp(diffusion + jump))
        self.t += 1
