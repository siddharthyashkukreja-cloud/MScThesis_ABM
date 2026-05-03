# mscthesis_abm/model/globals.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    # Population
    n_noise: int
    n_fundamental: int
    n_momentum: int

    # Price impact and fundamental process
    lambda_: float          # permanent price impact (volume mode only)
    lambda_tran: float      # transitory impact (volume mode only)
    rho_tran: float         # transient mean-reversion (volume mode only)
    v0: float               # initial fundamental value (price level)
    m0: float               # initial momentum
    price_distortion: float # initial p0 = v0 - distortion
    mu_v: float             # log-drift of fundamental (geometric GRW)
    sigma_v: float          # log-vol of fundamental

    # Fundamental trader
    kappa: float            # mean-reversion speed (volume mode only)
    delta: float            # dead-band half-width (log units in log mode)

    # Momentum trader
    alpha: float            # EWMA weight for momentum
    beta: float             # momentum impact coefficient
    gamma: float            # fundamental pull coefficient

    # Noise trader
    sigma_n: float          # noise shock std (log units in log mode)
    p_noise: float          # per-step participation probability
    noise_size_dist: str    # "geometric" | "poisson" | "fixed"
    noise_size_param: float # distribution parameter

    # Mode switch
    # True  -> log-price mode: agents output log-price increments (KF-consistent)
    # False -> volume mode:    agents output order volumes (legacy)
    log_price_mode: bool = True

    # ── Balance-sheet / margin parameters ──────────────────────────────────
    # These are per-agent defaults; individual agents can override post-init.
    initial_wealth: float = 1_000.0   # starting equity (same units as price)
    im_rate: float = 0.10             # initial margin as fraction of notional
    #   Justification: 10% IM is the EMIR-mandated floor for equity derivatives
    #   (ESMA RTS 2016/2251).  Reasonable for a stylised CCP model.
    vm_floor: float = 0.0             # variation margin floor (0 = full MtM loss)


class GlobalState:
    """
    Fundamental value process.

    log_price_mode=True  -> geometric random walk:
        v_{t+1} = v_t * exp(mu_v + sigma_v * Z),  Z ~ N(0,1)
    log_price_mode=False -> arithmetic walk (legacy):
        v_{t+1} = v_t + mu_v + sigma_v * Z
    """

    def __init__(self, params: ModelParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.v = params.v0
        self.t = 0

    def step_fundamental(self):
        if self.params.log_price_mode:
            log_shock = self.params.mu_v + self.params.sigma_v * self.rng.normal()
            self.v = self.v * np.exp(log_shock)
        else:
            dv = self.params.mu_v + self.params.sigma_v * self.rng.normal()
            self.v += dv
        self.t += 1
