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
    lambda_: float          # price impact coefficient λ
    v0: float               # initial fundamental value
    m0: float               # initial momentum
    price_distortion: float # initial p0 = v0 - distortion
    mu_v: float             # log-drift of fundamental (geometric GRW)
    sigma_v: float          # log-vol of fundamental

    # Fundamental trader parameters (linear for now, can add cubic later)
    kappa: float
    # kappa_3: float

    # Momentum trader
    alpha: float            # EWMA weight for momentum
    beta: float             # momentum impact coefficient
    gamma: float            # fundamental pull coefficient

    # Noise trader parameter
    sigma_n: float          # order size


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
