# mscthesis_abm/model/globals.py
from dataclasses import dataclass
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
    mu_v: float             # drift of fundamental
    sigma_v: float          # vol of fundamental

    # Fundamental trader parameters (linear for now, can add cubic later)
    kappa: float
    # kappa_3: float

    # Momentum trader parameters
    alpha: float            # EWMA weight for momentum
    beta: float
    gamma: float

    # Noise trader parameter
    sigma_n: float          # order size


class GlobalState:
    """
    Holds global model parameters and the fundamental value process.
    """

    def __init__(self, params: ModelParams, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.v = params.v0        # current fundamental value
        self.t = 0                # time index

    def step_fundamental(self):
        """
        Advance the fundamental value one step.
        This is a placeholder; later we can replace with Heston or
        a data-driven process.
        """
        dv = self.params.mu_v + self.params.sigma_v * self.rng.normal()
        self.v += dv
        self.t += 1