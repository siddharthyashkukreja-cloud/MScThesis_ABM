from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Regime-specific V_t CSV paths (Kalman-smoothed placeholder; will be replaced
# by Merton JD trajectories once Lee-Mykland calibration is implemented).
FV_CSV = {
    "calm":     "data/fv_calm.csv",
    "stressed": "data/fv_stressed.csv",
}

# Per-regime σ_v: per-step log-return std of V_t.
# Currently Kalman-derived (σ_η_daily / sqrt(78)); pending JD replacement.
SIGMA_V = {
    "calm":     0.000926,
    "stressed": 0.005521,
}

# Per-regime opening V_t level (from the loaded fv_*.csv first row).
V0 = {
    "calm":     2501.04,
    "stressed": 3233.82,
}


@dataclass
class ModelParams:
    # ── Direct populations (typically 0 in real-bank-tier setup) ──────────
    n_zi: int
    n_fundamental: int
    n_mm: int

    # ── Asset / cadence ──────────────────────────────────────────────────
    v0: float
    tick_size: float
    dt_minutes: float
    order_ttl: int                  # default LOB TTL (≈ 10 min at dt=5)

    # ── ZI (Cont-Stoikov 2008 + Bouchaud 2002) ────────────────────────────
    zi_alpha: float                 # limit arrivals per minute
    zi_mu: float                    # market arrivals per minute
    zi_delta: float                 # per-resting cancellation rate per minute
    p_zi: float                     # Geometric depth parameter (mid-anchored)

    # ── FT (ODD §Agents) ──────────────────────────────────────────────────
    ft_alpha: float                 # FT arrival rate per minute
    ft_sigma_c: float               # σ_fundamental = ft_sigma_c · σ_v · v0
    ft_threshold_bps: float = 50.0  # dead-band: FT skips submission when
                                    # |reservation − mid| < ft_threshold_bps · V_t / 10000

    # ── Regime toggle (auto-populates σ_v / fv_csv if not overridden) ─────
    stressed: bool = False
    sigma_v: Optional[float] = None
    fv_csv: Optional[str] = None
    sigma_fundamental: Optional[float] = None    # computed in __post_init__

    # ── Quantity range (uniform across ZI limits, ZI markets, FT, fire-sale) ─
    qty_min: int = 1
    qty_max: int = 10

    # ── FT per-order TTL ceiling: per-order TTL ~ U{1, ft_ttl_max} ───────
    ft_ttl_max: int = 10

    # ── Clearing tier ─────────────────────────────────────────────────────
    n_bcm: int = 0
    n_bcm_mm: int = 0
    n_bcm_with_clients: int = 0
    n_nbcm: int = 0
    clients_per_book: int = 6
    client_book_ft: int = 3
    client_book_zi: int = 3

    # ── MM (HFABM Cao 2024 §3.2 single-quote variant) ────────────────────
    mm_qty: int = 50
    mm_p_edge: int = 4              # tick-depth ceiling for d ~ U{0, ..., mm_p_edge}
    mm_inventory_limit: int = 1000
    mm_inventory_safe: int = 500    # post-liquidation target

    # ── Cap-ratio floor (Basel III) ───────────────────────────────────────
    cap_ratio_floor: float = 0.08

    def __post_init__(self):
        regime = "stressed" if self.stressed else "calm"
        if self.sigma_v is None:
            self.sigma_v = SIGMA_V[regime]
        if self.fv_csv is None:
            self.fv_csv = FV_CSV[regime]
        if self.sigma_fundamental is None:
            self.sigma_fundamental = self.ft_sigma_c * self.sigma_v * self.v0


@dataclass
class SimContext:
    """Per-step state passed to every trader's submit_orders.

    V_t is exogenous (loaded from CSV by Simulation); not evolved in-sim.
    """
    v: float
    mid_price: float
    tick: int
    traders_by_id: Dict[int, Any]
