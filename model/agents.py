import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .globals import ModelParams, SimContext
from .lob import LOB


@dataclass
class BaseTrader:
    agent_id: int
    cash: float
    inventory: int = 0
    pnl: float = 0.0
    margin_posted: float = 0.0
    defaulted: bool = False
    # ODD §Interactions: client trader margin flows through their CM (Stage 4+).
    # None for direct market participants; integer = agent_id of clearing CM.
    clearing_member_id: Optional[int] = None

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price

    @property
    def equity(self) -> float:
        return self.cash + self.pnl - self.margin_posted


@dataclass
class ZeroIntelligenceTrader(BaseTrader):
    """
    Zero-intelligence noise trader.

    Continuous-time Cont & Stoikov (2008) order-event semantics adapted
    to a discrete dt_minutes step:
      - n_limit  ~ Poisson(zi_alpha  * dt_minutes)
      - n_market ~ Poisson(zi_mu     * dt_minutes)
      - per-resting-order cancel prob = 1 - exp(-zi_delta * dt_minutes)
        (Farmer-Daniels 2003 per-order cancellation).
    Each arrival has independent side (50/50), qty ~ U{1, zi_qty_max},
    and (for limits) depth k ~ Geometric(zi_offset_p) capped at
    zi_offset_max ticks measured from the opposite best quote
    (Cont-Stoikov empirical exponential depth profile lambda(i)).
    Empty-book fallback: anchor on fundamental V.
    """
    _queued_order_ids: List[int] = field(default_factory=list)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        dt = params.dt_minutes

        # Cancellations: per-resting Bernoulli with rate-rescaled prob
        p_cancel = 1.0 - float(np.exp(-params.zi_delta * dt))
        surviving = []
        for oid in self._queued_order_ids:
            if rng.random() < p_cancel:
                lob.cancel(oid)
            else:
                surviving.append(oid)
        self._queued_order_ids = surviving

        # Gao 2023 noise-demand: order qty scales with current vol,
        # qty_max_eff = round(zi_qty_max * sigma_t / sqrt(theta_v)).
        # No-op when CIR disabled (v_var == sigma_v^2 = theta_v fallback).
        if ctx.v_var > 0.0 and params.theta_v > 0.0:
            vol_scale = float(np.sqrt(ctx.v_var / params.theta_v))
            qty_max_eff = max(params.zi_qty_min,
                              min(int(round(params.zi_qty_max * vol_scale)),
                                  10 * params.zi_qty_max))   # cap at 10x to avoid runaway
        else:
            qty_max_eff = params.zi_qty_max

        # Limit-order arrivals: Poisson per step
        n_limit = int(rng.poisson(params.zi_alpha * dt))
        for _ in range(n_limit):
            side = 1 if rng.random() < 0.5 else -1
            qty = int(rng.integers(params.zi_qty_min, qty_max_eff + 1))
            k = min(int(rng.geometric(params.zi_offset_p)), params.zi_offset_max)
            if side == 1:
                anchor = lob.best_ask if not np.isnan(lob.best_ask) else ctx.v
                price = anchor - k * params.tick_size
            else:
                anchor = lob.best_bid if not np.isnan(lob.best_bid) else ctx.v
                price = anchor + k * params.tick_size
            price = max(price, params.tick_size)
            oid = lob.add_limit(self.agent_id, side, price, qty)
            self._queued_order_ids.append(oid)

        # Market-order arrivals: Poisson per step (also vol-scaled)
        n_market = int(rng.poisson(params.zi_mu * dt))
        for _ in range(n_market):
            side = 1 if rng.random() < 0.5 else -1
            qty = int(rng.integers(params.zi_qty_min, qty_max_eff + 1))
            lob.add_market(self.agent_id, side, qty)


@dataclass
class FundamentalTrader(BaseTrader):
    """
    Chiarella-style fundamental trader, ODD §Prediction placement.

    Each FT has a fixed private valuation offset z ~ N(0,1) drawn once at
    init (ODD §Stochasticity, Mech #1: HeterogeneousPrivateValuation).
    Reservation price r_i = V_t + z_i * sigma_F. Side determined by
    sign(r_i - mid_price): if reservation > mid, agent values the asset
    above the market and submits a buy at r_i; if r_i < mid, sell at r_i.
    Quantity ~ Uniform{1, zi_qty_max} per ODD §Stochasticity.

    No demand-magnitude scaling (kappa) -- ODD's CMs use fixed-size random
    orders with sign from private val vs market. This deviates from
    Majewski et al. (2018) Chiarella demand kappa*(V-P) -- flagged
    explicitly: ODD spec is followed for heterogeneity, Majewski's kappa
    is reintroduced only if needed for calibration in Stage 8.

    Empty-book bootstrap: when mid_price is NaN, anchor on V (fundamental)
    so the first step still places a sensible quote.
    """
    z_score: float = 0.0

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        # V-relative reservation (regime-invariant): V * (1 + z * ft_sigma_rel)
        reservation = ctx.v * (1.0 + self.z_score * params.ft_sigma_rel)
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        if reservation > ref:
            side = 1
        elif reservation < ref:
            side = -1
        else:
            return
        price = max(reservation, params.tick_size)
        qty = int(rng.integers(params.zi_qty_min, params.zi_qty_max + 1))
        lob.add_limit(self.agent_id, side, price, qty)


@dataclass
class BankingClearingMember(FundamentalTrader):
    """
    Banking Clearing Member (BCM) -- ODD §Agents.

    Two trading modes:
      'fundamental' : trade as a FundamentalTrader (V+z*sigma placement);
                      may carry clients
      'market_maker': cancel previous quotes; post symmetric bid+ask
                      around mid with inventory-skew (Stoikov 2008);
                      no clients in baseline (MMs are pure liquidity)

    Capital ratio aggregates own and client mark-to-market exposure
    (cash / [(|own_inv| + sum |client_inv|) * mid_price]). On
    cap_ratio <= cap_ratio_floor (Basel III 8% floor, ODD §Mech #2),
    enters fire-sale mode regardless of mode: aggressive market-order
    liquidation of own inventory until cap_ratio is restored.
    """
    cm_type: str = "BCM"
    mode: str = "fundamental"  # 'fundamental' | 'market_maker'
    client_ids: List[int] = field(default_factory=list)
    cap_ratio_floor: float = 0.08
    _mm_order_ids: List[int] = field(default_factory=list)

    def capital_ratio(self, mid_price: float, traders_by_id: Optional[dict]) -> float:
        notional = abs(self.inventory) * mid_price
        if traders_by_id is not None:
            for cid in self.client_ids:
                client = traders_by_id.get(cid)
                if client is not None:
                    notional += abs(client.inventory) * mid_price
        if notional == 0.0:
            return float("inf")
        return self.cash / notional

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        cap = self.capital_ratio(ref, ctx.traders_by_id)
        if cap <= self.cap_ratio_floor:
            self._fire_sale(lob, params)
            return
        if self.mode == "market_maker":
            self._market_make(lob, params, ref)
        else:  # 'fundamental'
            super().submit_orders(lob, params, ctx, rng)

    def _fire_sale(self, lob: LOB, params: ModelParams):
        if self.inventory > 0:
            qty = min(self.inventory, params.zi_qty_max)
            lob.add_market(self.agent_id, -1, qty)
        elif self.inventory < 0:
            qty = min(-self.inventory, params.zi_qty_max)
            lob.add_market(self.agent_id, 1, qty)

    def _market_make(self, lob: LOB, params: ModelParams, ref: float):
        # Cancel previous quotes (one full requote per step)
        for oid in self._mm_order_ids:
            lob.cancel(oid)
        self._mm_order_ids = []

        # All scales in basis points of ref price (V-relative):
        shift_frac = -params.mm_inventory_skew_bps * self.inventory * 1e-4
        half_frac  = params.mm_half_spread_bps * 1e-4
        bid_px = ref * (1.0 + shift_frac - half_frac)
        ask_px = ref * (1.0 + shift_frac + half_frac)
        bid_px = max(bid_px, params.tick_size)
        ask_px = max(ask_px, params.tick_size)
        if ask_px <= bid_px:
            ask_px = bid_px + params.tick_size

        oid_b = lob.add_limit(self.agent_id, +1, bid_px, params.mm_qty)
        oid_a = lob.add_limit(self.agent_id, -1, ask_px, params.mm_qty)
        self._mm_order_ids = [oid_b, oid_a]


@dataclass
class NonBankingClearingMember(BaseTrader):
    """
    Non-Banking Clearing Member (NBCM) -- ODD §Agents, structural-only.

    Pure client-clearing entity: holds cash + a client_ids list but does
    NOT trade on own account (inventory stays 0). Capital ratio is
    measured against AGGREGATE CLIENT exposure (cash divided by client
    notional), so when clients accumulate large positions the NBCM cap
    ratio drops. Cap-ratio breach behaviour (stop-out: instruct clients
    to halt new submissions, ODD §Mech #2 NBCM branch) is enforced at the
    Stage 4 margin layer; at Stage 3 the NBCM is a passive structural
    placeholder that submit_orders ignores.
    """
    cm_type: str = "NBCM"
    client_ids: List[int] = field(default_factory=list)
    cap_ratio_floor: float = 0.08

    def capital_ratio(self, mid_price: float, traders_by_id: Optional[dict]) -> float:
        notional = 0.0
        if traders_by_id is not None:
            for cid in self.client_ids:
                client = traders_by_id.get(cid)
                if client is not None:
                    notional += abs(client.inventory) * mid_price
        if notional == 0.0:
            return float("inf")
        return self.cash / notional

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        return  # NBCMs do not trade; client stop-out enforced at Stage 4


@dataclass
class MomentumTrader(BaseTrader):
    """
    Chiarella-style momentum trader (Majewski et al. 2018 EWMA signal),
    ODD-style heterogeneous placement.

    Direction is determined by the sign of an EWMA log-return momentum
    signal M_t maintained in Simulation:
        M_t = lambda * M_{t-1} + (1 - lambda) * (log P_t - log P_{t-1}).
    Below an absolute threshold |M_t| < mt_threshold, the agent stays
    out (no zero-crossing churn).

    Placement reuses the ODD §Prediction private-valuation formula:
    price = V_t + z_i * sigma_M, with z_i ~ N(0,1) fixed at init.
    Anchoring on V (rather than mid) is a deliberate Stage 2 choice to
    keep a single ODD-faithful placement rule across all non-ZI agents;
    the ODD itself does not specify MTs (CMs only), so this is an
    explicit extension. Quantity ~ Uniform{1, zi_qty_max}.
    """
    z_score: float = 0.0

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        if abs(ctx.momentum) < params.mt_threshold:
            return
        side = 1 if ctx.momentum > 0 else -1
        # V-relative placement (regime-invariant)
        price = ctx.v * (1.0 + self.z_score * params.mt_sigma_rel)
        price = max(price, params.tick_size)
        qty = int(rng.integers(params.zi_qty_min, params.zi_qty_max + 1))
        lob.add_limit(self.agent_id, side, price, qty)
