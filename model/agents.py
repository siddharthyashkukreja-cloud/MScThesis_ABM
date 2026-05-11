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
    Noise trader, ODD-style placement scaled by current vol sqrt(nu_t).

    Per step (Bernoulli, ODD-faithful):
      with prob zi_alpha:  submit limit at V*(1 + z*sqrt(nu_t)), z fresh ~ N(0,1)
                           side from sign(reservation - mid)
      with prob zi_mu:     submit market order, random side
      qty ~ U{zi_qty_min, zi_qty_max} per ODD §Stochasticity

    No persistent state, no cancellations (order_ttl=1 makes orders expire).
    """

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        sigma_t = float(np.sqrt(max(0.0, ctx.v_var)))
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v

        # Limit-order submission (Bernoulli)
        if rng.random() < params.zi_alpha:
            z = float(rng.standard_normal())
            reservation = ctx.v * (1.0 + z * sigma_t)
            reservation = max(reservation, params.tick_size)
            if reservation > ref:
                side = 1
            elif reservation < ref:
                side = -1
            else:
                side = 0
            if side != 0:
                qty = int(rng.integers(params.zi_qty_min, params.zi_qty_max + 1))
                lob.add_limit(self.agent_id, side, reservation, qty)

        # Market-order submission (Bernoulli, random side)
        if rng.random() < params.zi_mu:
            side = 1 if rng.random() < 0.5 else -1
            qty = int(rng.integers(params.zi_qty_min, params.zi_qty_max + 1))
            lob.add_market(self.agent_id, side, qty)


@dataclass
class FundamentalTrader(BaseTrader):
    """
    Chiarella-style fundamental trader, ODD §Prediction placement with
    vol-scaled spread (Gao 2023-style sigma_t = sqrt(nu_t)).

    Each FT has fixed z ~ N(0,1) at init.
    Per step:
        reservation = V * (1 + z * sqrt(nu_t))
        side = sign(reservation - mid)
        submit limit at reservation, qty ~ U{dir_qty_min, dir_qty_max}
    Empty-book bootstrap: anchor on V.
    """
    z_score: float = 0.0

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        sigma_t = float(np.sqrt(max(0.0, ctx.v_var)))
        reservation = ctx.v * (1.0 + self.z_score * sigma_t)
        reservation = max(reservation, params.tick_size)
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        if reservation > ref:
            side = 1
        elif reservation < ref:
            side = -1
        else:
            return
        qty = int(rng.integers(params.dir_qty_min, params.dir_qty_max + 1))
        lob.add_limit(self.agent_id, side, reservation, qty)


@dataclass
class MomentumTrader(BaseTrader):
    """
    Chiarella-style momentum trader (Majewski 2018 / Krishnen) with
    vol-scaled placement spread.

    Direction = sign of EWMA log-return momentum:
        M_t = lambda * M_{t-1} + (1 - lambda) * (log P_t - log P_{t-1})
    Skip if |M_t| < mt_threshold.

    Placement: price = mid * (1 + z * sqrt(nu_t)),   z ~ N(0,1) at init.
    Empty-book fallback: anchor on V.
    qty ~ U{dir_qty_min, dir_qty_max}.
    """
    z_score: float = 0.0

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        if abs(ctx.momentum) < params.mt_threshold:
            return
        sigma_t = float(np.sqrt(max(0.0, ctx.v_var)))
        side = 1 if ctx.momentum > 0 else -1
        anchor = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        price = anchor * (1.0 + self.z_score * sigma_t)
        price = max(price, params.tick_size)
        qty = int(rng.integers(params.dir_qty_min, params.dir_qty_max + 1))
        lob.add_limit(self.agent_id, side, price, qty)


@dataclass
class BankingClearingMember(FundamentalTrader):
    """
    Banking Clearing Member (BCM), two modes:
      - 'fundamental': trade as FT (V·(1 + z·sqrt(nu_t)) reservation)
      - 'market_maker': HFABM Cao 2024 §3.2 ladder

    Cap-ratio governance: at any breach (cap_ratio <= cap_ratio_floor),
    enter fire-sale (market-order liquidate own inventory). MM mode also
    enforces an absolute inventory limit (HFABM): when |inv| >
    mm_inventory_limit, market-liquidate down to half the limit.

    Capital ratio aggregates own + client mark-to-market exposure.
    """
    cm_type: str = "BCM"
    mode: str = "fundamental"  # 'fundamental' | 'market_maker'
    client_ids: List[int] = field(default_factory=list)
    cap_ratio_floor: float = 0.08

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
            # HFABM ladder; first check inventory limit
            if abs(self.inventory) > params.mm_inventory_limit:
                self._inventory_liquidate(lob, params)
                return
            self._post_ladder(lob, params, ref)
        else:  # 'fundamental'
            super().submit_orders(lob, params, ctx, rng)

    def _fire_sale(self, lob: LOB, params: ModelParams):
        """Cap-ratio breach: liquidate own inventory in market-order chunks."""
        if self.inventory > 0:
            qty = min(self.inventory, params.dir_qty_max)
            lob.add_market(self.agent_id, -1, qty)
        elif self.inventory < 0:
            qty = min(-self.inventory, params.dir_qty_max)
            lob.add_market(self.agent_id, 1, qty)

    def _inventory_liquidate(self, lob: LOB, params: ModelParams):
        """HFABM inventory limit breach: liquidate down to half the limit."""
        target = params.mm_inventory_limit // 2
        if self.inventory > target:
            qty = min(self.inventory - target, params.dir_qty_max)
            lob.add_market(self.agent_id, -1, qty)
        elif self.inventory < -target:
            qty = min(-self.inventory - target, params.dir_qty_max)
            lob.add_market(self.agent_id, 1, qty)

    def _post_ladder(self, lob: LOB, params: ModelParams, ref: float):
        """HFABM Cao 2024 §3.2: post symmetric ladder at fixed tick offsets."""
        for k in range(1, params.mm_n_levels + 1):
            bid_px = max(ref - k * params.tick_size, params.tick_size)
            ask_px = max(ref + k * params.tick_size, params.tick_size)
            lob.add_limit(self.agent_id, +1, bid_px, params.mm_qty)
            lob.add_limit(self.agent_id, -1, ask_px, params.mm_qty)


@dataclass
class NonBankingClearingMember(BaseTrader):
    """
    Non-Banking Clearing Member (NBCM): clears for clients, never trades.

    Capital ratio aggregates client exposure only (own inventory == 0).
    Stop-out on cap-ratio breach is enforced at Stage 4 margin layer.
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
        return
