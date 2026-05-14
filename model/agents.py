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
    clearing_member_id: Optional[int] = None

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price

    @property
    def equity(self) -> float:
        return self.cash + self.pnl - self.margin_posted


# ── helpers ──────────────────────────────────────────────────────────────────

def _draw_qty(params: ModelParams, rng: np.random.Generator) -> int:
    return int(rng.integers(params.qty_min, params.qty_max + 1))


def _resting_oids(open_oids: list, lob: LOB) -> list:
    """Drop oids that have been filled / expired / cancelled in the book."""
    return [oid for oid in open_oids if lob.is_resting(oid)]


# ── Zero-Intelligence Trader (Cont-Stoikov 2008 + Bouchaud 2002) ─────────────

@dataclass
class ZeroIntelligenceTrader(BaseTrader):
    """
    Three independent Poisson processes per step:
      - Limit arrivals:   n_limits  ~ Poisson(zi_alpha · dt_minutes)
      - Market arrivals:  n_markets ~ Poisson(zi_mu    · dt_minutes)
      - Per-resting cancellation: Bernoulli with p = 1 − exp(−zi_delta · dt_minutes)

    For each limit: side ~ Uniform{±1}; price = mid ± k · tick_size,
    k ~ Geometric(p_zi), k ≥ 1; qty ~ U[qty_min, qty_max].
    Empty-book fallback: anchor = V_t.
    """
    _open_oids: List[int] = field(default_factory=list, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        self._open_oids = _resting_oids(self._open_oids, lob)

        # Per-resting cancellation pass
        if params.zi_delta > 0 and self._open_oids:
            p_cancel = 1.0 - np.exp(-params.zi_delta * params.dt_minutes)
            kept = []
            for oid in self._open_oids:
                if rng.random() < p_cancel:
                    lob.cancel(oid)
                else:
                    kept.append(oid)
            self._open_oids = kept

        anchor = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v

        # Limit arrivals
        n_limits = int(rng.poisson(params.zi_alpha * params.dt_minutes))
        for _ in range(n_limits):
            side = 1 if rng.random() < 0.5 else -1
            k = int(rng.geometric(params.p_zi))  # k >= 1
            price = anchor - k * params.tick_size if side == 1 else anchor + k * params.tick_size
            price = max(price, params.tick_size)
            qty = _draw_qty(params, rng)
            oid = lob.add_limit(self.agent_id, side, price, qty)
            self._open_oids.append(oid)

        # Market arrivals
        n_markets = int(rng.poisson(params.zi_mu * params.dt_minutes))
        for _ in range(n_markets):
            side = 1 if rng.random() < 0.5 else -1
            qty = _draw_qty(params, rng)
            lob.add_market(self.agent_id, side, qty)


# ── Fundamental Trader (ODD §Agents) ─────────────────────────────────────────

@dataclass
class FundamentalTrader(BaseTrader):
    """
    Per agent z_score ~ N(0,1) fixed at init. Per step:
      n_arrivals ~ Poisson(ft_alpha · dt_minutes)
      reservation = V_t + z_score · σ_fundamental
      side        = sign(reservation − mid)
      each arrival: limit at reservation, qty ~ U[qty_min, qty_max],
                    per-order TTL ~ U{1, ft_ttl_max}.
    """
    z_score: float = 0.0

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        reservation = ctx.v + self.z_score * params.sigma_fundamental
        reservation = max(reservation, params.tick_size)

        # Dead-band trigger on the FT's own mispricing |reservation - mid|.
        # side = sign(reservation - mid): high-z FTs lean buy, low-z lean sell,
        # which keeps order flow two-sided even when V_t drifts. A collective
        # |V_t - mid| trigger was tried and rejected — it synchronises every FT
        # onto one side, empties a book side, and decouples mid from V.
        threshold = params.ft_threshold_bps * 1e-4 * ctx.v
        diff = reservation - ref
        if diff > threshold:
            side = 1
        elif diff < -threshold:
            side = -1
        else:
            return

        n_arrivals = int(rng.poisson(params.ft_alpha * params.dt_minutes))
        for _ in range(n_arrivals):
            qty = _draw_qty(params, rng)
            ttl = int(rng.integers(1, params.ft_ttl_max + 1))
            lob.add_limit(self.agent_id, side, reservation, qty, ttl=ttl)


# ── Market Maker (HFABM Cao 2024 §3.2, single-quote stochastic-spread variant) ─

@dataclass
class MarketMaker(BaseTrader):
    """
    Per step:
      1. Cancel all MM resting quotes from prior step.
      2. If |inventory| > mm_inventory_limit: market-order liquidate to
         mm_inventory_safe; skip quoting this step.
      3. Otherwise quote one bid + one ask:
           d_bid ~ U{0, ..., mm_p_edge}; bid = mid − d_bid · tick
           d_ask ~ U{0, ..., mm_p_edge}; ask = mid + d_ask · tick
         qty = mm_qty each side.
    """
    _open_oids: List[int] = field(default_factory=list, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        for oid in self._open_oids:
            lob.cancel(oid)
        self._open_oids = []

        if abs(self.inventory) > params.mm_inventory_limit:
            self._inventory_liquidate(lob, params)
            return

        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        d_bid = int(rng.integers(0, params.mm_p_edge + 1))
        d_ask = int(rng.integers(0, params.mm_p_edge + 1))
        bid_px = max(ref - d_bid * params.tick_size, params.tick_size)
        ask_px = ref + d_ask * params.tick_size
        oid_b = lob.add_limit(self.agent_id, +1, bid_px, params.mm_qty)
        oid_a = lob.add_limit(self.agent_id, -1, ask_px, params.mm_qty)
        self._open_oids = [oid_b, oid_a]

    def _inventory_liquidate(self, lob: LOB, params: ModelParams):
        target = params.mm_inventory_safe
        if self.inventory > target:
            lob.add_market(self.agent_id, -1, self.inventory - target)
        elif self.inventory < -target:
            lob.add_market(self.agent_id, +1, -self.inventory - target)


# ── Banking Clearing Member ──────────────────────────────────────────────────

@dataclass
class BankingClearingMember(BaseTrader):
    """
    Delegates own-account trading to FT or MM depending on `mode`.
    Cap-ratio fire-sale on breach, regardless of mode.
    """
    cm_type: str = "BCM"
    mode: str = "fundamental"      # 'fundamental' or 'market_maker'
    client_ids: List[int] = field(default_factory=list)
    z_score: float = 0.0

    _ft_delegate: Optional[object] = field(default=None, init=False, repr=False)
    _mm_delegate: Optional[object] = field(default=None, init=False, repr=False)

    def _get_ft(self) -> "FundamentalTrader":
        if self._ft_delegate is None:
            self._ft_delegate = FundamentalTrader(
                agent_id=self.agent_id, cash=self.cash, z_score=self.z_score)
        self._ft_delegate.inventory = self.inventory
        return self._ft_delegate

    def _get_mm(self) -> "MarketMaker":
        if self._mm_delegate is None:
            self._mm_delegate = MarketMaker(
                agent_id=self.agent_id, cash=self.cash)
        self._mm_delegate.inventory = self.inventory
        return self._mm_delegate

    def capital_ratio(self, mid_price: float, traders_by_id) -> float:
        notional = abs(self.inventory) * mid_price
        if traders_by_id is not None:
            for cid in self.client_ids:
                client = traders_by_id.get(cid)
                if client is not None:
                    notional += abs(client.inventory) * mid_price
        return self.cash / notional if notional > 0.0 else float("inf")

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        if self.capital_ratio(ref, ctx.traders_by_id) <= params.cap_ratio_floor:
            self._fire_sale(lob, params)
            return
        if self.mode == "market_maker":
            self._get_mm().submit_orders(lob, params, ctx, rng)
        else:
            self._get_ft().submit_orders(lob, params, ctx, rng)

    def _fire_sale(self, lob: LOB, params: ModelParams):
        chunk = params.qty_max
        if self.inventory > 0:
            lob.add_market(self.agent_id, -1, min(self.inventory, chunk))
        elif self.inventory < 0:
            lob.add_market(self.agent_id, +1, min(-self.inventory, chunk))


# ── Non-Banking Clearing Member ──────────────────────────────────────────────

@dataclass
class NonBankingClearingMember(BaseTrader):
    """Clears for clients only; never trades on own account."""
    cm_type: str = "NBCM"
    client_ids: List[int] = field(default_factory=list)

    def capital_ratio(self, mid_price: float, traders_by_id) -> float:
        notional = 0.0
        if traders_by_id is not None:
            for cid in self.client_ids:
                client = traders_by_id.get(cid)
                if client is not None:
                    notional += abs(client.inventory) * mid_price
        return self.cash / notional if notional > 0.0 else float("inf")

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        return
