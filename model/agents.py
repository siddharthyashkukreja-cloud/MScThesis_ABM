import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .globals import ModelParams, SimContext
from .lob import LOB
from .clearing import BalanceSheet


@dataclass
class BaseTrader:
    agent_id: int
    cash: float
    inventory: int = 0
    pnl: float = 0.0
    # Id of the clearing member this trader clears through; None for clearing
    # members themselves. Set for every cleared client (FT/MT/ZI).
    clearing_member_id: Optional[int] = None
    # Client clearing-account state. Cleared clients post variation margin from
    # their own cash each margin cycle; `_stopped` freezes a client that breaches the
    # 8% capital floor; `has_defaulted` marks cash exhaustion, on which its CM absorbs
    # the shortfall. Unused by the CMs themselves (they track default on the balance
    # sheet).
    _stopped: bool = field(default=False, init=False, repr=False)
    has_defaulted: bool = field(default=False, init=False, repr=False)
    _cm_last_mark: float = field(default=0.0, init=False, repr=False)
    # Signed fills since the last VM mark. The margin cycle settles P&L against the
    # average filled price (ODD §Margin Call: P/L = N·(P_market − P_filled)), not
    # position·Δmid alone — so execution slippage (fire-sale book walks) is realised.
    _fill_qty: int = field(default=0, init=False, repr=False)
    _fill_cost: float = field(default=0.0, init=False, repr=False)
    # Initial margin physically posted to the CCP (G.IM_ESCROW). Cash is moved into
    # this account each margin cycle and released as the position shrinks or on default.
    _posted_im: float = field(default=0.0, init=False, repr=False)

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price


# ── helpers ──────────────────────────────────────────────────────────────────

def _draw_qty(params: ModelParams, rng: np.random.Generator) -> int:
    """Uniform `U[qty_min, qty_max]` order size — ODD §Stochasticity. Each model
    qty unit represents an institutional block of contracts (the VOLUME_LOT
    relabeling in globals.py scales to empirical ES levels)."""
    return int(rng.integers(params.qty_min, params.qty_max + 1))


def _draw_depth(params: ModelParams, rng: np.random.Generator) -> int:
    """Geometric placement depth in ticks, k >= 1 — shared by ZI and MT
    mid-anchored limit orders. k ~ Geometric(p_zi), the Cont-Stoikov-Talreja
    (2008) geometric, with p_zi fit by MBP-10 MLE (globals.P_ZI). Dense at the mid
    (mode k=1) keeps the near-mid book thick so market orders don't walk a sparse
    book."""
    return int(rng.geometric(params.p_zi))


def _resting_oids(open_oids: list, lob: LOB) -> list:
    """Drop oids that have been filled or cancelled in the book."""
    return [oid for oid in open_oids if lob.is_resting(oid)]


def _bernoulli_cancel(open_oids: list, lob: LOB, delta: float,
                      rng: np.random.Generator) -> list:
    """Per-resting Bernoulli cancellation pass with rate `delta` per step.
    Returns the surviving oid list. Cont-Stoikov (2008) geometric lifetime."""
    open_oids = _resting_oids(open_oids, lob)
    if delta <= 0 or not open_oids:
        return open_oids
    kept = []
    for oid in open_oids:
        if rng.random() < delta:
            lob.cancel(oid)
        else:
            kept.append(oid)
    return kept


def _client_cap_qty(trader, side: int, qty: int, params: ModelParams,
                    mid: float) -> int:
    """Position cap applied before an opening order. Two regimes:
    • a cleared CLIENT (clearing_member_id set) cannot open beyond what its free cash
      can margin — |pos| <= cash / (house_im · VOLUME_LOT · CONTRACT_USD · mid), where
      house_im is the broker house margin (CCP_CALIBRATION im_percent = 20%, i.e. 5×),
      uniform across clients;
    • a banking CM's OWN account (balance_sheet.is_banking) is bounded by the static
      gross-leverage limit |pos| <= POSITION_LIMIT_X·cash / (VOLUME_LOT · CONTRACT_USD ·
      mid) (POSITION_LIMIT_X > 0; respecting POSITION_LIMIT_CLIENTS_ONLY), bounding prop
      accumulation in a trend.
    Orders that reduce/flatten the position are never capped; other agents are uncapped."""
    if qty <= 0:
        return qty
    inv = trader.inventory
    if not ((side > 0 and inv >= 0) or (side < 0 and inv <= 0)):
        return qty                                   # reducing — never capped
    from .globals import CONTRACT_USD, CCP_CALIBRATION
    px = mid if (mid == mid and mid > 0) else params.v0
    base = params.volume_lot * CONTRACT_USD * px
    if trader.clearing_member_id is not None:        # cleared client — margin-capacity cap
        denom = CCP_CALIBRATION["im_percent"] * base
    elif getattr(trader, "balance_sheet", None) is not None and trader.balance_sheet.is_banking:
        from .globals import (POSITION_LIMIT_X, POSITION_LIMIT_CLIENTS_ONLY,
                              POSITION_LIMIT_X_HOUSE)
        if POSITION_LIMIT_X > 0.0:                    # static own-book leverage cap (no vol input)
            if POSITION_LIMIT_CLIENTS_ONLY and not getattr(trader, "client_ids", None):
                if POSITION_LIMIT_X_HOUSE <= 0.0:
                    return qty                       # house-only BCM: uncapped (8% floor only)
                denom = base / POSITION_LIMIT_X_HOUSE  # house-only BCM: looser finite leverage cap
            else:
                denom = base / POSITION_LIMIT_X      # client-clearing BCM: |own| <= X·cash/base
        else:
            return qty
    else:
        return qty                                   # uncleared / non-CM — uncapped
    if denom <= 0:
        return qty
    room = int(trader.cash / denom) - abs(inv)
    return max(0, min(int(qty), room))


# ── Zero-Intelligence Trader (Cont-Stoikov 2008) — background noise floor ──

@dataclass
class ZeroIntelligenceTrader(BaseTrader):
    """
    Cont-Stoikov (2008) noise trader — provides the continuous background
    flow that smooths per-step price impact.

    Per step, three independent Bernoulli draws (per-step probabilities at the
    1-min cadence; no dt rescaling):
      - per-resting limit cancelled w.p. zi_delta
      - submit one limit order w.p. zi_alpha → random side, depth k from
                                   the shared geometric `_draw_depth`
                                   (data-fit p_zi), qty ~ U[qty_min, qty_max]
      - submit one market order w.p. zi_mu → random side, qty ~ U[…]

    zi_alpha (limit arrival) and zi_delta (per-resting cancellation) are
    calibrated; zi_mu (market arrival) is pinned at the ODD §Calibration baseline
    0.025. zi_delta is the sole control on ZI order lifetime; the geometric
    placement depth is shared with MT.
    """
    _open_oids: List[int] = field(default_factory=list, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        if self._stopped or self.has_defaulted:      # frozen / defaulted client
            return
        self._open_oids = _bernoulli_cancel(self._open_oids, lob,
                                            params.zi_delta, rng)
        anchor = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        if rng.random() < params.zi_alpha:
            side = 1 if rng.random() < 0.5 else -1
            k = _draw_depth(params, rng)
            price = max(anchor - side * k * params.tick_size, params.tick_size)
            qty = _client_cap_qty(self, side, _draw_qty(params, rng), params, anchor)
            if qty > 0:
                oid = lob.add_limit(self.agent_id, side, price, qty)
                self._open_oids.append(oid)
        if rng.random() < params.zi_mu:
            side = 1 if rng.random() < 0.5 else -1
            qty = _client_cap_qty(self, side, _draw_qty(params, rng), params, anchor)
            if qty > 0:
                lob.add_market(self.agent_id, side, qty)


# ── Fundamental Trader (ODD §Agents) ─────────────────────────────────────────

@dataclass
class FundamentalTrader(BaseTrader):
    """
    Per agent z_score ~ N(0,1) fixed at init — persistent heterogeneous beliefs
    (Chiarella-Iori-Perelló). Per step:
      reservation = V_t · (1 + z_score · ft_sigma_c · σ_t)
                        (σ_t = EWMA realised vol of V_t, read off SimContext;
                        falls back to params.sigma_v when absent. ft_sigma_c is
                        calibrated per regime — the dominant return-tail lever.
                        The offset z·ft_sigma_c·σ_t is fractional, so the belief
                        cloud is scale-invariant around V_t — no v0 level constant.)
      side = sign(reservation − mid)
      activation = Bernoulli(ft_alpha)
      on activation: replace-on-new — cancel the FT's standing limit (if still
                     resting), place one fresh limit at the reservation,
                     qty ~ U[qty_min, qty_max].

    There is no dead-band: the persistent z_score supplies the FT heterogeneity,
    so the FT acts whenever its reservation differs from the mid.

    Order management is replace-on-new: the FT holds at most one resting limit,
    refreshed on the next activation. It trades every step (ft_alpha=1), so the
    order is refreshed each step in practice. There is no per-resting cancellation
    rate and no TTL — an un-refreshed order persists until filled.
    """
    z_score: float = 0.0
    _open_oid: Optional[int] = field(default=None, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        if self._stopped or self.has_defaulted:      # frozen / defaulted client
            return
        # Drop the standing-order reference if it was filled / TTL-expired.
        if self._open_oid is not None and not lob.is_resting(self._open_oid):
            self._open_oid = None

        # Stochastic cancellation of the standing order (campaign E5; CST-2008 /
        # Farmer ZI cancel rate). Off by default (ft_delta=0.0, replace-on-new).
        if self._open_oid is not None and params.ft_delta > 0.0 and rng.random() < params.ft_delta:
            lob.cancel(self._open_oid); self._open_oid = None
        # Optional Bernoulli activation gate: skip this step w.p. 1-ft_alpha.
        # Off by default (ft_alpha=1.0, so FTs act every step).
        if params.ft_alpha < 1.0 and rng.random() >= params.ft_alpha:
            return

        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        # Reservation as a FRACTIONAL belief offset around the current
        # fundamental: R = V_t * (1 + z * ft_sigma_c * sigma_t). z~N(0,1) is
        # fixed per FT, so the belief cloud's fractional std is ft_sigma_c *
        # sigma_t — scale-invariant, with no v0 level constant in the price-
        # formation law. sigma_t is the EWMA realised vol of V_t (read off
        # SimContext; falls back to params.sigma_v when absent). ft_sigma_c is
        # calibrated — the dominant return-tail lever; note it does NOT transmit
        # volatility clustering (the adaptive width damps high-vol bursts), which
        # enters via the V_t path and the momentum traders.
        sigma_t = ctx.sigma_t if ctx.sigma_t > 0.0 else params.sigma_v
        reservation = ctx.v * (1.0 + self.z_score * params.ft_sigma_c * sigma_t)
        reservation = max(reservation, params.tick_size)

        # Side from the sign of the FT's own mispricing: using the per-agent
        # reservation rather than the collective V_t keeps order flow two-sided.
        # No dead-band — z_score supplies the heterogeneity.
        diff = reservation - ref
        if diff > 0:
            side = 1
        elif diff < 0:
            side = -1
        else:
            return

        # FT trades every step (ODD §Step Sequence step 3: "CMs
        # submit BuyOrder / SellOrder messages based on capital ratio check
        # and limit price vs market price comparison"). Replace-on-new
        # without a Bernoulli gate — `ft_alpha` is pinned at 1.0 and out of
        # the calibration loop.
        qty = _client_cap_qty(self, side, _draw_qty(params, rng), params, ref)
        if qty <= 0:
            return                                   # at margin cap on this side — hold
        if self._open_oid is not None:
            lob.cancel(self._open_oid)
        self._open_oid = lob.add_limit(self.agent_id, side, reservation, qty)


# ── Momentum Trader (single-type EWMA chartist) ─────────────────────────────

@dataclass
class MomentumTrader(BaseTrader):
    """
    Single-type EWMA chartist. EWMA momentum on mid log-returns:

        r_t = log(mid_{t-1}) − log(mid_{t-2})
        M_t = (1 − mt_lambda) · M_{t-1} + mt_lambda · r_t

    `mt_lambda` is shared by every MT and pinned at 0.05: Majewski et al. (2018)
    fix the trend horizon externally (α = 1/7, τ = 6 months, from a CTA-index
    correlation) rather than estimating it, so pinning it follows their practice.

    `M_t` must clear a tiny floor `mt_eps` (skips the EWMA warm-up); the sign of
    M_t fixes the order side. The MT is limit-only (`mt_mu` is unused). It trades
    every step:

      limit (every step) — passive quote, mid-anchored:
        k ~ Geometric(p_zi) placement depth (_draw_depth — shared with ZI)
        price = mid - side · k · tick (buy below mid, sell above)

    The placement depth is the shared data-fit geometric `_draw_depth` (see
    globals.P_ZI). The limit branch is replace-on-new — at most one standing
    limit, cancelled and replaced each step.

    Single cohort. Each MT still carries its own `lambda_decay`
    field (per-agent init kwarg) so a long cohort can be re-enabled by
    setting `n_momentum_long > 0`, but at runtime all MTs use `mt_lambda`.
    """
    lambda_decay: float = 0.1   # per-agent EWMA decay; set at construction
    _M: float = field(default=0.0, init=False, repr=False)
    _prev_mid: float = field(default=float("nan"), init=False, repr=False)
    _open_oid: Optional[int] = field(default=None, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        if self._stopped or self.has_defaulted:      # frozen / defaulted client
            return
        # Drop the standing-order reference if it was filled / TTL-expired.
        if self._open_oid is not None and not lob.is_resting(self._open_oid):
            self._open_oid = None

        # EWMA momentum update — runs EVERY step regardless of activation.
        cur_mid = ctx.mid_price
        if (not np.isnan(cur_mid) and not np.isnan(self._prev_mid)
                and cur_mid > 0 and self._prev_mid > 0):
            r = np.log(cur_mid) - np.log(self._prev_mid)
            self._M = (1.0 - self.lambda_decay) * self._M + self.lambda_decay * r
        if not np.isnan(cur_mid):
            self._prev_mid = cur_mid

        # Stochastic cancellation + Bernoulli gate (campaign E5) — placed AFTER the
        # EWMA update so the trend signal keeps updating on skipped steps. Off by
        # default (mt_delta=0.0 replace-on-new, mt_alpha=1.0 trades every step).
        if self._open_oid is not None and params.mt_delta > 0.0 and rng.random() < params.mt_delta:
            lob.cancel(self._open_oid); self._open_oid = None
        if params.mt_alpha < 1.0 and rng.random() >= params.mt_alpha:
            return

        am = abs(self._M)
        if am < params.mt_eps:
            return
        # Activation scales with trend strength: P(trade) = tanh(|M_t| / (mt_gamma·sigma_v)),
        # so a larger SHARE of momentum traders act when the signal is strong. Replaces the old
        # hard sign-only gate, under which a barely-above-floor M_t traded as forcefully as a
        # strong trend (the demand now responds to signal strength via the activated fraction).
        p_act = np.tanh(am / (params.mt_gamma * max(params.sigma_v, 1e-12)))
        if rng.random() >= p_act:
            return
        side = 1 if self._M > 0 else -1

        # Market branch removed. MT is limit-only;
        # `mt_mu` no longer used (kept on ModelParams default 0.0 for
        # backward compat; out of PARAM_BOUNDS).

        # Limit branch — passive mid-anchored quote, posted every step
        #.
        anchor = cur_mid if not np.isnan(cur_mid) else ctx.v
        k = _draw_depth(params, rng)          # shared log-normal depth
        price = anchor - side * k * params.tick_size
        price = max(price, params.tick_size)
        qty = _client_cap_qty(self, side, _draw_qty(params, rng), params, anchor)
        if qty <= 0:
            return                                   # at margin cap on this side — hold
        # Replace-on-new: cancel the standing order, place a fresh one.
        if self._open_oid is not None:
            lob.cancel(self._open_oid)
        self._open_oid = lob.add_limit(self.agent_id, side, price, qty)


# ── Clearing tier ───────

@dataclass
class BankingClearingMember(FundamentalTrader):
    """
    Banking clearing member (ODD §Agents BankingClearingMember). Cast from a
    FundamentalTrader: the ODD's CMs price via z_score·σ_fundamental offset
    from the fundamental signal — identical to the thesis FT — so the BCM
    trades its OWN account through the inherited FT `submit_orders`. At the
    pre-margin scaffold stage its order flow is exactly an FT's, so the 8-d
    market-layer calibration is unchanged.

    Beyond own-account trading the BCM also CLEARS a client book (thesis
    client-clearing extension). It carries a `BalanceSheet`, a CCP
    membership link (`ccp_id`) and a `client_ids` list. The ODD §Mech #2
    capital-adequacy fire-sale is deferred to the stress stage.
    """
    balance_sheet: Optional[BalanceSheet] = None
    ccp_id: Optional[int] = None
    client_ids: List[int] = field(default_factory=list)
    _liq_slices: List[int] = field(default_factory=list, repr=False)  # AC fire-sale queue
    _liq_side: int = 0

    def __post_init__(self):
        if self.balance_sheet is None:
            self.balance_sheet = BalanceSheet(owner_id=self.agent_id,
                                              is_banking=True)

    def start_firesale(self, qty: int, urgency: float, horizon: int) -> None:
        """ODD §Mech #2 forced deleveraging via Almgren-Chriss liquidation: on an
        8%-capital-ratio breach, sell down `qty` of the OWN position over
        `horizon` steps to restore compliance (sell if long, buy if short).
        Slices are sent as LOB market orders by submit_orders — the book walk is
        the temporary impact (permanent impact GAMMA_PERM ≈ 0 empirically)."""
        from model.clearing import ac_slices
        queued = sum(self._liq_slices)                       # don't double-schedule
        qty = min(int(abs(qty)), abs(self.inventory) - queued)
        if qty <= 0:
            return
        self._liq_side = -1 if self.inventory > 0 else 1
        self._liq_slices.extend(ac_slices(qty, horizon, urgency))  # EXTEND: clustered
        #                                       client defaults in one cycle must all queue

    def submit_orders(self, lob, params, ctx, rng):
        """In a fire-sale, send the next AC liquidation slice as a market order
        and skip normal trading; otherwise trade own-account as an FT."""
        if self._liq_slices:
            qty = int(self._liq_slices.pop(0))
            if qty > 0:
                lob.add_market(self.agent_id, self._liq_side, qty)
            return
        if self._stopped:           # below the leverage floor / frozen: add no new own risk
            return
        super().submit_orders(lob, params, ctx, rng)

    def capital_ratio(self, mid: float, sigma_t: float = 0.0) -> float:
        """Basel III CAPITAL ADEQUACY RATIO (D78): capital (cash, net of escrowed IM) over risk
        exposure (own + client cleared notional). `cash/exposure >= 8%` mirrors the Basel III
        total-capital minimum (8% of RWA; CET1 4.5 / Tier 1 6 / Total 8) and is the ODD's
        capital-adequacy constraint (cash/|tradePosition|, "mirrors Basel III") — NOT the 3%
        leverage ratio and NOT a liquidity ratio. On breach the BCM deleverages its own book
        (Almgren-Chriss) to restore the ratio (the forced-deleveraging / leverage-cycle channel);
        an own-account-only BCM is held well above 8% by POSITION_LIMIT_X, so only client-carrying
        members bind. (sigma_t kept for signature compatibility; the CAR is exposure-based.)"""
        from model.globals import CONTRACT_USD
        own_notional = abs(self.inventory) * self.balance_sheet.volume_lot * CONTRACT_USD * mid
        exposure = own_notional + self.balance_sheet.client_notional(mid)
        if exposure <= 0.0:
            return float("inf")
        return self.cash / exposure


@dataclass
class NonBankingClearingMember(BaseTrader):
    """
    Non-banking clearing member (ODD §Agents NonBankingClearingMember). A
    pure clearing intermediary: it holds NO own position and submits NO
    orders to the LOB — its only exposure is the client book. It is
    therefore not in the LOB-trading `traders` list; it is registered with
    the CCP and carries a `BalanceSheet`, a CCP link and a `client_ids`
    list. `cash ~ U[5M, 10M]` (ODD §Initialization). The ODD §Mech #2
    stop-out is deferred to the stress stage.
    """
    balance_sheet: Optional[BalanceSheet] = None
    ccp_id: Optional[int] = None
    client_ids: List[int] = field(default_factory=list)
    # _stopped (ODD §Mech #2 stop-out, capital_ratio <= floor) is inherited from BaseTrader.
    # Fire-sale queue for a DEFAULTED client's position the NBCM has assumed.
    # The NBCM has no LOB access, so Simulation routes these slices through the CCP,
    # attributed to the NBCM, so the fills mark down its assumed `inventory` — the loss
    # is realised by marking the assumed book to market (deficit-consistent), not a flat
    # (1-recovery) haircut. Empty in normal operation (NBCM carries no position).
    _liq_slices: List[int] = field(default_factory=list, repr=False)
    _liq_side: int = 0

    def __post_init__(self):
        if self.balance_sheet is None:
            self.balance_sheet = BalanceSheet(owner_id=self.agent_id,
                                              is_banking=False)

    def start_firesale(self, qty: int, urgency: float, horizon: int) -> None:
        """Almgren-Chriss liquidation of an ASSUMED defaulted-client position.
        The NBCM holds no own trading book, so `inventory` is only ever a position it
        has assumed on a client default; this schedules its disposal over `horizon`
        steps (sell if long, buy if short). Slices are sent to the LOB by Simulation
        (via the CCP, attributed to this NBCM) so the book-walk impact is endogenous."""
        from model.clearing import ac_slices
        queued = sum(self._liq_slices)                       # don't double-schedule
        qty = min(int(abs(qty)), abs(self.inventory) - queued)
        if qty <= 0:
            return
        self._liq_side = -1 if self.inventory > 0 else 1
        self._liq_slices.extend(ac_slices(qty, horizon, urgency))  # EXTEND: clustered
        #                                       client defaults in one cycle must all queue

    def capital_ratio(self, mid: float, sigma_t: float = 0.0) -> float:
        """Basel III CAPITAL ADEQUACY RATIO for a non-banking CM (D78): capital (cash) over
        client-book risk exposure (+ any assumed defaulted-client position still being
        liquidated). `cash/exposure >= 8%` — the same capital-adequacy constraint as the BCM
        (the ODD's cash/|tradePosition|), but on breach the NBCM STOPS OUT (it holds no own book
        to deleverage) — the ODD BCM-deleverage / NBCM-stop-out asymmetry."""
        from model.globals import CONTRACT_USD
        assumed = abs(self.inventory) * self.balance_sheet.volume_lot * CONTRACT_USD * mid
        exposure = self.balance_sheet.client_notional(mid) + assumed
        if exposure <= 0.0:
            return float("inf")
        return self.cash / exposure
