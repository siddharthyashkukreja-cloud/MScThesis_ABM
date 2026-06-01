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
    # Client-clearing link (D28): id of the clearing member this trader
    # clears through; None for clearing members themselves and for direct
    # exchange participants (ZI stay direct — ODD §Initialization).
    clearing_member_id: Optional[int] = None

    def update_pnl(self, price: float):
        self.pnl = self.inventory * price

    # @property
    # def equity(self) -> float:
    #     return self.cash + self.pnl - self.margin_posted


# ── helpers ──────────────────────────────────────────────────────────────────

def _draw_qty(params: ModelParams, rng: np.random.Generator) -> int:
    """Uniform `U[qty_min, qty_max]` order size — ODD §Stochasticity. D49
    (Pareto) was reverted: with our discrete LOB matching, impact is
    ~linear in qty, so the Gabaix-Plerou (2003) sqrt-impact result didn't
    fire and power-law sizes degraded Hill instead of helping. Volume
    scaling to empirical ES levels is handled by the `VOLUME_LOT`
    relabeling in `globals.py` (D50) — each model qty unit represents
    a 50-contract institutional lot."""
    return int(rng.integers(params.qty_min, params.qty_max + 1))


def _draw_depth(params: ModelParams, rng: np.random.Generator) -> int:
    """Geometric placement depth in ticks, k >= 1 — SHARED by ZI and MT limit
    orders (mid-anchored). k ~ Geometric(p_zi) with the data-fit p_zi (MBP-10
    MLE; globals.P_ZI), reverting the D20 log-normal to the D3 Cont-Stoikov-
    Talreja (2008) geometric. Dense at the mid (mode k=1) keeps the near-mid
    book thick so market orders don't walk a sparse book — the fix for the ~5x
    mid-vs-fundamental kurtosis amplification. p_zi is data-fixed, not calibrated."""
    return int(rng.geometric(params.p_zi))


def _resting_oids(open_oids: list, lob: LOB) -> list:
    """Drop oids that have been filled / expired / cancelled in the book."""
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


def ac_schedule(Q: float, T: int, sigma: float, eta: float, gamma: float,
                lambda_risk: float) -> np.ndarray:
    """Almgren-Chriss (2000) optimal liquidation schedule. Discrete-time,
    linear-impact, constant-coefficient closed form (eqs 18-19).

    Returns array [n_1, n_2, ..., n_T] of per-step trade sizes summing to Q,
    front-loaded according to the AC tradeoff between transaction cost
    (η, γ) and price-variance risk (σ²) under risk aversion λ.

        κ² = λ·σ²/η                (curvature; γ correction omitted — small effect)
        n_k = (2 sinh(κ/2) / sinh(κT)) · cosh(κ(T − (k − ½))) · Q

    κ·T → 0 (low aversion / low impact) → near-linear schedule (≈ Q/T per step)
    κ·T → ∞ (high aversion / high impact) → exponentially front-loaded

    Reused by MM (D10e MM liquidation) and BCM/CCP fire-sale at Stage 4+.
    """
    if Q <= 0 or T < 1:
        return np.array([], dtype=float)
    if T == 1:
        return np.array([float(Q)])
    if eta <= 0 or lambda_risk <= 0:
        return np.full(T, Q / T)              # degenerate: linear
    kappa = float(np.sqrt(lambda_risk * sigma * sigma / eta))
    if kappa * T < 1e-6:                       # near-linear regime
        return np.full(T, Q / T)
    ks = np.arange(1, T + 1, dtype=float)
    n = ((2.0 * np.sinh(kappa / 2.0) / np.sinh(kappa * T))
         * np.cosh(kappa * (T - (ks - 0.5))) * Q)
    # Numerical safety: normalize so the schedule sums exactly to Q.
    total = n.sum()
    if total > 0:
        n = n * (Q / total)
    return n


# ── Zero-Intelligence Trader (Cont-Stoikov 2008) — background noise floor ──

@dataclass
class ZeroIntelligenceTrader(BaseTrader):
    """
    Cont-Stoikov (2008) noise trader — provides continuous background flow
    that smooths per-step price impact between MM quotes and aggressive
    agents. Re-introduced under D14e after dropping (D14b) proved the
    burst-only model produces too-heavy tails and no clustering.

    Per step, three independent Bernoulli draws (ODD-native per-step
    probabilities at the 1-min cadence; no dt rescaling):
      - per-resting limit cancelled w.p. zi_delta
      - submit one limit order   w.p. zi_alpha → random side, depth k from
                                   the shared log-normal `_draw_depth`
                                   (D20), qty ~ U[qty_min, qty_max]
      - submit one market order  w.p. zi_mu    → random side, qty ~ U[…]

    The three Bernoulli rates are fixed at ODD-baseline values; population
    n_zi is structural. The placement depth `depth_mean` is calibrated but
    SHARED with MT (D20) — it is a market-microstructure parameter, not a
    ZI-specific one — so ZI keeps no calibrated parameter of its own.
    """
    _open_oids: List[int] = field(default_factory=list, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        self._open_oids = _bernoulli_cancel(self._open_oids, lob,
                                            params.zi_delta, rng)
        anchor = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        if rng.random() < params.zi_alpha:
            side = 1 if rng.random() < 0.5 else -1
            k = _draw_depth(params, rng)
            price = max(anchor - side * k * params.tick_size, params.tick_size)
            qty = _draw_qty(params, rng)
            oid = lob.add_limit(self.agent_id, side, price, qty)
            self._open_oids.append(oid)
        if rng.random() < params.zi_mu:
            side = 1 if rng.random() < 0.5 else -1
            qty = _draw_qty(params, rng)
            lob.add_market(self.agent_id, side, qty)


# ── Fundamental Trader (ODD §Agents) ─────────────────────────────────────────

@dataclass
class FundamentalTrader(BaseTrader):
    """
    Per agent z_score ~ N(0,1) fixed at init (persistent heterogeneous beliefs
    — Chiarella-Iori-Perelló; see D6b). Per step:
      σ_fundamental_t = ft_sigma_c · σ_t · v0    (D34 — stochastic; σ_t is
                        the current Vasicek-OU SV from D33, read off
                        SimContext; falls back to params.sigma_v when SV
                        is off. ft_sigma_c pinned at √390 — one daily
                        scale; see globals.FT_SIGMA_C_DEFAULT)
      reservation     = V_t + z_score · σ_fundamental_t
      side            = sign(reservation − mid)
      activation      = Bernoulli(ft_alpha)
      on activation: REPLACE-ON-NEW (D5d) — cancel the FT's standing limit
                     (if still resting), place one fresh limit at the
                     reservation, qty ~ U[qty_min, qty_max].

    There is NO dead-band (D23 — removed): the persistent z_score already
    supplies the FT heterogeneity, so the FT acts whenever its reservation
    differs from the mid. (A flat `ft_threshold_bps` band and a wide
    per-agent Simudyne band U[0.01·V, 0.10·V] were both trialled and
    dropped — D9b/D22/D23.)

    Order management is replace-on-new: the FT holds at most one resting
    limit, refreshed on the next activation; otherwise it persists until the
    hard order_ttl ceiling (10 steps). No per-resting cancellation rate.
    """
    z_score: float = 0.0
    _open_oid: Optional[int] = field(default=None, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        # Drop the standing-order reference if it was filled / TTL-expired.
        if self._open_oid is not None and not lob.is_resting(self._open_oid):
            self._open_oid = None

        ref = ctx.mid_price if not np.isnan(ctx.mid_price) else ctx.v
        # D34 — FT belief width tracks the stochastic V_t volatility σ_t
        # (Deloitte convention: `theta_v = sigma_fundamental`). When σ_t
        # is high, the FT reservation cloud widens (less reactive); when
        # low, it tightens (more reactive). This is the channel through
        # which the SV-V_t clustering (D33) reaches the mid.
        sigma_t = ctx.sigma_t if ctx.sigma_t > 0.0 else params.sigma_v
        sigma_fund_t = params.ft_sigma_c * sigma_t * params.v0
        reservation = ctx.v + self.z_score * sigma_fund_t
        reservation = max(reservation, params.tick_size)

        # Side from the sign of the FT's own mispricing (D9c: the per-agent
        # reservation, not the collective V_t, keeps flow two-sided). No
        # dead-band — z_score supplies the heterogeneity (D23).
        diff = reservation - ref
        if diff > 0:
            side = 1
        elif diff < 0:
            side = -1
        else:
            return

        # D36 — FT trades every step (ODD §Step Sequence step 3: "CMs
        # submit BuyOrder / SellOrder messages based on capital ratio check
        # and limit price vs market price comparison"). Replace-on-new
        # without a Bernoulli gate — `ft_alpha` is pinned at 1.0 and out of
        # the calibration loop.
        if self._open_oid is not None:
            lob.cancel(self._open_oid)
        qty = _draw_qty(params, rng)
        self._open_oid = lob.add_limit(self.agent_id, side, reservation, qty)


# ── Momentum Trader (chartist — single-type, D13f; market branch, D27) ──────

@dataclass
class MomentumTrader(BaseTrader):
    """
    Single-type EWMA chartist (D13f — folded back from the D13e two-cohort
    long/short split once long-horizon volatility clustering was conceded
    as structurally unreachable, D18f). EWMA momentum on mid log-returns:

        r_t = log(mid_{t-1}) − log(mid_{t-2})
        M_t = (1 − mt_lambda) · M_{t-1} + mt_lambda · r_t

    `mt_lambda` is shared by every MT and PINNED at 0.05 (D44 — out of the
    calibration loop). Note on the timescale: Majewski et al. (2018) FIX the
    trend horizon externally (α = 1/7, τ = 6 months, from a CTA-index
    correlation) rather than estimating it, so pinning `mt_lambda` is
    consistent with their practice (the earlier "estimated from data" wording
    was a misreading and has been corrected).

    `M_t` must clear a tiny floor `mt_eps` (skips the EWMA warm-up); the
    sign of M_t fixes the order SIDE. The MT is LIMIT-ONLY (D40 — the D27
    market branch was reverted; trend-direction market flow corrupted the
    return ACF). `mt_mu` stays on ModelParams at 0.0 for back-compat but is
    unused. The MT trades every step (D36 — `mt_alpha` pinned at 1.0):

      limit (every step) — passive quote, mid-anchored:
        k     ~ LogNormal placement depth (_draw_depth — shared with ZI, D20)
        price = mid - side · k · tick   (buy below mid, sell above)

    The placement depth is the shared log-normal `_draw_depth` (D20 — the
    prior signal-driven depth k_base*sigma_v/|M_t| was removed; `depth_mean`
    and `depth_sigma` are calibrated and shared with ZI). The limit branch is
    REPLACE-ON-NEW (D5d) — at most one standing limit, cancelled and replaced
    each step.

    Single cohort (D44 — the D35 two-cohort long/short split was dropped;
    `n_momentum_long = 0`). Each MT still carries its own `lambda_decay`
    field (per-agent init kwarg) so a long cohort can be re-enabled by
    setting `n_momentum_long > 0`, but at runtime all MTs use `mt_lambda`.
    """
    lambda_decay: float = 0.1   # per-agent EWMA decay; set at construction (D35)
    _M: float = field(default=0.0, init=False, repr=False)
    _prev_mid: float = field(default=float("nan"), init=False, repr=False)
    _open_oid: Optional[int] = field(default=None, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
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

        am = abs(self._M)
        if am < params.mt_eps:
            return
        side = 1 if self._M > 0 else -1

        # Market branch removed (D40 — reverts D27). MT is limit-only;
        # `mt_mu` no longer used (kept on ModelParams default 0.0 for
        # backward compat; out of PARAM_BOUNDS).

        # Limit branch — passive mid-anchored quote, posted every step
        # (D36 — `mt_alpha` pinned at 1.0 and out of the calibration loop).
        anchor = cur_mid if not np.isnan(cur_mid) else ctx.v
        k = _draw_depth(params, rng)          # shared log-normal depth (D20)
        price = anchor - side * k * params.tick_size
        price = max(price, params.tick_size)
        qty = _draw_qty(params, rng)
        # Replace-on-new: cancel the standing order, place a fresh one.
        if self._open_oid is not None:
            lob.cancel(self._open_oid)
        self._open_oid = lob.add_limit(self.agent_id, side, price, qty)


# ── Cont threshold trader (Cont 2005 §4.1 — D39; DORMANT: n_ct=0, removed D44) ─

@dataclass
class ContTrader(BaseTrader):
    """
    Cont 2005 §4 threshold-with-inertia trader (D39). Each agent carries
    its own threshold `θ_i(t)` representing its subjective view on
    volatility. Per step:

        ε_t = log V_t − log V_{t-1}          (common news signal)
        if |ε_t| > θ_i(t):  market order, side = sign(ε_t), qty = ct_qty
        else:               inactive
        with probability `ct_update_prob`:  θ_i(t+1) = |r_mid(t)|

    The asynchronous updating creates **heavy-tailed durations** of
    inactivity vs activity regimes — Cont's mechanism for long-memory
    |r| ACF (§3.4: Markov SV alone gives only short-range clustering;
    long-range needs renewal switching with heavy-tailed regime durations).
    At low `ct_update_prob` (~0.05) some agents hold a stale threshold for
    many steps; if it's high they stay inactive across multiple regimes
    of V_t, if it's low they fire repeatedly — producing the persistence
    that direct vol-scaled noise (D38) cannot.

    Directional (not random): when V_t innovates up, all ContTraders that
    fire trade up. Couples FT (level-based, V_t + z·σ_fund) with a
    rate-of-change signal — V_t innovation drives ContTrader, V_t level
    drives FT.

    Initial `θ_i(0)` set per-agent at construction (U[0, 2·σ_v]).
    """
    threshold: float = 0.0
    _prev_mid: float = field(default=float("nan"), init=False, repr=False)
    _prev_v:   float = field(default=float("nan"), init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        # Common news signal — V_t log-return innovation
        if (not np.isnan(self._prev_v) and ctx.v > 0 and self._prev_v > 0):
            eps_t = float(np.log(ctx.v) - np.log(self._prev_v))
        else:
            eps_t = 0.0
        self._prev_v = ctx.v

        # Recent |r_mid| for threshold update
        cur_mid = ctx.mid_price
        r_mid_abs = 0.0
        if (not np.isnan(cur_mid) and not np.isnan(self._prev_mid)
                and cur_mid > 0 and self._prev_mid > 0):
            r_mid_abs = float(abs(np.log(cur_mid) - np.log(self._prev_mid)))
        if not np.isnan(cur_mid):
            self._prev_mid = cur_mid

        # Asynchronous threshold update — Cont §4.1
        if rng.random() < params.ct_update_prob:
            self.threshold = r_mid_abs

        # Trade if external signal exceeds threshold
        if abs(eps_t) > self.threshold:
            side = +1 if eps_t > 0 else -1
            qty = max(1, int(round(params.ct_qty)))
            lob.add_market(self.agent_id, side, qty)


# ── Volatility Trader (Gao et al. 2023 §3.1.3 — D38; DORMANT: n_vt=0, removed D44) ─

@dataclass
class VolatilityTrader(BaseTrader):
    """
    Gao 2023 Chiarella-Heston volatility trader (D38). Submits a market
    order every step in random direction (±1, equiprobable) with quantity
    scaled by the current `σ_t / params.sigma_v` — the LOB-form of Gao's
    `D^vol(t) = ω·√Σ_t · dW_t^S` continuous-time demand. Vol-scaled noise
    directly on the mid: when V_t's stochastic vol is high, VT market-order
    flow is correspondingly larger; when V_t vol is low, smaller. This is
    the channel through which the D33 Vasicek-OU σ_t couples to mid-return
    magnitude, propagating clustering to the mid (Cont 2005 §3.3 / Gao §3).

    Re-introduces the D24/D25 VolatilityTrader trial under the new context.
    Previous attempts failed because the D10g MM at `mm_qty = 50` clamped
    spread and absorbed VT impact; under D37 (no MM, dense FT/MT-every-step
    book) the spread widens with σ_t and VT market orders carry through.

    Parameters (all structural at this stage; consider calibrating
    `vt_qty_base` later):
        vt_qty_base   — base quantity at `σ_t = sigma_v` (1-min total vol).
    """
    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        sigma_now = ctx.sigma_t if ctx.sigma_t > 0 else params.sigma_v
        scale = sigma_now / params.sigma_v
        qty = max(1, int(round(params.vt_qty_base * scale)))
        side = +1 if rng.random() < 0.5 else -1
        lob.add_market(self.agent_id, side, qty)


# ── Market Maker (HFABM Gao et al. 2022 §3.6 — mid-anchored, D31; n_mm=4 D48) ─

@dataclass
class MarketMaker(BaseTrader):
    """
    HFABM-style **mid-anchored** market maker (D31 — supersedes both the
    D10g V_t-anchored design and the D21 MM drop). Per step:

      1. Cancel all MM resting quotes from the prior step.
      2. Quote a bid + ask around the prev-step mid (v0 fallback at t=0),
         at a random per-side tick offset:
           anchor      = mid_price (v0 at t=0)
           d_bid, d_ask ~ U{0, ..., mm_p_edge}   (independent)
           bid         = anchor − d_bid · tick_size
           ask         = anchor + d_ask · tick_size
         qty = mm_qty on each side. No inventory skew (HFABM Gao et al.
         2022 §3.6 convention). The MM ALWAYS quotes — no going dark.
         (Gao et al.'s MM also has an inventory-limit "hot-potato" regime
         switch — central to their flash-crash study — which is NOT modelled
         here; the always-quote variant is used instead.)

    The MM is the kurtosis-fix mechanism (D31). The no-MM market layer
    (D21–D30) produced a static-then-jump return pattern → kurt ~100–500;
    a low-qty mid-anchored MM provides continuous near-mid liquidity, so
    market orders no longer walk a sparse book in one go and small
    inter-jump price movement is restored — `kurt` drops 2–10× toward the
    empirical level. Sweep evidence (prototype): 2 HFABM MMs @ `mm_qty=1`
    cut calm kurt 126→44 and lifted Hill 2.49→3.10 (essentially the target
    3.00), at the cost of a small bid-ask bounce in `acf_r_1` (~−0.04).

    D10g's V_t anchor pinned mid to V_t (degenerate); D21 dropped the MM
    entirely. D31 splits the difference: **mid-anchored** (no V_t pin),
    low `mm_qty` so the MM provides liquidity without dominating price
    formation. As of D48 `mm_qty = 2` is PINNED structural (globals.MM_QTY)
    and `n_mm = 4`; the calibrated MM dial is `mm_p_edge` (the spread-width
    ceiling) — this supersedes the D31 "`mm_qty` calibrated" convention.
    Inventory skew is dropped (Skew + low qty produced positive `acf_r_1`
    in the prototype sweep); without skew the MM may drift in inventory
    over long runs — a soft skew can be re-added if needed. The POV /
    Almgren-Chriss helpers (`ac_schedule`, `mm_pov`, `mm_inventory_*`)
    stay reserved for the Stage-4+ BCM fire-sale.
    """
    _open_oids: List[int] = field(default_factory=list, init=False, repr=False)

    def submit_orders(self, lob: LOB, params: ModelParams,
                      ctx: SimContext, rng: np.random.Generator):
        for oid in self._open_oids:
            lob.cancel(oid)
        self._open_oids = []

        # Mid-anchored: no V_t pin, no inventory skew (D31). `mm_p_edge` is
        # CALIBRATED (D48) as a float; quantise to integer ticks for the
        # uniform draw bound (rng.integers requires an integer endpoint).
        anchor = ctx.mid_price if not np.isnan(ctx.mid_price) else params.v0
        p_edge = max(1, int(round(params.mm_p_edge)))
        d_bid = int(rng.integers(0, p_edge + 1))
        d_ask = int(rng.integers(0, p_edge + 1))
        bid_px = max(anchor - d_bid * params.tick_size, params.tick_size)
        ask_px = anchor + d_ask * params.tick_size
        qty = max(1, int(round(params.mm_qty)))   # pinned (D48); quantise to integer
        oid_b = lob.add_limit(self.agent_id, +1, bid_px, qty)
        oid_a = lob.add_limit(self.agent_id, -1, ask_px, qty)
        self._open_oids = [oid_b, oid_a]


# A VolatilityTrader was trialled here (D24) — an endogenous-volatility
# noise trader scaling its order flow with the model's realised vol σ̂_t.
# It did not produce clustering: random market orders wash out on
# aggregation, vol-scaled limit orders add depth and damp. Removed (D25);
# volatility clustering now comes from the Merton jumps in V_t (data/v_gbm.py).


# ── Clearing tier (D28 — thesis client-clearing extension of the ODD) ───────

@dataclass
class BankingClearingMember(FundamentalTrader):
    """
    Banking clearing member (ODD §Agents BankingClearingMember). Cast from a
    FundamentalTrader: the ODD's CMs price via z_score·σ_fundamental offset
    from the fundamental signal — identical to the thesis FT — so the BCM
    trades its OWN account through the inherited FT `submit_orders`. At the
    pre-margin scaffold stage its order flow is exactly an FT's, so the 8-d
    market-layer calibration is unchanged (D28).

    Beyond own-account trading the BCM also CLEARS a client book (thesis
    client-clearing extension). It carries a `BalanceSheet`, a CCP
    membership link (`ccp_id`) and a `client_ids` list. The ODD §Mech #2
    capital-adequacy fire-sale is deferred to the stress stage.
    """
    balance_sheet: Optional[BalanceSheet] = None
    ccp_id: Optional[int] = None
    client_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.balance_sheet is None:
            self.balance_sheet = BalanceSheet(owner_id=self.agent_id,
                                              is_banking=True)

    def capital_ratio(self, mid: float) -> float:
        """ODD §Mech #2 capital-adequacy ratio: cash / USD notional exposure
        (D50). Exposure = |own position|·VOLUME_LOT·CONTRACT_USD·mid +
        client_notional (already USD). The 8% floor drives the deferred
        fire-sale; with the D50 institutional-lot relabeling, this ratio
        now sits in a regime where the floor can plausibly bind."""
        from model.globals import VOLUME_LOT, CONTRACT_USD
        own_notional = abs(self.inventory) * VOLUME_LOT * CONTRACT_USD * mid
        exposure = own_notional + self.balance_sheet.client_notional(mid)
        return float("inf") if exposure <= 0.0 else self.cash / exposure


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

    def __post_init__(self):
        if self.balance_sheet is None:
            self.balance_sheet = BalanceSheet(owner_id=self.agent_id,
                                              is_banking=False)

    def capital_ratio(self, mid: float) -> float:
        """ODD §Mech #2 for a non-banking CM: cash / USD client-book
        notional (no own position; `client_notional` is already USD via
        D50 — VOLUME_LOT · CONTRACT_USD · mid · qty)."""
        exposure = self.balance_sheet.client_notional(mid)
        return float("inf") if exposure <= 0.0 else self.cash / exposure
