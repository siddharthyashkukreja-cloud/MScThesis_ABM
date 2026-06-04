"""
Central-clearing tier — the thesis client-clearing extension of the
Simudyne CCP ODD (§Agents, §Interactions, §Initialization Network).

This module supplies the per-member balance-sheet record, the central-
counterparty registry and the counterparty/link structure, plus the
default-management mechanics: the cover-2 default fund (recompute_default_fund),
the five-level waterfall (default_waterfall) and the Almgren-Chriss fire-sale
schedule (ac_slices / start_firesale; ODD §Staging Stage 3-5). The variation-
margin cycle that drives them lives in simulation.py.

The ODD's clearing members trade for their own account and have no client
tier; the thesis adds client clearing — the BCM/NBCM carry client books.
Banking CMs trade their own account AND clear clients; non-banking CMs
hold no own position and only clear clients (D28).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


def ac_slices(total: int, horizon: int, urgency: float) -> List[int]:
    """Almgren-Chriss (2000) liquidation schedule — split `total` units over
    `horizon` steps. The remaining-inventory trajectory is
    x_j / X = sinh(κ(H−j)) / sinh(κH) with κH = `urgency`; the per-step slices
    are its decrements. urgency→0 is TWAP (the risk-neutral limit); larger
    urgency front-loads the sale (risk-averse / distressed). Returns int slices
    summing to `total`. Each slice's temporary impact is realised endogenously
    when it is sent as a market order to the LOB (the book walk)."""
    total = int(abs(total)); H = max(1, int(horizon))
    if total <= 0:
        return []
    if urgency <= 1e-6:
        x = [total * (1.0 - j / H) for j in range(H + 1)]            # TWAP
    else:
        s = math.sinh(urgency)
        x = [total * math.sinh(urgency * (1.0 - j / H)) / s for j in range(H + 1)]
    slices = [max(0, int(round(x[j] - x[j + 1]))) for j in range(H)]
    drift = total - sum(slices)
    if drift:
        slices[0] += drift
    return [s_ for s_ in slices if s_ > 0] or [total]


@dataclass
class BalanceSheet:
    """Per-clearing-member balance-sheet record (ODD §Agents `balanceSheet`;
    the §Interactions `CMBalanceSheet` message snapshots it). Cash, own
    position and PnL live on the owning agent (BaseTrader); this record
    carries the clearing-specific state. The margin / default-fund fields
    are scaffolded at zero — populated once those stages are built."""
    owner_id: int
    is_banking: bool                                 # BCM trades own account; NBCM does not
    counterparty_id: Optional[int] = None            # the CCP, post-novation (ODD star topology)
    client_positions: Dict[int, int] = field(default_factory=dict)  # client_id -> net qty
    initial_margin: float = 0.0                      # ODD imPercent — Stage: margin
    maintenance_margin: float = 0.0                  # ODD mmPercent — Stage: margin
    variation_margin: float = 0.0                    # cumulative VM settled — Stage: margin
    df_contribution: float = 0.0                     # ODD dfAmount — Stage: default fund
    call_indicator: bool = False                     # margin call this cycle (D29)
    has_defaulted: bool = False                      # ODD hasDefaulted flag — Stage: waterfall
    _last_mark: float = 0.0                          # price the book was last VM-settled at (D29)
    volume_lot: float = 50.0                         # contracts per model lot (regime-specific; set at build)

    def client_notional(self, mid: float) -> float:
        """USD notional of the client book = Σ|client_pos| · mid ·
        VOLUME_LOT · CONTRACT_USD (D50 / Stage 4a). Each model qty unit
        represents a 50-contract institutional ES lot; each contract has a
        $50/point CME multiplier (CME E-mini ES futures contract spec). So
        one model-qty point of exposure = $2,500 of USD notional."""
        from model.globals import CONTRACT_USD
        gross_contracts = sum(abs(q) for q in self.client_positions.values()) * self.volume_lot
        return gross_contracts * mid * CONTRACT_USD

    def book_position(self, own_inventory: int) -> int:
        """Signed net position of the cleared book — own position plus client
        positions. The variation-margin base (D29). `own_inventory` is the BCM's
        own-account book or, for an NBCM, a defaulted-client position it has
        assumed and is liquidating (D55); it is 0 for an NBCM in normal operation."""
        return own_inventory + sum(self.client_positions.values())


@dataclass
class CentralCounterparty:
    """Single central counterparty (the clearing role of the ODD
    MatchingEngine; order matching itself stays in LOB). After novation the
    CCP is the counterparty to every clearing member — the ODD §Scales star
    topology, wired by the bidirectional CMToEX / EXToCM links. Holds the
    member registry and the default-fund accounts; the cover-2 sizing and
    the 5-level waterfall are deferred to later stages (ODD §Staging)."""
    ccp_id: int
    cash: float                                      # ODD MatchingEngine cash
    own_df: float = 0.0                              # exchange skin-in-the-game (ODD exDFRatio)
    total_df: float = 0.0                            # pooled member default fund (ODD cover-2)
    member_ids: List[int] = field(default_factory=list)
    members: Dict[int, object] = field(default_factory=dict, repr=False)  # id -> CM agent
    default_list: List[int] = field(default_factory=list)
    _liq_slices: List[int] = field(default_factory=list, repr=False)  # AC fire-sale queue
    _liq_side: int = 0                                                # +1 buy / -1 sell to close

    def register_member(self, cm) -> None:
        """Add a clearing member (BCM or NBCM) to the CCP star topology and
        wire the bidirectional CM<->CCP link (ODD §Initialization Network:
        CMToEXLink / EXToCMLink) — recorded as `cm.ccp_id` on the member
        side and `member_ids` / `members` on the CCP side."""
        self.member_ids.append(cm.agent_id)
        self.members[cm.agent_id] = cm
        cm.ccp_id = self.ccp_id
        cm.balance_sheet.counterparty_id = self.ccp_id

    def start_firesale(self, net_position: int, urgency: float, horizon: int) -> None:
        """Take over a defaulted member's net cleared book and schedule its
        Almgren-Chriss liquidation (Stage C). net_position > 0 (CCP inherits a
        long) -> sell to close (side -1); < 0 -> buy (+1). Slices execute one
        per step in Simulation.step via LOB market orders (endogenous impact)."""
        net_position = int(net_position)
        if net_position == 0:
            return
        self._liq_side = -1 if net_position > 0 else 1
        self._liq_slices.extend(ac_slices(net_position, horizon, urgency))

    def next_firesale_slice(self) -> tuple:
        """(side, qty) of the next pending fire-sale slice, or (0, 0) if none."""
        if not self._liq_slices:
            return 0, 0
        return self._liq_side, int(self._liq_slices.pop(0))

    def recompute_default_fund(self, mid: float, usd: float, sigma_t: float,
                               ex_df_ratio: float, cover_number: int,
                               buffer: float) -> None:
        """Cover-2 default fund via the Stress Loss Over Initial Margins (SLOIM)
        method — Euronext Clearing module A9 §3 / EMIR cover-2 (D55, supersedes the
        flat df_percent·notional haircut). For each active member,
            SLOIM = max(0, stress_loss − posted IM)
        under an extreme-but-plausible stress move (globals.df_stress_move); with a
        common margin fraction this is max(0, (stress_move − im_frac))·|net book|·USD.
        The fund covers the `cover_number` most exposed members' SLOIM, plus a buffer:
            Total DF = (Σ top-2 SLOIM) · (1 + buffer)   [Euronext A9 §3, buffer 10%].
        The exchange pre-funds ex_df_ratio as skin-in-the-game (bounded by its own
        cash); members fund the remainder pro-rata by IM share (= notional share here),
        the ENXC contribution-quota basis CQx ∝ IMx. Recomputed every df_interval."""
        from model.globals import im_fraction, df_stress_move
        im_frac = im_fraction(sigma_t)
        stress_move = df_stress_move(sigma_t)
        active = [cm for cm in self.members.values()
                  if not cm.balance_sheet.has_defaulted]
        notionals = [abs(cm.balance_sheet.book_position(cm.inventory)) * usd * mid
                     for cm in active]
        sloim = sorted((max(0.0, (stress_move - im_frac) * n) for n in notionals),
                       reverse=True)
        self.total_df = float(sum(sloim[:cover_number])) * (1.0 + buffer)
        # SITG is the exchange's first-loss contribution, but bounded by the CCP's
        # own prefunded capital — it cannot pledge more skin-in-the-game than it holds.
        self.own_df = min(ex_df_ratio * self.total_df, max(0.0, self.cash))
        member_pool = self.total_df - self.own_df
        total_pos = sum(notionals) or 1.0
        for cm, n in zip(active, notionals):
            cm.balance_sheet.df_contribution = member_pool * n / total_pos

    def default_waterfall(self, dcm, deficit: float) -> int:
        """ODD §Mech #5 five-level waterfall on a member default. `deficit` is the
        defaulted CM's unpaid cash shortfall — the loss beyond its own capital that
        it owes the CCP but cannot pay (its VM losses + absorbed client losses that
        drove cash < 0). The AC fire-sale separately disposes of the book with real
        market impact (transmitted via the next mark), so the mutualised loss is the
        realized deficit, NOT a flat (1-recovery)·notional haircut — the haircut
        overstated close-out losses by ~40% of gross notional and made a moderate
        crash insolvent the entire CCP (D55 fix). Absorbed sequentially: L1
        defaulted-CM DF contribution, L2 exchange SITG, L3 pooled surviving-member
        DF, L4 surviving-CM cash pro-rata (mutualised loss-sharing — the contagion
        channel), L5 exchange (CCP cash). Returns the deepest level reached."""
        bs = dcm.balance_sheet
        loss = max(0.0, deficit)
        survivors = [cm for cm in self.members.values()
                     if not cm.balance_sheet.has_defaulted and cm.agent_id != dcm.agent_id]
        level = 0
        # L1 — defaulted CM's own pre-funded DF contribution (IM is not escrowed
        # separately from cash in the model, so the deficit already nets it out)
        if loss > 0:
            level = 1
            use = min(loss, bs.df_contribution); bs.df_contribution -= use; loss -= use
        # L2 — exchange skin-in-the-game
        if loss > 0:
            level = 2
            use = min(loss, self.own_df); self.own_df -= use; self.cash -= use; loss -= use
        # L3 — pooled surviving-member default fund
        if loss > 0:
            contribs = [(cm, cm.balance_sheet.df_contribution) for cm in survivors]
            pool = sum(c for _, c in contribs)
            if pool > 0:
                level = 3
                use = min(loss, pool)
                for cm, c in contribs:
                    cm.balance_sheet.df_contribution -= use * c / pool
                loss -= use
        # L4 — surviving-member cash pro-rata (mutualisation → contagion)
        if loss > 0:
            weights = [(cm, max(0.0, cm.cash)) for cm in survivors]
            tot = sum(w for _, w in weights)
            if tot > 0:
                level = 4
                use = min(loss, tot)
                for cm, w in weights:
                    cm.cash -= use * w / tot
                loss -= use
        # L5 — exchange (CCP cash) absorbs the remainder
        if loss > 0:
            level = 5
            self.cash -= loss
        return level
