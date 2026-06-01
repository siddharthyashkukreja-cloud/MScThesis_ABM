"""
Central-clearing tier — the thesis client-clearing extension of the
Simudyne CCP ODD (§Agents, §Interactions, §Initialization Network).

SCAFFOLD ONLY (D28). This module supplies the balance-sheet record, the
central-counterparty registry and the counterparty/link structure. The
mechanics — variation-margin calls, the cover-2 default fund and the
5-level waterfall (ODD §Staging Stage 3-5) — are deferred to later stages.

The ODD's clearing members trade for their own account and have no client
tier; the thesis adds client clearing — the BCM/NBCM carry client books.
Banking CMs trade their own account AND clear clients; non-banking CMs
hold no own position and only clear clients (D28).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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

    def client_notional(self, mid: float) -> float:
        """USD notional of the client book = Σ|client_pos| · mid ·
        VOLUME_LOT · CONTRACT_USD (D50 / Stage 4a). Each model qty unit
        represents a 50-contract institutional ES lot; each contract has a
        $50/point CME multiplier (CME E-mini ES futures contract spec). So
        one model-qty point of exposure = $2,500 of USD notional."""
        from model.globals import VOLUME_LOT, CONTRACT_USD
        gross_contracts = sum(abs(q) for q in self.client_positions.values()) * VOLUME_LOT
        return gross_contracts * mid * CONTRACT_USD

    def book_position(self, own_inventory: int) -> int:
        """Signed net position of the cleared book — own position (banking
        CMs only) plus client positions. The variation-margin base (D29)."""
        own = own_inventory if self.is_banking else 0
        return own + sum(self.client_positions.values())


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

    def register_member(self, cm) -> None:
        """Add a clearing member (BCM or NBCM) to the CCP star topology and
        wire the bidirectional CM<->CCP link (ODD §Initialization Network:
        CMToEXLink / EXToCMLink) — recorded as `cm.ccp_id` on the member
        side and `member_ids` / `members` on the CCP side."""
        self.member_ids.append(cm.agent_id)
        self.members[cm.agent_id] = cm
        cm.ccp_id = self.ccp_id
        cm.balance_sheet.counterparty_id = self.ccp_id
