# mscthesis_abm/model/margin.py
"""
CCP Margin Engine  --  runs every simulation step.

Design principles
-----------------
1. Vectorised over agents using numpy arrays for speed.
2. No agent-specific Python loops in the hot path.
3. All parameters are thesis-justifiable:

   IM rate = 10%  (EMIR RTS 2016/2251 equity derivative floor)
   VM      = full mark-to-market loss (EMIR Art. 41 -- daily VM settlement)
   Default = equity < 0 after VM call; CCP closes position at current price
             (standard CCP waterfall: defaulter pays first).

Usage
-----
    engine = MarginEngine(traders, params)
    result = engine.step(current_price)
    # result is a MarginStepResult with aggregate counts and per-agent arrays.

The engine mutates trader.cash, trader.margin_posted, trader.margin_called,
trader.defaulted in-place.  simulation.py calls engine.step() after inventory
and cash are updated from the trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from model.agents import BaseTrader
from model.globals import ModelParams


@dataclass
class MarginStepResult:
    n_calls:       int           # agents receiving a margin call this step
    n_defaults:    int           # cumulative defaults
    n_new_default: int           # agents newly defaulted this step
    system_equity: float         # sum of equity across all non-defaulted agents
    total_margin:  float         # total collateral posted system-wide
    # Per-agent arrays (length = n_agents, ordered as traders list)
    equity:        np.ndarray = field(default_factory=lambda: np.array([]))
    margin_posted: np.ndarray = field(default_factory=lambda: np.array([]))
    called:        np.ndarray = field(default_factory=lambda: np.array([]))
    defaulted:     np.ndarray = field(default_factory=lambda: np.array([]))


class MarginEngine:
    """
    Vectorised CCP margin engine.

    Parameters
    ----------
    traders   : list of BaseTrader  (all agents in the simulation)
    params    : ModelParams
    """

    def __init__(self, traders: List[BaseTrader], params: ModelParams):
        self.traders   = traders
        self.params    = params
        self.n         = len(traders)

        # Pre-allocate numpy arrays -- reused every step (no allocation in hot path)
        self._inventory = np.zeros(self.n)
        self._cash      = np.zeros(self.n)
        self._entry     = np.zeros(self.n)
        self._posted    = np.zeros(self.n)
        self._defaulted = np.zeros(self.n, dtype=bool)

        for i, t in enumerate(traders):
            self._entry[i]     = t.entry_price
            self._posted[i]    = t.margin_posted
            self._defaulted[i] = t.defaulted

    def step(self, current_price: float) -> MarginStepResult:
        """
        Run one margin step at current_price.

        Steps (all vectorised):
          1. Sync mutable state from trader objects into numpy arrays
          2. Compute notional, IM requirement, VM requirement
          3. Identify shortfalls
          4. Attempt to cover shortfall from cash
          5. Flag defaults (equity < 0 after attempted cover)
          6. Force-close defaulted positions (set inventory=0 at current price)
          7. Write results back to trader objects
        """
        p = current_price

        # 1. Sync from objects (only non-defaulted)
        for i, t in enumerate(self.traders):
            if not self._defaulted[i]:
                self._inventory[i] = t.inventory
                self._cash[i]      = t.cash
                self._entry[i]     = t.entry_price
                self._posted[i]    = t.margin_posted

        # 2. Mark-to-market P&L
        mtm = self._inventory * (p - self._entry)

        # 3. Margin requirements
        notional    = np.abs(self._inventory) * p
        im_required = notional * self.params.im_rate
        # VM = full unrealised loss (EMIR Art. 41 daily cash VM)
        vm_required = np.maximum(-mtm, 0.0)
        total_req   = im_required + vm_required

        # 4. Shortfall and cash cover
        shortfall = np.maximum(total_req - self._posted, 0.0)
        mask_call = (shortfall > 0) & (~self._defaulted)
        equity    = self._cash + mtm
        cover     = np.minimum(shortfall, np.maximum(equity, 0.0))
        self._cash   -= np.where(mask_call, cover, 0.0)
        self._posted += np.where(mask_call, cover, 0.0)
        equity        = self._cash + mtm

        # 5. Default detection
        prev_defaulted   = self._defaulted.copy()
        new_default_mask = (equity < 0) & (~self._defaulted)
        self._defaulted |= new_default_mask

        # 6. Force-close defaulted positions at current price
        #    CCP standard waterfall: defaulter pays first (CPMI-IOSCO 2012)
        if new_default_mask.any():
            for i in np.where(new_default_mask)[0]:
                realised           = self._inventory[i] * (p - self._entry[i])
                self._cash[i]     += realised
                self._inventory[i] = 0.0
                self._posted[i]    = 0.0
                equity[i]          = max(self._cash[i], 0.0)

        # 7. Write back to trader objects
        for i, t in enumerate(self.traders):
            t.cash          = float(self._cash[i])
            t.margin_posted = float(self._posted[i])
            t.mtm_pnl       = float(mtm[i])
            t.equity        = float(equity[i])
            t.margin_called = bool(mask_call[i])
            t.defaulted     = bool(self._defaulted[i])
            if new_default_mask[i]:
                t.inventory        = 0.0
                self._inventory[i] = 0.0

        n_calls   = int(mask_call.sum())
        n_def_tot = int(self._defaulted.sum())
        n_new_def = int(new_default_mask.sum())
        live_mask = ~self._defaulted
        sys_eq    = float(equity[live_mask].sum()) if live_mask.any() else 0.0
        tot_marg  = float(self._posted.sum())

        return MarginStepResult(
            n_calls       = n_calls,
            n_defaults    = n_def_tot,
            n_new_default = n_new_def,
            system_equity = sys_eq,
            total_margin  = tot_marg,
            equity        = equity.copy(),
            margin_posted = self._posted.copy(),
            called        = mask_call.copy(),
            defaulted     = self._defaulted.copy(),
        )
