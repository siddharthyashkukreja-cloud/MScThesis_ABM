"""
Simulation driver. V_t is exogenous: loaded from params.fv_csv at init,
indexed per-step into SimContext.v. No GlobalState evolution in-sim.

Step sequence (ODD §Step Sequence, subset active):
  1. Reset LOB step_fills accumulator.
  2. Build SimContext (V_t, prev mid, tick, traders_by_id, last_volume).
  3. Each trader submit_orders.
  4. LOB.match() — call-auction clearing.
  5. Apply fills to buyer/seller inventory + cash.
  6. LOB.age_orders() — hard TTL expiry (D5d).
  7. Mark-to-market PnL on every trader.
  8. Margin cycle every `margin_interval` ticks — variation-margin
     settlement on the clearing tier (D29; ODD §Step Sequence step 11).
  9. Append snapshot to history.

Order lifetime (D5d) is governed by three layers: FT/MT use replace-on-new
(refresh their single standing order on each activation); ZI uses
per-resting Bernoulli cancellation at `zi_delta`; and ALL orders face the
hard `order_ttl` ceiling (10 steps, ODD §Mech #7) enforced by age_orders().
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from .globals import ModelParams, SimContext, CCP_CALIBRATION
from .lob import LOB, Fill


class Simulation:
    def __init__(self, params: ModelParams, traders: list, seed: int = 42,
                 ccp=None):
        self.params = params
        self.lob = LOB(params.tick_size, params.order_ttl)
        self.traders = traders
        self.traders_by_id: Dict[int, object] = {t.agent_id: t for t in traders}
        self.rng = np.random.default_rng(seed)
        # Clearing tier (D28/D29). `ccp` is the CentralCounterparty when the
        # clearing scaffold is wired (run_simulation); None on the
        # calibration path, which runs the bare market layer. When present,
        # the variation-margin cycle (D29) runs every `margin_interval`
        # ticks; clearing_history records one row per CM per cycle.
        self.ccp = ccp
        self._margin_interval = int(CCP_CALIBRATION["margin_interval"])
        self.clearing_history: List[dict] = []
        if self.ccp is not None:
            for cm in self.ccp.members.values():
                cm.balance_sheet._last_mark = params.v0

        # V_t path — the `V_smooth` column of the fv CSV (Stein-Stein SV
        # jump-diffusion, D33). `sigma_t` is the per-step stochastic vol
        # (D33), piped into the FT belief width via SimContext (D34); if
        # absent (legacy fv csv) the SV channel is off and FTs fall back
        # to params.sigma_v.
        df = pd.read_csv(params.fv_csv)
        v_col = "V_smooth" if "V_smooth" in df.columns else df.columns[1]
        self.v_array: np.ndarray = df[v_col].to_numpy()
        self.sigma_t_array = (df["sigma_t"].to_numpy()
                              if "sigma_t" in df.columns else None)

        self.t: int = 0
        self._prev_mid: float = float("nan")
        self._prev_volume: int = 0

        self.history: Dict[str, List] = {
            "t": [], "mid_price": [], "spread": [],
            "bid_depth": [], "ask_depth": [], "volume": [],
            "fundamental": [],
        }

    def _v_at(self, t: int) -> float:
        idx = t if t < len(self.v_array) else len(self.v_array) - 1
        return float(self.v_array[idx])

    def _sigma_at(self, t: int) -> float:
        if self.sigma_t_array is None:
            return float(self.params.sigma_v)
        idx = t if t < len(self.sigma_t_array) else len(self.sigma_t_array) - 1
        return float(self.sigma_t_array[idx])

    def step(self) -> dict:
        self.lob.step_fills = []
        v_now = self._v_at(self.t)
        sigma_now = self._sigma_at(self.t)

        ctx = SimContext(
            v=v_now,
            mid_price=self._prev_mid,
            tick=self.t,
            traders_by_id=self.traders_by_id,
            last_volume=self._prev_volume,
            sigma_t=sigma_now,
        )

        for trader in self.traders:
            trader.submit_orders(self.lob, self.params, ctx, self.rng)

        self.lob.match()
        for f in self.lob.step_fills:
            self._apply_fill(f)
        # Hard TTL ceiling (D5d, ODD §Mech #7): every resting order expires
        # after params.order_ttl steps (10). Layered on top of replace-on-new
        # (FT/MT) and Bernoulli cancellation (ZI).
        self.lob.age_orders()

        snap = self.lob.snapshot()
        mark_price = snap["mid_price"] if not np.isnan(snap["mid_price"]) else v_now
        for tr in self.traders:
            tr.update_pnl(mark_price)

        # Margin cycle (D29, ODD §Step Sequence step 11) — hourly at the
        # 1-min cadence: (t+1) % margin_interval == 0 (ODD §V&V).
        if self.ccp is not None and (self.t + 1) % self._margin_interval == 0:
            self._margin_cycle(mark_price)

        self.history["t"].append(self.t)
        self.history["mid_price"].append(snap["mid_price"])
        self.history["spread"].append(snap["spread"])
        self.history["bid_depth"].append(snap["bid_depth"])
        self.history["ask_depth"].append(snap["ask_depth"])
        self.history["volume"].append(snap["volume"])
        self.history["fundamental"].append(v_now)

        self._prev_mid = snap["mid_price"]
        self._prev_volume = int(snap["volume"])
        self.t += 1
        return snap

    def _apply_fill(self, fill: Fill):
        # Clearing members use futures-margin accounting (D29): a fill moves
        # only the position — no full notional is paid; cash is settled by
        # the variation-margin cycle. Non-CM traders keep nominal cash
        # accounting (inert — it gates nothing).
        buyer = self.traders_by_id.get(fill.buyer_id)
        seller = self.traders_by_id.get(fill.seller_id)
        if buyer is not None:
            buyer.inventory += fill.qty
            if not hasattr(buyer, "balance_sheet"):
                buyer.cash -= fill.price * fill.qty
        if seller is not None:
            seller.inventory -= fill.qty
            if not hasattr(seller, "balance_sheet"):
                seller.cash += fill.price * fill.qty

    def _margin_cycle(self, mid: float):
        """ODD §Step Sequence step 11 / §Mech #3 — hourly variation-margin
        settlement on the clearing tier (D29). For each clearing member:
        settle the cleared book's mark-to-market move since the last cycle
        into cash (VM); recompute initial / maintenance margin off the
        gross exposure; flag a margin call when |VM| exceeds (1-mmPercent)
        of IM; set the default flag on cash exhaustion (the 5-level
        waterfall is deferred). One clearing_history row per CM per cycle."""
        im = CCP_CALIBRATION["im_percent"]
        mm = CCP_CALIBRATION["mm_percent"]
        for cm in self.ccp.members.values():
            bs = cm.balance_sheet
            book_pos = bs.book_position(cm.inventory)
            vm = book_pos * (mid - bs._last_mark)
            cm.cash += vm
            bs.variation_margin += vm
            bs._last_mark = mid

            own_notional = abs(cm.inventory) * mid if bs.is_banking else 0.0
            exposure = own_notional + bs.client_notional(mid)
            bs.initial_margin = im * exposure
            bs.maintenance_margin = mm * bs.initial_margin
            bs.call_indicator = (bs.initial_margin > 0.0
                                 and abs(vm) > (1.0 - mm) * bs.initial_margin)
            if cm.cash <= 0.0 and not bs.has_defaulted:
                bs.has_defaulted = True
                self.ccp.default_list.append(cm.agent_id)

            self.clearing_history.append({
                "t": self.t, "agent_id": cm.agent_id,
                "kind": "BCM" if bs.is_banking else "NBCM",
                "cash": cm.cash, "own_position": cm.inventory,
                "client_notional": bs.client_notional(mid),
                "capital_ratio": cm.capital_ratio(mid),
                "initial_margin": bs.initial_margin,
                "maintenance_margin": bs.maintenance_margin,
                "vm_cycle": vm, "vm_cumulative": bs.variation_margin,
                "call_indicator": bs.call_indicator,
                "has_defaulted": bs.has_defaulted,
            })

    def run(self, n_steps: int) -> Dict[str, List]:
        for _ in range(n_steps):
            self.step()
        return self.history
