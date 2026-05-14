"""
Simulation driver. V_t is exogenous: loaded from params.fv_csv at init,
indexed per-step into SimContext.v. No GlobalState evolution in-sim.

Step sequence (ODD §Step Sequence, subset active):
  1. Reset LOB step_fills accumulator.
  2. Build SimContext (V_t, prev mid, tick, traders_by_id).
  3. Each trader submit_orders.
  4. LOB.match() — call-auction clearing.
  5. Apply fills to buyer/seller inventory + cash.
  6. LOB.age_orders() — TTL decrement / expiry.
  7. Mark-to-market PnL on every trader.
  8. Append snapshot to history.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from .globals import ModelParams, SimContext
from .lob import LOB, Fill


class Simulation:
    def __init__(self, params: ModelParams, traders: list, seed: int = 42):
        self.params = params
        self.lob = LOB(params.tick_size, params.order_ttl)
        self.traders = traders
        self.traders_by_id: Dict[int, object] = {t.agent_id: t for t in traders}
        self.rng = np.random.default_rng(seed)

        df = pd.read_csv(params.fv_csv)
        v_col = "V_smooth" if "V_smooth" in df.columns else df.columns[-1]
        self.v_array: np.ndarray = df[v_col].to_numpy()

        self.t: int = 0
        self._prev_mid: float = float("nan")

        self.history: Dict[str, List] = {
            "t": [], "mid_price": [], "spread": [],
            "bid_depth": [], "ask_depth": [], "volume": [],
            "fundamental": [],
        }

    def _v_at(self, t: int) -> float:
        idx = t if t < len(self.v_array) else len(self.v_array) - 1
        return float(self.v_array[idx])

    def step(self) -> dict:
        self.lob.step_fills = []
        v_now = self._v_at(self.t)

        ctx = SimContext(
            v=v_now,
            mid_price=self._prev_mid,
            tick=self.t,
            traders_by_id=self.traders_by_id,
        )

        for trader in self.traders:
            trader.submit_orders(self.lob, self.params, ctx, self.rng)

        self.lob.match()
        for f in self.lob.step_fills:
            self._apply_fill(f)
        self.lob.age_orders()

        snap = self.lob.snapshot()
        mark_price = snap["mid_price"] if not np.isnan(snap["mid_price"]) else v_now
        for tr in self.traders:
            tr.update_pnl(mark_price)

        self.history["t"].append(self.t)
        self.history["mid_price"].append(snap["mid_price"])
        self.history["spread"].append(snap["spread"])
        self.history["bid_depth"].append(snap["bid_depth"])
        self.history["ask_depth"].append(snap["ask_depth"])
        self.history["volume"].append(snap["volume"])
        self.history["fundamental"].append(v_now)

        self._prev_mid = snap["mid_price"]
        self.t += 1
        return snap

    def _apply_fill(self, fill: Fill):
        buyer = self.traders_by_id.get(fill.buyer_id)
        seller = self.traders_by_id.get(fill.seller_id)
        if buyer is not None:
            buyer.inventory += fill.qty
            buyer.cash -= fill.price * fill.qty
        if seller is not None:
            seller.inventory -= fill.qty
            seller.cash += fill.price * fill.qty

    def run(self, n_steps: int) -> Dict[str, List]:
        for _ in range(n_steps):
            self.step()
        return self.history
