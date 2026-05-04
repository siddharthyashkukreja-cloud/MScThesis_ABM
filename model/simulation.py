"""
Simulation driver -- Stages 1-3 (LOB + ZI + FT + MT + BCM + NBCM).

Step sequence follows the Simudyne ODD §Step Sequence (subset active):
  1. Reset LOB step_fills accumulator.
  2. Build SimContext (V_t, prev mid, EWMA momentum, traders_by_id).
  3. All traders submit_orders (uniform interface; ZI cancellations and
     BCM fire-sale market orders may produce immediate fills here).
  4. MatchingEngine: call-auction LOB.match() clears crossed quotes;
     fills accumulate into lob.step_fills.
  5. Apply all fills (market + match) to buyer/seller inventory + cash.
  6. LOB.age_orders(): TTL decrement / expiry.
  7. Mark-to-market PnL update on every trader.
  8. Record snapshot.
  9. Update EWMA momentum from new mid.
 10. GlobalState advances V_t via Merton (1976) jump-diffusion.
"""

from typing import List, Dict
import numpy as np

from .globals import ModelParams, GlobalState, SimContext
from .lob import LOB, Fill


class Simulation:
    def __init__(self, params: ModelParams, traders: list, seed: int = 42):
        self.params = params
        self.gs = GlobalState(params, seed=seed)
        self.lob = LOB(params.tick_size, params.order_ttl)
        self.traders = traders
        self.traders_by_id: Dict[int, object] = {t.agent_id: t for t in traders}

        self.momentum: float = 0.0
        self._prev_mid: float = float("nan")

        self.history: Dict[str, List] = {
            "t": [], "mid_price": [], "spread": [],
            "bid_depth": [], "ask_depth": [], "volume": [],
            "fundamental": [], "momentum": [],
        }

    def step(self) -> dict:
        self.lob.step_fills = []

        ctx = SimContext(
            v=self.gs.v,
            mid_price=self._prev_mid,
            momentum=self.momentum,
            tick=self.gs.t,
            traders_by_id=self.traders_by_id,
        )

        for trader in self.traders:
            trader.submit_orders(self.lob, self.params, ctx, self.gs.rng)

        self.lob.match()

        for f in self.lob.step_fills:
            self._apply_fill(f)

        self.lob.age_orders()
        snap = self.lob.snapshot()

        mark_price = snap["mid_price"] if not np.isnan(snap["mid_price"]) else self.gs.v
        for t in self.traders:
            t.update_pnl(mark_price)

        self.history["t"].append(self.gs.t)
        self.history["mid_price"].append(snap["mid_price"])
        self.history["spread"].append(snap["spread"])
        self.history["bid_depth"].append(snap["bid_depth"])
        self.history["ask_depth"].append(snap["ask_depth"])
        self.history["volume"].append(snap["volume"])
        self.history["fundamental"].append(self.gs.v)
        self.history["momentum"].append(self.momentum)

        new_mid = snap["mid_price"]
        if (not np.isnan(self._prev_mid)) and (not np.isnan(new_mid)) \
                and self._prev_mid > 0 and new_mid > 0:
            log_ret = float(np.log(new_mid) - np.log(self._prev_mid))
            lam = self.params.mt_lambda_ewma
            self.momentum = lam * self.momentum + (1.0 - lam) * log_ret
        self._prev_mid = new_mid

        self.gs.step()
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
