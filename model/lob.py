"""
Discrete-time call-auction limit order book.

Design
------
- Single-asset, price-time priority, bids/asks stored as sorted dicts.
- Call auction: orders accumulate during a step, then clear once.
- Resting orders leave the book only by fill or explicit cancellation — there is
  no blanket time-to-live. Order lifetime is governed by the agents: ZI cancels each
  resting order w.p. `zi_delta` per step (Cont-Stoikov-Talreja 2008 / Farmer et al.),
  FT/MT are replace-on-new.
- Mid-price = (best_bid + best_ask) / 2 after each auction.
- Spread and depth are outputs, not inputs -- they emerge from order flow.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class Order:
    order_id: int
    agent_id: int
    side: int        # +1 buy, -1 sell
    price: float
    qty: int


@dataclass
class Fill:
    buyer_id: int
    seller_id: int
    price: float
    qty: int


class LOB:
    def __init__(self, tick_size: float):
        self.tick_size = tick_size

        self._bids: Dict[float, List[Order]] = {}
        self._asks: Dict[float, List[Order]] = {}
        self._order_index: Dict[int, tuple] = {}  # oid -> (side, price) for cancel

        self.mid_price: float = np.nan
        self.best_bid: float = np.nan
        self.best_ask: float = np.nan
        self.spread: float = np.nan

        self._next_id = 0
        self.step_fills: List[Fill] = []
        self.last_price: float = np.nan   # most recent fill price (mid fallback)

    # -- Order submission -------------------------------------------------

    def add_limit(self, agent_id: int, side: int, price: float, qty: int) -> int:
        price = self._round(price)
        oid = self._next_id; self._next_id += 1
        order = Order(oid, agent_id, side, price, qty)
        book = self._bids if side == 1 else self._asks
        book.setdefault(price, []).append(order)
        self._order_index[oid] = (side, price)
        return oid

    def is_resting(self, order_id: int) -> bool:
        """True if the order is still in the book (not filled or cancelled)."""
        return order_id in self._order_index

    def add_market(self, agent_id: int, side: int, qty: int) -> List[Fill]:
        """Market order executes immediately against resting quotes."""
        fills = []
        remaining = qty
        if side == 1:          # buy hits asks
            for px in sorted(self._asks):
                if remaining <= 0:
                    break
                queue = self._asks[px]
                i = 0
                while i < len(queue) and remaining > 0:
                    o = queue[i]
                    traded = min(o.qty, remaining)
                    fills.append(Fill(agent_id, o.agent_id, px, traded))
                    o.qty -= traded
                    remaining -= traded
                    if o.qty == 0:
                        self._order_index.pop(o.order_id, None)
                        queue.pop(i)
                    else:
                        i += 1
                if not queue:
                    del self._asks[px]
        else:                  # sell hits bids
            for px in sorted(self._bids, reverse=True):
                if remaining <= 0:
                    break
                queue = self._bids[px]
                i = 0
                while i < len(queue) and remaining > 0:
                    o = queue[i]
                    traded = min(o.qty, remaining)
                    fills.append(Fill(o.agent_id, agent_id, px, traded))
                    o.qty -= traded
                    remaining -= traded
                    if o.qty == 0:
                        self._order_index.pop(o.order_id, None)
                        queue.pop(i)
                    else:
                        i += 1
                if not queue:
                    del self._bids[px]
        self.step_fills.extend(fills)
        return fills

    def cancel(self, order_id: int):
        """Remove a resting limit order by ID. No-op if already filled."""
        if order_id not in self._order_index:
            return
        side, price = self._order_index.pop(order_id)
        book = self._bids if side == 1 else self._asks
        if price in book:
            book[price] = [o for o in book[price] if o.order_id != order_id]
            if not book[price]:
                del book[price]

    def reprice(self, factor: float):
        """Scale every resting order's price by `factor` — an overnight session
        gap. The book reopens at the gapped level with its shape (depth, order ids,
        sizes) intact, so the new RTH day starts at the gapped V_t with a normal
        spread: no empty-book warm-up. Relative bid/ask ordering is preserved, so no
        new crosses."""
        if not (factor == factor) or factor <= 0:
            return
        for name in ("_bids", "_asks"):
            new: Dict[float, List[Order]] = {}
            for px, queue in getattr(self, name).items():
                npx = self._round(px * factor)
                for o in queue:
                    o.price = npx
                new.setdefault(npx, []).extend(queue)
            setattr(self, name, new)
        self._order_index = {o.order_id: (o.side, o.price)
                             for bk in (self._bids, self._asks)
                             for q in bk.values() for o in q}
        for a in ("best_bid", "best_ask", "mid_price", "last_price"):
            v = getattr(self, a)
            if v == v:                      # not NaN
                setattr(self, a, v * factor)
        self.spread = (self.best_ask - self.best_bid
                       if (self.best_bid == self.best_bid and self.best_ask == self.best_ask)
                       else np.nan)

    # -- Call auction -----------------------------------------------------

    def match(self) -> List[Fill]:
        """
        Price-time priority call auction.
        Crosses all buy orders >= best ask against ask queue.
        Updates mid_price, best_bid, best_ask after clearing.
        """
        fills = []
        # Uniform clearing price: one price per step (the pre-auction mid of the crossing book),
        # so every fill this minute prints at a single price. Dampens the directional bounce from
        # pricing each cross at the resting ask; a large one-sided order moves the post-auction mid
        # through the residual book rather than walking the queue at execution.
        clear_px = ((max(self._bids) + min(self._asks)) / 2.0
                    if self._bids and self._asks and max(self._bids) >= min(self._asks) else np.nan)
        while self._bids and self._asks:
            best_bid_px = max(self._bids)
            best_ask_px = min(self._asks)
            if best_bid_px < best_ask_px:
                break
            bid_queue = self._bids[best_bid_px]
            ask_queue = self._asks[best_ask_px]
            b = bid_queue[0]
            a = ask_queue[0]
            traded = min(b.qty, a.qty)
            fills.append(Fill(b.agent_id, a.agent_id, clear_px, traded))
            b.qty -= traded
            a.qty -= traded
            if b.qty == 0:
                self._order_index.pop(b.order_id, None)
                bid_queue.pop(0)
                if not bid_queue:
                    del self._bids[best_bid_px]
            if a.qty == 0:
                self._order_index.pop(a.order_id, None)
                ask_queue.pop(0)
                if not ask_queue:
                    del self._asks[best_ask_px]

        self.step_fills.extend(fills)
        self._update_quotes()
        return fills

    # -- Observables ------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "mid_price": self.mid_price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "bid_depth": sum(o.qty for q in self._bids.values() for o in q),
            "ask_depth": sum(o.qty for q in self._asks.values() for o in q),
            "n_fills": len(self.step_fills),
            "volume": sum(f.qty for f in self.step_fills),
        }

    def _update_quotes(self):
        if self.step_fills:
            self.last_price = self.step_fills[-1].price
        self.best_bid = max(self._bids) if self._bids else np.nan
        self.best_ask = min(self._asks) if self._asks else np.nan
        if not (np.isnan(self.best_bid) or np.isnan(self.best_ask)):
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
        else:
            # One or both sides empty: fall back to the last trade price (= the uniform clearing
            # price) rather than carrying a stale mid. Carrying produced long flat-return stretches
            # in this thin book (~28% zero returns) -> spurious volatility clustering and a surrogate
            # that could not learn the loss surface; last_price keeps the series moving.
            if not np.isnan(self.last_price):
                self.mid_price = self.last_price
            self.spread = np.nan

    def _round(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
