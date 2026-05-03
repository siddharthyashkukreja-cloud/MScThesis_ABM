"""
Discrete-time call-auction limit order book.

Design
------
- Single-asset, price-time priority, bids/asks stored as sorted dicts.
- Call auction: orders accumulate during a step, then clear once.
- Orders expire after `ttl` steps (ODD: 1-10 steps at 1-min; here 1-2 at 5-min).
- Mid-price = (best_bid + best_ask) / 2 after each auction.
- Spread and depth are outputs, not inputs -- they emerge from order flow.

Extension hooks
---------------
- `record` list stores per-step snapshots; add fields here for CCP layer.
- `execute_large_order()` stub reserved for Almgren-Chriss liquidation (Stage 5).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Order:
    order_id: int
    agent_id: int
    side: int        # +1 buy, -1 sell
    price: float
    qty: int
    ttl: int         # steps remaining before expiry


@dataclass
class Fill:
    buyer_id: int
    seller_id: int
    price: float
    qty: int


class LOB:
    def __init__(self, tick_size: float, order_ttl: int):
        self.tick_size = tick_size
        self.order_ttl = order_ttl

        self._bids: Dict[float, List[Order]] = {}
        self._asks: Dict[float, List[Order]] = {}
        self._order_index: Dict[int, tuple] = {}  # oid -> (side, price) for cancel

        self.mid_price: float = np.nan
        self.best_bid: float = np.nan
        self.best_ask: float = np.nan
        self.spread: float = np.nan

        self._next_id = 0
        self.step_fills: List[Fill] = []

    # -- Order submission -------------------------------------------------

    def add_limit(self, agent_id: int, side: int, price: float, qty: int) -> int:
        price = self._round(price)
        oid = self._next_id; self._next_id += 1
        order = Order(oid, agent_id, side, price, qty, self.order_ttl)
        book = self._bids if side == 1 else self._asks
        book.setdefault(price, []).append(order)
        self._order_index[oid] = (side, price)
        return oid

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
        """Remove a resting limit order by ID. No-op if already filled/expired."""
        if order_id not in self._order_index:
            return
        side, price = self._order_index.pop(order_id)
        book = self._bids if side == 1 else self._asks
        if price in book:
            book[price] = [o for o in book[price] if o.order_id != order_id]
            if not book[price]:
                del book[price]

    # -- Call auction -----------------------------------------------------

    def match(self) -> List[Fill]:
        """
        Price-time priority call auction.
        Crosses all buy orders >= best ask against ask queue.
        Updates mid_price, best_bid, best_ask after clearing.
        """
        fills = []
        while self._bids and self._asks:
            best_bid_px = max(self._bids)
            best_ask_px = min(self._asks)
            if best_bid_px < best_ask_px:
                break
            clear_px = best_ask_px
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

        self.step_fills = fills
        self._update_quotes()
        return fills

    # -- Expiry -----------------------------------------------------------

    def age_orders(self):
        for book in (self._bids, self._asks):
            for px in list(book):
                surviving = []
                for o in book[px]:
                    o.ttl -= 1
                    if o.ttl > 0:
                        surviving.append(o)
                    else:
                        self._order_index.pop(o.order_id, None)
                if surviving:
                    book[px] = surviving
                else:
                    del book[px]

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
        self.best_bid = max(self._bids) if self._bids else np.nan
        self.best_ask = min(self._asks) if self._asks else np.nan
        if not (np.isnan(self.best_bid) or np.isnan(self.best_ask)):
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
        elif not np.isnan(self.best_bid):
            self.mid_price = self.best_bid
        elif not np.isnan(self.best_ask):
            self.mid_price = self.best_ask

    def _round(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
