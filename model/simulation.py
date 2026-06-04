"""
Simulation driver. V_t is exogenous: loaded from params.fv_csv at init,
indexed per-step into SimContext.v. No GlobalState evolution in-sim.

Step sequence (ODD §Step Sequence, subset active):
  1. Reset LOB step_fills accumulator.
  2. Build SimContext (V_t, prev mid, tick, traders_by_id, last_volume).
  3. Each trader submit_orders.
  4. LOB.match() — call-auction clearing.
  5. Apply fills to buyer/seller inventory + cash.
  6. Mark-to-market PnL on every trader.
  7. Margin cycle every `margin_interval` ticks — variation-margin
     settlement on the clearing tier (D29; ODD §Step Sequence step 11).
  8. Append snapshot to history.

Order lifetime (D58) is governed by the agents alone — there is no blanket TTL
(the ODD §Mech #7 10-step ceiling was removed; see lob.py). FT/MT use
replace-on-new (refresh their single standing order on each activation); ZI
cancels each resting order w.p. `zi_delta` per step (Cont-Stoikov-Talreja 2008 /
Farmer et al.). Resting orders otherwise leave the book only by fill.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from .globals import ModelParams, SimContext, CCP_CALIBRATION
from .lob import LOB, Fill


class Simulation:
    def __init__(self, params: ModelParams, traders: list, seed: int = 42,
                 ccp=None, v_start: int = 0):
        self.params = params
        # Offset into the V_t series — lets a run start at any row of fv_{regime}.csv
        # (e.g. a specific COVID crash window) rather than always at row 0 (D55).
        self._v_start = int(v_start)
        self.lob = LOB(params.tick_size)
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
        self._df_interval = int(CCP_CALIBRATION["df_interval"])
        self._bars_per_day = 390   # legacy fallback grid for a fv csv without `ts`;
                                   # the real boundary is `_day_start_rows` (D57)
        self.clearing_history: List[dict] = []
        self.client_history: List[dict] = []
        if self.ccp is not None:
            for cm in self.ccp.members.values():
                cm.balance_sheet._last_mark = params.v0
                # NBCMs are not in the LOB `traders` list; register them in the id
                # map so a fire-sale fill on an ASSUMED defaulted-client position
                # marks down their inventory (D55 deficit-consistent close-out).
                self.traders_by_id.setdefault(cm.agent_id, cm)
            for t in traders:                       # client VM mark (D52)
                if t.clearing_member_id is not None:
                    t._cm_last_mark = params.v0

        # V_t path — the `V_smooth` column of the fv CSV (Stein-Stein SV
        # jump-diffusion, D33). `sigma_t` is the per-step stochastic vol
        # (D33), piped into the FT belief width via SimContext (D34); if
        # absent (legacy fv csv) the SV channel is off and FTs fall back
        # to params.sigma_v.
        df = pd.read_csv(params.fv_csv)
        v_col = "V_smooth" if "V_smooth" in df.columns else df.columns[1]
        self.v_array: np.ndarray = df[v_col].to_numpy()[self._v_start:]
        self.sigma_t_array = (df["sigma_t"].to_numpy()[self._v_start:]
                              if "sigma_t" in df.columns else None)

        # Real RTH-session opens (D57): the V_t path splices per-day series at real
        # levels, and a real session is ~405 bars and varies (149–406), not a fixed
        # 390. Read the open rows from the fv `ts` date-changes so the overnight
        # reset fires on the true boundary; fall back to a 390-grid for a legacy fv
        # csv without timestamps.
        if "ts" in df.columns:
            _d = pd.to_datetime(df["ts"]).dt.normalize().to_numpy()
            _opens = np.flatnonzero(np.concatenate(([True], _d[1:] != _d[:-1])))
        else:
            _opens = np.arange(0, len(df), self._bars_per_day)
        self._day_start_rows = frozenset(int(i) for i in _opens)

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

        # Overnight session gap (D56): at each RTH-day boundary the fundamental
        # V_t carries the real overnight move, so REPRICE the book by the gap
        # factor — it reopens at the gapped V_t with its shape intact (no empty-book
        # warm-up that would pollute the intraday returns). The gap is then a clean
        # between-session jump (the margin cycle marks the book across it — gap
        # risk), not a slow intraday drift; the day-boundary return is dropped from
        # the calibration moments (which target intraday returns, matching the
        # empirical convention). The boundary is the real session open (D57:
        # `_day_start_rows`, data-driven), offset by `_v_start`.
        if self.t > 0 and (self._v_start + self.t) in self._day_start_rows:
            if self._prev_mid == self._prev_mid and self._prev_mid > 0:
                self.lob.reprice(v_now / self._prev_mid)
            self._prev_mid = v_now
            # Re-anchor each trader's intraday price memory at the gapped open, so
            # the overnight gap is NOT seen as a one-step intraday return — without
            # this the MomentumTrader's EWMA reads the gap as trend and amplifies it
            # into the new session, inflating the intraday return tail (D56).
            for tr in self.traders:
                if hasattr(tr, "_prev_mid"):
                    tr._prev_mid = v_now

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

        # CCP fire-sale (Stage C): liquidate a defaulted member's book one
        # Almgren-Chriss slice per step as a market order — the book walk is the
        # price impact that transmits the default to surviving CMs' next mark.
        if self.ccp is not None:
            fs_side, fs_qty = self.ccp.next_firesale_slice()
            if fs_qty > 0:
                self.lob.add_market(self.ccp.ccp_id, fs_side, fs_qty)
            # NBCM assumed-client-book liquidation (D55): the NBCM has no LOB access,
            # so the CCP routes its AC slices, attributed to the NBCM agent_id so the
            # fill walks the book AND marks down the NBCM's assumed inventory.
            for cm in self.ccp.members.values():
                if cm.balance_sheet.is_banking or not cm._liq_slices:
                    continue
                qty = int(cm._liq_slices.pop(0))
                if qty > 0:
                    self.lob.add_market(cm.agent_id, cm._liq_side, qty)

        self.lob.match()
        for f in self.lob.step_fills:
            self._apply_fill(f)
        # No TTL expiry (D58): resting orders leave the book only by fill or
        # explicit cancellation — replace-on-new (FT/MT) and the ZI `zi_delta`
        # per-resting cancellation govern order lifetime.

        snap = self.lob.snapshot()
        mark_price = snap["mid_price"] if not np.isnan(snap["mid_price"]) else v_now
        for tr in self.traders:
            tr.update_pnl(mark_price)

        # Margin cycle (D29, ODD §Step Sequence step 11) — hourly at the
        # 1-min cadence: (t+1) % margin_interval == 0 (ODD §V&V).
        # The procyclical CCP margin (D55) is driven by the REGIME day-scale return
        # std `params.sigma_v` (the calibrated 1-min ES return std → daily via √390),
        # NOT the per-minute Kalman SV `sigma_now`: real CCPs reset IM daily off a
        # vol estimate, and the CSV `sigma_t` is the microstructure-noise scale used
        # for FT beliefs (it does not separate the regimes and would peg IM at the
        # APC floor in both). IM(σ_v): ~6% calm / ~12% stressed.
        if self.ccp is not None and (self.t + 1) % self._margin_interval == 0:
            self._margin_cycle(mark_price, self.params.sigma_v)
        # Cover-2 default-fund recalculation (ODD §Step Sequence step 8 / §Mech #4).
        if self.ccp is not None and (self.t + 1) % self._df_interval == 0:
            from model.globals import CONTRACT_USD
            _c = CCP_CALIBRATION
            self.ccp.recompute_default_fund(
                mark_price, self.params.volume_lot * CONTRACT_USD, self.params.sigma_v,
                _c["ex_df_ratio"], _c["cover_number"], _c["df_buffer"])

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
        # Futures-margin accounting (D29/D52): a fill moves only the position —
        # no full notional is paid. Clearing members AND cleared clients settle
        # cash through the variation-margin cycle, so neither takes the nominal
        # notional debit. Only a genuinely uncleared participant (no balance
        # sheet and no clearing member) keeps nominal cash accounting.
        buyer = self.traders_by_id.get(fill.buyer_id)
        seller = self.traders_by_id.get(fill.seller_id)
        if buyer is not None:
            buyer.inventory += fill.qty
            if not hasattr(buyer, "balance_sheet") and buyer.clearing_member_id is None:
                buyer.cash -= fill.price * fill.qty
        if seller is not None:
            seller.inventory -= fill.qty
            if not hasattr(seller, "balance_sheet") and seller.clearing_member_id is None:
                seller.cash += fill.price * fill.qty

    def _cancel_open(self, trader):
        """Cancel a defaulted client's resting LOB orders so its book stops (D52)."""
        oid = getattr(trader, "_open_oid", None)
        if oid is not None:
            if self.lob.is_resting(oid):
                self.lob.cancel(oid)
            trader._open_oid = None
        oids = getattr(trader, "_open_oids", None)
        if oids:
            for o in list(oids):
                if self.lob.is_resting(o):
                    self.lob.cancel(o)
            trader._open_oids = []

    def _margin_cycle(self, mid: float, sigma_t: float):
        """ODD §Step Sequence step 11 / §Mech #3 — hourly variation-margin
        settlement (D29/D52), in two phases. `sigma_t` is the regime day-scale
        return std (params.sigma_v) that sets the procyclical CCP initial margin
        (D55 — VaR/SPAN via globals.im_fraction; ~6% calm / ~12% stressed).

        Phase 1 — CLIENT margin calls (client → CM → CCP). Each cleared client is
        marked and pays its own VM from its own cash; the CM relays it to the CCP
        (pass-through). A client breaching the 8% capital floor freezes (stops
        trading); one that exhausts cash DEFAULTS — its CM (guarantor) absorbs the
        shortfall and closes the position. This is the client→CM contagion channel.

        Phase 2 — the CM settles its OWN-account VM, recomputes IM / capital ratio,
        deleverages (BCM AC fire-sale) or stops out (NBCM) at the floor, and DEFAULTS
        on cash exhaustion — now including any absorbed client losses → 5-level
        waterfall + CCP fire-sale. One clearing_history row per CM per cycle; one
        client_history row per client default."""
        from model.globals import CONTRACT_USD, FIRE_SALE_URGENCY, AC_HORIZON, im_fraction
        im_frac = im_fraction(sigma_t)                  # procyclical CCP IM (D55)
        mm = CCP_CALIBRATION["mm_percent"]
        floor = CCP_CALIBRATION["cap_ratio_floor"]
        usd = self.params.volume_lot * CONTRACT_USD     # USD per (lot · price point)

        # ── Phase 1: client margin calls, freeze and default ─────────────────
        absorbed: Dict[int, float] = {}     # cm_id -> client loss absorbed this cycle
        ndef: Dict[int, int] = {}           # cm_id -> client defaults this cycle
        for cm in self.ccp.members.values():
            bs = cm.balance_sheet
            if bs.has_defaulted:
                continue
            for cid in list(bs.client_positions):
                client = self.traders_by_id.get(cid)
                if client is None or client.has_defaulted:
                    continue
                cvm = client.inventory * (mid - client._cm_last_mark) * usd
                client.cash += cvm
                client._cm_last_mark = mid
                bs.client_positions[cid] = client.inventory
                if client.cash <= 0.0:
                    # Client default (D54): the CM ASSUMES the position and LIQUIDATES
                    # it. It first covers the uncollected VM (shortfall). Then a BCM
                    # takes the position onto its OWN book and self-liquidates it via
                    # Almgren-Chriss (the loss is realised through the assumed book's VM
                    # plus the fire-sale book-walk impact — no flat haircut). An NBCM has
                    # no LOB access, so the CCP liquidates the assumed position on its
                    # behalf (book-walk impact) and the NBCM bears the (1-recovery)
                    # close-out haircut (ODD §Mech #6) as its realised-loss proxy.
                    shortfall = -client.cash
                    pos = int(client.inventory)
                    cm.cash -= shortfall
                    client.cash = 0.0
                    client.has_defaulted = True
                    client._stopped = True
                    self._cancel_open(client)
                    bs.client_positions[cid] = 0
                    client.inventory = 0
                    # Both BCM and NBCM ASSUME the position onto their book and
                    # liquidate it via Almgren-Chriss; the close-out loss is realised
                    # by marking the assumed position to market as the fire-sale walks
                    # the price (D55 deficit-consistent), NOT a flat (1-recovery)·notional
                    # haircut. The BCM sends its own slices (submit_orders); the NBCM's
                    # are routed by the CCP (step), both attributed to the CM so fills
                    # mark down its inventory.
                    cm.inventory += pos
                    cm.start_firesale(abs(pos), FIRE_SALE_URGENCY, AC_HORIZON)
                    closeout = 0.0
                    absorbed[cm.agent_id] = absorbed.get(cm.agent_id, 0.0) + shortfall
                    ndef[cm.agent_id] = ndef.get(cm.agent_id, 0) + 1
                    self.client_history.append({
                        "t": self.t, "client_id": cid,
                        "kind": type(client).__name__, "cm_id": cm.agent_id,
                        "cm_kind": "BCM" if bs.is_banking else "NBCM",
                        "shortfall": shortfall, "closeout_loss": closeout,
                        "assumed_pos": pos})
                else:
                    cexp = abs(client.inventory) * usd * mid
                    client._stopped = (cexp > 0.0 and client.cash / cexp <= floor)

        # ── Phase 2: CM own-account VM + CM capital / default ────────────────
        for cm in self.ccp.members.values():
            bs = cm.balance_sheet
            if bs.has_defaulted:
                continue
            # VM on the own/assumed position. For a BCM this is its own-account book;
            # for an NBCM it is 0 unless it is carrying a defaulted client's position
            # it has assumed and is liquidating (D55) — marking it realises the
            # close-out loss as the fire-sale walks the price.
            own_pos = cm.inventory
            vm = own_pos * (mid - bs._last_mark) * usd
            cm.cash += vm
            bs.variation_margin += vm
            bs._last_mark = mid

            own_notional = abs(cm.inventory) * usd * mid
            exposure = own_notional + bs.client_notional(mid)
            bs.initial_margin = im_frac * exposure       # procyclical CCP IM (D55)
            bs.maintenance_margin = mm * bs.initial_margin
            bs.call_indicator = (bs.initial_margin > 0.0
                                 and abs(vm) > (1.0 - mm) * bs.initial_margin)
            book_pos = bs.book_position(cm.inventory)
            wf_level = 0
            if cm.cash <= 0.0:
                bs.has_defaulted = True
                self.ccp.default_list.append(cm.agent_id)
                deficit = -cm.cash                  # unpaid shortfall to mutualise
                cm.cash = 0.0
                wf_level = self.ccp.default_waterfall(cm, deficit)
                self.ccp.start_firesale(book_pos, FIRE_SALE_URGENCY, AC_HORIZON)
            elif cm.capital_ratio(mid, sigma_t) <= floor:
                if bs.is_banking:
                    if not cm._liq_slices:
                        # Restore cash/IM = floor (CFTC Reg 1.17): target IM = cash/floor,
                        # so target notional = (cash/floor)/im_frac. Only the own position
                        # can be shed (the client book can't be deleveraged intraday).
                        target_notional = (cm.cash / floor) / im_frac if im_frac > 0 else 0.0
                        target_own = max(0.0, target_notional - bs.client_notional(mid))
                        target_inv = target_own / (usd * mid) if usd * mid > 0 else 0.0
                        cm.start_firesale(abs(cm.inventory) - target_inv,
                                          FIRE_SALE_URGENCY, AC_HORIZON)
                else:
                    # NBCM operational stop-out (ODD §Mech #2 / CFTC Reg 1.17): on a
                    # cash/IM breach the NBCM stops taking client risk — freeze its
                    # clients (they port/close via the CCP on an actual default). Now
                    # wired LIVE (D55): cash/IM breaches only in distress, whereas the
                    # earlier cash/gross-notional ratio breached routinely and froze the
                    # book in normal trading (an FCM clears 8-30× its capital in notional).
                    cm._stopped = True
                    for cid in bs.client_positions:
                        client = self.traders_by_id.get(cid)
                        if client is not None and not client.has_defaulted:
                            client._stopped = True

            self.clearing_history.append({
                "t": self.t, "agent_id": cm.agent_id,
                "kind": "BCM" if bs.is_banking else "NBCM",
                "cash": cm.cash, "own_position": cm.inventory,
                "client_notional": bs.client_notional(mid),
                "capital_ratio": cm.capital_ratio(mid, sigma_t),
                "initial_margin": bs.initial_margin,
                "maintenance_margin": bs.maintenance_margin,
                "vm_cycle": vm, "vm_cumulative": bs.variation_margin,
                "call_indicator": bs.call_indicator, "waterfall_level": wf_level,
                "client_defaults": ndef.get(cm.agent_id, 0),
                "client_loss_absorbed": absorbed.get(cm.agent_id, 0.0),
                "has_defaulted": bs.has_defaulted,
            })

    def run(self, n_steps: int) -> Dict[str, List]:
        for _ in range(n_steps):
            self.step()
        return self.history
