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
     settlement on the clearing tier (ODD §Step Sequence step 11).
  8. Append snapshot to history.

Order lifetime is governed by the agents alone — there is no blanket TTL
(the ODD §Mech #7 10-step ceiling was removed; see lob.py). FT/MT use
replace-on-new (refresh their single standing order on each activation); ZI
cancels each resting order w.p. `zi_delta` per step (Cont-Stoikov-Talreja 2008 /
Farmer et al.). Resting orders otherwise leave the book only by fill.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from . import globals as G
from .globals import ModelParams, SimContext, CCP_CALIBRATION
from .lob import LOB, Fill


class Simulation:
    def __init__(self, params: ModelParams, traders: list, seed: int = 42,
                 ccp=None, v_start: int = 0):
        self.params = params
        # Offset into the V_t series — lets a run start at any row of fv_{regime}.csv
        # (e.g. a specific COVID crash window) rather than always at row 0.
        self._v_start = int(v_start)
        self.lob = LOB(params.tick_size)
        self.traders = traders
        self.traders_by_id: Dict[int, object] = {t.agent_id: t for t in traders}
        self.rng = np.random.default_rng(seed)
        # Clearing tier. `ccp` is the CentralCounterparty when the
        # clearing scaffold is wired (run_simulation); None on the
        # calibration path, which runs the bare market layer. When present,
        # the variation-margin cycle runs every `margin_interval`
        # ticks; clearing_history records one row per CM per cycle.
        self.ccp = ccp
        self._margin_interval = int(CCP_CALIBRATION["margin_interval"])
        self._df_interval = int(CCP_CALIBRATION["df_interval"])
        # Reactive-IM EWMA state: rolling realised variance of the 1-min sim
        # mid return, seeded at the regime prior; half-life G.IM_VOL_HALFLIFE.
        self._im_var = float(params.sigma_v) ** 2
        self._im_lam = 1.0 - 0.5 ** (1.0 / float(G.IM_VOL_HALFLIFE))
        self._bars_per_day = 390   # legacy fallback grid for a fv csv without `ts`;
                                   # the real boundary is `_day_start_rows`
        self.clearing_history: List[dict] = []
        self.client_history: List[dict] = []
        self.freeze_log: List[dict] = []      # client freeze onsets (own-distress / CM-contagion)
        self.porting_log: List[dict] = []     # client porting on a member default (EMIR Art. 48)
        if self.ccp is not None:
            for cm in self.ccp.members.values():
                cm.balance_sheet._last_mark = params.v0
                # NBCMs are not in the LOB `traders` list; register them in the id
                # map so a fire-sale fill on an ASSUMED defaulted-client position
                # marks down their inventory.
                self.traders_by_id.setdefault(cm.agent_id, cm)
            # The CCP is registered too: its disposal fire-sale fills mark
            # down the assumed defaulted book (deficit-consistent close-out).
            self.traders_by_id.setdefault(self.ccp.ccp_id, self.ccp)
            for t in traders:                       # client VM mark
                if t.clearing_member_id is not None:
                    t._cm_last_mark = params.v0

        # V_t path — the `V_smooth` column of the fv CSV. `sigma_t` is the
        # per-step stochastic vol, piped into the FT belief width via SimContext;
        # if absent (legacy fv csv) the SV channel is off and FTs fall back to
        # params.sigma_v.
        df = pd.read_csv(params.fv_csv)
        v_col = "V_smooth" if "V_smooth" in df.columns else df.columns[1]
        self.v_array: np.ndarray = df[v_col].to_numpy()[self._v_start:]
        self.sigma_t_array = (df["sigma_t"].to_numpy()[self._v_start:]
                              if "sigma_t" in df.columns else None)

        # Real RTH-session opens: the V_t path splices per-day series at real
        # levels, and a real session is ~405 bars and varies (149–406), not a fixed
        # 390. Read the open rows from the fv `ts` date-changes so the overnight
        # reset fires on the true boundary; fall back to a 390-grid for a legacy fv
        # csv without timestamps.
        if "ts" in df.columns:
            _d = pd.to_datetime(df["ts"]).dt.normalize().to_numpy()
            _opens = np.flatnonzero(np.concatenate(([True], _d[1:] != _d[:-1])))
        else:
            _d = None
            _opens = np.arange(0, len(df), self._bars_per_day)
        self._day_start_rows = frozenset(int(i) for i in _opens)

        # Daily-IM (G.IM_DAILY): per-step daily close-to-close vol (RiskMetrics EWMA, warmed on
        # real pre-window history) looked up by each step's date — supersedes the intraday EWMA.
        self._sigma_daily_row = None
        if G.IM_DAILY and _d is not None:
            try:
                _ds = G.daily_sigma_series()
                # tz-robust reindex: coerce both the daily-vol series and the per-step dates to
                # tz-naive normalized days so the pre-window-warmed daily vol is matched onto every
                # session regardless of the fv `ts` tz format. (A tz mismatch here was otherwise
                # swallowed by the except below, silently dropping margin back to the cold intraday EWMA.)
                _dsi = pd.DatetimeIndex(_ds.index)
                _dsi = _dsi.tz_localize(None) if _dsi.tz is not None else _dsi
                _ds = pd.Series(_ds.to_numpy(), index=_dsi.normalize())
                _sd = pd.DatetimeIndex(_d[self._v_start:self._v_start + len(self.v_array)])
                _sd = _sd.tz_localize(None) if _sd.tz is not None else _sd
                self._sigma_daily_row = _ds.reindex(_sd.normalize(), method="ffill").to_numpy(dtype=float)
            except Exception:
                self._sigma_daily_row = None

        # Overnight-gap variance EWMA (G.IM_INCLUDE_GAPS): the intraday reactive EWMA misses the
        # session gap the margin must mark the book across, so a separate EWMA of squared boundary
        # gap returns is added to the daily variance (close-to-close coverage). Seeded from the
        # path's own overnight gaps (the regime's historical gap variance); half-life ~11 sessions.
        _gaps = []
        for _i in _opens:
            _j = int(_i) - self._v_start
            if 0 < _j < len(self.v_array) and self.v_array[_j] > 0 and self.v_array[_j - 1] > 0:
                _gaps.append(float(np.log(self.v_array[_j] / self.v_array[_j - 1])) ** 2)
        self._gap_var = float(np.mean(_gaps)) if _gaps else 0.0
        self._gap_lam = 1.0 - 0.5 ** (1.0 / 11.0)

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

        # Overnight session gap: at each RTH-day boundary the fundamental
        # V_t carries the real overnight move, so REPRICE the book by the gap
        # factor — it reopens at the gapped V_t with its shape intact (no empty-book
        # warm-up that would pollute the intraday returns). The gap is then a clean
        # between-session jump (the margin cycle marks the book across it — gap
        # risk), not a slow intraday drift; the day-boundary return is dropped from
        # the calibration moments (which target intraday returns, matching the
        # empirical convention). The boundary is the real session open
        # (`_day_start_rows`, data-driven), offset by `_v_start`.
        if self.t > 0 and (self._v_start + self.t) in self._day_start_rows:
            self._flush_liquidations()   # EOD: finish any open AC liquidation within the session
            if self._prev_mid == self._prev_mid and self._prev_mid > 0:
                if G.IM_INCLUDE_GAPS:   # feed the overnight gap into the close-to-close margin EWMA
                    _g = float(np.log(v_now / self._prev_mid))
                    self._gap_var = ((1.0 - self._gap_lam) * self._gap_var
                                     + self._gap_lam * _g * _g)
                self.lob.reprice(v_now / self._prev_mid)
            self._prev_mid = v_now
            # Re-anchor each trader's intraday price memory at the gapped open, so
            # the overnight gap is NOT seen as a one-step intraday return — without
            # this the MomentumTrader's EWMA reads the gap as trend and amplifies it
            # into the new session, inflating the intraday return tail.
            # Gated by G.REANCHOR_ON_GAP (default True): set False for the
            # open-disorder sensitivity (the gap becomes a disorderly open).
            if G.REANCHOR_ON_GAP:
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

        # CCP fire-sale: liquidate a defaulted member's book one
        # Almgren-Chriss slice per step as a market order — the book walk is the
        # price impact that transmits the default to surviving CMs' next mark.
        if self.ccp is not None:
            fs_side, fs_qty = self.ccp.next_firesale_slice()
            if fs_qty > 0:
                self.lob.add_market(self.ccp.ccp_id, fs_side, fs_qty)
            # NBCM assumed-client-book liquidation: the NBCM has no LOB access,
            # so the CCP routes its AC slices, attributed to the NBCM agent_id so the
            # fill walks the book AND marks down the NBCM's assumed inventory.
            for cm in self.ccp.members.values():
                if (cm.balance_sheet.is_banking or cm.balance_sheet.is_client
                        or cm.balance_sheet.has_defaulted or not cm._liq_slices):
                    continue
                qty = int(cm._liq_slices.pop(0))
                if qty > 0:
                    self.lob.add_market(cm.agent_id, cm._liq_side, qty)

        self.lob.match()
        for f in self.lob.step_fills:
            self._apply_fill(f)
        # No TTL expiry: resting orders leave the book only by fill or
        # explicit cancellation — replace-on-new (FT/MT) and the ZI `zi_delta`
        # per-resting cancellation govern order lifetime.

        snap = self.lob.snapshot()
        mark_price = snap["mid_price"] if not np.isnan(snap["mid_price"]) else v_now
        for tr in self.traders:
            tr.update_pnl(mark_price)

        # Margin cycle — hourly at the 1-min
        # cadence. The margin-driving sigma is set by G.IM_MODE:
        # "reactive" = EWMA of realised intraday sim returns (rolling VaR margin —
        # the realistic baseline; IM then ratchets up THROUGH a crash, as CME's ES
        # margin did in Mar-2020); "static" = the regime constant sigma_v (
        # two-point procyclicality); "flat" = im_fraction ignores sigma entirely.
        # The EWMA updates on intraday returns only (the day-boundary return was
        # re-anchored at the reprice, so overnight gaps never enter the estimator —
        # consistent with the SIGMA_V intraday convention).
        if self.ccp is not None:
            if (self._prev_mid == self._prev_mid and self._prev_mid > 0
                    and mark_price == mark_price and mark_price > 0):
                _r = float(np.log(mark_price / self._prev_mid))
                self._im_var = ((1.0 - self._im_lam) * self._im_var
                                + self._im_lam * _r * _r)
            if G.IM_DAILY and self._sigma_daily_row is not None:
                # Daily close-to-close vol (RiskMetrics EWMA, warmed on real pre-window history)
                # for this step's session; im_fraction re-scales by sqrt(390) so pass sigma/sqrt(390).
                # Close-to-close already covers the overnight gap (no separate gap-EWMA needed).
                _sd = self._sigma_daily_row[self.t] if self.t < len(self._sigma_daily_row) else float("nan")
                sigma_im = (float(_sd) / (G.TRADING_MINUTES_PER_DAY ** 0.5)
                            if (_sd == _sd and _sd > 0) else self.params.sigma_v)
            elif G.IM_MODE == "reactive":
                # Legacy intraday EWMA + separate gap-EWMA (superseded by IM_DAILY).
                _gv = (self._gap_var / float(G.TRADING_MINUTES_PER_DAY)
                       if G.IM_INCLUDE_GAPS else 0.0)
                sigma_im = (self._im_var + _gv) ** 0.5
            else:
                sigma_im = self.params.sigma_v
            if (self.t + 1) % self._margin_interval == 0:
                self._margin_cycle(mark_price, sigma_im)
            # Cover-2 default-fund recalculation (ODD §Step Sequence step 8 / §Mech #4).
            if (self.t + 1) % self._df_interval == 0:
                from model.globals import CONTRACT_USD
                _c = CCP_CALIBRATION
                self.ccp.recompute_default_fund(
                    mark_price, self.params.volume_lot * CONTRACT_USD, sigma_im,
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
        # Futures-margin accounting: a fill moves only the position —
        # no full notional is paid. Clearing members AND cleared clients settle
        # cash through the variation-margin cycle, so neither takes the nominal
        # notional debit. Only a genuinely uncleared participant (no balance
        # sheet and no clearing member) keeps nominal cash accounting.
        buyer = self.traders_by_id.get(fill.buyer_id)
        seller = self.traders_by_id.get(fill.seller_id)
        if buyer is not None:
            buyer.inventory += fill.qty
            buyer._fill_qty += fill.qty
            buyer._fill_cost += fill.price * fill.qty
            if not hasattr(buyer, "balance_sheet") and buyer.clearing_member_id is None:
                buyer.cash -= fill.price * fill.qty
        if seller is not None:
            seller.inventory -= fill.qty
            seller._fill_qty -= fill.qty
            seller._fill_cost -= fill.price * fill.qty
            if not hasattr(seller, "balance_sheet") and seller.clearing_member_id is None:
                seller.cash += fill.price * fill.qty

    def _cancel_open(self, trader):
        """Cancel a defaulted client's resting LOB orders so its book stops."""
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

    def _port_clients(self, dcm, mid: float, im_frac: float, usd: float,
                      floor: float) -> int:
        """Port a defaulted CM's surviving clients to other client-carrying CMs
        (EMIR Art. 48(5)-(6): transfer to a backup member, else liquidate; in the
        spirit of ODD §Mech #6, which moves a defaulted book to surviving members).
        Greedy by spare risk capacity: a receiver takes a client only while it
        stays above the 8% cash/IM floor with the ported notional — porting in
        stress is not guaranteed (OFR 2026: the modal client has a single clearing
        agent). Returns the net position of unported clients, which the CCP
        closes out; unported clients are frozen with their books flat."""
        bs = dcm.balance_sheet
        receivers = [c for c in self.ccp.members.values()
                     if c is not dcm and not c.balance_sheet.has_defaulted
                     and not c._stopped and c.balance_sheet.client_positions]
        head = {}
        for c in receivers:
            expo = (abs(c.inventory) * usd * mid
                    + c.balance_sheet.client_notional(mid))
            if getattr(G, "DIFFERENTIATED_FLOORS", False):
                # Option A: receiver headroom under its own rule — BCM cash/exposure >= LR_FLOOR_BCM
                # (max exposure = cash/LR_FLOOR_BCM); NBCM cash/IM >= REG117_FLOOR_NBCM
                # (max exposure = cash/(REG117_FLOOR_NBCM*im_frac)).
                if c.balance_sheet.is_banking:
                    head[c.agent_id] = (c.cash / G.LR_FLOOR_BCM - expo
                                        if G.LR_FLOOR_BCM > 0 else 0.0)
                else:
                    den = G.REG117_FLOOR_NBCM * im_frac
                    head[c.agent_id] = c.cash / den - expo if den > 0 else 0.0
            elif G.IM_ESCROW:
                # escrow ratio is cash/exposure >= floor -> max exposure = cash/floor
                head[c.agent_id] = c.cash / floor - expo if floor > 0 else 0.0
            else:
                # legacy cash/IM ratio -> max exposure = cash/(floor*im_frac)
                head[c.agent_id] = (c.cash / (floor * im_frac) - expo
                                    if im_frac > 0 and floor > 0 else 0.0)
        leftover = 0
        n_ported = n_unported = 0
        for cid in list(bs.client_positions):
            del bs.client_positions[cid]
            client = self.traders_by_id.get(cid)
            if client is None or client.has_defaulted:
                continue
            need = abs(client.inventory) * usd * mid
            best = max(receivers, key=lambda c: head[c.agent_id], default=None)
            if best is not None and head[best.agent_id] >= need:
                client.clearing_member_id = best.agent_id
                best.client_ids.append(cid)
                best.balance_sheet.client_positions[cid] = client.inventory
                head[best.agent_id] -= need
                n_ported += 1
            else:
                leftover += int(client.inventory)
                client._stopped = True
                client.inventory = 0
                self._cancel_open(client)
                n_unported += 1
        dcm.client_ids = []
        self.porting_log.append({"t": self.t, "defaulted_cm": dcm.agent_id,
            "cm_kind": "BCM" if bs.is_banking else "NBCM", "n_ported": n_ported,
            "n_unported": n_unported, "unported_position": int(leftover)})
        return leftover

    def _post_im(self, agent, required_im: float) -> bool:
        """Physically move the delta between `required_im` and the agent's currently
        posted IM between its cash and the CCP im_account (G.IM_ESCROW). Posting is
        bounded by available cash; returns True if the call was met in full, False if
        the agent ran out of cash (a liquidity shortfall → caller defaults it)."""
        delta = required_im - agent._posted_im
        if delta > 0.0:
            pay = min(delta, max(0.0, agent.cash))
            agent.cash -= pay
            agent._posted_im += pay
            self.ccp.im_account += pay
            return agent._posted_im >= required_im - 1e-6
        if delta < 0.0:                                   # position shrank — return excess IM
            agent.cash += -delta
            agent._posted_im += delta
            self.ccp.im_account += delta
        return True

    def _seize_im(self, agent, deficit: float) -> float:
        """On default, seize the agent's posted IM: release it from the CCP account and
        apply it to the deficit first (the defaulter's own collateral is first-loss). Any
        EXCESS over the loss returns to the defaulter's estate (its own cash) so system
        cash is conserved — it is not mutualised (the defaulter's surplus collateral is
        its property, EMIR Art. 48)."""
        im = agent._posted_im
        if im > 0.0:
            self.ccp.im_account -= im
            agent._posted_im = 0.0
            used = min(max(0.0, deficit), im)
            agent.cash += im - used                  # excess collateral back to the estate
            deficit = max(0.0, deficit - im)
        return deficit

    def _transfer_book(self, pos: int, candidates: list, mid: float):
        """Transfer a defaulted net position `pos` to the surviving candidate that most reduces
        its OWN net position (the largest-opposing book — the ODD PositionAuction / EMIR-48
        transfer target), so the auction goes to the member best able to absorb it. The receiver
        books it as a FILL AT `mid`, so its variation margin is charged from the transfer price
        rather than its prior mark; total system inventory is preserved (no flatten), which keeps
        the VM cycle zero-sum and the system cash-conserving. The (1-recovery) haircut is handled
        separately as the deadweight close-out loss. Returns the receiver, or None when there is
        no surviving counterparty (the caller warehouses the book at the CCP)."""
        if pos == 0 or not candidates:
            return None
        recv = min(candidates,
                   key=lambda c: (abs(int(getattr(c, "inventory", 0)) + pos), -max(0.0, c.cash)))
        recv.inventory = int(getattr(recv, "inventory", 0)) + pos
        recv._fill_qty += pos
        recv._fill_cost += pos * mid
        return recv

    def _ccp_warehouse(self, pos: int, dcm, mid: float) -> None:
        """No surviving counterparty to take the defaulter's book (tier collapse): the CCP holds
        it, booked as a fill at `mid`. The disposal-settlement block marks it each cycle and
        resumes the waterfall on losses, so inventory is preserved without any open-market
        disposal (no price impact, consistent with the transfer close-out)."""
        if pos == 0:
            return
        if (self.ccp._disposal_dcm is None and self.ccp.inventory == 0
                and self.ccp._fill_qty == 0):
            self.ccp._disposal_mark = mid
        self.ccp.inventory += pos
        self.ccp._fill_qty += pos
        self.ccp._fill_cost += pos * mid
        self.ccp._disposal_dcm = dcm

    def _flush_liquidations(self) -> None:
        """End-of-session (EOD) flush: complete any unfinished Almgren-Chriss liquidation so
        no assumed/deleveraged book is carried across the overnight gap — the close-out loss
        is realised within the session. Remaining slices collapse into one closing market
        order per liquidating member and for the CCP disposal queue."""
        if self.ccp is None:
            return
        for m in self.ccp.members.values():
            # In the direct-clearing counterfactual the members are plain clients that do
            # not run their own Almgren-Chriss liquidation (the CCP closes them out), so they
            # carry no _liq_slices — guard for that.
            if getattr(m, "_liq_slices", None):
                qty = int(sum(m._liq_slices)); m._liq_slices = []
                if qty > 0:
                    self.lob.add_market(m.agent_id, m._liq_side, qty)
        if self.ccp._liq_slices:
            qty = int(sum(self.ccp._liq_slices)); self.ccp._liq_slices = []
            if qty > 0:
                self.lob.add_market(self.ccp.ccp_id, self.ccp._liq_side, qty)

    def _margin_cycle(self, mid: float, sigma_t: float):
        """ODD §Step Sequence step 11 / §Mech #3 — hourly variation-margin
        settlement, in two phases. `sigma_t` is the regime day-scale
        return std (params.sigma_v) that sets the procyclical CCP initial margin
       .

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
        from model.globals import CONTRACT_USD, AC_HORIZON, im_fraction, ac_urgency
        fs_urgency = ac_urgency("stressed" if self.params.stressed else "calm")  # AC Eq.19, derived
        im_frac = im_fraction(sigma_t)                  # procyclical CCP IM
        mm = CCP_CALIBRATION["mm_percent"]
        floor = CCP_CALIBRATION["cap_ratio_floor"]
        # Option A: clients (leveraged end-users) freeze near a maintenance-margin breach, not at
        # the bank leverage floor (members use type-specific floors in Phase 2). Falls back to the
        # uniform floor when DIFFERENTIATED_FLOORS is off.
        client_floor = (G.CLIENT_FREEZE_FLOOR if getattr(G, "DIFFERENTIATED_FLOORS", False)
                        else floor)
        usd = self.params.volume_lot * CONTRACT_USD     # USD per (lot · price point)

        # ── Phase 0: DIRECT-cleared end users ────────
        # Identical client mechanics to the tiered run (fill-settled VM from own
        # cash, 8%-of-notional freeze, default at cash 0) — but with NO clearing-
        # member buffer: a defaulted client's deficit goes straight to the CCP
        # waterfall and its book to the CCP fire-sale. Direct clients are members
        # for DF/waterfall purposes (registered, cash-funded contributions) and
        # are skipped by the Phase-2 member logic.
        for cm in self.ccp.members.values():
            bs = cm.balance_sheet
            if not bs.is_client or bs.has_defaulted:
                continue
            client = cm
            cvm = (client.inventory * (mid - client._cm_last_mark)
                   + client._cm_last_mark * client._fill_qty
                   - client._fill_cost) * usd
            client._fill_qty = 0; client._fill_cost = 0.0
            client.cash += cvm
            client._cm_last_mark = mid
            bs.variation_margin += cvm
            met = True
            if G.IM_ESCROW and client.cash > 0.0:
                met = self._post_im(client, im_frac * abs(client.inventory) * usd * mid)
            if client.cash <= 0.0 or not met:
                bs.has_defaulted = True
                client.has_defaulted = True
                client._stopped = True
                self.ccp.default_list.append(client.agent_id)
                deficit = -client.cash
                client.cash = 0.0
                if G.IM_ESCROW:
                    deficit = self._seize_im(client, deficit)
                self._cancel_open(client)
                pos = int(client.inventory)
                if G.CLOSEOUT_MODE == "transfer":
                    # CCP AUCTION (default): the direct client's book is TRANSFERRED to the
                    # surviving direct client with the largest opposing position (ODD
                    # PositionAuction); inventory is preserved (no flatten), so VM stays zero-sum.
                    # The (1-recovery) haircut is the deadweight close-out loss, mutualised via the
                    # waterfall. No open-market disposal (only the CCP auctions).
                    deficit += (1.0 - G.CLOSEOUT_RECOVERY) * abs(pos) * usd * mid
                    wf = self.ccp.default_waterfall(client, deficit)
                    rxers = [c for c in self.ccp.members.values()
                             if c.balance_sheet.is_client and not c.balance_sheet.has_defaulted
                             and c.agent_id != client.agent_id]
                    if self._transfer_book(pos, rxers, mid) is None:
                        self._ccp_warehouse(pos, client, mid)
                    client.inventory = 0
                    assumed = 0
                else:
                    # firesale (deferred): CCP open-market Almgren-Chriss disposal.
                    wf = self.ccp.default_waterfall(client, deficit)
                    if (self.ccp._disposal_dcm is None and self.ccp.inventory == 0
                            and self.ccp._fill_qty == 0):
                        self.ccp._disposal_mark = mid   # see member-default note
                    self.ccp.inventory += pos
                    self.ccp._fill_qty += pos
                    self.ccp._fill_cost += pos * mid
                    self.ccp._disposal_dcm = client
                    client.inventory = 0
                    self.ccp.start_firesale(pos, fs_urgency, AC_HORIZON)
                    assumed = pos
                self.client_history.append({
                    "t": self.t, "client_id": client.agent_id,
                    "kind": type(client).__name__, "cm_id": self.ccp.ccp_id,
                    "cm_kind": "CCP", "shortfall": deficit,
                    "closeout_loss": 0.0, "assumed_pos": assumed})
                self.clearing_history.append({
                    "t": self.t, "agent_id": client.agent_id, "kind": "DCLIENT",
                    "cash": 0.0, "own_position": 0,
                    "client_notional": 0.0, "capital_ratio": 0.0,
                    "initial_margin": 0.0, "maintenance_margin": 0.0,
                    "vm_cycle": cvm, "vm_cumulative": bs.variation_margin,
                    "call_indicator": False, "waterfall_level": wf,
                    "client_defaults": 0, "client_loss_absorbed": 0.0,
                    "has_defaulted": True})
            else:
                cexp = abs(client.inventory) * usd * mid
                was = client._stopped
                client._stopped = (cexp > 0.0 and client.cash / cexp <= client_floor)
                if client._stopped and not was:
                    self._cancel_open(client)       # see Phase 1
                    self.freeze_log.append({"t": self.t, "client_id": client.agent_id,
                        "kind": type(client).__name__, "cm_id": self.ccp.ccp_id, "cm_kind": "CCP",
                        "reason": "own_distress", "kappa": client.cash / cexp if cexp > 0 else float("nan")})

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
                cvm = (client.inventory * (mid - client._cm_last_mark)
                       + client._cm_last_mark * client._fill_qty
                       - client._fill_cost) * usd   # P/L vs avg filled price (ODD)
                client._fill_qty = 0; client._fill_cost = 0.0
                client.cash += cvm
                client._cm_last_mark = mid
                bs.client_positions[cid] = client.inventory
                met = True
                if G.IM_ESCROW and client.cash > 0.0:
                    met = self._post_im(client, im_frac * abs(client.inventory) * usd * mid)
                if client.cash <= 0.0 or not met:
                    # Client default: the CM covers the uncollected VM (shortfall) and
                    # closes out the position. CLOSEOUT_MODE selects how (globals):
                    #  "transfer" (default) — the position is assigned to the CM's
                    #     largest-offsetting client at CLOSEOUT_RECOVERY; the CM bears the
                    #     (1-recovery) close-out haircut. No open-market liquidation.
                    #  "firesale" (deferred hypothesis) — the CM ASSUMES the position and
                    #     self-liquidates it via Almgren-Chriss; the loss is realised as the
                    #     fire-sale walks the LOB (the price-impact contagion channel).
                    shortfall = -client.cash
                    pos = int(client.inventory)
                    im_posted_val = (client._posted_im if G.IM_ESCROW  # actual escrowed IM,
                                     else CCP_CALIBRATION["im_percent"] * abs(pos) * usd * mid)
                    client.cash = 0.0                                   # captured before seize
                    client.has_defaulted = True
                    client._stopped = True
                    self._cancel_open(client)
                    bs.client_positions[cid] = 0
                    client.inventory = 0
                    if G.CLIENT_CLOSEOUT == "transfer":
                        # Transfer variant: assign to the member's largest-offsetting client at
                        # recovery (no market impact). Client first-loss for the VM shortfall +
                        # close-out haircut. Under IM_ESCROW it is the client's PHYSICALLY posted
                        # procyclical IM (seized here); else the notional 20% house-margin buffer.
                        haircut = (1.0 - G.CLOSEOUT_RECOVERY) * abs(pos) * usd * mid
                        if G.IM_ESCROW:
                            cm_loss = self._seize_im(client, shortfall + haircut)
                        else:
                            im_client = CCP_CALIBRATION["im_percent"] * abs(pos) * usd * mid
                            cm_loss = shortfall + max(0.0, haircut - im_client)
                        cm.cash -= cm_loss
                        closeout = max(0.0, cm_loss - shortfall)
                        # Transfer = the member (guarantor) ASSUMES the client's position onto its
                        # own book, booked at mid (inventory preserved; no open-market sale, no
                        # price impact); it is marked thereafter in Phase 2.
                        cm.inventory += pos
                        cm._fill_qty += pos
                        cm._fill_cost += pos * mid
                        assumed = pos
                    else:
                        # OPEN-MARKET close-out (DEFAULT): the member ASSUMES the position
                        # (booked as a fill at `mid` so Phase-2 VM does not retro-charge the
                        # already-settled move) and liquidates it via Almgren-Chriss — the BCM
                        # sends its own slices, the NBCM's are routed by the CCP, both marking
                        # down the assumed inventory. The book-walk realises the close-out loss.
                        # The client's posted IM is seized first against the VM shortfall.
                        if G.IM_ESCROW:
                            shortfall = self._seize_im(client, shortfall)
                        cm.cash -= shortfall
                        cm.inventory += pos
                        cm._fill_qty += pos
                        cm._fill_cost += pos * mid
                        cm.start_firesale(abs(pos), fs_urgency, AC_HORIZON)
                        closeout = 0.0
                        cm_loss = shortfall
                        assumed = pos
                    absorbed[cm.agent_id] = absorbed.get(cm.agent_id, 0.0) + cm_loss
                    ndef[cm.agent_id] = ndef.get(cm.agent_id, 0) + 1
                    self.client_history.append({
                        "t": self.t, "client_id": cid,
                        "kind": type(client).__name__, "cm_id": cm.agent_id,
                        "cm_kind": "BCM" if bs.is_banking else "NBCM",
                        "shortfall": shortfall, "closeout_loss": closeout,
                        "im_posted": im_posted_val,
                        "assumed_pos": assumed})
                else:
                    cexp = abs(client.inventory) * usd * mid
                    was = client._stopped
                    client._stopped = (cm._stopped or
                                       (cexp > 0.0 and client.cash / cexp <= client_floor))
                    if client._stopped and not was:
                        self._cancel_open(client)   # a newly frozen client's standing quotes leave the book
                        self.freeze_log.append({"t": self.t, "client_id": client.agent_id,
                            "kind": type(client).__name__, "cm_id": cm.agent_id,
                            "cm_kind": "BCM" if cm.balance_sheet.is_banking else "NBCM",
                            "reason": "cm_contagion" if cm._stopped else "own_distress",
                            "kappa": client.cash / cexp if cexp > 0 else float("nan")})

        # ── Phase 2: CM own-account VM + CM capital / default ────────────────
        for cm in self.ccp.members.values():
            bs = cm.balance_sheet
            if bs.has_defaulted or bs.is_client:
                continue
            # VM on the own/assumed position. For a BCM this is its own-account book;
            # for an NBCM it is 0 unless it is carrying a defaulted client's position
            # it has assumed and is liquidating — marking it realises the
            # close-out loss as the fire-sale walks the price.
            own_pos = cm.inventory
            vm = (own_pos * (mid - bs._last_mark)
                  + bs._last_mark * cm._fill_qty - cm._fill_cost) * usd
            cm._fill_qty = 0; cm._fill_cost = 0.0
            cm.cash += vm
            bs.variation_margin += vm
            bs._last_mark = mid

            own_notional = abs(cm.inventory) * usd * mid
            exposure = own_notional + bs.client_notional(mid)
            bs.initial_margin = im_frac * exposure       # procyclical CCP IM (own + client book)
            bs.maintenance_margin = mm * bs.initial_margin
            bs.call_indicator = (bs.initial_margin > 0.0
                                 and abs(vm) > (1.0 - mm) * bs.initial_margin)
            met = True
            if G.IM_ESCROW and cm.cash > 0.0:
                # the CM posts IM on its OWN book; clients fund their own IM (Phase 0/1)
                met = self._post_im(cm, im_frac * own_notional)
            wf_level = 0
            if cm.cash <= 0.0 or not met:
                bs.has_defaulted = True
                self.ccp.default_list.append(cm.agent_id)
                deficit = -cm.cash                  # unpaid VM shortfall to mutualise
                cm.cash = 0.0
                # the defaulted CM is REMOVED (ODD §Step Sequence 13): it stops trading;
                # surviving clients are PORTED to other client CMs with spare capacity
                # (EMIR Art. 48(5)-(6)); the unported net position is the residual book.
                ccp_book = int(cm.inventory) + self._port_clients(cm, mid, im_frac,
                                                                  usd, floor)
                if G.CLOSEOUT_MODE == "transfer":
                    # Close out at CLOSEOUT_RECOVERY; the (1-recovery) haircut joins the loss.
                    # The defaulter's posted IM is seized FIRST against the TOTAL loss (VM
                    # shortfall + close-out haircut) — IM is the defaulter's own first-loss
                    # resource (EMIR Art. 45(1)) — so only the residual reaches the waterfall.
                    deficit += (1.0 - G.CLOSEOUT_RECOVERY) * abs(ccp_book) * usd * mid
                    if G.IM_ESCROW:
                        deficit = self._seize_im(cm, deficit)
                    wf_level = self.ccp.default_waterfall(cm, deficit)
                    # TRANSFER the residual book to the surviving member with the largest opposing
                    # position (prefer banking members — the real auction participants); inventory
                    # is preserved (no flatten), so VM stays zero-sum and cash is conserved.
                    rxers = [m for m in self.ccp.members.values()
                             if not m.balance_sheet.has_defaulted and not m.balance_sheet.is_client
                             and m.agent_id != cm.agent_id and m.balance_sheet.is_banking]
                    if not rxers:
                        rxers = [m for m in self.ccp.members.values()
                                 if not m.balance_sheet.has_defaulted
                                 and not m.balance_sheet.is_client and m.agent_id != cm.agent_id]
                    if self._transfer_book(ccp_book, rxers, mid) is None:
                        self._ccp_warehouse(ccp_book, cm, mid)
                    cm.inventory = 0
                else:
                    # firesale (deferred hypothesis): seize IM against the known VM shortfall;
                    # the open-market disposal loss is realised later and resumes the waterfall.
                    if G.IM_ESCROW:
                        deficit = self._seize_im(cm, deficit)
                    wf_level = self.ccp.default_waterfall(cm, deficit)
                    if (self.ccp._disposal_dcm is None and self.ccp.inventory == 0
                            and self.ccp._fill_qty == 0):
                        self.ccp._disposal_mark = mid
                    self.ccp.inventory += ccp_book
                    self.ccp._fill_qty += ccp_book
                    self.ccp._fill_cost += ccp_book * mid
                    self.ccp._disposal_dcm = cm
                    cm.inventory = 0
                    self.ccp.start_firesale(ccp_book, fs_urgency, AC_HORIZON)
                cm._stopped = True
                cm._liq_slices = []
                self._cancel_open(cm)
            elif (cm.capital_ratio(mid, sigma_t) <= (
                    (G.REG117_FLOOR_NBCM
                     if (getattr(G, "UNIFIED_CLIENT_FLOOR", False) or not bs.is_banking)
                     else G.LR_FLOOR_BCM)
                    if getattr(G, "DIFFERENTIATED_FLOORS", False) else floor)
                  or (bs.is_banking and getattr(G, "BASEL_LR_BCM", False) and exposure > 0.0
                      and cm.cash / exposure <= G.LR_FLOOR_BCM)):
                # Below a solvency floor. Client clearing: CFTC Reg 1.17 cash/IM >= REG117_FLOOR_NBCM
                # (both tiers, when UNIFIED_CLIENT_FLOOR). A BANK member is ADDITIONALLY held to the
                # Basel III / eSLR leverage ratio on its whole cleared book (cash/exposure >=
                # LR_FLOOR_BCM, BASEL_LR_BCM) — the bank-wide constraint that drives the leverage
                # cycle (Haynes-McPhail-Zhu 2019). On breach a BCM takes corrective action:
                # DELEVERAGE its OWN book (Almgren-Chriss) toward CAP_DELEVERAGE_TARGET (Basel
                # III leverage ratio / CFTC net-capital require reducing exposure; the book-walk
                # is the forced-deleveraging contagion channel, Thurner 2012 / Aymanns-Farmer
                # 2015). Only the own book can be shed intraday, so a client-book-dominated
                # member sheds all own risk and freezes on the residual. An NBCM has no own book
                # and just freezes — the ODD's BCM-deleverage / NBCM-stop asymmetry. Either tier
                # freezes NEW risk (own account + clients) while in breach (cm._stopped).
                cm._stopped = True
                if bs.is_banking and not cm._liq_slices:
                    _tgt = (G.DELEVERAGE_TARGET_BCM if getattr(G, "DIFFERENTIATED_FLOORS", False)
                            else G.CAP_DELEVERAGE_TARGET)
                    target_expo = (cm.cash / _tgt if _tgt > 0 else 0.0)
                    target_own = max(0.0, target_expo - bs.client_notional(mid))
                    target_inv = target_own / (usd * mid) if usd * mid > 0 else 0.0
                    cm.start_firesale(abs(cm.inventory) - target_inv, fs_urgency, AC_HORIZON)
            else:
                cm._stopped = False   # ratio recovered above the floor -> unfreeze (BCM or NBCM)

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
                "df_contribution": bs.df_contribution,
                "has_defaulted": bs.has_defaulted,
            })

        # ── settle the CCP disposal account. P&L on the assumed defaulted
        # book (mark-to-mid + fill slippage) RESUMES the originating default's
        # waterfall on a loss — symmetric with the deficit-consistent client
        # close-out — and accrues to the CCP on a gain. A synthetic
        # "CCP" history row records the level so the deepest-waterfall metric
        # includes disposal-period losses.
        ccp = self.ccp
        if ccp._disposal_dcm is not None or ccp.inventory != 0 or ccp._fill_qty:
            dpnl = (ccp.inventory * (mid - ccp._disposal_mark)
                    + ccp._disposal_mark * ccp._fill_qty - ccp._fill_cost) * usd
            ccp._fill_qty = 0; ccp._fill_cost = 0.0
            ccp._disposal_mark = mid
            if dpnl < 0 and ccp._disposal_dcm is not None:
                lvl = ccp.default_waterfall(ccp._disposal_dcm, -dpnl)
                self.clearing_history.append({
                    "t": self.t, "agent_id": ccp.ccp_id, "kind": "CCP",
                    "cash": ccp.cash, "own_position": ccp.inventory,
                    "client_notional": 0.0, "capital_ratio": float("nan"),
                    "initial_margin": 0.0, "maintenance_margin": 0.0,
                    "vm_cycle": dpnl, "vm_cumulative": 0.0,
                    "call_indicator": False, "waterfall_level": lvl,
                    "client_defaults": 0, "client_loss_absorbed": 0.0,
                    "has_defaulted": False})
            else:
                ccp.cash += dpnl
            if ccp.inventory == 0 and not ccp._liq_slices:
                ccp._disposal_dcm = None

    def run(self, n_steps: int) -> Dict[str, List]:
        for _ in range(n_steps):
            self.step()
        return self.history
