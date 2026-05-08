"""
Stage 3 invariants and behavioural tests (LOB + ZI direct + BCM + NBCM
+ client clearing tier).

Covers:
- Determinism with the full Stage 3 population.
- Inventory + cash conservation across all market participants.
- BCM capital_ratio formula (own + client aggregate exposure).
- NBCM capital_ratio formula (client aggregate only; own inventory == 0).
- BCM cap-ratio breach -> fire-sale liquidation reduces |inventory|.
- NBCM cap-ratio breach -> stop-out (no own orders; trivially satisfied
  since NBCM never trades anyway, but we still verify zero inventory).
- Client linkage integrity: each client.clearing_member_id matches a CM
  whose client_ids list contains their id.
- ODD V&V CapitalRatioSeparation: BCM under stress liquidates while
  NBCM does not submit any orders of its own.
"""
import unittest
import numpy as np

from model.globals import ModelParams
from model.agents import (
    ZeroIntelligenceTrader, FundamentalTrader, MomentumTrader,
    BankingClearingMember, NonBankingClearingMember,
)
from model.simulation import Simulation
from run_simulation import build_traders


def _params(**overrides):
    base = dict(
        n_zi=0, n_fundamental=0, n_momentum=0,
        n_bcm=15, n_bcm_mm=8, n_bcm_with_clients=8, n_nbcm=5,
        clients_per_book=6, client_book_ft=2, client_book_mt=2, client_book_zi=2,
        v0=450.0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10,
        zi_offset_p=0.5, zi_offset_max=20,
        ft_sigma_rel=0.005,
        mt_sigma_rel=0.005, mt_lambda_ewma=0.95, mt_threshold=1e-4,
        mu_v=0.0, sigma_v=0.001,
        jump_lambda=0.0, jump_mean=0.0, jump_std=0.01,
        mm_half_spread_bps=30.0, mm_qty=50, mm_inventory_skew_bps=0.5,
    )
    base.update(overrides)
    return ModelParams(**base)


class TestStage3(unittest.TestCase):

    def test_determinism(self):
        p = _params()
        h1 = Simulation(p, build_traders(p, 42), seed=42).run(120)
        h2 = Simulation(p, build_traders(p, 42), seed=42).run(120)
        for k in h1:
            for a, b in zip(h1[k], h2[k]):
                if isinstance(a, float) and np.isnan(a) and np.isnan(b):
                    continue
                self.assertEqual(a, b, f"divergence in field {k}")

    def test_inventory_conservation(self):
        """Sum of all agent inventories == 0 at every step (fills come in pairs)."""
        p = _params()
        sim = Simulation(p, build_traders(p, 11), seed=11)
        for _ in range(100):
            sim.step()
            total = sum(t.inventory for t in sim.traders)
            self.assertEqual(total, 0,
                             f"inventory imbalance {total} at t={sim.gs.t}")

    def test_cash_conservation(self):
        """Sum of all agent cash is constant (cash flows agent-to-agent).

        Use relative tolerance: BCM cash ~$5-10B per agent => total cash
        on the order of 75B. Floating-point accumulation over thousands of
        fills produces ~ 1e-16 relative drift, which is at machine epsilon.
        """
        p = _params()
        traders = build_traders(p, 13)
        initial_cash = sum(t.cash for t in traders)
        sim = Simulation(p, traders, seed=13)
        sim.run(100)
        final_cash = sum(t.cash for t in sim.traders)
        rel = abs(final_cash - initial_cash) / max(abs(initial_cash), 1.0)
        self.assertLess(rel, 1e-12,
                        f"cash relative drift {rel:.2e} too large "
                        f"(initial={initial_cash:.4e}, final={final_cash:.4e})")

    def test_client_linkage_integrity(self):
        """Every client.clearing_member_id points to a CM whose client_ids includes them."""
        p = _params()
        traders = build_traders(p, 17)
        by_id = {t.agent_id: t for t in traders}
        for t in traders:
            if t.clearing_member_id is not None:
                cm = by_id.get(t.clearing_member_id)
                self.assertIsNotNone(cm, f"agent {t.agent_id} cites missing CM")
                self.assertIn(t.agent_id, cm.client_ids,
                              f"agent {t.agent_id} not in CM {cm.agent_id} client_ids")

    def test_nbcm_never_trades(self):
        """NBCM inventory stays exactly 0 across the run."""
        p = _params()
        sim = Simulation(p, build_traders(p, 19), seed=19)
        sim.run(300)
        for t in sim.traders:
            if isinstance(t, NonBankingClearingMember):
                self.assertEqual(t.inventory, 0,
                                 f"NBCM {t.agent_id} accrued inventory {t.inventory}")

    def test_bcm_cap_ratio_aggregates_clients(self):
        """BCM capital ratio includes own + client inventory in denominator."""
        p = _params()
        traders = build_traders(p, 23)
        by_id = {t.agent_id: t for t in traders}
        bcm = next(t for t in traders if isinstance(t, BankingClearingMember)
                   and t.client_ids)
        # Manually set inventories to known values
        bcm.inventory = 100
        for cid in bcm.client_ids:
            by_id[cid].inventory = 50
        mid = 10.0
        own = abs(bcm.inventory) * mid
        clients = sum(abs(by_id[c].inventory) * mid for c in bcm.client_ids)
        expected = bcm.cash / (own + clients)
        actual = bcm.capital_ratio(mid, by_id)
        self.assertAlmostEqual(actual, expected, places=8)

    def test_nbcm_cap_ratio_clients_only(self):
        """NBCM cap_ratio uses ONLY client exposure (own inventory is 0)."""
        p = _params()
        traders = build_traders(p, 29)
        by_id = {t.agent_id: t for t in traders}
        nbcm = next(t for t in traders if isinstance(t, NonBankingClearingMember))
        for cid in nbcm.client_ids:
            by_id[cid].inventory = 25
        mid = 10.0
        clients = sum(abs(by_id[c].inventory) * mid for c in nbcm.client_ids)
        expected = nbcm.cash / clients
        actual = nbcm.capital_ratio(mid, by_id)
        self.assertAlmostEqual(actual, expected, places=8)

    def test_bcm_mm_posts_two_sided_quotes(self):
        """MM-mode BCM posts a bid and an ask each step (after warm-up)."""
        p = _params(n_zi=0, n_fundamental=0, n_momentum=0,
                    n_bcm=2, n_bcm_mm=2, n_bcm_with_clients=0, n_nbcm=0)
        traders = build_traders(p, 41)
        sim = Simulation(p, traders, seed=41)
        sim.run(20)
        # After warm-up the LOB should have both bid and ask depth from MM quotes
        post = [s for s in zip(sim.history["bid_depth"], sim.history["ask_depth"])][5:]
        two_sided = sum(1 for b, a in post if b > 0 and a > 0)
        self.assertGreater(two_sided / max(len(post), 1), 0.8,
                           "MM not maintaining two-sided book")

    def test_bcm_mm_inventory_skew(self):
        """Long inventory shifts both quotes down; short shifts up."""
        p = _params()
        traders = build_traders(p, 51)
        mm = next(t for t in traders if isinstance(t, BankingClearingMember)
                  and t.mode == "market_maker")
        from model.lob import LOB
        from model.globals import SimContext
        import numpy as np

        rng = np.random.default_rng(0)
        # Use a price-scale (V=100) where bps shifts are within tick precision
        # at sufficient inventory. mm_inventory_skew_bps default = 0.5 -> need
        # at least 1 unit of inventory to shift > tick_size=0.01 at V=100
        # (1 unit -> 0.5bps = 0.005 which rounds to tick_size). Use 1000+ for
        # clear directional shift.
        ctx = SimContext(v=100.0, mid_price=100.0, momentum=0.0, tick=0,
                         traders_by_id={mm.agent_id: mm})

        def quote_prices(inv):
            lob = LOB(p.tick_size, p.order_ttl)
            mm.inventory = inv
            mm._mm_order_ids = []
            mm.submit_orders(lob, p, ctx, rng)
            lob.match()
            return float(lob.best_bid), float(lob.best_ask)

        bid_zero, ask_zero = quote_prices(0)
        bid_long, ask_long = quote_prices(10000)   # large inventory for visible shift
        bid_short, ask_short = quote_prices(-10000)

        self.assertLess(bid_long, bid_zero, "long inventory should lower bid")
        self.assertLess(ask_long, ask_zero, "long inventory should lower ask")
        self.assertGreater(bid_short, bid_zero, "short inventory should raise bid")
        self.assertGreater(ask_short, ask_zero, "short inventory should raise ask")
        mm.inventory = 0

    def test_bcm_fire_sale_liquidates(self):
        """Manually-stressed BCM (low cash, large position) liquidates inventory.

        ODD §V&V CapitalRatioSeparation: BCM under cap-ratio breach
        submits deleveraging orders; NBCM does not.
        """
        p = _params(n_zi=20, n_bcm=1, n_nbcm=0, n_bcm_with_clients=0,
                    n_fundamental=0, n_momentum=0)
        traders = build_traders(p, 31)
        bcm = next(t for t in traders if isinstance(t, BankingClearingMember))
        # Force breach: tiny cash, large positive inventory
        bcm.cash = 100.0
        bcm.inventory = 200
        sim = Simulation(p, traders, seed=31)
        start_inv = bcm.inventory
        sim.run(80)
        self.assertLess(bcm.inventory, start_inv,
                        "BCM under breach failed to reduce inventory")


if __name__ == "__main__":
    unittest.main()
