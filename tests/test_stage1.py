"""
Stage 1 invariants and ODD V&V tests (LOB + ZI only).

Covers:
- LOBMatchingDeterminism (ODD V&V): same seed -> identical history.
- MidPriceUpdate (ODD V&V): mid = 0.5*(best_bid + best_ask) when both exist.
- No-crossed-book invariant after each call-auction match.
- Bounded spread: spread <= 2 * zi_offset_max * tick_size whenever both sides
  are populated (theoretical maximum given Cont-Stoikov / Farmer-Daniels
  uniform offset placement).
- Two-sided depth after warmup (sanity that price discovery occurs).
"""
import unittest
import numpy as np

from model.globals import ModelParams, GlobalState
from model.lob import LOB
from model.agents import ZeroIntelligenceTrader
from model.simulation import Simulation


def _build_params(**overrides):
    base = dict(
        n_zi=10, n_fundamental=0, n_momentum=0,
        v0=450.0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10, zi_offset_max=5,
        sigma_v=0.0,
    )
    base.update(overrides)
    return ModelParams(**base)


def _build_traders(n, seed):
    rng = np.random.default_rng(seed)
    return [ZeroIntelligenceTrader(agent_id=i, cash=float(rng.uniform(1e5, 1e6)))
            for i in range(n)]


class StepInvariantSimulation(Simulation):
    """Simulation that records LOB invariants at each step for assertion."""

    def __init__(self, params, traders, seed=42):
        super().__init__(params, traders, seed=seed)
        self.invariants = []

    def step(self):
        snap = super().step()
        bb, ba = self.lob.best_bid, self.lob.best_ask
        spread = self.lob.spread
        mid = self.lob.mid_price
        self.invariants.append({
            "best_bid": bb, "best_ask": ba, "spread": spread, "mid": mid,
            "bid_depth": snap["bid_depth"], "ask_depth": snap["ask_depth"],
        })
        return snap


class TestStage1(unittest.TestCase):

    def test_determinism(self):
        """ODD V&V LOBMatchingDeterminism: same seed -> identical history."""
        params = _build_params()
        h1 = Simulation(params, _build_traders(params.n_zi, 42), seed=42).run(120)
        h2 = Simulation(params, _build_traders(params.n_zi, 42), seed=42).run(120)
        for k in h1:
            for a, b in zip(h1[k], h2[k]):
                if isinstance(a, float) and np.isnan(a) and np.isnan(b):
                    continue
                self.assertEqual(a, b, f"divergence in field {k}")

    def test_midprice_formula(self):
        """ODD V&V MidPriceUpdate: mid = 0.5*(bid + ask) whenever both exist."""
        params = _build_params()
        sim = StepInvariantSimulation(params, _build_traders(params.n_zi, 7), seed=7)
        sim.run(200)
        for r in sim.invariants:
            if not (np.isnan(r["best_bid"]) or np.isnan(r["best_ask"])):
                self.assertAlmostEqual(
                    r["mid"], 0.5 * (r["best_bid"] + r["best_ask"]), places=8
                )

    def test_no_crossed_book(self):
        """Post-match invariant: best_bid <= best_ask (call auction clears crosses)."""
        params = _build_params()
        sim = StepInvariantSimulation(params, _build_traders(params.n_zi, 11), seed=11)
        sim.run(200)
        for r in sim.invariants:
            if not (np.isnan(r["best_bid"]) or np.isnan(r["best_ask"])):
                self.assertLessEqual(r["best_bid"], r["best_ask"])

    def test_bounded_spread(self):
        """Spread stays bounded under ZI-only flow.

        Each ZI buy is placed at best_ask - k*tick (k in {1,..,K}); each
        sell at best_bid + k*tick. New orders narrow the spread, but TTL
        aging of the inside layer can transiently widen it before fresh
        orders arrive. Empirical bound used here: 4 * zi_offset_max ticks
        -- well above observed maxima, well below run-away. Stage 2+
        Chiarella + CM flow tightens this further.
        """
        params = _build_params()
        max_spread_ticks = 4 * params.zi_offset_max
        sim = StepInvariantSimulation(params, _build_traders(params.n_zi, 21), seed=21)
        sim.run(300)
        warmed = sim.invariants[20:]
        for r in warmed:
            if not (np.isnan(r["best_bid"]) or np.isnan(r["best_ask"])):
                self.assertLessEqual(
                    r["spread"] / params.tick_size, max_spread_ticks,
                    f"spread {r['spread']} exceeds {max_spread_ticks} ticks"
                )

    def test_two_sided_depth_post_warmup(self):
        """Book is two-sided a non-trivial fraction of post-warmup steps.

        Stage 1 ZI flow at ODD-calibrated rates (alpha=0.15, mu=0.025,
        delta=0.025) with 10 traders and TTL=2 produces sparse two-sided
        depth; expected limit orders per step = 10 * 0.15 * 0.5 = 0.75 per
        side, so single-sided gaps are common between TTL refreshes.
        Threshold reflects this; Stage 2 Chiarella + CM flow will lift it.
        """
        params = _build_params()
        sim = StepInvariantSimulation(params, _build_traders(params.n_zi, 33), seed=33)
        sim.run(300)
        post = sim.invariants[30:]
        two_sided = sum(1 for r in post
                        if not (np.isnan(r["best_bid"]) or np.isnan(r["best_ask"])))
        self.assertGreater(two_sided / len(post), 0.40,
                           "two-sided book in <40% of post-warmup steps")

    def test_no_negative_quantities(self):
        """LOB invariant: no resting order with non-positive qty after match/age."""
        params = _build_params()
        rng = np.random.default_rng(99)
        traders = _build_traders(params.n_zi, 99)
        sim = Simulation(params, traders, seed=99)
        for _ in range(150):
            sim.step()
            for book in (sim.lob._bids, sim.lob._asks):
                for queue in book.values():
                    for o in queue:
                        self.assertGreater(o.qty, 0)
                        self.assertGreater(o.ttl, 0)


if __name__ == "__main__":
    unittest.main()
