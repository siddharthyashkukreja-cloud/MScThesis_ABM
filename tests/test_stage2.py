"""
Stage 2 invariants and behavioural tests (LOB + ZI + FT + MT).

Covers:
- Determinism still holds with the unified SimContext + FT/MT in the loop.
- FT mean-reversion: with V_t frozen (sigma_v=0, jump_lambda=0), an FT-only
  population drives P_mid toward V (ODD §Mech #1: HeterogeneousPrivateValuation
  -> persistent reversion to V).
- MT trend-following: under a deterministic upward V drift (sigma_v=0,
  mu_v>0), the EWMA momentum signal becomes positive after warmup so MT
  fires predominantly buy-side orders.
- Full Stage 2 run sanity: 78-step sim with mixed population produces a
  finite, two-sided history with non-zero volume.
"""
import unittest
import numpy as np

from model.globals import ModelParams
from model.agents import (
    ZeroIntelligenceTrader, FundamentalTrader, MomentumTrader,
)
from model.simulation import Simulation


def _params(**overrides):
    base = dict(
        n_zi=10, n_fundamental=10, n_momentum=10,
        v0=450.0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10, zi_offset_max=5,
        ft_sigma=0.5,
        mt_sigma=0.5, mt_lambda_ewma=0.95, mt_threshold=1e-4,
        mu_v=0.0, sigma_v=0.001,
        jump_lambda=0.0, jump_mean=0.0, jump_std=0.01,
    )
    base.update(overrides)
    return ModelParams(**base)


def _build(params, seed):
    rng = np.random.default_rng(seed)
    traders, aid = [], 0
    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(agent_id=aid, cash=float(rng.uniform(1e5, 1e6))))
        aid += 1
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(
            agent_id=aid, cash=float(rng.uniform(1e5, 1e6)),
            z_score=float(rng.standard_normal())))
        aid += 1
    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(
            agent_id=aid, cash=float(rng.uniform(1e5, 1e6)),
            z_score=float(rng.standard_normal())))
        aid += 1
    return traders


class TestStage2(unittest.TestCase):

    def test_determinism(self):
        """Same seed -> identical history, mixed population."""
        p = _params()
        h1 = Simulation(p, _build(p, 42), seed=42).run(120)
        h2 = Simulation(p, _build(p, 42), seed=42).run(120)
        for k in h1:
            for a, b in zip(h1[k], h2[k]):
                if isinstance(a, float) and np.isnan(a) and np.isnan(b):
                    continue
                self.assertEqual(a, b, f"divergence in field {k}")

    def test_ft_mean_reversion(self):
        """FT-only with frozen V: P_mid stays close to V after warmup.

        V is constant at v0. FTs with z ~ N(0,1) have reservations spread
        around V, so the marketable side flips as P drifts away from V,
        keeping mid bounded around V0. Bound: |mid - V| <= 4 * ft_sigma
        (covers ~4-sigma tail of reservation distribution) post-warmup.
        """
        p = _params(n_zi=5, n_fundamental=20, n_momentum=0,
                    sigma_v=0.0, mu_v=0.0, jump_lambda=0.0)
        sim = Simulation(p, _build(p, 7), seed=7)
        sim.run(300)
        mids = np.array([m for m in sim.history["mid_price"][50:]
                         if not np.isnan(m)])
        self.assertGreater(len(mids), 0)
        max_dev = float(np.max(np.abs(mids - p.v0)))
        self.assertLessEqual(
            max_dev, 4.0 * p.ft_sigma,
            f"mid wandered {max_dev:.3f} from V (bound 4*ft_sigma={4*p.ft_sigma})"
        )

    def test_mt_trend_following(self):
        """V trending upward => MT order flow tilts long.

        With sigma_v=0 and a small positive mu_v, V drifts deterministically
        upward; recent log-returns become positive so EWMA momentum > 0;
        MT side should be +1 in the majority of triggered submissions.
        Test reads MT-side bias indirectly via the sign of the cumulative
        momentum signal post-warmup (which is what determines MT side).
        """
        p = _params(n_zi=0, n_fundamental=0, n_momentum=20,
                    sigma_v=0.0, mu_v=0.001, jump_lambda=0.0,
                    mt_threshold=0.0)
        sim = Simulation(p, _build(p, 9), seed=9)
        sim.run(200)
        post = sim.history["momentum"][50:]
        positive = sum(1 for m in post if m > 0)
        self.assertGreater(
            positive / len(post), 0.7,
            "momentum signal not predominantly positive under upward V drift"
        )

    def test_full_run_sanity(self):
        """78-step mixed-population run has finite mid + non-zero volume."""
        p = _params()
        sim = Simulation(p, _build(p, 42), seed=42)
        h = sim.run(78)
        mids = [m for m in h["mid_price"] if not np.isnan(m)]
        self.assertGreater(len(mids), 60, "too few non-NaN mid prices")
        self.assertGreater(sum(h["volume"]), 0, "no fills produced in Stage 2")
        self.assertTrue(
            all(np.isfinite(v) for v in h["fundamental"]),
            "Merton jump-diffusion produced non-finite V"
        )

    def test_jump_diffusion_no_drift_when_off(self):
        """sigma_v=0, jump_lambda=0 => V is exactly constant."""
        p = _params(sigma_v=0.0, mu_v=0.0, jump_lambda=0.0)
        sim = Simulation(p, _build(p, 1), seed=1)
        sim.run(50)
        self.assertTrue(all(v == p.v0 for v in sim.history["fundamental"]))

    def test_jump_diffusion_jumps_on(self):
        """jump_lambda>0 produces non-zero log-return jumps in V."""
        p = _params(sigma_v=0.0, mu_v=0.0,
                    jump_lambda=0.5, jump_mean=0.0, jump_std=0.05)
        sim = Simulation(p, _build(p, 3), seed=3)
        sim.run(200)
        v = np.array(sim.history["fundamental"])
        log_ret = np.diff(np.log(v))
        self.assertGreater(float(np.std(log_ret)), 0.0,
                           "jumps off but expected jump-driven dispersion")


if __name__ == "__main__":
    unittest.main()
