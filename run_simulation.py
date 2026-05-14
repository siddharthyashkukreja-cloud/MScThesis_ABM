"""
Entry point. Real-bank-tier population (98 agents):
  - 15 BCM: 8 MM-mode (with clients), 7 FT-prop (no clients)
  - 5  NBCM (all with clients)
  - 78 clients: 3 FT + 3 ZI per book × 13 books (8 BCM-MM books + 5 NBCM books)

Cash distribution (ODD §Initialization, claudereadme D14):
  BCM ~ U[5B, 10B]  | NBCM ~ U[5M, 10M]
  FT client ~ U[1M, 10M]  | ZI client ~ U[10k, 100k]

V_t is exogenous (loaded from params.fv_csv); no in-sim diffusion / jumps.
"""

import os
import numpy as np
import pandas as pd

from model.globals import ModelParams, V0
from model.agents import (
    ZeroIntelligenceTrader, FundamentalTrader,
    BankingClearingMember, NonBankingClearingMember,
)
from model.simulation import Simulation


def _make_client_book(rng, next_id, clearing_member_id, params):
    """Per-CM client book: params.client_book_ft FTs + .._zi ZIs."""
    clients = []
    for _ in range(params.client_book_ft):
        cid = next_id()
        clients.append(FundamentalTrader(
            agent_id=cid, cash=float(rng.uniform(1e6, 1e7)),
            z_score=float(rng.standard_normal()),
            clearing_member_id=clearing_member_id))
    for _ in range(params.client_book_zi):
        cid = next_id()
        clients.append(ZeroIntelligenceTrader(
            agent_id=cid, cash=float(rng.uniform(1e4, 1e5)),
            clearing_member_id=clearing_member_id))
    return clients


def build_traders(params: ModelParams, seed: int) -> list:
    rng = np.random.default_rng(seed)
    traders: list = []
    counter = [0]

    def next_id() -> int:
        i = counter[0]
        counter[0] += 1
        return i

    # 1) Direct ZI (0 by default)
    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e5, 1e6))))

    # 2) Direct FT (0 by default)
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e5, 1e6)),
            z_score=float(rng.standard_normal())))

    # 3) Banking CMs
    for i in range(params.n_bcm):
        is_mm = i < params.n_bcm_mm
        bcm = BankingClearingMember(
            agent_id=next_id(),
            cash=float(rng.uniform(5e9, 10e9)),
            z_score=float(rng.standard_normal()),
            mode="market_maker" if is_mm else "fundamental",
        )
        traders.append(bcm)
        if i < params.n_bcm_with_clients:
            book = _make_client_book(rng, next_id, bcm.agent_id, params)
            bcm.client_ids.extend(c.agent_id for c in book)
            traders.extend(book)

    # 4) Non-Banking CMs (all carry clients)
    for _ in range(params.n_nbcm):
        nbcm = NonBankingClearingMember(
            agent_id=next_id(),
            cash=float(rng.uniform(5e6, 10e6)),
        )
        traders.append(nbcm)
        book = _make_client_book(rng, next_id, nbcm.agent_id, params)
        nbcm.client_ids.extend(c.agent_id for c in book)
        traders.extend(book)

    return traders


def main():
    stressed = False
    regime = "stressed" if stressed else "calm"

    params = ModelParams(
        # Direct populations
        n_zi=0, n_fundamental=0, n_mm=0,
        # Clearing tier
        n_bcm=15, n_bcm_mm=8, n_bcm_with_clients=8, n_nbcm=5,
        clients_per_book=6, client_book_ft=3, client_book_zi=3,
        # Asset / cadence
        v0=V0[regime], tick_size=0.25, dt_minutes=5.0, order_ttl=2,
        # ZI
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        p_zi=0.30,
        # FT — ft_alpha=0.5 is a non-degenerate operating point (mid tracks V,
        #      corr≈0.93); the true value is set by the agent-calibration loop.
        ft_alpha=0.5, ft_sigma_c=10.0,
        # MM (HFABM defaults)
        mm_qty=50, mm_p_edge=4,
        mm_inventory_limit=1000, mm_inventory_safe=500,
        # Regime
        stressed=stressed,
    )

    traders = build_traders(params, seed=42)
    sim = Simulation(params, traders, seed=42)
    history = sim.run(n_steps=78)

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(history).to_csv(f"output/run_{regime}.csv", index=False)


if __name__ == "__main__":
    main()
