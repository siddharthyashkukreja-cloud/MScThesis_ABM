"""
Entry point for Stage 3: LOB + ZI direct + BCM + NBCM + client-clearing tier.

Topology (ODD-faithful + clients extension):
- 10 ZI direct
- 10 BCM (FundamentalTrader behaviour, V+z*sigma placement); 3 of 10 have a
  client book attached.
- 10 NBCM (passive structural entity; never trades on own account); ALL 10
  have a client book.
- Each client book = 1 FT + 1 MT + 1 ZI = 3 clients (cleared via that CM).

Cash ranges per ODD §Initialization scaled to SPY-relevant magnitudes:
BCM ~U[5B, 10B], NBCM ~U[5M, 10M], retail/direct ~U[100k, 1M].

Fundamental V_t evolves under Merton (1976) jump-diffusion; jumps off in
Stage 3 baseline. Calibration of all parameters lives in the calibration
stage.
"""

import os
import numpy as np
import pandas as pd

from model.globals import ModelParams
from model.agents import (
    ZeroIntelligenceTrader, FundamentalTrader, MomentumTrader,
    BankingClearingMember, NonBankingClearingMember,
)
from model.simulation import Simulation


def _make_client_book(rng, next_id, clearing_member_id, params):
    """Per-CM client book: params.client_book_ft FTs + .._mt MTs + .._zi ZIs.

    Cash per client by type:
      - FT/MT (institutional): U[$1M, $10M]
      - ZI    (retail):        U[$10k, $100k]
    """
    clients = []
    for _ in range(params.client_book_ft):
        cid = next_id()
        clients.append(FundamentalTrader(
            agent_id=cid, cash=float(rng.uniform(1e6, 1e7)),
            z_score=float(rng.standard_normal()),
            clearing_member_id=clearing_member_id))
    for _ in range(params.client_book_mt):
        cid = next_id()
        clients.append(MomentumTrader(
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

    # 1) Direct ZI
    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e5, 1e6))))

    # 2) Direct FT / MT (zero by default in Stage 3; kept for flexibility)
    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e5, 1e6)),
            z_score=float(rng.standard_normal())))
    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e5, 1e6)),
            z_score=float(rng.standard_normal())))

    # 3) Banking CMs (n_bcm_mm and n_bcm_with_clients are independent flags):
    #    - First n_bcm_mm BCMs run as market makers (mode='market_maker').
    #    - First n_bcm_with_clients BCMs (regardless of mode) carry clients.
    #    - Real-bank-tier default: MMs ALSO clear for clients
    #      (e.g., GS/JPM/Morgan Stanley both market-make and run prime brokerage).
    for i in range(params.n_bcm):
        is_mm = i < params.n_bcm_mm
        bcm = BankingClearingMember(
            agent_id=next_id(),
            cash=float(rng.uniform(5e9, 10e9)),     # ODD: ~U[5B, 10B]
            z_score=float(rng.standard_normal()),
            mode="market_maker" if is_mm else "fundamental",
        )
        traders.append(bcm)
        if i < params.n_bcm_with_clients:
            book = _make_client_book(rng, next_id, bcm.agent_id, params)
            bcm.client_ids.extend(c.agent_id for c in book)
            traders.extend(book)

    # 4) Non-Banking CMs (all carry client books)
    for _ in range(params.n_nbcm):
        nbcm = NonBankingClearingMember(
            agent_id=next_id(),
            cash=float(rng.uniform(5e6, 10e6)),     # ODD: ~U[5M, 10M]
        )
        traders.append(nbcm)
        book = _make_client_book(rng, next_id, nbcm.agent_id, params)
        nbcm.client_ids.extend(c.agent_id for c in book)
        traders.extend(book)

    return traders


def main():
    params = ModelParams(
        n_zi=0, n_fundamental=0, n_momentum=0,                     # no direct
        n_bcm=15, n_bcm_mm=8, n_bcm_with_clients=8, n_nbcm=5,      # 8 MM-clearing, 7 FT-prop, 5 NBCM
        clients_per_book=6, client_book_ft=2, client_book_mt=2, client_book_zi=2,
        v0=450.0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,   # rates / minute
        zi_qty_min=1, zi_qty_max=10,
        zi_offset_p=0.5, zi_offset_max=20,
        ft_sigma_rel=0.005,
        mt_sigma_rel=0.005, mt_lambda_ewma=0.95, mt_threshold=1e-4,
        mu_v=0.0, sigma_v=0.001,
        jump_lambda=0.0385, jump_mean=0.0, jump_std=0.01,   # ODD: 3 jumps/day = 3/78 per 5-min
        mm_half_spread_bps=30.0, mm_qty=50, mm_inventory_skew_bps=0.5,
    )

    traders = build_traders(params, seed=42)
    sim = Simulation(params, traders, seed=42)
    history = sim.run(n_steps=78)

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(history).to_csv("output/stage3_run.csv", index=False)


if __name__ == "__main__":
    main()
