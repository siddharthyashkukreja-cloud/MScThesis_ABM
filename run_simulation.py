"""
Entry point — market layer + scaffolded clearing tier.

Market layer (54 LOB agents): 10 FT + 10 BCM (FT-cast, own-account) +
10 MT (single cohort) + 20 ZI + 4 MM (HFABM mid-anchored, D48).
Clearing tier (D28/D29): 5 NBCM + 1 CCP off-LOB; 5 of 10 BCMs carry
client books (plain FT + MT round-robin); ZI direct exchange participants.

Margin cycle (D29) runs every 60 ticks: VM settles the cleared book's
M2M into CM cash, IM/MM recomputed, capital ratio recorded. Note: VM
currently uses raw model qty — D51 (pending) will USD-denominate it
consistently with the D50 VOLUME_LOT relabeling.

1-min cadence (ODD-native): 390 steps per 6.5h RTH day. V_t exogenous,
read from data/fv_{regime}.csv (SV-MJD path; D33). The 6-d behavioural
θ is taken per regime from globals.CALIBRATED.
"""

import os
import numpy as np
import pandas as pd

from model.globals import (
    ModelParams, V0, CALIBRATED,
    CCP_CASH, BCM_CASH_RANGE, NBCM_CASH_RANGE, CCP_CALIBRATION,
)
from model.agents import (
    FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader, MarketMaker,
    VolatilityTrader, ContTrader,
    BankingClearingMember, NonBankingClearingMember,
)
from model.clearing import CentralCounterparty
from model.simulation import Simulation

BARS_PER_DAY = 390   # 6.5-hour RTH day at the 1-min cadence


def build_traders(params: ModelParams, seed: int) -> list:
    """LOB-trading population. BCM is cast from the FT pool (own-account
    FT-style); calibration POP sets n_bcm=0 and folds the BCM count into
    n_fundamental, so the calibrated θ carries straight over to runtime."""
    rng = np.random.default_rng(seed)
    traders: list = []
    counter = [0]

    def next_id() -> int:
        i = counter[0]
        counter[0] += 1
        return i

    for _ in range(params.n_fundamental):
        traders.append(FundamentalTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e6, 1e7)),
            z_score=float(rng.standard_normal())))

    for _ in range(params.n_bcm):
        traders.append(BankingClearingMember(
            agent_id=next_id(), cash=float(rng.uniform(*BCM_CASH_RANGE)),
            z_score=float(rng.standard_normal())))

    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e6, 1e7)),
            lambda_decay=params.mt_lambda))

    for _ in range(params.n_momentum_long):
        traders.append(MomentumTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e6, 1e7)),
            lambda_decay=params.mt_lambda_long))

    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e4, 1e5))))

    for _ in range(params.n_vt):
        traders.append(VolatilityTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e4, 1e5))))

    for _ in range(params.n_ct):
        traders.append(ContTrader(
            agent_id=next_id(), cash=float(rng.uniform(1e4, 1e5)),
            threshold=float(rng.uniform(0.0, 2.0 * params.sigma_v))))

    for _ in range(params.n_mm):
        traders.append(MarketMaker(
            agent_id=next_id(), cash=float(rng.uniform(1e7, 1e8))))

    return traders


def build_clearing_tier(traders: list, params: ModelParams, seed: int):
    """Build the central-clearing scaffold (D28/D29): one CentralCounterparty,
    n_nbcm NonBankingClearingMembers, the bidirectional CM<->CCP links and
    the client-book assignment. All BCMs + NBCMs are CCP members; but only
    `n_bcm_with_clients` BCMs carry a client book — the rest are own-account
    only (D29 — client-book concentration as a study lever). The plain FT
    and MT clear through the client-carrying CMs (round-robin); ZI stay
    direct exchange participants (ODD §Initialization). Returns the
    CentralCounterparty (which holds every CM via `members`)."""
    rng = np.random.default_rng(seed + 7)
    start_id = len(traders)

    ccp = CentralCounterparty(
        ccp_id=start_id + params.n_nbcm, cash=float(CCP_CASH),
        own_df=CCP_CALIBRATION["ex_df_ratio"] * CCP_CASH)

    nbcms = [NonBankingClearingMember(
                 agent_id=start_id + k,
                 cash=float(rng.uniform(*NBCM_CASH_RANGE)))
             for k in range(params.n_nbcm)]

    bcms = [t for t in traders if isinstance(t, BankingClearingMember)]
    for cm in bcms + nbcms:
        ccp.register_member(cm)

    clients = [t for t in traders
               if isinstance(t, (FundamentalTrader, MomentumTrader))
               and not isinstance(t, BankingClearingMember)]
    # Only half the BCMs carry clients (D29); the rest are own-account only.
    client_cms = bcms[:params.n_bcm_with_clients] + nbcms
    for i, client in enumerate(clients):
        cm = client_cms[i % len(client_cms)]
        client.clearing_member_id = cm.agent_id
        cm.client_ids.append(client.agent_id)
        cm.balance_sheet.client_positions[client.agent_id] = 0

    return ccp


def main():
    stressed = False
    regime = "stressed" if stressed else "calm"

    # 50 LOB agents: 10 FT + 10 BCM + 10 MT + 20 ZI (no MM — removed; geometric
    # data-fit placement supplies near-mid liquidity). 5 NBCM + 1 CCP off-LOB;
    # 5 of 10 BCMs carry client books (D29). 3-d θ from CALIBRATED (zi_alpha,
    # zi_mu, zi_delta); p_zi / qty_max / σ_v auto-populate from globals per regime.
    params = ModelParams(
        n_fundamental=10, n_momentum=10, n_momentum_long=0,
        n_mm=0, n_zi=20, n_vt=0, n_ct=0,
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        **CALIBRATED[regime],
        stressed=stressed,
    )

    traders = build_traders(params, seed=42)
    ccp = build_clearing_tier(traders, params, seed=42)
    sim = Simulation(params, traders, seed=42, ccp=ccp)
    history = sim.run(n_steps=BARS_PER_DAY)   # one RTH day

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(history).to_csv(f"output/run_{regime}.csv", index=False)
    # Clearing-tier observables — one row per CM per margin cycle (D29).
    if sim.clearing_history:
        pd.DataFrame(sim.clearing_history).to_csv(
            f"output/clearing_{regime}.csv", index=False)


if __name__ == "__main__":
    main()
