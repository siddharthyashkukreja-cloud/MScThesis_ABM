"""
Entry point — market layer + central-clearing tier.

Market layer (100 LOB agents): 30 FT + 10 BCM (FT-cast, own-account) +
20 MT + 40 ZI. No market maker — geometric data-fit placement supplies the
near-mid liquidity.
Clearing tier: 5 NBCM + 1 CCP off-LOB; 5 of 10 BCMs carry client books,
all 5 NBCMs do; the 90 plain FT + MT + ZI clear through them with skewed
5-15 books (ZI clearing is a thesis extension to the ODD). The H1
counterfactual `build_clearing_tier(..., direct=True)` removes the client
tier entirely — every end user clears directly at the CCP.

Margin cycle runs every 60 min: USD variation margin (× VOLUME_LOT ×
CONTRACT_USD) settles the cleared book's M2M into CM cash, IM/MM and the
capital ratio recomputed, margin call raised; a capital-ratio breach
triggers an Almgren-Chriss deleverage, cash exhaustion a default + the
5-level waterfall + a CCP fire-sale. The cover-2 default fund is
recomputed each RTH day.

1-min cadence (ODD-native): 390 steps per 6.5h RTH day. V_t exogenous,
read from data/fv_{regime}.csv (Kalman efficient price; data/v_kalman.py).
The behavioural θ is taken per regime from globals.CALIBRATED.
"""

import os
import numpy as np
import pandas as pd

from model.globals import (
    ModelParams, V0, CALIBRATED,
    CCP_CASH, BCM_CASH_RANGE, NBCM_CASH_RANGE, CCP_CALIBRATION,
    FT_CLIENT_CASH, MT_CLIENT_CASH, ZI_CLIENT_CASH,
)
from model.agents import (
    FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
    BankingClearingMember, NonBankingClearingMember,
)
from model import globals as G
from model.clearing import CentralCounterparty
from model.simulation import Simulation

BARS_PER_DAY = 390   # 6.5-hour RTH day at the 1-min cadence


def build_traders(params: ModelParams, seed: int) -> list:
    """LOB-trading population. BCM is cast from the FT pool (own-account,
    FT-style); calibration sets n_bcm=0 and folds the BCM count into
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
            agent_id=next_id(), cash=float(rng.uniform(*FT_CLIENT_CASH)),
            z_score=float(rng.standard_normal())))

    for _ in range(params.n_bcm):
        traders.append(BankingClearingMember(
            agent_id=next_id(), cash=float(rng.uniform(*BCM_CASH_RANGE)),
            z_score=float(rng.standard_normal())))

    for _ in range(params.n_momentum):
        traders.append(MomentumTrader(
            agent_id=next_id(), cash=float(rng.uniform(*MT_CLIENT_CASH)),
            lambda_decay=params.mt_lambda))

    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(
            agent_id=next_id(), cash=float(rng.uniform(*ZI_CLIENT_CASH))))

    return traders


def build_clearing_tier(traders: list, params: ModelParams, seed: int,
                        direct: bool = False):
    """Build the central-clearing scaffold: one CentralCounterparty, n_nbcm
    NonBankingClearingMembers, the bidirectional CM<->CCP links and the
    client-book assignment. All BCMs + NBCMs are CCP members, but only
    `n_bcm_with_clients` BCMs carry a client book — the rest are own-account
    only. The plain FT, MT and ZI clear through the client-carrying CMs by a
    leverage-balanced (capacity-proportional) rule: each client is routed to the
    clearer whose resulting client-book-to-capital ratio would be lowest, so large
    clients clear through the high-capital bank-CMs and small accounts through the
    non-bank CMs. ZI clearing is a thesis extension — the ODD has ZI as direct,
    balance-less participants. Returns the CentralCounterparty (holds every CM via
    `members`)."""
    rng = np.random.default_rng(seed + 7)
    start_id = len(traders)

    if direct:
        # Direct-clearing counterfactual (H1 tiered-vs-direct; Duffie-Zhu 2011,
        # Galbiati-Soramaki 2013). No client tier: the NBCMs do not exist, the
        # BCMs are own-account members only, and every end user clears directly
        # at the CCP — registered as a participant for margin / default-fund /
        # waterfall purposes while keeping identical trading mechanics to the
        # tiered run (same house-margin position cap, same 8%-of-notional freeze,
        # default at cash 0), so the comparison isolates the loss-absorption
        # structure, not client behaviour. A defaulted direct client's deficit
        # goes straight to the CCP waterfall (no CM buffer) and its book to the
        # CCP fire-sale.
        from model.clearing import BalanceSheet
        ccp = CentralCounterparty(
            ccp_id=start_id, cash=float(CCP_CASH),
            own_df=CCP_CALIBRATION["ex_df_ratio"] * CCP_CASH)
        bcms = [t for t in traders if isinstance(t, BankingClearingMember)]
        for cm in bcms:
            cm.balance_sheet.volume_lot = params.volume_lot
            ccp.register_member(cm)
        clients = [t for t in traders
                   if isinstance(t, (FundamentalTrader, MomentumTrader,
                                     ZeroIntelligenceTrader))
                   and not isinstance(t, BankingClearingMember)]
        for client in clients:
            client.balance_sheet = BalanceSheet(
                owner_id=client.agent_id, is_banking=False, is_client=True)
            client.balance_sheet.volume_lot = params.volume_lot
            ccp.register_member(client)
            client.clearing_member_id = ccp.ccp_id   # house cap + VM mark wiring
        return ccp

    ccp = CentralCounterparty(
        ccp_id=start_id + params.n_nbcm, cash=float(CCP_CASH),
        own_df=CCP_CALIBRATION["ex_df_ratio"] * CCP_CASH)

    # NBCM adjusted net capital ~ uniform over the CFTC FCM non-bank range ($50M-$1B).
    nbcms = [NonBankingClearingMember(
                 agent_id=start_id + k,
                 cash=float(rng.uniform(*NBCM_CASH_RANGE)))
             for k in range(params.n_nbcm)]

    bcms = [t for t in traders if isinstance(t, BankingClearingMember)]
    for cm in bcms + nbcms:
        cm.balance_sheet.volume_lot = params.volume_lot   # per-regime contract lot
        ccp.register_member(cm)

    clients = [t for t in traders
               if isinstance(t, (FundamentalTrader, MomentumTrader,
                                 ZeroIntelligenceTrader))
               and not isinstance(t, BankingClearingMember)]
    # Client-clearing CMs: only n_bcm_with_clients BCMs carry clients (the rest own-account
    # only) plus all NBCMs. Clients are distributed across them by the leverage-balanced rule
    # below (capacity-proportional), so the high-capital bank-CMs hold the large books and the
    # non-bank CMs the small ones — the realistic ordering. (Sort is cosmetic; the assignment
    # rule is order-independent.)
    client_cms = sorted(bcms[:params.n_bcm_with_clients] + nbcms,
                        key=lambda cm: cm.cash, reverse=True)   # high-capital CMs first
    # Capacity-proportional (leverage-balanced) client assignment. Each client is routed to
    # the clearer whose RESULTING client-book-to-capital ratio would be lowest (largest
    # clients first), so every client-clearing CM ends near the same gross leverage. This
    # routes the large asset-manager (FT) books to the high-capital bank-CMs and the smaller
    # accounts to the non-bank CMs — the realistic ordering (big asset managers clear through
    # bank FCMs) — keeping calm clean (all CMs above the 8% floor) while the stressed crash
    # pushes every clearer toward the floor together (bank-CM deleverage vs non-bank-CM
    # stop-out, the ODD asymmetry). Client cash is the book proxy (positions are margin-capped
    # at ~5x cash). Replaces the earlier count-ramp that concentrated the biggest books on the
    # smallest NBCMs and drove their leverage ratio to ~0 once client balance sheets were
    # realistic.
    load = {cm.agent_id: 0.0 for cm in client_cms}      # Sigma assigned client cash per CM
    for client in sorted(clients, key=lambda c: c.cash, reverse=True):
        cm = min(client_cms, key=lambda m: (load[m.agent_id] + client.cash) / m.cash)
        client.clearing_member_id = cm.agent_id
        cm.client_ids.append(client.agent_id)
        cm.balance_sheet.client_positions[client.agent_id] = 0
        load[cm.agent_id] += client.cash

    return ccp


def main():
    stressed = False
    regime = "stressed" if stressed else "calm"

    # 100 LOB agents: 30 FT + 10 BCM + 20 MT + 40 ZI (no MM; geometric data-fit
    # placement supplies near-mid liquidity). 5 NBCM + 1 CCP off-LOB; 5 of 10 BCMs
    # carry client books. 90 cleared clients across 10 client-carrying CMs. FT-equivalent
    # = 30 FT + 10 BCM = 40, preserving the FT-equiv:MT:ZI = 40:20:40 price-formation
    # mix so the calibrated θ carries over. Behavioural θ from CALIBRATED {ft_sigma_c,
    # zi_alpha, zi_delta (+ p_zi stressed)}; qty_max / σ_v / VOLUME_LOT auto-populate
    # per regime.
    params = ModelParams(
        n_fundamental=30, n_momentum=20, n_zi=40,
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
    # Clearing-tier observables — one row per CM per margin cycle.
    if sim.clearing_history:
        pd.DataFrame(sim.clearing_history).to_csv(
            f"output/clearing_{regime}.csv", index=False)


if __name__ == "__main__":
    main()
