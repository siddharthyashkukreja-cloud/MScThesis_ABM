"""
Entry point — market layer + central-clearing tier.

Market layer (50 LOB agents): 10 FT + 10 BCM (FT-cast, own-account) +
10 MT (single cohort) + 20 ZI. No market maker (removed) — geometric
data-fit ZI placement supplies the near-mid liquidity.
Clearing tier: 5 NBCM + 1 CCP off-LOB; 5 of 10 BCMs carry client books,
all 5 NBCMs do; plain FT + MT + ZI clear through them round-robin (ZI
clearing is a thesis extension to the ODD).

Margin cycle runs every 60 min: USD variation margin (× VOLUME_LOT ×
CONTRACT_USD) settles the cleared book's M2M into CM cash, IM/MM and the
capital ratio recomputed, margin call flagged; a capital-ratio breach
triggers an Almgren-Chriss deleverage, cash exhaustion a default + the
5-level waterfall + a CCP fire-sale. The cover-2 default fund is
recomputed each RTH day.

1-min cadence (ODD-native): 390 steps per 6.5h RTH day. V_t exogenous,
read from data/fv_{regime}.csv (Kalman efficient price; data/v_kalman.py).
The behavioural θ is taken per regime from globals.CALIBRATED (grid optimum).
"""

import os
import numpy as np
import pandas as pd

from model.globals import (
    ModelParams, V0, CALIBRATED,
    CCP_CASH, BCM_CASH_RANGE, NBCM_CASH_RANGE, CCP_CALIBRATION,
    FT_CLIENT_CASH, MT_CLIENT_CASH, ZI_CLIENT_CASH, CLIENT_BOOK_RANGE,
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

    for _ in range(params.n_momentum_long):
        traders.append(MomentumTrader(
            agent_id=next_id(), cash=float(rng.uniform(*MT_CLIENT_CASH)),
            lambda_decay=params.mt_lambda_long))

    for _ in range(params.n_zi):
        traders.append(ZeroIntelligenceTrader(
            agent_id=next_id(), cash=float(rng.uniform(*ZI_CLIENT_CASH))))

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
    only. The plain FT, MT and ZI clear through the client-carrying CMs with
    SKEWED, size-ranked book sizes (D53c — heterogeneous counts in
    CLIENT_BOOK_RANGE, larger CMs by cash holding more clients). NB: ZI clearing
    is a thesis extension — the ODD has ZI as direct, balance-less participants.
    Returns the CentralCounterparty (holds every CM via `members`)."""
    rng = np.random.default_rng(seed + 7)
    start_id = len(traders)

    ccp = CentralCounterparty(
        ccp_id=start_id + params.n_nbcm, cash=float(CCP_CASH),
        own_df=CCP_CALIBRATION["ex_df_ratio"] * CCP_CASH)

    # NBCM adjusted net capital ~ UNIFORM over the CFTC FCM non-bank range ($50M-$1B).
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
    # Skewed, size-ranked client-book concentration (D53c). Only n_bcm_with_clients
    # BCMs carry clients (the rest own-account only). Book sizes are HETEROGENEOUS in
    # CLIENT_BOOK_RANGE; the NBCMs hold the LARGER books (no own position → all their
    # capacity is client-clearing), and WITHIN each type more clients = higher cash.
    # Clients are shuffled so each CM's FT/MT/ZI mix also varies. Concentrating the big
    # books on the NBCMs (pure intermediaries) is the client→NBCM contagion lever.
    client_cms = sorted(bcms[:params.n_bcm_with_clients] + nbcms,
                        key=lambda cm: (isinstance(cm, NonBankingClearingMember), cm.cash),
                        reverse=True)   # NBCMs first, then within-type by cash
    K, N = len(client_cms), len(clients)
    lo, hi = CLIENT_BOOK_RANGE
    if K and lo * K <= N <= hi * K:
        ramp = np.linspace(hi, lo, K) * (N / (0.5 * (hi + lo) * K))  # descending, mean N/K
        counts = np.clip(np.round(ramp), lo, hi).astype(int)
        j = 0
        while int(counts.sum()) != N:                               # fix rounding within [lo,hi]
            step = 1 if counts.sum() < N else -1
            nv = counts[j % K] + step
            if lo <= nv <= hi:
                counts[j % K] = nv
            j += 1
        counts = sorted(counts.tolist(), reverse=True)              # large CM -> more clients
    else:
        base = N // K
        counts = [base + (1 if i < N % K else 0) for i in range(K)]
    order = list(range(N)); rng.shuffle(order)                      # varied type-mix per CM
    pos = 0
    for cm, n in zip(client_cms, counts):
        for _ in range(int(n)):
            client = clients[order[pos]]; pos += 1
            client.clearing_member_id = cm.agent_id
            cm.client_ids.append(client.agent_id)
            cm.balance_sheet.client_positions[client.agent_id] = 0

    return ccp


def main():
    stressed = False
    regime = "stressed" if stressed else "calm"

    # 100 LOB agents (D53 — CLIENTS scaled 2×, CM count unchanged at 15): 30 FT +
    # 10 BCM + 20 MT + 40 ZI (no MM — removed; geometric data-fit placement supplies
    # near-mid liquidity). 5 NBCM + 1 CCP off-LOB; 5 of 10 BCMs carry client books
    # (D29). 90 cleared clients across 10 client-carrying CMs = 9 per CM. FT clients
    # are 30 (not 20) so FT-equivalent = 30 + 10 BCM = 40, preserving the pre-doubling
    # FT-equiv:MT:ZI = 40:20:40 (40/20/40) price-formation mix at 2× scale — the
    # calibrated θ carries over. Behavioural θ from CALIBRATED {ft_sigma_c, zi_alpha,
    # zi_delta (+ p_zi stressed)}; qty_max / σ_v / VOLUME_LOT auto-populate per regime.
    params = ModelParams(
        n_fundamental=30, n_momentum=20, n_momentum_long=0,
        n_mm=0, n_zi=40, n_vt=0, n_ct=0,
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
