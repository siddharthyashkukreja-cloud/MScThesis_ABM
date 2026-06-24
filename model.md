# The model — design, calibration, and results

In-depth reference for the agent-based model: the agents and equations, the fundamental
process, the clearing tier, every parameter and how it is set, the calibration method and
its locked result, and the contagion experiments. `README.md` is the short orientation
guide; `AGENT.md` is the working briefing and change history. This file is the technical
spine.

## 1. Overview

The model has two tiers. The first is a **market**: a limit order book where three kinds
of trader — fundamental (value-driven), momentum (trend-following) and zero-intelligence
(noise) — trade E-mini S&P 500 (ES) futures around an exogenous "fair value" path taken
from real ES data. This layer is calibrated so the simulated price reproduces the
empirical stylised facts of returns (fat tails, clustered volatility, no predictable
drift). It is the realistic stage on which the second tier acts.

The second tier is the **clearing system**. Trades are guaranteed by a central
counterparty (CCP); most participants ("clients") do not face the CCP directly but clear
through a member — a bank or broker (a banking or non-banking clearing member, BCM/NBCM).
Everyone posts margin sized to their risk, and the CCP holds a mutualised default fund. As
prices move, margin is exchanged; a participant that runs out of cash defaults, and its
losses flow through a fixed waterfall — its own collateral first, then the fund, then, in
the extreme, the surviving members.

The thesis question lives at that junction: **does clearing clients through members, rather
than directly, change how a stress propagates?** The model lets a client default flow up to
its clearing member and out into the rest of the system, and measures how severe a shock
must be before the mutualised layer is touched.

The asset is the ES front-month future. Prices move on a 0.25 tick; the simulation steps
every minute. Each trader submits to a shared book, and a call auction clears any crossed
quotes at the end of every step. The mid is `(best_bid + best_ask)/2`, holding at the last
trade price when a side is empty.

## 2. Market layer

### 2.1 Agents

**Order lifetime** is governed by the agents — there is no blanket time-to-live.
Fundamental, banking-member and momentum traders use *replace-on-new* management: each
holds at most one standing limit order and, the next time it acts, cancels that order and
places a fresh one, so its resting order always reflects its latest view (they act every
step). Zero-intelligence traders instead cancel each resting order with a fixed
probability `zi_delta` per step — the Cont-Stoikov-Talreja (2008) / Farmer et al. (2004)
cancellation rate, which is the sole control on ZI order lifetime. This deviates from the
Simudyne ODD's explicit order ceiling, which was redundant with `zi_delta` and masked it
whenever the cancellation rate was low.

**Fundamental trader (FT).** A value strategy (the Chiarella-Iori-Perelló family). Each FT
holds a fixed idiosyncratic draw `z ~ N(0,1)` and forms a reservation price
`R = V_t · (1 + z · ft_sigma_c · σ_t)` around the exogenous fundamental: the belief offset
`z · ft_sigma_c · σ_t` is a *fractional* dispersion (so the cloud is scale-invariant around
`V_t`, with no `v0` level constant in the price-formation law), and `σ_t` is the EWMA realised
volatility of `V_t` (EWMA of `r²`, 30-min half-life, reset each session) — so the belief cloud
widens in volatile regimes and tightens in calm ones. Every step
it places one limit order at the reservation on the implied side (`sign(R − mid)`),
replacing any standing order. The persistent `z` is fixed at initialisation, not redrawn:
this is a heterogeneous-beliefs specification (Chiarella-Iori-Perelló), so disagreement
persists and high-`z` FTs accumulate long positions while low-`z` FTs accumulate shorts,
building **concentrated inventories** over time. That concentration is the load-bearing
input to the clearing layer — margin calls, the default fund, and the waterfall only have
something to act on when member balance sheets are skewed by positions.

`ft_sigma_c` (the belief-width scale) is calibrated, and the ablation showed it is the
single dominant lever on the return tail: too wide and FTs overshoot the book, producing
kurtosis in the thousands; calibrated low it reproduces the empirical tail.

**Momentum trader (MT).** A trend-following chartist (Majewski, Ciliberti & Bouchaud 2018;
Gao et al. 2022). Each MT tracks an exponentially-weighted moving average of mid
log-returns, `M_t = (1−λ)·M_{t−1} + λ·r_t`, and on the side of the trend places a passive
limit order at a depth drawn from the shared geometric distribution. The MT is limit-only;
trend amplification comes from inside-the-spread MT limits being lifted by trend-following
flow. Activation is stochastic: each step an MT trades with probability
`tanh(|M_t| / (mt_gamma·σ_v))`, so the share of MTs acting scales with trend strength (a
near-flat market barely moves them; a strong trend pulls most in). The EWMA decay `mt_lambda`
(trend horizon) and the activation scale `mt_gamma` are both calibrated per regime.

**Zero-intelligence trader (ZI).** A Cont-Stoikov-Talreja (2008) / Farmer-Patelli-Zovko
(2004) noise trader providing the background liquidity floor. Each step it independently
may cancel a resting order (probability `zi_delta`), submit a limit order at a random side
and geometric depth (probability `zi_alpha`), or submit a small market order (probability
`zi_mu`), with no signal dependence. All three ZI rates are calibrated.

**Order placement (ZI and MT)** is geometric: a limit rests `k ~ Geometric(p_zi)` ticks
from the mid, with most mass at the touch (Cont-Stoikov-Talreja 2008; Farmer et al. 2004).
`p_zi` is calibrated per regime alongside the other behavioural parameters, so no separate
L2/MBP-10 order-book feed is required (the legacy MBP-10 MLE in `data/p_zi.py` is kept only as
an offline cross-check).

**Removed agents** (do not re-add without a strong citation). A *market maker* (Gao et al.
2022) was trialled and removed: the ablation confirmed it damps volatility without
improving clustering, and near-mid liquidity now comes from the data-fit geometric ZI book.
A *volatility trader* (Gao et al. 2023) helps the stressed tail but harms calm, so it is
held as a possible regime-specific option rather than a standing agent.

### 2.2 Clearing tier

Layered on the LOB population (Simudyne CCP ODD, extended here with the client-clearing
layer):

- **Banking clearing member (BCM, x10)** — a fundamental-trader subclass. It trades its own
  account through the inherited FT logic, so for the market calibration 30 FT + 10 BCM is a
  40-FT-equivalent population. It additionally carries a balance sheet, a CCP link, and a
  client book. Its **client-clearing solvency** is the FCM rule it shares with the
  NBCMs — CFTC Reg 1.17, `cash/IM ≥ 8%` on the client book (§3.3). As a bank it is **additionally**
  held to the Basel III / US-eSLR **leverage ratio** `cash/exposure(own+client) ≥ LR_FLOOR_BCM = 4.25%`
  (cash net of escrowed IM) — the constraint that actually binds dealer client-clearing. Its own
  (house) book is further capped by a **static gross-leverage limit** `|own notional| ≤ 2·cash`
  (`POSITION_LIMIT_X = 2`, no volatility input), keeping the proprietary book modest. On breaching the
  Basel LR the BCM **deleverages its own book** (Almgren-Chriss) toward a 6% buffer
  (`DELEVERAGE_TARGET_BCM`) and freezes new risk — the bank leverage cycle / forced-deleveraging
  channel (Thurner 2012; Aymanns-Farmer 2015); only the own book can be shed intraday, so a
  client-book-dominated member sheds all own risk and freezes on the residual. (These floors live only
  on the runtime BCM path, so the market calibration with plain FTs is unaffected; see §3.3.)
- **Non-banking clearing member (NBCM, x5)** — a pure clearing intermediary: no own position, no LOB
  orders; a balance sheet and a client book only. Not a bank, so it is held to the same FCM rule
  (CFTC Reg 1.17, `cash/IM ≥ 8%`) but **not** the Basel LR; on breach it stops out (the
  BCM-deleverage / NBCM-stop asymmetry). Capital `U[$0.5, $3]B` — the substantial non-bank clearers
  (Marex / ABN AMRO Clearing / Clear Street) that carry real client volume.
- **Central counterparty (CCP, x1)** — the clearing role of the ODD matching engine (order
  matching itself stays in the LOB). Holds the member registry, default-fund accounts, and
  the default list; wires the bidirectional member-CCP star topology.
- **Clients (90)** — 30 FT + 20 MT + 40 ZI, clearing through the client-carrying members
  (five of the ten BCMs plus all five NBCMs) by a **leverage-balanced (capacity-proportional)**
  assignment: each client is routed to the clearer whose resulting client-book-to-capital ratio
  would be lowest, so large asset-manager clients clear through high-capital bank-CMs and small
  accounts through non-bank CMs. Each client carries
  its own balance sheet and, each cycle, settles its variation margin in cash and posts its
  own procyclical initial margin (routed through its member) into the CCP escrow. A client can
  only open what its cash can margin against its house margin (a fixed broker leverage cap —
  15% house margin, ≈6.67x, uniform across clients); a client that cannot fund either call from
  its own cash **defaults on liquidity**. On default its member assumes the position and
  liquidates it (§3.3); the client's posted margin is first-loss and the member bears any
  residual.

### 2.3 Fundamental value process

`V_t` is exogenous — a path generated offline and read in as a time series. There are two
interchangeable generators; the model runs on the **empirical mid** fundamental, with the
SV-MJD generator kept as a robustness alternative.

**Empirical efficient mid (primary).** The fundamental is the observed ES 1-minute mid
itself, read per RTH day (so the overnight gap is never inside any single day's series) and
concatenated across days at the real levels, so the **overnight gaps are retained** — the
COVID episode then carries its true ~-33% drawdown rather than the ~-7% with gaps removed,
and overnight gap risk is a primary CCP default driver. A local-level Kalman MLE (state
`log V_{t+1} = log V_t + μ + σ_v·Z`, observation `log p_t = log V_t + ε_t`, fit per day) is
retained only as a diagnostic: on the 1-minute mid it finds almost no i.i.d. microstructure
noise to remove (σ_ε ≈ 1e-5, steady-state gain ≈ 1.0; the bid-ask bounce lives in trade
prices, not the mid), so the RTS smoother is near-identity (it moves the level by < 0.04
points) and is **not applied** — `V_t` is taken as the mid directly. This is in the spirit
of the data-derived fundamental of the Chiarella-Heston / XGB-Chiarella line (Gao et al.
2022) and of Majewski et al. (2018); the honest framing is that the Kalman step verifies
1-minute mid noise is negligible (justifying the mid as the efficient price), not that it
delivers material smoothing.

**Stochastic-volatility Merton jump-diffusion (alternative).** A geometric Brownian motion
whose instantaneous volatility mean-reverts as a Vasicek/Ornstein-Uhlenbeck process,
overlaid with Poisson jumps and empirical overnight gaps at every session boundary:

```
σ_{t+1}              = θ + e^{−α}(σ_t − θ) + s·Z_σ                 (Vasicek-OU)
log V_{t+1} − log V_t = μ + σ_t·Z + J,   J ~ N(0, δ²) w.p. λ/390    (intraday)
log V_{t+1} − log V_t = G ~ empirical overnight-return pool         (session boundaries)
```

Every parameter is measured from the ES data via realized measures: the diffusion σ_d from
daily bipower variation (jump-robust; Barndorff-Nielsen & Shephard 2004); the jump
intensity from Mancini (2009) threshold counts (λ ≈ 1.9/day calm, 1.0/day stressed); the
jump size from the RV−BV jump variance (share ~7% / ~4% of intraday variance, on the
Huang-Tauchen 2005 S&P benchmark); the OU parameters from a jump-robust bipower volatility
series; the drift from the mean intraday log-return; and the overnight gaps as nonparametric
bootstrap draws from the regime's observed overnight returns (Lou-Polk-Skouras 2019). The
formal OU-on-volatility model is Stein & Stein (1991) / Heston (1993); the jumps are Merton
(1976). The generated path carries only the measured jump/SV share of the tails — the rest
is the agent layer's to produce — so path kurtosis sits below empirical by construction.

The session boundary is read from the data: real ES sessions are ~405 bars and vary day to
day, not a fixed 390-bar grid. The simulator opens each session at the gapped `V_t` by
repricing the book — a clean between-session jump, with each trader's price memory
re-anchored so the gap is not read as intraday trend — and the boundary return is excluded
from the calibration moments. So the gaps stress the cleared book without contaminating the
intraday stylised-fact fit. (A flagged sensitivity, `REANCHOR_ON_GAP = False`, suppresses
the re-anchoring so the gap produces a disorderly open instead; see §6.4.)

### 2.4 Market formulas

Time is discrete; each step is one minute. Arrival rates are per-step Bernoulli
probabilities (ODD-native — no `dt` rescaling).

```
Fundamental trader (fixed z ~ N(0,1)):
  σ_t             = EWMA realised vol of V_t (EWMA of r², 30-min half-life, per session)
  reservation R   = V_t · (1 + z · ft_sigma_c · σ_t)   (fractional belief offset; scale-invariant)
  side            = sign(R − mid)
  every step: cancel the standing limit, place one fresh limit AT R, qty ~ U[qty_min, qty_max]

Momentum trader:
  M_t  = (1 − mt_lambda)·M_{t−1} + mt_lambda·(log mid_{t−1} − log mid_{t−2})
  |M_t| < ε → no trade;  side = sign(M_t)
  every step: replace the standing limit with one at mid − side·k·tick, k ~ Geometric(p_zi)

Zero-intelligence trader:
  each resting limit cancelled w.p. zi_delta
  submit a limit  w.p. zi_alpha at random side, depth k ~ Geometric(p_zi)
  submit a market w.p. zi_mu    at random side
  size ~ U[qty_min, qty_max]
```

## 3. Clearing mechanics

### 3.1 Margin cycle

Every 60 minutes, variation margin settles each cleared book's mark-to-market move into
member cash in USD (`x VOLUME_LOT x CONTRACT_USD`), against the average filled price
(`P/L = position·Δmark + Σq·(mark − fill price)`), so execution slippage in fire-sale book
walks is realised rather than dropped. Client positions are novated live into the clearing
member's book.

**Initial margin is a procyclical VaR scan, filtered-historical-simulation style.** For a single
linear future SPAN scanning risk equals VaR, so IM is a fraction of notional = a quantile of the
daily return over the margin period of risk (MPOR), floored by an anti-procyclicality floor:

```
σ_daily      = RiskMetrics EWMA (λ=0.94) of DAILY close-to-close ES returns, warmed on real
               pre-window daily history (no cold start; overnight gap covered natively)
im_fraction  = max(IM_FLOOR, q₉₉ · σ_daily · √MPOR)   (q₉₉=3.0, MPOR=1, floor=4%, no cap — IM_CAP=None, D77)
```

`q₉₉ = 3.0` is the EMPIRICAL 99% quantile of EWMA-standardised ES returns (filtered historical
simulation — the standard CCP IM method; LCH/ICE/Eurex), not the Gaussian 2.326, which understates the
fat-tailed ES 99% move by ~1/3 (measured 3.0 stressed / 3.2 calm). The driving σ is set by the margin
regime (`IM_MODE`):

- **reactive** (default) — a RiskMetrics EWMA (λ=0.94). The driving σ is a DAILY close-to-close
  EWMA warmed on real pre-window daily history (`IM_DAILY = True`; no cold start, native overnight-gap
  coverage); the older intraday-returns EWMA (~11-RTH-day half-life) + separate gap-EWMA is retained
  only as a fallback. IM ratchets up through a crash as CME's ES margin did in March 2020 (~4% calm
  floor → ~11% crash peak, ~2.8× (uncapped at the 1-day MPOR); CME ES IM went ~$6.3k→$12k, ~1.9–2.6× of notional).
- **static** — the regime constant `σ_v` (two-point procyclicality across regimes only).
- **flat** — a fixed fraction `IM_FLAT_FRAC` with no volatility response (the
  through-the-cycle comparator for H2).

A margin call flags when variation margin exceeds `(1 − mm%)` of initial margin. Members use
futures-margin accounting (a fill posts only the position; cash settles via the cycle).

**Margin escrow and the funding cycle (`IM_ESCROW = True`).** Initial margin moves as real
cash to the CCP's segregated `im_account` each cycle: a member posts on its own book and a
client posts its own procyclical IM, routed through its member. Posted IM is returned as the
position shrinks and is **seized first** if the participant defaults (the defaulter's own
first-loss resource, ahead of the default fund); any excess collateral over the realised loss
returns to the defaulter's estate, not the mutualised pool. A participant that cannot fund a
call from its cash **defaults on that call**, so default is a *liquidity* event — the
realistic failure mode of a clearing participant (Nasdaq 2018, LME 2022) — and is the channel
through which margin procyclicality transmits: a crash raises `im_fraction`, drains free cash,
and forces the most leveraged participants out, whose losses then propagate up the tier

Because IM is escrowed, member solvency is governed by the **net-position leverage ratio**
`κ = cash / exposure ≥ 8%` (cash already net of posted IM, exposure the own + client cleared
notional). This is the Basel leverage ratio — the operative constraint for derivative dealers
(Haynes, McPhail & Zhu 2019; Acosta-Smith et al. 2018) — and the channel through which
leverage-constrained intermediaries transmit stress (Thurner et al. 2012; Aymanns & Farmer
2015). The margin regime now transmits through the **numerator** (a funding-liquidity squeeze:
a higher reactive IM locks more cash, lowering κ), and leverage is procyclically self-bounded
at `cash / (8% + im_fraction)` (≈7× in calm, ≈2.6× under a stressed margin), which makes a
separate volatility-dependent VaR cap redundant. Under `IM_ESCROW=True` the capital ratio is
simply `κ = cash (net of posted initial margin) / exposure`.

### 3.2 Default fund

A cover-2 **Stress-Loss-Over-Initial-Margin** fund (Euronext Clearing module A9),
recomputed each RTH day: per member `SLOIM = max(0, stress_move·notional − initial_margin)`
under an extreme-but-plausible move; the total is the two largest members' SLOIM x (1 + 10%
buffer):

```
df_stress_move = min(DF_CAP, max(DF_FLOOR, q_df · σ_daily · √MPOR))   (q_df=3.9 FHS ~99.9%, MPOR=1, floor 8%, cap 35%)
total_df       = (Σ over the two largest of SLOIM) · (1 + buffer)
```

The exchange pre-funds a skin-in-the-game share (first loss); members pay their
contributions in cash into a CCP-held pool, pro-rata by exposure — so a post-draw daily
recompute is a replenishment cash call, the margin-liquidity channel the ESRB documented
for March 2020.

Note the coupling that matters for the H2 results: SLOIM subtracts the initial margin, so a
**higher IM mechanically shrinks the fund**. This is internally coherent (same-basis net)
but means the margin regime and the fund size are linked — see §6.2.

### 3.3 Waterfall and default management

Solvency floors separate the **client-clearing** role (shared by both tiers) from the
**bank-specific** constraints, so the two member types are measured comparably
(`DIFFERENTIATED_FLOORS`, `UNIFIED_CLIENT_FLOOR`, `BASEL_LR_BCM`).

- **Client-clearing solvency — both tiers, CFTC Reg 1.17.** A clearing member's adjusted net capital
  must be ≥ 8% of its client risk margin: `cash/IM ≥ REG117_FLOOR_NBCM = 8%` on the client book. This
  is the FCM rule that governs *every* clearing member, so a BCM and an NBCM are held to the *same*
  client-clearing yardstick — the comparison between them is then about capital and book size, not a
  different rule. A member that breaches freezes new risk; one that cannot fund a margin call from
  cash defaults (a liquidity event).
- **Bank leverage cycle — BCM only, Basel III / US-eSLR.** A banking member is additionally a bank,
  held to the leverage ratio `cash/exposure(own+client) ≥ LR_FLOOR_BCM = 4.25%` (the G-SIB 3%-base +
  surcharge buffer / post-2026 eSLR range — Haynes-McPhail-Zhu 2019; Acosta-Smith 2018) — the
  constraint that actually binds dealer client-clearing. On breach the BCM **deleverages its own
  book** (Almgren-Chriss) toward a 6% buffer (`DELEVERAGE_TARGET_BCM`) and freezes — the
  forced-deleveraging channel (Thurner 2012; Aymanns-Farmer 2015). Only the own book can be shed
  intraday, so a client-book-dominated member sheds all its own risk and freezes on the residual. An
  NBCM (non-bank, no own book) is not Basel-bound and simply stops out — the ODD's BCM-deleverage /
  NBCM-stop asymmetry.
- **Client freeze.** A client (a leveraged end-user, not a bank) is frozen by its FCM near distress,
  `cash/exposure ≤ CLIENT_FREEZE_FLOOR = 4%` (the house margin already caps opening leverage at
  1/`im_percent`).

This shape was reached in two steps (see `FLOOR_BALANCE_RESULTS.md`). The original *single* uniform 8%
`cash/exposure` floor both mis-labelled the Basel ratio (real minimum 3%) and — sitting just above the
endogenous stress `κ_min` (~4–6%) — was the load-bearing *trigger* of the COVID contagion; lowering it
to the accurate ~4% made COVID benign. But with tiny `$50M–1B` NBCMs the small non-banks were then the
*only* tier that ever defaulted — a capital-size artifact, not a risk-rule one (verified: unifying the
floor alone did not change it). The fix unifies the client-clearing measure (Reg 1.17, both tiers),
keeps the Basel LR as the bank deleverage trigger, and sizes NBCMs at the realistic `$0.5–3B` (§4).
Result: calm clean, COVID benign (a handful of localised client defaults, no member defaults, no
mutualisation — matching the historical record), the bank leverage cycle fires under deeper stress,
and member defaults are graded across both tiers rather than confined to the non-banks.

A defaulting member triggers the five-level waterfall on its realised loss (the unpaid VM
shortfall beyond its own capital, after its posted IM is seized, plus the close-out loss on
its book):

```
L1 defaulted-member DF → L2 exchange skin-in-the-game → L3 pooled member DF
   → L4 surviving-member cash pro-rata → L5 CCP cash
```

The member is then removed: its surviving clients are **ported** to other client-carrying
members with spare risk capacity above the floor (EMIR Art. 48(5)-(6); capacity-checked
against the receiver's escrow headroom `cash/floor − exposure`, not guaranteed — the modal
client has a single clearing agent, OFR 2026 — and unported clients are closed out and
frozen).

**Close-out: who liquidates, and how.** The two tiers resolve a default differently.

- A defaulted **tiered client** is closed out by its **member** in the open market
  (`CLIENT_CLOSEOUT = "firesale"`): the member assumes the position onto its book and
  liquidates it via an Almgren-Chriss schedule (`x_j/X = sinh(κ(H−j))/sinh(κH)`), so the
  book-walk realises the close-out loss as endogenous price impact — the client-level
  price-impact channel. A single client's position is small enough that open-market execution
  is realistic. The client's posted IM (under escrow, its physically-posted procyclical IM —
  seized via `_seize_im`) is first-loss against the VM shortfall, and the member bears the
  residual.
- A defaulted **member** — and, in the direct-clearing counterfactual, a defaulted **direct
  client** faced by the CCP — is resolved by the **CCP** at a recovery rate
  (`CLOSEOUT_MODE = "transfer"`, `CLOSEOUT_RECOVERY = 0.80`). After EMIR-48 porting, the residual
  net book is **transferred to the surviving member with the largest opposing position**
  (`_transfer_book`, booked as a fill at the prevailing mid — the ODD PositionAuction target, the
  member best able to absorb it), and the (1−recovery) haircut is the close-out loss mutualised
  through the waterfall. **Reassigning the book rather than flattening it keeps system inventory and
  cash conserved (the C1 fix);** the earlier flatten-without-transfer broke variation-margin
  zero-sum (a measured −\$14.8B / −7,427-lot leak in the direct arm). If no surviving counterparty
  exists the CCP warehouses the book (`_ccp_warehouse`). A whole member book is too large to push
  through the open market without unrealistic impact, which is why the member tier is auction-style.

The 0.80 recovery is a conservative stressed assumption: real default-management auctions
recover close to par on a hedged book (LCH used only ~35% of Lehman's initial margin and no
default fund in 2008; BIS 2018; IOSCO 2017), while concentrated/unhedged books recover less
(Nasdaq 2018), so 0.80 sits below the Lehman outcome and above the ODD's credit-style 0.60; it
is sensitivity-testable.

The **open-market Almgren-Chriss fire-sale of a whole member book** — the CCP liquidates the
defaulted member's book into the LOB, the price impact resuming the waterfall via a CCP
disposal account — is retained as a deferred **additional hypothesis** (`CLOSEOUT_MODE =
"firesale"`), alongside the reverse-stress amplifier. It is a deliberate deviation from the
ODD's position auction: real CCPs auction precisely to avoid that fire-sale impact (Vuillemey
2020), so it models the counterfactual that auctions are designed to prevent.

Total system cash **and inventory** are conserved to float precision through L4 cascades in both
tiered and direct modes (verified after the C1 close-out fix: Σ positions ≡ 0, and total cash falls
only by genuine close-out haircuts; every dollar destroyed is a written-off unpaid debt, and the
default-fund pool equals the sum of contributions to the cent). The lone exception is the
extreme-tail `_ccp_warehouse` fallback (no surviving counterparty), which adds the warehoused book's
full mark-down on top of the haircut and so overstates the apocalyptic-tail loss — report that tail
qualitatively (CCP backstop reached), not by its dollar figure.

## 4. Parameters

Grouped by how each is set.

**Market and order book (structural):**

| Parameter | Meaning | Value |
|---|---|---|
| `tick_size` | price increment | 0.25 |
| `dt_minutes` | step length | 1.0 |
| `qty_min`, `qty_max` | order size range | 1, 10 |
| populations | LOB counts (FT+BCM, MT, ZI) | 40-equiv, 20, 40 (100 on book) + 90 cleared clients |

**Fundamental `V_t` (data, per regime):** the empirical 1-minute ES mid itself (the Kalman MLE
is only a diagnostic confirming the mid is the efficient price — it is not applied as a
smoother); `σ_t` the per-minute local volatility (EWMA of `r²`, 30-min half-life); the SV-MJD
drift/diffusion/jump/OU parameters (robustness arm) from ES realized measures; the
overnight-gap pools from observed overnight returns.

**Behavioural (calibrated per regime):**

| Parameter | Meaning | Calm | Stressed |
|---|---|---|---|
| `ft_sigma_c` | FT belief-width scale (the tail lever) | calibrated | calibrated |
| `zi_alpha` | ZI limit-order arrival | calibrated | calibrated |
| `zi_delta` | ZI per-resting cancellation | calibrated (floor 0.02) | calibrated (floor 0.02) |
| `zi_mu` | ZI market-order arrival | calibrated | calibrated |
| `p_zi` | geometric placement depth | calibrated (floor 0.08) | calibrated (floor 0.08) |
| `mt_lambda` | MT EWMA decay (trend horizon) | calibrated | calibrated |
| `mt_gamma` | MT activation scale (`P=tanh(\|M\|/(mt_gamma·σ_v))`) | calibrated | calibrated |
| `ft_alpha` | FT activation (trades every step) | 1.0 (fixed) | 1.0 (fixed) |

`p_zi` is now calibrated in both regimes; this removes the dependence on L2/MBP-10 book data
(the only quantity that had been drawn from it), so the data inputs reduce to the 1-minute BBO,
1-minute OHLCV, and daily OHLCV feeds. `mt_lambda` and the new MT activation scale `mt_gamma`
are likewise calibrated, giving a single 7-parameter behavioural loop in both regimes.

**Clearing tier (regulatory / ODD constants — none calibrated):**

| Parameter | Meaning | Value | Source |
|---|---|---|---|
| `VOLUME_LOT` | contracts per model lot (per regime) | 18 / 32 | matched to empirical RTH front-month ES volume (~2,000 / ~3,500 per min) |
| `CONTRACT_USD` | CME ES point multiplier | 50 | CME contract spec |
| `margin_interval` | variation-margin cadence (min) | 60 | ODD |
| `df_interval` | default-fund recompute (min) | 390 | one RTH day |
| `IM_CONF_Z, IM_MPOR_DAYS, IM_FLOOR, IM_CAP` | 99% 1-day VaR rate (FHS empirical quantile; EWMA σ half-life ~11d, RiskMetrics λ≈0.94), APC floor, **no cap** (D77) | 3.0, 1, 4%, None | CME SPAN 2 (1-day ETD-futures liquidation); FHS (LCH/ICE/Eurex); EMIR Art. 41 (2-day = OTC min); ES IM ~3.9%→~10% of notional, Mar-2020 |
| `im_percent` (house margin) | client position cap (leverage) | 15% (≈6.67x) | post-2020 security-futures statutory minimum (17 CFR 242.400-406; lowered from the 2002-2020 20% floor) |
| `IM_ESCROW` | physical IM escrow (IM moved as cash to CCP; default on liquidity) | True | EMIR segregation; funding-liquidity channel |
| `POSITION_LIMIT_X`, `POSITION_LIMIT_CLIENTS_ONLY` | static BCM own-book gross-leverage cap (own notional ≤ X·cash, no vol input) | 2.0, False (all BCMs) | keeps the own book modest, so a floor breach (exposure ≳ 12.5×cash) is driven by a large client book; calm stays safe |
| `mm_percent` | maintenance-margin threshold | 95% | CME methodology |
| `DF_STRESS_Z, DF_STRESS_FLOOR, DF_STRESS_CAP` | ~99.9% VaR (FHS; kept > IM_CONF_Z so SLOIM>0), EBP floor + ceiling | 3.9, 10%, 35% | Euronext A9; CPMI-IOSCO |
| `df_buffer` | cover-2 SLOIM buffer | 10% | Euronext A9 |
| `ex_df_ratio` | exchange skin-in-the-game (share of fund) | 3% | empirical SITG/DF (CME ES ≈2.7%, Clarus); cf. EMIR Art. 45 (25% of capital) |
| `cover_number` | cover-N default fund | 2 | EMIR / Dodd-Frank |
| `DIFFERENTIATED_FLOORS`, `UNIFIED_CLIENT_FLOOR` | type-specific floors / unify the client-clearing floor across both tiers (vs legacy uniform `cap_ratio_floor=8%`) | True, True | — |
| `REG117_FLOOR_NBCM` | **client-clearing solvency floor — BOTH tiers** (`cash/IM`, FCM net capital) | 8% | CFTC Reg 1.17 (net capital ≥ 8% of risk margin) |
| `BASEL_LR_BCM`, `LR_FLOOR_BCM` | BCM-only bank leverage ratio (`cash/exposure`, own+client) — deleverage trigger | True, 4.25% | Basel III G-SIB / US-eSLR (Haynes-McPhail-Zhu 2019; Acosta-Smith 2018) |
| `CLIENT_FREEZE_FLOOR` | client near-distress freeze `cash/exposure` | 4% | FCM cut-off (house margin caps opening leverage) |
| `DELEVERAGE_TARGET_BCM` | breaching BCM deleverages own book (AC) toward this buffer; NBCM has no own book → stop-out | 6% | above `LR_FLOOR_BCM` so it does not re-breach |
| `CLIENT_MARGIN_NETTING` | CM client-margin basis | gross (US/CME) / net (EU omnibus) | CFTC gross margining; Duffie-Zhu 2011 (net) |
| `AC_HORIZON, FIRE_SALE_URGENCY` | Almgren-Chriss schedule (client close-out; deferred member fire-sale) | 30 min, κH = 2.0 | Almgren & Chriss 2000 |
| `CLOSEOUT_MODE` (member / direct client) | CCP resolution at recovery (no counterparty selected) | transfer, `CLOSEOUT_RECOVERY` 0.80 | ODD transfer; Lehman ~par (BIS 2018 / IOSCO 2017) |
| `CLIENT_CLOSEOUT` (tiered client) | member open-market Almgren-Chriss liquidation | firesale | client-level price-impact channel |

`VOLUME_LOT` and `CONTRACT_USD` apply at the reporting, notional and margin layer only —
never inside the matching engine — so log-returns, ACFs, and the tail index are invariant
under this multiplicative relabel.

**Loss-absorbing cash** (regulatory-scale; inert for price formation — it gates only the
clearing-layer default/deleverage): CCP $1.5B (CCP OWN capital — the SITG source + the
Level-5 backstop; a major CCP's own equity, distinct from total prefunded DF ~$9-10B = SITG +
member fund, which is member-funded, not CCP cash; at $1.5B the L5 backstop is reachable under
extreme stress yet inert at actual severity); BCM
`U[$5B, $10B]` (bank-FCM scale); NBCM `U[$0.5B, $3B]` (the substantial non-bank clearers —
Marex / ABN AMRO Clearing / Clear Street — that carry real client volume). Client cash by
type: FT `U[$0.5B, $3B]` (asset managers), MT `U[$0.2B, $1B]` (CTAs), ZI `U[$0.2B, $0.5B]`
(noise accounts), tuned so the 8% floor binds for a subset under stress but not in calm.

The calibrated values per regime live in `model/globals.py` (`CALIBRATED`) and are read by
`model/run_simulation.py`, the `scripts/` drivers, and the notebooks.

## 5. Calibration

The fundamental and impact inputs are prepared directly and offline: the empirical-mid
fundamental from the ES 1-minute mid (`data/v_kalman.py`, the Kalman MLE used only as the
efficient-price diagnostic); and an Almgren-Chriss market-impact regression on the 1-minute
data (`data/impact.py`, `|r|`~volume, kept for the liquidation/optimal-execution channel).
The geometric placement depth `p_zi` is no longer fixed offline — it is calibrated with the
behavioural parameters (the legacy MBP-10 MLE in `data/p_zi.py` is kept only as a cross-check).
These inputs are fixed per regime before any agent calibration.

The behavioural parameters are calibrated by matching simulated 1-minute return moments to
the empirical ES moments, regime by regime, as two independent problems. The target battery
is the Cont (2001) stylised-fact set; the loss is a grouped, standardised-moment distance
with five components:

| Component | What it matches | Source |
|---|---|---|
| KS | whole return distribution (2-sample Kolmogorov-Smirnov) | Gao et al. 2022 |
| V | return standard deviation | Gao et al. 2022 |
| ACF1 | autocorrelation of returns, lags {1, 5, 10, 20} | Cont 2001 |
| ACF2 | autocorrelation of |returns| (volatility clustering) | Cont 2001 |
| Hill | Hill (1975) tail index (banded) | Gao et al. 2022 |

Each moment's distance is divided by its empirical block-bootstrap sampling standard
deviation (Franke & Westerhoff 2012; Künsch 1989), so heterogeneous moments are
commensurable. Excess kurtosis is tracked as a diagnostic but not optimised (it is
outlier-dominated). KS and Hill are paired deliberately: KS alone lets the tail blow up
while Hill alone leaves the bulk distribution untargeted — each controls a failure mode the
other cannot.

The optimum is found by a **two-stage surrogate-assisted simulated method of moments**
(`scripts/calibrate.py run`; Gao et al. 2022 §4.2; Lamperti et al. 2018): stage one is an
XGBoost surrogate θ → D(θ) with a Sobol design and active learning (the global search), and
stage two is a tight local grid on the true simulator, centred on the surrogate optimum, that
sharpens it. A separate exhaustive global grid over the full box (Gao et al. 2023) is deferred
given its runtime cost. `D` is comparable only at a fixed seed count (the KS component's
standardiser is sim-sized).

**Regimes.** A calm regime (13 Sep–27 Dec 2019, 75 sessions) and a stressed regime (the full
COVID window, 17 Feb–28 May 2020, 73 sessions). Each simulation is capped to its regime's data
window, so simulated and empirical moments span the same period.

### 5.1 Locked baseline

> **Pending re-lock.** The numbers in this subsection are the prior-structure optima (calm 4-d /
> stressed 5-d, with `p_zi` L2-fixed in calm). The behavioural loop is now 7-d in both regimes
> (`p_zi`, `mt_lambda`, `mt_gamma` added; surrogate two-stage), so the θ, identifiability and `D`
> values below are a record pending the widened-bounds re-lock — the method and the moment
> battery are unchanged.

The behavioural θ is the grid optimum, reconfirmed by the final seed-count-matched relock
(`output/thesis_final/relock/`; matches `globals.CALIBRATED`). It is cross-checked four ways: a
surrogate run, a fresh-common-seed re-rank, beyond-floor probes, and out-of-sample moments. In
**calm** the grid argmin matches the lock exactly. In **stressed** the in-sample grid argmin drifts
to `ft_sigma_c = 0.85` (D_grid 5.16), but the fresh-seed re-rank exposes this as winner's curse:
re-scored on common fresh seeds that node ranks third, while the top fresh node returns the locked θ
(`ft_sigma_c = 0.25, zi_alpha = 0.32, p_zi = 0.18`; D_fresh 6.09). The lock is therefore retained — a
clean illustration of the weak identification of the stressed depth/tail levers, where the in-sample
argmin wanders a flat direction and only the fresh-seed re-rank pins the robust value.

| Regime | ft_sigma_c | zi_alpha | zi_delta | p_zi | zi_mu | D_grid (3-seed) |
|---|---|---|---|---|---|---|
| calm | 0.65 | 0.38 | 0.02 | 0.543 (L2-pinned) | 0.08125 | 43.79 |
| stressed | 0.25 | 0.32 | 0.02 | 0.18 | 0.0583 | 6.01 |

(Stressed row is the locked θ; the in-sample grid argmin reaches D 5.16 at `ft_sigma_c = 0.85` but
is rejected by the fresh-seed re-rank above. `D` compares only at a fixed seed count.)

Only `zi_alpha` is sharply identified in both regimes; `ft_sigma_c` is flat in calm and on the 0.25
box floor in stressed, and the ZI cancellation/market rates sit near their floors — so grid, surrogate
and fresh-seed agree on `zi_alpha` but differ on the weakly-identified levers (report the loss-surface
flatness, not a unique optimum). Grid `D` (3 seeds) and surrogate `D` (6 seeds) are not directly
comparable: the KS standardiser is sim-sized, so `D` only compares at a fixed seed count.

On the SV-MJD fundamental the **ZI rates are invariant** (grid optimum shares
`zi_alpha = 0.32, zi_delta = 0.02, zi_mu = 0.0583` with the baseline mid lock), while the two
identified levers shift to `ft_sigma_c = 0.45, p_zi = 0.08`; the surrogate interior
(`ft ≈ 0.32, p_zi ≈ 0.15`) sits between the two grid optima, so the synthetic-path optimum
lives in the same basin, only flatter (fit worse on the synthetic path, D 7.35 vs 5.76, as
expected). The robust statement is "same ZI rates; the tail/depth levers land in the same
neighbourhood," not "identical optimum."

### 5.2 Stylised facts reproduced

The model reproduces a fat-tailed return distribution (stressed Hill ≈ 3.03 vs empirical
≈ 3.18, calm ≈ 2.78 vs ≈ 2.96), near-zero linear return autocorrelation, and short-horizon
volatility clustering. Long-horizon clustering (absolute-return ACF beyond ~30 minutes)
remains a documented structural limit — a single-timescale momentum cohort cannot produce
multi-scale memory (Cont 2005). Excess kurtosis runs above empirical (outlier-dominated) and
is a diagnostic only; Hill is the tail measure the loss targets.

The calm fit floors at D ≈ 44, of which a large share is a window-selection artifact: the
calm sim runs the first 20 sessions of 2019 (elevated post-2018-selloff volatility) but is
scored against full-year targets. Scored against its own window the calm D drops to ≈ 26.5
with return std within 5% and |r|-ACF lag-1 essentially exact — the residual is the KS body
shape and a partial Hill gap.

### 5.3 Robustness campaign

A campaign (eight configurations x both regimes) tested whether richer agent dynamics
improve the fit. The verdict is a parsimony result: the baseline is hard to beat. Adding
FT/MT activation + per-order cancellation, a second momentum cohort, or more momentum
traders gave no reproducible gain — and a configuration that looked markedly better at
screening resolution failed to replicate at higher resolution (high surrogate R² does not
guarantee a reproducible optimum; replication across seeds is the real test). The only
consistent improvement was calibrating the momentum half-life (a small stressed gain); this is
now adopted — `mt_lambda` is in the loop alongside a new tanh activation scale `mt_gamma`
(distinct from the flat Bernoulli gate tested above), since the trend-strength-scaled momentum
channel shapes the volatility-clustering profile. One result doubles as a scope condition: calibrating
with the clearing tier active leaves calm unchanged but materially shifts the stressed fit,
so the market calibration is not clearing-invariant under stress, and the baseline is
calibrated bare-market.

### 5.4 Moment coverage (out-of-sample validation)

Beyond the point loss `D`, the locked θ is validated by the **Moment Coverage Ratio** (MCR; Franke
2009, Franke & Westerhoff 2012; the HFABM protocol §4.3). For each of K = 10 point moments the
empirical 95% interval is `m_emp ± 1.96·s_i`, where `s_i` is the moment's block-bootstrap sampling SD
(stationary bootstrap on the empirical series); the per-moment MCR is the fraction of model runs that
land inside, and the joint MCR the fraction inside all K at once (`scripts/calibrate.py mcr`;
`output/thesis_final/mcr_{regime}.json`, M = 80 runs).

**Stressed validates well: mean per-moment coverage 87%, with 9 of 10 moments inside the empirical
interval** — return std (100%), all four return-ACF lags, the |r|-ACF at 5/10/20 min, and the Hill
tail index (76-100% each). The sole miss is the shortest |r|-ACF lag (1 min: model median 0.315 vs
empirical 0.256 — slight over-clustering at the very short lag), which alone holds the **joint MCR to
2.5%** against a 62% bootstrap ceiling (the joint level the empirical series itself would clear).
Reported as "9 of 10 stylised-fact moments within empirical sampling error," the stressed fit is
strong.

Calm coverage is low against full-year intervals (mean 21%, joint 0%), and the per-moment
decomposition attributes this to the two effects already named in §5.2 rather than to a behavioural
misfit: return std and the lag-1 |r|-ACF miss because of the calm-window selection artifact (the model
matches its *own* 20-session window to 2-3 significant figures but is scored against full-2019
intervals), while the |r|-ACF beyond 5 min is the single-timescale long-horizon-clustering limit (Cont
2005). Calm MCR is therefore reported against both the full-year and a matched-window interval, the
latter lifting coverage materially.

## 6. Results — contagion experiments

> **Source.** Numbers below are the committed run `output/thesis_final/experiments/` (driver:
> `scripts/run_thesis_experiments.py`, **40 seeds/arm, both regimes**; θ = `globals.CALIBRATED`;
> IM_FLOOR=0.04, IM_CAP=0.12, MPOR=2, DF_STRESS_FLOOR=0.08). **⚠ STALE (D77):** the live config is now
> MPOR=1 + IM cap dropped (`IM_CAP=None`), so every IM dollar figure rescales ≈0.71× in stress and the H2
> band restates to 4%→~11.4% (2.8×, uncapped) — re-run `scripts/run_thesis_experiments.py` before quoting.
> Full per-arm tables, dispersion, and caveats live in `agent_context/EXPERIMENT_RESULTS.md`; figures in
> `results_figures.ipynb`.

All contagion experiments run the standard cleared population (no engineered fragile client).
The fundamental path `V_t` is a **single fixed historical series per regime** (calm ≈ −5.4%,
stressed ≈ −25.6% peak-to-trough), so the 40 seeds vary the **agent RNG only** and the drawdown
is ~identical across every arm: read this as **response-to-a-fixed-shock**, not path/scenario
uncertainty (the latter needs the GBM ensemble, currently non-runnable — see `RESULTS_PLAN.md` §3).
Clients post the bulk of margin (client IM share ≈75%, toward the empirical CME ~67-70% split).
Report dispersion (mean ± sd across seeds), not means only.

### 6.1 H1 — tiered vs direct clearing

The client tier strongly absorbs stress at the mutualised layer, by relocating loss to
individual members rather than erasing it. On the actual COVID-stressed path (−25.6%, 40 seeds),
**tiered clearing never draws the mutualised default fund (0/40 seeds, deepest waterfall level 0)** —
the ≈1.3 client defaults/seed are borne by the carrying member's own capital before they can reach
the pool — **while direct clearing reaches the mutualised layer (L≥3) in 80% of seeds** (mean depth
2.6/5), burning CCP skin-in-the-game (≈$94M) and ≈$1.6B of mutualised resources. So H1 is a
*mutualisation-threshold* result: the tier converts socialised losses into losses concentrated on
member balance sheets rather than erasing them (Galbiati & Soramäki 2013). **Confound to disclose:**
tiered also holds ≈3.2× the initial margin (≈$32.6B vs $10.3B direct), because members post house IM
on top of gross client IM — so report IM alongside waterfall depth and frame the protective buffer as
member *capital*, not margin. Calm: neither arm defaults or mutualises.

### 6.2 H2 — margin procyclicality

The reactive (procyclical) IM **fraction runs the full floor→cap band, 4%→12% — a 3.0× rise** —
over the stressed window (close-to-close daily VaR), with the 12% cap binding in ≈38% of stressed
sessions; in calm it sits at the 4% floor (≈89% of sessions). That ratchet is the funding-liquidity
channel: as the crash drives `im_fraction` up, more member cash is locked in CCP escrow, draining
free cash from the numerator of `κ = cash / exposure` and **forcing the leverage-ratio breach and the
consequent deleverage** (§3.1, §3.3). The margin *level* trades funding cost against defaults: cheap
flat-5% is under-margined (≈4.1 client defaults/seed), rich flat-12% is well-margined (≈0.65), reactive
sits between (≈1.3) — and reactive is cheap in calm (≈$14.8B posted, ~42% of flat-12's $35.4B) but
ratchets to flat-12 levels in the crash (≈$32.6B mean / $39.8B peak), spiking collateral demand exactly
when funding is scarcest (ESRB "dash for cash"; Glasserman-Wu 2018 trade-off). **Framing:** the price
path is exogenous and identical across all H2 arms, so this is a **collateral-demand / leverage-cycle**
result, *not* crash amplification — do not headline "amplifies the crash." **Caveat:** the `static` arm
is currently a no-op (≡ `reactive`: with `IM_DAILY=True` the daily-σ branch in `simulation.py` precedes
the `IM_MODE` check); the `flat` arms are unaffected — fix or drop `static` before quoting it.

*A coupling to interpret with care.* The mutualisation-depth comparison across margin regimes
is entangled with the SLOIM sizing of §3.2: the cover-2 fund is sized `(stress_move −
IM)·notional`, so a higher IM mechanically **shrinks** the pooled default fund even as it
raises members' posted collateral. Any statement about how the margin regime changes
mutualisation frequency is therefore as much a statement about the cover-2 sizing convention
as about procyclicality — do not headline "procyclical margin reduces contagion" off it. The
clean framing: in a cover-2 SLOIM framework the two margin levers trade off — reactive smooths
the solvency channel but spikes liquidity; through-the-cycle smooths liquidity but thins the
fund. A robustness variant that sizes the default fund on a fixed stress scenario, independent
of the current IM, would isolate the pure procyclicality channel.

### 6.3 NET — gross vs net client margining

Switching the client-margin basis from gross (US/CME) to net (EU omnibus) **reduces posted client
initial margin to ≈0.58× of gross — about a 42% saving** (`CLIENT_MARGIN_NETTING`; §3.1): ≈$19.0B
vs $32.6B stressed, ≈$8.3B vs $14.8B calm, with little change in defaults. Netting offsetting client
positions within an omnibus account lets the member post collateral on the net rather than the gross
exposure — the Duffie-Zhu (2011) netting benefit, measured directly on the cleared population. (Earlier
drafts said "halves"/0.81×; the committed figure is 0.58×.)

### 6.4 Close-out recovery — a secondary lever

The close-out recovery rate is a **secondary** lever next to the structural choices. Dropping
the recovery to the low "disorderly / Aas-class" setting (`CLOSEOUT_RECOVERY = 0.60`) draws the
mutualised default fund in only **1 of 5 GBM paths** — far less consequential than the
structural levers (direct clearing, and open-market disposal of the defaulted book), which
dominate the resilience outcome. The two recovery regimes have clean empirical anchors: Lehman
(2008) was a liquid book closed out within initial margin at high recovery (the baseline 0.80
"transfer" arm), whereas Aas/Nasdaq (2018) was a concentrated, illiquid book whose default fund
was drawn (Bell & Holden 2018, BIS) — the low-recovery stress arm.

### 6.5 E6 — overnight gaps vs a single shock

At matched total drawdown, a single concentrated shock is far more destructive than the
real overnight-gapped path. At `c = 2.0` (≈ -34%): the shock causes 27 client + 3.1 member
defaults and reaches pooled-DF mutualisation in 90% of seeds; the gapped path causes 20.7
client + 0.4 member defaults and reaches L3 in 0% of seeds (it does not touch L3 until
`c = 3.0`, and only in 7.5%). Contagion tracks the concentration and speed of a move, not
its total magnitude; the overnight-gap structure is a mitigant because variation margin
collects between jumps, de-risking the book before the next gap.

### 6.6 Open-disorder sensitivity

Suppressing the session-open re-anchoring (`REANCHOR_ON_GAP = False`) so the gap produces a
disorderly open adds only ~3-8% more client defaults and nudges L3 frequency at 4x from
0.325 to 0.40. A disorderly open worsens contagion modestly but is a second-order effect, so
the clean-reprice default is a defensible simplification.

## 7. Limitations

- **Long-horizon volatility clustering** is not fully captured (the single-timescale
  momentum limit; a two-cohort revival did not fix it).
- **Mutualisation-depth in H2 is coupled to the default-fund sizing convention** (§6.2); the
  liquidity side of H2 is unambiguous, the solvency side needs the fixed-fund robustness run.
- **The stressed calibration is not clearing-invariant** — the bare-market θ understates the
  clearing-active stressed dynamics.
- **The call-auction print convention** clears crossings at the resting ask, a minor
  asymmetry to disclose.

## 8. References (how each is used)

- **Simudyne CCP Risk Model ODD** and **Deloitte**, *Modelling CCP resilience* — clearing-tier
  scaffold, margin cadence, waterfall structure, star topology (deviations flagged: order TTL
  removed; a defaulted member is closed out at a recovery rate with the haircut mutualised —
  no counterparty-selecting PositionAuction; a defaulted tiered client is liquidated
  open-market by its member; a BCM breaching the floor deleverages its own book (AC) while an
  NBCM stops out — the ODD's asymmetry preserved).
- **Euronext Clearing A9** (cover-2 SLOIM, reverse-stress multiplier); **CME SPAN** and
  **EMIR Art. 41/45** (VaR initial margin, exchange SITG; physical IM escrow); **Basel
  leverage ratio** / **Haynes, McPhail & Zhu 2019** / **Acosta-Smith et al. 2018** (the
  net-position capital floor `κ = cash/exposure`; **CFTC Reg. 1.17** is the related FCM
  net-capital minimum); a **static gross-leverage position limit** on the BCM own book
  (cf. **Basel III FRTB**); **EMIR Art. 48(5)-(6)** (client porting);
  **BCBS-CPMI-IOSCO 2025** (the H2 margin-responsiveness metrics); **Glasserman & Wu 2018**
  (the through-the-cycle anchor for H2).
- **Majewski, Ciliberti & Bouchaud 2018** (FT value + MT momentum; externally-fixed trend
  horizon); **Gao et al. 2022** (surrogate-assisted SMM, the data-derived fundamental, the KS and
  Hill targets); **Gao et al. 2023** (grid-search calibration); **Cont, Stoikov & Talreja
  2008** and **Farmer, Patelli & Zovko 2004** (ZI rates, geometric placement, the
  cancellation that governs order lifetime).
- **Barndorff-Nielsen & Shephard 2004**, **Mancini 2009**, **Huang & Tauchen 2005**, **Lou,
  Polk & Skouras 2019** (the SV-MJD realized-measure parameters and overnight-gap pool);
  **Stein & Stein 1991 / Heston 1993** and **Merton 1976** (the SV-MJD architecture).
- **Almgren & Chriss 2000** (fire-sale schedule); **Cont 2001** (stylised facts), **Cont
  2005** (multi-scale clustering); **Hill 1975** (tail index); **Franke & Westerhoff 2012**
  and **Künsch 1989** (block-bootstrap moment weights); **Lamperti, Roventini & Sani 2018**
  (surrogate-calibration inspiration).
- **Duffie & Zhu 2011**, **Galbiati & Soramäki 2013**, **OFR 2026** (the tiering trade-off and
  single-agent dependency framing for H1); **Paddrik, Rajan & Young 2020** (the closest
  modelling cousin — fixed network + exogenous shocks, vs endogenous LOB price formation and a
  client tier here).

### 8.1 Clearing-layer element → source (for the methodology write-up)

Citable basis for each clearing mechanism, for the methodology chapter. "bib" = key in the
thesis `references.bib`; "ADD" = primary disclosure to add if you cite the specific figure.

| Element (model) | Real-world basis | Citation | bib |
|---|---|---|---|
| Initial margin = 99% VaR, 1-day MPOR, 4% floor, no cap | SPAN/VaR; 99%; CME 1-day futures liquidation; APC 10-yr vol floor | CME SPAN 2; EMIR RTS 153/2013 Art 24/26/28 (2-day = OTC min) | `emir_rts`; CME SPAN (ADD) |
| Variation margin, fill-price, hourly | ≥twice-daily MtM + intraday calls; settled-to-market | CME Financial Safeguards | CME FSG (ADD) |
| Default fund = cover-2 SLOIM (+10% buffer) | cover-2; stress-loss-over-IM | EMIR 648/2012 Art 43; CPMI-IOSCO PFMI P4; Euronext A9 | `emir_rts`; `euronext_a9`; PFMI (ADD) |
| DF stress move (FHS ~99.9% q=3.9, 8–35%) | extreme-but-plausible 1-day scenario | Euronext A9 §2.1 | `euronext_a9` |
| Skin-in-the-game = 3% of fund | empirical SITG/DF ~1–4% (CME ES ≈2.7%); EMIR DOR = 25% of capital | EMIR 648/2012 Art 45; Clarus | `emir_rts`; Clarus (ADD) |
| Capital floor 8% (κ = cash/exposure) | net-position leverage ratio (Basel) ≥ 8%; cf. adjusted-net-capital ≥ 8% risk margin | Basel leverage ratio; Haynes-McPhail-Zhu 2019; Acosta-Smith 2018; CFTC Reg 1.17 | `haynes_mcphail_zhu_2019`; `acosta_smith_2018`; `cftc_reg117` |
| BCM own-book static gross-leverage cap (own notional ≤ 2·cash) | flat position limit, no vol feedback | prop-book position limit; cf. Basel III FRTB | `basel_frtb` |
| 5-level waterfall order | defaulter → SITG → pooled DF → survivor cash → CCP | CME Financial Safeguards; EMIR Art 45 | `emir_rts`; CME FSG (ADD) |
| Client porting | transfer to back-up member, else liquidate | EMIR 648/2012 Art 48(5)-(6); CFTC Part 190 | `emir_rts`; CFTC 190 (ADD) |
| Member / direct-client close-out = CCP transfer at 80% recovery (no counterparty selected) | auction/transfer recovers near par (Lehman ~35% of IM); ODD 0.60 anchor | BIS Quarterly Review Dec 2018; IOSCO PD657 (2017); Simudyne ODD | BIS 2018 (ADD); IOSCO (ADD); `simudyne_odd` |
| Tiered-client close-out = member open-market Almgren-Chriss liquidation | member liquidates an assumed (small) client book into the market | Almgren & Chriss 2000 | `almgren_chriss_2000` |
| Open-market AC fire-sale of a member book (deferred hypothesis) | auctions exist to avoid open-market price impact | Almgren & Chriss 2000; Vuillemey 2020 | `almgren_chriss_2000`; `vuillemey_2020` |
| Member/client cash scale | FCM adjusted-net-capital ($5–10B bank / $50m–1B non-bank); CPMI-IOSCO PQD | CFTC Financial Data for FCMs; Clarus | `cftc_fcm_data`; Clarus (ADD) |
| Clients = majority of CCP margin (H1) | clients ~73–82% of IM | OFR 2026; FIA (CME 82%) | `ofr_2026`; FIA (ADD) |
| Procyclicality / liquidity (H2) | margin spiral; COVID dash-for-cash; TTC | Brunnermeier-Pedersen 2009; Glasserman-Wu 2018; Cont 2017; ESRB 2020; BCBS-CPMI-IOSCO 2022/2025 | `brunnermeier_pedersen_2009`; `glasserman_wu_2018`; `cont_2017`; `esrb_2020`; `bcbs_cpmi_iosco_2022/2025` |
| Tiering trade-off (H1) | netting/concentration vs loss-absorption | Duffie-Zhu 2011; Galbiati-Soramaki 2013; Borovkova 2013 | `duffie_zhu_2011`; `galbiati_soramaki_2013`; `borovkova_2013` |

Real-scale grounding numbers (CME IM ≈$292B, prefunded DF ≈$9.4B, SITG ≈$100M, DF/IM ≈3–4%,
client share ≈82%) and full source URLs are in `CLEARING_LAYER_REVIEW.md` (§3, §7, §8).
