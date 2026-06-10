# An Agent-Based Model of Central Clearing

An agent-based model (ABM) of a centrally-cleared single-asset futures market —
the E-mini S&P 500 (ES front-month) — built to study CCP systemic risk: clearing-
member default risk, margin procyclicality, and client-clearing contagion under
calm and stressed market regimes. The market-microstructure layer is built and
calibrated first; the central-clearing and default-management layer is layered on
top once the market dynamics reproduce the empirical stylised facts.

This README documents the **current, decided state** of the model — agents,
equations, the fundamental process, parameters, how everything is calibrated, and
why each choice was made. The detailed change history and the AI-collaboration
rules live in `AGENT.md`.

> **Writing the thesis?** See **`writing.md`** for the results with exact numbers, the
> narrative, the chapter structure, the honest limitations, and how each reference is used.
> **`README.md` + `writing.md` are the complete context pair** — this file is the *model /
> code reference*, `writing.md` is the *findings / writing reference*. Nothing else is needed
> for writing (`AGENT.md` is dev-only).

## The model in brief

The model has **two tiers**. The first is a **market**: a limit order book where three kinds of
traders — *fundamental* (value-driven), *momentum* (trend-following) and *zero-intelligence*
(noise) — trade S&P-500 (ES) futures around an exogenous "fair value" path taken from real ES
data. This layer is *calibrated* so the simulated price reproduces the real statistical
fingerprints of the market (fat-tailed returns, volatility that clusters in bursts, no
predictable drift). It is the realistic stage on which the second tier acts.

The second tier is the **clearing system**. Trades are guaranteed by a **central counterparty
(CCP)**; most participants ("clients") don't face the CCP directly but **clear through a member**
— a bank or broker (a banking or non-banking clearing member, BCM/NBCM). Everyone posts **margin**
(collateral) sized to their risk, and the CCP also holds a mutualised **default fund**. As prices
move, margin is exchanged; a participant that runs out of cash **defaults**, and its losses are
absorbed by a fixed **waterfall** — first its own collateral, then the fund, then, in the extreme,
the *surviving members* (the "mutualisation" that spreads one default's loss across the system).

The thesis question lives at that junction: **does clearing clients through members, rather than
directly, change how a stress propagates?** The model lets us watch a client default flow up to
its clearing member and out into the rest of the system, and measure how severe a shock must be
before the mutualised layer is touched. The sections below specify each element — agents, the
fundamental, the margin/default machinery, and how everything is calibrated and grounded in the
literature.

## Research questions

The full (market + clearing) model is built to address:

- Does tiered **client clearing** change the frequency and severity of CCP stress
  versus direct clearing?
- Do banks that both trade proprietarily and clear clients (the banking clearing
  members here) face higher default risk?
- Does **transient price impact** in distressed liquidations amplify contagion?
- How does **margin methodology** interact with client clearing and concentration?

The market layer produces realistic price/volume/volatility dynamics so the clearing
layer has a credible environment to act in. The margin-call → fire-sale → price-impact
feedback that transmits clearing-member distress to surviving members runs on top of it;
the real COVID-crash window exercises it end to end (client default, member default,
waterfall, and CCP fire-sale all fire), and a reverse-stress sweep locates the cover-2
breach point.

## Current state

**Implemented and verified (market layer):**

- A single limit order book with 1-minute call-auction clearing. Real RTH
  sessions are ≈405 bars and vary day to day (D57: session boundaries are read
  from the data, not a fixed 390-bar grid), and the simulator opens each session
  at the gapped fundamental.
- A three-type LOB trading population calibrated on real ES data: fundamental
  traders, momentum traders, and zero-intelligence traders. (The market maker and
  the volatility trader were both removed — see *Agents* and `AGENT.md`.)
- An **exogenous fundamental** read from a pre-generated path. The primary
  fundamental is a **Kalman-filtered real efficient price** of the ES mid; a
  synthetic stochastic-volatility Merton jump-diffusion is retained as an
  alternative for robustness.
- A two-method offline-and-agent calibration pipeline: data-side parameters fit
  directly from ES data and order-book data; behavioural parameters fit by
  matching simulated to empirical return moments, via **both** a surrogate-assisted
  simulated method of moments and an exhaustive grid search.
- A 12-cell ablation study validating each design choice (see `AGENT.md`).

**Implemented and verified (clearing layer):**

The central-clearing tier is built end to end: one central counterparty, ten banking
clearing members (cast from the fundamental type; they trade their own account *and*
clear a client book), five non-banking clearing members, balance sheets, and the
star-topology member↔CCP links. The 60-minute variation-margin cycle settles each
cleared book's mark-to-market move into member cash in **USD** (× `VOLUME_LOT` ×
`CONTRACT_USD`); client positions are **novated** live into the clearing member's book.
The margin model is grounded in real methodology: **initial margin is a procyclical
VaR/SPAN scan** (`im_fraction(σ)` — a 99% / 2-day VaR floored by an anti-procyclicality
floor anchored to the CME ES margin, ≈6% calm rising to ≈12% stressed); the **capital
ratio is cash / initial margin** (CFTC Reg 1.17: adjusted net capital ≥ 8% of risk
margin, not gross notional); the **default fund is a cover-2 Stress-Loss-Over-Initial-
Margin** fund (Euronext Clearing module A9 — the two most-exposed members' loss above
posted margin under an extreme-but-plausible move, ×(1 + 10% buffer)). A banking CM's
own (house) book runs under a **VaR risk limit** (Basel FRTB) so it cannot accumulate an
unbounded prop position. On a capital breach a member **deleverages via Almgren–Chriss**
(banking CM fire-sale; non-banking CM stop-out); on cash exhaustion it **defaults**, the
CCP runs the **five-level waterfall** on the member's realised **cash deficit** (its DF
→ exchange skin-in-the-game → pooled member DF → surviving-member cash → exchange) and
**fire-sells the book** as Almgren–Chriss market orders whose price impact transmits the
loss. A client default is handled the same way: the CM **assumes and liquidates** the
position, the close-out loss realised by marking it as the fire-sale walks the price
(deficit-consistent — no flat haircut).

Verified end to end on the **real COVID-crash window** (the sim is pointed at the
ES 2020-03 rows of the stressed series, **overnight gaps retained** — see *Fundamental*),
with the standard cleared population — no engineered fragile client. At actual COVID
severity (the gapped window draws down ≈−19% over ten RTH days, ≈−33% over the full
episode) the cover-2 framework **contains** the book: a handful of thin-capital clients
default but are **absorbed by their clearing members** — no member default, no
mutualisation. A reverse-stress sweep (Euronext A9 §5 — amplify the real path by a
multiplier *c*) then traces the escalation: client defaults grow with severity but stay
CM-absorbed through ≈−35%; cross-member **mutualisation** (waterfall Level 3–4) onsets
only at **≈2.5× COVID (≈−41%)**, where non-banking members default. The CCP stays solvent
and the default fund stays at a realistic ≈$0.5B throughout — the framework absorbs
individual defaults and mutualises only under a beyond-COVID shock, as a sound cover-2
design should.

## Market and agents

The asset is the ES front-month future. Prices move on a 0.25 tick; the simulation
steps every minute. Each LOB trader submits orders to a shared book, and a call
auction clears any crossed quotes at the end of every step. The mid is
`(best_bid + best_ask)/2`, holding at the last trade price when a side is empty.

**Order lifetime** is governed by the agents alone — there is no blanket time-to-live
(the ODD's 10-step order ceiling was removed, D58). Fundamental, banking-clearing-member,
and momentum traders use *replace-on-new* management: each holds at most one standing
limit order and, the next time it acts, cancels that order and places a fresh one, so its
resting order always reflects its latest view (they act every step). Zero-intelligence
traders instead cancel each resting order with a fixed probability `zi_delta` per step —
the Cont-Stoikov-Talreja (2008) / Farmer et al. ZI cancellation rate, which is now the
*sole* control on ZI order lifetime (calibrated, floored at 0.05 so the book cannot
accumulate stale orders). Resting orders otherwise leave the book only by fill. This
deviates from Simudyne ODD §Mech #7 — flagged because the explicit per-order cancellation
rate is the standard ZI order-lifetime mechanism and the blanket TTL was redundant with it
(and masked it whenever `zi_delta` was low).

There are three LOB trader types.

**Fundamental trader (FT).** A value strategy (Simudyne CCP ODD §Agents; the
Chiarella–Iori–Perelló family). Each FT holds a fixed idiosyncratic draw
`z ~ N(0,1)` and forms a reservation price `R = V_t + z · σ_fundamental_t` around
the exogenous fundamental, where `σ_fundamental_t = ft_sigma_c · σ_t · v0` tracks
the current local volatility `σ_t` of `V_t` — the belief cloud widens in
high-volatility regimes and tightens in calm ones. Every step it places one limit
order *at* the reservation on the implied side (`sign(R − mid)`), replacing any
order it had standing. The sign split across the population keeps order flow
two-sided as the fundamental drifts; there is no dead-band, since the persistent
`z` already supplies the heterogeneity.

The draw `z` is fixed at initialisation, not redrawn each step. This is a
*heterogeneous-beliefs* specification (Chiarella–Iori–Perelló): agents disagree about
value and the disagreement persists, so high-`z` FTs accumulate long positions and
low-`z` FTs accumulate shorts, building **concentrated inventories** over time. That
concentration is the load-bearing input to the clearing layer — margin calls, the
default fund, and the waterfall only have something to act on when clearing-member
balance sheets are skewed by positions.

`ft_sigma_c` (the belief-width scale) is **calibrated**, and the ablation showed it
is the single dominant lever on the return tail: too wide and the FTs overshoot the
book, producing kurtosis in the thousands; calibrated low (≈0.6) it reproduces the
empirical tail. This supersedes the earlier convention of pinning it at √390 ("one
daily fundamental standard deviation").

**Momentum trader (MT).** A trend-following chartist (Majewski, Ciliberti & Bouchaud
2018; Gao et al. 2022 HFABM). Each MT tracks an exponentially-weighted moving
average of mid log-returns, `M_t = (1−λ)·M_{t−1} + λ·r_t`, and on the side of the
trend places a passive limit order at a placement depth drawn from the shared
geometric distribution. The MT is limit-only — trend amplification comes from
inside-the-spread MT limits being lifted by trend-following market flow. All MTs
share a single EWMA decay `mt_lambda = 0.05`, pinned as a design constant: Majewski
et al. (2018) likewise fix the trend horizon externally rather than estimating it.

**Zero-intelligence trader (ZI).** A Cont, Stoikov & Talreja (2008) / Farmer, Patelli
& Zovko (2004) noise trader providing the background liquidity floor. Each step it
independently may cancel a resting order (probability `zi_delta`), submit a limit
order at a random side and geometric depth (probability `zi_alpha`), or submit a
small market order (probability `zi_mu`) — with no signal dependence. `zi_alpha` and
`zi_delta` are calibrated; `zi_mu` is pinned at the literature baseline 0.025 (the
ablation showed calibrating it changes the fit negligibly). The ZIs absorb aggressive
market orders and keep the book from gapping.

**Order placement (ZI and MT)** is geometric: a limit rests `k ~ Geometric(p_zi)`
ticks from the mid, with most mass at the touch (Cont, Stoikov & Talreja 2008;
Farmer, Patelli & Zovko 2004). `p_zi` is fit directly from real L2 (MBP-10) book
depth (`data/p_zi.py`), per regime. This replaced an earlier calibrated log-normal
placement; the geometric form is the one grounded in measured book depth.

**Removed agents** (do not re-add without a strong citation; see `AGENT.md`):

- *Market maker* — an always-quoting mid-anchored MM (Gao et al. 2022 §3.6) was
  trialled, removed, re-added, and removed again. The ablation confirmed it hurts
  both regimes: it damps volatility without improving clustering. Near-mid liquidity
  now comes from the data-fit geometric ZI book.
- *Volatility trader* — a Gao et al. (2023) stochastic-volatility construct. The
  ablation found it helps the *stressed* tail markedly but harms calm, so it is held
  as a possible regime-specific option rather than a standing agent.

### Clearing tier

Layered on top of the LOB population (Simudyne CCP ODD, extended with this thesis's
client-clearing layer):

- **Banking clearing member (BCM, ×10)** — a fundamental-trader subclass. It trades
  its own account through the inherited FT logic, so for the market-layer calibration
  30 FT + 10 BCM is simply a 40-FT-equivalent population. It additionally carries a
  balance sheet, a CCP link, and a client book. Capital ratio = `cash / initial_margin`
  (CFTC Reg 1.17 — adjusted net capital ≥ 8% of *risk margin*, not gross notional). Its
  own (house) book is capped by a **VaR risk limit** (own 99%/1-day VaR ≤ 5% of capital;
  Basel FRTB) — without it the FT logic accumulates an unbounded prop position in a trend
  that would dwarf the client books and inflate the default fund. The limit is procyclical
  (tightens in the stressed regime to ≈0.6× capital) and lives only on the runtime BCM
  path, so the market-layer calibration (which uses plain FTs) is unaffected.
- **Non-banking clearing member (NBCM, ×5)** — a pure clearing intermediary: no own
  position, no LOB orders. Carries a balance sheet and a client book only.
- **Central counterparty (CCP, ×1)** — the clearing role of the ODD matching engine
  (order matching itself stays in the LOB). Holds the member registry, default-fund
  accounts, and the default list; wires the bidirectional member↔CCP star topology.
- **Margin cycle** — every 60 minutes, variation margin settles each cleared book's
  mark-to-market move into member cash in USD; **procyclical initial margin** (a VaR/SPAN
  scan, `im_fraction(σ)`) and maintenance margin are recomputed off gross USD exposure; a
  margin call flags when variation margin exceeds `(1 − mm%)` of initial margin; and a
  member defaults on cash exhaustion. Members use futures-margin accounting (a fill posts
  only the position; cash is settled by the margin cycle).
- **Client clearing (the contagion channel)** — the 90 clients (30 FT + 20 MT + 40 ZI)
  clear through the client-carrying CMs (five of the ten BCMs plus all five NBCMs),
  round-robin, **9 per CM** (ZI clearing is a thesis extension — the ODD has ZI as
  direct, balance-less participants; the other five BCMs are own-account only, so
  client-book concentration stays a study lever). Each client carries its own balance
  sheet: the margin cycle issues it its own variation-margin call, paid from its **own
  cash** (the CM relays it to the CCP, and the client's live inventory is novated into
  the CM book). A client can only open what its cash can margin against its **house
  margin** (a fixed broker leverage cap — 5×, 20% house margin, uniform across clients);
  variation-margin losses then erode that cash until it **freezes** at the
  8% floor (stops trading) and **defaults** at zero. On default the **CM assumes and
  liquidates** the position: it covers the uncollected margin (the client's cash
  shortfall) and takes the position onto its own book, disposing of it via Almgren–Chriss
  (the banking CM sends its own orders; the non-banking CM's are routed by the CCP, both
  attributed to the CM). The close-out loss is realised by **marking the assumed position
  as the fire-sale walks the price** — deficit-consistent, not a flat `(1 − recovery)`
  haircut — draining CM cash and potentially toppling the CM. A non-banking CM fails
  purely through this channel; a banking CM through it *or* its own book.
- **Default fund** — a cover-2 **Stress-Loss-Over-Initial-Margin** fund (Euronext Clearing
  module A9 §3), recomputed each RTH day: per member `SLOIM = max(0, stress_move·notional
  − initial_margin)` under an extreme-but-plausible move (`df_stress_move(σ)`, ≥15%); the
  total is the two largest members' SLOIM × (1 + 10% buffer). The exchange pre-funds a
  skin-in-the-game share (first loss), members fund the remainder pro-rata by margin share.
- **Default management** — a solvent member breaching the 8% **cash/IM** floor deleverages
  via an Almgren–Chriss schedule (banking CM sells its own book down to the floor;
  non-banking CM stops out — freezes its clients). A defaulting member triggers the
  **five-level waterfall** on its realised **cash deficit** (the loss beyond its own
  capital it cannot pay): defaulted-member DF → exchange skin-in-the-game → pooled member
  DF → surviving-member cash pro-rata → CCP cash; its book is handed to the CCP for an
  Almgren–Chriss fire-sale, whose book-walk price impact is the contagion channel into
  surviving members' next mark. The deficit basis replaces an earlier flat
  `(1 − recovery)·notional` haircut that overstated close-out losses ~40× and spuriously
  insolvented the CCP in a moderate crash.

## Fundamental value process

The fundamental value `V_t` is exogenous to the simulation — a path generated offline
and read in as a time series. There are two interchangeable generators; the model is
currently run on the **Kalman** fundamental.

**Kalman-filtered real efficient price (primary, `data/v_kalman.py`).** The observed
ES mid is treated as a latent efficient price plus microstructure noise — a local-level
state-space model: `log V_{t+1} = log V_t + μ + σ_v·Z` (state), `log p_t = log V_t +
ε_t` (observation), with `ε_t ~ N(0, σ_ε²)`. The noise split `(μ, σ_v, σ_ε)` is fit by
Kalman maximum likelihood (per RTH day, so the overnight gap is never inside any single
day's filter), and an RTS smoother extracts `E[log V_t | all obs]` — the efficient
log-price with bid-ask bounce removed. The per-day series are concatenated at their
**real levels, so the overnight gaps are retained** (D56 — the COVID episode then carries
its true ≈−33% drawdown, not the ≈−7% with gaps removed; overnight gap risk is a primary
CCP default driver). The simulator opens each RTH session at the gapped V_t by **repricing
the book** at the session boundary — read from the data (the fv timestamps), since real ES
sessions are ≈405 bars and vary, not a fixed 390-bar day (D57) — a clean between-session
jump, no empty-book warm-up; and the session-boundary return is excluded from the calibration
moments at those same real boundaries, which target intraday returns (matching the empirical
convention) — so the gaps stress the cleared book without contaminating the intraday
stylised-fact fit. The per-minute **local volatility**
`σ_t = √EWMA[r²]` (half-life 30 min) carries the real, multi-scale volatility
clustering and feeds the FT belief width. This is the data-derived fundamental of the
Chiarella–Heston / XGB-Chiarella line (Gao et al. 2022, §2.5.2), and it is consistent
with the spirit of Majewski et al. (2018), who treat the fundamental as a latent state
filtered from price. Because it is the real efficient price (lightly smoothed,
`σ_ε ≪ σ_v`), it carries the real return tails — the agent layer must reproduce them
rather than having them injected by a synthetic process.

**Stochastic-volatility Merton jump-diffusion (alternative, `data/v_gbm.py`).** A
geometric Brownian motion whose instantaneous volatility mean-reverts as a Vasicek /
Ornstein–Uhlenbeck process, overlaid with Poisson jumps:

```
σ_{t+1}              = θ + e^{−α}(σ_t − θ) + s·Z_σ                 (Vasicek-OU)
log V_{t+1} − log V_t = (μ_v − ½σ_t²) + σ_t·Z + J,   J ~ N(0, δ²) w.p. λ/390
```

Diffusion and jump parameters come from the variance and excess kurtosis of ES
1-minute returns; the OU parameters from a 30-minute realised-volatility series; the
long-run level is re-anchored so the SV layer adds clustering without inflating total
variance. The SV architecture follows a Deloitte CCP Risk Model webinar treatment (the
published ODD is incomplete on the fundamental); the formal Ornstein-Uhlenbeck-on-
volatility model is Stein & Stein (1991) / Heston (1993); the jumps are Merton (1976),
with `λ = 3/day` the ODD value.

The ablation compared the two head-to-head: with the tail controlled in the loss
(below), the Kalman fundamental fits *better* than the synthetic SV-MJD in both
regimes, and is far easier to defend (it is the real efficient price, with no synthetic
jump/vol parameters to justify and no risk of "injecting the clustering you claim to
reproduce"). Hence Kalman is primary; SV-MJD is kept as a robustness check.

## Formulas

Time is discrete; each step is one minute. Agent arrival rates are per-step Bernoulli
probabilities (ODD-native — no `dt` rescaling).

Fundamental trader, per step (fixed `z ~ N(0,1)`):

```
σ_fundamental_t = ft_sigma_c · σ_t · v0          (σ_t = local vol of V_t; ft_sigma_c calibrated)
reservation R   = V_t + z · σ_fundamental_t
side            = sign(R − mid)                  (no dead-band)
acts every step: cancel the standing limit, place one fresh limit AT R, qty ~ U[qty_min, qty_max]
```

Momentum trader, per step:

```
M_t  = (1 − mt_lambda)·M_{t−1} + mt_lambda·(log mid_{t−1} − log mid_{t−2})
|M_t| < ε  → no trade
side = sign(M_t)
acts every step: cancel the standing limit, place one fresh limit at mid − side·k·tick,
                 k ~ Geometric(p_zi)
```

Zero-intelligence trader, per step:

```
each resting limit cancelled w.p. zi_delta
submit a limit order  w.p. zi_alpha  at random side, depth k ~ Geometric(p_zi)
submit a market order w.p. zi_mu     at random side          (zi_mu pinned at 0.025)
size ~ U[qty_min, qty_max]
```

Procyclical initial margin (VaR/SPAN scan — for a single linear future SPAN = VaR):

```
σ_daily       = σ_regime · √390                          (1-min return std → daily)
im_fraction   = max(IM_FLOOR, z₉₉ · σ_daily · √MPOR)     (z₉₉=2.326, MPOR=2, floor≈6%)
df_stress_move= max(DF_FLOOR, z_df · σ_daily · √MPOR)    (z_df=3.0, floor 15% — extreme-but-plausible)
```

Clearing, per 60-minute margin cycle (per member):

```
book_position    = own_inventory + Σ client_positions    (own = BCM book, or an NBCM's assumed defaulted-client position)
variation_margin = book_position · (mid − last_mark) · VOLUME_LOT · CONTRACT_USD
exposure         = (|own_inventory| + client_notional) · VOLUME_LOT · CONTRACT_USD · mid
initial_margin   = im_fraction · exposure;   margin call if |VM| > (1 − mm%)·IM
capital_ratio    = cash / initial_margin;    deleverage (Almgren–Chriss) if ≤ 8%   (CFTC Reg 1.17)
default if cash ≤ 0
BCM own-account house limit:  |own notional| ≤ HOUSE_VAR_BUDGET · cash / (z₉₉ · σ_daily)   (β=5%; Basel FRTB)
client house cap:             |position| ≤ cash / (house_im · VOLUME_LOT · CONTRACT_USD · mid)   (house_im=20%)
```

Default management, on a member default — waterfall absorbs the member's realised cash
deficit (its unpaid shortfall beyond its own capital; the fire-sale disposes of the book):

```
cover-2 default fund = (Σ over the two largest of  max(0, (df_stress_move − im_fraction)·|net book|·USD)) · (1 + buffer)
waterfall (absorb the deficit in order):
  L1 defaulted-member DF → L2 exchange SITG → L3 pooled member DF
  → L4 surviving-member cash pro-rata → L5 CCP cash
fire-sale schedule (Almgren–Chriss): x_j / X = sinh(κ(H − j)) / sinh(κH),  κH = urgency
```

## Parameters

Grouped by how each is set: a structural constant, calibrated offline from data, or
calibrated in the agent-parameter loop.

Market and order book (structural):

| Parameter | Meaning | Value |
|---|---|---|
| `tick_size` | price increment | 0.25 |
| `dt_minutes` | step length | 1.0 |
| `qty_min`, `qty_max` | order size range | 1, 10 |
| populations | LOB counts (FT+BCM, MT, ZI) | 40-equiv, 20, 40 (100 on book) + 90 cleared clients |

Fundamental value `V_t` (data-calibrated, per regime):

| Parameter | Meaning | Set by |
|---|---|---|
| `μ`, `σ_v`, `σ_ε` | drift, efficient-price vol, microstructure-noise vol | Kalman MLE (`v_kalman.py`) |
| `σ_t` | per-minute local volatility (EWMA of r², 30-min half-life) | data (`v_kalman.py`) |
| `μ_v, σ_d, λ, δ, α, θ, σ_vol` | SV-MJD diffusion/jump/OU parameters (alternative) | data (`v_gbm.py`); `λ=3/day` (ODD) |

Behavioural parameters — **regime-specific calibrated loop**:

| Parameter | Meaning | Calm (3-d) | Stressed (4-d) |
|---|---|---|---|
| `ft_sigma_c` | FT belief-width scale (the tail lever) | calibrated | calibrated |
| `zi_alpha` | ZI limit-order arrival rate | calibrated | calibrated |
| `zi_delta` | ZI per-resting cancellation rate | calibrated | calibrated |
| `p_zi` | geometric placement-depth parameter | **fixed at L2 value** | **calibrated** |
| `zi_mu` | ZI market-order arrival rate | fixed 0.025 (CST) | fixed 0.025 (CST) |
| `ft_alpha`, `mt_alpha` | FT/MT activation (trade every step) | pinned 1.0 | pinned 1.0 |
| `mt_lambda` | MT EWMA decay | pinned 0.05 | pinned 0.05 |

`p_zi` is calibrated only in the stressed regime (the ablation showed a sparser book
sharpens the stressed fit); in calm it stays at its directly-measured L2 value, which
is the more defensible choice when the measurement is available. `zi_mu` left the loop
because the ablation found calibrating it changes the loss negligibly.

Clearing tier — all regulatory / ODD constants (none SMM-calibrated):

| Parameter | Meaning | Value | Source |
|---|---|---|---|
| `VOLUME_LOT` | contracts per model lot (per regime) | 30 / 60 | matched to empirical ES volume (100-agent pop) |
| `CONTRACT_USD` | CME E-mini ES point multiplier | 50 | CME contract spec |
| `margin_interval` | variation-margin cadence (minutes) | 60 | ODD §Scales |
| `df_interval` | default-fund recalculation (minutes) | 390 | one RTH day |
| `IM_CONF_Z`, `IM_MPOR_DAYS`, `IM_FLOOR` | procyclical IM: 99% VaR, 2-day MPOR, APC floor | 2.326, 2, 6% | EMIR Art. 41 / CME SPAN; CME ES margin |
| `im_percent` (house margin) | **client** position cap (leverage) — not the CCP IM | 20% (5×) | broker house margin over exchange min |
| `HOUSE_VAR_BUDGET` | BCM own-book VaR limit (≤ β·capital) | 5% | Basel FRTB / prop-desk practice |
| `mm_percent` | maintenance-margin threshold | 95% | CME methodology |
| `DF_STRESS_Z`, `DF_STRESS_FLOOR` | DF stress move: ~99.9% VaR, extreme-but-plausible floor | 3.0, 15% | Euronext A9 §2.1; CPMI-IOSCO |
| `df_buffer` | cover-2 SLOIM buffer | 10% | Euronext Clearing module A9 §3 |
| `ex_df_ratio` | exchange skin-in-the-game | 10% | EMIR Art. 45 |
| `cover_number` | cover-N default fund | 2 | EMIR / Dodd-Frank; Euronext A9 |
| `cap_ratio_floor` | CM capital floor on **cash / IM** | 8% | CFTC Reg 1.17 (FCM net-capital rule) |
| `AC_HORIZON`, `FIRE_SALE_URGENCY` | Almgren–Chriss fire-sale schedule | 30 min, κH = 2.0 | Almgren & Chriss 2000 |

(`df_percent` and `recovery_rate` are retained as constants but no longer drive the fund
or the waterfall — superseded by the cover-2 SLOIM and the deficit-based close-out.)

Clearing-member cash (loss-absorbing capital, regulatory-scale; inert for price
formation — it gates no order submission, only the clearing-layer default/deleverage):
CCP $7.5B (LCH SwapClear default-fund cap), BCM `U[$5B, $10B]` (bank-FCM scale), NBCM
`U[$50M, $1B]` uniform (CFTC FCM adjusted-net-capital range, Apr 2026: small clearing
FCMs ~$30–140M up to Marex / Clear Street / ABN AMRO ~$0.8–0.9B; the small end is where
a concentrated client cluster can topple the member).

Client cash (cleared end-users, by entity type — **active**: it caps each client's
position by margin capacity and drives its freeze/default): FT `U[$100M, $500M]` (asset
managers, the largest books), MT `U[$30M, $150M]` (CTAs / trend institutions), ZI
`U[$60M, $150M]` (noise accounts; raised so their random-walk inventories default only on
the unlucky tail). Tuned so the 8% floor binds for a subset under stress but not in calm.

`VOLUME_LOT` and `CONTRACT_USD` are applied at the reporting, notional and margin layer
only — never inside the matching engine — so log-returns, ACFs, and the tail index are
invariant under this multiplicative relabel.

The calibrated values per regime are stored in `model/globals.py` (`CALIBRATED`) and
read by `run_simulation.py` and the analysis notebooks.

## Calibration

The fundamental and order-book parameters are calibrated **directly and offline**: the
Kalman fundamental from the ES 1-minute mid (`data/v_kalman.py`); the geometric
placement-depth `p_zi` from real L2/MBP-10 book depth (`data/p_zi.py`); and a
market-impact regression on the 1-minute best-bid-offer data (`data/impact.py`), kept
for a later optimal-execution extension but not currently wired into any agent. These
are fixed per regime before any agent calibration.

The behavioural parameters are calibrated by **matching simulated 1-minute return
moments to the empirical ES moments**, regime by regime, as two independent problems.
The target battery is the Cont (2001) stylised-fact set, and the loss is a grouped,
standardised-moment distance with **five components**:

| Component | What it matches | Source |
|---|---|---|
| ΔKS | whole return distribution (2-sample Kolmogorov-Smirnov) | XGB-Chiarella (Gao et al. 2022) §3.2.4 |
| ΔV | return standard deviation | Gao et al. 2022 |
| ΔACF1 | autocorrelation of returns, lags {1, 5, 10, 20} | Cont 2001 |
| ΔACF2 | autocorrelation of \|returns\| (volatility clustering), lags {1, 5, 10, 20} | Cont 2001, 2005 |
| ΔHill | Hill (1975) tail index (banded) | HFABM (Gao et al. 2022) §4.1.1 |

Each moment's distance is divided by its empirical **block-bootstrap sampling standard
deviation** (Franke & Westerhoff 2012, diagonal inverse-variance form; Künsch 1989
moving-block bootstrap), so heterogeneous moments are commensurable. Excess kurtosis is
tracked as a diagnostic but not optimised (it is outlier-dominated; matching it exactly
is noise-chasing). The KS and Hill terms are paired deliberately: the ablation showed KS
alone lets the tail blow up (kurtosis > 100) while Hill alone leaves the bulk
distribution untargeted — each controls a failure mode the other cannot.

The optimum is found by **two methods, reported together** as a cross-check:

1. **Surrogate-assisted simulated method of moments** (`calibrate.py run`; XGB-Chiarella,
   Gao et al. 2022) — sample the parameter space with a Sobol/Latin-hypercube design,
   fit a single XGBoost regressor θ → D(θ), run a few rounds of greedy active learning to
   focus simulator runs near the optimum, minimise D on the cheap surrogate (pool
   argmin), refine on the true simulator in a tight box (stage-2 grid), and validate on
   fresh seeds. Efficient: ~200 true-simulator evaluations.

2. **Exhaustive grid search** (`calibrate.py grid`; Gao et al. 2023 "Deeper Hedging" /
   Chiarella-Heston §3.3, which calibrates its parameters by grid search minimising
   D(ϑ)) — evaluate the true loss on a regular grid over the regime's parameter box and
   take the minimum. Transparent and exhaustive, which is defensible at this low
   dimension (3–4 parameters), and it yields the full loss surface.

Both are methods from the same research group and should land on the same optimum;
reporting both is a genuine cross-validation. The grid is the easy-to-defend headline,
the surrogate demonstrates efficient convergence.

**Regimes.** Two are used throughout: a **calm** regime (2019) and a **stressed** regime
(the COVID-crash window, late February to early April 2020). The stressed simulation is
capped to the exact length of its ~29-day data window, so the simulated and empirical
moments span the same period and the fundamental is never run past the data.

### Stylised facts reproduced

The calibration targets, and the model's match (Kalman fundamental, baseline locked at D60),
are: a fat-tailed return distribution (calm Hill ≈ 3.0 vs empirical ≈ 2.96 —
the Hill term is what the loss targets, and it matches; excess kurtosis is tracked only as a
diagnostic and runs above empirical, being dominated by a few extremes, so it is not a fit
criterion), near-zero linear return autocorrelation, and short-horizon volatility clustering
(absolute-return autocorrelation, matched at the short lags). Long-horizon clustering
(absolute-return ACF at lags ≥ ~30 minutes) remains a documented structural limit — a
single-timescale momentum cohort cannot produce multi-scale memory (Cont 2005). These
are visualised against the empirical ES tape in `empirical_analysis.ipynb` and the
model-design notebook.

### Calibration robustness & current status

A robustness campaign (eight configurations × both regimes; results in `output/campaign*/` and
`output/e5_confirm/`) tested whether richer agent dynamics improve the fit. The verdict is a
**parsimony result — the baseline is hard to beat.** Adding FT/MT activation probability +
per-order cancellation (E5), a second momentum cohort or more momentum traders (E4), or combining
the changes gave **no reproducible gain**: E5's apparent −34% stressed improvement at screening
resolution did **not** replicate at higher resolution (a poorly-identified four-parameter
addition — high surrogate R² did not guarantee a reproducible optimum). The only consistent,
well-identified improvement was calibrating the momentum half-life `mt_lambda` (a small stressed
gain; optional). The **thesis-final baseline θ is locked (D60)**: the **grid optimum** is the
wired headline (calm 7³ / stressed 5⁴ nodes, 3 seeds — `output/baseline_grid/`), cross-validated
by the high-res surrogate (`output/baseline_hires/`) landing on the same optimum — calm
`ft_sigma_c=0.80, zi_alpha=0.34, zi_delta=0.05` (D_grid 43.8), stressed `ft_sigma_c=0.50,
zi_alpha=0.26, zi_delta=0.05, p_zi=0.15` (D_grid 7.9), now in `globals.CALIBRATED`. Note D is
comparable only at a fixed seed count (the KS component's `s_KS` is sim-sized). Full campaign
numbers, the gaps-vs-shock contagion result, and the clearing-in-loop sensitivity are written up
in **`writing.md`**.

## Repository structure

```
.
├── README.md                  this file — current thesis state
├── AGENT.md                   AI-collaboration rules + full change history + ablation (start at "Status & handoff")
├── writing.md                 thesis-writing context pack (findings, numbers, refs, structure) — pairs with README
├── model_design.ipynb         presentation notebook — market + clearing design, figures, stylised facts
├── build_nb.py                generator for model_design.ipynb (edit here, re-run to rebuild the notebook)
├── data/
│   ├── data.py                DataBento ingest (OHLCV-1m, BBO-1m, MBP-10)
│   ├── roll.py                ES front-month roll + 1-min resample
│   ├── v_kalman.py            Kalman-filtered efficient-price fundamental (primary)
│   ├── v_gbm.py               SV-MJD fundamental (alternative / robustness)
│   ├── p_zi.py                geometric placement-depth fit from MBP-10
│   ├── impact.py              market-impact regression from BBO-1m (Almgren–Chriss η)
│   ├── processed/             rolled 1-minute ES series (calm + stressed)
│   └── fv_{calm,stressed}.csv generated V_t paths (V_smooth + sigma_t)
├── model/
│   ├── globals.py             ModelParams, SimContext, regime dicts, CALIBRATED, CCP consts
│   ├── lob.py                 Order, Fill, LOB
│   ├── agents.py              FT, MT, ZI + BCM/NBCM clearing members
│   ├── clearing.py            BalanceSheet + CentralCounterparty (margin, cover-2 DF, 5-level waterfall, Almgren–Chriss)
│   └── simulation.py          driver + USD variation-margin cycle, novation, waterfall, CCP fire-sale
├── run_simulation.py          entry point + population/clearing-tier builder
├── covid_contagion.py         client-clearing contagion experiment (COVID window + reverse stress)
├── calibrate.py               agent calibration — surrogate-SMM (`run`) + grid (`grid`)
├── analysis_long_run.py       long-horizon 1-min/daily moment validation
├── empirical_analysis.ipynb   empirical ES stylised facts
└── output/                    calibration + simulation outputs
```

## How to run

```bash
# 1. Offline data calibration (one-shot)
python3 data/v_kalman.py generate-all        # Kalman fundamental → data/fv_*.csv (primary)
python3 data/v_gbm.py calibrate              # SV-MJD parameters (alternative)
python3 data/v_gbm.py generate-all 42        # SV-MJD path (only if using the alternative)
python3 data/p_zi.py calibrate               # geometric placement-depth from MBP-10
python3 data/impact.py                        # Almgren–Chriss impact η (fire-sale temp impact is endogenous)

# 2. Run one RTH day of the simulation
python3 run_simulation.py

# 2b. Client-clearing contagion experiment on the real COVID crash window
python3 covid_contagion.py trace             # one cascade run at actual COVID severity
python3 covid_contagion.py reverse           # reverse-stress sweep → cover-2 breach multiplier

# 3. Calibrate the agent parameters — two methods, reported together.
python3 calibrate.py run                      # surrogate-assisted SMM
python3 calibrate.py grid calm 7              # grid search, calm (3-d, 7 points/dim)
python3 calibrate.py grid stressed 6          # grid search, stressed (4-d)
#    both write to output/ (calibrated_params.json / _grid.json); the chosen optimum is wired
#    into globals.CALIBRATED — currently the D60 grid headline (output/baseline_grid/),
#    cross-validated by the high-res surrogate. See writing.md for the calibration campaign.

# 4. Inspect
#    open model_design.ipynb (design + clearing + stylised facts) / empirical_analysis.ipynb
```

## What the key design decisions changed

The model was built market-layer-first, then the clearing tier, then made
regulation-faithful, then stress-realistic. The decisions that shaped it most, and what
each one actually did:

**Doubling the client population, holding the clearing members fixed.** The cleared
population was scaled 2× — to 30 FT + 20 MT + 40 ZI = **90 clients** — while the **15
clearing members were held fixed** and the price-formation mix was **preserved**
(FT-equivalent : MT : ZI = 40 : 20 : 40, exactly the pre-doubling ratio at 2× scale; the
calibration population folds the 10 banking CMs into the 40 FT-equivalents). Why this way:
the market-layer calibration is a function of the agent *mix*, not the head-count, so
doubling at a fixed mix leaves the calibrated parameters valid — **no price-formation re-fit
needed** — while handing the clearing tier a far richer client base. Effect: each
client-carrying CM now clears a **heterogeneous, skewed book of 5–15 clients** (larger CMs
hold more; the non-banking CMs hold the largest, since all their balance-sheet capacity is
client-clearing). That concentration heterogeneity is the lever the contagion study turns —
how hard a client default hits its CM depends on how concentrated that CM's book is — so
doubling-with-preserved-mix buys the study's degrees of freedom essentially for free.

**Adding the clearing tier — what it introduced.** On top of the order book the model gained
a 60-minute **variation-margin cycle** (each cleared book's mark-to-market settles into member
cash in USD), per-cycle **initial margin** and a **capital ratio**, a **cover-2 default fund**,
a **five-level waterfall**, and **Almgren–Chriss fire-sales**. The load-bearing new dynamic is
the **client → CM → CCP cascade**: a client posts its own margin, freezes at the capital floor,
and defaults on cash exhaustion, whereupon its CM assumes and liquidates the position — draining
CM cash and, if the loss is large enough, toppling the CM into the waterfall, whose
mutualisation step and fire-sale price-impact transmit the loss to surviving members. This is
what turns a calibrated *price* model into a *systemic-risk* model.

**Making the margins regulation-faithful — and what it fixed.** The first clearing build used
placeholder constants (flat 20% initial margin, a flat `df%·notional` fund, a
`(1−recovery)·notional` close-out). Replacing them with real methodology changed the behaviour
materially:

- *Procyclical VaR/SPAN initial margin* (≈6% calm → ≈12% stressed) instead of flat 20% — margin
  now rises with volatility, as a real CCP's does.
- *Capital ratio = cash / initial margin* (CFTC Reg 1.17) instead of cash / gross notional. This
  **revived a dead control**: measured against gross client notional, an FCM (levered 8–30×)
  breaches the 8% floor constantly, so the non-banking-CM stop-out never fired; measured against
  *risk margin* it binds only near distress, as intended.
- *Cover-2 Stress-Loss-Over-Initial-Margin default fund* (Euronext A9), plus a **VaR house limit
  on banking-CM proprietary books**, instead of flat `df%·notional`. Together these shrank the
  fund from an inflated **≈$7.7B** (driven by unbounded $30–40B prop books) to a realistic
  **≈$0.5B**.
- *Deficit-based waterfall* (mutualise the defaulter's actual unpaid cash shortfall) instead of a
  40%-of-notional haircut. The haircut had **spuriously bankrupted the whole CCP** (cash → −$79B)
  on a moderate −16% move; the deficit basis makes losses realistic and the cascade credible.

Net effect: the clearing tier now behaves like a sound, regulation-grounded CCP — single
defaults are *contained* by the defaulter's own posted resources, and mutualisation is a genuine
tail event rather than an artefact of placeholder accounting.

**Letting overnight gaps through — and what it changed.** The fundamental originally spliced the
RTH days continuously, removing the overnight gaps, which understated COVID (≈−7% rather than its
true ≈−33% — the largest COVID moves were overnight limit-down opens). Retaining the gaps lets
the cleared book be **marked across them** (overnight gap risk is a primary CCP default driver),
so the clearing tier now sees realistic crisis stress: client defaults rise (≈4–9 vs ≈1 under the
gap-free path) and the reverse-stress **mutualisation onset moves from ≈4× to ≈2.5× COVID**.
An A/B test confirmed the gaps themselves do **not** fatten the intraday tail (same kurtosis with
gaps retained vs re-spliced gap-free) — the gap is handled as a clean between-session **reprice**
of the book and the boundary return is excluded from the calibration moments. **D57** then found
that exclusion had been *misaligned*: it dropped a fixed 390-bar grid while the real sessions are
≈405 bars and vary, so ≈28 overnight-gap returns were leaking into the stressed "intraday" moments
and inflating the simulated stressed kurtosis. Reading the session boundaries from the data and
excluding at the real opens removes the leak (simulated stressed excess kurtosis ≈690 → ≈190); the
residual above-empirical kurtosis is a *pre-existing* property of the current parameters, for the
re-calibration to tune (Hill, not kurtosis, is the tail measure the loss targets).

## What remains

The market and clearing layers are both built. The open items are:

- **Contagion is now exercised (resolved).** The earlier oversized cover-2 fund — caused
  by unbounded BCM prop accumulation inflating member books to tens of $B — is fixed by
  the **VaR house limit**, which shrinks the fund to a realistic $0.3–0.8B in the
  stressed/crash regime (it is looser in calm, where low vol permits larger VaR-limited
  books, but calm is default-free so the fund is untested there); the
  **deficit-based waterfall** and **deficit-consistent client close-out** make losses
  realistic; and the **COVID-window reverse-stress** sweep activates the full waterfall.
  The standing result (standard population, no engineered client; the COVID fundamental
  now retains its overnight gaps, D56, so it carries the true ≈−33% drawdown): at actual
  COVID severity the cover-2 framework *contains* the book — a handful of thin-capital
  clients default but are absorbed by their CMs; cross-member **mutualisation (Level 3–4)
  onsets only at ≈2.5× COVID (≈−41%)**, CCP solvent throughout. The remaining refinement
  is to run this as a **Monte-Carlo ensemble over seeds** (distributions of the breach
  multiplier, default count, waterfall depth, mutualised loss) rather than the single-seed
  traces shown so far.
- **Long-horizon volatility clustering.** The absolute-return ACF is matched at short
  lags but undershoots beyond ~30 minutes; a single-timescale momentum cohort cannot
  produce multi-scale memory (Cont 2005). A multi-horizon momentum cohort is the
  candidate fix.
- **The analysis plan for the research questions.** Default and contagion outcomes are
  studied by **Monte-Carlo ensembles over seeds** (distributions of default counts,
  waterfall depth, and mutualised loss) and by **designed sweeps** over the clearing-tier
  structure — clearing-member count, client-book concentration, margin methodology, and
  tiered-versus-direct clearing — rather than by enlarging the calibrated trading
  population (which would force a recalibration for no analytic gain).

## References

- **Simudyne CCP Risk Model ODD** — [Simudyne docs](https://docs.simudyne.com/commercial_models/ccp#agents-and-structure); **Deloitte**, *Modelling CCP resilience* — [white paper](https://www.deloitte.com/content/dam/assets-zone2/uk/en/docs/services/audit-assurance/2023/deloitte-uk-modelling-ccp-resilience.pdf) (clearing-tier mechanics; the stochastic-volatility motivation came from a Deloitte CCP Risk Model webinar).
- **Margin & default-fund methodology** — **Euronext Clearing**, *Default Fund — module A9* (cover-2 Stress-Loss-Over-Initial-Margin, monthly/daily stress add-ons, the reverse-stress multiplier test); **CME SPAN** and **EMIR** RTS Art. 41 (VaR initial margin, 99% / 2-day MPOR for ETD futures); **CFTC Reg. 1.17** (FCM adjusted-net-capital ≥ 8% of risk margin) and the **CFTC monthly FCM financial data** (member adjusted-net-capital scale); **Basel III FRTB** (trading-book VaR limit for the BCM house book).
- **Majewski, Ciliberti & Bouchaud (2018)** — *Co-existence of trend and value*, extended Chiarella estimation — [arXiv:1807.11751](https://arxiv.org/pdf/1807.11751); [Simudyne calibration](https://docs.simudyne.com/tutorials/extended_chiarella#calibration).
- **Gao et al. (2022)** — *High-frequency financial market simulation* (HFABM): market-maker spec, Hill tail target, surrogate-assisted calibration — [arXiv:2208.13654](https://arxiv.org/pdf/2208.13654). The companion XGB-Chiarella surrogate-SMM / Kalman-fundamental paper — [arXiv:2208.14207](https://arxiv.org/pdf/2208.14207).
- **Gao et al. (2023)** — *Deeper Hedging* / Chiarella-Heston: grid-search calibration of D(ϑ) — [arXiv:2310.18755](https://arxiv.org/pdf/2310.18755).
- **Cont, Stoikov & Talreja (2008)** — stochastic limit-order-book model — [paper](https://www.columbia.edu/~ww2040/orderbook.pdf); **Farmer, Patelli & Zovko (2004)** — zero-intelligence baseline — [arXiv:cond-mat/0309233](https://arxiv.org/pdf/cond-mat/0309233).
- **Vytelingum et al. (2025)** — agent-based liquidity-risk modelling — [arXiv:2505.15296](https://arxiv.org/pdf/2505.15296).
- **Stein & Stein (1991); Heston (1993)** — stochastic volatility; **Merton (1976)** — jump diffusion.
- **Almgren & Chriss (2000)** — optimal execution / market impact (the CM and CCP fire-sale schedule).
- **Lamperti, Roventini & Sani (2018)** — ML-surrogate ABM calibration (surrogate inspiration) — [paper](https://www.sciencedirect.com/science/article/pii/S0165188918301088).
- **Franke & Westerhoff (2012); Künsch (1989)** — block-bootstrap moment weights.
- **Cont (2001)** — stylised facts; **Cont (2005)** — volatility clustering in ABMs; **Hill (1975); Resnick (2007)** — tail-index estimator.
