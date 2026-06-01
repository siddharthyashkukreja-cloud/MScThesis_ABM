# MScThesis_ABM

An agent-based model of a centrally-cleared single-asset futures market (the
E-mini S&P 500, ES front-month), built to study CCP systemic risk: clearing-
member default risk, margin procyclicality, and contagion under calm and
stressed market regimes. The market-microstructure layer is built and
calibrated first; the central-clearing and default-management layer is layered
on top once the market dynamics reproduce the empirical stylised facts.

This file documents the part of the model that is built and decided. It is kept
deliberately short — working notes, open questions, and the full deviation log
live in `claudereadme.md`.

## Next steps

- Re-check the agent calibration on the current model. `globals.CALIBRATED`
  still holds pre-D48 placeholders; the validated D48 stressed run lives in
  `output/calibrated_params.json` but has not been copied back, and the calm
  regime has not been run at the current six-parameter layout.
- Close the stressed tail. The dominant calibration residual is the Hill
  tail index (model ≈ 1.1 vs empirical ≈ 3.25): the fundamental-value jump
  layer is over-attributing excess kurtosis to jumps. The candidate fix is to
  let the stochastic-volatility layer absorb part of the kurtosis (cap the
  jump-variance share / lower the jump intensity).
- Long-horizon volatility clustering. The single-factor OU volatility gives
  short-horizon clustering only; the long-lag absolute-return autocorrelation
  is still unreached. A second volatility timescale (two-factor SV or a
  threshold-inertia agent) is the open candidate.
- Build out the clearing layer: USD-denominate the variation-margin cycle,
  make client books live (trade novation), and validate cash depletion under
  stress before the cover-2 default fund and the five-level waterfall.

## Current state

What is implemented and verified:

- A single limit order book with 1-minute call-auction clearing — 390 steps per
  6.5-hour regular-trading-hours day (the ODD-native cadence).
- A 54-agent LOB population — 10 fundamental traders, 10 banking clearing
  members (cast from the fundamental type; they trade their own account
  identically and additionally clear a client book), 10 momentum traders,
  20 zero-intelligence traders, and 4 market makers. Off the order book sit
  5 non-banking clearing members and one central counterparty (the
  star-topology clearing tier of the Simudyne CCP ODD). Five of the ten
  banking clearing members carry client books; the zero-intelligence traders
  remain direct exchange participants.
- An exogenous fundamental value process — a Stein-Stein-style
  stochastic-volatility Merton jump-diffusion — calibrated to real ES futures
  data.
- An offline data-calibration pipeline (the fundamental process, the
  order-placement depth, and a market-impact regression) and an
  agent-parameter calibration pipeline built around an XGBoost surrogate. The
  behavioural calibration is six-dimensional per regime.

The market layer is a four-type agent-based market (fundamental, momentum,
market maker, zero-intelligence). An earlier volatility trader — a
stochastic-volatility construct driven by a separate Heston process (Gao et al.
2023) — was removed: the volatility it traded on was not the volatility of
anything else in the model. Time-varying volatility now enters through the
stochastic-volatility fundamental, and the procyclical volatility relevant to
the research question is expected to emerge endogenously from the clearing
layer.

The central-clearing tier is scaffolded and the 60-tick variation-margin cycle
runs, but the cover-2 default fund, the five-level default waterfall, and
client-trade novation are not yet implemented — they are the planned next layer
(see the end of this file). Client books are currently inert
(`client_positions` at zero), so the clearing-member capital ratios reflect own
positions only.

## Market and agents

The asset is the ES front-month future. Prices move on a 0.25 tick and the
simulation steps every minute. Each LOB trader submits orders directly to a
shared order book; a call auction clears any crossed quotes at the end of every
step.

Order lifetime works in three layers. The fundamental, banking-clearing-member,
and momentum traders use *replace-on-new* management: each holds at most one
standing limit order, and the next time it acts it cancels that order and places
a fresh one, so the standing order always reflects the trader's latest view. The
zero-intelligence traders instead cancel each resting order with a fixed
probability per step. On top of both, a hard ceiling expires any order still
resting after ten steps.

There are four LOB trader types.

**Fundamental trader.** A value strategy (Simudyne CCP ODD §Agents; the
Chiarella–Iori–Perelló family). Each FT holds a fixed idiosyncratic draw `z` and
a reservation price `V_t + z * sigma_fundamental` around the exogenous
fundamental, where `sigma_fundamental` tracks the current stochastic volatility
`sigma_t` of `V_t` — the belief cloud widens in high-volatility regimes and
tightens in calm ones. Every step it places one limit order at the reservation
on the implied side, replacing any order it already had standing. The sign split
across the FT population — some above, some below the fundamental — keeps order
flow two-sided as the fundamental drifts. There is no dead-band: the persistent
`z` already supplies the heterogeneity.

The draw `z` is fixed at agent initialisation, not redrawn each step. This is a
*heterogeneous-beliefs* specification (Simudyne CCP ODD §Agents; Chiarella–Iori–
Perelló): agents disagree about value, and the disagreement persists. The
alternative — drawing a fresh observation-noise term each step — would be a
*noisy-information* specification (Glosten–Milgrom 1985; ABIDES `ValueAgent`)
under which every FT agrees on the value on average. The choice here is
deliberate and downstream-driven: persistent `z` produces a stable cross-section
in which high-`z` FTs accumulate long positions and low-`z` FTs accumulate short
positions, building up concentrated inventories over time. That concentration is
the load-bearing input to the clearing layer added later — margin calls, the
default fund, and the waterfall only have something to act on when individual
clearing-member balance sheets are skewed by client positions.

**Momentum trader.** A trend-following strategy. Each MT tracks an
exponentially weighted moving average of mid log-returns and, on the side of the
trend, places a passive limit order at a placement depth drawn from the shared
log-normal distribution. The MT is limit-only — trend amplification comes from
inside-the-spread MT limits being filled by market orders that move with the
trend. All momentum traders share a single EWMA decay (`mt_lambda`), pinned at
0.05: Majewski et al. (2018) likewise fix the trend horizon externally (they set
it from a CTA-index correlation rather than estimating it), so treating it as a
fixed design constant has a literature home.

**Market maker.** Liquidity provision (Gao et al. 2022, the high-frequency ABM,
§3.6, single-quote mid-anchored variant). Each step the MM cancels its previous
quotes and reposts one bid and one ask, each at a uniformly random tick offset
from the previous-step mid, with no inventory skew, always quoting. Its size
(`mm_qty = 2`) is structural; the spread width (`mm_p_edge`) is calibrated. The
continuous near-mid liquidity stops the book from walking in one go between
fundamental jumps, which controls return kurtosis. The percent-of-volume and
Almgren-Chriss liquidation helpers in the class are retained for the deferred
clearing-member fire-sale; the live MM does not use them. (Gao et al.'s MM also
carries an inventory-limit "hot-potato" regime switch central to their
flash-crash study; this thesis uses the always-quote variant and does not model
that switch.)

**Zero-intelligence trader.** A Cont, Stoikov & Talreja (2008) / Farmer, Patelli
& Zovko (2004) noise trader providing a background liquidity floor. Each step it
independently may cancel a resting order (probability `zi_delta`), submit a limit
order at a random side and depth (probability `zi_alpha`), or submit a small
market order (probability `zi_mu`) — with no signal dependence. The three rates
are calibrated. Its role is the steady background flow that absorbs aggressive
market orders and keeps the book from gapping.

## Fundamental value process

The fundamental value `V_t` is exogenous to the simulation: a path generated
offline and read in as a time series. It is a Stein-Stein-style
stochastic-volatility Merton jump-diffusion — a geometric Brownian motion whose
instantaneous volatility `sigma_t` mean-reverts as a Vasicek / Ornstein-
Uhlenbeck process, overlaid with Poisson jumps. The persistent `sigma_t` induces
the short-horizon volatility clustering that the earlier constant-volatility
specification lacked.

Calibration is layered (`data/v_gbm.py`). The diffusion and jump parameters come
from the variance and excess kurtosis of ES 1-minute log-returns (overnight
returns excluded); the OU volatility parameters from a 30-minute realised-
volatility series; the long-run volatility level is then re-anchored so the
stochastic-vol layer adds clustering without inflating the total return variance.

The stochastic-volatility architecture follows a Deloitte CCP Risk Model webinar
treatment (the published ODD is incomplete on the fundamental process); the
formal Ornstein-Uhlenbeck-on-volatility model is Stein-Stein (1991). The jump
intensity `lambda = 3/day` is the ODD §Calibration value; the jump size is
data-calibrated. Piping `sigma_t` into the fundamental trader's belief width is
a modelling choice of this thesis.

## Formulas

Time is discrete; the step length is one minute (the ODD-native cadence). Agent
arrival rates are per-step Bernoulli probabilities: each step an agent acts, or
does not, with its rate as the probability.

Fundamental value, stochastic-volatility Merton jump-diffusion:

```
sigma_{t+1}      = theta + e^{-alpha}(sigma_t - theta) + s * Z_sigma   (Vasicek-OU)
log V_{t+1} - log V_t = (mu_v - 0.5 * sigma_t^2) + sigma_t * Z + J
J = jump:  with prob lambda/390 a draw from N(0, delta^2), else 0
```

Fundamental trader, per step (each FT holds a fixed `z ~ N(0, 1)`):

```
sigma_fundamental_t = ft_sigma_c * sigma_t * v0      (ft_sigma_c = sqrt(390))
reservation       R = V_t + z * sigma_fundamental_t
side = sign(R - mid)                                  (no dead-band)
acts every step (ft_alpha = 1.0): cancel the standing limit, place one fresh
limit at R, qty ~ U[qty_min, qty_max]
```

Momentum trader, per step:

```
M_t  = (1 - mt_lambda) * M_{t-1} + mt_lambda * (log mid_{t-1} - log mid_{t-2})
|M_t| < mt_eps  -> no trade
side = sign(M_t)
acts every step (mt_alpha = 1.0): cancel the standing limit, place one fresh
limit at  mid - side * k * tick_size,  k ~ shared log-normal placement depth
```

Market maker, per step:

```
anchor = previous-step mid (v0 at t = 0)
bid = anchor - d_bid * tick_size,   ask = anchor + d_ask * tick_size
d_bid, d_ask ~ Uniform{0, ..., mm_p_edge}             (no inventory skew)
quote size on each side = mm_qty;  always quotes
```

Zero-intelligence trader, per step:

```
each resting limit cancelled w.p. zi_delta
submit a limit order  w.p. zi_alpha  at random side, depth k ~ shared log-normal
submit a market order w.p. zi_mu     at random side
size ~ Uniform{qty_min, ..., qty_max}
```

Shared log-normal placement depth (ZI and MT): `k ~ LogNormal(mu, depth_sigma)`
with `mu = ln(depth_mean) - depth_sigma^2 / 2`, rounded and floored at 1 tick.

Order book: `mid = (best_bid + best_ask) / 2`; if one side is empty, `mid` holds
at the last trade price.

## Parameters

Grouped by where each parameter is used. The right column gives how it is set: a
structural constant, calibrated offline from data, or calibrated in the
agent-parameter loop (`calibrate.py`).

Market and order book:

| Parameter | Meaning | Set by |
|---|---|---|
| `tick_size` | price increment (0.25) | structural |
| `dt_minutes` | step length in minutes (1.0) | structural |
| `order_ttl` | hard order-lifetime ceiling, steps (10) | structural |
| `qty_min`, `qty_max` | order size range (1; 10) | structural |
| `n_fundamental`, `n_momentum`, `n_mm`, `n_zi` | LOB trader counts (10+10 FT/BCM, 10, 4, 20) | structural |

Fundamental value `V_t` (stochastic-volatility Merton jump-diffusion):

| Parameter | Meaning | Set by |
|---|---|---|
| `mu_v`, `sigma_v`, `v0` | per-step drift, total 1-min vol, initial level | data, per regime (`v_gbm.py`) |
| `sigma_d`, `lambda`, `delta` | diffusion vol, jump intensity, jump size | data; `lambda = 3/day` (ODD) |
| `alpha`, `theta`, `sigma_vol` | OU mean-reversion speed, level, vol-of-vol | data (30-min realised vol) |

Fundamental trader:

| Parameter | Meaning | Set by |
|---|---|---|
| `ft_alpha` | activation probability per step (1.0 — trades every step) | structural (pinned) |
| `ft_sigma_c` | reservation-dispersion scale (`sqrt(390)`) | structural |

Momentum trader:

| Parameter | Meaning | Set by |
|---|---|---|
| `mt_alpha` | activation probability per step (1.0 — trades every step) | structural (pinned) |
| `mt_lambda` | EWMA decay of the momentum signal (0.05) | structural (pinned) |
| `mt_mu` | market-order branch (0.0 — limit-only) | structural (pinned) |

Market maker:

| Parameter | Meaning | Set by |
|---|---|---|
| `mm_qty` | size per quote (2) | structural |
| `mm_p_edge` | maximum tick offset of a quote from the mid | agent loop, per regime |

Zero-intelligence trader and placement depth:

| Parameter | Meaning | Set by |
|---|---|---|
| `zi_alpha`, `zi_mu`, `zi_delta` | limit- / market-order / cancellation probabilities | agent loop, per regime |
| `depth_mean`, `depth_sigma` | shared log-normal placement-depth mean and shape | agent loop, per regime |

The six calibrated values per regime are `depth_mean`, `depth_sigma`,
`zi_alpha`, `zi_mu`, `zi_delta`, and `mm_p_edge` — calibrated as two independent
problems with no parameter shared between calm and stressed. The values are
stored in `model/globals.py` (the `CALIBRATED` dict) and read by both
`run_simulation.py` and the analysis notebooks.

## Calibration

Calibration runs in two stages.

The data-side parameters are calibrated directly and offline. The fundamental
process is calibrated from the ES 1-minute series (`data/v_gbm.py`). The
geometric placement-depth parameter is reported as a book diagnostic from the L2
order-book data (`data/p_zi.py`), though live placement now uses the calibrated
log-normal. A market-impact regression on the 1-minute best-bid-offer data
(`data/impact.py`) produces temporary and permanent impact coefficients, kept
for a later optimal-execution extension but not currently wired into any agent.
These are all fixed per regime before any agent calibration.

The six behavioural parameters are calibrated by matching simulated 1-minute
return moments to the empirical ES moments (`calibrate.py`). The target battery
is the Cont (2001) stylised facts — return standard deviation and the
autocorrelation of returns and of absolute returns at several lags — plus the
Hill (1975) tail index as a robust heavy-tails measure. The loss follows Gao et
al. (2022): a four-component grouped-moment distance (Hill, return std, return
ACF, absolute-return ACF), with each moment standardised by its empirical
block-bootstrap sampling standard deviation (Franke & Westerhoff 2012, diagonal
inverse-variance form). The pipeline is surrogate-assisted simulated method of
moments: the simulator is sampled across the parameter space with a Latin
hypercube, an XGBoost regressor is fit per moment, a few rounds of greedy active
learning focus extra simulator runs on the most promising regions, the loss is
minimised on the cheap surrogate, and that optimum is refined by a grid search
on the true simulator and validated on fresh seeds.

Two market regimes are used throughout: a calm regime (2019) and a stressed
regime (the COVID-crash window, late February to early April 2020).

## Repository structure

```
MScThesis_ABM/
├── data/
│   ├── data.py            DataBento ingest (OHLCV, BBO, MBP-10)
│   ├── roll.py            ES front-month roll and 1-minute resample
│   ├── v_gbm.py           SV-MJD calibration and V_t path generation
│   ├── v_kalman.py        Kalman-filter / Roll-1984 noise diagnostic
│   ├── p_zi.py            placement-depth diagnostic from MBP-10
│   ├── impact.py          market-impact regression from BBO-1m
│   ├── processed/         rolled front-month 1-minute series
│   └── fv_*.csv           generated V_t paths (V_smooth + sigma_t), per regime
├── model/
│   ├── globals.py         ModelParams, SimContext, regime dicts, CALIBRATED, CCP consts
│   ├── lob.py             order book
│   ├── agents.py          the four trader types + BCM/NBCM clearing members
│   ├── clearing.py        BalanceSheet and CentralCounterparty
│   └── simulation.py      simulation driver and variation-margin cycle
├── run_simulation.py      entry point and population builder
├── calibrate.py           agent-parameter calibration (XGBoost surrogate)
├── analysis.ipynb         post-run market-layer analysis
├── clearing_analysis.ipynb  clearing-tier analysis (capital ratios, margin calls)
├── AGENT.md               AI-agent collaboration briefing
└── output/
```

## How to run

```
# 1. Offline data calibration (one-shot)
python data/v_gbm.py calibrate
python data/v_gbm.py generate-all 42
python data/p_zi.py calibrate
python data/impact.py

# 2. Run the simulation
python run_simulation.py

# 3. Calibrate the agent parameters, then copy theta_stage2 into globals.CALIBRATED
python calibrate.py run

# 4. Inspect
#    open analysis.ipynb and clearing_analysis.ipynb and run all cells
```

## Planned extension: the CCP layer

Once the market dynamics are calibrated, the clearing layer is completed: the
banking and non-banking clearing members and the client-clearing tier already
exist in `agents.py` and `clearing.py` and the variation-margin cycle runs, but
initial-and-variation-margin USD denomination, client-trade novation, a cover-2
default fund, and a five-level default waterfall with position auctioning remain.
The research questions the full model is meant to address — whether tiered
client clearing changes the frequency and severity of CCP stress, whether banks
that both trade proprietarily and clear clients face higher default risk, whether
transient price impact in liquidations amplifies contagion, and how margin
methodology interacts with client clearing — depend on that layer. The
margin-call to fire-sale to price-impact feedback is also where time-varying,
procyclical volatility is expected to emerge endogenously.

## References

- Simudyne CCP Risk Model ODD; Deloitte CCP Risk Model webinar / CCP resilience
  white paper (clearing-tier mechanics and the stochastic-volatility motivation).
- Stein & Stein (1991); Heston (1993) — stochastic volatility (the
  Ornstein-Uhlenbeck-on-volatility fundamental). Merton (1976) — jump diffusion.
- Majewski, Ciliberti & Bouchaud (2018). Co-existence of trend and value in
  financial markets.
- Cont, Stoikov & Talreja (2008); Farmer, Patelli & Zovko (2004) — limit order
  book and zero-intelligence baselines.
- Gao et al. (2022). High-frequency financial market simulation and flash-crash
  scenarios analysis (market-maker spec, loss function, surrogate-assisted
  calibration). Gao et al. (2023) — Chiarella-Heston / deep hedging (the
  volatility-trader channel, now removed).
- Vytelingum et al. (2025). Agent-based liquidity-risk modelling for financial
  markets.
- Almgren & Chriss (2000); Almgren, Thum, Hauptmann & Li (2005) — optimal
  execution and percent-of-volume impact.
- Lamperti, Roventini & Sani (2018). Agent-based model calibration using machine
  learning surrogates (surrogate-assisted calibration; inspiration).
- Franke & Westerhoff (2012); Künsch (1989) — block-bootstrap moment weights.
- Mike & Farmer (2008); Bouchaud, Mézard & Potters (2002) — empirical order
  placement and book depth.
- Lee & Mykland (2008) — jump detection. Cont (2001) — stylised facts. Cont
  (2005) — volatility clustering and agent-based models. Hill (1975); Resnick
  (2007) — tail-index estimator.
```
