# MScThesis_ABM

Agent-based simulation of a CCP-cleared financial market, built to study systemic risk, margin procyclicality, and contagion dynamics under calm and stressed market regimes.

---

## Next Steps

- Redo order scaling logic, currently parameters zi_qty_min and zi_qty_min uniform [1,10] but instead of 1 min steps we have 5 min steps
- Depth of limit order for noise trader is geometric offset(discrete time exponential), perhaps dosent need maximum but does need to be calibrated from data
- ft_sigma_rel for fundamental sigma offset is currently calibrated, shoudl just be sigma of fundamental
- Exact volatility scaling process from Gao, perhaps for later?
- Intraday data for calibration? - L2 book for noise trader limit offset, calibrating Cont-Stoikov; for FV signal drift and diffusion and jumps 
- FT, MT and MM order cancellation after 10 ticks (maybe 3 as we have 10 min), if new order sent last order cancelled, also overrules zi_delta so maybe just remove that 
- CHange limit price for MT and FT to offset by sigma_fundamental or sigma_momentum
- Change market maker to match Krishnen paper, and change claude documentation to amke it different trader type
- Fix parameters to fix, calibrate on empirical data, or calibrate using surrogate modelling 

---

## Questions for Krishnen

- 


---

## Overview

This project implements a heterogeneous agent-based model (HABM), calibrated to real market data. The simulation reproduces stylised facts of financial markets — fat-tailed returns, volatility clustering, autocorrelation structure — and overlays a CCP margin and default framework to study how clearing mechanics interact with price dynamics under stress. The model can be thought of as an extension to the Deloitte-Simudyne CCP Risk Model with client tiers and a richer market environment.

---

## Repository Structure

```
MScThesis_ABM/
├── data/
│   ├── thesis_data_calm.csv        # Real market data – calm regime (2018-2019)
│   └── thesis_data_stressed.csv    # Real market data – stressed regime (2008-2009)
│
├── model/
│   ├── globals.py      # ModelParams dataclass; GlobalState (fundamental value process)
│   ├── agents.py       # BaseTrader + FundamentalTrader, MomentumTrader, Noise Vol. Trader (+ maybe market maker)
│   ├── lob.py       # 
│   ├── margin.py       # 
│   └── simulation.py   # 
│
├── calibrate.py        #   
├── run_simulation.py   # 
├── analysis.ipynb      # Exploratory analysis and plots
│
└── output/

```

---

## How to Run

```bash
# 1. Calibrate KF+EM on real data
python calibrate.py

# 2. Run Monte Carlo simulation
python run_simulation.py

# 3. Open notebook for analysis
jupyter notebook analysis.ipynb
```

Step 2 reads from `output/calibrated_params.csv`. If calibration has already been run, step 1 can be skipped.


---

## Model Description

The model covers 3 stages, and parameters are calibrated on SPY data. The three stages are:

 1. Financial Market Simulation

 2. Margin Call Framework

 3. Default Management Framework

---

### Questions for Krishnen

1. Model calibrated parameters twice - once for calm and once for stressed?
2. 
--- 

## Financial Market Simulation


### Agents

The model includes trading agents interacting through a limit order book, plus a clearing tier or banks and non-bank clearing members. Bank Clearing Members are also of one of these trader types, and may clear on behalf of other client traders. Non-Bank CMs only clear on behalf of client traders. 

#### Four Trader Types: 


**Zero Intelligence (ZI)/Noise Trader with Vol. Scaling (Count: )**

Provide background liquidity and order flow, proxying for other trader types such as retail and corporate traders, and traders with hedging demands.

- **Events per step (length `dt_minutes`):**
  - Limit orders: `N_limit ~ Poisson(zi_alpha * dt_minutes)`
  - Market orders: `N_market ~ Poisson(zi_mu * dt_minutes)`
  - Cancellation: each resting order is cancelled with prob `p_cancel = 1 - exp(-zi_delta * dt_minutes)`

- **Side and size:**
  - Side is 50/50 buy vs sell.
  - Size: `q ~ Uniform{zi_qty_min, ..., zi_qty_max}`, optionally scaled up when current variance `v_var` exceeds target `theta_v`.

- **Limit price placement:**
  - Depth in ticks: `k = min(Geometric(zi_offset_p), zi_offset_max)`
  - Buy limits: anchored off `best_ask` (or `V_t` if book empty) at  
    `price = anchor - k * tick_size`
  - Sell limits: anchored off `best_bid` (or `V_t`) at  
    `price = anchor + k * tick_size`
  - Prices are floored at `tick_size` and rounded to the tick grid.

ZI traders carry inventory and cash like other agents but have no strategic beliefs or balance‑sheet constraints.

**Fundamental Trader (Count: )**

Fundamental traders implement a value strategy with heterogeneous private valuations, following the extended Chiarella model and the Simudyne CCP ODD.

**Private valuation**

- Each FT draws a fixed `z_i ~ Normal(0, 1)` at initialisation.
- With fundamental value `V_t` and relative dispersion `ft_sigma_rel`, the reservation price is

  `r_i,t = V_t * (1 + z_i * ft_sigma_rel)`

**Direction rule**

- Reference price is the midprice if available, otherwise `V_t`.
- If `r_i,t > ref_price`: submit a **buy** limit at `r_i,t`.
- If `r_i,t < ref_price`: submit a **sell** limit at `r_i,t`.
- If equal: no order.

**Order size**

- Size is independent of mispricing:

  `q ~ Uniform{zi_qty_min, ..., zi_qty_max}`

**Notes**

- Offsets are specified as a fraction of `V_t` (basis‑point style), so FT behaviour is invariant to the price level across calm vs stressed regimes.
- There is no `kappa * (V_t - P_t)` demand‑magnitude term at this stage; that can be added later if needed for calibration, as in Majewski et al. (extended Chiarella).

**Momentum Trader (Count: )**

### Momentum Trader (MT)

### Momentum Trader (MT)

Momentum traders follow an EWMA trend signal on log returns and place limit orders
relative to the current midprice, with no dependence on fundamental value.

**Momentum signal**

- Global EWMA on log‑returns, updated each step:

  `M_t = mt_lambda_ewma * M_{t-1} + (1 - mt_lambda_ewma) * (log P_t - log P_{t-1})`

- `mt_lambda_ewma` controls memory length; `M_0 = 0`.

**Activation and direction**

- MT acts only when `abs(M_t) >= mt_threshold`.
- If `M_t > 0`: submit a **buy** limit.
- If `M_t < 0`: submit a **sell** limit.
- If `abs(M_t) < mt_threshold`: no order.

**Limit price placement**

- Each MT draws a fixed `z_i ~ Normal(0, 1)` at initialisation.
- Reference price is the current midprice if available, otherwise last traded price.
- Buy limit:

  `price = mid_t + z_i * mt_sigma_abs`

- Sell limit:

  `price = mid_t - z_i * mt_sigma_abs`

- `mt_sigma_abs` is an absolute offset in price units (e.g. dollars or ticks).
- Price is floored at `tick_size` and rounded to the tick grid.

**Order size**

- Size is independent of signal magnitude:

  `q ~ Uniform{zi_qty_min, ..., zi_qty_max}`

**Notes**

- Placing relative to `mid_t` keeps MTs as market‑reactive agents with no fundamental belief, in contrast to FTs. `mt_sigma_abs` controls how aggressively they cross the spread.[cite:203]
- Signal‑scaled aggressiveness via `tanh(mt_kappa * M_t)` is a natural extension if urgency effects are needed later.

**Market Marker (Count: )**


#### One LOB/Market Engine/CCP:




#### Two types of clearing members:

**Bank Clearing Member (Count: )**


**Non-Bank Clearing Member (Count: )**



### Fundamental Value Process


### Parameters and Calibration

**ZI/Noise/Vol. Trader**

zi_alpha (α): limit order arrival rate
zi_mu (μ): market order arrival rate 
zi_delta (δ): order cancellation rate (can maybe be deleted if orders removed every 3 steps regardless)

Limit orders placed at mid price + exponential(geometric for discrete) depth 

All can be estimated from L2 order book data

Order size range: scaled with volatality signal (Gao Deep Hedging) which follows Heston process interacted with GBM for Wts with is correlated with variance with rho - price and volatility have negative correlation 

**Heston Volatility Process**



--- 

## Margin Call 

### Initial Margin

### Variation Margin

### Default Fund


---

## Default Management Framework 

--- 

## Calibration


---

## Step Sequence


---

## Staged Development Plan

The model is being built incrementally to isolate bugs at each layer before adding complexity:

| Stage | What is added | Validation target |
|---|---|---|
| 1 | Baseline market + ZI noise only | Price discovery, return distribution |
| 2 | + Fundamental and momentum traders | Stylised facts: fat tails, vol clustering, ACF |
| 3 | + Balance sheets and IM/VM margin (current) | Cash depletion dynamics, margin call frequency |
| 4 | + Default fund | Default fund adequacy under stress |
| 5 | + Full CCP waterfall + fire-sale contagion | Cascade absorption, margin spiral |
| 6 | + Almgren-Chriss distressed liquidation | Price impact from forced sales |
| 7 | Stressed data | Stressed default probability, procyclicality |

---

## Future/Possible Extensions

The following are on the roadmap but not yet implemented:

- **Volatility traders** — Heston-style demand proportional to $\nu_t$ (stochastic variance), as in Gao et al. (2023). Demand: $\Delta \log P \propto \zeta \cdot \nu_t$, where $\nu_t$ follows a CIR mean-reverting process.
- **Almgren-Chriss liquidation** — distressed agents (capital ratio ≤ 8%) slice positions optimally, generating measurable temporary price impact haircuts.
- **Basel III deleveraging** — bank CMs deleverage when capital ratio approaches 8% floor; non-bank CMs freeze order submission.
- **Client clearing delay** — non-bank CMs face a one-step delay in passing margin calls to clients, creating transient funding gaps.
- **Strategy switching** — agents switch between fundamental, momentum, and noise strategies based on recent performance, following the reinforcement learning framework in Gao et al. (2023).
- **Cover-2 default fund** — DF sized to cover the two largest member defaults (EMIR Art. 42); contributions proportional to exposure; recalculated daily.
- **LOB / market maker** — optional limit order book with a market-making agent to provide liquidity between call auctions.

---

## Research Questions/Hypotheses

1. Is Cover-2 sufficient to absorb a stressed default scenario calibrated to 2008-2009 data?
2. Do bank CMs that clear for clients face higher default probability than proprietary traders under margin spirals?
3. How does trade concentration (crowded positions) amplify contagion through the CCP waterfall?
4. What is the conditional probability of a second default given a first?
5. Does CoMargin (correlated margin) reduce procyclicality relative to individual SPAN-style IM?
6. How does CCP fire-sale price impact differ between calm and stressed regimes?

---

## References

- Majewski, A., Ciliberti, S., & Bouchaud, J.-P. (2018). *Co-existence of Trend and Value in Financial Markets: Estimating an Extended Chiarella Model.* arXiv:1807.11751.
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk.
- Gao, X. et al. (2023). *Deeper Hedging: A New Agent-based Model for Effective Deep Hedging.* arXiv:2310.18755.
- Bookstaber, R., Paddrik, M. & Tivnan, B. (2014). *An Agent-based Model for Financial Vulnerability.* OFR Working Paper 2014-05.
- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading.* Econometrica, 53(6), 1315–1335.
- CPMI-IOSCO (2012). *Principles for Financial Market Infrastructures.*
- ESMA (2016). *RTS 2016/2251 — Margin Requirements for Non-Centrally Cleared OTC Derivatives.*
