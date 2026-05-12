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

