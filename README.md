# MScThesis_ABM

Agent-based simulation of a CCP-cleared financial market, built to study systemic risk, margin procyclicality, and contagion dynamics under calm and stressed market regimes.

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
│   ├── market.py       # 
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

The model covers 3 stages, and parameters are calibrated on S\&P E-mini futures data. The three stages are:

 1. Financial Market Simulation

 2. Margin Call Framework

 3. Default Management Framework



--- 

## Financial Market Simulation

Choices to be made:
1. LOB or no?
2. FV signal: calibrated as a parameter or externally fed jump diffusion?
3. Market making agent?
4. Almgren-Chriss execution?

### Agents

Three heterogeneous trader types (+ possible Market Maker):

**Fundamental Trader**


**Momentum Trader**


**Noise Trader**


One Market Engine/CCP:




Two types of clearing members:

**Bank Clearing Member**


**Non-Bank Clearing Member**


### Market Clearing


### Fundamental Value Process

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
