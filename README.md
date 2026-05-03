# MScThesis_ABM

Agent-based simulation of a CCP-cleared financial market, built to study systemic risk, margin procyclicality, and contagion dynamics under calm and stressed market regimes.

---

## Overview

This project implements an extended Chiarella-Iori heterogeneous agent-based model (HABM), calibrated to real market data via Kalman Filter + Expectation-Maximisation (KF-EM). The simulation reproduces stylised facts of financial markets — fat-tailed returns, volatility clustering, autocorrelation structure — and overlays a CCP margin and default framework to study how clearing mechanics interact with price dynamics under stress.

The model sits at the intersection of four strands of literature:

- **Heterogeneous agent price formation** — Majewski, Ciliberti & Bouchaud (2018): fundamental value as a hidden variable, estimated by Bayesian filtering; price dynamics governed by linear price impact interacting with heterogeneous demand.
- **Optimal execution and price impact** — Almgren & Chriss (2000): permanent and temporary price impact decomposition; used for distressed liquidation.
- **CCP systemic risk** — Bookstaber, Paddrik & Tivnan (2014) and the Deloitte/Simudyne CCP Risk Model: Cover-2 default fund, margin spiral mechanics, fire-sale contagion.
- **Deep hedging and synthetic markets** — Gao et al. (2023): volatility traders with Heston-style demand; ABM as a synthetic data engine for strategy evaluation.

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
│   ├── agents.py       # BaseTrader + FundamentalTrader, MomentumTrader, NoiseTrader
│   ├── market.py       # Market clearing: log-price mode (KF-consistent) and volume mode
│   ├── margin.py       # Vectorised CCP margin engine (IM, VM, default, force-close)
│   └── simulation.py   # Step loop: agents → clearing → VWAP accounting → margin
│
├── calibrate.py        # KF-EM calibration → output/calibrated_params.csv
├── run_simulation.py   # Monte Carlo runner → output/sim_{regime}_*.csv
├── analysis.ipynb      # Exploratory analysis and plots
│
└── output/
    ├── calibrated_params.csv           # Estimated parameters per regime
    ├── fundamental_value_{regime}.csv  # Filtered/smoothed fundamental value path
    ├── sim_{regime}_runs.csv           # Per-step market data, all Monte Carlo runs
    ├── sim_{regime}_trades.csv         # Per-agent trade log
    ├── sim_{regime}_margin.csv         # Per-step margin/balance-sheet aggregates
    └── sim_{regime}_median.csv         # Median + IQR summary across runs
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

### CLI Options — `calibrate.py`

```
--calm        Path to calm regime CSV       (default: data/thesis_data_calm.csv)
--stressed    Path to stressed regime CSV   (default: data/thesis_data_stressed.csv)
--regime      Which regime to run           (default: both | calm | stressed)
--out         Output path for params CSV    (default: output/calibrated_params.csv)
--max_iter    Max EM iterations             (default: 200)
```

---

## Model Description

### Agents

Three heterogeneous trader types, all operating in log-price mode (see Calibration section):

**Fundamental Trader**
Observes the latent fundamental value $V_t$ and trades when log-mispricing $\log(V_t/P_t)$ exceeds a dead-band $\delta$. Demand contribution:

$$\Delta \log P \propto \gamma \cdot \log(V_t / P_t)$$

where $\gamma$ is estimated by KF-EM. The dead-band $\delta = 0.5 \cdot \sigma_n$ (half the noise standard deviation) prevents overreaction to noise-driven price fluctuations.

**Momentum Trader**
Tracks an EWMA momentum signal $M_t$ of past log-returns, with decay rate $\alpha = 1/(1+5)$ (one-week horizon, fixed per Majewski et al.). Demand saturates for large signals via:

$$\Delta \log P \propto \beta \cdot \tanh(M_t)$$

The tanh saturation reflects risk aversion at large momentum values.

**Noise Trader**
Participates with probability $p_{\text{noise}}$ each step. Demand drawn from:

$$\Delta \log P \sim \mathcal{N}(0, \sigma_n^2)$$

where $\sigma_n$ is estimated from data. In log-price mode, noise traders represent the residual unexplained return variation — the KF observation noise.

### Market Clearing

In log-price mode (default), agent decisions aggregate directly into a net log-price increment:

$$\log P_{t+1} = \log P_t + \sum_i s_i \cdot m_i$$

where $s_i \in \{-1, 0, +1\}$ is agent $i$'s side and $m_i$ is their magnitude. This is the discrete-time equivalent of Kyle's (1985) linear price impact with $\lambda = 1$ normalised into agent magnitudes — consistent with the KF state equation used in calibration.

Momentum is updated as an EWMA of the net log-return:

$$M_{t+1} = \alpha \cdot \Delta \log P_t + (1-\alpha) \cdot M_t$$

Volatility is tracked via an EWMA of squared log-returns with a 20-day window ($\alpha_{\text{vol}} = 1/21$).

### Fundamental Value Process

$$V_{t+1} = V_t \cdot \exp(\mu_v + \sigma_v \cdot \varepsilon_t), \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

Parameters $\mu_v$ (log-drift $g$) and $\sigma_v$ are estimated by EM. $V_t$ is latent — never observed by agents directly; they observe only $P_t$.

### Balance Sheet and Margin

Each agent holds a cash account initialised at `initial_wealth = 1000` (normalised units). Positions are tracked in log-increment units with VWAP entry prices.

The `MarginEngine` runs every step, fully vectorised over all agents:

| Quantity | Formula | Regulatory basis |
|---|---|---|
| Notional | \|inventory\| · P_t | — |
| Initial Margin (IM) | 10% of notional | EMIR RTS 2016/2251, equity derivative floor |
| Variation Margin (VM) | max(-MtM PnL, 0) | EMIR Art. 41, daily cash VM settlement |
| Default trigger | equity < 0 after VM cover attempt | CPMI-IOSCO 2012, Principle 12 |
| Force-close | position closed at P_t; collateral released | CCP defaulter-pays-first waterfall |

---

## Calibration

Calibration follows Majewski, Ciliberti & Bouchaud (2018). The latent fundamental value $V_t$ is treated as a hidden variable in a linear state-space model:

**Transition:** $\quad V_{t+1} = V_t + g + \sigma_v \varepsilon_t$

**Observation:** $\quad P_{t+1} = (1-\gamma) P_t + \gamma V_t + \beta u_t + \sigma_n \eta_t$

where $u_t = \tanh(M_t)$ is the observable momentum control. Parameters estimated by EM:

| Parameter | Meaning | Estimated? |
|---|---|---|
| $\gamma$ | Fundamental pull on price | ✓ EM |
| $\beta$ | Momentum impact on price | ✓ EM |
| $\alpha$ | EWMA decay (momentum horizon) | Fixed: 1/6 (5-day horizon) |
| $g$ | Fundamental drift | ✓ EM |
| $\sigma_v$ | Fundamental volatility | ✓ EM |
| $\sigma_n$ | Observation (noise) volatility | ✓ EM |
| $V_0$ | Initial fundamental value | ✓ EM (smoothed) |
| $\lambda$ | Kyle's lambda | Not estimated (price-only data) |

The E-step runs a Kalman Filter (forward pass) followed by a Rauch-Tung-Striebel smoother (backward pass). The M-step solves a weighted OLS system in closed form, correcting for the uncertainty in $V_t$ through the posterior variance $P_t$.

**Log-price mode**: simulation agents operate directly in log-price increments, so calibrated parameters $(\gamma, \beta, \sigma_v, \sigma_n)$ slot in without rescaling. Kyle's $\lambda$ is left as `NaN` in `calibrated_params.csv` because it cannot be identified from price-only data; it is unused in log-price mode.

---

## Staged Development Plan

The model is being built incrementally to isolate bugs at each layer before adding complexity:

| Stage | What is added | Validation target |
|---|---|---|
| 1 | Baseline market + ZI noise only | Price discovery, return distribution |
| 2 | + Fundamental and momentum traders (KF-EM calibrated) | Stylised facts: fat tails, vol clustering, ACF |
| 3 | + Balance sheets and IM/VM margin (current) | Cash depletion dynamics, margin call frequency |
| 4 | + Default fund (Cover-2) | Default fund adequacy under stress |
| 5 | + Full CCP waterfall + fire-sale contagion | Cascade absorption, margin spiral |
| 6 | + Almgren-Chriss distressed liquidation | Price impact from forced sales |
| 7 | Stressed data | Stressed default probability, procyclicality |

---

## Planned Extensions

The following are on the roadmap but not yet implemented:

- **Volatility traders** — Heston-style demand proportional to $\nu_t$ (stochastic variance), as in Gao et al. (2023). Demand: $\Delta \log P \propto \zeta \cdot \nu_t$, where $\nu_t$ follows a CIR mean-reverting process.
- **Almgren-Chriss liquidation** — distressed agents (capital ratio ≤ 8%) slice positions optimally, generating measurable temporary price impact haircuts.
- **Basel III deleveraging** — bank CMs deleverage when capital ratio approaches 8% floor; non-bank CMs freeze order submission.
- **Client clearing delay** — non-bank CMs face a one-step delay in passing margin calls to clients, creating transient funding gaps.
- **Strategy switching** — agents switch between fundamental, momentum, and noise strategies based on recent performance, following the reinforcement learning framework in Gao et al. (2023).
- **Cover-2 default fund** — DF sized to cover the two largest member defaults (EMIR Art. 42); contributions proportional to exposure; recalculated daily.
- **LOB / market maker** — optional limit order book with a market-making agent to provide liquidity between call auctions.

---

## Research Questions

1. Is Cover-2 sufficient to absorb a stressed default scenario calibrated to 2008-2009 data?
2. Do bank CMs that clear for clients face higher default probability than proprietary traders under margin spirals?
3. How does trade concentration (crowded positions) amplify contagion through the CCP waterfall?
4. What is the conditional probability of a second default given a first?
5. Does CoMargin (correlated margin) reduce procyclicality relative to individual SPAN-style IM?
6. How does CCP fire-sale price impact differ between calm and stressed regimes?

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
jupyter
```

No additional build steps required. All simulation code is pure Python.

---

## References

- Majewski, A., Ciliberti, S., & Bouchaud, J.-P. (2018). *Co-existence of Trend and Value in Financial Markets: Estimating an Extended Chiarella Model.* arXiv:1807.11751.
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk.
- Gao, X. et al. (2023). *Deeper Hedging: A New Agent-based Model for Effective Deep Hedging.* arXiv:2310.18755.
- Bookstaber, R., Paddrik, M. & Tivnan, B. (2014). *An Agent-based Model for Financial Vulnerability.* OFR Working Paper 2014-05.
- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading.* Econometrica, 53(6), 1315–1335.
- CPMI-IOSCO (2012). *Principles for Financial Market Infrastructures.*
- ESMA (2016). *RTS 2016/2251 — Margin Requirements for Non-Centrally Cleared OTC Derivatives.*
