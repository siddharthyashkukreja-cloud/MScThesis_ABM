# MScThesis_ABM — CCP Systemic Risk Agent-Based Model

MSc Thesis: *An Agent-Based Model of Central Clearing: Client Clearing, Contagion and Systemic Risk*

---

## Architecture

```
model/
  globals.py       ModelParams dataclass + GlobalState (fundamental process)
  lob.py           Discrete-time call-auction limit order book (matching engine)
  agents.py        BaseTrader + ZeroIntelligenceTrader (Stage 1); stubs for FV / Momentum
  simulation.py    Simulation driver (step sequence mirrors Simudyne CCP ODD)
data/
  thesis_data_calm.csv      SPY daily order-flow data, 2013-2018 (calm regime)
  thesis_data_stressed.csv  SPY daily order-flow data, 2014-2016 (stressed regime)
calibrate.py       ZI parameter estimation from BuyVol/SellVol data (to do)
run_simulation.py  Entry point
output/            Generated CSVs (git-ignored)
```

---

## Staged Development Plan

| Stage | Components | Status |
|-------|------------|--------|
| 1 | LOB matching engine + ZI traders | Done |
| 2 | FundamentalTrader + MomentumTrader + GBM fundamental | - |
| 3 | Variation margin / IM per agent (60-step calls) | - |
| 4 | Cover-2 default fund (510-step recalculation) | - |
| 5 | Five-level waterfall + position auction | - |
| 6 | Stressed vs normal data regimes | - |

---

## Data

Both CSVs contain **daily** SPY (S&P 500 ETF) observations with columns:

| Column | Description |
|--------|-------------|
| `OPrice` / `DPrice` | Open / close price |
| `Ret_mkt_t` | Daily log return |
| `BuyVol_LR1` / `SellVol_LR1` | Daily buy/sell volume (Lee-Ready, 1-tick) |
| `IVol_t_m` | Intraday realised variance (minutes) |
| `VarianceRatio1/2` | Variance ratios (microstructure noise test) |

**Calibration plan (Stage 1):** estimate Cont-Stoikov arrival rates
(lambda, mu, delta) from daily `BuyVol / SellVol` ratios and `IVol_t_m`.
High-frequency TAQ data is not required at this stage; the daily aggregates
are sufficient to anchor the order-of-magnitude of ZI rates at 5-min resolution.
HF data can be added later for more precise depth calibration.

---

## Running

```bash
pip install numpy pandas
python run_simulation.py
```
