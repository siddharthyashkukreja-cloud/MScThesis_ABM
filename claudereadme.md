# claudereadme.md — internal model state-of-affairs

> Living reference. Updated whenever code changes. Separate from README.md
> (which Sid maintains for thesis-facing documentation). Use this file to
> bring any AI / collaborator up to speed on what's implemented, what's
> deferred, and why every design choice was made.

**Last sync:** Stage 3 build complete (test suite passing) — ready for Stage 4 (margin layer).

---

## 0 · Project frame

Pure-Python ABM of a CCP-cleared single-asset market (SPY), built to
study CCP systemic-risk dynamics, margin procyclicality, and contagion
under calm vs. stressed regimes. Single LOB with **batched call-auction
clearing** at 5-min cadence (78 steps per NYSE 6.5 h trading day).

Reference priority chain (project rule: every design decision cites one;
deviations flagged):
1. Simudyne CCP Risk Model **ODD** (`Simudyne_CCPRiskModel_ODDDoc.md`)
2. Deloitte / Simudyne CCP Resilience paper
3. Majewski, Ciliberti, Bouchaud (2018) — Extended Chiarella estimation
4. Gao et al. (2023) — Deeper Hedging / stochastic-vol Chiarella
5. ABM Liquidity Risk paper (Krishnen)
6. Cont & Stoikov (2008) — Stochastic LOB models
7. Farmer, Patelli, Zovko / Daniels (2003, 2005) — ZI baseline

---

## 1 · Repository layout

```
MScThesis_ABM/
├── claudereadme.md              # this file (internal AI/collab reference)
├── README.md                    # thesis-facing readme (Sid maintains)
├── data/
│   ├── thesis_data_calm.csv     # WRDS Intraday Indicators, SPY 2013-2014
│   └── thesis_data_stressed.csv # WRDS Intraday Indicators, SPY 2008-2009 era
├── model/
│   ├── globals.py               # ModelParams + GlobalState + SimContext
│   ├── lob.py                   # Order, Fill, LOB (call auction)
│   ├── agents.py                # All trader classes
│   └── simulation.py            # Simulation driver
├── tests/
│   ├── test_stage1.py           # 6 tests, ZI baseline
│   ├── test_stage2.py           # 6 tests, FT/MT + Merton
│   └── test_stage3.py           # 8 tests, BCM/NBCM + clients
├── run_simulation.py            # Stage 3 entry point
├── calibrate.py                 # legacy; redesign deferred to Stage 8
├── analysis.ipynb               # exploratory; not yet aligned with Stage 3
└── output/
    ├── stage1_run.csv (legacy)
    └── stage2_run.csv / stage3_run.csv (newer entry points)
```

---

## 2 · Current model — what's implemented

### 2.1 LOB (`model/lob.py`)
- Single asset, price-time priority, dict-of-price-levels with FIFO queue per level.
- `add_limit(side, price, qty)` — places a resting order (rounded to tick_size).
- `add_market(side, qty)` — sweeps the opposite side at best available price; fills append to `step_fills`.
- `cancel(order_id)` — removes a resting order.
- `match()` — call-auction clearing of any crossed quotes; fills extend `step_fills`. Updates best_bid / best_ask / mid / spread.
- `age_orders()` — TTL decrement; expiry by removal.
- `snapshot()` — mid_price, best_bid, best_ask, spread, bid_depth, ask_depth, n_fills, volume.
- **Bug fixed in Stage 3:** `match()` previously *overwrote* `step_fills`, dropping any market-order fills generated during submission. Now it `extend`s.

### 2.2 GlobalState (`model/globals.py`)
- `v: float` — Merton (1976) jump-diffusion fundamental.
  - `log V_{t+1} = log V_t + (mu_v − σ_v²/2) + σ_v · ε + Σ J_i`
  - `n_jumps ~ Poisson(jump_lambda)`, each `J ~ N(jump_mean, jump_std)`
- `t: int` — discrete tick counter.
- `rng: np.random.Generator` — single seeded RNG; all stochastic draws route through it for ODD §V&V `LOBMatchingDeterminism`.

### 2.3 SimContext (`model/globals.py`)
Dataclass passed to every trader's `submit_orders`:
- `v` — current fundamental.
- `mid_price` — post-match mid from previous step (NaN before first clear).
- `momentum` — EWMA log-return signal.
- `tick` — current tick.
- `traders_by_id` — dict for CMs to look up client inventories.

### 2.4 Trader classes (`model/agents.py`)
All share signature `submit_orders(lob, params, ctx, rng)`.

| Class | Direction signal | Limit price | Qty | Notes |
|---|---|---|---|---|
| `ZeroIntelligenceTrader` | uniform 50/50 | best opposite ± k·tick (k ~ U{1, zi_offset_max}) | U{1, zi_qty_max} | Cont-Stoikov 2008 + Farmer-Daniels 2003. Submits limit (α), market (μ), cancels (per-resting δ). Empty-book fallback: anchor on V. |
| `FundamentalTrader` | sign(reservation − mid) | reservation = V + z·σ_F | U{1, zi_qty_max} | ODD §Prediction. z ~ N(0,1) drawn once at init. |
| `MomentumTrader` | sign(M_t) above threshold | V + z·σ_M | U{1, zi_qty_max} | EWMA momentum (Majewski 2018). Placement is ODD-style on V (deliberate Stage 2 extension; ODD has no MT). |
| `BankingClearingMember` | extends FT | V + z·σ_F | U{1, zi_qty_max} | Has cash (ODD ~U[5B,10B]), client_ids list, mode='fundamental' (or 'market_maker' Stage 4+). cap_ratio = cash / (own + Σ client) · mid. Breach → fire-sale market orders. |
| `NonBankingClearingMember` | does not trade | — | — | Pure structural/clearing entity. cash ~U[5M,10M]. cap_ratio = cash / Σ client · mid. Stop-out enforced at Stage 4 margin layer. |

Common balance-sheet fields on `BaseTrader`: `cash`, `inventory`, `pnl`,
`margin_posted`, `defaulted`, `clearing_member_id` (None = direct).

### 2.5 Simulation driver (`model/simulation.py`)
Per-step sequence (ODD §Step Sequence subset):
1. Reset `lob.step_fills = []`.
2. Build `SimContext` (V, prev mid, momentum, traders_by_id).
3. Each trader `submit_orders(lob, params, ctx, rng)`.
4. `lob.match()` — call-auction clearing.
5. Apply every fill to buyer/seller `inventory` & `cash`.
6. `lob.age_orders()` — TTL decrement.
7. Mark-to-market `update_pnl(mid)` on every trader.
8. Append snapshot to `history`.
9. EWMA momentum update: `M_t = λ M_{t-1} + (1−λ) (log P_t − log P_{t-1})`.
10. `gs.step()` advances V via Merton jump-diffusion.

### 2.6 Population builder (`run_simulation.py`)
Stage 3 default (all counts in `ModelParams`):
- 10 ZI direct
- 10 BCM, 3 of which have a client book attached
- 10 NBCM, all of which have a client book
- Each client book = 1 FT + 1 MT + 1 ZI (cleared via that CM)
- **Total: 30 direct + 39 clients = 69 agents.**

ID assignment: deterministic, sequential, in order: ZI direct → FT direct
→ MT direct → (BCM, then its clients) → (NBCM, then its clients).

---

## 3 · Parameters (`model/globals.py::ModelParams`)

All parameters live in one dataclass. Required (no default) first;
optional / Stage-2/3 defaults at the bottom (avoids Python dataclass
field-ordering errors).

### Required
| Name | Type | Meaning | Source |
|---|---|---|---|
| `n_zi`, `n_fundamental`, `n_momentum` | int | Direct populations | — |
| `v0` | float | Initial fundamental | calibration target |
| `tick_size` | float | Min price increment ($0.01 SPY) | NYSE rule |
| `dt_minutes` | float | Step length (5.0) | computational tractability vs ODD's 1-min |
| `order_ttl` | int | Steps before expiry (2 = 10-min, equivalent to ODD's 10×1-min) | ODD §Mech #7 |
| `zi_alpha`, `zi_mu`, `zi_delta` | float | ZI rates per step (defaults 0.15 / 0.025 / 0.025) | ODD §Calibration; Cont-Stoikov 2008 |
| `zi_qty_min`, `zi_qty_max` | int | Order qty range (1, 10) | ODD §Stochasticity |

### Defaults / Stage 2+
| Name | Default | Source |
|---|---|---|
| `zi_offset_max` | 5 ticks | Farmer-Daniels uniform-offset simplification of Cont-Stoikov empirical λ(i) |
| `ft_sigma` | 0.5 ($0.50 = 50 ticks) | placeholder; calibrate Stage 8 |
| `mt_sigma` | 0.5 | same |
| `mt_lambda_ewma` | 0.95 | Majewski 2018; calibrate later |
| `mt_threshold` | 1e-4 | min \|M_t\| to act |
| `mu_v` | 0.0 | per-step drift, ~0 for short horizons |
| `sigma_v` | 0.001 | ~9% annualised at 5-min cadence |
| `jump_lambda` | 0.0 | Poisson rate per step; ODD stressed value ≈ 0.038 (3 jumps / 78 steps) |
| `jump_mean`, `jump_std` | 0.0, 0.01 | ODD §Stochasticity |
| `kappa_v`, `theta_v`, `xi_v` | 0.0 | reserved for Gao 2023 stoch-vol extension |
| `n_bcm`, `n_nbcm` | 0 | activated at Stage 3 (10 each) |
| `n_bcm_with_clients` | 0 | how many BCMs hold a client book (default 3 at Stage 3) |

---

## 4 · Stage history & validation

| Stage | Status | Population | Tests |
|---|---|---|---|
| 1 | ✅ done | ZI only (10) | 6 / 6 pass — determinism, midprice formula, no-crossed-book, bounded spread, two-sided depth, no-negative-qty |
| 2 | ✅ done | + 10 FT + 10 MT (no CMs) | 6 / 6 pass — determinism, FT mean-reversion, MT trend-following, full-run sanity, jump-diffusion off/on |
| 3 | ✅ done | + 10 BCM + 10 NBCM + 39 clients | 8 / 8 pass — determinism, inventory + cash conservation, NBCM never trades, BCM/NBCM cap_ratio formulas, BCM fire-sale liquidates, client linkage integrity |
| 4 | pending | + IM + VM margin call (60-tick / 12-step cadence) | — |
| 5 | pending | + Default fund (Cover-2 EMIR) | — |
| 6 | pending | + 5-level default waterfall + position auction | — |
| 7 | pending | + Almgren-Chriss distressed liquidation | — |
| 8 | pending | + Stressed data; calibration | — |

### 4.1 Stage 1 annual run (78 × 252 = 19,656 steps, ZI only, sigma_v=0)
- Mid range [449.84, 450.17], drift ≈ 0.001, std 0.029
- Spread mean 3.2 ticks, p99 10, max 19
- Volume 3,880 across 4.5% of steps
- ACF(r,1) = −0.144 (Roll 1984 bid-ask bounce, expected for ZI)
- ACF(\|r\|) ≈ 0 (no vol clustering yet — emerges with MT)

### 4.2 Stage 2 annual run (10 ZI + 10 FT + 10 MT, sigma_v=0.001, jumps off)
- corr(mid, V) = 1.000 → FT mean-reversion working
- Spread mean 8.78 ticks, p99 55, max 192 (FT placement at V±z·σ_F creates wider spreads — calibrate σ_F at Stage 8)
- Volume 595k across 97.2% of steps
- ACF(r,1) = +0.010 (Cont 2001 fact #1: ~0) ✓
- Excess kurtosis = +0.57 (Cont fact #2: > 0) ✓ (mild)
- ACF(\|r\|, k) negative / ~0 — Cont fact #3 (vol clustering) **not yet**; FT dominates MT at current params; calibration target for Stage 8.

### 4.3 Stage 3 smoke (full population)
- All invariants hold (15/15 in inline smoke test).
- BCM fire-sale demo: stress BCM (cash=$100, inventory=200) → after 80 steps, inventory liquidated to 107 (53% reduction).

---

## 5 · Deviations from ODD (explicit, by design)

| # | Deviation | Reason |
|---|---|---|
| D1 | Single SPY LOB (vs ODD's dual FTSE 100/250) | Data is SPY-only |
| D2 | 5-min steps × 78/day (vs ODD's 1-min × 510/day) | Computational tractability; 78 == NObsUsed1 in WRDS row |
| D3 | ZI placement: Farmer-Daniels uniform `k ~ U{1, K}` from opposite best (vs ODD's unspecified / Cont-Stoikov empirical depth profile) | Stage 1 simplification; replace with fitted profile at Stage 8 |
| D4 | MT exists at all (ODD has only ZI + CMs) | Project staging plan: ZI → Chiarella → CMs → margin → … |
| D5 | MT placement uses ODD's V + z·σ scheme even though ODD has no MT | Keeps placement geometry consistent across non-ZI agents |
| D6 | Direct FT/MT in Stage 2 (collapsed back into BCM at Stage 3) | Test Chiarella mechanics in isolation before adding CM/balance-sheet layer |
| D7 | Client-clearing tier (39 client traders attached to CMs) | User extension; ODD has no client tier |
| D8 | Some BCMs have no clients, all NBCMs do | User design (BCMs may be prop-only, NBCMs are pure clearing) |
| D9 | NBCMs do not trade on own account; ODD describes both as similar traders | User correction (NBCM = clearing-only, no own positions) |
| D10 | No ft_kappa demand-magnitude scaling (Majewski has it; ODD does not) | ODD-faithful: CMs use fixed Uniform{1,10} qty regardless of mispricing |
| D11 | Staging order: README/project (ZI→Chiarella→CM→…) used over ODD (ZI+CM in stage 1 itself) | User decision to debug trader behaviour before adding capital constraints |

---

## 6 · Open questions / planned design decisions

- **calibrate.py is legacy.** It assumes tick-level event-typed data
  (`emini_ticks.csv`) but `data/thesis_data_*.csv` are daily WRDS
  Intraday Indicators aggregates. Redesign deferred to Stage 8. Likely
  approach: (a) calibrate Merton (μ, σ, λ_J, σ_J) from daily SPY returns
  + IVol_t_m bar-variance; (b) calibrate FT/MT params via surrogate
  matching of stylized facts (Gao 2023 / Krishnen ABM Liquidity Risk).
- **Joint Kalman + EM for FV**, à la Majewski 2018: deferred to
  Stage 8. Stage 2 baseline uses externally-fed Merton process so trader
  mechanics can be debugged in isolation.
- **BCM market-maker mode** (`mode='market_maker'`) is reserved on the
  class but not implemented. Will post symmetric bid+ask quotes around
  mid.
- **Vol clustering not yet present** at Stage 2 / 3 baselines (FT
  dominates MT at current σ values). Stage 8 calibration target.

---

## 7 · WRDS data notes

`data/thesis_data_calm.csv` (calm: 2013-…) and
`data/thesis_data_stressed.csv` (stressed: 2008-2009 era).
One row = one trading day. Columns from WRDS Intraday Indicators:

| Column | Meaning |
|---|---|
| `OPrice` | open |
| `DPrice` | close |
| `Ret_mkt_t` | daily market return |
| `BuyVol_LR1`, `SellVol_LR1` | Lee-Ready signed volume, 1-second |
| `BuyVol_LRi`, `SellVol_LRi` | Lee-Ready signed volume, i-second aggregate |
| `IVol_t_m`, `IVol_q_m` | integrated volatility (trade-based / quote-based) |
| `VarianceRatio1`, `VarianceRatio2` | variance ratio statistics |
| `TSignSqrtDVol1/2` | t-stat of signed sqrt-volume |
| `NObsUsed1` | 78 = number of 5-min bars (perfectly aligned with our step grid) |

Stage 8 calibration plan uses `IVol_t_m / NObsUsed1` as per-bar variance
target for `sigma_v`, `Ret_mkt_t` distribution for jump parameters.

---

## 8 · Roadmap of planned extensions (post-Stage-8)

From README's "Future / Possible Extensions" — preserved here for AI
context but not yet on the implementation path:

- **Volatility traders** — Gao 2023 Heston-style demand ∝ ν_t (CIR
  variance). Reserved param slots `kappa_v`, `theta_v`, `xi_v` already
  in `ModelParams`.
- **Almgren-Chriss liquidation** — distressed-CM fire-sales with optimal
  slicing → measurable temporary price-impact haircut.
- **Basel III deleveraging asymmetry** — already partially modeled in
  Stage 3 (BCM fire-sale vs NBCM stop-out); Stage 4 will add proper
  margin-call cadence.
- **Client clearing delay** — NBCM lag in passing margin calls to clients
  → transient funding gaps.
- **Strategy switching (Gao 2023)** — agents switch between FT / MT / ZI
  based on recent PnL; reinforcement-learning framework.
- **Cover-2 default fund** (Stage 5) — DF sized to cover 2 largest CM
  defaults; daily recalc; pro-rata contributions.
- **CoMargin (correlated margin)** — alternative IM scheme, compare to
  SPAN.

---

## 9 · How to run

```bash
# Stage 3 baseline
python run_simulation.py            # writes output/stage3_run.csv

# Tests
python -m unittest tests.test_stage1 tests.test_stage2 tests.test_stage3 -v
```

For analysis: `analysis.ipynb` is now Stage-3-aware. It runs a fresh
Stage 3 simulation inline, captures per-agent inventory + cash at every
step, and visualises (1) mid vs V, (2) spread evolution + distribution,
(3) bid/ask depth, (4) EWMA momentum signal, (5) per-step volume,
(6) Cont 2001 stylised facts (return distribution, ACF(r), ACF(|r|);
ACF computed inline so statsmodels is not required), (7) end-of-run
agent state by type, (8) per-CM capital-ratio time series with the 8%
floor marked, (9) sample agent trajectories per type. Just open and
run-all.

---

## 10 · Maintenance protocol for this file

When code changes, update:
- §2 component description if behaviour changed
- §3 parameter table if `ModelParams` changed
- §4 status / annual-run numbers if a new stage is built or a new run is logged
- §5 deviations table if any new deliberate departure from ODD
- §6 open-questions list when a design choice is locked or a new one opens
- §8 if an extension is moved on-path or off-path

Keep tone terse and citation-anchored. This is an AI/collab context
document, not a tutorial.
