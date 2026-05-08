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
| `BankingClearingMember` (FT-prop) | extends FT | V·(1 + z·ft_sigma_rel) | U{1, zi_qty_max} | mode='fundamental'. May carry clients. |
| `BankingClearingMember` (MM)      | mid (with inv skew) | mid·(1 + skew_frac ± half_spread_bps·1e-4) | mm_qty | mode='market_maker'. V-relative quotes; cancels and re-posts each step. No clients in baseline. |
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
- 10 BCM total split into:
  - **4 MMs** (mode='market_maker', no clients, pure liquidity providers)
  - **6 FT-prop** (mode='fundamental'); 3 of these carry a client book
- 10 NBCM, all of which have a client book
- Each client book = 1 FT + 1 MT + 1 ZI (cleared via that CM)
- **Total: 30 direct + 39 clients = 69 agents.**

ID assignment: deterministic, sequential, in order: ZI direct → FT direct
→ MT direct → (BCM, then its clients) → (NBCM, then its clients).

### 2.7 Topology — exchange access vs CCP clearing tier

Two distinct relationships, easy to conflate:

| Layer | Relationship | Population |
|---|---|---|
| **Exchange access** | Submits orders directly to LOB | All 69 agents (every trader's `submit_orders` writes to the LOB) |
| **CCP clearing — direct** | Posts margin / DF to CCP directly | 10 BCM + 10 NBCM = 20 direct CMs |
| **CCP clearing — indirect** | Posts margin to a CM, who passes it to CCP | 39 clients (3 each via 13 CMs-with-clients) |
| **CCP clearing — none** | Exogenous noise per ODD; no CCP relationship modelled | 10 direct ZI |

In the real-world / FCM-tier sense: clients are *trading-direct, clearing-indirect* — they hit the LOB on their own behalf but their post-trade clearing flows through their CM. This becomes load-bearing at Stage 4 when margin is enforced.

The 10 direct ZI mirror the ODD's 10 ZIs: pure exogenous liquidity providers without an explicit clearing layer. Adding a clearing relationship for them is a Stage 4+ extension if needed for waterfall realism.

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
| `zi_alpha`, `zi_mu`, `zi_delta` | float | ZI **rates per minute** (defaults 0.15 / 0.025 / 0.025); ZI.submit_orders draws Poisson(rate × dt_minutes) for arrivals + Bernoulli with prob 1−exp(−delta·dt) per resting order for cancellation | ODD §Calibration; Cont-Stoikov 2008 |
| `zi_qty_min`, `zi_qty_max` | int | Order qty range (1, 10) | ODD §Stochasticity |

### Defaults / Stage 2+
| Name | Default | Source |
|---|---|---|
| `zi_offset_p` | 0.5 (Geometric param) | Cont-Stoikov 2008 exponential-decay depth profile, discrete-tick analog |
| `zi_offset_max` | 20 ticks | Truncation cap on Geometric tail |
| `ft_sigma_rel` | 0.005 (= 50 bps of V) | V-relative offset; regime-invariant; calibrate Stage 8 |
| `mt_sigma_rel` | 0.005 | same |
| `mt_lambda_ewma` | 0.95 | Majewski 2018; calibrate later |
| `mt_threshold` | 1e-4 | min \|M_t\| to act |
| `mu_v` | 0.0 | per-step drift, ~0 for short horizons |
| `sigma_v` | 0.001 | ~9% annualised at 5-min cadence |
| `jump_lambda` | 0.0 | Poisson rate per step; ODD stressed value ≈ 0.038 (3 jumps / 78 steps) |
| `jump_mean`, `jump_std` | 0.0, 0.01 | ODD §Stochasticity |
| `kappa_v`, `theta_v`, `xi_v` | 0.0 | reserved for Gao 2023 stoch-vol extension |
| `n_bcm`, `n_nbcm` | 0 | activated at Stage 3 (10 each) |
| `n_bcm_mm` | 0 | of n_bcm, how many run in market-maker mode (default 4 at Stage 3.5) |
| `n_bcm_with_clients` | 0 | of the FT-prop BCMs, how many hold a client book (default 3) |
| `mm_half_spread_bps` | 30.0 (bps of mid) | MM half-spread; V-relative (Stoikov 2008) |
| `mm_qty` | 50 | MM quote size per side |
| `mm_inventory_skew_bps` | 0.5 (bps/unit of mid) | per-unit-inventory quote shift; bps of mid per unit |

---

## 4 · Stage history & validation

| Stage | Status | Population | Tests |
|---|---|---|---|
| 1 | ✅ done | ZI only (10) | 6 / 6 pass — determinism, midprice formula, no-crossed-book, bounded spread, two-sided depth, no-negative-qty |
| 2 | ✅ done | + 10 FT + 10 MT (no CMs) | 6 / 6 pass — determinism, FT mean-reversion, MT trend-following, full-run sanity, jump-diffusion off/on |
| 3 | ✅ done | + 10 BCM + 10 NBCM + 39 clients | 8 / 8 pass — determinism, inventory + cash conservation, NBCM never trades, BCM/NBCM cap_ratio formulas, BCM fire-sale liquidates, client linkage integrity |
| 8a | ✅ done | direct calibration (sigma_v from data) + indirect grid search | calibrate.py works; 12-point grid found vol-clustering params |
| 4 | pending | + IM + VM margin call (60-tick / 12-step cadence) | — |
| 5 | pending | + Default fund (Cover-2 EMIR) | — |
| 6 | pending | + 5-level default waterfall + position auction | — |
| 7 | pending | + Almgren-Chriss distressed liquidation | — |
| 8b | pending | + Kalman-Filter + EM comparison; final parameter selection on stressed data | — |

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

## 4b · Calibration (Stage 8, run BEFORE Stage 4 margin per project plan)

`calibrate.py` does direct + indirect calibration:

### Direct (data-derived parameters)

| Parameter | Source | Calibrated value (calm) |
|---|---|---|
| `v0` | mean OPrice across calm dataset | 178.82 |
| `mu_v` (per 5-min) | mean of log(DPrice/OPrice) / 78 | ~4.2e-6 (≈ 0) |
| `sigma_v` (per 5-min) | empirical OC daily std / sqrt(78) | 6.52e-4 |

`calibrate.py verify` cross-checks WRDS `IVol_t_m` units against three hypotheses (per-second RV, daily IV, per-bar var). Best fit (log-ratio metric) is **H1: per-second RV**, giving an implied 5-min std of 1.46e-3 — about **2.24× the empirical-OC-derived value of 6.52e-4**. This is a known microstructure-noise overshoot in high-frequency RV estimators for SPY. We use the empirical-OC value as primary (`source='empirical'` in `extract_direct_params`) and treat IVol as a secondary cross-check.

### Indirect (stylized-fact minimization, Krishnen + Gao 2023)

Targets from WRDS calm-regime daily log-returns (Cont 2001 facts):

| Moment | Empirical |
|---|---|
| `ret_std` | 0.0070 |
| `ret_kurtosis_excess` | +1.31 |
| `ACF(r, 1)` | −0.051 |
| `ACF(r, 5)` | −0.062 |
| `ACF(|r|, 1)` | +0.164 |
| `ACF(|r|, 5)` | +0.060 |
| `ACF(|r|, 20)` | −0.059 |

Grid search (12-point: `ft_sigma ∈ {0.1, 0.5}`, `mt_sigma ∈ {0.1, 1.0}`, `mt_lambda_ewma ∈ {0.7, 0.9, 0.95}`, `n_steps = 78×30`, `n_runs = 2`):

**Best fit:** `ft_sigma=0.5, mt_sigma=0.1, mt_lambda=0.9` (loss 1.20)
- `ret_std`: 0.005 (slightly under target 0.007 — sim mid is smoother than V; bumping `sigma_v` to 9.1e-4 closes the gap)
- `kurt`: +2.29 (target +1.31; mild overshoot)
- `ACF(r,1)`: −0.006 ✓ (target ~0)
- `ACF(|r|,1)`: +0.127 ✓ (target +0.164) — **vol clustering achieved**
- `ACF(|r|,5)`: +0.052 ✓ (target +0.060) — **slow decay**

This is the first parameter set where Cont 2001 fact #3 (vol clustering) holds. The Stage 3 default (`mt_sigma=0.5, mt_lambda=0.95`) had `ACF(|r|,1)≈0`. Calibration was the missing piece.

To use the calibrated params in a sim, pass them explicitly to `ModelParams(...)` in `run_simulation.py`. The current globals.py defaults are still the pre-calibration baseline; once you're satisfied with calibration, update them.

### Recalibration after BCM-MM hybrid (Stage 3.5)

Adding 4 MMs (mode='market_maker') changed the dynamics enough that the calibrated grid search shifted:

| Top result by criterion | params | ret_std | kurt | ACF(\|r\|, 1) | ACF(\|r\|, 5) | loss |
|---|---|---|---|---|---|---|
| Lowest loss | `ft=0.1, mt=0.1, λ=0.7` | 0.0063 | +0.68 | −0.05 | +0.03 | 0.51 |
| Best vol clustering | `ft=0.5, mt=0.1, λ=0.95` | 0.0057 | +0.28 | **+0.27** | −0.01 | 1.24 |

MMs damp directional flow at the inside spread, so vol clustering (ACF\|r\|) becomes harder to elicit — the loss-minimum doesn't coincide with the vol-clustering-maximum any more. Two ways forward depending on user priority:
- Reweight loss to favour `ACF(|r|)` over `ret_std` exact match.
- Add a small fast-MT subpopulation (Krishnen high-η) to amplify trend-following without overshooting std.

### Stressed-data caveat — DATA ISSUE

`thesis_data_stressed.csv` is dated **2014-01-02 → 2014-12-31** (252 rows). 2014 was a calm SPY year. Direct calibration shows `σ_v` ratio of just **1.04×** vs calm; the file isn't actually a stressed regime. Project instructions and the ODD reference **2008–2009** GFC data. The stressed-regime calibration framework is in place and runs — but the data file needs to be re-pulled from WRDS TAQ for an actual stress period (2008-Q4, 2020-Q1 COVID, or 2015-Q3 China devaluation) before stress dynamics will show up.

### Kalman + EM comparison (still pending)

User asked to also try Kalman-Filter + EM (Majewski 2018) for comparison. Not yet implemented; meaningful next step once stressed data is corrected.

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
| D12 | ZI rates `zi_alpha`/`zi_mu`/`zi_delta` are now **per-minute rates** (Poisson arrivals + per-resting Bernoulli cancellation rescaled by dt_minutes), not Bernoulli probs per step | Cont-Stoikov continuous-time semantics; preserves expected order rate when changing dt_minutes |
| D13 | ZI limit-price depth uses `Geometric(zi_offset_p)` capped at `zi_offset_max` instead of uniform | Cont-Stoikov 2008 fitted exponential decay |
| D14 | Calibration ordered before margin (Stage 8 before Stage 4) | Lock down market dynamics before adding capital-constraint feedback |
| D15 | BCM split into 4 MMs + 6 FT-prop (vs ODD's homogeneous BCM population) | Real bank-tier mix; MMs provide explicit liquidity layer; matches dealer-bank role in market microstructure |
| D16 | All agent placement scales (`ft_sigma_rel`, `mt_sigma_rel`, `mm_half_spread_bps`, `mm_inventory_skew_bps`) are **V-relative / basis points** rather than absolute dollars | Regime-invariant calibration: same params work for V≈$92 (stressed) and V≈$179 (calm). Fixed the +64% stress amplification observed under absolute-cents scales. |
| D17 | CIR stochastic volatility (Heston/Gao 2023) added on top of Merton jump-diffusion (`xi_v=0` falls back to constant σ_v) | Stage 8 prep: long-memory vol clustering (Cont 2001 fact #3 at long lags) needs vol-of-vol. Available but typically `xi_v` small for baseline; calibrated by LHS. |

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

## 7b · HFABM (Cao et al. 2024 JASSS) extensions to consider

[High-frequency financial market simulation and flash crash scenarios analysis](https://www.jasss.org/27/2/8.html) (arXiv [2208.13654](https://arxiv.org/abs/2208.13654)) presents a millisecond-cadence ABM for E-mini S&P 500 used to recreate the 2010 Flash Crash. Two design choices differ from ours and are worth flagging as extension options:

**MM inventory limit** (HFABM §3.2, "Market Maker"). In HFABM the MM posts limits during normal trading; **once |inventory| exceeds an absolute limit, MM submits market orders to liquidate**. This is similar in spirit to our BCM `_fire_sale` but tuned for normal flow (not capital-ratio breach). They show in Monte Carlo that the inventory limit is a key driver of mini-flash-crash amplitude. Could add as `mm_inventory_limit` parameter; if `|inventory| > limit`, switch from quote-posting to market-order liquidation.

**Fundamental trader as market-order-only** (HFABM §3.3). Their FT submits market orders only — opposite of our ODD-faithful FT-as-limit-orders. They calibrate FT trading frequency to E-mini bar data. Our design follows ODD §Prediction (limit orders at private valuation); HFABM's design is more HFT-realistic and would interact more aggressively with MM quotes. Worth implementing as an alternative `FundamentalTrader_HFABM` for comparison if Stage 8 calibration struggles.

**ML surrogate calibration** (HFABM §4.2). They calibrate via a neural-network surrogate trained on grid evaluations, reducing thousands of objective evaluations to one neural-network forward pass per candidate. Our LHS approach is simpler; if calibration becomes the bottleneck, this is a documented upgrade path.

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

For analysis: `analysis.ipynb` is now Stage-8 calibration-progress notebook. It runs the simulation under both calm AND stressed regimes (fix-agents-swap-environment), computes Cont 2001 stylized-fact moments for each, and visualises:
1. Empirical reference (calm 2013-14 vs stressed Sept 2008 - Mar 2009)
2. Empirical Cont 2001 target moments
3. Sim runs under both regimes (60 days × 3 seeds averaged)
4. **Side-by-side moment comparison** with sim/target ratios — the headline diagnostic
5. Mid-vs-V plots for both regimes
6. Return distribution histograms (sim overlaid on empirical)
7. ACF(r) and ACF(|r|) overlay (sim vs empirical, both regimes, both lags 0-20)
8. CIR variance trajectory under each regime
9. MM inventory + quote dynamics (HFABM-style diagnostic)
10. CM cap_ratio plots (BCM split by mode, NBCM)
11. End-of-run agent state by type
12. **Calibration progress**: loads `output/calibration_lhs.csv` if present (created by `python calibrate.py lhs 200 90 3` on user's laptop) and shows loss landscape across the LHS search.

Open and run-all in Jupyter.

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
