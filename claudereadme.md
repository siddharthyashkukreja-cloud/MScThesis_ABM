# claudereadme.md — internal model state-of-affairs

> Living reference. Updated whenever code changes. Separate from README.md
> (which Sid maintains for thesis-facing documentation). Use this file to
> bring any AI / collaborator up to speed on what's implemented, what's
> deferred, and why every design choice was made.

**Last sync:** Stage 3 + 3.5 (BCM-MM hybrid + V-relative scales) + 6prep
(CIR stochastic vol) + 8a (LHS calibration framework) complete. Population
restructured to 98 agents with all non-CM traders attached as clients (no
"direct" market participants). MT placement switched from V-anchor to
mid-anchor. Gao 2023 noise-trader vol scaling and ODD-faithful jumps now
active by default. 24/24 tests pass.

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
8. HFABM — Cao et al. 2024 JASSS (extension reference)

---

## 1 · Repository layout

```
MScThesis_ABM/
├── claudereadme.md              # this file (internal AI/collab reference)
├── README.md                    # thesis-facing readme (Sid maintains)
├── data/
│   ├── thesis_data_calm.csv     # WRDS Intraday Indicators, SPY 2013-2014
│   └── thesis_data_stressed.csv # WRDS Intraday Indicators, SPY 2008-09 (Sept 2008 - Mar 2009)
├── model/
│   ├── globals.py               # ModelParams + GlobalState + SimContext
│   ├── lob.py                   # Order, Fill, LOB (call-auction)
│   ├── agents.py                # All trader classes
│   └── simulation.py            # Simulation driver
├── tests/
│   ├── test_stage1.py           # 6 tests, ZI baseline invariants
│   ├── test_stage2.py           # 8 tests, FT/MT + Merton + CIR
│   └── test_stage3.py           # 10 tests, BCM/NBCM + clients + MM
├── run_simulation.py            # entry point + build_traders()
├── calibrate.py                 # direct + indirect (LHS) calibration
├── analysis.ipynb               # 26-cell Stage-8 calibration-progress notebook
└── output/
    ├── calibration_lhs.csv      # produced by `python calibrate.py lhs ...`
    └── stage*_run.csv           # ad-hoc run outputs
```

---

## 2 · Current model — what's implemented

### 2.1 LOB (`model/lob.py`)
- Single asset, price-time priority, dict-of-price-levels with FIFO queue per level.
- `add_limit(side, price, qty)` — places a resting order (price rounded to tick_size).
- `add_market(side, qty)` — sweeps the opposite side at best available price; fills append to `step_fills`.
- `cancel(order_id)` — removes a resting order.
- `match()` — call-auction clearing of any crossed quotes; fills extend `step_fills`. Updates best_bid / best_ask / mid / spread.
- `age_orders()` — TTL decrement; expiry by removal.
- `snapshot()` — mid_price, best_bid, best_ask, spread, bid_depth, ask_depth, n_fills, volume.
- **Key invariant**: `match()` extends rather than overwrites `step_fills` so market-order fills generated during submission survive into the per-step fills list.

### 2.2 GlobalState (`model/globals.py`)
- `v: float` — fundamental price.
- `v_var: float` — instantaneous variance (CIR state when xi_v>0; otherwise sigma_v²).
- `t: int` — discrete tick counter.
- `rng: np.random.Generator` — single seeded RNG for ODD §V&V `LOBMatchingDeterminism`.

Per-step update (`gs.step()`):
1. **CIR variance update** (only if xi_v>0):
   `v_var_{t+1} = max(0, v_var_t + κ_v(θ_v − v_var_t) + ξ_v · √v_var_t · Z_2)`
2. **Diffusion**: σ_t = √v_var_t; `log V_{t+1} = log V_t + (μ_v − σ_t²/2) + σ_t · Z_1`
3. **Jumps** (Merton 1976, only if jump_lambda>0): n_jumps ~ Poisson(jump_lambda), each J ~ N(jump_mean, jump_std), summed and added to log V.

### 2.3 SimContext (`model/globals.py`)
Dataclass passed to every trader's `submit_orders`:
- `v` — current fundamental.
- `mid_price` — post-match mid from previous step (NaN before first clear).
- `momentum` — EWMA log-return signal.
- `tick` — current tick.
- `traders_by_id` — dict for CMs to look up client inventories.
- `v_var` — current CIR variance state (used by Gao ZI vol scaling).

### 2.4 Trader classes (`model/agents.py`)
All share signature `submit_orders(lob, params, ctx, rng)`.

| Class | Direction | Limit price | Qty | Notes |
|---|---|---|---|---|
| `ZeroIntelligenceTrader` | uniform 50/50 | best opposite ± k·tick, k ~ Geom(zi_offset_p) capped at zi_offset_max | U{1, qty_max_eff} where qty_max_eff = round(zi_qty_max·√(v_var/θ_v)) | Cont-Stoikov 2008 + Farmer-Daniels 2003 + **Gao 2023 vol-scaled qty**. Submits limit (Poisson α·dt arrivals), market (Poisson μ·dt), cancels (per-resting Bernoulli with prob 1−exp(−δ·dt)). Empty-book fallback: anchor on V. |
| `FundamentalTrader` | sign(reservation − mid) | reservation = V·(1 + z·ft_sigma_rel) | U{dir_qty_min, dir_qty_max} | ODD §Prediction. z ~ N(0,1) drawn once at init. V-relative offset (regime-invariant). |
| `MomentumTrader` | sign(M_t) above threshold | mid·(1 + z·mt_sigma_rel), V-fallback for empty book | U{dir_qty_min, dir_qty_max} | EWMA momentum (Majewski 2018, Krishnen). **Mid-anchored** — MTs trade trends in observed prices, not fundamentals. |
| `BankingClearingMember` (FT-prop) | extends FT | V·(1 + z·ft_sigma_rel) | U{dir_qty_min, dir_qty_max} | mode='fundamental'. May carry clients. cap_ratio = cash / (\|own_inv\| + Σ\|client_inv\|)·mid. Fire-sale lots also use dir_qty_max. |
| `BankingClearingMember` (MM) | mid (with inv skew) | bid = mid·(1 + skew_frac − half_bps·1e-4); ask = mid·(1 + skew_frac + half_bps·1e-4) | mm_qty | mode='market_maker'. Cancels prior + reposts each step. **MMs CAN have clients** in current default (matches GS/JPM real-bank-tier model). |
| `NonBankingClearingMember` | does not trade | — | — | Pure clearing entity. cash ~U[5M,10M]. cap_ratio = cash / Σ\|client_inv\|·mid. Stop-out enforced at Stage 4 margin layer. |

Cap-ratio breach (≤ 0.08) on any BCM triggers `_fire_sale`: market-order liquidate own inventory in chunks of `zi_qty_max`.

Common `BaseTrader` fields: `cash`, `inventory`, `pnl`, `margin_posted`, `defaulted`, `clearing_member_id` (None = direct, but in current default no agent is direct).

### 2.5 Simulation driver (`model/simulation.py`)
Per-step sequence (ODD §Step Sequence subset):
1. Reset `lob.step_fills = []`.
2. Build `SimContext` (V, prev mid, momentum, traders_by_id, v_var).
3. Each trader `submit_orders(lob, params, ctx, rng)`.
4. `lob.match()` — call-auction clearing.
5. Apply every fill to buyer/seller `inventory` & `cash`.
6. `lob.age_orders()` — TTL decrement.
7. Mark-to-market `update_pnl(mid)` on every trader.
8. Append snapshot to `history`.
9. EWMA momentum update from new mid.
10. `gs.step()` advances V (and v_var if CIR active).

### 2.6 Population builder (`run_simulation.py`)
**Real-bank-tier population — 98 agents** (project plan):
- **0 direct ZI / FT / MT** — every non-CM trader is attached to a CM as a client (no exogenous noise floats).
- **15 BCMs** total:
  - **8 MMs WITH clients** (mode='market_maker', cleared as well as quoting — like GS/JPM/MS)
  - **7 FT-prop without clients** (proprietary banks)
- **5 NBCMs** all with clients (small clearing-only firms).
- **Each CM-with-clients** carries `clients_per_book = 6` (default 2 FT + 2 MT + 2 ZI).
- **78 clients total** (8 + 5 = 13 books × 6).
- **Cash distribution**: BCM ~U[5B, 10B] (ODD), NBCM ~U[5M, 10M] (ODD), institutional clients (FT/MT) ~U[1M, 10M], retail clients (ZI) ~U[10k, 100k].

ID assignment: deterministic, sequential — direct ZI/FT/MT (currently empty) → (BCM, then its clients) → (NBCM, then its clients).

### 2.7 Topology — exchange access vs CCP clearing tier

| Layer | Relationship | Population (98-agent default) |
|---|---|---|
| **Exchange access** | Submits orders directly to LOB | All 98 agents |
| **CCP clearing — direct** | Posts margin / DF to CCP directly | 15 BCM + 5 NBCM = **20 direct CMs** |
| **CCP clearing — indirect** | Posts margin to a CM, who passes it to CCP | **78 clients** (8 BCM-with-clients books + 5 NBCM books, 6 clients each) |
| **CCP clearing — none** | Exogenous noise; no CCP relationship modelled | **0** in current default (was 10 ZIs in earlier setup) |

Clients are *trading-direct, clearing-indirect* — they hit the LOB on their own behalf but post-trade clearing flows through their CM. This becomes load-bearing at Stage 4 when margin enforcement activates.

---

## 3 · Parameters (`model/globals.py::ModelParams`)

All parameters live in one dataclass. Required (no default) first;
optional / Stage-2/3 defaults at the bottom (Python dataclass field-ordering).

### Required
| Name | Type | Meaning | Source |
|---|---|---|---|
| `n_zi`, `n_fundamental`, `n_momentum` | int | **Direct** populations (default 0 in real-bank-tier setup) | — |
| `v0` | float | Initial fundamental | calibration target |
| `tick_size` | float | Min price increment ($0.01 SPY) | NYSE rule |
| `dt_minutes` | float | Step length (5.0) | computational tractability vs ODD's 1-min |
| `order_ttl` | int | Steps before expiry (2 = 10-min, equivalent to ODD's 10×1-min) | ODD §Mech #7 |
| `zi_alpha`, `zi_mu`, `zi_delta` | float | ZI **rates per minute** (defaults 0.15 / 0.025 / 0.025); ZI.submit_orders draws Poisson(rate × dt) for arrivals + Bernoulli with prob 1−exp(−δ·dt) per resting order for cancellation | ODD §Calibration; Cont-Stoikov 2008 |
| `zi_qty_min`, `zi_qty_max` | int | ZI order qty range (1, 10) — Poisson arrivals preserve per-minute volume across dt | ODD §Stochasticity |

### Defaults / Stage 2+ (V-relative scales)
| Name | Default | Source |
|---|---|---|
| `zi_offset_p` | 0.5 (Geometric param) | Cont-Stoikov 2008 exponential-decay depth profile, discrete-tick analog |
| `zi_offset_max` | 20 ticks | Truncation cap on Geometric tail |
| `dir_qty_min`, `dir_qty_max` | 5, 50 | Directional-agent order qty (FT, MT, BCM-FT-prop, BCM fire-sale lots, FT/MT clients). ODD's `U[1,10]` × `dt_minutes=5`. |
| `ft_sigma_rel` | 0.005 (= 50 bps of V) | V-relative offset; regime-invariant; calibrate Stage 8 |
| `mt_sigma_rel` | 0.005 | same; mid-anchored placement |
| `mt_lambda_ewma` | 0.95 | Majewski 2018; calibrate later |
| `mt_threshold` | 1e-4 | min \|M_t\| to act |

### Fundamental dynamics
| Name | Default | Source |
|---|---|---|
| `mu_v` | 0.0 | per-step drift, ~0 for short horizons |
| `sigma_v` | 0.001 | constant-vol fallback when CIR off (~9% annualised at 5-min cadence) |
| `jump_lambda` | **0.0385** | per-step Poisson rate; ODD §Stochasticity (3/78) — **active by default** |
| `jump_mean`, `jump_std` | 0.0, 0.01 | ODD §Stochasticity |
| `kappa_v`, `theta_v`, `xi_v` | 0.0 | CIR mean-reversion / long-run var / vol-of-vol; CIR active when xi_v > 0 |

### Population (Stage 3+, real-bank-tier defaults)
| Name | Default | Meaning |
|---|---|---|
| `n_bcm` | 0 (set 15 in run_simulation.py) | Banking CMs total |
| `n_bcm_mm` | 0 (set 8) | of n_bcm, how many are MMs |
| `n_bcm_with_clients` | 0 (set 8) | of n_bcm, how many carry clients (independent of mode flag) |
| `n_nbcm` | 0 (set 5) | Non-Banking CMs (all carry clients) |
| `clients_per_book` | 6 | clients per CM-with-clients |
| `client_book_ft`, `client_book_mt`, `client_book_zi` | 2, 2, 2 | client type composition per book (must sum to clients_per_book) |

### MM quoting (Stoikov 2008-style, V-relative)
| Name | Default | Source |
|---|---|---|
| `mm_half_spread_bps` | 30.0 (bps of mid) | MM half-spread above/below mid |
| `mm_qty` | 50 | MM quote size per side |
| `mm_inventory_skew_bps` | 0.5 (bps/unit of mid) | per-unit-inventory quote shift; skews against position |

---

## 4 · Stage history & validation

| Stage | Status | Population | Tests |
|---|---|---|---|
| 1 | ✅ done | ZI only | 6 / 6 pass — determinism, midprice formula, no-crossed-book, bounded spread, two-sided depth, no-negative-qty |
| 2 | ✅ done | + FT + MT + Merton + CIR | 8 / 8 pass — determinism, FT mean-reversion, MT trend-following, full-run sanity, jumps off/on, CIR off/on |
| 3 | ✅ done | + BCM (FT-prop and MM modes) + NBCM + clients | 10 / 10 pass — determinism, inventory + cash conservation, NBCM never trades, BCM/NBCM cap_ratio formulas, BCM fire-sale, MM two-sided quotes, MM inventory skew, client linkage integrity |
| 3.5 | ✅ done | BCM-MM hybrid; V-relative scales (D16) | tests still pass |
| 6prep | ✅ done | CIR stochastic volatility (Heston/Gao 2023) | tests still pass; xi_v=0 falls back to constant σ_v |
| 8a | ✅ done | Direct + LHS indirect calibration framework | calibrate.py CLI: `verify`, `direct`, `calibrate`, `lhs` |
| 8b | pending | Bigger LHS run + Kalman+EM comparison | run on user's laptop: `python calibrate.py lhs 200 90 3` |
| 4 | pending | + IM + VM margin call (60-tick / 12-step cadence) | — |
| 5 | pending | + Default fund (Cover-2 EMIR) | — |
| 6 | pending | + 5-level default waterfall + position auction | — |
| 7 | pending | + Almgren-Chriss distressed liquidation | — |

### 4.1 Calm vs Stressed run (current default population, ft=mt=0.005, λ=0.95, jumps on, CIR with xi=1e-5)

```
                       ret_std    kurt      ACF(r,1)   ACF(|r|,1)   ACF(|r|,5)
EMPIRICAL CALM     :   0.0070     +1.31     -0.051       +0.164       +0.060
SIMULATED CALM     :   0.0198     +0.19     -0.090       +0.062       +0.060
                       +183%                              positive ✓   match ✓

EMPIRICAL STRESS   :   0.0340     +0.74     -0.142       -0.004       +0.275
SIMULATED STRESS   :   0.0392     +0.14     +0.011       +0.104       +0.056
                       +15%                                            
```

**Direct σ_v** scales correctly: calm 6.5e-4 → stressed 3.18e-3 (4.87× ratio matches the 4.87× ratio in empirical OC daily std).

**Stress amplification fixed** by V-relative scales (was +64% pre-fix, now +15%). Calm overshoot (+183%) is the new structural mismatch from the bigger 98-agent population + ODD jumps + Gao vol scaling — needs LHS recalibration to find lower (ft_sigma_rel, mt_sigma_rel) values.

### 4.2 Stage 1 annual run (legacy ZI-only)
Mid range [449.84, 450.17], spread mean 3.2 ticks, ACF(|r|) ≈ 0 — no vol clustering with pure ZI, expected.

### 4.3 Sanity invariants in current 98-agent setup
- Total inventory across all agents = 0 every step ✓
- Cash conservation up to floating-point ✓
- NBCM inventory exactly 0 ✓
- BCM cap_ratios stay astronomical (~1e3 to 1e6) under 60-day runs — no breaches yet (margin layer not active).

---

## 4b · Calibration — XGBoost surrogate (Stage 8)

`calibrate.py` is now a focused 4-step pipeline using **XGBoost as a surrogate** of the simulator-to-moments map (SMAC-style, Hutter et al. 2011 / surrogate-assisted calibration in HFABM Cao 2024 §4.2 / Gao 2023). Replaces the legacy 12-point grid-search and standalone LHS workflows.

```bash
python calibrate.py verify              # WRDS IVol_t_m unit-hypothesis check
python calibrate.py direct              # data-derived sigma_v, v0, mu_v (both regimes)
python calibrate.py run [N D R]         # end-to-end (default 200 90 3, ~15 min)
```

The `run` subcommand executes:
1. **LHS training data** — sample N parameter vectors θ from the 5-D search space; for each, simulate both regimes for D days × R seeds; record daily-return moment vector. Saves to `output/calibration_lhs.csv`.
2. **Train XGBoost surrogate** — one `XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)` per (regime, moment) pair; 14 regressors total (7 moments × 2 regimes). XGBoost handles 200-row training sets well, robust to MC-noisy targets, supports quantile regression for uncertainty estimates.
3. **Multi-start L-BFGS-B optimisation** — surrogate loss = per-moment-normalised L2 distance, regime-summed; 30 random L-BFGS-B starts on the surrogate (sub-second total). Falls back to dense random search if scipy unavailable.
4. **Validate** — run actual simulator at θ* and compare predicted vs simulated moments. If close (~5% relative error per moment) the surrogate isn't overfitting.

Outputs:
- `output/calibration_lhs.csv` — raw training data
- `output/calibrated_params.json` — final calibrated θ + validated moments + all metadata
- `output/calibration_feature_importance.csv` — per-(regime, moment) param importance ranking

### Search space (5-D, V-relative)

| Parameter | Bounds | Notes |
|---|---|---|
| `ft_sigma_rel` | [0.0005, 0.05] | 5 bps to 500 bps of V |
| `mt_sigma_rel` | [0.0005, 0.05] | same |
| `mt_lambda_ewma` | [0.50, 0.99] | EWMA decay |
| `kappa_v` | [0.001, 1.0] | CIR mean-reversion speed |
| `log10(xi_v)` | [−7, −3] | xi_v ∈ [1e-7, 1e-3], log-spaced |

**Per-regime `theta_v` is set directly from each regime's empirical σ_v²** (not in the search space) — this is what makes the calibrated θ regime-invariant. `jump_lambda`/`jump_mean`/`jump_std` are anchored at ODD values (0.0385/0/0.01) and not searched.

### Loss function

Per-moment-normalised weighted L2:
```
loss = Σ_regime regime_weight × Σ_moment moment_weight × ((sim - target) / max(|target|, 0.05))²
```

Default moment weights `[1.0, 1.0, 0.5, 0.3, 2.0, 1.5, 1.0]` for `[ret_std, kurt, ACF(r,1), ACF(r,5), ACF(|r|,1), ACF(|r|,5), ACF(|r|,20)]` — emphasises vol-clustering moments (Cont 2001 fact #3). Regime weights default to (1.0, 1.0).

### Cont 2001 stylized-fact targets (empirical, daily SPY)

| Moment | Calm (2013-14, 503 days) | Stressed (Sept 2008–Mar 2009, 146 days) |
|---|---|---|
| `ret_std` | 0.0070 | 0.0340 |
| `ret_kurtosis_excess` | +1.31 | +0.74 |
| `ACF(r, 1)` | −0.051 | −0.142 |
| `ACF(\|r\|, 1)` | +0.164 | −0.004 |
| `ACF(\|r\|, 5)` | +0.060 | +0.275 |
| `ACF(\|r\|, 20)` | −0.059 | +0.076 |

Stressed shows **delayed vol clustering** (lag-5 ACF dominates over lag-1) — characteristic of GFC where vol regimes persisted multi-day. CIR captures this; jumps capture the kurtosis component.

### Why XGBoost (vs alternatives)

| Method | Sims needed | Quality | Interpretability | Status |
|---|---|---|---|---|
| LHS pick-best (legacy) | 200 | crude — discrete | low | removed from CLI |
| **XGBoost surrogate** (current) | **200 + 30s training** | smooth interpolation | **feature importance** | **ready to run** |
| Bayesian opt (gp_minimize) | 80-120 | smooth, uncertainty-aware | medium | needs `skopt` |
| ML-NN surrogate (HFABM) | 500+ | flexible, data-hungry | low | Stage 9+ |
| Kalman + EM (Majewski) | one-shot MLE | rigorous, model-locked | high | Stage 8b |

XGBoost is the same data efficiency as LHS but extracts much more value from those 200 simulator calls — gives you a callable function plus feature importance plus (optionally) quantile uncertainty bands. The L-BFGS-B refinement on the surrogate finds a better optimum than picking the lowest-loss LHS row.

### Dependencies (laptop-side)

The XGBoost path needs `xgboost` and `scipy`. The cowork sandbox doesn't have these, so the `run` subcommand executes on your laptop:

```bash
pip install xgboost scipy            # one-time
python calibrate.py run 200 90 3     # ~15 min
```

`verify` and `direct` don't need any extra packages and run anywhere.

### Active learning / Bayesian-opt loop (Stage 8b extension)

Form 1 (above) is "train once, optimise once." A natural follow-on is **active-learning iteration**:

```
Repeat for K rounds:
  1. Train XGBoost on accumulated (θ, m) pairs
  2. Surrogate-optimise to propose N_iter candidates
  3. Run actual simulator at candidates; append to training set
```

This converges in 60-80% fewer simulator calls than blind LHS. Optional Stage 8b extension; current Form 1 framework is the foundation.

### Kalman + EM comparison (Stage 8b extension)

Majewski 2018-style joint state-space estimation (V latent + behavioural params) remains an open task — useful as an independent methodology to validate the XGBoost surrogate's calibrated θ.

---

## 5 · Deviations from ODD (explicit, by design)

| # | Deviation | Reason |
|---|---|---|
| D1 | Single SPY LOB (vs ODD's dual FTSE 100/250) | Data is SPY-only |
| D2 | 5-min steps × 78/day (vs ODD's 1-min × 510/day) | Computational tractability; 78 == NObsUsed1 in WRDS row |
| D3 | ZI placement: Cont-Stoikov 2008 Geometric depth `k ~ Geom(zi_offset_p)` capped at zi_offset_max (vs ODD's unspecified placement) | Cont-Stoikov 2008 fitted exponential depth profile, discrete-tick analog |
| D4 | MT exists at all (ODD has only ZI + CMs) | Project staging plan: ZI → Chiarella → CMs → margin → … |
| D5 | MT placement is **mid-anchored** `mid·(1 + z·mt_sigma_rel)`, V-fallback for empty book | Momentum traders react to observed price action, so anchor on mid (not V); matches Majewski 2018 / Krishnen Chiarella momentum spec. Changed from V-anchor in current revision. |
| D6 | Direct FT/MT in Stage 2 (collapsed back into BCM/clients at Stage 3+) | Test Chiarella mechanics in isolation before adding CM/balance-sheet layer |
| D7 | Client-clearing tier (78 client traders attached to CMs) | User extension; ODD has no client tier |
| D8 | Some BCMs have clients, all NBCMs do; **MMs are among those with clients** (real-bank-tier) | User design — banks like GS/JPM market-make AND clear for clients |
| D9 | NBCMs do not trade on own account; ODD describes both as similar traders | User correction: NBCM = clearing-only |
| D10 | No ft_kappa demand-magnitude scaling (Majewski has it; ODD does not) | ODD-faithful: CMs use fixed Uniform{1,10} qty regardless of mispricing |
| D11 | Staging order: README/project (ZI→Chiarella→CM→…) used over ODD's (ZI+CM in stage 1 itself) | User decision to debug trader behaviour before adding capital constraints |
| D12 | ZI rates `zi_alpha`/`zi_mu`/`zi_delta` are now **per-minute rates** (Poisson arrivals + per-resting Bernoulli cancellation rescaled by dt_minutes), not Bernoulli probs per step | Cont-Stoikov continuous-time semantics; preserves expected order rate when changing dt_minutes |
| D13 | ZI limit-price depth uses `Geometric(zi_offset_p)` capped at `zi_offset_max` (vs uniform Farmer-Daniels) | Cont-Stoikov 2008 fitted exponential decay |
| D14 | Calibration ordered before margin (Stage 8 before Stage 4) | Lock down market dynamics before adding capital-constraint feedback |
| D15 | BCM split into **8 MMs (with clients) + 7 FT-prop (no clients)** out of 15 (vs ODD's homogeneous BCM population) | Real-bank-tier mix (GS/JPM/MS structure) |
| D16 | All agent placement scales (`ft_sigma_rel`, `mt_sigma_rel`, `mm_half_spread_bps`, `mm_inventory_skew_bps`) are **V-relative / basis points** rather than absolute dollars | Regime-invariant calibration: same params work for V≈$92 (stressed) and V≈$179 (calm). Fixed +64% stress amplification observed under absolute-cents scales |
| D17 | CIR stochastic volatility (Heston/Gao 2023) added on top of Merton jump-diffusion (`xi_v=0` falls back to constant σ_v) | Stage 8 prep: long-memory vol clustering (Cont 2001 fact #3 at long lags) needs vol-of-vol |
| D18 | Population restructured to **15 BCM + 5 NBCM + 78 clients = 98 agents**, with **0 direct ZI/FT/MT** (every non-CM trader is a client of some CM) | Real-exchange-tier mimicry: every trade has a clearing path, no exogenous-noise floats |
| D19 | ZI order quantity scaled by current vol per Gao 2023: `qty_max_eff = round(zi_qty_max · √(v_var/θ_v))`, capped at 10× | Gao 2023 noise-demand spec `D_N = σ · ζ`; amplifies Cont fact #3 in high-vol regimes; no-op when CIR disabled |
| D20 | ODD-faithful jumps **enabled by default**: jump_lambda = 0.0385 (= 3/78), jump_mean = 0, jump_std = 0.01 | ODD §Stochasticity (3 jumps per 510-tick day = 3/78 per 5-min step); applies to both regimes |
| D21 | Variable client-book composition + heterogeneous client cash (institutional FT/MT vs retail ZI) | Real-exchange tiering: dealer banks clear for both institutional (~$1-10M cash) and retail (~$10-100k cash) |
| D22 | Order-quantity range **split** between agent types: `zi_qty ∈ U[1,10]` (Poisson cadence-invariant) but `dir_qty ∈ U[5,50]` (= ODD U[1,10] × dt_minutes=5) for FT/MT/BCM-FT/clients | FT/MT/CMs are deterministic 1-order-per-step submitters, so per-minute volume scales as 1/dt_minutes. To preserve ODD's per-minute volume target (calibrated at 1-min cadence), directional-agent qty is multiplied by dt_minutes. ZI doesn't need this because Poisson(α·dt) arrivals already preserve per-minute order count. |

---

## 6 · Open questions / planned design decisions

- **`calibrate.py` is no longer legacy** — fully rewritten for WRDS Intraday Indicators data. Direct calibration extracts σ_v from empirical OC std (preferred over IVol_t_m due to microstructure-noise bias). LHS handles indirect calibration.
- **Big LHS run pending on user's laptop** (200 points × 90 days × 3 seeds, ~15 min wall). Will pin final calibrated parameters for current 98-agent topology.
- **BCM market-maker mode is implemented** (Stoikov 2008 with V-relative basis points). MM inventory limit (HFABM-style hard cap) not yet implemented — tracked as Stage 9 extension.
- **Vol clustering signature** present but undershoots target at default params (`ACF(|r|,1) = +0.062` vs target +0.164); LHS recalibration on the new 98-agent + jumps + Gao vol-scaling structure should improve this.
- **Long-lag vol clustering** under stress (`ACF(|r|, 5) = +0.275` empirical) requires CIR to be active with non-trivial xi_v. Calibration target.
- **Margin layer (Stage 4)** is the next functional addition once calibration locks. Will introduce variation-margin cash transfers every 60 ticks (12 5-min steps), at which point cap_ratios will actually compress and breaches become possible.
- **Kalman+EM comparison (Majewski 2018)** for comparison with LHS surrogate matching: Stage 8b.

---

## 7 · WRDS data notes

`data/thesis_data_calm.csv` (calm: 2013-01-02 → 2014-12-31, 503 rows) and
`data/thesis_data_stressed.csv` (stressed: 2008-09-02 → 2009-03-31, 146 rows).

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

Stressed period covers the GFC trough: SPY range $67.97–$130.06 (48% drawdown captured), daily log-return std 3.41% (vs calm 0.70%), max abs daily move 10.88%.

---

## 7b · HFABM (Cao et al. 2024 JASSS) extensions to consider

[High-frequency financial market simulation and flash crash scenarios analysis](https://www.jasss.org/27/2/8.html) (arXiv [2208.13654](https://arxiv.org/abs/2208.13654)) — millisecond-cadence ABM for E-mini S&P 500. Two design choices differ from ours and are worth flagging as extensions:

**MM inventory limit** (HFABM §3.2). MM posts limits during normal trading; when |inventory| exceeds a hard limit, MM submits market orders to liquidate. Different from our cap-ratio fire-sale (which only fires below 8% Basel floor, very rarely). HFABM shows the inventory limit drives mini-flash-crash amplitude. Could add `mm_inventory_limit` parameter.

**Fundamental trader as market-order-only** (HFABM §3.3). Their FT submits market orders only — opposite of our ODD-faithful FT-as-limit-orders. More HFT-realistic; would interact more aggressively with MM quotes. Worth implementing as alternative `FundamentalTrader_HFABM` if Stage 8 calibration plateaus.

**ML surrogate calibration** (HFABM §4.2). Neural-network surrogate trained on grid evaluations, replacing thousands of objective evaluations with one forward pass. Our LHS is simpler; this is a documented upgrade path if calibration becomes a bottleneck.

---

## 8 · Roadmap of planned extensions (post-Stage-8)

- **Volatility traders** — Gao 2023 explicit ν_t-driven trader (we already have CIR ν_t but no dedicated agent reacting to it). Demand `∝ ζ · ν_t`.
- **Almgren-Chriss liquidation** — distressed-CM fire-sales with optimal slicing → measurable temporary price impact.
- **Basel III deleveraging asymmetry** — partially modelled in Stage 3 (BCM fire-sale vs NBCM stop-out); Stage 4 will add proper margin-call cadence.
- **Client clearing delay** — NBCM lag in passing margin calls to clients → transient funding gaps.
- **Strategy switching (Gao 2023)** — agents switch between FT/MT/ZI based on recent PnL.
- **Cover-2 default fund** (Stage 5) — DF sized to cover 2 largest CM defaults; daily recalc; pro-rata contributions.
- **CoMargin** — alternative IM scheme, compare to SPAN.
- **MM inventory limit** (HFABM) — hard cap that triggers market-order liquidation.
- **Multi-scale MTs** (Krishnen) — fast (high-η) + slow (low-η) MT subpopulations.
- **HFABM-style market-order-only FT** — alternative trader spec for comparison.

---

## 9 · How to run

```bash
# Run the model
python run_simulation.py            # writes output/stage*_run.csv

# Tests
python -m unittest tests.test_stage1 tests.test_stage2 tests.test_stage3 -v

# Calibration (XGBoost surrogate)
python calibrate.py verify          # WRDS unit-hypothesis sanity check
python calibrate.py direct          # v0, sigma_v, mu_v from data
python calibrate.py run 200 90 3    # end-to-end: LHS + XGB + L-BFGS + validate
                                    # ~15 min on laptop; writes
                                    #   output/calibration_lhs.csv
                                    #   output/calibrated_params.json
                                    #   output/calibration_feature_importance.csv
```

`run` requires `pip install xgboost scipy` on your laptop. `verify` and `direct` work without these.

`analysis.ipynb` is the Stage-8 calibration-progress notebook (26 cells). Open in Jupyter and run-all. It runs the simulation under both calm AND stressed regimes (fix-agents-swap-environment), computes Cont 2001 stylized-fact moments for each, and visualises:

1. Empirical reference data — calm 2013-14 vs stressed Sept 2008 - Mar 2009
2. Empirical Cont 2001 target moments
3. Both-regime sim (60 days × 3 seeds averaged)
4. **Side-by-side moment comparison** with sim/target ratios — headline diagnostic
5. Mid-vs-V plots, both regimes
6. Return distribution histograms (sim overlaid on empirical)
7. ACF(r) and ACF(|r|) overlays (both regimes, both lags 0–20)
8. CIR variance trajectory under each regime
9. MM inventory + quote dynamics (HFABM-style)
10. CM cap_ratio plots (BCM split by mode, NBCM)
11. End-of-run agent state by type
12. **Calibration progress** — auto-loads `output/calibration_lhs.csv` (LHS training data) and `output/calibration_feature_importance.csv` (XGBoost per-(regime, moment) importance heatmap) if present. Cell 1 also auto-loads `output/calibrated_params.json` so the rest of the notebook simulates at the calibrated θ rather than pre-calibration defaults.

---

## 10 · Maintenance protocol for this file

When code changes, update:
- §2 component description if behaviour changed
- §3 parameter table if `ModelParams` changed
- §4 status / annual-run numbers if a new stage or run is logged
- §5 deviations table if any new deliberate departure from ODD (D1–D21 currently)
- §6 open-questions list when a design choice locks or a new one opens
- §8 if an extension is moved on-path or off-path

Keep tone terse and citation-anchored. This is an AI/collab context document, not a tutorial.
