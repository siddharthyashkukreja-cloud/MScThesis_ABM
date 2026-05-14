# claudereadme.md — internal model state-of-affairs

> Living reference. Updated whenever code changes. Separate from README.md
> (which Sid maintains for thesis-facing documentation).

**Last sync:** Design refactor (May 2026). Agent typology re-spec'd to a clean
ZI + FT + MM + (BCM/NBCM/clients) baseline. Momentum trader parked as future
extension. V_t becomes an exogenous time series (Merton jump-diffusion
calibrated offline via Lee-Mykland, pending) replacing in-sim GBM/CIR
dynamics. Mid-anchored geometric ZI depth (Cont-Stoikov / Bouchaud), V_t +
z·σ_fundamental FT reservation (ODD §Agents), single-quote uniform-depth MM
(HFABM Cao 2024 §3.2 flavour). Production code is being brought into sync
with this spec.

---

## 0 · Project frame

Pure-Python ABM of a CCP-cleared single-asset market (ES front-month futures),
built to study CCP systemic-risk dynamics, margin procyclicality, and contagion
under calm vs stressed regimes. Single LOB with batched call-auction clearing
at 5-min cadence (78 steps per RTH 6.5h trading day).

Reference priority chain (every design decision cites one; deviations flagged):
1. Simudyne CCP Risk Model **ODD** (`Simudyne_CCPRiskModel_ODDDoc.md`)
2. Deloitte / Simudyne CCP Resilience paper
3. Majewski, Ciliberti, Bouchaud (2018) — Extended Chiarella estimation
4. Gao et al. (2023) — Deep hedging / stochastic-vol Chiarella
5. Krishnen — ABM Liquidity Risk paper
6. Cont & Stoikov (2008) — Stochastic LOB models
7. Farmer, Patelli, Zovko / Daniels (2003, 2005) — ZI baseline
8. Cao et al. (2024) HFABM — flash-crash ABM, MM spec

Supporting (current refactor):
- Merton (1976) — jump-diffusion fundamental
- Lee & Mykland (2008) — jump detection statistic
- Barndorff-Nielsen & Shephard (2004) — bipower variation
- Bouchaud, Mézard, Potters (2002) — mid-anchored empirical depth profile

---

## 1 · Repository layout

```
MScThesis_ABM/
├── claudereadme.md
├── README.md
├── data/
│   ├── data.py            # DataBento ingest (OHLCV-1m, BBO-1m, MBO-10)
│   ├── roll.py            # ES front-month roll + 5-min resample
│   ├── glbx-*.csv.zst     # DataBento raw (zstd)
│   ├── processed/         # rolled front-month series
│   ├── fv_calm.csv        # daily-forward-filled fundamental, 2019 (Kalman placeholder)
│   └── fv_stressed.csv    # daily-forward-filled fundamental, 2020-02-24 → 2020-04-02
├── Kalman_FV.ipynb        # placeholder smoother, to be retired
├── model/
│   ├── globals.py         # ModelParams + SimContext
│   ├── lob.py             # Order, Fill, LOB
│   ├── agents.py          # BaseTrader, ZI, FT, MM, BCM, NBCM
│   └── simulation.py      # Simulation driver
├── tests/                 # stage1/2/3 — pending rewrite to new spec
├── run_simulation.py      # entry point + build_traders()
├── calibrate.py           # pending re-spec to new 5-D agent search
├── analysis.ipynb         # calibration-progress notebook
└── output/
```

---

## 2 · Agent specification (locked)

All non-MM agents share signature `submit_orders(lob, params, ctx, rng)`.
SimContext provides exogenous V_t, prev-step mid, tick, traders-by-id lookup.

### 2.1 Zero-Intelligence Trader (Cont-Stoikov 2008 + Bouchaud 2002)

Per step, three independent Poisson processes:

| Event | Rate (per min) | Per-step semantics |
|---|---|---|
| Limit-order arrival | `zi_alpha` | `n_limits ~ Poisson(zi_alpha · dt_minutes)` |
| Market-order arrival | `zi_mu` | `n_markets ~ Poisson(zi_mu · dt_minutes)` |
| Per-resting cancellation | `zi_delta` | per resting order, Bernoulli with `p = 1 − exp(−zi_delta · dt_minutes)` |

For each limit arrival: side ~ Uniform{±1}; depth `k ~ Geometric(p_zi)` ticks, k ≥ 1; price = `mid ± k · tick_size`; qty ~ U[qty_min, qty_max]. Empty-book fallback: anchor = V_t. Default LOB TTL applies.

For each market arrival: side ~ Uniform{±1}; qty ~ U[qty_min, qty_max].

Cancellation is per-resting-order, not arrival-rate-driven — different process semantics.

**True zero intelligence**: no signal dependence in side or placement.

### 2.2 Fundamental Trader (ODD §Agents)

Per-agent persistent `z_score ~ N(0, 1)` at init. Per step:

| Step | Behaviour |
|---|---|
| Reservation | `R = V_t + z_score · σ_fundamental` |
| Dead-band check | if `|R − mid| ≤ ft_threshold_bps · V_t / 10000` → skip (no submission this step) |
| Side | `+1` if `R − mid > +threshold`; `−1` if `R − mid < −threshold` |
| Arrivals | `n_arrivals ~ Poisson(ft_alpha · dt_minutes)` (sampled after threshold pass) |
| Per arrival | submit limit at `R`, qty ~ U[qty_min, qty_max], per-order TTL ~ U{1, ft_ttl_max} |

FT limits can be marketable when `|R − mid|` is large — `lob.match()` clears the cross at end of step. The dead-band avoids submitting orders that would persist for U{1, ft_ttl_max} steps and clear near mid (= near V_t) without informational content. No active cancellation rate.

**Trigger is `|reservation − mid|`, not `|V_t − mid|`.** A collective `|V_t − mid|` trigger was implemented and rejected during verification: it synchronises every FT onto one side when V_t drifts, which empties a book side and decouples mid from V. The `|reservation − mid|` form splits FTs by z-sign (high-z lean buy, low-z lean sell), keeping order flow two-sided. See D9c.

### 2.3 Market Maker (HFABM Cao 2024 §3.2 — stochastic single-quote variant)

Per step:

| Step | Behaviour |
|---|---|
| Cancel | All MM resting quotes from prior step cancelled |
| Inventory check | If `\|inventory\| > mm_inventory_limit`: market-order liquidate to `mm_inventory_safe`; skip quoting this step |
| Quote bid | `d_bid ~ U{0, 1, ..., mm_p_edge}`; price = `mid − d_bid · tick_size`; qty = `mm_qty` |
| Quote ask | `d_ask ~ U{0, 1, ..., mm_p_edge}` (independent); price = `mid + d_ask · tick_size`; qty = `mm_qty` |

Single quote per side per step. Random tick-offset spread. Empty-book fallback: anchor = V_t.

### 2.4 Banking Clearing Member

Mode: `'fundamental'` (delegates to FT) or `'market_maker'` (delegates to MM). May carry a client book (`client_ids`). Per step:

1. `cap_ratio = cash / [(|own_inv| + Σ_clients |client_inv|) · mid]`.
2. If `cap_ratio ≤ cap_ratio_floor` (0.08, Basel III): fire-sale own inventory via market orders in U[qty_min, qty_max] chunks; skip delegate.
3. Otherwise delegate.

Cap-ratio guard runs before the MM inventory-limit check inside the MM delegate path.

### 2.5 Non-Banking Clearing Member

No-op `submit_orders`. Cap-ratio:

    cap_ratio_NBCM = cash / [Σ_clients |client_inv| · mid]

Cap-ratio breach response activates at Stage 4 (margin); currently inert.

### 2.6 Parked — future extensions

- **Momentum Trader** (Majewski 2018 chartist; EWMA momentum signal; possibly market-order spec for HFABM-flavour).
- **ZI vol-dependent demand** (Gao 2023 `D_N ∝ ζ · σ_t` scaling on Heston-like vol).
- Both retired from current population; class stubs may remain in `agents.py` for future re-activation.

### 2.7 SimContext

```python
@dataclass
class SimContext:
    v: float                 # exogenous V_t this step
    mid_price: float         # prev-step post-match mid (NaN at t=0)
    tick: int
    traders_by_id: dict      # CMs look up clients
```

No `momentum` field. No `v_var` field. No GlobalState.

### 2.8 LOB

Price-time priority, FIFO at each level, `match()` clears any cross. `add_limit(..., ttl=None)` accepts a per-order TTL override (used by FT for U{1, ft_ttl_max}); default falls back to `self.order_ttl`. `is_resting(oid) → bool` helper for trader-side stale-oid pruning.

`last_price` tracks the most recent fill price. `_update_quotes()`: when both book sides are populated, `mid = (best_bid + best_ask) / 2`; when one or both sides are empty, `mid` falls back to `last_price` (not to the surviving side's best limit — that "teleports" mid onto an arbitrary FT reservation when aggressive flow empties a side).

---

## 3 · Parameters

### 3.1 Structural (ODD-fixed, not calibrated)

| Name | Value | Source |
|---|---|---|
| `tick_size` | 0.25 | CME ES |
| `dt_minutes` | 5.0 | sim cadence |
| `order_ttl` | 2 | default LOB TTL (≈ 10 min); ODD §Mech #7 |
| `ft_ttl_max` | 10 | FT per-order TTL ceiling (≈ 50 min) |
| `ft_threshold_bps` | 50.0 | FT dead-band: skip when `|R − mid| ≤ 0.5% · V_t` |
| `qty_min, qty_max` | 1, 10 | ODD §Stochasticity (preserved across cadences via Poisson arrivals) |
| `mm_qty` | 50 | per-quote size; HFABM default |
| `mm_p_edge` | 4 | MM tick-depth ceiling; matches HFABM mm_n_levels=4 span |
| `mm_inventory_limit` | 1000 | HFABM default |
| `mm_inventory_safe` | 500 | post-liquidation target |
| `cap_ratio_floor` | 0.08 | Basel III |
| `clients_per_book` | 6 (3 FT + 3 ZI; MT slots reallocated) | — |
| `λ` (V_t jump intensity) | 0.0385 / step | ODD §Stochasticity (3 / RTH day) |

### 3.2 Data-calibrated offline (per regime, fixed before agent loop)

| Name | Method | Status |
|---|---|---|
| `v0` per regime | empirical opening price | done (calm 2463.00, stressed 3249.25) |
| `σ_v` per regime | RTH 5-min log-return std on non-jump bars (BNS 2004) | done (calm 6.28e-4, stressed 3.84e-3) |
| `m, s` (jump mean, std) per regime | Lee-Mykland 2008 flagged bars | done (calm m=−0.00117, s=0.00383; stressed m=+0.00291, s=0.02647) |
| `λ_emp` diagnostic per regime | Lee-Mykland count / total bars | done (calm 0.0028/step, stressed 0.0031/step — both ~0.22–0.24 jumps/RTH-day, ~12× lower than ODD's 3/day) |
| `p_zi` per regime | Geometric MLE on MBO-10 placed-distance distribution (mid-anchored) | pending |

JSON output: `output/v_jd_params.json`. CLI: `python data/v_jd.py {calibrate,generate,generate-all}`. V_t paths written to `data/fv_{regime}.csv` (drop-in replacement for Kalman placeholder).

### 3.3 Agent calibration loop (per regime unless noted)

| Name | Description | Bound (initial) |
|---|---|---|
| `zi_alpha` | ZI limit-order Poisson rate per minute | [0.05, 1.0] |
| `zi_mu` | ZI market-order Poisson rate per minute | [0.005, 0.2] |
| `zi_delta` | ZI per-resting cancellation rate per minute | [0.005, 0.2] |
| `ft_alpha` | FT limit-order Poisson rate per minute | [0.05, 2.0] |
| `ft_sigma_c` (regime-invariant) | σ_fundamental = `ft_sigma_c · σ_v · v0` | [1, 100] |

9 free values total across both regimes (4 per-regime + 1 shared `ft_sigma_c`).

**Non-degenerate operating point** (pre-calibration default in `run_simulation.py`): `ft_alpha=0.5`. At the original `ft_alpha=0.05` the FT layer cannot generate enough flow to move mid against 8 MMs × 50 depth — mid decouples from V (corr ≈ 0, mid/V std ratio ≈ 0.06). At `ft_alpha=0.5` mid tracks V (calm corr 0.93, mid/V ratio 1.28; stressed corr 0.88, ratio 0.56). The MM/FT flow balance — `ft_alpha` vs `mm_qty · n_bcm_mm` — is the key thing the agent-calibration loop must pin down.

### 3.4 Population (real-bank-tier, 98 agents)

| Group | Count | Behaviour |
|---|---|---|
| BCM-MM with clients | 8 | HFABM stochastic-single-quote MM; clears for 6 clients each |
| BCM-FT-prop, no clients | 7 | FT spec on own account |
| NBCM with clients | 5 | No own trading; clears for 6 clients each |
| FT clients | 39 | 3 per book × 13 books |
| ZI clients | 39 | 3 per book × 13 books |

Cash per ODD §Initialization: BCM ~U[5B, 10B]; NBCM ~U[5M, 10M]; FT clients ~U[1M, 10M]; ZI clients ~U[10k, 100k].

---

## 4 · Stage history

| Stage | Status |
|---|---|
| 1 — ZI baseline | ✅ legacy |
| 2 — + FT, MT (parked), jumps/CIR (retired) | superseded |
| 3 — + BCM/NBCM/MM/clients | ✅ logic intact; agent specs updated by current refactor |
| **Design refactor (May 2026)** | in progress (this doc) |
| 4 — Margin (IM + VM, 60-tick cadence) | pending |
| 5 — Default fund (Cover-2 EMIR) | pending |
| 6 — 5-level waterfall + position auction | pending |
| 7 — Almgren-Chriss distressed liquidation | pending |
| 8 — Calibration (XGBoost surrogate on 5-D agent space) | pending |

Tests (`tests/`): stage1/stage2 require rewrite against new spec; stage3 mostly retains validity. Calibration code in `calibrate.py` targets the old parameter set and needs re-spec.

---

## 5 · Deviations from ODD (locked)

| # | Deviation | Reason |
|---|---|---|
| D1 | Single ES LOB (vs ODD's dual FTSE 100/250) | Data is ES-only |
| D2 | 5-min steps × 78/day (vs ODD's 1-min × 510/day) | Computational tractability; aligns with RTH 6.5h |
| D3 | ZI placement: geometric depth from mid (vs ODD unspecified) | Cont-Stoikov 2008 / Bouchaud 2002 |
| D4 | ZI arrivals: `Poisson(rate · dt_minutes)` counts per step (multiple per step possible) | Continuous-time-faithful; linear `rate·dt` Bernoulli undercounts at our cadence |
| D5 | ZI cancellation is per-resting Bernoulli `1 − exp(−zi_delta · dt_minutes)` | Cont-Stoikov continuous-time |
| D6 | FT places at `V_t + z · σ_fundamental` (ODD §Agents), `z ~ N(0,1)` per agent | ODD-faithful |
| D7 | `σ_fundamental = ft_sigma_c · σ_v · v0` (per regime, single shared `ft_sigma_c`) | Regime-invariant calibration; ties FT noise to JD V_t vol |
| D8 | FT has Poisson `ft_alpha` arrival gate; no market orders | Patient Chiarella fundamentalist |
| D9 | FT per-order TTL ~ U{1, ft_ttl_max} (vs LOB-level fixed) | Heterogeneous order lifetime without active cancellation rate |
| D9b | FT dead-band: skip submission when `|reservation − mid| ≤ ft_threshold_bps · V_t / 10000`; default 50 bps | Avoid persisting orders near V_t that execute uninformatively; pragmatic addition not in ODD or priority chain |
| D9c | FT trigger on `|reservation − mid|` (not `|V_t − mid|`) | Verification showed a collective `|V_t − mid|` trigger synchronises all FTs onto one side and empties a book side; `|reservation − mid|` splits FTs by z-sign and keeps flow two-sided |
| D9d | LOB mid falls back to `last_price` when a book side empties | Prevents mid teleporting onto an arbitrary surviving resting limit; robustness fix surfaced during D9c verification |
| D10 | MM = single quote per side per step at d ~ U{0, ..., mm_p_edge} | HFABM Cao 2024 §3.2 flavour; user spec; stochastic spread variant of HFABM ladder |
| D11 | V_t is exogenous, loaded from CSV; no GlobalState evolution in sim | Decouples V calibration (offline, one-shot) from agent calibration |
| D12 | V_t generated offline via Merton JD calibrated by Lee-Mykland | ODD §Mechanism #9; currently Kalman-smoothed placeholder pending JD |
| D13 | MT and ZI vol scaling parked | Future extensions; baseline focuses on FT + ZI + MM dynamics |
| D14 | Client-clearing tier (78 client traders attached to CMs) | User extension; ODD has no client tier |
| D15 | Some BCMs are MM-mode with clients (GS/JPM model) | Real-bank-tier; partial HFABM precedent |
| D16 | NBCMs do not trade on own account | User clarification of ODD |
| D17 | Client books = 3 FT + 3 ZI per book | MT parked |
| D18 | Calibration ordered before margin (Stage 8 before Stage 4) | Lock market dynamics before adding capital-constraint feedback |

---

## 6 · Open items

- **MM/FT flow balance**: the single biggest pre-calibration knob. `ft_alpha` vs `mm_qty · n_bcm_mm` determines whether mid tracks V. Current default `ft_alpha=0.5` is a hand-found non-degenerate point, not calibrated. The agent-calibration loop must pin this against empirical return moments.
- **High return kurtosis**: smoke runs show mid-return kurtosis ≈ 130–150, far above empirical 5-min ES (~10–15). Driven by (a) ODD λ over-representing jumps in V_t, (b) LOB microstructure amplification. Expected to compress under calibration; watch closely.
- **`p_zi` calibration**: from MBO-10 book snapshots — fit Geometric to placed-distance distribution from mid. Currently using placeholder `p_zi=0.30`.
- **Tests**: stage1/stage2 need rewrite against new spec; stage3 mostly intact.
- **`calibrate.py`**: existing XGBoost surrogate pipeline targets old parameter set; needs re-spec to current 9-value loop.
- **Margin (Stage 4)**: next functional addition after calibration locks.

Closed:
- ✅ Lee-Mykland V_t calibration → `data/v_jd.py calibrate`
- ✅ JD path generator → `data/v_jd.py generate{,-all}`
- ✅ Smoke test with dynamic V_t — FT firing across all 39 agents both regimes; mid tracks V (calm corr 0.93, stressed 0.88); inventory conserves to 0; no NaN mids.
- ✅ λ identification: locked at ODD's 0.0385/step (= 3/RTH day). L-M finds ~0.25 jumps/RTH-day empirically (~12× lower); documented as deviation. σ, m, s remain regime-calibrated.
- ✅ FT dead-band threshold → D9b (default 50 bps of V_t).
- ✅ FT trigger form → D9c (`|reservation − mid|`, not collective `|V_t − mid|`).
- ✅ LOB empty-side mid robustness → D9d (`last_price` fallback).

---

## 7 · Data notes

- **Source**: DataBento, CME Globex MDP3 feed. Schemas: OHLCV-1m, BBO-1m, MBO-10 (separate storage).
- **Roll**: ES quarterly (Mar/Jun/Sep/Dec); switch 8 calendar days before expiry (`data/roll.py`).
- **RTH**: 13:30–20:15 UTC (08:30–15:15 CT). Overnight Globex excluded from calibration.
- **Regimes**:
  - **Calm**: 2019 full year (~252 trading days). V₀ ≈ 2501.
  - **Stressed**: 2020-02-24 → 2020-04-02 (COVID trough, ~28 days). V₀ ≈ 3234, range ~$2200–$3400.
- **V_t source**: currently Kalman-smoothed daily forward-filled to 5-min; placeholder for Merton JD paths once Lee-Mykland calibration is implemented.

---

## 8 · How to run

```bash
python run_simulation.py
python -m unittest tests.test_stage1 tests.test_stage2 tests.test_stage3 -v   # pending rewrite
```

`calibrate.py` and `analysis.ipynb` will be re-spec'd when calibration scope is reopened.

---

## 9 · Maintenance protocol

When code changes, update:
- §2 if agent behaviour changed
- §3 if `ModelParams` changed
- §4 if a stage is closed or a new refactor opens
- §5 if a new deliberate deviation is added
- §6 when an open item closes or a new one opens
- §8 if instructions change

Keep tone terse and citation-anchored.
