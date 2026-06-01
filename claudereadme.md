# claudereadme.md — internal model state-of-affairs

> Living reference. Updated whenever code changes. Separate from README.md
> (which Sid maintains for thesis-facing documentation).

## Current state (D51 pending — May 2026)

**Roster (54 LOB + 5 NBCM + 1 CCP off-LOB):**
10 FT + 10 BCM (FT-cast, own-account) + 10 MT (single cohort) +
20 ZI + 4 MM (HFABM mid-anchored) on the LOB; 5 NBCM + 1 CCP off-LOB.
5 of 10 BCMs carry client books (plain FT + MT round-robin); ZI direct.

**V_t (D33):** Stein-Stein SV Merton jump-diffusion. σ_t Vasicek-OU,
`α ≈ 0.0125/min` (1/α ≈ 80 min). θ re-anchored so `E[σ_t²] = σ_d²`.
`λ = 3/day` ODD-pinned; `δ` data-calibrated. σ_t piped to FT belief
width (D34: `σ_fund_t = √390 · σ_t · v0`).

**Calibration (`calibrate.py`):** 6-d θ per regime —
`{depth_mean, depth_sigma, zi_alpha, zi_mu, zi_delta, mm_p_edge}`.
HFABM surrogate-SMM: LHS → XGBoost surrogate → active learning → L-BFGS-B
→ stage-2 true-grid → validate. 4-component loss (Hill + V + ACF1 + ACF2),
Franke inverse-bootstrap-variance weights. ACF1 lags {1,5,10}; ACF2 lags
{1,2,5,10,15,30} (D45). Kurt diagnostic-only.

**Pinned (not in loop):** `ft_alpha = mt_alpha = 1.0` (D36); `mt_lambda
= 0.05`, `mt_mu = 0` (D40, D44); `ft_sigma_c = √390` (D34); `qty_max
= 10` (D47); `mm_qty = 2`, `mm_skew = 0.05`.

**D50 — reporting/notional relabeling:** Each model qty unit = 50-contract
institutional ES block (`VOLUME_LOT = 50`); CME ES point multiplier
`CONTRACT_USD = 50`. Applied at `client_notional`, `BCM/NBCM.capital_ratio`,
and volume-reporting layer only. **LOB dynamics, log-returns, ACFs and
Hill are invariant** under this multiplicative relabeling. With it, calm
1-day model volume = 3,008/min (empirical 2,524/min ✓).

**Latest calibration (D48, stressed only):**
validated D = 21.67. ΔHill 9.53, ΔV 0.55, ΔACF1 2.60, ΔACF2 8.98.
4 MMs + calibrated `mm_p_edge` did NOT close the Hill gap (1.13 vs
target 3.25). MMs help V and ACF1 marginally and the volume mean
materially; tails remain V_t-side limited.

**Calm θ** in `globals.CALIBRATED` is the pre-D48 5-d overnight result
(D = 27.5). Needs a 6-d re-run after the model lock.

### Structural limits (documented, not solved)

1. **Hill tail-index (~9.5 SD short stressed).** ODD `λ = 3/day` + D25
   jump-share cap `f ≤ 0.95` over-attribute kurtosis to jumps. Fix
   candidates (untried): `λ ↓ 1`, `f_cap ≤ 0.5`. Cite ABD 2007.
2. **Long-lag |r| ACF (1-min lags ≥ 30 and daily lag-1).** Single OU σ_t
   has one timescale; empirical equity vol has multi-scale memory (HAR-
   RV, ABDL 2001/3). D46 α_mult override hit target but the optimizer
   exploited a "thin-trading clustering" escape that broke V; reverted.
   Owned as Cont 2005 structural limit.
3. **VM cycle units (D51 pending).** `_margin_cycle` line 159 in
   `simulation.py` uses raw `book_pos × Δmid` — not USD post-D50. One-
   line fix unlocks Stage 4 contagion: `vm *= VOLUME_LOT · CONTRACT_USD`.
4. **CM-layer roster too thin.** 1-2 clients per BCM, homogeneous
   `BCM_CASH_RANGE = U[5B, 10B]`. Needs synthetic-client cohort and
   heavy-tailed BCM cash before Mech #2 can fire.

### Decision log (top — full history below)

- **D51 (pending) —** USD-denominate `_margin_cycle` VM.
- **D50 —** `VOLUME_LOT = 50` institutional-lot relabeling for volume +
  capital ratios. Eisler-Bouchaud-Kockelkoren 2012 ref. LOB dynamics
  unchanged.
- **D49 —** Bounded-Pareto `_draw_qty` (Gabaix-Plerou α≈1.5). Tested
  + reverted — discrete LOB has linear impact, so Gabaix theorem
  doesn't fire; power-law sizes degraded Hill.
- **D48 —** Reintroduce 4 HFABM mid-anchored MMs; `mm_p_edge` enters
  the loop as 6th θ. Calibrated D = 21.67 stressed; Hill gap unchanged.
- **D47 —** `qty_max` stressed reverted 5 → 10 (match ODD baseline +
  scale CM-layer exposure).
- **D46 —** α_mult = 0.1 V_t persistence override. Tested + reverted —
  optimizer found "thin-trading clustering" loophole; broke V. Long-
  lag ACF gap owned as Cont 2005 structural.
- **D45 —** ACF2 lag set refocused to {1,2,5,10,15,30} (short-lag).
- **D44 —** Pin `mt_lambda = 0.05`, drop VolatilityTrader (D38b) +
  ContTrader (D39). Lean roster: FT/BCM/MT/ZI/MM only.

(Older D-rows preserved below for reference. The above is the
current architecture.)

---

**(D33 — May 2026)** **V_t becomes a Stein-Stein-style stochastic-vol
jump-diffusion** — Vasicek-OU on σ_t (the Deloitte/Simudyne CCP-model
architecture). `data/v_gbm.py` rewritten: σ_t mean-reverts to θ_eff with
speed α and per-step shock σ_vol; V_t evolves with stochastic σ_t plus
the existing Merton jumps. Method-of-moments calibration on 30-min
realised-vol time series gives α ≈ 0.013/min (~55-min half-life) and
ρ_1(30m) ≈ 0.68 — strong persistence. V_t paths now exhibit the long-
absent volatility clustering at the fundamental level: calm `acf_absr`
1/10/30 = 0.25/0.23/0.19 vs empirical 0.31/0.25/0.21 (single OU decays
too fast at lags ≥ 60 — a two-factor SV could close that). The D31
mid-anchored MM is the channel that should transmit this to the mid.
See D33 + §3.2.

**(D31/D32 — May 2026)** **MarketMaker re-introduced (D31)** as HFABM-style
mid-anchored at low qty (`n_mm = 2`, `mm_qty` 9th calibrated parameter,
bound `(1.0, 5.0)`). Reverts D21's MM drop and supersedes the D10g V_t
anchor that pinned the mid. The mechanism: continuous near-mid liquidity
absorbs market orders so the book doesn't walk between V_t-jump steps —
the static-then-jump return pattern that drove kurt ~100-500 is broken.
Prototype sweep (`outputs/test_mm.py`) shows calm 2 MMs @ `mm_qty=1` cut
kurt 126→44 and lifted Hill from 2.49 to 3.10 (≈ target 3.0). HFABM (no
skew) preferred over Avellaneda-Stoikov skew (Skew + low qty introduced
positive `acf_r_1` of +0.5–0.7 stressed). **ΔACF2 reinstated (D32)** —
4-component loss — since the MM is the continuous-quoting mechanism
Deloitte/Simudyne assume when calibrating volatility clustering. See
D31 + D32 + §2.4.

**(D29 — May 2026)** The **margin stage is live** — `Simulation._margin_cycle`
runs every 60 ticks (ODD §Step Sequence step 11): variation margin settles
each clearing member's cleared-book mark-to-market into cash, IM/MM are
recomputed, a margin call is flagged when `|VM| > 5%·IM`, and the capital
ratio is recorded. Clearing members switch to **futures-margin accounting**
(a fill posts only the position; cash is settled by VM — ES is a future).
5 of the 10 BCMs carry client books (`n_bcm_with_clients`); the rest are
own-account only. Per-cycle observables → `output/clearing_{regime}.csv`,
read by `clearing_analysis.ipynb`. With ODD-scale CM capital the 8% floor
does not bind in calm (by design). See D29 + §2.5.

**(D28 — May 2026)** The **central-clearing tier is scaffolded** — the
thesis client-clearing extension of the Simudyne CCP ODD. 10 FTs are
re-cast as `BankingClearingMember` (FT subclass — trades own account);
5 `NonBankingClearingMember` are added (pure clearing intermediaries — no
own position, no LOB orders); 1 `CentralCounterparty` (`model/clearing.py`).
Each CM carries a `BalanceSheet`; CM↔CCP and client-book links are wired
(plain FT + MT as clients, ZI direct). Only the STRUCTURE is built —
margin calls, the cover-2 default fund and the 5-level waterfall are
deferred to later stages. The BCM trades own-account FT-style, so the
8-d market-layer calibration is unchanged (`calibrate.py` keeps
`n_bcm = 0`). See D28 + §2.5.

**(D27 — May 2026)** The **MomentumTrader gains a market-order branch** —
on activation it submits a trend-direction marketable order w.p. `mt_mu`
(a separate Bernoulli rate, parallel to ZI `zi_mu`; not the D13c k≤0
cliff) alongside its passive limit w.p. `mt_alpha`. Trialled as a
clustering mechanism — it does **not** deliver clustering (`acf_absr`
lifts at lag 1 only), so D26's accepted limitation stands. It is kept as
a calibrated parameter because the test sweep showed real traction on the
tail moments: excess kurtosis falls toward empirical and the Hill index
rises (lighter tails). Theta is now **8-d per regime**. See D27.

**(D24/D25 — May 2026)** V_t is now a **Merton jump-diffusion** — GBM +
Poisson(λ=3/day) jumps, σ_d/δ calibrated from the ES variance + kurtosis
(the ODD §Stochasticity structure, reverting D12). A **VolatilityTrader**
was re-introduced to chase volatility clustering and then **removed**: in
four tested forms it never produced clustering — random order flow washes
out, σ̂-scaled liquidity damps (D24). The jumps give V_t the right excess
kurtosis but did not induce clustering at the mid either — the no-MM
microstructure layer's own i.i.d. bursts dominate. **Volatility clustering
is confirmed structurally unreached** (≈7 mechanisms tried — D25, §6) and is
deferred to the Stage-4 fire-sale feedback.

**(D21/D23 — May 2026)** The MarketMaker is **dropped**. The whole MM-redesign
arc (D10–D10g) ended here: at the 1-min cadence 4 MMs unavoidably *became* the
price-setter — a V_t-anchored MM pinned price to V_t (no agent traction, the
calibration was degenerate); a last-mid-anchored MM froze the price. Removing
the MM lets FT/MT/ZI order flow drive price (the Chiarella / Majewski
excess-demand mechanism the thesis is built on) and gives the agent
parameters real traction. Population is now 50: 20 FT + 10 MT + 20 ZI. The FT
dead-band is removed (z_score alone supplies FT heterogeneity). The three ZI
rates entered the calibration loop — theta is now 8-d per regime (with
`mt_mu`, D27): `ft_alpha, mt_alpha, mt_mu, depth_mean, mt_lambda,
zi_alpha, zi_mu, zi_delta`. The
market layer reproduces the volatility level and the flat return ACF;
excess kurtosis / clustering remain a structural gap (see D24/D25, §6).
The calibration loss was rebuilt in parallel: see D18g.

**(D14j)** VolatilityTrader removed from the model. It was a Gao-2023
Chiarella-Heston construct trading an exogenous Heston `σ_t` that was not the
volatility of anything else in the model (`V_t` is plain GBM). Its primary
justification — long-horizon volatility clustering — had already been conceded
as structurally unreachable (D18f). Class deleted from `agents.py`;
`n_volatility` / `vt_alpha_market` / `vt_alpha_limit` removed from
`ModelParams`; `SimContext.vol` / `z_vol_s` removed; `simulation.py` no longer
loads `sigma_t` / `z_vol_s`. `data/v_heston.py` is retired (V_t now comes from
the plain-GBM `V_smooth` column, which `v_gbm.py` can generate directly). The
procyclical-volatility channel VT nominally provided is deferred to the
endogenous CCP-tier fire-sale feedback (Stage 4+).

**(D13f)** MomentumTrader folded back from the D13e two-cohort long/short
split to a single momentum type — the split's only justification was the
clustering now conceded. `mt_lambda` (EWMA decay) re-enters the calibration
loop: with one momentum horizon it is the model's primary trend-strength dial,
and Majewski et al. (2018) estimate the trend timescale from data, so it has a
literature home as an estimated quantity.

**50-agent flat FT + MT + ZI population (no MM — D21). 8-d theta per
regime** (`ft_alpha`, `mt_alpha`, `mt_mu`, `depth_mean`, `mt_lambda`,
`zi_alpha`, `zi_mu`, `zi_delta`), independent calm/stressed runs. The
market layer is a three-type ABM (Simudyne ODD + Cont-Stoikov + HFABM) in
which FT/MT/ZI order flow drives price. `globals.CALIBRATED` carries stale
placeholders — **an 8-d re-calibration is pending** (see §6).

---

## 0 · Project frame

Pure-Python ABM of a CCP-cleared single-asset market (ES front-month futures),
built to study CCP systemic-risk dynamics, margin procyclicality, and contagion
under calm vs stressed regimes. Single LOB with batched call-auction clearing
at 1-min cadence (390 steps per RTH 6.5h trading day — ODD-native).

Reference priority chain (every design decision cites one; deviations flagged):
1. Simudyne CCP Risk Model **ODD** (`Simudyne_CCPRiskModel_ODDDoc.md`)
2. Deloitte CCP Risk Model webinar + CCP resilience white paper — clearing-tier
   mechanics; the SV-V_t motivation (D33) came from the WEBINAR (the white paper
   has no equations / no σ-process; the published ODD is incomplete on the
   fundamental). The formal OU-on-σ is cited to Stein-Stein (1991) below.
3. Majewski, Ciliberti, Bouchaud (2018) — Extended Chiarella estimation
4. Gao et al. (2023) — Deep hedging / stochastic-vol Chiarella
5. Vytelingum et al. (2025) — ABM Liquidity Risk (Simudyne WP008,
   arXiv:2505.15296; "Krishnen" was a misread of Peru-krishnen Vytelingum)
6. Cont, Stoikov & Talreja (2008) — Stochastic LOB models
7. Farmer, Patelli, Zovko / Daniels (2003, 2005) — ZI baseline
8. Gao et al. (2022) HFABM — flash-crash ABM, MM spec, calibration
   (arXiv:2208.13654; previously mis-cited in this repo as "Cao et al. 2024")

Supporting:
- Stein & Stein (1991); Heston (1993) — stochastic-volatility V_t; the formal
  OU-on-σ model behind the D33 SV layer (the Deloitte webinar was the prompt)
- Mike & Farmer (2008) — empirical log-normal order-placement distribution
- Bouchaud, Mézard, Potters (2002) — empirical depth profile near the best
  quote (NB: measured from the opposite best, not the mid — see D3c)
- Lee & Mykland (2008) — jump-detection diagnostic (justified dropping jumps)
- Cont (2001) — stylised facts of asset returns
- Hill (1975); Resnick (2007) — tail-index estimator
- Almgren & Chriss (2000); Almgren, Thum, Hauptmann & Li (2005) — optimal
  execution; POV impact (MM liquidation)
- Franke & Westerhoff (2012); Künsch (1989) — block-bootstrap moment weights
  (NB: F&W use the FULL inverse-covariance matrix; this repo uses the diagonal
  inverse-variance form — their §4.4 moment-coverage heuristic)
- Lamperti, Roventini & Sani (2018) — surrogate calibration (inspiration only;
  they use one surrogate on an aggregate measure — regression AND classification
  — whereas we fit a per-moment regression surrogate; see `calibrate.py`)

Note: Gao et al. (2023) was the basis for the now-removed VolatilityTrader
(D14j). It remains in the chain as a project reference but no longer justifies
a live mechanism.

---

## 1 · Repository layout

```
MScThesis_ABM/
├── claudereadme.md
├── README.md
├── AGENT.md              # AI-agent collaboration briefing
├── data/
│   ├── data.py           # DataBento ingest (OHLCV-1m, BBO-1m, MBP-10)
│   ├── roll.py           # ES front-month roll + 1-min resample (close + mid)
│   ├── v_gbm.py          # GBM calibration (on mid) + V_t path generator
│   ├── v_kalman.py       # KF / Roll-1984 σ_v diagnostic (citable cross-check)
│   ├── p_zi.py           # MBP-10 geometric-depth calibration (streaming)
│   ├── impact.py         # BBO-1m Hasbrouck/AC impact (η, γ) — for future AC
│   ├── glbx-*.csv.zst    # DataBento raw (zstd)
│   ├── processed/        # rolled front-month 1-min series (ES_front_{regime}_1m.csv)
│   ├── fv_calm.csv       # V_t path, calm regime (V_smooth column read by the sim)
│   └── fv_stressed.csv   # V_t path, stressed regime (2020 COVID crash)
├── model/
│   ├── globals.py        # ModelParams + SimContext + regime dicts + CALIBRATED + CCP consts
│   ├── lob.py            # Order, Fill, LOB
│   ├── agents.py         # BaseTrader, ZI, FT, MT, BCM, NBCM (MM dormant — D21)
│   ├── clearing.py       # BalanceSheet + CentralCounterparty (D28)
│   └── simulation.py     # Simulation driver
├── run_simulation.py     # entry point + build_traders() + build_clearing_tier()
├── calibrate.py          # surrogate-assisted SMM (8-d theta per regime)
├── analysis.ipynb        # post-run market-layer analysis notebook
├── clearing_analysis.ipynb   # clearing-tier analysis — capital ratios, margin calls, balance sheets (D29)
├── empirical_analysis.ipynb  # empirical ES stylised facts (Hill, kurtosis, ACF) — HFABM/Cont
└── output/
    ├── v_gbm_params.json      # GBM calibration (μ, σ, v0 per regime)
    ├── v_kalman_params.json   # KF diagnostic (citable; not wired in)
    ├── p_zi_params.json       # MBP-10 depth + zi_delta diagnostic
    ├── impact_params.json     # η/γ impact regression (not wired in)
    ├── calibration_lhs_{regime}.csv      # per-regime surrogate training set (regenerated)
    ├── calibration_stage2_{regime}.csv   # per-regime stage-2 grid output (regenerated)
    ├── calibrated_params.json # final theta + validated moments, nested by regime
    ├── run_{regime}.csv       # price-history output of run_simulation.py
    └── clearing_{regime}.csv  # clearing-tier observables — per CM per margin cycle (D29)
```

L2 MBP-10 data lives **outside** the repo at `../L2_data_thesis/` (gigabyte files): 4 calm days (`glbx-mdp3-2019060[3-6].mbp-10.csv.zst`) + 4 stressed days (`glbx-mdp3-2020031[6-9].mbp-10.csv.zst`).

---

## 2 · Agent specification (50 LOB traders + clearing tier — D28)

All LOB-trading agents share signature `submit_orders(lob, params, ctx,
rng)`. SimContext provides exogenous V_t, prev-step mid, and tick. The
MarketMaker is dropped (D21) — FT/BCM/MT/ZI order flow drives price. The
clearing tier (BCM/NBCM/CCP — §2.5) is layered on top: BCMs trade the LOB
own-account FT-style; NBCMs and the CCP do not trade.

**Order lifetime (D5d)** — three layers:
- FT / MT: **replace-on-new**. Each holds at most one standing limit
  (`_open_oid`). On the next activation it cancels that order (if still
  resting) and places a fresh one. No per-resting cancellation rate.
- ZI: per-resting **Bernoulli cancellation** at rate `zi_delta` (calibrated).
- ALL types: a hard `order_ttl = 10`-step ceiling enforced by
  `LOB.age_orders()` — whichever removes the order first wins.

### 2.1 Fundamental Trader (ODD §Agents)

Per-agent persistent `z_score ~ N(0, 1)` at init. Per step:

| Step | Behaviour |
|---|---|
| Stale-oid drop | clear `_open_oid` if it is no longer resting (filled / TTL-expired) |
| Reservation | `R = V_t + z_score · σ_fundamental`,  `σ_fundamental = ft_sigma_c · σ_v · v0` with `ft_sigma_c = √390` pinned (one daily V_t std — D7b) |
| Side | `+1` if `R > mid`, `−1` if `R < mid` (skip the measure-zero `R = mid`) |
| Activation | Bernoulli(`ft_alpha`) per step |
| On activation | **replace-on-new**: cancel the standing limit, place one fresh limit at `R`, qty ~ U[qty_min, qty_max] |

**No dead-band (D23)** — the persistent `z` (D6b) already supplies the FT
heterogeneity and the per-agent-reservation trigger keeps flow two-sided
(D9c — trigger on `R − mid`, not the collective `V_t − mid`). A flat band
(D9b) and a wide per-agent Simudyne band (D22) were trialled and dropped.
Persistent `z` also builds the concentrated FT inventories the Stage-4+
CCP/margin layer bites on.

### 2.2 Momentum Trader (chartist — single-type; D13f, limit-only D40)

Single momentum type — every MT shares the EWMA decay `params.mt_lambda`
(pinned at 0.05 — D44; out of the loop). Each MT carries its own `_M` and
`_prev_mid` state. **Limit-only** (D40 — the D27 market branch was reverted;
`mt_mu` stays on `ModelParams` at 0.0 for back-compat but is unused).
**Trades every step** (D36 — `mt_alpha` pinned at 1.0, out of the loop).
Per step:

| Step | Behaviour |
|---|---|
| Stale-oid drop | clear `_open_oid` if no longer resting |
| Realised return | `r = log(mid_{t-1}) − log(mid_{t-2})` (skipped during warm-up) |
| EWMA update | `M_t = (1 − mt_lambda)·M_{t-1} + mt_lambda·r` — runs every step |
| Tiny floor | skip if `\|M_t\| < mt_eps` (1e−6 structural) |
| Side | `sign(M_t)` |
| Limit (every step) | **replace-on-new** — cancel standing limit, place fresh limit at `mid − sign(M_t)·k·tick` (passive: buy below mid, sell above), `k` from the shared log-normal `_draw_depth` (D20 — see §2.3), qty `~ U[qty_min, qty_max]` |

The signal `M_t` drives only the order side (the prior signal-driven depth
`k_base·σ_v/|M_t|` was removed — D20; placement depth is the shared log-normal
`depth_mean`/`depth_sigma`). **The D27 market-order branch was reverted (D40)**:
trend-direction market flow corrupted the return ACF. None of `mt_alpha`,
`mt_mu`, `mt_lambda` is in the calibration loop now (all pinned). On the
naming of the trend timescale: Majewski et al. (2018) likewise FIX the trend
horizon externally (α = 1/7, τ = 6 months, from a CTA-index correlation) rather
than estimating it — so pinning `mt_lambda` is consistent with their practice,
not a departure from it (the earlier D13f/D35 "estimated from data" wording was
wrong and is corrected here).

### 2.3 Zero-Intelligence Trader (Cont, Stoikov & Talreja 2008; Farmer-Patelli-Zovko 2004)

The three-event structure (limit / market / per-resting cancellation) is
faithful to both papers; the α/μ/δ naming is actually Farmer-Patelli-Zovko's.
NB the baseline rates 0.15 / 0.025 / 0.025 are the ODD §Calibration values —
they are NOT reported in Cont-Stoikov-Talreja (which estimates per-second rates
for one TSE stock); here all three are calibrated anyway (D21), so the baseline
is only the bound anchor.

`n_zi = 20` (D23 — doubled from 10). Per step, three independent Bernoulli
draws (ODD-native per-step probabilities — no `dt` rescaling):

| Draw | Behaviour |
|---|---|
| Per-resting cancellation | each ZI resting limit cancelled w.p. `zi_delta` |
| Limit arrival | w.p. `zi_alpha` — random side, depth `k` from `_draw_depth`, qty ~ U[qty_min, qty_max] |
| Market arrival | w.p. `zi_mu` — random side, qty ~ U[qty_min, qty_max] |

True zero intelligence — no signal dependence. All three rates `zi_alpha,
zi_mu, zi_delta` are **calibrated** (D21 — they entered the loop when the MM
was dropped and ZI became a primary price/liquidity driver). Doubling the ZI
count (D23) densified the near-mid book and removed the FT-flow choppiness
(`acf_r₁` −0.14 → ≈ 0).

**Shared placement depth — `_draw_depth` (D20).** ZI and MT both rest limit
orders `k` ticks from the **mid** (a simplification — Cont-Stoikov-Talreja and
Bouchaud-Mézard-Potters measure depth from the opposite best quote, not the
mid; deviation D3c), `k ~ LogNormal(μ, depth_sigma)` with the distribution mean
`depth_mean` AND the shape `depth_sigma` both calibrated (D41; `μ = ln(depth_mean)
− depth_sigma²/2`; `k` rounded, floored at 1). Supersedes the ZI L2-geometric
depth (D3) and the MT signal-driven depth (D13d).

### 2.4 Market Maker — HFABM-style mid-anchored, low qty (D31; n_mm=4 + mm_p_edge calibrated, D48)

`n_mm = 4` (D48 — up from the D31 re-introduction at 2). The class is
**mid-anchored, no inventory skew, always quotes** (Gao et al. 2022 HFABM §3.6
single-quote style). Per step: cancel prior MM quotes; quote one bid + one ask
at `anchor ± U{0, mm_p_edge}·tick`, `qty = mm_qty` on each side; `anchor =
prev-step mid` (`v0` at t=0). **`mm_qty = 2` is now PINNED structural** (D48 —
from `globals.MM_QTY`); the calibrated MM dial is **`mm_p_edge`** (the 6th theta,
bound `(1.0, 8.0)`, rounded to int at submit) — D48 supersedes the D31 "`mm_qty`
calibrated" convention. (Gao et al.'s MM also carries an inventory-limit
"hot-potato" regime switch — the core of their flash-crash study — which this
thesis does NOT model; the always-quote variant is used instead.) The MM is the
kurtosis-fix mechanism: continuous
near-mid liquidity absorbs market orders so the book does not walk in
one go between V_t-jump steps — the static-then-jump return pattern that
drove the D21-era kurt ~100–500 is broken. Sweep evidence (`outputs/
test_mm.py`): calm 2 MMs @ `mm_qty=1` cut kurt 126→44 and lifted Hill to
3.10 ≈ target. D31 fixes both the D10g pin (V_t anchor) and the D21
drop (no MM at all); the third historical configuration — last-mid-
anchored with high qty — froze the price and is not revisited. The POV /
Almgren-Chriss helpers (`ac_schedule`, `mm_pov`, `mm_inventory_*`) stay
reserved for the Stage-4+ BCM fire-sale.

### 2.5 Clearing tier — BCM / NBCM / CCP (D28 scaffold + D29 margin)

The central-clearing tier is live with the **variation-margin cycle**
(D29): agents, balance sheets, counterparty links and the 60-tick margin
mechanism all run; the default-fund and waterfall MECHANICS are still
deferred. It extends the Simudyne CCP ODD — whose CMs trade for their own
account with no client tier — with the thesis's client-clearing layer.

- **`BankingClearingMember`** (`agents.py`, ×10) — a `FundamentalTrader`
  subclass. Cast from the FT pool: it trades its OWN account through the
  inherited FT `submit_orders` (the ODD's CMs price via `z_score·σ_fund`,
  identical to the thesis FT), so 10 FT + 10 BCM is the 20-FT-equivalent
  the calibration runs against — the 8-d theta carries over unchanged. It
  additionally carries a `BalanceSheet`, a `ccp_id` link and a
  `client_ids` book. `capital_ratio(mid) = cash / [(|own_inv| +
  client-book notional)·mid]` (ODD §Mech #2).
- **`NonBankingClearingMember`** (`agents.py`, ×5) — a `BaseTrader`
  subclass. Pure clearing intermediary: NO own position, NO LOB orders
  (not in the trading `traders` list). Carries a `BalanceSheet`, a
  `ccp_id` link and a `client_ids` book. `capital_ratio(mid) = cash /
  [client-book notional·mid]` — no own position.
- **`CentralCounterparty`** (`clearing.py`, ×1) — the clearing role of the
  ODD MatchingEngine (matching itself stays in `LOB`). Holds the member
  registry (`member_ids` / `members`), the default-fund accounts
  (`own_df` = 10% exchange SITG, `total_df` = 0 until the DF stage) and
  `default_list`. `register_member()` wires the bidirectional CM↔CCP link
  (ODD CMToEX / EXToCM star topology).
- **`BalanceSheet`** (`clearing.py`) — per-CM record (ODD §Agents): cash /
  own position / PnL stay on the agent; the record carries
  `client_positions`, `initial_margin`, `maintenance_margin`,
  `variation_margin`, `call_indicator`, `has_defaulted`,
  `counterparty_id`, `_last_mark`. `df_contribution` is scaffolded at zero
  until the DF stage.
- **Margin cycle** (D29) — `Simulation._margin_cycle`, every
  `margin_interval = 60` ticks (ODD §Step Sequence step 11): variation
  margin settles the cleared book's MtM move into CM cash, IM/MM are
  recomputed, `call_indicator` fires when `|VM| > 5%·IM`, the capital
  ratio is recorded, `has_defaulted` is set on cash exhaustion. CMs use
  **futures-margin accounting** — a fill posts only the position, cash is
  settled by the VM cycle (D29). One `clearing_history` row per CM per
  cycle → `output/clearing_{regime}.csv`.
- **Client links** — plain FT + MT are assigned as clients; only the
  `n_bcm_with_clients = 5` client-carrying BCMs + the 5 NBCMs hold books
  (the other 5 BCMs are own-account only — D29). ZI stay direct exchange
  participants (ODD §Initialization). Client books are still **inert** —
  `client_positions` at 0; clients trade the LOB directly, trade novation
  through the CM is a later stage.

Deferred (later stages): the cover-2 default fund, the 5-level waterfall,
position auction, BCM fire-sale / NBCM stop-out, client-trade novation.
The ODD §Calibration constants live in `globals.CCP_CALIBRATION`.

### 2.6 SimContext

```python
@dataclass
class SimContext:
    v: float                 # exogenous V_t this step
    mid_price: float         # prev-step post-match mid (NaN at t=0)
    tick: int
    traders_by_id: dict      # id → trader lookup (for the CM tier, Stage 4+)
    last_volume: int = 0     # prior-step transacted volume (unused since the
                             # MM was dropped, D21; retained for Stage 4+)
```

### 2.7 LOB

Price-time priority, FIFO at each level, `match()` clears any cross.
`add_limit(...)` appends a resting order — it does **not** auto-cancel any
prior order from the same agent (replace-on-new is handled agent-side).
`age_orders()` decrements each resting order's TTL and removes it after
`order_ttl` (10) steps. `is_resting(oid) → bool` for agent-side stale-oid
pruning. `last_price` tracks the most recent fill. `_update_quotes()`:
mid = (best_bid + best_ask) / 2 when both sides populated; falls back to
`last_price` when one side empties (D9d).

---

## 3 · Parameters

### 3.1 Structural (not calibrated)

| Name | Value | Source |
|---|---|---|
| `tick_size` | 0.25 | CME ES |
| `dt_minutes` | 1.0 | sim cadence (ODD-native 1-min) |
| `order_ttl` | 10 | hard TTL ceiling — ODD §Mech #7; ≈10 min at 1-min cadence (D5d) |
| `ft_sigma_c` | √390 ≈ 19.7 | one daily V_t std (Chiarella heterogeneous-beliefs; ABIDES `ValueAgent`) — D7b |
| `mt_eps` | 1e−6 | tiny floor on `|M_t|` before EWMA warms up |
| `mt_lambda` | 0.05 | MT EWMA decay — pinned (D44) |
| `ft_alpha, mt_alpha` | 1.0 | FT/MT trade every step — pinned (D36) |
| `mt_mu` | 0.0 | MT market branch off — limit-only (D40) |
| `mm_qty` | 2 | MM per-side quote size — pinned (D48; `globals.MM_QTY`) |
| `qty_min, qty_max` | 1; 10 both regimes | ODD §Stochasticity (`QTY_MAX`; D49 reverted) |

The FT has no dead-band (D23 — `ft_threshold_bps` removed). The MarketMaker is
LIVE again (D48 — `n_mm = 4`, HFABM mid-anchored); the AC `ETA/GAMMA` dicts and
the POV helpers stay dormant in `globals.py` for the Stage-4+ BCM fire-sale.
The three ZI rates and `depth_sigma` are no longer structural — they are
calibrated (§3.3; `mm_p_edge` too — D48).

### 3.2 Data-calibrated offline (per regime, fixed before agent loop)

V_t is a **Stein-Stein-style stochastic-vol Merton jump-diffusion** (D33 —
supersedes the constant-σ D25 model): `dσ_t = α(θ − σ_t)dt + σ_vol·dW^σ`,
`Δlog V = (μ − ½σ_t²) + σ_t·Z + J`, `J` a Merton jump (prob `λ/390` per
step) of size `N(0, δ²)`. `data/v_gbm.py` calibrates the jump-diffusion
layer (`σ_d`, `δ`) from the empirical variance + kurtosis decomposition,
then the Vasicek-OU layer (`α`, `σ_vol`) by method of moments on a 30-min
realised-vol time series; `θ` is re-anchored so `E[σ_t²] = σ_d²` (the SV
layer adds clustering without inflating total vol).

Source: the SV architecture was prompted by a **Deloitte CCP Risk Model
webinar** (the white paper has no equations / no σ-process; the ODD is
incomplete on the fundamental). The formal OU-on-σ is **Stein-Stein (1991)**;
`λ = 3/day` is the ODD value; piping `σ_t` into the FT belief width (D34) is
this thesis's own modelling choice. **Caveat (Hill-gap cause):** the kurtosis
decomposition assigns ALL excess kurtosis to jumps assuming constant σ (line
~155 `v_gbm.py`), THEN adds the OU layer without re-deriving kurtosis — so `δ`
is biased high and the validated 1-min Hill comes out too heavy (≈1.1 vs 3.25).
The fix is to let the SV layer absorb part of the kurtosis (`f_cap ≤ 0.5` /
`λ ↓ 1`); see Structural limits #1.

| Name | Method | Status |
|---|---|---|
| `v0` per regime | first ES front-month **mid** of the regime sample | ✅ **calm 2463.125, stressed 3253.875** (`v_gbm.py`) |
| `σ_d` (long-run diffusion) | jump-diffusion decomposition of 1-min variance + kurtosis | ✅ **calm 2.62e-4, stressed 1.53e-3** (D25/D33) |
| `λ, δ` (jumps) | λ fixed 3/day (ODD); δ from the excess kurtosis | ✅ **δ calm 1.61e-3, stressed 1.14e-2** (jumps carry 23% / 30% of variance) |
| `α, σ_vol, θ_eff` (Vasicek-OU SV) | MoM on 30-min RV: α=−ln(ρ_30m)/30, σ_vol²/(2α)=Var(RV/√30); θ_eff re-anchored to σ_d² | ✅ **calm α=0.0125/min, σ_vol=2.48e-5, θ_eff=2.09e-4; stressed α=0.0127/min, σ_vol=1.26e-4, θ_eff=1.31e-3** — ρ_30m ≈ 0.68 both regimes (strong persistence; D33) |
| `σ_v` (total) | √(E[σ_t²] + λ_step·δ²) = √(σ_d² + λ_step·δ²) by construction — total 1-min vol, the FT/agent scale | ✅ **calm 2.97e-4, stressed 1.82e-3** |
| `μ_v` per regime | mean of RTH 1-min mid log-returns (same valid set) | ✅ **calm +1.21e-6, stressed −6.36e-6** |
| `p_zi` per regime | streaming Geometric MLE on MBP-10 add-event depths | ⚠️ **DIAGNOSTIC ONLY (D20)** — placement depth is now a calibrated log-normal; `p_zi.py` still reports the geometric fit (calm 0.5430 / stressed 0.3431) as a book diagnostic |
| `η, γ` (AC impact) | Hasbrouck-1991 / AC-2000 linear-impact regression on BBO-1m (`data/impact.py`) | ⚠️ **calm η=1.5e-4, stressed η=7.0e-4; γ floored at 0**. NOT wired into any agent — kept for the Stage-4+ AC fire-sale (D10e) |

CLI: `python data/v_gbm.py {calibrate, generate, generate-all}` —
`generate-all` writes the V_t path to `data/fv_{regime}.csv` (the `V_smooth`
column the simulator reads); `python data/p_zi.py {calibrate, ...}`;
`python data/v_kalman.py calibrate`; `python data/impact.py`. `σ_v`/`v0`
live in `globals.py` regime-keyed dicts; `ModelParams.__post_init__` fills them
from the `stressed` toggle.

**Stochastic-vol jump-diffusion (D33; D25, reverts D12)**: V_t carries
Merton jumps (D25, ODD §Stochasticity) AND a Vasicek-OU stochastic
volatility (D33). On the calibrated paths, V_t now exhibits the
volatility clustering D24-D26 deemed structurally unreachable: calm
`acf_absr` 1/10/30 = 0.25/0.23/0.19 vs empirical 0.31/0.25/0.21
(captures ~70-90% of empirical clustering out to lag 30; the single OU
decays a bit too fast at lags ≥ 60). kurt and ret_std match empirical at
the V_t level (calm kurt 19.7 vs 19.9; calm ret_std 3.04e-4 vs 2.97e-4).
Lee-Mykland (2008) detects only ~0.25
*statistically-significant* jumps/RTH-day, but the model uses the ODD's
more-frequent λ=3/day with smaller δ so the kurtosis is a stable sample
feature. NB the jumps did NOT deliver the hoped-for volatility clustering at
the mid — see the D24/D25 deviation rows and §6.

### 3.3 Agent calibration loop (6-d theta PER REGIME, independent runs)

Each regime is calibrated as a separate optimisation problem (D18b). No
parameter is shared across calm and stressed. The 6 free dimensions
(`calibrate.py` `PARAM_BOUNDS`):

| Name | Description | Bound |
|---|---|---|
| `depth_mean` | shared ZI+MT log-normal placement-depth mean, ticks — D20 | [1.0, 10.0] |
| `depth_sigma` | shared ZI+MT log-normal placement-depth shape — D41 | [0.1, 1.0] |
| `zi_alpha` | ZI limit-order arrival per step — D21/D30 | [0.02, 0.50] |
| `zi_mu` | ZI market-order arrival per step — D21/D30 | [0.005, 0.10] |
| `zi_delta` | ZI per-resting cancellation per step — D21 | [0.005, 0.50] |
| `mm_p_edge` | HFABM MM tick-depth ceiling, `d ~ U{0,..,p_edge}` — D48 | [1.0, 8.0] |

6 free values × 2 regimes = 12 total, calm and stressed independent. Pinned /
out of the loop: `ft_alpha = mt_alpha = 1.0` (D36); `mt_mu = 0` (D40);
`mt_lambda = 0.05` (D44); `mm_qty = 2` (D48); `ft_sigma_c` (√390 — D7b);
`order_ttl` (10 — D5d); `qty_max` (10 both regimes — D49 reverted);
`σ_v`/`v0`/`μ` (data-side, offline). NB many earlier comments in `calibrate.py`
still say "5-d"/"8-d" — the live `PARAM_BOUNDS` dict (6 keys) is the truth.

### 3.4 Population (54 LOB traders + clearing tier — D28/D48)

**LOB-trading layer** (`build_traders`) — 54 agents:

| Group | Count | Class |
|---|---|---|
| Fundamental Trader | 10 | `FundamentalTrader` |
| Banking Clearing Member | 10 | `BankingClearingMember` (FT subclass — trades own account; D28) |
| Momentum Trader | 10 | `MomentumTrader` (single cohort, `lambda_decay = mt_lambda`; limit-only D40) |
| Zero-Intelligence Trader | 20 | `ZeroIntelligenceTrader` (D23 — doubled) |
| Market Maker | 4 | `MarketMaker` (HFABM mid-anchored, `mm_qty = 2`, `mm_p_edge` calibrated; D48) |

VolatilityTrader (D38) and the long-MT cohort (D35) and ContTrader (D39) were
all **removed (D40/D44)** — `n_vt = n_ct = n_momentum_long = 0`. The classes
stay dormant in `agents.py` for back-compat.

**Clearing tier** (`build_clearing_tier`, D28/D29) — not LOB traders:

| Group | Count | Class |
|---|---|---|
| Non-Banking Clearing Member | 5 | `NonBankingClearingMember` (no own position) |
| Central Counterparty | 1 | `CentralCounterparty` |

10 FT were re-cast as BCM (the user's design); the 20-FT-equivalent for price
formation is unchanged, so `calibrate.py` `POP` keeps `n_bcm = 0` and
instantiates `n_fundamental = 20` + the 4 MMs (it mirrors the runtime LOB
layer exactly — 20 FT-equiv + 10 MT + 20 ZI + 4 MM = 54). Cash (inert for
non-CMs): FT/MT ~U[1M, 10M]; ZI ~U[10k, 100k]; MM ~U[10M, 100M]; BCM
~U[5B, 10B]; NBCM ~U[5M, 10M]; CCP 10M (ODD §Initialization). Clients: plain
FT + MT (20) round-robin assigned to `n_bcm_with_clients = 5` client-carrying
BCMs + 5 NBCMs; the other 5 BCMs are own-account only (D29). ZI stay direct
(ODD §Initialization). Client books are still INERT (`client_positions` at 0 —
novation deferred), so NBCM capital ratios are `inf`. CM↔CCP bidirectional
links (ODD star topology).

**Clearing constants** (`globals.CCP_CALIBRATION`, ODD §Calibration —
regulatory, none SMM-calibrated; wired stage by stage): `im_percent` 0.20,
`mm_percent` 0.95, `df_percent` 0.20, `ex_df_ratio` 0.10, `cover_number`
2, `recovery_rate` 0.60, `cap_ratio_floor` 0.08, `margin_interval` 60,
`df_interval` 390.

### 3.5 Calibrated values — `globals.CALIBRATED` is STALE (re-copy + calm re-run pending)

`globals.CALIBRATED` holds 6-d theta per regime, but **neither entry reflects a
clean current-model validated run**:

| Parameter | Calm (in globals) | Stressed (in globals) | Stressed (validated D48, on disk) |
|---|---|---|---|
| `depth_mean` | 5.7149 | 7.8733 | 7.2226 |
| `depth_sigma` | 0.3 | 0.3 | 0.9507 *(rail @ 1.0)* |
| `zi_alpha` | 0.4722 | 0.4957 | 0.1483 |
| `zi_mu` | 0.0374 | 0.0615 | 0.0212 |
| `zi_delta` | 0.3618 | 0.0799 | 0.2749 |
| `mm_p_edge` | 4.0 | 4.0 | 1.4262 *(rail @ 1.0)* |

The stressed entry in `globals` is a **pre-D48 placeholder**; the validated D48
stressed θ lives in `output/calibrated_params.json` (right column) but was
**never copied back** (the AGENT.md §5 step was skipped). Calm has **never been
run at 6-d** (absent from the JSON — the stressed-only run overwrote it,
gotcha #3); its `globals` entry is the pre-D48 5-d overnight result with
`mm_p_edge = 4.0` default. **Pending:** copy the D48 stressed θ (note the two
rails) and run calm at 6-d.

Validated D48 stressed **D(θ) = 21.67** (ΔHill 9.53 + ΔV 0.55 + ΔACF1 2.60 +
ΔACF2 8.98). ret_std 1.9e-3 (target 1.8e-3 ✓); but **Hill 1.13 vs 3.25**, kurt
395 vs 35 (out of loss) — the structural heavy-tail gap (§3.2 caveat,
Structural limits #1). Critically, the D33 SV clustering did NOT reach the mid
in stressed: `acf_absr` lags 1/10/30 = 0.07 / −0.005 / −0.004 vs empirical
0.27 / 0.21 / 0.19 — verified at the fundamental layer, lost at the mid.

---

## 4 · Stage history

| Stage | Status |
|---|---|
| 1 — ZI baseline | ✅ legacy |
| 2 — + FT, MT, jumps/CIR (retired) | superseded |
| 3 — 50-agent FT+MT+ZI market-dynamics calibration (8-d, D30) | ✅ ran; ΔHill dominant residual |
| 3b — clearing-tier scaffold (BCM/NBCM/CCP, balance sheets, links — D28) | ✅ structure live |
| 3c — MM re-introduced HFABM-style (D31) + ΔACF2 reinstated (D32) + Stein-Stein SV V_t (D33) | ✅ code live; V_t clustering verified at the fundamental layer; 9-d ABM re-calibration pending (§6) |
| **Current — Stage 4: margin (IM/MM + 60-tick variation-margin cycle — D29)** | ✅ VM cycle live; `clearing_analysis.ipynb` reads `output/clearing_{regime}.csv` |
| 5 — Default fund (Cover-2 EMIR) | next |
| 6 — 5-level waterfall + position auction | pending |
| 7 — Almgren-Chriss distressed liquidation | `ac_schedule()` + `impact.py` ready; not wired |
| 8 — Full agent calibration scale-up | pipeline working; thesis-final budgets pending |

Stages 4–6 build the margin / default-fund / waterfall mechanics onto the
D28 clearing scaffold; the ODD §Calibration constants are pre-staged in
`globals.CCP_CALIBRATION`. Tests (`tests/`) need rewrite against the
current spec.

---

## 5 · Deviations from ODD (locked)

| # | Deviation | Reason |
|---|---|---|
| D1 | Single ES LOB (vs ODD's dual FTSE 100/250) | Data is ES-only |
| D2 | 1-min steps × 390/day (ODD-native cadence) | Matches ES 6.5h RTH |
| D3 | ZI placement: geometric depth from mid, `k ~ Geometric(p_zi)` | Cont-Stoikov 2008, mid-anchored per Bouchaud 2002. Log-normal trialled and rejected (rounding artifact); `data/p_zi.py` reports both fits |
| D3b | `zi_delta` empirically saturates to 1.0 in stressed (rate 35.6/order/min) | Real L2 finding; ZI runs the ODD value 0.025 instead — the L2 value over-churns the book |
| D3c | Placement depth measured from **mid**, not opposite-best | Mid-anchored is simpler; half-spread bias bounded by ~1 tick on ES |
| D4 | ZI/FT/MT arrivals: per-step Bernoulli at the agent's rate | ODD-faithful — 1-min cadence is ODD-native (no `dt` rescaling) |
| D5 | Cancellation is per-resting-order Bernoulli | ODD-native per-step probability |
| D5b | (superseded by D5d) FT/MT carried calibrated per-resting cancellation rates | Added during the volatility-clustering push |
| D5c | (superseded by D5d) all per-order TTLs removed; Bernoulli δ the sole lifetime mechanism | Cont-Stoikov geometric lifetime |
| D5d | **Order management reworked. FT/MT use REPLACE-ON-NEW** — at most one standing limit (`_open_oid`); the next activation cancels the old order and places a fresh one. **ZI keeps per-resting Bernoulli cancellation** at `zi_delta`. **A hard `order_ttl = 10`-step ceiling** (`LOB.age_orders()`) sits on top. `ft_delta` / `mt_delta` removed from the calibration loop | ODD §Agents refreshes a CM's order on each action — replace-on-new is the ODD-native form for the signal-driven agents (their standing order should always reflect the latest reservation / momentum signal). The hard TTL is the ODD §Mech #7 ceiling. (Originally also covered VT, which shared ZI's `zi_delta`; VT removed under D14j.) |
| D6 | FT places at `V_t + z · σ_fundamental` (ODD §Agents) | ODD-faithful |
| D6b | `z` is **persistent** (fixed at agent init), not redrawn each step | Heterogeneous-beliefs (ODD; Chiarella-Iori-Perelló) over noisy-information (Glosten-Milgrom; ABIDES `ValueAgent`) — builds concentrated FT inventories for Stage 4+ |
| D7 | `σ_fundamental = ft_sigma_c · σ_v · v0` | Regime-invariant; ties FT noise to GBM V_t vol |
| D7b | **`ft_sigma_c` pinned at √390 ≈ 19.7 (structural).** σ_fundamental = one daily V_t std per regime | Chiarella heterogeneous-beliefs at the daily-news scale; ABIDES `ValueAgent`. FT belief dispersion is a design choice, not a market observable |
| D8 | FT has a Bernoulli `ft_alpha` activation gate; submits LIMIT orders only | Follows the Simudyne ODD (ref #1) over HFABM's market-only FT — needed for concentrated inventories at Stage 4+ |
| D9b | FT dead-band: skip when `|reservation − mid| ≤ ft_threshold_bps · V_t / 10000`; default 50 bps | Avoid persisting uninformative orders near V_t |
| D9c | FT trigger on `|reservation − mid|` (not `|V_t − mid|`) | Collective `|V_t − mid|` trigger synchronises FTs onto one side, empties a book side. `|reservation − mid|` splits FTs by z-sign, keeps flow two-sided |
| D9d | LOB mid falls back to `last_price` when a book side empties | Prevents mid teleporting onto an arbitrary surviving resting limit |
| D10 | MM = single quote per side per step at `d ~ U{0, ..., mm_p_edge}` | HFABM Gao et al. (2022) §3.2 stochastic-spread variant |
| D10b | (retired) `mm_qty` / `mm_inventory_limit` briefly in the calibration loop | Calibration pushed `mm_inventory_limit` to 3781 — over-fat tails. Superseded by D10c |
| D10c | **All four MM size parameters structural — none calibrated.** `mm_qty` from `globals.MM_QTY`; `mm_inventory_limit = 1000`; `mm_inventory_safe = 800`; `mm_p_edge = 4` | HFABM convention: MM size reflects book reality, not moment-matching |
| D10d | (superseded by D10f) MM liquidation chunked into `qty_max`-sized market orders via a `_liquidating` state machine | Fixed the single-shot slam, but a fixed chunk ignores current liquidity |
| D10e | (tabled) Almgren-Chriss optimal-execution schedule for MM liquidation. `ac_schedule()` retained in `agents.py`; `ETA_TEMP`/`GAMMA_PERM` retained in `globals.py`; `data/impact.py` calibrates η/γ | AC needs per-agent risk aversion λ and a fixed horizon T — extra parameters with no market observable. Deferred; `ac_schedule()` will be reused for the Stage-4+ BCM/CCP fire-sale where a horizon is natural |
| D10f | (superseded by D10g) **MM liquidation is POV (percent-of-volume).** While liquidating, MM submits `chunk = max(1, round(mm_pov · last_volume))`-sized market orders (capped at the safe target). Single structural parameter `mm_pov = 0.10` | Adapts to current liquidity with one structural knob instead of AC's λ/T pair. Almgren-Thum-Hauptmann-Li (2005); ABIDES `AdaptiveMarketMakerAgent` |
| D10g | **MM redesigned to fundamental-anchored inventory-skewed quoting.** The MM quotes one bid + one ask around `reservation = V_t − mm_skew·inventory·tick`, with `d_bid, d_ask ~ U{0,…,mm_p_edge}`. It ALWAYS quotes — the D10/D10f quote-or-go-dark liquidation state machine is removed. `mm_skew` is structural (0.05; D10c upheld — no MM parameter calibrated). The POV/AC machinery (`mm_pov`, `mm_inventory_limit/safe`, `ac_schedule()`) is retained for the Stage-4+ BCM/CCP fire-sale but is no longer used by the MM | The old MM hit its inventory limit ~54% of steps (4 MMs vs 20 persistent-z FTs), went dark, gapped the book, and the mid `(best_bid+best_ask)/2` teleported across scattered FT/MT limits — kurtosis ~10³, ret_std ~8× target. Those fat tails were a price-formation artifact, not an emergent fact. Anchoring the reservation to V_t and never withdrawing keeps the book tight (spread ~2 ticks): the mid tracks V_t, the market layer reproduces the volatility level and the flat return ACF. Excess kurtosis / clustering are NOT reproduced (V_t is plain GBM, D12; no volatility-state agent, D18f) — deferred to Stage-4 fire-sale feedback. Avellaneda & Stoikov (2008) inventory skew |
| D11 | V_t is exogenous, loaded from CSV; no GlobalState evolution in sim | Decouples V calibration (offline) from agent calibration |
| D12 | V_t is plain GBM (no jumps), calibrated offline via `data/v_gbm.py` | Lee-Mykland flagged only ~0.25 jumps/RTH-day. Departs from ODD §Mech #9 |
| D12b | V_t GBM drift μ = data-calibrated mean RTH 1-min log-return, not imposed at 0 | Honest data fit; matters for long horizons |
| D12c | σ_v retained from direct GBM fit; KF / Roll-1984 diagnostic confirms the correction is immaterial | `data/v_kalman.py` — citable noise cross-check; not wired in |
| D13 | (superseded) MT had a discrete market-vs-limit switch on `|M_t|` | HFABM/Vytelingum/Majewski `tanh` simplified to a step switch |
| D13b | (superseded by D13e, then D13f) HFABM long/short MT subgroup split trialled, reverted, re-adopted, reverted again | See D13e/D13f |
| D13c | (superseded by D13d) MT had a natural-marketable branch (k ≤ 0 → market order) | The calibration drove `k_base` to ≈ 0.5 — synchronised 10-MT bursts, calm kurt > 2400 |
| D13d | **MT is always-limit (no market branch).** `k = max(1, round(k_base · σ_v / max(|M_t|, mt_eps)))`; submit a limit at depth k | Removes the cliff-driven burst pathology. Trend amplification relies on inside-spread MT limits being filled passively (Cont-Stoikov 2008) |
| D13e | (superseded by D13f) MT split into a two-cohort population (HFABM §3.4 long/short) with structural EWMA decays `mt_lambda_short=0.5` / `mt_lambda_long=0.02` | Targeted long-horizon volatility clustering. The split *could* produce clustering at the midpoint but the calibrated optimum traded it away |
| D13f | **MT folded back to a SINGLE momentum type.** `mt_lambda_short`/`mt_lambda_long` replaced by a single `mt_lambda`; `build_traders` builds one MT loop with no cohort split; `MomentumTrader` reads `params.mt_lambda`. `mt_lambda` re-enters the **calibration loop** | The two-cohort split's only justification was long-horizon clustering, which was conceded as structurally unreachable (D18f) — so the extra cohort was pure complexity. With a single momentum horizon `mt_lambda` becomes the model's primary trend-strength dial; Majewski et al. (2018) estimate the trend timescale from data, giving it a literature home as an *estimated* quantity (so calibrating it is consistent with "calibrate as few as possible, prefer empirics/literature") |
| D14 | Flat direct population; BCM/NBCM/client clearing tier deferred to Stage 4+ | Calibrate market dynamics on the flat ODD star topology first |
| D14b | (superseded by D14e) ZeroIntelligenceTrader dropped from active set | After adding VT, ZI wasn't pulling weight in any moment |
| D14c | **`n_mm` bumped from 2 → 4** | Stressed top-of-book depth was structurally too thin. HFABM Gao et al. (2022) §3.2 uses 4–8 MMs |
| D14d | **`qty_max` regime-scaled via `globals.QTY_MAX` — 10 calm, 5 stressed** | Real traders size down in stressed markets; stressed top-of-book is ~5× thinner. A deviation from ODD §Stochasticity justified by realism |
| D14e | **ZeroIntelligenceTrader re-introduced as a background noise floor.** Three Bernoulli draws/step; all three rates FIXED at ODD-baseline values — no calibrated ZI parameters. `n_zi=30` at introduction | ZI provides the steady background flow that absorbs aggressive market orders and smooths per-step impact |
| D14f | (superseded by D14g) VolatilityTrader dropped from active set | VT iterations produced burst-driven heavy tails; dropped to simplify |
| D14g | (superseded by D14j) VolatilityTrader re-introduced (tanh-saturated σ_t activity) | Re-added to chase long-horizon clustering the no-VT model lacked |
| D14h | (experiment, superseded by D14i) `n_zi` set to 0 | Raised clustering modestly but destroyed Hill and inflated ret_std |
| D14i | **`n_zi` set to 10** — partial noise floor | Middle ground between D14e's 30 and D14h's 0 |
| D14j | **VolatilityTrader REMOVED from the model.** Class deleted from `agents.py`; `n_volatility` / `vt_alpha_market` / `vt_alpha_limit` removed from `ModelParams`; `SimContext.vol` / `z_vol_s` removed; `simulation.py` no longer loads `sigma_t` / `z_vol_s`; `data/v_heston.py` retired (V_t now comes from the plain-GBM `V_smooth` column). Population 54 → 44; theta 5-d → (with D13f) 4-d | VT was a Gao-2023 Chiarella-Heston construct trading an exogenous Heston `σ_t` that was not the volatility of anything else in the model — `V_t` is plain GBM, the Heston `σ_t` drove only the VT, and even the leverage ρ was "agent-side only." Its primary justification (long-horizon clustering) was already conceded as structurally unreachable (D18f). The procyclical-volatility channel it nominally provided is better delivered *endogenously* by the Stage-4+ CCP-tier fire-sale feedback (margin call → liquidation → price impact → vol → margin), which is the systemic-risk mechanism the thesis is about. Removing VT collapses the model to a classic four-type ABM (FT/MT/MM/ZI) — simpler and far more defensible |
| D15 | (deferred) Client-clearing tier: client traders attached to CMs | User extension; ODD has no client tier — re-introduced at Stage 4+ |
| D16 | (history — VT removed under D14j) Gao 2023 Chiarella-Heston VT, probability-scaled by `(σ_t/σ_v)·|Z^S|` | The `|Z^S|` multiplier washed out σ_t persistence |
| D16b | (history — VT removed) VT activity depended only on `σ_t/σ_v` (linear) | No saturation — extreme-σ_t windows produced runaway activity |
| D16c | (history — VT removed) VT activity tanh-saturated: `p = vt_alpha · tanh(σ_t/σ_v)` | The final VT form before removal (D14j) |
| D16d | (history — VT removed) a σ_t-rescaling change reverted after a misdiagnosis was corrected by a smoke test | — |
| D17 | Calibration ordered before margin (Stage 8 before Stage 4) | Lock market dynamics before adding capital-constraint feedback |
| D18 | Calibration attribution: surrogate-assisted SMM (XGBoost regressors; LHS + greedy active-learning; multi-start L-BFGS-B). Not pure Lamperti (classification) | Documented at top of `calibrate.py` |
| D18b | **Per-regime independent calibration.** Calm and stressed are two separate optimisation problems — no shared parameters. Per-regime LHS caches; combined output nested under `results[regime]` | The regimes are economically distinct — forcing shared parameters introduces compromise instead of fit |
| D18c | **HFABM-literal loss + two-stage refinement.** Loss `D(θ) = Σ_c Δ_c / σ²_c` is HFABM Gao et al. (2022) eqs 4-9 literal: L1, grouped components (ΔHill, ΔV, ΔACF1; ΔACF2 dropped — D18f); kurtosis dropped (Hill is the robust tail measure); weights `1/σ²_c` from a 60-min-block bootstrap (Franke & Westerhoff 2012; Künsch 1989); Stage 2 a tight LHS grid search on the TRUE simulator around the surrogate optimum | Brings the loss in line with HFABM Gao et al. (2022) §4.1.4 + §4.2 |
| D18e | **ACF lag centers are {30, 60, 90} (1-min cadence).** At our 1-min cadence the same lag numbers HFABM uses at 100 ms cover 30 min – 1.5 h — an intraday span matching the margin-cycle horizons relevant to CCP work. Each "lag X" is a 3-lag average. Lag 1 dropped from ΔACF1 | HFABM's lag set is cadence-specific — match the horizon span, not the lag numbers |
| D18f | **ΔACF2 (volatility clustering) DROPPED from the loss.** The loss is 3-component (ΔHill + ΔV + ΔACF1). The individual `acf_absr_*` moments are still computed and printed as diagnostics | Long-memory absr clustering at lags 30/60/90 is structurally unreachable: no agent in the FT+MT+MM+ZI model carries a persistent volatility state (the two-cohort MT split and the VT, both of which targeted it, were removed — D13f/D14j). Keeping ΔACF2 in the loss only injects an unsatisfiable target. Documented as an accepted limitation (§6) |
| D19 | (folded into D18c) loss-scaling history — per-moment LHS-std normalisation, then HFABM bootstrap-variance weights | See D18c |
| D18e (revised) | **ACF lag battery moved to where the empirical signal is.** ΔACF1 (return ACF) lags `{30,60,90}` → `{1,5,10}`; ΔACF2 (\|return\| ACF) lags `{1,30,60,90}` → `{1,10,30,60,90}`. Smoothing changed from forward 3-lag `{c,c+1,c+2}` to CENTRED 3-lag `{c−1,c,c+1}` (F&W 2012; lag 1 → `{1,2}`) | The empirical ES 1-min return ACF is flat (~0) at 30/60/90 in both regimes — the prior ACF1 lags carried no information. The real signal is short: a bid-ask/microstructure term at lag 1 and a transient-impact mean-reversion peaking near lag ~9 in stress (−0.05, −5.6 SE). ACF2 (\|r\| clustering) is significant at every lag; `{1,10,30,60,90}` samples the level, the steep knee and the slow tail |
| D18f (reversed by D18g) | ΔACF2 had been dropped from the loss | — |
| D18g | **Calibration loss rebuilt — Franke per-moment standardisation; ACF2 restored.** Loss is `D(θ) = ΔHill + ΔV + ΔACF1 + ΔACF2` (4 components, equal weight). Each component is the mean over its moments of the Franke-standardised distance `|m_sim − m_hist| / s_i`, where `s_i` is moment `i`'s empirical block-bootstrap sampling SD (Künsch 1989 moving-block, block 390 ≫ longest lag 91 — the prior block 60 was shorter than the lag-90 moment, a bug). `s_i` is computed once per regime and fixed. Empirical targets and `s_i` are computed on the BBO **mid** `(best_bid+best_ask)/2` (data/roll.py now emits a `mid` column; v_gbm `σ_v` also moved to mid) — matching the simulator's own mid observable. ΔACF2 is back in the loss | The prior loss `Σ_c Δ_c/σ²_c` weighted by the variance of the grouped *absolute* distance was scale-pathological: `ret_std`'s tiny bootstrap variance handed ΔV a weight ~1e9× ΔHill's, collapsing the stressed Hill to 1.14. Standardising each moment by its own sampling SD makes every component dimensionless (a count of sampling SDs), so the four are comparable and equal-weighted, and `D(θ)` is **comparable across model versions** (s_i and the targets are fixed empirical quantities). Franke & Westerhoff (2012) inverse-sampling-variability weighting; HFABM Gao et al. (2022) eqs 3–10 grouping (this repo's loss uses |r| for ΔACF2 where Gao use r²). ΔACF2 is restored as a structural-gap benchmark — the FT+MT+MM+ZI layer on a plain-GBM V_t reproduces no clustering, so ΔACF2 currently measures that gap rather than being optimisable; keeping it in the loss makes `D` track the gap a future clustering mechanism would close |
| D20 | **Limit-order placement depth unified and calibrated.** ZI and MT both rest limit orders k ticks from the mid, `k ~ LogNormal(μ, depth_sigma)` with `depth_sigma = 0.3` structural and the distribution mean `depth_mean` calibrated (one shared parameter, per regime; `μ = ln(depth_mean) − depth_sigma²/2`). Supersedes (a) the ZI L2-geometric depth `k ~ Geometric(p_zi)` (D3) and (b) the MT signal-driven depth `k = max(1, round(k_base·σ_v/|M_t|))` (D13d). `k_base` removed from `ModelParams` and the calibration loop; `p_zi` removed from `ModelParams` (the `P_ZI` dict stays as a book diagnostic). MT's signal `M_t` now drives only the order side and the activation gate | The L2 `p_zi` was fit on placement depth from the *event-time* mid, but the agents anchor to the 1-min-stale previous-step mid — intra-minute drift (~2 ticks calm / ~17 ticks stressed) means the L2 fit is not the quantity the agent uses, so a calibrated depth is more honest than a mis-anchored offline fit. Unifying ZI and MT removes `k_base` and the signal-driven depth, whose only role (tightening placement as the trend strengthens) added complexity with no calibration moment constraining it. `depth_sigma` pinned at 0.3 (HFABM placement family). D3 had rejected a log-normal *fit to L2* for a tick-rounding artifact; D20 does not fit L2 — `depth_mean` is calibrated against the return moments — so that objection does not apply |
| D21 | **MarketMaker dropped — FT/MT/ZI order flow drives price.** Population is now 50: 20 FT + 10 MT + 20 ZI (with D23), `n_mm = 0`. The `MarketMaker` class stays defined-but-dormant in `agents.py`; `mm_*` params, `MM_QTY`, `ETA/GAMMA` go dormant. The three ZI rates (`zi_alpha, zi_mu, zi_delta`) ENTER the calibration loop — theta is now 7-d per regime: `ft_alpha, mt_alpha, depth_mean, mt_lambda, zi_alpha, zi_mu, zi_delta` | At the 1-min cadence 4 MMs *became* the price-setter: the D10g V_t-anchored MM pinned the mid to V_t (`ret_std` matched but the agent parameters had ≤4% traction on every moment — the calibration was degenerate); a last-mid-anchored MM froze the price (`ret_std` collapsed ~10×). HFABM's mid-quoting MM works only because its 100 ms cadence has many participants driving price. Dropping the MM lets FT/MT/ZI flow drive price (the Chiarella / Majewski excess-demand mechanism the thesis is built on). Verified: with the MM gone the agent parameters swing `ret_std` ~10× and `kurt` ~50× — a real optimum exists, so the calibration does genuine work. ZI is now a primary price/liquidity driver, hence its rates are calibrated |
| D22 | (trialled, reverted by D23) FT dead-band set to a per-agent Simudyne Δ ~ U[0.01·V, 0.10·V] (replacing the flat `ft_threshold_bps`) | Intended to let price drift further from V_t before the FT corrects. But with σ_fundamental at the daily scale the FT reservations cluster within ±0.6% of V_t, far tighter than the 1–10% band — so when price drifted, *every* FT crossed its band at once, a synchronised snap-back wave → kurtosis ~4600, uncalibratable. (σ_fundamental at the *annual* scale was also trialled — it de-synchronised the FTs but widened the belief cloud to ±28% of V and blew `ret_std` up 80–300×.) |
| D23 | **FT dead-band removed; ZI count doubled 10 → 20.** The FT acts whenever `reservation ≠ mid` (`side = sign(reservation − mid)`), gated only by `ft_alpha`; `ft_threshold_bps` removed from `ModelParams`. ZI count 10 → 20 (population 40 → 50) | The dead-band is unnecessary: the persistent `z_score` already supplies FT heterogeneity (D6b) and the per-agent-reservation trigger keeps flow two-sided (D9c) without a band — a flat band (D9b) and the wide Simudyne band (D22) added complexity without earning it. Doubling ZI densifies the near-mid book: it removed the lag-1 overshoot-revert choppiness (`acf_r₁` −0.14 → ≈ 0, matching the empirical efficient-market signature) and pulled `ret_std` from ~4× toward ~2.4× the target. σ_fundamental kept at the daily scale (D7b) |
| D24 | (trialled, removed by D25) **VolatilityTrader re-introduced** — an endogenous-volatility noise trader whose order flow scales with the model's own realised volatility `σ̂_t = EWMA(\|mid return\|)`, maintained on `SimContext` (decay `vt_lambda`). Tested in four forms: random-side market orders w.p. `vt_alpha·tanh(σ̂/σ_v)`; `σ̂`-scaled random market orders (MDH size scaling); `σ̂`-scaled limit orders; `σ̂`-scaled FT/MT order size | Aimed at the volatility clustering the FT+MT+ZI layer lacks, fixing the D14j flaw by keying on the model's *own* vol (not an exogenous Heston). The `σ̂_t` state IS strongly persistent (acf 0.97/0.64/0.26) — but **no order channel transmits that persistence to `\|r\|`**: random market orders wash out on aggregation (more VTs → *lower* `ret_std` — they trade against each other), and `σ̂`-scaled limit/size add resting depth → damp. `acf_absr` stayed ≈ 0 in every variant. Removed |
| D25 | **V_t is a Merton jump-diffusion (reverts D12); VolatilityTrader removed.** `Δlog V = (μ − ½σ_d²) + σ_d·Z + J`, `J` a jump w.p. `λ/390` of size `N(0,δ²)`; `λ` fixed at the ODD 3/day, `σ_d` and `δ` calibrated per regime from the empirical variance + excess kurtosis (`data/v_gbm.py`). The ODD §Stochasticity jump structure. The VT (D24), `σ̂` machinery and `vt_*` params are deleted; theta is back to 7-d | A plain GBM has zero excess kurtosis; jumps give V_t the empirical leptokurtosis, and were hoped to induce clustering via the FT price-catch-up after each jump. Outcome: V_t now carries the right kurtosis, but at the MID both the kurtosis (~220 calm / ~300 stressed) and the still-absent clustering (`acf_absr` ≈ 0) are dominated by the no-MM microstructure layer's own large, i.i.d.-timed bursts, which drown the V_t-jump signal. Volatility clustering remains structurally unreached — the verified conclusion after ≈7 mechanisms (the MT two-cohort split, the VT in four forms, σ̂-scaled sizing, jump-V_t) — confirming D18f's concession. It is deferred to the Stage-4 fire-sale feedback (the thesis's genuine endogenous-volatility channel). The jump-diffusion V_t is kept regardless: it is the ODD-faithful fundamental and gives V_t correct tails |
| D26 | **ΔACF2 removed from the calibration loss** (re-instates D18f; reverts D18g's restoration). The loss is now 3-component: `D(θ) = ΔHill + ΔV + ΔACF1`. The `acf_absr_*` moments stay in `MOMENT_NAMES` — computed, surrogate-trained, and printed in the validation table plus a grouped-ΔACF2 diagnostic line — but ΔACF2 is no longer summed into `D` | D24/D25 confirmed volatility clustering is structurally unreachable: `acf_absr` is pinned ≈ 0 across the whole parameter space. As a loss term ΔACF2 is then a near-constant ~18 — it cannot be optimised, does not inform the argmin, burdens the XGBoost surrogate with an unfittable moment (R² ≈ 0 on `acf_absr_*`), and inflates `D` so the reachable misfit is harder to read. Removing it makes `D` an honest, interpretable measure of *reachable* misfit; the clustering gap stays visible via the diagnostic. Accepted limitation — clustering deferred to Stage 4 (§6) |
| D27 | **MT gains a market-order branch — `mt_mu` calibrated (reverts D13d's always-limit; theta now 8-d).** `MomentumTrader.submit_orders` runs two independent activation gates: a passive mid-anchored limit w.p. `mt_alpha` (unchanged) and a trend-direction MARKET order w.p. `mt_mu`, qty `~ U[qty_min, qty_max]`. `mt_mu` is a SEPARATE Bernoulli rate (parallels ZI `zi_mu`) — **not** the D13c natural-marketable cliff `k ≤ 0 → market` that produced synchronised 10-MT bursts (calm kurt > 2400). Added to `ModelParams` (default 0.0, back-compatible), `calibrate.py` `PARAM_BOUNDS` `(0.0, 0.20)`, `_theta_to_params`, and `globals.CALIBRATED`. Theta is 8-d: `ft_alpha, mt_alpha, mt_mu, depth_mean, mt_lambda, zi_alpha, zi_mu, zi_delta` | Trialled as a clustering mechanism — directional chartist flow should not wash out on aggregation the way the VT's random market flow did (D24). **It does not deliver clustering**: a prototype `mt_mu` sweep {0, .05, .1, .2, .5} (6 runs × 2 days/regime) lifts `acf_absr` only at lag 1 (calm 0.044 → 0.067, stressed 0.027 → 0.13) while lags 10/30/60 stay flat ≈ 0 — the lag-1 bump is just consecutive-step momentum trades, no long memory. So D26's accepted-limitation concession stands. `mt_mu` is retained as a calibrated parameter for a DIFFERENT reason it earned in the sweep: it has genuine traction on the tail moments — excess kurtosis falls toward the empirical level (calm 195 → 89, stressed 414 → 50) and the Hill index rises (lighter tails: calm 0.56 → 0.88, stressed 0.54 → 1.10), both in the loss via ΔHill. The market flow erodes the i.i.d.-timed burst pathology. Upper bound held at 0.20: at `mt_mu = 0.5` the trend-direction flow corrupts the return ACF (`acf_r₁ → −0.13` stressed, a bid-ask-bounce artifact) — within `[0, 0.20]` `acf_r₁` stays ≈ 0 and ΔACF1 self-regulates `mt_mu` away from that corner |
| D28 | **Central-clearing tier scaffolded — BCM / NBCM / CCP, balance sheets, counterparty links.** 10 FTs re-cast as `BankingClearingMember` (FT subclass, trades own account); 5 `NonBankingClearingMember` added (no own position, no LOB orders); 1 `CentralCounterparty` (`model/clearing.py`). Each CM carries a `BalanceSheet` (ODD §Agents); CM↔CCP bidirectional links + client-book assignment (plain FT + MT as clients; ZI direct). `ModelParams.n_bcm/n_nbcm` (default 0 → `calibrate.py` unaffected); `Simulation` gains an optional `ccp` arg; `BaseTrader.clearing_member_id`; `globals.CCP_CALIBRATION` holds the ODD §Calibration constants. Margin / default-fund / waterfall MECHANICS deferred to later stages | The thesis's client-clearing question requires the CM/CCP tier. Built as a scaffold so the calibrated market layer is untouched: the BCM trades own-account FT-style, so 10 FT + 10 BCM ≡ the 20-FT layer the 8-d calibration runs against — calibration carries over with no re-run. **Deviations from the Simudyne CCP ODD, flagged:** (a) the ODD's CMs trade for their own account with no client tier — the thesis adds client clearing (BCM/NBCM carry client books); this client-clearing layer IS the thesis contribution. (b) 10 BCM + **5** NBCM, vs the ODD's 10 + 10 (the user's design — NBCM is the scarcer pure-intermediary type). (c) NBCM holds **no own position** (the user's design; ODD's NBCM does trade own-account — it differs from BCM only in stress behaviour). (d) `df_interval = 390` (one thesis RTH day) vs the ODD's 510. (e) Round-robin client assignment is a neutral placeholder topology — heterogeneous client-book concentration is a later scenario choice |
| D29 | **Margin stage — variation-margin cycle + futures-margin accounting + half-BCM client books.** `Simulation._margin_cycle` runs every `margin_interval = 60` ticks (ODD §Step Sequence step 11 / §Mech #3): for each CM it settles the cleared book's mark-to-market move since the last cycle into cash (variation margin), recomputes IM = `im_percent`·exposure and MM = `mm_percent`·IM, flags `call_indicator` when `\|VM\| > (1−mm_percent)·IM` (5% of IM), records the capital ratio, and sets `has_defaulted` on cash exhaustion (the 5-level waterfall stays deferred). One `clearing_history` row per CM per cycle → `output/clearing_{regime}.csv`; `clearing_analysis.ipynb` reads it. `n_bcm_with_clients` (=5) — only half the BCMs carry a client book, the rest are own-account only. **Futures-margin accounting:** `_apply_fill` no longer moves cash for clearing members — ES is a future, so a fill posts only the position and cash is settled by the VM cycle (`cash = starting capital + Σ VM`); non-CM cash keeps the old nominal accounting (inert). | The thesis's capital-adequacy question needs the margin mechanism live (ODD §Mech #2-3). The capital ratio binds only when variation margin depletes cash and cash is commensurate with the book — at the ODD's `U[5B,10B]` capital (kept — the user's choice) the ratio sits ~10³-10⁴ in calm and the 8% floor is exercised only under stress, matching the ODD's normal-regime expectation (`default_count_normal == 0`). The old full-notional `cash -= price·qty` was stock accounting — wrong for a future and double-counts once VM is added; corrected to futures-margin accounting (CM order flow is unchanged, so the calibrated market layer and `calibrate.py` — which sets `n_bcm = 0` — are untouched). Half the BCMs carry clients (the user's design): this seeds the client-book concentration heterogeneity the thesis studies. Deferred still: cover-2 default fund, 5-level waterfall, position auction, BCM fire-sale / NBCM stop-out, client-trade novation (client books remain inert — `client_positions` at 0) |
| D31 | **MarketMaker re-introduced — HFABM-style mid-anchored at low qty; `mm_qty` calibrated.** Reverts D21's MM drop. `MarketMaker` class rewritten in `agents.py`: anchor = prev-step mid (v0 fallback at t=0), random per-side tick offset `d ~ U{0, mm_p_edge}`, NO inventory skew, ALWAYS quotes. `n_mm = 2` structurally; `mm_qty` (per-side quote size) becomes the 9th calibrated parameter with bound `(1.0, 5.0)` (rounded to int at submit). Added to `build_traders`, `calibrate.py` `POP` (`n_mm = 2`), `_theta_to_params`, `globals.CALIBRATED` (placeholder 2). LHS / stage-2 caches deleted. | A prototype MM sweep (`outputs/test_mm.py`: 2 styles × n_mm ∈ {1,2,4} × mm_qty ∈ {1,5,10}) showed a low-qty MM cleanly closes the residual kurtosis gap the no-MM layer left: calm 2 MMs @ q=1 cut kurt 126→44 and lifted Hill 2.49→3.10 (essentially the target 3.00), at the cost of a small `acf_r_1` bounce (~−0.04). The D21 pathology was MM **size** (qty 50 dominated price formation), not MM **presence**; at qty ~1-5 the MM provides continuous near-mid liquidity without dominating. The MID anchor (rather than D10g's V_t anchor) is essential — V_t-anchored at any qty pins the mid to V_t. Inventory skew (Avellaneda-Stoikov) was tested and rejected: Skew + low qty produced positive `acf_r_1` (especially stressed at +0.5–0.7, the inventory-skew trend-amplification). HFABM (no skew) keeps `acf_r_1` near zero. `n_mm` fixed structurally because the sweep showed `n_mm` and `mm_qty` are roughly substitutable on the moments (total MM liquidity ≈ `n_mm · mm_qty` controls); calibrating an integer `n_mm` is awkward for L-BFGS-B and adds nothing the surrogate can't pick up via `mm_qty` |
| D38 | **VolatilityTrader added — Gao 2023 Chiarella-Heston vol-noise channel.** `agents.py` adds `VolatilityTrader` class: every step, each VT submits a market order with random ±1 direction and `qty = max(1, round(vt_qty_base · σ_t / sigma_v))`. Five VTs added (`n_vt = 5`); `vt_qty_base = 2` structural. LOB-form of Gao §3.1.3's continuous-time vol-scaled demand `ω·√Σ_t · dW_t^S`. Re-introduces the D24/D25 VT trial under the D37 no-MM context. | The D24/D25 VTs failed because the D10g MM (qty=50) clamped spread and absorbed VT impact. Post-D37 (no MM, FT/MT every step) the spread is free to widen with σ_t and VT market orders walk it. Cont 2005 §3 surveys clustering mechanisms — `acf_absr` slow decay comes from vol-scaled order flow + heavy-tailed activity-regime durations; Gao gives the noise channel, Cont the explanation. Smoke (5d × 4 seeds calm @ placeholder θ + D33-D38 stacked): **kurt 13 → 20** (target 19.9 — essentially matched), **ret_std 2.93e-4** (target 2.97e-4 — essentially matched), lag-1 `|r|` ACF 0.189 → 0.212. Headline moments now hit target without calibration. Long-lag clustering still capped at ~0.04 — Gao's VT is a short-horizon vol-noise channel; long-memory probably needs Cont's threshold-inertia mechanism or σ_t piped into more agents (MT activation, ZI rate) |
| D37 | **MM dropped again — `n_mm = 0`; `mm_qty` removed from PARAM_BOUNDS. Theta = 7-d.** `run_simulation` and `calibrate.py` `POP` both set `n_mm = 0`. The `MarketMaker` class stays dormant in `agents.py` (re-introducible by flipping `n_mm`). | The user's hypothesis: with FT/MT trading every step (D36) the dense order flow gives enough natural liquidity, AND removing the MM lets the spread widen with σ_t (the channel D31's tight MM was clamping). Placeholder-θ smoke (5d × 4 seeds calm, D34/D35/D36/D37 stacked): kurt **13** (no return of the pre-D31 800+ burst pathology — the dense FT-every-step + 15 MTs provide near-mid liquidity); ret_std **2.89e-4** (calm target 2.97e-4 — essentially perfect); lag-1 `|r|` ACF **0.189** (with MM was 0.144); lag-10 `0.048` (with MM 0.014, **3.4× lift**); lag-30/60/90 all ~2× lifts. Best market-layer numbers of the project at an unoptimised θ. Supersedes D31's MM re-introduction; the kurt-fix mechanism is now D36's FT/MT-every-step instead of an MM. (D31's MM-vs-no-MM trade-off was: D31 needed dense liquidity to stop the burst pathology, but the MM provided it via mid-pinning, which clamped spread variability and blocked long-memory clustering. D36 provides dense liquidity via agent order flow without the pinning, breaking the trade-off.) |
| D36 | **FT and MT trade every step — `ft_alpha` / `mt_alpha` pinned at 1.0 and removed from the calibration loop.** ODD-faithful (§Step Sequence step 3: "BCM, NBCM, ZI submit BuyOrder / SellOrder messages... based on capital ratio check and limit price vs market price comparison"). `FundamentalTrader.submit_orders` drops the `if rng.random() < params.ft_alpha:` gate; `MomentumTrader.submit_orders` drops the `if rng.random() >= params.mt_alpha: return` gate (market branch is still gated by `mt_mu`). `ModelParams.ft_alpha` and `mt_alpha` now default to 1.0; PARAM_BOUNDS shrinks to 8-d (`mt_mu, depth_mean, mt_lambda, mt_lambda_long, zi_alpha, zi_mu, zi_delta, mm_qty`). | Pass-2 / Pass-3-with-D35 LHS still capped lag-10+ `acf_absr` near zero; the bid-ask is structurally tight because MM and ZI keep the spread narrow at all σ_t (the user's diagnosis). Doubling FT order arrival (calm `ft_alpha` was ~0.49 → 1.0) densifies near-mid flow and gives FTs a constant presence to which σ_t-driven belief shifts can register — so book churn directly tracks vol. Placeholder-θ smoke (5d × 4 seeds calm, D34+D35+D36): lag-1 `|r|` ACF 0.077 → 0.144 (2× lift from D35 alone), kurt 14 → 16, ret_std 2.36e-4 → 2.65e-4 (closer to 2.97e-4 target); long-lag still flat — calibration will show how far the new channel reaches |
| D35 | **Long-horizon MT cohort added — per-agent `lambda_decay`, `mt_lambda_long` calibrated.** `MomentumTrader` gains a per-agent `lambda_decay: float` field; `submit_orders` uses `self.lambda_decay` instead of `params.mt_lambda`. `ModelParams.n_momentum_long: int = 0` and `mt_lambda_long: float = 0.02` added. `build_traders` instantiates `n_momentum` short-horizon MTs (`lambda_decay = params.mt_lambda`) and `n_momentum_long` long-horizon MTs (`lambda_decay = params.mt_lambda_long`); `run_simulation.main()` sets `n_momentum_long = 5` (15 MTs total). `calibrate.py` `POP` adds `n_momentum_long=5`; `mt_lambda_long` enters `PARAM_BOUNDS` at `(0.005, 0.05)` (EWMA half-life ≥ 14 min, up to ~140 min). Theta becomes 10-d. | Pass-1 / Pass-2 LHS reachability showed lag-1 `acf_absr` near target (~0.27 vs 0.31) but lags 10/30/60/90 capped at 0.04-0.06 — short-memory clustering reaches the mid (V_t-jump → FT-herd burst), long-memory does not. The OU σ_t has 55-min half-life; the existing MT has `mt_lambda` bounded above 0.02 (EWMA half-life ≤ 35 min), too fast to track hour-scale vol persistence. A second MT cohort with `mt_lambda_long` ∈ [0.005, 0.05] tracks return momentum over hourly windows, so periods of persistently high σ_t generate persistent directional flow that propagates `|r|` clustering at long lags. Revisits D13e's two-cohort split — that one was dropped (D13f) because ΔACF2 was out of the loss and the optimiser traded the long cohort away; with ΔACF2 reinstated (D32) the long cohort now has incentive. Placeholder-θ smoke (5d × 4 seeds): lag-1 `|r|` ACF 0.048 → 0.077, lag-90 0.007 → 0.037 — modest lift across the curve; the Pass-3 LHS reachability is the real test |
| D34 | **σ_t piped into FT belief width** — FT reservation `V_t + z · σ_fund_t` now uses `σ_fund_t = ft_sigma_c · ctx.sigma_t · v0` instead of the static `params.sigma_fundamental`. Implementation: `Simulation.__init__` loads `sigma_t` from `fv_csv` (added by D33) into `self.sigma_t_array`; `SimContext.sigma_t` field; `FundamentalTrader.submit_orders` reads `ctx.sigma_t` (fallback to `params.sigma_v` when SV is off / column missing — back-compatible). This matches the Deloitte convention `theta_v = sigma_fundamental` literally and was the second pass of the user's "first without, then with" split. | The Pass-1 calibration confirmed the diagnosis: validated calm D = 38.84 with ΔACF2 = 20.1 dominant (LHS reachability for `acf_absr` lags 10/30/60/90 all stuck near 0 even though V_t has clustering at those lags — D33 verified). With FT belief width static at the long-run scale, the SV-V_t persistence had no channel into the mid — FTs reacted to V_t with equal aggression regardless of vol regime, so the mid caught up to V_t in a step and σ_t persistence flattened. With σ_t scaling FT cloud width, FTs are less reactive during high-vol windows (Deloitte mechanism), allowing σ_t persistence to spread out over multiple mid-return steps. Smoke at placeholder θ: kurt **800 → 14** (the wider FT cloud damps the burst-spike-response pathology), `ret_std` 2.36e-4 vs target 2.97e-4, lag-1 `|r|` ACF +0.06. Whether the optimum at Pass-2 lifts long-lag clustering is what the re-calibration tests — if ΔACF2 still dominates after Pass-2, the next step is piping σ_t into MT/MM/ZI as well |
| D33 | **V_t becomes a Stein-Stein-style stochastic-vol jump-diffusion — Vasicek-OU on σ_t.** `data/v_gbm.py` rewrite: σ_t follows `dσ = α(θ − σ)dt + σ_vol·dW`; V_t evolves `Δlog V = (μ − ½σ_t²) + σ_t·Z + J` with the existing Merton jumps. Method-of-moments calibration on a 30-min realised-vol time series from 1-min ES returns: θ = mean σ_per_min, σ_vol²/(2α) = Var, `α = −ln(ρ_30m)/30`. θ is then re-anchored so `E[σ_t²] = σ_d²`, keeping total return variance equal to the D25 jump-diffusion-only calibration (the SV layer adds clustering without inflating the vol level). Path schema adds a `sigma_t` column to `fv_{regime}.csv` for diagnostics. | The thesis's residual loss is dominated by ΔHill and (under D32) ΔACF2 — both are tail / volatility-clustering features. The constant-σ Merton model (D25) injects clustering only through the burst of FT price-catch-up after each jump, so `acf_absr` was effectively pinned near zero and clustering was the "structurally unreachable" gap D26 acknowledged. Deloitte's CCP webinar parameter set names per-asset `alpha_v / theta_v / sigma_v` with `theta_v` equal to the static `sigma_fundamental` — an OU process driving σ, not a rates Vasicek. Calibration check on the new V_t paths (no agents yet, just the fundamental): calm `acf_absr` at lags 1/10/30/60/90 = 0.25/0.23/0.19/0.13/0.10 vs empirical 0.31/0.25/0.21/0.18/0.16 (model captures ~70-95% of the empirical clustering out to lag 30, decays a bit too fast past lag 60 because a single OU has one timescale; a two-factor SV could close that if needed). kurt and ret_std match empirical at the V_t level (calm kurt 19.7 vs 19.9; calm ret_std 3.04e-4 vs 2.97e-4). Whether this V_t clustering propagates to the mid is what the post-D33 ABM calibration tests — the D31 mid-anchored MM is now the channel that should let it through |
| D32 | **ΔACF2 reinstated in the loss — 4-component D(θ), reverts D26.** `COMPONENT_NAMES = ("Hill", "V", "ACF1", "ACF2")`. The `acf_absr_*` moments at lags {1,10,30,60,90} (already computed for the diagnostic) now feed into the summed loss. | D26 dropped ΔACF2 because volatility clustering was structurally unreachable in the no-MM market layer — `acf_absr` was pinned near zero across the whole parameter space (D24/D25). The D31 mid-anchored MM is precisely the continuous-quoting mechanism Deloitte/Simudyne and HFABM assume when they calibrate clustering off a population that has both a momentum trader and an MM, so the structural pin is plausibly broken. Restoring ΔACF2 lets the optimiser pull on it; if clustering remains unreachable post-D31, ΔACF2 will again dominate the loss and the term can be dropped — but speculating one way or the other without giving the optimiser the lever is the worse failure mode |
| D30 | **ZI rate bounds tightened to literature ranges; `N_RUNS` 3→8.** `calibrate.py` `PARAM_BOUNDS`: `zi_alpha` `(0.02, 1.0)→(0.02, 0.50)`, `zi_mu` `(0.005, 1.0)→(0.005, 0.10)`; `zi_delta` unchanged. `N_RUNS` default 3→8. LHS / stage-2 caches deleted (sampled from the old box). | First 8-d calibration run (`run 150 30 2 3 30 60`) exposed two faults. (1) The `→1.0` ZI bounds — widened earlier — let the optimiser run `zi_mu` to 0.52 (calm) / 0.40 (stressed) and `zi_alpha` to 0.62 / 0.83, i.e. ~20× / ~5× the Cont-Stoikov 2008 / ODD §Calibration baselines (0.025 / 0.15). A market-order arrival rate of ~0.5 means half of all ZI activity is book-walking aggression → validated kurtosis 1204 (calm) / 772 (stressed) and a negative `acf_r_1` bid-ask bounce. The wide bound was the un-grounded deviation; the new caps keep market orders a clear minority of ZI flow (`zi_mu` ≤ 4× baseline, `zi_alpha` ≤ 3.3×). (2) `N_RUNS = 2` made the loss noise-dominated: the stage-2 grid optimum (D ≈ 12.9 calm / 9.7 stressed) did NOT reproduce at validation (D ≈ 16.0 / 16.7) for the *same* theta — the grid selects on favourable per-run noise. 8 seeds ≈ halves the moment sampling SD. The poor first-run theta (`ΔHill` ≈ 8-10 dominating) is NOT copied into `globals.CALIBRATED` — it is superseded by the re-run under the corrected bounds |

---

## 6 · Open items

- **Clearing tier — scaffold (D28) + margin cycle (D29) live; default
  fund / waterfall pending.** BCM / NBCM / CCP, balance sheets, the
  counterparty/link structure (`model/clearing.py`, §2.5) and the 60-tick
  variation-margin cycle all run; `clearing_analysis.ipynb` reads
  `output/clearing_{regime}.csv`. Next is **Stage 5 — cover-2 default
  fund**: `total_df` = sum of the 2 largest CM loss scenarios
  (`df_percent`·exposure), recalculated every `df_interval = 390` ticks,
  with the `ex_df_ratio = 10%` exchange SITG (constants in
  `globals.CCP_CALIBRATION`). Then **Stage 6** — the 5-level waterfall +
  position auction. Also pending: client-trade novation (client books are
  inert — `client_positions` at 0, so NBCM capital ratios are still
  `inf`), and the BCM fire-sale / NBCM stop-out at the stress stage. With
  ODD-scale capital the 8% floor does not bind in calm — capital adequacy
  is a stress-regime phenomenon (by design — the user kept `U[5B,10B]`).
- **D31/D32/D33/D34 Pass-2 re-calibration PENDING.** Theta is 9-d
  (`mm_qty` from D31); loss is 4-component (ΔACF2 reinstated, D32); V_t
  carries Stein-Stein SV (D33); FT belief width now scales with `σ_t`
  (D34). Re-run command: `python3 calibrate.py run 150 30 8 3 30 60`
  (≈2.5 h total). LHS / stage-2 caches deleted. **Pass-1 baseline (no σ_t
  feedback to FT)**: calm D = 38.85 (ΔACF2 = 20.1 dominant; lag-10+
  `acf_absr` LHS-unreachable); stressed D = 17.22. **Pass-2 expectation**:
  kurt drops (D34 wider-cloud damps burst-spike response — smoke
  800→14); ΔACF2 should drop materially as σ_t persistence spreads
  through the FT channel. If long-lag clustering is still LHS-unreachable
  in Pass-2, the next step is piping `σ_t` into MT / MM as well.
- **Structural heavy-tail gap (quantified by the first run).** Best
  achievable Hill is ≈1.5–1.7 vs the empirical ≈3.0–3.25 (`ΔHill` ≈ 8–10,
  the dominant loss term). The model *can* reach Hill 8–9 but only with a
  wide book (`depth_mean` 7–9), which mechanically produces a bid-ask
  bounce (`acf_r_1` −0.10…−0.18) — light tails and a tight book conflict
  in the no-MM mechanism. This is the D25 heavy-tail limitation, now
  measured; kurtosis (1200/770) is out of the loss so the optimiser does
  not control it. A kurtosis guard / the wide-book–bounce tension are open
  design questions if the post-D30 fit is still unsatisfactory.
- **Loss-weighting pathology — FIXED (D18g).** The loss is now
  `Σ_c Δ_c` with each moment Franke-standardised by its empirical sampling
  SD; ΔV no longer dominates (smoke: ΔV ≈ 0.3–1.4 sampling-SDs, balanced
  against ΔHill / ΔACF1 / ΔACF2). `D(θ)` is comparable across model
  versions.
- **Volatility clustering — confirmed structurally unreached; ACCEPTED
  limitation.** The market layer reproduces the volatility level and the
  flat return ACF but NOT `acf_absr` clustering. ≈8 mechanisms were tried
  and failed (D25/D27): the MT two-cohort split (D13e), the VolatilityTrader
  in four forms (D24), σ̂-scaled FT/MT sizing, the Merton-jump V_t (D25),
  and the MT market-order branch (D27 — `acf_absr` lifts at lag 1 only,
  long lags stay flat).
  The consistent reason — the no-MM microstructure layer produces large,
  i.i.d.-timed bursts that drown any clustering signal; random order flow
  washes out and σ̂-scaled liquidity damps. ΔACF2 stays in the loss as the
  documented structural-gap benchmark (D18g). Clustering is deferred to the
  Stage-4 fire-sale feedback — the thesis's genuine endogenous-volatility
  channel. The jump-diffusion V_t (D25) does give V_t itself the empirical
  excess kurtosis, but at the mid the kurtosis is microstructure-dominated.
  The stressed return-ACF lag-9 transient-impact reversion is likewise not
  reproduced.
- **Sync `empirical_analysis.ipynb`.** Its helpers (forward 3-lag
  smoothing, `close`-based returns, old ACF lags) no longer mirror
  `calibrate.py` after D18e-revised/D18g — needs an update pass.
- **Refresh `MM_QTY` from L2.** `globals.MM_QTY` uses placeholder values
  (50 calm, 10 stressed). Extend `data/p_zi.py`'s streaming pass to record
  mean top-of-book passive size per regime; re-run; copy into the dict.
- **Tests**: stage1/stage2 need rewrite against the current spec; stage3
  obsolete (CM-tier deferred).
- **CCP clearing tier (Stage 4)**: re-introduce BCM/NBCM/client topology +
  margin. The persistent-z FT (D6b) is load-bearing — it provides the
  concentrated-inventory cross-section margin calls bite on. The retained
  POV / `ac_schedule()` machinery (D10f/D10g) carries over to BCM fire-sales.

Closed:
- ✅ **Model simplification (May 2026)**: VolatilityTrader removed (D14j),
  MomentumTrader folded to single-type (D13f). Population 54 → 44; theta
  5-d → 4-d. Market layer is now a classic four-type ABM.
- ✅ Order-management refactor: replace-on-new for FT/MT, Bernoulli for ZI,
  hard `order_ttl = 10` ceiling (D5d).
- ✅ MM POV liquidation (D10f); AC schedule tabled (D10e); `data/impact.py`
  η/γ regression complete.
- ✅ HFABM-literal loss with bootstrap-variance weights; ACF lags moved to
  the 1-min-cadence set; ΔACF2 dropped from the loss (D18c/e/f).
- ✅ Per-regime independent calibration (D18b).
- ✅ V_t model: GBM (D12); KF diagnostic (D12c); jump-diffusion dropped.

---

## 7 · Data notes

- **Source**: DataBento, CME Globex MDP3 feed. Schemas: OHLCV-1m, BBO-1m
  (`data/`), MBP-10 (`../L2_data_thesis/`).
- **Roll**: ES quarterly (Mar/Jun/Sep/Dec); switch 8 calendar days before
  expiry (`data/roll.py`).
- **RTH**: 13:30–20:15 UTC (08:30–15:15 CT). Overnight Globex excluded.
- **Regimes** (σ_v / v0 windows):
  - **Calm**: 2019 full year (~264 RTH days). V₀ = 2463.50.
  - **Stressed**: 2020-02-24 → 2020-04-02 (COVID trough, ~29 days). V₀ = 3255.00.
- **MBP-10 sample windows** (for `p_zi`): calm 2019-06-03…06, stressed
  2020-03-16…19 (4 days each).
- **V_t source**: plain-GBM paths in `data/fv_{regime}.csv` (the `V_smooth`
  column). `data/v_gbm.py generate-all` writes them.

---

## 8 · How to run

```bash
# 1. Offline data calibration (one-shot)
python data/v_gbm.py calibrate                   # output/v_gbm_params.json
python data/v_gbm.py generate-all                # data/fv_{calm,stressed}.csv (V_t path)
python data/p_zi.py calibrate                    # output/p_zi_params.json (book diagnostic — D20)
python data/v_kalman.py calibrate                # output/v_kalman_params.json (diagnostic)
python data/impact.py                            # output/impact_params.json (η/γ; not wired in)

# 2. Run the simulation (one RTH day, 50 agents, calibrated theta)
python run_simulation.py

# 3. Inspect (notebook)
#    open analysis.ipynb, run-all — mid vs V, spread/volume, moments, agent state

# 4. Agent calibration (surrogate-assisted SMM, 8-d per-regime loop, D18b)
python calibrate.py targets                              # print empirical moment targets
python calibrate.py run                                  # both regimes independent
python calibrate.py run calm                             # calm only
python calibrate.py run stressed                         # stressed only
                                                         # → output/calibrated_params.json (nested by regime)
#    then copy results[regime].theta_stage2 into globals.CALIBRATED
```

**Heads-up before calibration runs:** if `PARAM_KEYS` changed since the last
run (any calibrated parameter added/removed), delete
`output/calibration_lhs_{calm,stressed}.csv` — the cached training data is
incompatible. The VT-removal/MT-fold refactor changed the theta layout, so
the existing caches **must** be deleted before the next run. `calibrate.py`
requires `xgboost` (with libomp) and `scipy` — run it locally, not in the
sandbox.

---

## 9 · Maintenance protocol

When code changes, update:
- §2 if agent behaviour changed
- §3 if `ModelParams` / `globals.CALIBRATED` changed
- §4 if a stage is closed or a new refactor opens
- §5 if a new deliberate deviation is added
- §6 when an open item closes or a new one opens
- §8 if instructions change

Keep tone terse and citation-anchored.
