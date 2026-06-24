> **HISTORICAL SNAPSHOT (2026-06-18).** This document records an earlier audit/state and is NOT the current model. For the current committed model and results see AGENT.md and THESIS_SYNTHESIS.md. Notably superseded since: BCM prop cap 6×→2×, house margin 20%→15%, leverage-balanced client→clearer assignment, client balance-sheet rebalance, and the close-out/GBM-ensemble findings.

# Clearing layer — comprehensive review and recommended final design

Synthesis of six parallel investigations (current code, the Simudyne ODD, real-world regulation/
methodology, CCP public disclosures, the academic modelling literature, and a live simulation
realism check). For each component it answers the four questions you asked: **how it is now**, **how
it is in the ODD**, **how it is realistically / in other papers**, and **how the hypotheses can be
tested with it**. Code claims carry `file:line`; real-world figures carry sources. Date: 2026-06-15.

---

## 0. Executive summary

**The clearing layer is well-grounded — most parameters are literally CME/EMIR numbers** (99% IM,
2-day MPOR, 8% capital floor, $50M NBCM floor, cover-2 default fund, SITG-before-pooled-DF ordering,
EMIR Art. 48 porting). The code substantially *upgrades* the ODD it extends (flat 20% IM → procyclical
VaR; bookkeeping fund → cash-prefunded; position auction → open-market fire-sale; + a client tier).

**Five concrete fixes for the "best final design"**, in priority order (detail in §5):

| # | Fix | Why | Effort |
|---|-----|-----|--------|
| 1 | **Cap / smooth stressed IM** | Reactive EWMA overshoots ~2.5× at COVID (24.9% vs ~10% real); IM/notional exceeds 100% at c≥4 — unphysical (no cap) | small |
| 2 | **Run the fixed-stress-fund arm** | DF is coupled to IM (SLOIM), confounding H2 solvency *and* inflating calm DF/IM to ~44% (real ~3-4%) | ~2 lines + re-run |
| 3 | **Lower SITG `ex_df_ratio` 10%→2-4% + relabel** | Real SITG is ~1-4% of the fund (CME ES ≈2.7%); "EMIR Art 45 = 25%" conflates %-of-capital with %-of-fund | 1 line |
| 4 | **Make L4 reachable** (near-floor BCM + directional client concentration) | L4 / client→NBCM→survivor-mutualisation — the most policy-relevant channel — never fires (BCMs over-capitalised 12-114× floor) | build change + re-run |
| 5 | **Add loss-location + collateral-demand decomposition; use paired stats** | Converts H1/H2 into the literature's metrics; enables H3 (Duffie-Scheicher-Vuillemey) | ~instrumentation |

Everything else (IM = 99% VaR floored, cover-2 SLOIM + 10% buffer, the five-level waterfall, EMIR-Art-28
volatility-floor APC, EMIR-48 porting, the fire-sale-as-contagion-channel) is faithful to practice and
should be kept and disclosed, not changed.

---

## 1. Component-by-component: now / ODD / real / verdict

### 1.1 Initial margin (IM)

- **Now:** `im_fraction(σ) = max(IM_FLOOR, z₉₉·σ_daily·√MPOR)`, `σ_daily = σ·√390`, `z₉₉=2.326`,
  `MPOR=2`, `IM_FLOOR=0.06` (`globals.py:233-244`). Three modes (`IM_MODE`): **reactive** (EWMA of
  realised 1-min sim returns, half-life 2 RTH days, gaps excluded — `simulation.py:196-204`), **static**
  (regime σ_v), **flat** (`IM_FLAT_FRAC`). Recomputed every 60 min (`simulation.py:203,475`). A separate
  `im_percent=0.20` is the client *position cap* (5× leverage), not the CCP IM.
- **ODD:** flat `imPercent = 0.20` of gross notional; no VaR scan; capital ratio = `cash/|position|`
  (gross). (ODD §Calibration L183, §Mech #3.)
- **Real:** CME **SPAN / SPAN 2** (filtered historical VaR, ≥10-yr lookback + liquidity/concentration
  add-ons). Confidence **99% for listed futures** (EMIR RTS Art 24; CME Financial Safeguards). **MPOR
  2-day** non-OTC minimum (EMIR RTS Art 26). Real ES IM ≈ **3.1-4.7% of notional** (~$11.7-18k on ~$379k);
  in March 2020 CME raised ES IM ~6× in 3 weeks ($6,300→$12,000, ~+90%). APC: EMIR RTS Art 28 — pick ≥1 of
  {25% buffer / 25% stressed-weight / 10-yr vol floor}.
- **Papers:** Glasserman-Wu 2018 (risk-sensitive vs through-the-cycle; the quantile-gap/buffer);
  Aymanns-Farmer 2015 (procyclical-VaR amplifies the leverage cycle); BCBS-CPMI-IOSCO 2025 (the
  responsiveness metric ΔIM%/Δvol%).
- **Verdict:** calm IM (7-9% in-sim, §4) is **realistic** and the parameter set is literally
  CME/EMIR. Two issues: **(a)** stressed reactive IM **overshoots ~2.5×** (24.9% at c=1 vs ~10% real)
  because the EWMA of r² picks up the model's fat tails faster than a clean intraday vol estimator;
  **(b)** there is **no IM cap**, so IM/notional > 100% at c≥4 (the notional shrinks as clients close
  out while σ keeps rising) — an artifact. The procyclicality *direction* is correct (this is the H2
  mechanism); only the absolute stressed level is off. `z₉₉=2.326` is a Gaussian quantile vs real
  *historical* VaR — a conservative-direction simplification to disclose. The `IM_FLOOR` is exactly
  EMIR Art 28 option (c).

### 1.2 Variation margin (VM)

- **Now:** fill-price settlement `P/L = pos·Δmark + Σq·(mark−fill)`, ×`VOLUME_LOT×CONTRACT_USD`,
  every 60 min, clients novated to the CM each cycle (`simulation.py:395-399,465-469`). Symmetric (gains
  paid). The fire-sale slippage is realised (the D61 change).
- **ODD:** "PnL vs maintenance-margin threshold, settle VM, every 60 ticks" — no fill-price formula.
- **Real:** CME marks to market **≥ twice daily** (intraday + EOD) plus continuous intraday monitoring
  and **ad-hoc intraday calls** (heavily used Mar-2020); PFMI floor = daily. Futures VM is
  **settled-to-market** (cash, title passes, MtM resets) — which the model matches. CME settlement
  variation averaged ~$3.2B/day (2011), peaked $18.5B (Oct-2008).
- **Papers:** Cont 2017 (clearing transforms counterparty risk into *liquidity* risk via VM); ESRB
  2020 (COVID VM "trapped" overnight; HF cleared VM ×10).
- **Verdict:** **defensible, intensified.** 60-min VM is more frequent than CME's twice-daily routine,
  but it's a reasonable proxy for CME's intraday-call regime and is *conservative* (collecting VM more
  often de-risks the book — exactly the E6 result). Disclose the real cadence. Optional: a twice-daily
  VM robustness arm as the literal comparator.

### 1.3 Default fund (DF)

- **Now:** cover-2 SLOIM, `SLOIM_i = max(0, (stress_move − im_frac)·notional_i)`,
  `total_DF = Σ(top-2)·(1+0.10)`, `stress_move = max(0.15, 3.0·σ_daily·√2)` (`clearing.py:163-171`,
  `globals.py:247-254`). **Cash-prefunded** (member cash → `df_cash` pool each daily recompute, draws
  disburse it, replenishment = a cash call — `clearing.py:181-193`). Recomputed daily (390 steps).
- **ODD:** flat `dfPercent = 0.20` of gross notional per member, top-2, every 510 ticks;
  `recoveryRate=0.60`; the fund was **bookkeeping** (L3 cost survivors nothing).
- **Real:** CPMI-IOSCO PFMI Principle 4 + EMIR Art 43(2) = **cover-2** for systemic CCPs (modern CME Base
  is cover-2). Mechanism = **stress-loss-over-IM** (the fund covers the tail *beyond* IM; daily stress
  tests). **DF/IM ≈ 3-4% for liquid futures CCPs** (CME 3.2%, LCH 3.7%); 11-14% for options-heavy
  (OCC, Eurex). CME Base DF/cover-2 ≈ **1.24** (DF $9.4B vs cover-2 stress $7.6B).
- **Papers:** Duffie-Scheicher-Vuillemey 2015 (collateral demand = IM + frictional VM); the SLOIM
  framing is standard (Euronext A9).
- **Verdict:** mechanism is **sound and standard** (cover-2 SLOIM + 10% buffer matches EMIR/CME).
  Two issues: **(a)** the **SLOIM-IM coupling** (higher IM → smaller fund) is genuine stress-loss-over-IM
  math but **confounds H2's solvency result** and **(b)** inflates **calm DF/IM to ~44%** (because the
  DF floor 15% sits far above the IM floor 6%, so SLOIM = 9% of notional on a 6% IM base). Stressed
  DF/IM (5.6-11.4%) is in the real band; calm is not. **Fix = the fixed-stress-fund robustness arm**
  (size the fund on a fixed scenario independent of current IM) — this both cleans H2 and brings calm
  DF/IM toward the real ~3-5%. Keep `df_buffer=0.10` (matches CME's ~1.24, conservatively).

### 1.4 Skin-in-the-game (SITG)

- **Now:** `own_df = min(ex_df_ratio·total_DF, ccp.cash)`, `ex_df_ratio = 0.10` → SITG = 10% of the
  fund, drawn at L2 (`clearing.py:174`). `CCP_CASH = $7.5B` labelled "SITG capacity."
- **ODD:** `exDFRatio = 0.10`, consumed at L2 before L3.
- **Real:** EMIR Art 45 = dedicated own resources **= 25% of the CCP's regulatory capital** (a small
  absolute number), positioned after the defaulter, before pooled DF. Empirically SITG is **~1-4% of the
  fund**: CME $250M / $9.4B ≈ **2.7%**; ICE Europe $197M ≈ ~2%; the academic critique is that real SITG
  is *too small*. CME Base SITG = **$100M**.
- **Verdict:** **position is right, size is ~3-7× too large and mislabelled.** `ex_df_ratio=10%` is not
  "EMIR Art 45 = 25%" (that's 25% of *capital*, not 10% of the *fund*). **Lower to ~2-4% and relabel**
  (source: Clarus CCP-disclosure empirics, not EMIR). Separately, **`CCP_CASH=$7.5B` is DF-scale**
  (member-funded fund range: CME $9.4B, LCH $10.3B), **not** CCP own SITG ($100-250M) — split the
  construct: SITG ≈ $100-250M as the CCP's own first-loss; keep $7.5B only as a total-resources cap.

### 1.5 Default waterfall + porting + close-out

- **Now:** 5 levels — L1 defaulter DF → L2 SITG → L3 pooled DF (pro-rata by contribution) → L4 survivor
  cash (pro-rata by cash) → L5 CCP cash (`clearing.py:195-247`). EMIR-48 porting (greedy by headroom,
  unported closed out — `simulation.py:261-300`). Defaulted book → **CCP open-market Almgren-Chriss
  fire-sale** (`AC_HORIZON=30`, `κH=2.0`), disposal P&L resumes the waterfall (`simulation.py:492-571`).
- **ODD:** same 5-level order; **position auction** to the surviving CM with the largest opposing position
  at `recoveryRate=0.60` (no open-market impact); no client tier, no porting.
- **Real:** identical waterfall order (CME/EMIR). Close-out is a **default-management auction** with
  mandatory participation + juniorisation, *designed to avoid* open-market price impact (Vuillemey 2023).
  Porting = EMIR Art 48 (CCP transfers client positions to a back-up member if pre-agreed, else
  liquidates; CFTC Part 190 5-day grace). L4 in reality = **capped assessments** (CME Base ≤2.75×/5.5×
  the fund).
- **Verdict:** the order and porting are **faithful**. The **fire-sale-instead-of-auction is a deliberate,
  correct, well-cited deviation** — Vuillemey 2023 shows auctions exist *to suppress* the price-impact
  channel, so the model intentionally exposes that channel because it is the object of study. L4
  "pro-rata" is a simplification of capped assessments — disclose, not a problem.

### 1.6 Member structure, cash & scale

- **Now:** 10 BCM (own-account, FT-cast) + 5 NBCM (pure intermediaries) + 1 CCP; 90 cleared clients.
  CCP $7.5B; BCM U[$5B,$10B]; NBCM U[$50M,$1B]; clients FT U[$100M,$500M] / MT U[$30M,$150M] / ZI
  U[$60M,$150M]; ES `CONTRACT_USD=$50`, `VOLUME_LOT=30/60` (`globals.py:117-151`).
- **ODD:** 10 BCM + 10 NBCM + 10 ZI (no clients); BCM U[$5B,$10B]; NBCM **U[$5M,$10M]**; CCP $10M.
- **Real:** CME total IM ~$292B (**82% client**); DF $9.4B; bank-FCM adjusted net capital ~$5-20B;
  non-bank FCM ANC ~$30M-$0.9B; ES $50 multiplier, ~$379k notional, OI ~1.9M. Client share of IM (CME
  82%, LCH 60%) **corroborates the thesis's "clients ≈ 73% / OFR 2026" premise** for H1.
- **Verdict:** **BCM / NBCM / client cash bands are realistic and well-anchored — keep.** The NBCM
  U[$50M,$1B] is exactly the real non-bank FCM ANC range; $50M is literally CME's OTC-clearing minimum.
  Two notes: BCMs are **over-capitalised relative to the books they carry** (12-114× the floor in-sim),
  which is *why* L4 is dormant (fix #4); and `CCP_CASH=$7.5B`-as-SITG is mislabelled (§1.4).

### 1.7 Frequencies

| Process | Now | Real (CME) | Verdict |
|---|---|---|---|
| IM recompute | continuous EWMA (reactive), 60-min apply | daily + ad-hoc intraday | model ⊇ real; conservative |
| VM settle | 60 min | ≥ twice daily + intraday calls | intensified, conservative — disclose |
| DF recompute | daily (390 steps) | quarterly (Base) / monthly (IRS) | accelerated = ESRB replenishment-liquidity channel — deliberate, disclose |
| Confidence / MPOR | 99% / 2-day | 99% / 1-2 day | matches |
| Capital floor | 8% cash/IM | 8% risk-based ANC (CFTC) | matches |

---

## 2. Not implemented (vs a real clearing layer)

From the code audit (`clearing.py`/`simulation.py`): **intraday VM** (only hourly); **netting** (notional
is gross `Σ|pos|`, never nets longs/shorts — the EU net-omnibus lever); **portfolio/cross-margining**
(single asset); **default auction** (replaced by fire-sale, by design); **asymmetric VM**; **interest on
posted margin**; **account segregation** (single `df_cash` pool, no omnibus/individual); **IM cap /
concentration add-ons**; the **EMIR Art 28 25% APC buffer** (only the floor is implemented); a **cure /
grace period** before default. Most are reasonable single-asset simplifications; the ones worth adding are
the **IM cap** (fix #1), the **fixed-stress fund** (fix #2), and optionally **net-omnibus netting** (a
clean third H-axis).

---

## 3. Real-world scale (to ground the cash balances)

CPMI-IOSCO Public Quantitative Disclosures, Q2-Q3 2025 (FIA CCP Tracker / Clarus, from the PQDs; CME/EMIR
primary where cited):

| CCP | Initial margin | Prefunded default fund | SITG | DF/IM | Members |
|---|---|---|---|---|---|
| **CME (all)** | **$291.8B** (82% client) | **$9.4B** | $250M (Base $100M) | **3.2%** | — |
| CME Base (F&O, incl. ES) | $254.5B | (part of $9.4B) | $100M | ~1.4% | cover-2 stress $7.6B |
| LCH Ltd | $279.2B (60% client) | $10.3B | ~1% | **3.7%** | 62 (SwapClear) |
| OCC | $141.8B | $19.6B | — | 13.8% | 102 |
| Eurex | ~$90B | $10.3B | — | 11.4% | 81 |
| ICE Clear Europe | $83.7B | — | $197M | ~2% | — |

ES contract: **$50 multiplier**, ~$379k notional, IM ~3.1-4.7% of notional, OI ~1.9M. Sources: FIA CCP
Tracker Q2-2025; Clarus "What's new in CCP disclosures Q3-2025"; CME Financial Safeguards; CME ES
contract/margin specs; OFR WP 26-04 (client clearing). **Key ratios to hit:** DF/IM ≈ **3-4%** (liquid
futures, not the 11-14% options band); SITG ≈ **1-4% of the fund**; DF/cover-2 ≈ **1.1-1.25**.

---

## 4. Simulation behaviour & realism (live run, seed 42)

**Calm day** (6 cycles): IM/notional 7.2→9.0% (**realistic**, CME ES 8.5-10%); BCM cash flat
(VM ~$10M/day, negligible vs $72B); BCM capital 12-114× floor; NBCM cash/IM 0.75-5.25; **0 defaults, L0**.

**Reverse-stress sweep** (COVID window):

| c | drawdown | client defaults | CM defaults | deepest WF | DF | IM peak | SITG used |
|---|---|---|---|---|---|---|---|
| 1.0 | −18.8% | 3 (all MT) | 0 | L0 | $0.71B | $15.4B | 0 |
| 2.0 | −34.1% | 22 | 0 | L0 | $1.84B | $34.0B | 0 |
| 3.0 | −46.3% | 41 | 0 | L0 | $2.08B | $47.3B | 0 |
| 4.0 | −56.4% | 53 | 4 (all NBCM) | **L2** | $5.21B | $71.2B | $0.038B (5%) |

Default ordering MT→FT→ZI is mechanically correct (trend-followers carry the directional risk).
Contagion runs **client → NBCM** (the intended channel); **no BCM ever fails** → **L3/L4 dormant** (BCMs
over-capitalised). At c=4 the CCP uses only 5% of SITG — solvent by a wide margin.

**Realism flags (quantitative):** (i) stressed IM ~2.5× too high at c=1 (24.9% vs ~10%) + IM>100% at c≥4
(no cap) — fix #1; (ii) calm DF/IM ~44% vs real 3-4% (floor coupling) — fix #2; (iii) BCM capital 12-114×
floor is unrealistically safe — fix #4; (iv) L4 dormant — fix #4. VM magnitudes (calm ~$25M/cycle,
stressed up to $5B/cycle ≈ CME's ~$5B/day COVID) are realistic; waterfall fires in the right order.

---

## 5. Recommended "best final design"

Keep the architecture; make these changes (priority order from §0):

1. **IM cap + (optional) cleaner stressed vol.** Add an upper bound `im_fraction ≤ IM_CAP` (e.g.
   ~25-30% of notional, the real APC-smoothed ceiling) so IM never exceeds the notional and the stressed
   overshoot is bounded; optionally drive the reactive EWMA from a jump-robust (bipower) intraday vol so
   it doesn't double-count the fat tails. Direction of procyclicality (the H2 mechanism) is unchanged.
2. **Fixed-stress default fund (robustness arm).** Size SLOIM on a fixed stress scenario independent of
   the current IM (replace `im_frac` in the SLOIM coefficient with the constant `IM_FLOOR`, or decouple
   `df_stress_move` from σ). This (a) cleans the H2 solvency result of the SLOIM confound and (b) brings
   calm DF/IM from ~44% toward the real ~3-5%. ~2 lines + re-run; report both coupled and fixed.
3. **SITG: `ex_df_ratio` 10% → 2-4%, relabel; split `CCP_CASH`.** Set SITG to ~2-4% of the fund (CME ES
   ≈2.7%), source it to Clarus empirics (not "EMIR Art 45 = 25%"). Give the CCP a genuine own SITG of
   ~$100-250M; keep $7.5B only as a total-resources cap.
4. **Make L4 / the client→NBCM→mutualisation channel fire** (highest-leverage, AGENT.md's own open item).
   Combine (a) a near-floor BCM/NBCM cash draw (at least one small, toppleable member) with (b)
   **directional** client concentration (route same-sign high-z FT clients onto one small NBCM so its net
   book — not gross — exceeds its cash on a down-gap). Size heterogeneity alone is insufficient because the
   5× margin cap keeps gross-but-net-zero books small.
5. **Instrument for the literature's metrics** (§6): loss-location decomposition, concentration index,
   collateral-demand decomposition `C = IM + frictional VM`; report distributions + paired tests + bootstrap
   CIs, not medians.

Disclose (don't change): the 60-min VM and daily DF cadences (intensified vs real, conservative); the
fire-sale-vs-auction deviation (Vuillemey 2023 — it's the contagion channel); the L4 pro-rata
simplification; the Gaussian-vs-historical VaR quantile. Optional new axis: **net-omnibus netting**
(gross → net `|Σ pos|`) for a gross-vs-net margining comparison.

---

## 6. How to test the hypotheses with this layer

**Novelty (state it once, lead with it):** every CCP-contagion paper holds the price/loss exogenous over a
fixed network; this is the first to put a **client tier on an endogenous LOB whose fire-sale walk is the
contagion channel**. That endogeneity is the defensibility — it lets you *locate the severity at which a
trade-off turns* and *let the spiral emerge*, which static models cannot.

**H1 — tiered vs direct.** Hold everything constant except the clearing graph (same agents, cash, V_t,
seeds; conserve total resources — state the cash-conservation audit). The literature (Galbiati-Soramaki
2013, Borovkova 2013, Duffie-Zhu 2011) demands the **two-sided** result, so a one-sided "tiering is safer"
would be a red flag. Report: (a) **breach-multiplier survival curve** per waterfall level (your metric);
(b) **loss-location decomposition** — client / member-balance-sheet / mutualised-DF / CCP (the single most
defensible H1 figure; operationalises "tiering relocates, doesn't erase" — Duffie-Scheicher-Vuillemey
distributional finding); (c) **member-default count vs mutualisation frequency** (the trade-off); (d)
**concentration index** (Herfindahl / max-single-member-loss vs c — literally Galbiati-Soramaki). Locate
the c at which the ordering flips (Borovkova predicts it can) — that crossover is a genuine contribution.

**H2 — procyclical vs through-the-cycle margin.** Split the two stories:
- *Liquidity side (clean headline):* **IM responsiveness ΔIM%/Δvol%** (the literal BCBS-CPMI-IOSCO 2025
  metric — you already report 0.76→3.98); **peak-to-trough IM ratio** (reactive 1.96→6.01 vs flat ~1.7-2.2);
  **calm collateral cost** ($28.6B flat-12 vs $13.8B reactive vs $11.5B flat-5 → "TTC costs ~2× collateral
  to avoid a ~3× spike" = Glasserman-Wu's buffer cost). Validate externally against ESRB 2020 (EU IM +⅓)
  / CPMI-IOSCO 2022 (cleared IM +~$300B; HF VM ×10).
- *Solvency side (only with fix #2):* breach-multiplier by regime, presented **twice** — SLOIM-coupled
  (caveated) and **fixed-fund (clean headline)**; an emergent **liquidity-spiral diagnostic** (regress
  per-step fire-sale price impact on lagged margin-call cash, compare across regimes — the *emergent*
  Brunnermeier-Pedersen / Aymanns-Farmer spiral, your contribution over assumed-spiral models).

**H3 (recommended, low-risk).** Collateral-demand decomposition `C = IM + frictional VM`
(Duffie-Scheicher-Vuillemey 2015 — the PDF is in `extras/references/`). Log IM (by tier) + VM buffers per
step; show how C splits across client/member/CCP and shifts between tiered/direct and across IM regimes —
the collateral-side analogue of H1's loss relocation, done *dynamically under stress with an endogenous
price*, which DSV (static) cannot. Reuses existing instrumentation.

---

## 7. Sources

**Code:** `model/clearing.py`, `model/simulation.py`, `model/globals.py`, `run_simulation.py`,
`covid_contagion.py`; `model.md` §2.2/§3/§4/§6/§8; `AGENT.md` (D55/D58/D61/D63 + Known structural limits).
**ODD:** `extras/references/Simudyne_CCPRiskModel_ODDDoc.md`.
**Regulation:** EMIR RTS 153/2013 Arts 24/26/28/35; EMIR Arts 43/45/48; CPMI-IOSCO PFMI Principles 4/6;
CME Financial Safeguards; CME SPAN 2 methodology; BCBS-CPMI-IOSCO 2022 (d526/d537) & 2025 (d590).
**Disclosures:** FIA CCP Tracker Q2-2025; Clarus CCP disclosures Q3-2025; CME ES contract/margin specs;
OFR WP 26-04; FIA Procyclicality (Oct-2020); ESRB June-2020 margin report; Clarus "CCP Skin in the Game."
**Papers:** Duffie-Zhu 2011; Galbiati-Soramaki 2013; Duffie-Scheicher-Vuillemey 2015; Paddrik-Rajan-Young
2020; Menkveld-Vuillemey 2021; Cont 2017; Vuillemey 2023; Borovkova-El Mouttalibi 2013; Glasserman-Wu
2018; Brunnermeier-Pedersen 2009; Aymanns-Farmer 2015; Bookstaber-Paddrik-Tivnan 2018; Almgren-Chriss 2000.

---

## Update — changes implemented (D68)

Fixes #1-#4 from §5 are in (see AGENT.md D68). Verified outcomes (COVID window, seed 42):

- **#1 IM cap** `IM_CAP=0.30` + **companion DF stress cap** `DF_STRESS_CAP=0.35`: removes the IM>notional
  artifact AND keeps the fund realistic (DF ~$0.6-1B over the window, not the ~$16B the IM cap alone
  produced). Stressed DF/IM now ~3-5% (real CME/LCH band). **Side effect: the right-sized fund is now
  breached to L3 (pooled-DF mutualisation) at c≈4** — deeper mutualisation via the realistic fund +
  reverse-stress, not a hack.
- **#2 `DF_DECOUPLE_IM`** (default False): H2 robustness lever — sizes the fund on a fixed reference IM so
  it doesn't shrink as live IM rises. Verified reaches L3 at c=4 when on. Run both arms for H2.
- **#3 SITG** `ex_df_ratio` 10%→3%, relabelled (empirical SITG/DF, not "EMIR 25%"); `CCP_CASH` relabelled
  as total resources. Verified own_df/total_df = 0.030.
- **#4 fragile-NBCM concentration** (`CONCENTRATE_FT_ON_FRAGILE_NBCM`, default **OFF**): verified
  counterproductive — under cover-2 SLOIM the concentrated member becomes a top-2 member so the fund
  *grows* to cover it (absorbed at L1), and the freeze/stop-out bounds the directional book. **L4 dormancy
  is therefore partly STRUCTURAL to cover-2 sizing** (realistic; L4 assessments are rare). The reverse-stress
  c (now reaching L3) is the lever for deeper mutualisation. Kept as an off-by-default lever.
- Topology map added: `clearing_topology.ipynb` / `.png`.
- Remaining disclosed item (not changed): **calm DF/IM ~44%** from the `DF_STRESS_FLOOR` 15% vs `IM_FLOOR`
  6% mismatch — 15% is a defensible extreme-but-plausible floor; lowering it is the optional further fix.

---

## 8. Sources for the thesis (clearing-layer element → citation)

Citable basis for each element, for writing the model chapter. "bib" = key already in
`references.bib`; "ADD" = source to add (mostly primary disclosures the bib doesn't yet carry).

| Element (model) | Real-world basis | Citation | bib |
|---|---|---|---|
| Initial margin = 99% VaR, 2-day MPOR | EMIR RTS 153/2013 Art 24 (99% non-OTC), Art 26 (2-day MPOR); CME SPAN/SPAN 2 | EMIR RTS 153/2013; CME SPAN 2 Methodology | `emir_rts`; CME SPAN (ADD) |
| IM anti-procyclicality floor (6%) | EMIR RTS Art 28(c) 10-yr volatility floor; CME ES maintenance margin | EMIR RTS 153/2013 Art 28 | `emir_rts` |
| IM cap / smoothing (30%) | APC smoothing; margin-responsiveness work | BCBS-CPMI-IOSCO 2022/2025 | `bcbs_cpmi_iosco_2022/2025` |
| Variation margin, fill-price, intraday | CME ≥twice-daily MtM + intraday calls; settled-to-market | CME Financial Safeguards; CFTC | CME FSG (ADD) |
| Default fund cover-2 | EMIR Art 43(2); CPMI-IOSCO PFMI Principle 4 | EMIR 648/2012 Art 43; CPMI-IOSCO PFMI 2012 | `emir_rts`; PFMI (ADD) |
| DF sizing = SLOIM (stress-loss-over-IM) | Euronext Clearing module A9; daily stress testing | Euronext A9; CME daily stress tests | `euronext_a9` |
| DF stress move (~99.9%, 15-35%) | extreme-but-plausible 2-day stress scenario | Euronext A9 §2.1; CPMI-IOSCO | `euronext_a9` |
| Skin-in-the-game (3% of fund) | EMIR Art 45 (DOR = 25% of CCP capital); empirical SITG/DF ~1-4% (CME ES ≈2.7%) | EMIR 648/2012 Art 45; Clarus "CCP Skin in the Game" | `emir_rts`; Clarus (ADD) |
| Capital floor 8% (cash/IM) | CFTC Reg 1.17 adjusted-net-capital ≥ 8% risk margin | CFTC Reg 1.17 | `cftc_reg117` |
| BCM house-book VaR limit (5%) | Basel III FRTB | BCBS FRTB | `basel_frtb` |
| 5-level waterfall order | CME Financial Safeguards waterfall; EMIR Art 45 | CME FSG; EMIR 648/2012 Art 45 | `emir_rts`; CME FSG (ADD) |
| Client porting | EMIR Art 48(5)-(6); CFTC Part 190 grace period | EMIR 648/2012 Art 48 | `emir_rts`; CFTC Part 190 (ADD) |
| Close-out = open-market AC fire-sale (NOT the ODD auction) | Almgren-Chriss execution schedule; CCPs use auctions to *avoid* fire-sale (deliberate deviation) | Almgren & Chriss 2000; Vuillemey 2023 | `almgren_chriss_2000`; `vuillemey_2023` |
| Member/client cash scale | CFTC FCM adjusted-net-capital; CPMI-IOSCO PQD | FIA CCP Tracker; Clarus; CFTC FCM data | FIA/Clarus (ADD) |
| Clients = majority of CCP margin (H1 motivation) | clients ~73-82% of IM at large CCPs | OFR 2026; FIA (CME 82%) | `ofr_2026`; FIA (ADD) |
| Procyclicality / margin→liquidity (H2) | margin spiral; COVID dash-for-cash; TTC margin | Brunnermeier-Pedersen 2009; Glasserman-Wu 2018; Cont 2017; ESRB 2020; BCBS-CPMI-IOSCO 2022/2025 | `brunnermeier_pedersen_2009`; `glasserman_wu_2018`; `cont_2017`; `esrb_2020`; `bcbs_cpmi_iosco_2022/2025` |
| Tiering trade-off (H1) | netting/concentration vs loss-absorption | Duffie-Zhu 2011; Galbiati-Soramaki 2013; Borovkova 2013; Paddrik-Rajan-Young 2020 | `duffie_zhu_2011`; `galbiati_soramaki_2013`; `borovkova_2013`; `paddrik_rajan_young_2020` |
| Collateral-demand decomposition (H3) | C = IM + frictional VM | Duffie, Scheicher & Vuillemey 2015 | extras PDF (ADD bib key) |

Primary-source URLs are in §7 and the subagent reports. Keys marked ADD are mostly CME/CPMI-IOSCO/FIA/Clarus
primary disclosures — add to `references.bib` if you cite the specific figures (real IM $292B, DF $9.4B,
SITG $100M, DF/IM ~3-4%).
