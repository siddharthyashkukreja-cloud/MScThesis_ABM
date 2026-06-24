> **HISTORICAL SNAPSHOT (2026-06-18).** This document records an earlier audit/state and is NOT the current model. For the current committed model and results see AGENT.md and THESIS_SYNTHESIS.md. Notably superseded since: BCM prop cap 6×→2×, house margin 20%→15%, leverage-balanced client→clearer assignment, client balance-sheet rebalance, and the close-out/GBM-ensemble findings.

# Model design review

Read-only audit of the model design and code, run as six parallel component reviews
(FV, trader agents, clearing/balance-sheets/margin, LOB/simulation engine, hypothesis-testing
harness, calibration/validation). Every finding is anchored to `file:line`. Date: 2026-06-14.

This is a defensibility + correctness review aimed at a high-level thesis — i.e. "what would an
examiner attack, and what is the highest-leverage fix." Nothing here was changed; it is analysis only.

Two claims were spot-verified against the code before writing: the H2 fund-sizing confound
(`clearing.py:169`, confirmed) and the "im_responsiveness is unpopulated" flag (FALSE — the column
is 5520/5520 populated, so the §6.2 responsiveness numbers are reproducible; dismissed).

---

## 0. Priority summary

Ranked by leverage for a strong thesis. Detail in the linked sections.

| # | Item | Type | Effort | Where |
|---|------|------|--------|-------|
| 1 | **H2 fund sizing confounds the mutualisation result** — `(stress_move − im_frac)` means higher IM shrinks the fund. Run the fixed-stress-fund robustness. | Confound (disclosed, not fixed) | ~2 lines + re-run | `clearing.py:169`; §2.1 |
| 2 | **L4 / client→NBCM channel never fires** — the survivor-mutualisation cascade (the most policy-relevant) is never exercised tiered. Needs *directional* client concentration on a small NBCM. | Structural limit (AGENT.md "highest-leverage open item") | build change + re-run | `run_simulation.py:149-176`; §2.2 |
| 3 | **No statistical inference despite a clean paired (CRN) design** — only marginal means/medians; no CIs, no significance tests; the breach-`c` median is censored. | Statistical gap | ~30 lines | `run_experiments.py:296-313`; §2.3 |
| 4 | **Stressed grid too coarse to call "exhaustive"** — 4 pts/axis, optimum sits on the box floor (`ft_sigma_c=0.25`) while the surrogate says interior ≈0.34. | Defensibility | re-run finer | `calibrate.py:200`; §2.4 |
| 5 | **Kalman "fundamental" is near-identity** (σ_ε≈1e-5, gain≈1.0 ⇒ V_t ≈ mid). Reframe as verification, not smoothing. | Framing | wording | `v_kalman.py:161`; §2.5 |
| 6 | **Add Moment Coverage Ratio (+ optional moment p-values)** — the validation the bare market layer is missing; machinery already exists. | New validation (you asked) | ~30 lines | §4 |
| 7 | **Opening call auction** — realism/print-symmetry improvement; will not change H1/H2/E6 conclusions. | Optional design (you asked) | ~2–3 days | §3 |

---

## 1. What is solid (state these as strengths)

These were checked and hold up; lead with them in the viva.

- **Cash conservation genuinely holds.** The ledger was traced end-to-end: fills never create/destroy
  cash for cleared accounts (futures-margin gate, `simulation.py:238,244`); VM is a pure transfer; the
  default fund is a custody move (`df_cash ≡ Σ contributions`); the only destruction is the written-off
  residual at L5 (`clearing.py:244-246`). The phantom-double-charge bug is genuinely fixed (the assumed
  book is booked as a fill at mark, `simulation.py:434-435`).
- **The H1 counterfactual is clean.** Tiered vs direct share the fundamental path and seed (CRN), and
  direct clients keep identical trading mechanics — same 5×/20% house cap, same 8% freeze, same default-at-0
  (`simulation.py:380` vs `:448-450`, `:344` vs `:402`). The *only* structural difference is loss
  absorption (CM buffer vs straight-to-waterfall). Verified 40 identical seeds across arms.
- **E6 drawdown-matching is fair.** Shock trough == gapped trough to float precision at every `c`, and
  σ_t (the IM driver) is held identical, so only price-delivery structure differs (`run_experiments.py:168-172`).
- **VM fill-price settlement is correct and identical in all four settlement sites** (`P/L = pos·Δmark + Σq·(mark − fill)`).
- **Fixed-z FT is the right call** (Chiarella-Iori-Perelló heterogeneous beliefs; redrawing z would kill
  the persistent-inventory mechanism the clearing layer needs).
- **The calibration loss, inverse-SD weights, Hill estimator, and KS are correctly implemented;** the
  identifiability story (ft_sigma_c + p_zi identified; ZI rates flat) is well-supported by the fresh-seed data.

---

## 2. Highest-priority issues

### 2.1 H2: the fund-sizing convention confounds the mutualisation-depth result  [verified]

`clearing.py:169`:
```
sloim = sorted((max(0.0, (stress_move - im_frac) * n) for n in notionals), reverse=True)
```
The cover-2 fund coefficient is `(stress_move − im_frac)`. Both terms are driven by the *same* σ, so
raising IM (higher regime σ under reactive, or a higher `IM_FLAT_FRAC`) **mechanically shrinks the pooled
fund** and simultaneously lowers every member's cash/IM ratio. In the data this is a 3× fund difference:
flat-12 gives coefficient `0.15−0.12=0.03` vs flat-05/reactive-calm `0.15−0.06=0.09` (kalman flat12
`df_mean≈$0.5-0.7B` vs flat05 `≈$1.7-2.2B`).

So "high-margin regimes mutualise more often" (model.md §6.2: static L3 0.925, flat-12 0.85 vs reactive
0.375) is **partly an artifact of a smaller fund being breached sooner**, not a clean procyclicality→contagion
finding. The *liquidity/cost* side of H2 is clean and uncontaminated (calm IM $28.6B flat-12 vs $13.8B
reactive; reactive crisis peak-to-trough 1.96→6.01). model.md §6.2/§7 already disclose this honestly — but
**the decoupling robustness run is absent**, so the solvency half of H2 currently cannot be attributed to
procyclicality.

Two latent foot-guns in the same place:
- If anyone sets `IM_FLAT_FRAC > 0.15` (a plausible robustness arm), SLOIM goes to **exactly zero** for all
  members, silently, because `df_stress_move` never consults `IM_MODE` and the `max(0,·)` masks it
  (`globals.py:241-248`, `clearing.py:169`).
- The fund is non-monotonic in σ near the floor crossover (the `(3.0−2.326)·σ` coefficient dips below the
  `0.15−0.06` floor regime over a σ band) — undocumented.

**Fix (the right one, already in AGENT.md open item #2).** Size the fund on a *fixed* stress scenario
independent of current IM — e.g. replace `im_frac` in the SLOIM coefficient with the constant APC floor
`IM_FLOOR`, or decouple `df_stress_move` from σ entirely. ~2 lines at `clearing.py:164-169`, then re-run H2.
If the mutualisation ordering survives, procyclicality is the driver; if it collapses, it was the convention.
Pair with an assertion that `df_stress_move ≥ im_fraction` so the clamp can't silently zero the fund.

### 2.2 L4 and the client→NBCM channel never fire

Confirmed from code why the survivor-mutualisation cascade — arguably the headline CCP mechanism — is never
exercised in the tiered arm:
- Client books are margin-capacity-bounded at 5× cash (`agents.py:96-97`), so they are small.
- Only small ZI accounts default under stress (`ZI_CLIENT_CASH $60-150M`, `globals.py:136-139`); their
  losses stay under FCM-scale NBCM cash (`NBCM $50M-$1B`, `globals.py:121`), so no NBCM topples.
- BCM own book is VaR-capped and deleveraged at the 8% floor before it can build an L4-scale deficit.
- CCP cash ($7.5B, `globals.py:117`) + SITG + pooled DF absorb any realised deficit the current population
  can generate, so **L3 is the realistic floor; L4/L5 are unreachable** tiered (model.md §6.1: tiered L4
  2.5% at 4×). The direct arm *does* reach L4 (95% by 3×), so the machinery is validated — the tiered
  narrative just can't demonstrate it.

**Why the proposed lever (size heterogeneity) is necessary but not sufficient.** The build already gives the
larger books to NBCMs (`run_simulation.py:149-176`). The binding constraint is the 5× margin cap: an all-ZI
book is large gross but ~0 net (ZI mean-reverts), so VM losses are small. To topple an NBCM you need
**directional concentration** — route a cluster of same-sign (high-z, persistently long) FT clients onto one
*small* NBCM so its book is net-long and a down-gap produces a one-sided VM call exceeding its cash. That is a
targeted `build_clearing_tier` change (concentrate same-sign FT clients on one small NBCM, optionally tighten
the small-NBCM cash floor toward the $30-140M CFTC small-FCM end), not just a wider size range. This is the
single highest-leverage change to make the thesis's central channel actually fire.

### 2.3 No statistical inference despite a clean CRN paired design

The harness is *built* for paired comparison — identical seeds across arms, verified — but
`aggregate()` collapses to **marginal** means/medians/fractions and discards the pairing
(`run_experiments.py:303-313`). There is **no scipy.stats / bootstrap / CI / significance test** anywhere in
the experiment pipeline (grep-confirmed). All headline claims (H1 breach gap, H2 ordering, E6 27-vs-21
defaults) are point estimates with p10–p90 *dispersion* bands that are easily misread as *precision* bands.

The pairing is real signal being thrown away: reconstructed by hand, the within-seed
`breach_c(tiered) − breach_c(direct)` is **+1.23 c-units** over the 15 doubly-finite seeds — exactly the
"~1.25× headroom" headline, but computed nowhere.

Related metric problem: the breach-`c` **median is conditional on breaching** (`run_experiments.py:296-298`,
`bc.dropna().median()`). For kalman tiered L3 only 15/40 seeds ever breach through c=4, so the reported
"median 3.5" is over the breaching minority and reads as far more reliable than it is. Report
`frac_seeds_breached` (correct) + a censored statement, not a conditional median.

**Fix (~30 lines in `aggregate()`):** per-seed paired differences with bootstrap 95% CIs and a paired
sign/Wilcoxon (H1 tiered−direct; H2 reactive−flat12; E6 gapped−shock); a paired McNemar/proportion test on
`mutualised_l3` at a fixed `c`; seed-bootstrap CIs on every ensemble point estimate. This converts the
qualitative claims into defensible significant results — cheap, because the CRN design already delivers the
variance reduction.

### 2.4 The stressed grid is too coarse to defend as "exhaustive"

Stressed grid is `n_per_dim=4` on a 5-d box: `ft_sigma_c ∈ {0.25, 0.45, 0.65, 0.85}`. The locked optimum sits
on the **box floor 0.25** (`calibrate.py:200`), while the surrogate lands interior at `ft_sigma_c≈0.34` with
comparable D (7.0–7.4). The 0.20 node spacing straddles the surrogate's answer, so the grid cannot resolve
whether the stressed tail optimum is the boundary or the interior. Calm is fine (5 pts/axis, interior 0.95).

**Fix:** re-run the stressed grid at `n_per_dim≥6` near the basin, or report the surrogate interior as the
stressed point estimate with the grid as a coarse confirmation. Pairs naturally with the calibration
presentation in §5.

### 2.5 The Kalman "fundamental" is near-identity — reframe honestly

model.md §2.3 and `v_kalman.py` already admit σ_ε≈1e-5, steady-state gain≈1.0, so the RTS smoother is
effectively identity and `V_t` is the observed mid bar-for-bar. The phrase "latent efficient log-price"
(`v_kalman.py:161`) oversells it; an examiner who reads §2.3 will ask "what noise does the filter remove?"
and the honest answer is "essentially none."

This is a *finding*, not a flaw: frame it as "the Kalman MLE confirms 1-minute mid microstructure noise is
negligible, so V_t is the mid, used as the exogenous trajectory agents trade around — the model therefore does
not double-count microstructure." Anchor to Gao et al. (2022, §2.5.2), who use the historical price as the
signal. The associated point — fundamental and calibration targets both come from the same ES data
(circularity) — should be stated, with the SV-MJD robustness run (same data, different fundamental, *same* ZI
θ) cited as the control.

---

## 3. Component-by-component findings

### 3.1 Fundamental value process

Bugs / fixes:

| Sev | Where | Issue |
|-----|-------|-------|
| Med | `v_kalman.py:201` | `_local_vol` docstring claims `√390·σ_t·v0`; live FT formula is `ft_sigma_c·σ_t·v0` (no √390). Stale from the old pin. Doc bug. |
| Low | `v_kalman.py:206-209` | per-day EWMA σ_t initialised at first-bar `r²`, not the stationary mean ⇒ ~30-min warm-up transient each day. Use `mean(rets²)` or the day's BV/n. (SV-MJD path is cleaner — inits at θ_eff.) |
| Low | `v_gbm.py:119` | bipower variation missing the `n/(n-1)` Barndorff-Nielsen-Shephard finite-sample correction (~0.26% BV understatement). One-line fix. |
| Low | `v_gbm.py:175` | Mancini threshold uses a daily-constant σ_loc, not a spot estimate; `THRESH_C=4.0` is a continuous-record value at 390 obs. Disclose / sensitivity. |

Design notes: V_t is fully exogenous (standard for this family, fine). The σ_fundamental scaling uses `v0`
(the init price) as a *constant* dimensional factor, so in fractional terms the FT belief cloud widens ~1.5×
by a 33% COVID drawdown — not a bug, but worth a sentence. The overnight-gap retention and boundary-return
exclusion are correctly implemented.

### 3.2 Trader agents

| Sev | Where | Issue |
|-----|-------|-------|
| Low | `agents.py:101` | BCM house-VaR cap uses static `params.sigma_v`, not live `ctx.sigma_t`, so the cap does **not** tighten intracrash — contradicts model.md §2.2's "tight in stress" claim within the stressed run. Thread σ_t through `_client_cap_qty`. |
| Low | `agents.py:182` | FT `z_score` default is `0.0` (a degenerate value-follower). Production always draws N(0,1), but the silent default is a trap — use a NaN sentinel + guard. |
| Low | `agents.py:216-223` | FT returns on `diff==0` without cancelling its standing order, orphaning it for ≤1 step. Vanishingly rare at 0.25 tick; disclose only. |

Design notes (all defensible, worth pre-empting in the text): the MT is **limit-only**, so trend
amplification is *indirect* (passive limits on the trend side get lifted by other flow) — weaker than
Majewski's marketable momentum, and the documented long-horizon-clustering limit follows from it. State this
explicitly so it doesn't read as a bug. Per-step Bernoulli (no dt rescaling), D65b cancel-on-freeze,
client cap math, and the σ_t→FT routing all verified correct. The BCM VaR cap is correctly isolated from the
market calibration (plain FTs have no balance sheet, so the branch is skipped).

### 3.3 Clearing tier, balance sheets, margin

| Sev | Where | Issue |
|-----|-------|-------|
| **High** | `clearing.py:169` | SLOIM-IM coupling — see §2.1. |
| High(narrative) | `run_simulation.py:149-176` | L4 dormant — see §2.2. |
| Low | `clearing.py:191-193` | DF "refund" branch can return custody cash to a survivor whose contribution was just pro-rata-debited at L3, partially **un-mutualising a realised loss** across daily recomputes. Ledger still balances; economically questionable. Floor the refund. |
| Low(doc) | `simulation.py:437` | `closeout = 0.0` hard-coded; the docstring + model.md §3.3 promise a "(1−recovery) close-out haircut" that is never applied. The code is actually *better* (deficit-consistent fire-sale mark), but reconcile the stale prose — the ODD `recoveryRate=0.60` is abandoned, which is fine but should read as a deliberate deviation. |
| Low | `lob.py:73-113` | unfilled fire-sale quantity is silently dropped (book runs thin) — position/cash conservation still holds (inventory only moves by what filled), but a fire-sale can stall. Feature, not bug; know it. |

Margin methodology is otherwise sound: reactive/static/flat IM, VaR=SPAN equivalence for a single linear
future, APC floor, cover-2 SLOIM, cash-prefunded fund with replenishment (faithful to the ESRB-2020 channel
and an improvement over the ODD's free-bookkeeping fund). Flag the hand-tuned cash bands (`globals.py:121-145`,
"tuned so the floor binds under stress but not calm") as the one place the "no invented calibration" rule is
softest — and the fire-sale urgency `κH=2.0` (`globals.py:85-86`) drives the entire contagion-channel speed,
so it deserves a sensitivity sweep.

**Add a runtime cash-conservation assertion** (turn the model.md §3.3 audit into a guard at the end of
`_margin_cycle`) — it makes the float-precision claim reproducible and would catch any regression of the
phantom-charge class automatically.

### 3.4 LOB / matching / simulation loop

| Sev | Where | Issue |
|-----|-------|-------|
| Moderate | `lob.py:167` | call auction clears every crossing at the **resting ask** (`clear_px = best_ask_px`), a persistent bias toward lower fill prices (≤ ¼ tick/trade). Mid (the calibration target) is independent of fill price, so stylised facts are unaffected; but the fill-price VM component is systematically advantaged for buyers. Disclose; the opening auction (§3) fixes it at opens. |
| Low-mod | `lob.py:70-113`, `simulation.py:159-172` | market orders (ZI `zi_mu`, fire-sales) execute **immediately** under continuous rules while limits wait for the end-of-step call auction — a mixed regime not labelled as a design choice. Small effect at calibrated `zi_mu`; disclose. |
| Low | `lob.py:136-141` | `reprice()` tick-rounding can merge two price levels and scramble FIFO at the boundary. Only at the session-open step, where orders are replaced next step anyway. |

Loop ordering verified clean: agents see only the prior-step mid (no look-ahead); FT uses end-of-bar V_t
(intended); the day-boundary reprice→re-anchor→submit→match sequence is correct; fire-sales run before the
match against passive prices (correct).

### 3.5 Hypothesis-testing harness

Architecture is strong: a deduplicated master-spec so shared arms are the *same* runs, resumable, CRN paired
(`run_experiments.py:183-227`). The clean H1 counterfactual, fair E6 matching, and correctly-wired
open-disorder toggle are confirmed (see §1). The two real weaknesses are statistical (§2.3), not structural,
plus the H2 confound it inherits from the clearing layer (§2.1). `im_responsiveness` is fully populated
(false alarm, dismissed). Disclose that reactive≡static in *calm* (both clamp to the IM floor), so the
calm-cost panel has 3 distinct levels, not 4.

Optional H3 is mostly free: the IM series is already logged and `IM_MPOR_DAYS` toggles 1d-vs-2d, so a
system-wide collateral-demand decomposition (procyclical IM call vs DF replenishment cash-call — the ESRB
"dash for cash") is largely an aggregation over existing columns.

### 3.6 Calibration & validation

| Sev | Where | Issue |
|-----|-------|-------|
| Med | `calibrate.py:200` | stressed grid coarse, optimum on box floor — see §2.4. |
| Low | — | D reported across mixed seed counts (grid n_runs=3 vs surrogate n_runs=6); the KS standardiser is sim-sized so D is only comparable at fixed n_runs. Agreement is legitimately claimed on **θ**; any D-to-D sentence needs the caveat. |
| Low | `calibrate.py:603` | dead defensive filter (`if k in lhs_df.columns`) in `_loss_column` that `_true_loss` then contradicts by re-indexing the full moment list — remove or make skip-tolerant. |
| Doc | AGENT.md / model.md | prose still says calm loop is 3-d `{ft_sigma_c, zi_alpha, zi_delta}`; code re-added `zi_mu` at D65, so it is 4-d (calm) / 5-d (stressed). Reconcile. |

Note: ACF smoothing is **forward** (Gao XGB-Chiarella), correctly cited — but Franke & Westerhoff 2012 use
**centred**; keep the moment vector consistent if you add FW-style p-values (§4).

---

## 4. Validation you asked for: moment p-values + Moment Coverage Ratio

**Source.** The repo prose attributes these to "XGB-Chiarella following Franke & Westerhoff." In fact the
XGB-Chiarella paper only *cites* the concept; the machinery is entirely in **Franke & Westerhoff (2012)**
(`extras/references/frank_weights_stylisedfacts.pdf`, §4.1 + §4.3 + App. A2), which is the source to use.
Lamperti et al. (2018) uses the FW MCR without extending it.

### 4.1 The method, precisely

**Bootstrap covariance.** Block-bootstrap the empirical returns B times (FW use 5000); each resample → a
moment vector m_b; `Σ̂ = cov(M)`. FW use non-overlapping blocks; for long |r|-ACF lags they use a *longer*
block to avoid downward bias (only relevant if you add long lags — your max lag is 22, so one block is fine).

**Moment p-value (J-based).** `J(m) = (m − m_emp)ᵀ W (m − m_emp)`, the Mahalanobis distance with
`W = Σ̂⁻¹`. Build the bootstrap J-distribution and its 95% quantile `J_0.95`. Then run a Monte-Carlo set of
**model** simulations *each of empirical length T′* (this matters — you cannot compare a long sim's J to a
short-T′ bootstrap), and the model's p-value is where `J_0.95` sits within the model's MC J-distribution
(≈ fraction of bootstrap J ≥ the model's typical J). Per-moment p-values restrict J to one coordinate.

**Moment Coverage Ratio (the cleaner, more reportable statistic).** For each moment, `CI_i = m_emp,i ± 1.96·s_i`.
Run the model M times at empirical length T′; single-moment coverage = fraction of runs with simulated `m_i`
inside `CI_i`; **joint MCR** = fraction of runs where *all* moments are simultaneously in-band. Benchmark
against the bootstrap's own joint self-coverage. Key point (FW): the J p-value and MCR can rank models
differently — MCR is the more honest per-dimension diagnostic, because J can hide a systematic single-moment
miss through cancellation in the quadratic form.

### 4.2 Concrete recipe for this repo (≈30 lines)

You already have ~90% of it.

- **Moments:** the 10 point moments — `ret_std`, `acf_r_{1,5,10,20}`, `acf_absr_{1,5,10,20}`, `hill_tail_index`.
  Exclude `ks_stat` (a 2-sample distance with no point-value CI; report it separately) and `ret_kurtosis`
  (diagnostic).
- **Reuse `empirical_moment_sd` (`calibrate.py:674-734`)** — it already runs the Künsch block bootstrap and
  builds `samples = {m: [...]}` at lines 715-724, then throws everything but the SD away. Refactor it to
  return the full B×K matrix; that gives `s_i` (for CIs), `Σ̂ = cov` (for J), and the bootstrap J-distribution
  for free. Bump `N_BOOTSTRAP` from 200 (`calibrate.py:306`) to ~5000 for the J tail (200 is fine for SDs).
- **CIs:** `CI_i = m_emp,i ± 1.96·s_i` (the `s_i` you already compute).
- **Model MC:** loop `simulate_moments(θ, regime, n_days, 1, seed=...)` for ~200–500 reps at **empirical
  length T′** (calm = the ~20-session scoring window; stressed = the ~29-day window `_sim_steps` already
  returns). Coverage per moment + joint MCR over the reps.
- **Report:** a `moments` CLI subcommand reading the locked θ from `globals.CALIBRATED` / `calibrated_params_grid.json`,
  printing per-moment (emp, CI, model median, in-band?, coverage%) + joint MCR + optional J p-value; write
  `output/relock/mcr_{regime}.json`.

**Predicted result (from the locked moments).** Calm `acf_absr_1` is ~6 SD low and calm `ret_std` ~10 SD high
(the window-selection artifact) ⇒ **calm joint MCR ≈ 0%**; stressed covers far better (ret_std 0.00175 vs
0.00182, Hill 3.02 vs 3.18 — within a couple SD) ⇒ **stressed joint MCR modest but nonzero**. That asymmetry
is itself a result that corroborates §5.2, and the *within-window* calm MCR will jump (matching the D
43.9→26.5 story). Make the per-moment CI plot the headline validation figure.

**Recommendation:** report **MCR** (fully cited, loss-agnostic, drops in cleanly). The J p-value is optional —
it needs the quadratic `W` form (your loss is diagonal-L1), so only add it if you want the single-number test;
MCR alone is the lower-risk, better-supported choice.

---

## 5. Presenting the calibration

### 5.1 Calibrated-parameter table

| Parameter | Meaning | Calm | Stressed | How set | Identified? |
|---|---|---|---|---|---|
| `ft_sigma_c` | FT belief-width / tail lever | **0.95** | **0.25** | calibrated (grid+surrogate) | **Yes** (dominant) |
| `p_zi` | geometric placement depth | 0.543 | **0.18** | calm: measured L2; stressed: calibrated | **Yes** (stressed) |
| `zi_alpha` | ZI limit arrival | 0.26 | 0.32 | calibrated | Weak (flat ±~2 D) |
| `zi_delta` | ZI cancellation | 0.02 | 0.02 | calibrated (at floor) | Weak; floor-bound |
| `zi_mu` | ZI market arrival | 0.0125 | 0.0583 | calibrated | Weak |
| `mt_lambda` | MT EWMA decay | 0.05 | 0.05 | pinned (Majewski) | — |
| `ft_alpha`=`mt_alpha` | FT/MT activation | 1.0 | 1.0 | pinned (every step) | — |
| `μ, σ_v, σ_ε` | Kalman fundamental | per-regime | per-regime | measured | — |
| D_grid (3-seed) | locked loss | 43.93 | 5.76 | — | — |

Legend: **measured** (offline/data) · **calibrated** (SMM/grid) · **pinned** (literature). Footnote: calm
`ft_sigma_c` is interior; stressed sits on the 0.25 box floor (see §2.4). Add the SV-MJD robustness row (same
ZI rates; `ft_sigma_c=0.45`, `p_zi=0.08`) as the fundamental-invariance control.

### 5.2 XGB vs grid

| | calm grid | calm surrogate | stressed grid | stressed surrogate |
|---|---|---|---|---|
| `ft_sigma_c` | 0.95 | 1.04 | 0.25 | 0.34 |
| `zi_alpha` | 0.26 | 0.49 | 0.32 | 0.35 |
| `zi_delta` | 0.02 | 0.19 | 0.02 | 0.17 |
| `zi_mu` | 0.0125 | 0.027 | 0.0583 | 0.118 |
| `p_zi` | 0.543 (L2) | 0.543 (L2) | 0.18 | 0.128 |
| D *(different n_runs — not comparable)* | 43.93 (3s) | 58.66 (6s) | 5.76 (3s) | 7.42 (6s) |
| surrogate R² | — | 0.95 | — | 0.84 |

Caption: agreement on the identified levers (`ft_sigma_c`, `p_zi` — same basin) and disagreement on the weak
ZI rates is the identifiability result, not a contradiction. Header must carry the "D at different seed counts"
caveat.

Figures (all from `output/relock/grid_surface_{calm,stressed}.csv`, which store all 5 component deltas +
moments per node — use these, not the older smaller `calibration_grid_*.csv`):
1. **1-D loss profiles** D vs each lever at the optimum — sharp `ft_sigma_c` min (identified) vs flat
   `zi_alpha`/`zi_delta` (weak). Most persuasive identifiability figure.
2. **2-D heatmaps** D over (`ft_sigma_c`, `zi_alpha`) and (`ft_sigma_c`, `p_zi`), grid argmin ★ + surrogate ●
   in the same valley.
3. **Component-stacked bar** (`dKS,dV,dACF1,dACF2,dHill`) at each optimum — KS+ACF2 dominate calm (artifact),
   stressed balanced+small; motivates the MCR.
4. **Surrogate accuracy scatter** (predicted vs true D, R² 0.95/0.84).
5. **Fresh-seed re-rank dot plot** (D_grid vs D_fresh, top-5, locked node at #1) — the winner's-curse defence.
6. **MCR/CI plot** (§4) — per-moment emp ±1.96 s_i band + model MC spread, in-band green / out red + joint MCR.

---

## 6. Opening call auction (the "start the day with an auction?" idea)

**Verdict: implementable and a genuine realism/defensibility win at the session open, but it will not change
the H1/H2/E6 conclusions — only magnitudes.** Worth doing if time allows; not a results-changer.

Currently each session opens by repricing the resting book by the gap factor `v_now/prev_mid` and re-anchoring
trader memory (`lob.py:126-151`, `simulation.py:129-142`) — a mechanical jump with no agent price discovery
and no opening print, and the first prints inherit the resting-ask asymmetry (§3.4).

**Design.** Add a short pre-open window (P_pre ≈ 5 steps; CME RTH pre-open is ~5 min, so no new calibrated
parameter): during pre-open the book accumulates limit orders but does not match (`LOB._preopen` flag; market
orders are no-ops in pre-open). At the open, **uncross at the price that maximises executable volume** (the
MEQ/equilibrium price), with the standard tie rule (excess-buy → top of tie range, excess-sell → bottom, else
nearest the repriced prior close). Execute all opening fills at that single uniform price (this removes the
resting-ask bias on the most important print), then resume continuous trading.

- **Functions:** new `LOB.preopen_uncross(ref_price)` + an `_preopen` flag in `lob.py`; in `simulation.py`
  replace the day-boundary block with enter-preopen → (P_pre steps) → uncross; defer/route fire-sales during
  pre-open; skip the reactive-IM EWMA update on pre-open steps (no fills).
- **What it fixes:** real agent-driven gap price discovery (the opening price reflects the spread of FT
  reservations around V_open, not a mechanical scale); a symmetric opening print; realistic opening-volume
  concentration.
- **Calibration impact:** the boundary return is *already* excluded from the moments, so you only need to also
  exclude the P_pre pre-open steps (≈1.25% of the calm series) — a bookkeeping fix, not a re-calibration. A
  1–2 seed sanity check at the locked θ suffices.
- **Effect on results:** E6 first-bar open prices shift by a few ticks; client-default counts may move ±1–2;
  the "gap is a mitigant" mechanism (VM collects between jumps) is unchanged. H1/H2 are about structure/margin,
  essentially unaffected. Open-disorder (`REANCHOR_ON_GAP=False`) would need a re-run (the MT sees the gap only
  after the uncross), but its second-order conclusion holds.
- **Effort/risk:** ~2–3 days; the main risk is the calibration boundary-exclusion bookkeeping.

If you don't build it, at least **disclose the resting-ask print convention** (`lob.py:167`) as model.md §7
already begins to — an opening auction is the principled fix and a good "future work" item even if unbuilt.

---

## 7. Consolidated quick-fix list (small, low-risk)

Doc/code hygiene that costs little and removes examiner snags:
- `v_kalman.py:201` — fix the `√390` docstring (no longer matches the code).
- `v_gbm.py:119` — add the `n/(n-1)` BV correction.
- `calibrate.py:603` — remove the dead `if k in lhs_df.columns` guard (or make `_true_loss` skip-tolerant).
- AGENT.md / model.md — reconcile "calm 3-d loop" prose with the 4-d code (`zi_mu` re-added at D65).
- `simulation.py:407-410,437` + model.md §3.3 — reconcile the stale "(1−recovery) close-out" prose with the
  deficit-consistent fire-sale that is actually used.
- `globals.py:241-248` / `clearing.py:169` — assert `df_stress_move ≥ im_fraction` so the SLOIM clamp can't
  silently zero the fund.
- Add a runtime cash-conservation assertion at the end of `_margin_cycle`.
- Disclose: reactive≡static in calm (floor-clamped); the σ_t-init warm-up; the κH=2.0 fire-sale-urgency
  sensitivity; the hand-tuned client/NBCM cash bands.

---

## 8. Suggested sequence

1. **Fixed-stress-fund H2 robustness** (§2.1) — decouple the fund from IM, re-run H2. Cleans the headline.
2. **Paired statistics + CIs** (§2.3) — ~30 lines; makes every result defensible.
3. **MCR validation** (§4) — ~30 lines on the existing bootstrap; the missing market-layer validation.
4. **L4/directional-concentration** (§2.2) — the highest-leverage *structural* change, if scope allows.
5. **Stressed grid re-run finer** (§2.4) + the §5 presentation tables/figures.
6. **Opening auction** (§6) and the §7 quick-fixes — realism/hygiene, optional.
