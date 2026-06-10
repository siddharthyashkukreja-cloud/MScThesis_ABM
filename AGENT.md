# AGENT.md — collaboration briefing + change history

Briefing for any AI assistant working on this thesis. `README.md` is the
thesis-facing description of the **current** model; this file covers *how to
work here* and the **full history of changes and the effect each had** —
including the ablation study. Read the "Project at a glance" snapshot first,
then the relevant code, before suggesting anything.

## Status & handoff (read first)

**Phase:** the model is COMPLETE and the calibration is LOCKED (D60) — the project is now in the
**thesis-writing** stage. A fresh chat/agent needs only three files: **`README.md`** (model / code
reference), **`writing.md`** (thesis findings, numbers, references, chapter structure — the
writing-agent context pack), and **this `AGENT.md`** (dev history + how-to-work-here). README +
writing.md are the self-contained context pair; AGENT.md is the dev log.

**Locked baseline θ** (`globals.CALIBRATED`; grid headline, surrogate cross-validated — D60):
calm `ft_sigma_c=0.80, zi_alpha=0.34, zi_delta=0.05`; stressed `ft_sigma_c=0.50, zi_alpha=0.26,
zi_delta=0.05, p_zi=0.15`; `zi_mu` pinned 0.025. D_grid 43.8 / 7.9 — **D is comparable only at a
fixed seed count** (the KS component's `s_KS` is sim-sized; see writing.md §4.1).

**Open decisions (writer):** (1) pick the RQ — RQ-A/B/C in writing.md §2 (the model most directly
answers B, then A); (2) optionally adopt E2's calibrated `mt_lambda` for a small stressed gain, or
keep the parsimonious locked baseline. The calibration-extension campaign (E1–E6) is DONE — verdict
(parsimony; E5/combo/E4 rejected on confirmation) in writing.md §4.3 and the D59 entry below.

**Where things live.** Code → `model/`, `data/`, `calibrate.py`, `run_simulation.py`,
`covid_contagion.py`. Notebooks → `model_design.ipynb` (built by **`build_nb.py`**),
`empirical_analysis.ipynb`. Calibration results → `output/campaign/` (E0–E6 screen),
`output/campaign_v2/` + `output/e5_confirm/` (E2/E5 lock-in), **`output/baseline_grid/`** (the locked
grid θ + `output/calibration_grid_*.csv` loss surfaces), `output/baseline_hires/` (surrogate
cross-check). Run scripts: `run_calibration_all.sh` / `run_overnight.sh` (full pipeline),
`run_baseline_grid.sh` (the lock; has a single-instance lock + `< /dev/null`), the campaign
`run_*.sh` (provenance, done). `calibration_campaign.md` = the Claude-Code campaign brief.

**Git:** everything is on branch **`calibration-campaign`** with uncommitted changes — commit (and
merge to `main` if desired) at migration. Nothing has been pushed.

## Project at a glance

MSc thesis ABM of a CCP-cleared single-asset (ES front-month futures) market,
pure Python (no Simudyne SDK). Two tiers: a market-microstructure layer (LOB +
traders + exogenous V_t) and a central-clearing tier (CCP + clearing members +
balance sheets + margin).

**Current roster (D53 — clients scaled 2×, CMs unchanged at 15, mix preserved).**
Calibration population: 40 FT + 20 MT + 40 ZI (folds 30 FT + 10 BCM into
`n_fundamental`; no market maker, no volatility trader). Runtime: 30 FT + 10 BCM
(FT-cast, own-account) + 20 MT + 40 ZI on the LOB (100 agents); 5 NBCM + 1 CCP off
the book; 5 of the 10 BCMs carry client books. 90 clients clear through the 10
client-carrying CMs, 9 per CM. FT clients are 30 (not 20) so FT-equiv = 40,
preserving the pre-doubling FT-equiv:MT:ZI = 40:20:40 mix at 2× scale. The behavioural θ has
since been **re-calibrated and locked at thesis-final resolution (D60 — see Status & handoff)**.

**Fundamental V_t.** Primary: Kalman-filtered real efficient price of the ES
mid (`data/v_kalman.py`) + real local volatility `σ_t` (EWMA of r², 30-min
half-life) piped into the FT belief width. Alternative (robustness): SV-MJD
(`data/v_gbm.py`). The `fv_*.csv` paths are currently Kalman.

**Calibration.** Regime-specific behavioural loop: calm 3-d `{ft_sigma_c,
zi_alpha, zi_delta}`, stressed 4-d `{+ p_zi}`. `zi_mu` pinned 0.025; `ft_alpha
= mt_alpha = 1.0`, `mt_lambda = 0.05` pinned; `p_zi` calm = L2 value, stressed
calibrated. Two methods, reported together: surrogate-assisted SMM
(`calibrate.py run`) and exhaustive grid search (`calibrate.py grid`). Loss is a
5-component standardised-moment distance: **KS + V + ACF1 + ACF2 + Hill** (ACF1/ACF2
forward-3-lag-smoothed at lags {1, 5, 10, 20}), Franke–Westerhoff inverse-bootstrap-SD
weights. Kurtosis is diagnostic only. Run both at thesis resolution via
`./run_calibration_all.sh` (deletes the stale LHS cache, runs grid + surrogate for both
regimes, records every moment, merges both regimes into one JSON).

**Clearing.** Built end to end. USD variation-margin cycle every 60 min (`× VOLUME_LOT
× CONTRACT_USD`; per-regime `VOLUME_LOT = 30 / 60`, `CONTRACT_USD = 50`); live client
novation; capital-ratio (8% floor) Almgren–Chriss deleveraging; cover-2 default fund;
five-level waterfall; CCP Almgren–Chriss fire-sale of the defaulted book. All ODD /
regulatory constants live in `globals.CCP_CALIBRATION`. Calm yields rare pooled-DF-level
defaults; stressed yields a default cluster that consumes the exchange skin-in-the-game.

**Client clearing (D52 — the contagion channel).** Each client (FT/MT/ZI) carries its own
balance sheet, posts its own variation margin from its own cash, is position-bounded by
margin capacity (IM ≤ cash ⇒ exposure ≤ 5× cash ⇒ ≈ 20% capital ratio), **freezes** at the
8% floor and **defaults** at cash ≤ 0 — on which its CM absorbs the uncollected margin plus
the (1−recovery) close-out haircut, draining CM cash and potentially toppling the CM
(client → CM → CCP cascade). Client cash is type-matched (FT asset-manager scale > MT CTA >
ZI noise), tuned so the floor binds under stress but not calm. Stressed currently defaults
the small ZI accounts; making that reliably topple an NBCM is the deferred tuning (NBCM size
heterogeneity + client-book concentration).

## Cardinal rule

**Read the current repo files before suggesting anything.** This project iterates
aggressively — agents and mechanisms have been added, removed, and re-added in
different forms (market maker, volatility trader, ContTrader, the loss
composition, the placement law, the fundamental process). Without grounding in
the current state you will reinvent something already tried and rejected. The
"Change history" section below records what was tried and what happened.

## Reference priority chain

Every design decision cites one of these; higher-ranked refs win on conflict.

1. **Simudyne CCP Risk Model ODD** (primary spec).
2. **Deloitte** CCP Risk Model webinar + CCP-resilience white paper (clearing-tier
   mechanics; the SV-V_t prompt came from the *webinar* — the white paper has no
   equations and the ODD is incomplete on the fundamental).
3. **Majewski, Ciliberti & Bouchaud (2018)** — extended Chiarella estimation (the
   fundamental is a latent state filtered from price).
4. **Gao et al. (2022)** HFABM (arXiv:2208.13654) — MM spec, Hill tail target,
   surrogate-SMM; the companion XGB-Chiarella (arXiv:2208.14207) — Kalman
   fundamental (§2.5.2), KS distributional target (§3.2.4).
5. **Gao et al. (2023)** "Deeper Hedging" / Chiarella-Heston (arXiv:2310.18755) —
   grid-search calibration of D(ϑ).
6. **Vytelingum et al. (2025)** — ABM liquidity risk (arXiv:2505.15296).
7. **Cont, Stoikov & Talreja (2008)** — stochastic LOB; **Farmer, Patelli & Zovko
   (2004)** — ZI baseline + geometric placement.

Supporting: Stein & Stein (1991) / Heston (1993) (OU-on-σ); Merton (1976) (jumps);
Cont (2001) stylised facts; Cont (2005) multi-scale vol memory; Franke &
Westerhoff (2012) / Künsch (1989) (block-bootstrap weights); Hill (1975) /
Resnick (2007) (tail index); Almgren & Chriss (2000) (impact; CM/CCP fire-sale);
Lamperti, Roventini & Sani (2018) (surrogate-calibration inspiration).

Note: earlier code mis-cited the HFABM paper as "Cao et al. 2024" and Vytelingum
as "Krishnen" — both corrected.

## Hard rules (Sid's preferences)

1. **No invented calibration.** Numbers come from literature, data, or
   calibrated θ. No made-up defaults.
2. **No synthetic data.** Empirical inputs only (DataBento ES under `data/`).
3. **No print statements** in production code (calibration / data CLIs are fine).
4. **No saved images unless asked.**
5. **No excessive comments.** One docstring per class/function with intent +
   citation; one-line comments for non-obvious mechanics only.
6. **Snippets in chat — don't push to GitHub unless told.**
7. **Update `README.md` / `AGENT.md` only when asked.**
8. **No emojis in files.**
9. **Citations in code** for every meaningful design choice.
10. **Flag deviations explicitly** from a higher-ranked ref, with the reason.
11. **Prefer simplicity** — complexity costs defensibility; removing a
    weakly-justified mechanism is valid.

## Repo layout

```
README.md                  thesis-facing current state
AGENT.md                   this file
model_design.ipynb         presentation notebook — market + clearing design, figures, stylised facts
data/  data.py roll.py v_kalman.py v_gbm.py p_zi.py impact.py  processed/  fv_*.csv
model/ globals.py lob.py agents.py clearing.py simulation.py
run_simulation.py          entry point + population/clearing builder
calibrate.py               agent calibration: `run` (surrogate) + `grid`
analysis_long_run.py       long-horizon 1-min/daily moment validation
empirical_analysis.ipynb   empirical ES stylised facts   (.gitignore for caches)
output/                    calibration + simulation outputs
```

## Calibrated state + re-copy

`globals.CALIBRATED` is the source `run_simulation.py` and the notebooks read. The
**grid optimum** (`output/calibrated_params_grid.json`) is the wired headline; the
surrogate (`output/calibrated_params.json`, `theta_stage2`) is reported as a
cross-check. The two agree on the identified levers (`ft_sigma_c`, `p_zi`) but differ
on the weakly-identified ZI rates (`zi_alpha`, `zi_delta`) at near-flat loss — so the
grid is the defensible headline. Re-copy the grid optimum into `CALIBRATED` after each run.

## Calibration pipeline

Offline (one-shot): `data/v_kalman.py generate-all` (Kalman fv, primary);
`data/v_gbm.py calibrate` + `generate-all 42` (SV-MJD, alternative); `data/p_zi.py
calibrate` (geometric depth from MBP-10).

Agent loop (`calibrate.py`):
- `python3 calibrate.py run [calm|stressed] [N_LHS N_DAYS N_RUNS N_REFINE N_PER_REFINE N_STAGE2]` — surrogate-assisted SMM.
- `python3 calibrate.py grid [calm|stressed] [N_PER_DIM N_DAYS N_RUNS]` — exhaustive grid search (Gao 2023).
- The stressed sim auto-caps to its ~29-day data window; calm subsamples to N_DAYS.
- **Cache:** delete `output/calibration_lhs_*.csv` when the param set or population
  changes (the guard catches param/moment-name changes; a structural change at the
  same param set is NOT caught — delete manually). The calm LHS cache regenerates
  automatically after the `zi_mu`-drop param-set change.

## Change history & effects

The decision log, most-recent first. Each entry is *what changed* and *the effect
it had*. Older micro-iterations are condensed; full per-step history is in git.

### Recent architecture (current model)

- **D60 — thesis-final baseline θ LOCKED (grid headline, surrogate cross-validated).** Grid search
  (Gao 2023; calm 7³=343 nodes, stressed 5⁴=625, n_days=20, n_runs=3 — `output/baseline_grid/`,
  full surface in `output/calibration_grid_{regime}.csv`) and the high-res surrogate (160 LHS /
  6 seeds, 2 refine — `output/baseline_hires/`) land on the SAME optimum: calm ft_sigma_c 0.80/0.83,
  zi_alpha 0.34/0.343, zi_delta 0.05/0.063; stressed 0.50/0.53, 0.26/0.271, 0.05/0.123, p_zi
  0.15/0.177 (grid/surrogate). Grid wired into `globals.CALIBRATED` per the stated convention:
  calm `{0.80, 0.34, 0.05}` D_grid 43.77; stressed `{0.50, 0.26, 0.05, p_zi 0.15}` D_grid 7.92.
  **Methodology findings recorded:** (i) the KS loss component rescales with seed count (s_KS is
  sim-sized: calm KS 16.8 at 2 seeds ≙ 28.6 at 6 — observed ×1.70 vs predicted ×1.73), so **D is
  comparable only at fixed n_runs**; (ii) calm `ft_sigma_c` bound tightened 2.0→1.1 (weakly
  identified tail lever drifted to a worse interior basin ~1.35, Hill Δ7.4/D 57, under the wide
  bound — the good basin is 0.5–0.9); (iii) boundary-sitting at the lock: stressed ft_sigma_c at
  its 0.5 floor, zi_delta at the 0.05 D58 book-stability floor (a near-flat loss direction — calm
  D 43.8 at 0.05 vs 44.0 at 0.48), stressed p_zi at 0.15 — all justified bounds, disclosed.
  Operational fixes en route: `< /dev/null` stdin redirect in the run scripts (an unattended
  second python died with "init_sys_streams: Bad file descriptor" after the launching terminal
  closed) and a single-instance lock in `run_baseline_grid.sh` (double-launches had clobbered
  shared caches twice).

- **D59 — calibration robustness campaign + `writing.md` context pair.** Ran a campaign (surrogate
  SMM only, both regimes, env-flag-gated configs in `calibrate.py`; results under `output/campaign/`,
  `output/campaign_v2/`, `output/e5_confirm/`, `output/baseline_hires/`): E0 baseline; E1 clearing
  tier active in the loop; E2 `mt_lambda` in the loop; E3 `mt_lambda` pinned ~3h half-life; E4a/E4b
  more / two-cohort momentum; E5 FT/MT activation prob + per-order cancellation (revives the
  D36-rejected gate, post-TTL-removal); COMBO E2+E5; E6 gaps-vs-shock contagion. **Verdict
  (parsimony):** the baseline is hard to beat. E5's apparent −34% stressed win at screening (D 13.8→9.2,
  192 LHS) **did not replicate** at higher resolution (256 LHS / 5 seeds → D 16.9, worse than
  baseline) — a poorly-identified 4-param addition with high run-to-run variance; **high surrogate R²
  (~0.9) did not guarantee a reproducible optimum** (replicate across seeds). E4a/E4b/COMBO: no robust
  gain; the two-cohort MT did NOT revive long-horizon clustering (the single-timescale limit stands).
  **Only consistent win: E2** (`mt_lambda` calibrated) — small (~1 D) reliable stressed gain, one
  well-identified param; optional. **E1:** stressed calibration is NOT clearing-invariant (D 13.8→28
  with the tier active) — methodology flag. **E6:** at equal drawdown a single concentrated shock is
  far worse than the gapped path (~3× client defaults; mutualisation at c=1 vs c≈2.5–3) — contagion
  tracks move concentration/speed, not magnitude; the overnight-gap structure is a mitigant (VM
  collects between jumps). `globals.CALIBRATED` left at the baseline; **thesis-final baseline θ being
  locked** via `run_baseline_hires.sh` (160 LHS / 6 seeds). Nothing pushed; all on the
  `calibration-campaign` branch. Also added **`writing.md`** — the thesis-writing context pack
  (results/numbers, references-usage, chapter structure, limitations); README + writing.md are now the
  self-contained context pair, AGENT.md stays the dev-only log.

- **D58 — removed the LOB order-TTL; ZI cancellation governs order lifetime.** The hard 10-step
  order expiry (`LOB.age_orders`, `Order.ttl`, `order_ttl`, ODD §Mech #7) is gone — resting orders
  now leave the book only by fill or explicit cancellation. FT/MT already manage their single order
  by replace-on-new (they act every step, so the TTL never bound them); ZI cancels each resting order
  w.p. `zi_delta` per step (Cont-Stoikov-Talreja 2008 / Farmer et al.), which is now the SOLE control
  on ZI order lifetime. **Rationale:** the TTL only ever acted on ZI (the one type that accumulates
  resting orders) and was redundant with `zi_delta` — worse, it MASKED it: order life is
  `min(Geometric(zi_delta), 10)`, so whenever `1/zi_delta ≳ 10` (i.e. `zi_delta ≲ 0.1`, as in the old
  stressed fit at 0.045) the TTL dominated and `zi_delta` was near-unidentifiable. Removing it makes
  `zi_delta` a clean, calibratable lever. **Deviation from the ODD flagged explicitly** (the CST/Farmer
  per-order cancellation rate is the standard ZI order-lifetime mechanism, superseding the blanket
  ceiling). Guard: `zi_delta` lower bound raised 0.005→0.05 (mean life ≤20 steps) so the book cannot
  accumulate stale orders without the backstop — verified depth stays bounded (calm ~220, stressed
  ~270, no drift over a 20-day run). At the current θ (`zi_delta` 0.33–0.48) the TTL rarely bound, so
  the loss targets (KS/V/ACF/Hill) barely move; the diagnostic kurtosis rises (the few long-resting
  orders are no longer culled). **Re-run calibration** to re-settle `zi_delta`. Also fixed stale ZI/FT
  docstrings (the ZI rates ARE calibrated; FT no longer relies on the TTL backstop).

- **D57 — data-driven RTH-session boundaries (replaces the fixed 390-bar day).** Real ES sessions in
  the processed data are ≈405 bars and vary (149–406), not the hardcoded 390. `data/v_kalman.py` now
  writes the REAL per-bar timestamps to `fv_{regime}.csv` (was a synthetic 390-per-business-day stamp),
  and `globals.day_start_steps(regime)` recovers the true session opens from the `ts` date-changes.
  Every consumer of the day boundary now reads it from the data: the simulator reprices/re-anchors at
  the real open (`Simulation._day_start_rows`, offset by `_v_start`), and the overnight-return exclusion
  + daily-close sampling are data-driven in `calibrate._intraday_logret`, `analysis_long_run`,
  `covid_contagion` (window anchored to the Mar-9-2020 session), and the notebook. Effect: fixed the two
  stressed artifacts — the "0-volume block" (was un-repriced overnight gaps landing mid-sim-day from the
  390-vs-405 misalignment; now 0.5% thin no-cross steps localised to the COVID vol peak, max run 5 min)
  and the "1-day graph that wasn't one day" (now a real 406-bar session). The misalignment had been
  leaking ≈28 overnight-gap returns into the "intraday" calibration moments, inflating the simulated
  stressed kurtosis (≈690 → ≈190 once excluded at the real boundary). Also vectorised the Kalman MLE
  across days (numerically identical; `generate-all` 43s+ → ~6.5s). `SIGMA_V`/`V0` unchanged.
  **Re-run calibration after this** (the corrected moments change the loss surface).

- **D56 — overnight gaps retained in the fundamental.** `data/v_kalman.py` now concatenates the
  per-day smoothed series at their REAL levels (was a continuous gap-free splice), so the COVID
  episode carries its true ≈−33% drawdown (was ≈−7%) — overnight limit-down gaps are a primary CCP
  default driver. The simulator opens each RTH day at the gapped V_t by **repricing the book** at
  the day boundary (`LOB.reprice` — a clean between-session jump, not an empty-book warm-up),
  re-anchors each trader's intraday price memory (so the MT EWMA doesn't read the gap as trend), and
  **excludes the day-boundary return** from the calibration moments (`calibrate._intraday_logret`,
  `analysis_long_run`, notebook), matching the empirical overnight exclusion. Effect: the clearing
  tier now sees gap risk (≈4–9 client defaults under gapped COVID vs ≈1 gap-free, still CM-absorbed;
  reverse-stress mutualisation onset drops from ≈4× to ≈2.5× COVID). An A/B vs a gap-free re-splice
  confirmed the intraday calibration moments are essentially unchanged by the gaps (kurt ≈700 either
  way — a PRE-EXISTING stressed-tail property of the current params for re-calibration to tune, NOT
  a gap artifact). `SIGMA_V`/`V0` unchanged (intraday / day-1 open). **Re-run calibration after this.**

- **D55 — real margin methodology (IM/VM/DF) + the COVID contagion experiment.** Flat 20% IM →
  **procyclical VaR/SPAN scan** (`globals.im_fraction(σ)`: 99%/2-day VaR floored ~6% ≈ CME ES margin
  → ~12% stressed; EMIR Art. 41 / CME SPAN). CM **capital ratio → cash / initial margin** (CFTC Reg
  1.17 — net capital ≥ 8% of risk margin, not gross notional; fixed the dead NBCM stop-out). Default
  fund → **cover-2 Stress-Loss-Over-Initial-Margin** (Euronext Clearing module A9: top-2 members'
  loss above posted margin under an extreme-but-plausible move, ×1.1 buffer). Waterfall → absorbs the
  defaulter's **realised cash deficit** (not a flat (1−recovery)·notional haircut, which overstated
  close-outs and spuriously insolvented the CCP at −16%). **BCM house book capped by a VaR limit**
  (FRTB, β=5% of capital) — unbounded FT-style prop accumulation had reached ~$30–40B and inflated
  the DF to ~$7.7B; the limit shrinks it to a realistic ~$0.5B. Added `covid_contagion.py` (cascade
  trace + reverse-stress breach multiplier) and `Simulation(v_start=…)` to run any window.

- **D54 — client default → CM assumes + liquidates.** On a client default the CM ASSUMES the
  position onto its book and liquidates it via Almgren-Chriss (BCM sends its own orders; the NBCM's
  are routed by the CCP, attributed to the NBCM so fills mark down its assumed inventory), covering
  the uncollected VM shortfall — the loss is realised by marking the assumed book as the fire-sale
  walks the price (deficit-consistent, no flat haircut).

- **D53 — client population scaled 2× (CMs held at 15), price-formation mix PRESERVED.**
  Clients scaled to 30 FT + 20 MT + 40 ZI = 90 (9 per client-CM); `n_bcm`/`n_nbcm`/
  `n_bcm_with_clients` unchanged. FT clients set to **30 (not 20)** so FT-equiv = 30 + 10
  BCM = 40, keeping FT-equiv:MT:ZI = **40:20:40** (= the pre-doubling 2:1:2 mix at 2×
  scale; calibration POP `n_fundamental=40`). An initial "double the clients" pass used
  20 FT (mix drifted to 33/22/44, ZI-heavy) and the loss rose — corrected by the 30-FT
  bump. Effect: more attached clients ⇒ more client defaults + heavier per-CM loss
  concentration. The existing θ carries over (loss validated similar at the old params:
  calm 47.8→50.5, stressed 10.9→8.8; the residual is a finite-size tail effect — 2× more
  agents thins calm tails / fattens stressed clustering). Full re-calibration planned.
- **D52 — client-clearing contagion mechanism.** Clients gained their own balance sheet:
  per-cycle VM paid from own cash; margin-capacity position bound (IM ≤ cash); freeze at
  the 8% floor; default at cash ≤ 0 → CM absorbs uncollected VM + (1−recovery) close-out
  haircut. `agents._client_cap_qty` + a stop/has_defaulted gate on FT/MT/ZI submit; the
  margin cycle restructured into two phases (client calls → CM own VM + default). Type-
  matched client cash (FT/MT/ZI bands, globals). This is the thesis's central client →
  CM → CCP contagion channel.
- **Start-to-end audit (every file vs docs + papers).** Fixed: (i) `_apply_fill` was
  giving cleared clients the nominal full-notional debit on top of VM (double-counting) —
  now cleared accounts settle via VM only; (ii) `analysis_long_run._acf_smooth` used
  CENTRED smoothing while the loss uses FORWARD (and falsely claimed they matched) — now
  forward; (iii) stale `calibrate.py` comment block ("Hill dropped / 4-component / ACF1
  {1,5,10}") rewritten to the real 5-component loss; (iv) progress-banner off-by-one
  (`n_steps`). Flagged (author's call, not fixed): p_zi geometric cited to CST/Farmer
  (power-law) and v_kalman cited to XGB-Chiarella §2.5.2 (local-level, not their Chiarella
  state) — both better worded "in the spirit of".
- **ACF1 gained a 5-min lag** ({1,10,20} → {1,5,10,20}), matching ACF2's short-lag set.
- **`run_calibration_all.sh` + merge-save.** One thesis-resolution runner: deletes stale
  cache, runs grid + surrogate for both regimes, records every moment (grid JSON now keeps
  the optimum's moments too), prints a cross-method comparison. `_merge_save` makes per-
  regime runs accumulate in one JSON (resolves the "rewritten per regime" gotcha).
- **Clearing layer built end to end (was scaffold + margin).** USD-denominated VM
  (`book_pos · Δmid · VOLUME_LOT · CONTRACT_USD`); live client **novation** (each
  client's LOB inventory pulled into its CM book per cycle); capital-ratio (8%)
  Almgren–Chriss deleveraging (BCM fire-sale / NBCM stop-out); **cover-2 default fund**
  recomputed each RTH day; **five-level waterfall** (defaulted IM+DF → exchange SITG →
  pooled DF → surviving-CM cash → CCP); CCP **Almgren–Chriss fire-sale** of the
  defaulted book via LOB market orders (endogenous impact = the contagion channel).
  Per-regime `VOLUME_LOT = 32 / 72` matched to empirical ES volume; regulatory-scale CM
  cash (CCP $7.5B, BCM `U[5B,10B]`, NBCM `U[0.2B,1B]`). Effect: calm gives rare
  pooled-DF-level defaults (CCP solvent); stressed gives a 3-BCM/5-NBCM default cluster
  that consumes the exchange SITG to $0 (CCP still solvent). All ODD/regulatory pins in
  `globals.CCP_CALIBRATION` — none SMM-calibrated.
- **Grid optimum wired as the calibration headline.** `globals.CALIBRATED` now holds the
  grid optimum (calm D=47.8, stressed D=11.6); the surrogate run is reported alongside as
  a cross-check. They agree on `ft_sigma_c` (≈0.5) and `p_zi` (≈0.15); the ZI rates are
  weakly identified (flat loss) and differ between methods — hence grid is the headline.
- **Notebook figures aligned to the literature.** `model_design.ipynb` price panels now
  overlay the real ES mid + Kalman fundamental + simulated mid on one axis (real mid is
  the same series the fundamental smooths, so the fundamental is *less* noisy, not more);
  stylised-fact panels follow HFABM / XGB-Chiarella — return / |return| / squared-return
  ACF as offset stems (empirical vs simulated) + the ±1.96/√N band.
- **Repo cleanup.** Deleted the two stale notebooks (`analysis.ipynb`,
  `clearing_analysis.ipynb` — superseded by `model_design.ipynb`); untracked
  `__pycache__`/`.DS_Store` and added a `.gitignore`.
- **Grid-search calibration added (Gao 2023).** Added a `grid` CLI that minimises
  the same loss D on a regular grid over the regime box, reported alongside the
  surrogate. Effect: a transparent, exhaustive, low-dimension-defensible method
  that cross-validates the surrogate optimum and yields the full loss surface.
- **Regime-specific parameter sets + `zi_mu` dropped.** Calm 3-d `{ft_sigma_c,
  zi_alpha, zi_delta}`; stressed 4-d `{+ p_zi}`. `zi_mu` pinned at the CST 0.025
  baseline. Effect: a leaner, cleaner loop; `p_zi`-for-stressed sharpens the
  stressed fit (ablation C9) while calm keeps the directly-measured L2 depth;
  dropping `zi_mu` costs ~0 (ablation C6).
- **Stressed simulation capped to its data window.** `_sim_steps` makes the
  stressed sim span the full ~29-day COVID window and never run past it. Effect:
  sim and empirical moments cover the same period; no frozen-fundamental artifact
  at the end of long runs.
- **Hill re-added to the loss → 5 components (KS + V + ACF1 + ACF2 + Hill).** Hill
  had been dropped (kurtosis deemed "adequate"); under the Kalman fundamental the
  calm kurtosis blew up to ~108. Re-adding Hill (HFABM §4.1.1) alongside KS
  (XGB-Chiarella §3.2.4). Effect (ablation C0 vs C10): the calm tail collapsed back
  into line — kurtosis 108 → 9, Hill 1.67 → 3.02 — because Hill gives the optimiser
  a direct, reachable tail-index lever that pulls `ft_sigma_c` down. KS alone is
  insufficient (tail blows up); Hill alone leaves the bulk distribution untargeted.
- **ACF lag sets refocused.** ACF1 (return ACF) lags `{1, 10, 20}` (later `{1, 5, 10, 20}`
  — see the D53 audit entry above), forward 3-lag smoothed; ACF2 (|return| ACF) lags
  `{1, 5, 10, 20}`. Effect: |r| is far less noisy than r², so clustering actually counts.
- **Surrogate stage-1 switched to pool-argmin.** Replaced multi-start L-BFGS-B
  (a tree surrogate has no usable gradient) with argmin over a Sobol candidate pool;
  single XGBoost regressor θ → D (not per-moment). Effect: stable convergence on the
  true surrogate landscape.
- **`ft_sigma_c` unpinned and calibrated.** Was pinned at √390 ("one daily V_t
  std"). Effect: it is THE return-tail lever — at √390 the FTs sweep the book to
  their outermost reservation and kurtosis explodes (ablation C4: kurt 5256/1409,
  D doubles); calibrated low (≈0.6) it reproduces the empirical tail.
- **Fundamental switched SV-MJD → Kalman.** The fundamental is now the
  Kalman-smoothed real efficient price (`v_kalman.py`) rather than the synthetic
  SV-MJD. Effect: it carries the *real* return tails and real multi-scale clustering
  (via `σ_t`), and is far more defensible (no synthetic jump/vol parameters; no
  "injected clustering" critique). The ablation (C1) confirmed Kalman fits *better*
  than SV-MJD once the tail is controlled. SV-MJD retained for robustness.
- **Placement reverted log-normal → geometric, data-fit.** Limit placement depth is
  `k ~ Geometric(p_zi)` with `p_zi` fit from real L2/MBP-10 depth (`data/p_zi.py`),
  reverting the calibrated log-normal `depth_mean`/`depth_sigma`. Effect: a thicker
  near-mid book grounded in measured depth; `depth_mean`/`depth_sigma` parked.
- **Market maker removed (n_mm 4 → 0).** Effect: the always-quoting MM was damping
  volatility and clustering; removing it let FT/MT/ZI order flow drive price and
  revived short-horizon clustering. Confirmed by ablation C2 (MM hurts both regimes).
- **Documentation + citations reconciled.** Cao → Gao (2022), Krishnen → Vytelingum,
  HFABM MM is §3.6 (not §3.2), SV-V_t prompt is the Deloitte *webinar*.

### Earlier history (condensed)

- **Clearing tier (scaffold + margin).** CCP + BCM (FT-cast) + NBCM + balance sheets
  + bidirectional links built; the 60-tick variation-margin cycle runs (VM settles
  M2M into cash, IM/MM recomputed, call flagged, default on cash exhaustion;
  futures-margin accounting). (Default fund, waterfall, USD VM and novation since built —
  see "Recent architecture".)
- **`VOLUME_LOT` / `CONTRACT_USD` relabel.** 1 model lot = an institutional ES block;
  applied at reporting + notional + margin only, so returns/ACFs/Hill are invariant.
  Effect: simulated contract volume and USD notional match real ES scale without touching
  the calibrated dynamics. (Later made per-regime `45 / 82`, then `32 / 72` at the D53
  doubling, and the VM cycle put on this
  USD scale — see "Recent architecture".)
- **Fundamental lineage.** Constant-σ GBM → Merton jump-diffusion (jumps gave excess
  kurtosis but no clustering) → SV-MJD (Vasicek-OU σ_t; gave short-horizon clustering;
  Deloitte-webinar architecture) → Kalman efficient price (current).
- **Volatility / Cont traders trialled and removed.** Multiple forms; none produced
  durable clustering at this LOB (see "What's removed").
- **Momentum trader** went two-cohort (long/short) and folded back to single-cohort
  EWMA; the D27 market-order branch was reverted (it corrupted the return ACF) — MT
  is now limit-only.
- **ZI rates entered the loop** when the MM was first dropped and ZI became the
  primary liquidity driver; doubling the ZI count densified the near-mid book and
  removed FT-flow choppiness.

## Ablation study (C0–C11)

12 cells, one design change each, on the KS+Hill 5-component loss, measured against
the C0 baseline (calm D = 48.36, stressed D = 29.11). Full per-cell detail was in
`ablation/LOG.md` (now folded here; the folder is removed).

| Cell | Change | calm D | stressed D | Finding |
|------|--------|--------|-----------|---------|
| C0 | Baseline (Kalman, KS+Hill) | 48.36 | 29.11 | reference (calm kurt 9, Hill 3.02) |
| C1 | SV-MJD fundamental | 47.17 | 36.92 | worse tail in both; **Kalman is better** |
| C2 | Market maker on (n_mm=4) | 51.46 | 33.42 | both worse — **confirms MM removal** |
| C3 | Volatility trader on (n_vt=10) | 52.80 | 21.27 | **stressed big win** (Hill 2.44→3.17), calm liability → regime-asymmetric |
| C4 | Fix ft_sigma_c = √390 | 102.69 | 71.90 | catastrophic (kurt 5256/1409) — **ft_sigma_c must be calibrated** |
| C5 | Fix zi_alpha = 0.15 | 66.35 | 26.68 | calm much worse — zi_alpha matters (calm) |
| C6 | Fix zi_mu = 0.025 | 48.28 | 29.95 | flat — **zi_mu is freely droppable** |
| C7 | Fix zi_delta = 0.15 | 50.77 | 30.35 | calm mildly worse — moderate sensitivity |
| C8 | Fix ALL ZI rates at CST | 59.07 | 33.43 | both worse — **calibrating ZI rates is justified** |
| C9 | p_zi in the loop (5-d) | 49.70 | 23.00 | **stressed win** (sparser book); calm flat |
| C10 | KS-only (drop Hill) | 51.59* | 25.18* | calm tail **EXPLODES** (kurt 109, Hill 1.67) |
| C11 | Hill-only (drop KS) | 24.95* | 10.75* | tail fine but KS untargeted |

*C10/C11 D not comparable to the others — different component count.

**Headline conclusions baked into the current model:**

1. `ft_sigma_c` is the dominant lever (C4); calibrate it, never pin √390.
2. KS + Hill together are vindicated (C10 vs C11): KS pins the body, Hill pins the
   tail; each controls a failure the other cannot.
3. Kalman beats SV-MJD once the tail is controlled (C1) — and is more defensible.
4. The market maker hurts both regimes (C2); the volatility trader and free `p_zi`
   are *stressed-specific* improvements (C3, C9) → regime-specific structure, not a
   global toggle. `p_zi`-for-stressed is adopted; the VT is held in reserve.
5. Among ZI rates, calibration value ranks zi_alpha > zi_delta > zi_mu (C5/C7/C6);
   `zi_mu` dropped, the other two kept.

## Known structural limits (document; don't try to fix without a strong cite)

1. **Long-horizon volatility clustering** (|r| ACF at lags ≥ ~30 min). A
   single-timescale momentum cohort cannot produce multi-scale memory (Cont 2005;
   HAR-RV, ABDL 2001/3). The two-timescale momentum trader or the volatility trader
   (regime-specific) are the open candidates. Currently owned as a documented limit.
2. **Client → NBCM default not yet firing; waterfall Level 4 still dormant.** Client
   positions are now margin-capacity bounded (D52), so client books are small; only the
   small ZI accounts default in stress and their losses, though larger after the D53
   doubling ($1.37B absorbed, 8/CM), still stay under FCM-scale NBCM cash, so no NBCM
   fails through the client channel and the surviving-member mutualisation step (**L4**)
   never fires (stressed defaults absorbed by L1–L3 down to the SITG; BCM own-book
   accumulation is still unbounded until the 8% deleverage). The deferred tuning (#63 —
   NBCM size heterogeneity + client-book concentration) is the lever. **Highest-leverage
   open item.**
3. **Population scaling is calibration-coupled.** D53 doubled the FT/MT/ZI clients for
   more defaults and concentration (8/CM) — but they are also the price-formation
   population, so this forced the running re-calibration (CALIBRATED is stale until it
   finishes). Beyond headcount, default/contagion power comes from Monte-Carlo over seeds
   (outcome distributions) and designed sweeps (CM count, client-book concentration,
   margin methodology, tiered-vs-direct). NBCMs are off-LOB, so the non-banking tier scales
   without recalibrating; the LOB traders do not.

## What's removed (do NOT re-add without a strong cite)

- **Market maker** — damps volatility/clustering without improving the tail (C2).
- **Volatility trader** — helps stressed, harms calm (C3); retained only as a
  possible regime-specific stressed option, not a standing agent.
- **ContTrader** — Cont (2005) threshold-with-inertia trader; no durable clustering
  gain at this LOB.
- **Two-cohort momentum / MT market-order branch** — the market branch corrupted the
  return ACF.
- **Pareto `_draw_qty`** — power-law trade sizes need sqrt-impact (Gabaix); the
  discrete LOB gives linear impact, so they degraded Hill. Volume scale is handled by
  the `VOLUME_LOT` relabel instead.

## Gotchas

1. **Delete `output/calibration_lhs_*.csv` on a param-set / population / structural
   change.** The cache guard catches param/moment-name changes but not a structural
   change at the same param set.
2. **Re-copy `globals.CALIBRATED` after every calibration.**
3. **`calibrated_params.json` is rewritten per regime** — a calm-only run overwrites
   the stressed entry. Run sequentially or merge by hand.
4. **Per-step Bernoulli is ODD-native — no `dt` rescaling.**
5. **FT triggers on `|reservation − mid|`, not `|V_t − mid|`.**
6. **MT is limit-only; FT/MT trade every step** (`ft_alpha = mt_alpha = 1`,
   replace-on-new). Don't re-add a Bernoulli activation gate.
7. **`σ_t` reaches the LOB only through the FT belief width.** To reach another agent,
   pass it explicitly via `SimContext`.
8. **`VOLUME_LOT` / `CONTRACT_USD` apply at reporting / capital-ratio only** — never
   inside the matching engine.
9. **No xgboost / scipy in the sandbox.** `calibrate.py` is local-only; sandbox smoke
   uses `run_simulation.py` / `analysis_long_run.py` or direct function calls.
10. **Stressed sim length is data-capped**, not `N_DAYS` — the `N_DAYS` arg only
    affects calm and the KS sample-size for the s_KS weight.

## Stage roadmap

| Stage | Status |
|---|---|
| Market layer (FT/MT/ZI, calibrated) | done; grid + surrogate cross-validated |
| Clearing scaffold (CCP/BCM/NBCM + balance sheets) | done |
| USD variation-margin cycle (60-min) | done |
| VOLUME_LOT matched to empirical ES volume | done (per-regime 30 / 60; globals.py is source of truth) |
| Live client books (trade novation) | done |
| Cover-2 default fund (EMIR) | done |
| 5-level waterfall | done |
| Almgren–Chriss deleveraging + CCP fire-sale | done |
| Right-size cover-2 DF → activate L4 mutualisation | open (highest-leverage) |
| Monte-Carlo ensembles + clearing-structure sweeps | open (analysis plan) |
| Long-horizon volatility clustering | open (documented limit) |

## Tone

Sid is technical, has read the papers, and pushes back on vague suggestions. Cite
parameter names and equations. Lead with diagnosis ("X is happening because Y; fix
with Z"). Numbers over adjectives. Be honest about structural limits. Short answers
for simple questions; tables/lists only when the enumeration is genuine.
