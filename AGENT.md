# AGENT.md — working briefing + change history

Briefing for any developer or AI assistant working on this project. **`model.md` is the
in-depth technical reference** (agents, equations, the clearing tier, parameters, calibration,
results) — read it for any model detail. `README.md` is the short orientation guide. This file
covers *how to work here*: current status, conventions, the repo map, gotchas, and the full
change history (at the bottom). Read this snapshot and the relevant code before suggesting
anything.

## Status & handoff (read first)

**Where the project is.** Both layers are built; the project is in its analysis/writing phase.
The market layer is calibrated and LOCKED; the clearing layer is built end to end and
ledger-audited (total system cash conserved to float precision through L4 cascades, in both
tiered and direct modes). The committed clearing model is the **D74 escrow model with the
balance-sheet rebalance** (see "Committed model state" below and the D74 change-log entry). The
headline experiments — H1 (tiered vs direct), H2 (margin regimes), and NET (gross vs net client
margining) — have been run; the kept harnesses are `run_thesis_experiments.py` (canonical) and
`gbm_lever.py` (GBM ensemble + close-out / leverage stress arms). **The committed
`output/thesis_final/experiments/` run is current** (40 seeds/arm, both regimes, regenerated
2026-06-21 under the rebalance, 15% house margin, and IM floor/cap 4%/12%; per-arm tables +
caveats in `agent_context/EXPERIMENT_RESULTS.md`) — treat any stored number as stale; the live
headline numbers are in this file's "Headline results" and in `THESIS_SYNTHESIS.md`.

### Committed model state (source of truth)

- **Population** (unchanged; calibration is bare-market so unaffected): 30 FT + 20 MT + 40 ZI
  clients; 10 BCM (5 with client books) + 5 NBCM + 1 CCP.
- **Client balance sheets:** FT $0.5–3B, MT $0.2–1B, ZI $0.2–0.5B. Member capital BCM $5–10B,
  NBCM **$0.5–3B** (raised from $50M–1B — the substantial non-bank clearers; see the floor scheme
  below). (Old client sizes — FT $100–500M, MT $30–150M, ZI $60–150M — SUPERSEDED.)
- **BCM proprietary cap** `POSITION_LIMIT_X=2.0` (`|own| ≤ 2·cash`, flat gross-leverage, no vol
  feedback). **Was 6 — now 2.**
- **House margin** `im_percent=0.15` → client leverage ≈6.67× (post-2020 security-futures
  statutory minimum). **Was 20% (5×) — now 15%.** At 15% the stressed leverage cycle is materially
  more active.
- **Client→clearer assignment:** leverage-balanced / capacity-proportional (largest clients →
  highest-capital CMs; `run_simulation.py`). **Replaces the old round-robin "9 per CM" / "concentrate
  the biggest books on the smallest NBCMs" rule.**
- **IM:** physical escrow (`IM_ESCROW=True`), procyclical VaR by **filtered historical simulation**
  (`IM_CONF_Z=3.0` = the empirical 99% quantile of vol-standardised ES returns, NOT the Gaussian 2.326
  which understates the fat tail ~1/3; **MPOR=1** (CME/CFTC 1-day ETD-futures horizon, D77); floor 4%,
  **cap dropped** (`IM_CAP=None` — non-binding at the 1-day MPOR)). The vol estimator is now a
  **DAILY-frequency close-to-close RiskMetrics EWMA** (`IM_DAILY=True`, λ=0.94), **WARMED on real
  pre-window daily history** (`data/processed/ES_front_daily_1d.csv`) → genuine recent vol at session 0
  (no cold start) + native overnight-gap coverage. This SUPERSEDES the old intraday-EWMA (`IM_VOL_HALFLIFE`
  11 RTH days) + separate overnight-gap EWMA (`IM_INCLUDE_GAPS`), which are retained only as the fallback
  if the daily series is unavailable. Procyclicality: IM ~4% floor pre-crash → ~11% crash peak (2.8×,
  uncapped at 1-day MPOR; ES IM moved ~3.9%→~10% of notional in Mar-2020). Recomputed hourly;
  VM hourly; DF cover-2 daily (`DF_STRESS_Z=3.9`, kept
  above IM so SLOIM>0). `VOLUME_LOT` **18/32** (matched to empirical RTH front-month ~2,000/~3,500 ES
  contracts/min).
- **Solvency floors (unified FCM client-clearing + bank Basel-LR — `DIFFERENTIATED_FLOORS=True`,
  `UNIFIED_CLIENT_FLOOR=True`, `BASEL_LR_BCM=True`).** Both tiers' **client-clearing solvency** is the
  same FCM rule, **CFTC Reg 1.17** `cash/IM ≥ 8%` (`REG117_FLOOR_NBCM`) on the client book →
  freeze/default. A **BCM is additionally a bank**: held to the **Basel III / US-eSLR leverage ratio**
  `cash/exposure(own+client) ≥ LR_FLOOR_BCM = 4.25%`, which triggers its own-book **deleverage**
  toward `DELEVERAGE_TARGET_BCM=6%` (the leverage cycle, Haynes-McPhail-Zhu); its house book is also
  bounded by the `2×`-cash cap. NBCMs (non-banks) get only Reg 1.17 → stop-out. Clients freeze at
  `cash/exposure ≤ 4%`. **Why this shape (D75b, see `FLOOR_BALANCE_RESULTS.md`):** the earlier
  *single* Basel-LR-on-everything BCM floor + tiny `$50M–1B` NBCMs made the **small NBCMs the only tier
  that ever defaulted** — a *capital-size* artifact (de-confounding the floor alone did not change it).
  Fix = (i) measure client-clearing solvency the same way for both (Reg 1.17), (ii) raise NBCM capital
  to `$0.5–3B` (`NBCM_CASH_RANGE`, the substantial non-bank clearers — Marex/ABN/Clear Street), and
  (iii) keep the Basel LR as the **bank deleverage trigger** so the leverage cycle survives the
  unification. Verified: calm clean, COVID benign (2 client defaults), IM share ~73%, the BCM Basel LR
  fires under deep stress (cash/exposure → 0.03, banks deleverage), defaults graded across tiers.
- **Close-out:** member / direct-client book `CLOSEOUT_MODE="transfer"` @ `CLOSEOUT_RECOVERY=0.80`
  (Lehman-class). **C1 FIX:** transfer now **reassigns** the defaulter's net book to the surviving
  participant with the largest opposing position (`_transfer_book`; `_ccp_warehouse` fallback),
  booked as a fill at mid — so inventory and system cash are **conserved** (the old flatten-without-
  transfer leaked −$14.8B / −7,427 lots in the direct arm). Client book `CLIENT_CLOSEOUT="firesale"`
  (member assumes + Almgren-Chriss). **Stress arms, OFF:** open-market CCP disposal
  (`CLOSEOUT_MODE="firesale"`), recovery 0.60 (Aas / disorderly). Waterfall: defaulter IM (L0) → L1
  defaulter DF → L2 SITG → L3 pooled DF → L4 survivor → L5 CCP. Porting EMIR Art. 48.
- **Calibration:** bare-market, `N_DAYS=75`, calm (2019) / stressed (COVID 2020); MCR + Franke
  J-test; θ locked, unaffected by the clearing changes.
- **Experiment window:** stressed sessions 10–30; calm first 20.

**Locked baseline θ** — **PLACEHOLDER, pending the widened-bounds 7-parameter re-lock.** The
calibrated loop is now the same 7-d set in both regimes (`{ft_sigma_c, zi_alpha, zi_delta, zi_mu,
p_zi, mt_lambda, mt_gamma}`), but `globals.CALIBRATED` still holds the OLD optima below (calm 4-d
with no `p_zi`/`mt_lambda`/`mt_gamma` value; stressed 5-d); these point values are stale and must be
re-locked. The displayed numbers (`globals.CALIBRATED`; D67 V_t-form relock grid headline;
`output/relock/`): calm `ft_sigma_c=0.65, zi_alpha=0.38, zi_delta=0.02, zi_mu=0.08125` (D_grid 43.94);
stressed `ft_sigma_c=0.25, zi_alpha=0.32, zi_delta=0.02, p_zi=0.18, zi_mu=0.05833` (D_grid 5.49). D
is comparable only at a fixed seed count (KS standardiser is sim-sized — so the
grid D at 3 seeds is NOT comparable to the surrogate D at 6 seeds). On the fresh-seed re-rank the
grid argmin is the top node within ≤1 D in both regimes (fresh-seed winner differs only in a
weakly-identified param: calm ft_sigma_c 0.50, stressed zi_mu 0.104). Only `zi_alpha` is sharply
identified; `ft_sigma_c` is flat in calm / on the 0.25 box floor in stressed. The SV-MJD
robustness θ is in `output/relock_gbm/` (same ZI rates, ft_sigma_c=0.45, p_zi=0.08). The
behavioural θ is bare-market, so the clearing rebalance does NOT change it.

**Headline results** (committed config — sandbox single-/few-seed checks; **regenerate the full
ensembles before quoting in the thesis**):
> **⚠ D77 (2026-06-21): margin config changed — MPOR 2→1, IM cap dropped (`IM_CAP=None`).** Every IM
> figure below is from the MPOR=2 / 12%-cap run (`output/thesis_final/experiments`) and is **STALE pending
> a re-run**. Expected effect: stressed IM ≈0.71× (floor unaffected), H2 band 4%→~11.4% (2.8×, no cap), and
> likely a modest rise in client defaults (less collateral). Re-run `scripts/run_thesis_experiments.py`.
- **System margin split:** clients post **≈73–75%** of system IM (toward the CME ≈82% reality).
- **Calm: clean.** Every clearer stays above its floor; no defaults, no deleverage.
- **COVID is benign — the accurate-floor result.** Under the Option-A floors the COVID window gives
  only a handful of client defaults (≈7–12, localised), **no member defaults, no mutualisation
  (waterfall 0)** — matching the historical record (March 2020 was a margin/funding event, not a
  member-default cascade). The old "~37 defaults / leverage cycle fires at COVID" was an artifact of
  the inaccurate 8% floor (it sat just above the endogenous κ_min ~0.04–0.06 and *triggered* the
  cascade; see the floor sweep in `FLOOR_AND_MARGIN_ANALYSIS.md`).
- **H1 (tiered vs direct) — tiering raises the mutualisation threshold.** On the COVID-stressed path
  (40 seeds) the tier **localises**: tiered draws the mutualised fund in **0/40 seeds** (client defaults
  absorbed by member capital), while **direct reaches L≥3 in 80% of seeds** (mean depth 2.6/5, ~$94M CCP
  cash + ~$1.6B mutualised). Report as waterfall-depth / DF-draw frequency vs drawdown, not a single
  count, and **disclose the IM confound** (tiered holds ~3.2× the IM: $32.6B vs $10.3B). (All conserved
  after the C1 fix.)
- **H2 (margin regime) — procyclicality is a funding channel.** The daily close-to-close IM fraction runs
  the **4% floor → 12% cap (3.0×; cap binds ~38% of stressed sessions)** — in line with CME ES IM
  (~3.9%→~10% of notional, Mar-2020) — and escrow makes the spike a real funding drain. Defaults trade
  off against level: flat-5% 4.1/seed, reactive 1.3, flat-12% 0.65; reactive's calm cost is ~42% of
  flat-12's. (Collateral-demand / leverage-cycle, *not* crash amplification — the price path is exogenous.)
- **NET — netting helps twice.** Net (vs gross) client margining cuts posted client IM to **≈0.58×**
  (≈42% saving: ≈$19.0B vs $32.6B stressed, ≈$8.3B vs $14.8B calm) and modestly reduces contagion.
- **Recovery 0.60 / open-market disposal (secondary arms):** deepen the draw modestly; the structural
  levers (direct clearing, deeper stress) dominate. **Anchors:** Lehman = high recovery / within IM;
  Aas / Bell & Holden 2018 = DF drawn.

**Open decisions.** (a) **Re-run the full suite under the committed config** (unified Reg-1.17 client
floor + bank Basel-LR trigger + $0.5–3B NBCM, C1 close-out fix, FHS+gap margin, `VOLUME_LOT` 18/32)
and **report H1 as a waterfall-depth-vs-drawdown curve** (tiered vs direct) plus the H2
collateral-demand series — the existing `output/thesis_final/experiments` is stale. Expand the GBM
ensemble (≥5 agent seeds × ≥20 paths; the per-seed `data/fv_gbm_stressed_s*.csv` need regenerating via
`data/v_gbm.py`). (b) **Write the thesis results / discussion / conclusion chapters**
(intro/model/calibration drafts in `extras/tex_drafts/`). *(Docs synced to the committed floor scheme
2026-06-19: `model.md`, `clearing_writeups.tex` §A/§C, `README`, `THESIS_SYNTHESIS`, topology.)*

**This session's reviews/changes (2026-06-18, D75)** are written up in `SECOND_READER_REVIEW.md`
(full audit), `FLOOR_AND_MARGIN_ANALYSIS.md` (the floor sweep), `OPTION_A_RESULTS.md`,
`VOL_ESTIMATOR_AUDIT.md`, and `VM_AND_VOLUME_NOTES.md`. Reproducers: `audit_conservation.py`,
`floor_sweep.py`, `optA_test.py`, `vol_audit.py`.

**Reproduce.** (run from repo root) `./scripts/run_relock.sh` re-runs the locked calibration;
`python3 scripts/run_thesis_experiments.py` re-runs the H1/H2/NET ensemble; `python3 scripts/gbm_lever.py`
runs the GBM ensemble + close-out / leverage stress arms; `python3 scripts/verify_rebalance.py` checks the
calm-clean / stress-contagion split and the house-margin sensitivity. `./scripts/run_thesis_final.sh`
chains the whole pipeline. (`scripts/*.py` self-bootstrap the repo root onto `sys.path`.)

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
client-carrying CMs (5 client-carrying BCMs + 5 NBCMs); the assignment is now
**leverage-balanced / capacity-proportional** (largest clients → highest-capital CMs —
see "Committed model state"), NOT the old flat "9 per CM" / smallest-NBCM concentration.
FT clients are 30 (not 20) so FT-equiv = 40, preserving the pre-doubling FT-equiv:MT:ZI =
40:20:40 mix at 2× scale. The behavioural θ has since been **re-calibrated and locked at
thesis-final resolution (D67 — see Status & handoff)**.

**Fundamental V_t.** Primary: the contemporaneous empirical 1-min ES BBO mid —
NOT Kalman-smoothed. The local-level Kalman MLE (`data/v_kalman.py`) is kept ONLY
as the diagnostic that justifies using the raw mid (σ_ε ≈ 1e-5 → steady-state gain
≈ 1 → the RTS smoother is near-identity, moving the level < 0.04 index pts, so it is
NOT applied; the `fv_*.csv` `V_smooth` column now equals the mid exactly). Real local
volatility `σ_t` (EWMA of r², 30-min half-life) is piped into the FT belief width.
DEFERRED robustness arm: SV-MJD (`data/v_gbm.py`). Data inputs reduce to 1-min BBO +
1-min OHLCV + daily OHLCV (MBP-10 / L2 order-book data is no longer required — see
Calibration; `data/p_zi.py` is legacy/optional).

**Calibration.** Behavioural loop = **7 parameters per regime, the SAME set in BOTH
regimes**: `{ft_sigma_c, zi_alpha, zi_delta, zi_mu, p_zi, mt_lambda, mt_gamma}`.
`zi_mu` is calibrated (the old C6 "freely droppable" finding is stale post-D58);
`p_zi` is now CALIBRATED in BOTH regimes (previously calm was fixed at the MBP-10
L2-MLE value 0.543 — that book-data dependency is removed); `mt_lambda` (MT EWMA
decay) is now CALIBRATED (was pinned 0.05) and `mt_gamma` (MT tanh-activation scale,
new) is calibrated. `ft_alpha = mt_alpha = 1.0` pinned. **Calibrated point values are
placeholders pending a widened-bounds 7-parameter surrogate re-lock** — `globals.CALIBRATED`
still holds the old 4-d (calm) / 5-d (stressed) optima, and calm has no
`p_zi`/`mt_lambda`/`mt_gamma` value wired yet. Two methods, reported together:
surrogate-assisted SMM
(`calibrate.py run`) and exhaustive grid search (`calibrate.py grid`). Loss is a
5-component standardised-moment distance: **KS + V + ACF1 + ACF2 + Hill** (ACF1/ACF2
forward-3-lag-smoothed at lags {1, 5, 10, 20}), Franke–Westerhoff inverse-bootstrap-SD
weights. Kurtosis is diagnostic only. Run both at thesis resolution via
`./run_calibration_all.sh` (deletes the stale LHS cache, runs grid + surrogate for both
regimes, records every moment, merges both regimes into one JSON).

**Clearing.** Built end to end. USD variation-margin cycle every 60 min (`× VOLUME_LOT
× CONTRACT_USD`; per-regime `VOLUME_LOT = 18 / 32`, `CONTRACT_USD = 50`); live client
novation; **physical IM escrow** (IM moved as cash to the CCP, seized first on default);
**client-clearing solvency** `cash/IM ≥ 8%` (CFTC Reg 1.17, both tiers) plus a **bank-only Basel-LR
trigger** `cash/exposure ≥ 4.25%`; a **static own-book position limit** `|own| ≤ 2·cash`
(`POSITION_LIMIT_X=2`, no vol feedback); a BCM breaching the Basel LR **deleverages its own book
toward a 6% buffer then freezes**, an NBCM **stops out**; cover-2 default fund; five-level waterfall. **Close-out is split:** a defaulted
tiered client is liquidated open-market by its member (Almgren-Chriss, `CLIENT_CLOSEOUT="firesale"`);
a defaulted member / direct client is closed out by the CCP at 0.80 recovery
(`CLOSEOUT_MODE="transfer"`) with the haircut mutualised. All ODD / regulatory constants live in
`globals.CCP_CALIBRATION` (+ the escrow/limit toggles `IM_ESCROW`, `POSITION_LIMIT_X`,
`CLOSEOUT_MODE`, `CLIENT_CLOSEOUT` at module scope). At the 15% house margin: calm is clean (no
defaults), the stressed crash fires the leverage cycle through the bank-CMs' client books, and the
member tier localises the loss (the H1 result — see "Headline results"). (See the D74 change-log
entry + the balance-sheet rebalance.)

**Client clearing (D52 — the contagion channel; escrow update D74; rebalance — see "Committed
model state").** Each client (FT/MT/ZI) carries its own balance sheet, settles its own VM in
cash and (under `IM_ESCROW`) posts its own procyclical IM through its CM, and is position-bounded
by the `im_percent=0.15` house margin (IM ≤ cash ⇒ exposure ≤ ~6.67× cash; was 5× at the old 20%).
A client that cannot fund a VM or IM call from its own cash **defaults on liquidity** — on which
its CM seizes the client's posted IM first, covers the uncollected VM, and **liquidates the
position open-market via Almgren-Chriss** (`CLIENT_CLOSEOUT="firesale"`), draining CM cash and
potentially toppling the CM (client → CM → CCP cascade). Client cash is type-matched to realistic
asset-manager scale (FT $0.5–3B > MT CTA $0.2–1B > ZI noise $0.2–0.5B) and assigned to clearers
by the leverage-balanced rule (largest clients → highest-capital CMs). At the 15% margin the
stressed crash pushes the bank-CMs through the leverage floor via their CLIENT books (the
leverage-cycle / contagion channel); calm stays clean.

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
# ROOT — kept deliberately minimal (2026-06-19 cleanup): 3 docs + 3 notebooks + folders
README.md                  short orientation guide
model.md                   in-depth model + calibration + results reference
AGENT.md                   this file — working briefing + change history
calibration_figures.ipynb  calibration-section figures (stylised facts, ACFs, moment table)
results_figures.ipynb      results-section figures (H1 waterfall-vs-drawdown, H2 IM, NET, IM share)
clearing_topology.ipynb    tiered-clearing network map
model/   globals.py agents.py simulation.py clearing.py lob.py  run_simulation.py (entry point + builder)
scripts/ calibrate.py validate_grid_fresh.py run_thesis_experiments.py gbm_lever.py verify_rebalance.py
         analyze_volume_margin.py analysis_long_run.py  run_relock.sh run_thesis_final.sh (pipelines)
data/    data.py roll.py v_kalman.py v_gbm.py p_zi.py impact.py  processed/  fv_*.csv
agent_context/  review/analysis docs, prompts, .tex drafts, scratch scripts, superseded notebooks
output/  relock/ relock_gbm/ (locked calibration) + thesis_final/ (experiments)
extras/  archived tex drafts, finished campaign scripts, reference PDFs, old output
```
Run everything from the repo root. The `scripts/*.py` self-bootstrap the repo root onto `sys.path`
(so `python3 scripts/gbm_lever.py` works directly); code imports the entry point as
`from model.run_simulation import build_traders, build_clearing_tier`. The notebooks run from root.

## FILE INVENTORY — canonical vs scratch

What is load-bearing, what is a kept harness, and what is safe to delete. (See "Open decisions"
for what still needs re-running.)

**CANONICAL MODEL** — the model itself; do not break:
- `model/` — `globals.py`, `agents.py`, `simulation.py`, `clearing.py`, `lob.py`,
  `run_simulation.py` (entry point + population/clearing builder). `model/__init__.py` makes it a package.

**CALIBRATION:**
- `scripts/calibrate.py`, `scripts/validate_grid_fresh.py`, `data/v_kalman.py`, `data/v_gbm.py`

**EXPERIMENTS (kept harnesses — all in `scripts/`, self-bootstrap the repo root onto `sys.path`):**
- `scripts/run_thesis_experiments.py` — canonical H1/H2/NET driver.
- `scripts/gbm_lever.py` — close-out & leverage stress levers + GBM ensemble (arms: baseline / direct /
  flat05 / flat03 / rec06 / rec06_dir / lowrec05 / ccp_fs / ccp_fs_dir / combo / flat05_dir).
- `scripts/verify_rebalance.py` — calm/stress rebalance verification + `HOUSE_MARGIN` sensitivity.
- `scripts/analyze_volume_margin.py` — volume + margin-by-type figures.
- `scripts/analysis_long_run.py` — long-horizon (1-min/daily) moment validation.
- `scripts/run_relock.sh`, `scripts/run_thesis_final.sh` — pipeline drivers (run `./scripts/run_relock.sh` from root).

**FIGURES (canonical — root notebooks, figures inline, no separate .png saved):**
- `calibration_figures.ipynb`, `results_figures.ipynb`, `clearing_topology.ipynb`

**DOCS (live):**
- Root: `AGENT.md`, `model.md`, `README.md` (the only three root docs).
- `agent_context/` — `THESIS_SYNTHESIS.md`, `CALIBRATION_VALIDATION.md`, the review/analysis docs
  (`SECOND_READER_REVIEW.md`, `FLOOR_AND_MARGIN_ANALYSIS.md`, `FLOOR_BALANCE_RESULTS.md`,
  `OPTION_A_RESULTS.md`, `VOL_ESTIMATOR_AUDIT.md`, `VM_AND_VOLUME_NOTES.md`), the agent prompts
  (`NEW_AGENT_PROMPT.md`, `CLEARING_METHODOLOGY_PROMPT.md`), the `.tex` write-up drafts, and the
  historical snapshots (`OVERNIGHT_REVIEW.md`, `DESIGN_REVIEW.md`, `CLEARING_LAYER_REVIEW.md`).

**SCRATCH (in `agent_context/` — regenerable reproducers + superseded artefacts):**
- Reproducers: `audit_conservation.py`, `floor_sweep.py`, `floor_balance.py`, `optA_test.py`,
  `vol_audit.py`, `calib_figures.py`, `results_figures.py`, `clearing_topology.py` (the notebooks supersede these).
- Superseded: `covid_contagion.py`, `run_experiments.py`, `plot_experiments.py` (PRE-rebalance
  experiment path — superseded by `scripts/run_thesis_experiments.py` / `scripts/gbm_lever.py`);
  `model.ipynb`, `results.ipynb` (old notebooks); `clearing_topology.png`.
- `data/fv_gbm_stressed_s*.csv` (regenerate via `data/v_gbm.py`).

Note: `scripts/run_overnight_experiments.sh` and `scripts/run_recalib_vt.sh` are older campaign
drivers kept for reference; the canonical pipeline is `run_relock.sh` -> `run_thesis_final.sh`.

## Calibrated state + re-copy

`globals.CALIBRATED` is the source `model/run_simulation.py` and the notebooks read. The
**grid optimum** (`output/calibrated_params_grid.json`) is the wired headline; the
surrogate (`output/calibrated_params.json`, `theta_stage2`) is reported as a
cross-check. The two agree on the identified levers (`ft_sigma_c`, `p_zi`) but differ
on the weakly-identified ZI rates (`zi_alpha`, `zi_delta`) at near-flat loss — so the
grid is the defensible headline. Re-copy the grid optimum into `CALIBRATED` after each run.

## Calibration pipeline

Offline (one-shot): `data/v_kalman.py generate-all` (writes the raw 1-min mid as V_t —
the Kalman MLE is the justifying diagnostic only, no smoothing applied); `data/v_gbm.py
calibrate` + `generate-all 42` (SV-MJD, DEFERRED robustness arm). `data/p_zi.py` is now
legacy/optional — `p_zi` is calibrated in the loop in both regimes, so the MBP-10 geometric
depth is no longer a calibration input.

Agent loop (`scripts/calibrate.py`, run from repo root):
- `python3 scripts/calibrate.py run [calm|stressed] [N_LHS N_DAYS N_RUNS N_REFINE N_PER_REFINE N_STAGE2]` — surrogate-assisted SMM.
- `python3 scripts/calibrate.py grid [calm|stressed] [N_PER_DIM N_DAYS N_RUNS]` — exhaustive grid search (Gao 2023).
- `FV_GBM=1` prefix → either command runs on the SV-MJD fundamental (D63: own day
  boundaries, separate `_gbm` LHS cache, calm ft_sigma_c box widened to (0.5, 2.0)).
- `python3 scripts/validate_grid_fresh.py <regime>` — fresh-seed re-rank of the top grid nodes.
- The stressed sim auto-caps to its data window (the full 2020-02-17..2020-05-28 window, 73 sessions,
  28,866 1-min obs — the old ~29-day crash-only window is no longer used); calm subsamples to N_DAYS.
- **Cache:** delete `output/calibration_lhs_*.csv` when the param set or population
  changes (the guard catches param/moment-name changes; a structural change at the
  same param set is NOT caught — delete manually). The calm LHS cache regenerates
  automatically after the `zi_mu`-drop param-set change.

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
2. **Deep mutualisation (L4 survivor assessments) is reached only under the stress arms.** Under
   the committed config the stressed crash DOES fire the leverage cycle (~18 client defaults per
   seed, ≈37 summed over the 2-seed check, at the 15% margin; the bank-CMs breach through their client books). But in the TIERED arm the loss
   is localised — the member balance sheet absorbs it and the pooled/mutualised layer is rarely
   reached (the H1 result: tiered draws the mutualised DF in 0/6 GBM paths, direct in 4/6). Deeper
   mutualisation (recovery 0.60 / open-market disposal) is the `gbm_lever` stress arm; L4 is
   partly STRUCTURAL to cover-2 sizing (the fund is sized to cover the two largest, so a single
   member's default is normally absorbed by L1–L3 — realistic; L4 assessments are rare in practice).
   The reverse-stress / recovery-0.60 arms are the lever for breaching deeper.
3. **Population scaling is calibration-coupled — but θ is currently LOCKED.** The FT/MT/ZI clients
   are also the price-formation population, so changing the CLIENT COUNT would force a
   re-calibration. The current rebalance changed only client CASH (balance sheets), not counts,
   and calibration is bare-market, so `globals.CALIBRATED` is NOT stale (do not re-run calibration
   for the rebalance). Beyond headcount, default/contagion power comes from Monte-Carlo over seeds
   (outcome distributions) and designed sweeps (margin methodology, tiered-vs-direct, close-out /
   recovery arms). NBCMs are off-LOB, so the non-banking tier scales without recalibrating; the
   LOB traders do not.

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
6. **MT is limit-only; FT trades every step** (`ft_alpha = 1`, replace-on-new). **MT now
   activates STOCHASTICALLY**: `P(trade) = tanh(|M_t| / (mt_gamma·σ_v))`, so the SHARE of MTs
   acting scales with trend strength (replaces the old every-step / hard sign-only gate; the
   EWMA momentum `M_t` still updates every step regardless of activation). Don't re-add the
   MT market-order branch.
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
| Cash-prefunded DF + fill-settled VM + porting + CCP disposal (D61) | done |
| H1 direct-clearing mode + H2 margin regimes (D62) | done (levers built) |
| SV-MJD scenario generator re-grounded + overnight gaps (D63) | done |
| Market-side completion run — lock validated, gbm θ recorded (D64) | done |
| Dead-code removal, behaviour-neutral (D64) | done |
| Physical IM escrow + net-position ratio + static position cap + split close-out (D74) | done |
| Balance-sheet rebalance + leverage-balanced assignment + 15% house margin | done (committed) |
| Re-run H1/H2/NET ensembles under the committed config | open (Open decision (a)) |
| Fold gbm_lever stress arms into run_thesis_experiments.py | open (Open decision (a)) |
| Sync model.md / clearing_writeups.tex to the committed state | open (Open decision (b)) |
| Thesis results / discussion / conclusion chapters | open (Open decision (c)) |
| Long-horizon volatility clustering | open (documented limit) |

## Tone

Sid is technical, has read the papers, and pushes back on vague suggestions. Cite
parameter names and equations. Lead with diagnosis ("X is happening because Y; fix
with Z"). Numbers over adjectives. Be honest about structural limits. Short answers
for simple questions; tables/lists only when the enumeration is genuine.

## Change history & effects

The decision log, most-recent first. Each entry is *what changed* and *the effect
it had*. Older micro-iterations are condensed; full per-step history is in git.

### Recent architecture (current model)

- **D77 — MPOR 2→1 day + IM cap dropped (margin realism).** Two `model/globals.py` changes, motivated by
  ES/CME disclosures (Clarus, FIA, BIS Bulletin 13) and the EMIR APC framework. **(1) `IM_MPOR_DAYS` 2→1.**
  1-day is CME/CFTC's actual liquidation horizon for liquid exchange-traded futures (SPAN = 99% VaR over
  1 day); the 2-day was the EMIR OTC-derivative minimum, an over-conservative deviation for an ES product.
  This rescales the VaR by 1/√2; `df_stress_move` picks up the same MPOR so the SLOIM stays consistent
  (`DF_STRESS_Z 3.9 > IM_CONF_Z 3.0` ⇒ SLOIM≥0 regardless of MPOR). **(2) `IM_CAP` 0.12→`None` (dropped).**
  At the 1-day MPOR the reactive FHS VaR self-bounds at **~11.4% peak** on the COVID path (0/73 stressed
  sessions exceed 12%), so the old 12% cap never bound — dropping it lets the VaR express its full
  **4%→~11.4% range (2.8×, no flat-top artifact)** while staying within the realised CME ES band (~4%→~8-10%
  of notional, Mar-2020). Real CCPs don't cap IM (EMIR Art. 28 APC tools are floors/buffers, not caps).
  `im_fraction` applies a cap only when `IM_CAP is not None`, so the mechanism is preserved — **if the
  deferred reverse-stress (c>1) amplifier is ever reactivated, set `IM_CAP` back to a fraction (e.g. 1.0)
  to restore the physical IM≤notional bound**, since uncapped VaR can exceed notional at extreme c. Verified:
  band reproduces (calm 4% flat; stressed 4%→11.37%/2.84×, 0 cap-binds), `CALIBRATED` untouched (bare-market
  θ), sim runs clean. **Consequence: `output/thesis_final/experiments` is now STALE** — every IM dollar
  figure rescales ≈0.71× in stress (floor-bound calm ≈unchanged), the H2 band restates to 4%→~11.4%/2.8×,
  and client defaults likely rise modestly (less collateral); **re-run `scripts/run_thesis_experiments.py`**
  and re-derive before quoting. Docs synced (config): `model.md` §4/§8.1, `RESULTS_METHODOLOGY.md`, this file;
  result docs banner-flagged stale (`EXPERIMENT_RESULTS.md`, `RESULTS_PLAN.md`, `THESIS_SYNTHESIS.md`).
  `clearing_writeups.tex` IM formula still shows the old floor/MPOR (`.tex`, flagged in `WRITING_FIXES.md`).

- **D76 — IM realism retune + headline experiments re-run + figure/notebook redesigns.**
  (1) **IM band retuned to ES reality:** `IM_FLOOR` 0.06→**0.04**, `IM_CAP` 0.30→**0.12**,
  `DF_STRESS_FLOOR` 0.10→**0.08** (FHS `IM_CONF_Z=3.0` kept). The old 6%/30% band let the reactive
  fraction peg the 30% cap (~5× cap-saturated, unrealistic); the 4%/12% band gives a **3.0× crash
  rise (4%→12%)** with the cap binding ~38% of stressed sessions — in line with CME ES IM (~3.9%→~10%
  of notional, ~1.9–2.6×, Mar-2020). (2) **Headline experiments regenerated** (40 seeds/arm, both
  regimes) → `output/thesis_final/experiments/`; full analysis in `agent_context/EXPERIMENT_RESULTS.md`.
  Numbers: **H1** tiered draws the mutualised fund in 0/40 vs **direct 80% reach L≥3** (disclose the IM
  confound: tiered $32.6B vs direct $10.3B); **H2** defaults flat05 4.1 / reactive 1.3 / flat12 0.65,
  reactive calm cost ~42% of flat12; **NET** posted client IM **0.58×** gross (was mis-stated as
  "halves"/0.81×). (3) **Found `static` ≡ `reactive`** (no-op: with `IM_DAILY=True` the daily-σ branch
  in `simulation.py` precedes the `IM_MODE` check; `flat` arms unaffected) — fix or drop the static arm;
  and H2 is a collateral-demand/leverage-cycle result (exogenous price), not crash amplification.
  (4) **Figures:** `calibration_figures.ipynb` — tail decay redesigned as a **Hill tail-index plot**,
  ACF x-axis to integer minute lags, price path → 2×2 (calm+stressed, full window + intraday zoom), and
  the **clearing network added inline**; `clearing_topology.ipynb`/`.png` redesigned (radial, members
  alternating BCMc/BCMo/NBCM around a large CCP hub, clients on long straight spokes, node area ∝ capital).
  Also hardened the daily-IM tz reindex and fixed `wire_lock.py` to wire all 7 params.

- **D75 — second-reader audit: Option-A floors, C1 close-out fix, FHS+close-to-close margin,
  VOLUME_LOT re-match (2026-06-18).** A full critical audit (see `SECOND_READER_REVIEW.md`) plus five
  committed changes. **(1) Type-differentiated solvency floors (Option A, `DIFFERENTIATED_FLOORS=True`).**
  The old uniform `cash/exposure ≥ 8%` floor mis-labelled the Basel leverage ratio (real min 3%,
  G-SIB/eSLR ~3.5–4.25%) and, sitting just above the endogenous stress κ_min (~0.04–0.06), was the
  *load-bearing trigger* of the COVID contagion (floor sweep: at 0.04 the COVID window is benign, at
  0.08 it cascades — `FLOOR_AND_MARGIN_ANALYSIS.md`). Now: BCM `cash/exposure ≥ LR_FLOOR_BCM=4.25%`
  (Basel/eSLR, deleverage to `DELEVERAGE_TARGET_BCM=6%`); NBCM `cash/IM ≥ REG117_FLOOR_NBCM=8%` (CFTC
  Reg 1.17 — non-banks aren't Basel-bound; liquidity defaulters); clients freeze at
  `CLIENT_FREEZE_FLOOR=4%`. Result: **COVID is benign** (a handful of localised client defaults, 0
  member defaults, no mutualisation) — matching March-2020 reality; contagion is conditional on
  deeper stress. **(2) C1 close-out conservation fix.** `transfer` mode used to flatten a defaulted
  book without reassigning it, breaking VM zero-sum (measured −$14.8B / −7,427 lots leak in the direct
  arm). It now `_transfer_book`s the net book to the largest-opposing survivor (booked at mid;
  `_ccp_warehouse` fallback) — inventory + cash conserved (direct-arm COVID leak → −$0.30B = the
  genuine haircut). The bug had *inflated* the direct contagion. **(3) FHS initial margin.**
  `IM_CONF_Z` 2.326 → **3.0** (the empirical 99% quantile of vol-standardised ES returns; filtered
  historical simulation, the standard CCP method — the Gaussian understated the fat tail ~1/3);
  `DF_STRESS_Z` 3.0 → **3.9** (kept above IM so SLOIM>0); `IM_VOL_HALFLIFE` 2 → **11 RTH days**
  (RiskMetrics λ≈0.94 — 2d was ~5× too reactive). **(4) Close-to-close margin** (`IM_INCLUDE_GAPS=True`):
  a separate overnight-gap EWMA is added to the daily variance, so margin covers the gap it marks
  across (~45% of stressed daily variance). Effect: stressed IM ≈10→14%, procyclicality ~1.5×→**~2×**
  — matching CME's actual ~2× ES-margin hike in March 2020. **(5) VOLUME_LOT 30/60 → 18/32**
  (`VM_AND_VOLUME_NOTES.md`): the old map overshot empirical volume ~1.3–1.6× (agent flow rose with
  the D67 relock); re-matched to the RTH front-month ~2,000/~3,500 contracts/min. Behaviour-neutral
  (position caps are notional-based, so exposure/IM/share are VOLUME_LOT-invariant — verified). **VM
  design audited and confirmed correct** (`pos·(mid−last_mark) + last_mark·fill_qty − fill_cost` is
  exact M2M with average-fill-price). **Still flagged, not changed:** the 15% house-margin cites the
  security-futures rule, but ES is a broad-based index future (SPAN ~5–7%) — re-cite as an FCM house
  add-on; and the whole model is uncommitted to git (HEAD = D60).

- **D74 — physical IM escrow + net-position capital ratio + static position limit + close-out
  split (2026-06-18).** The committed clearing model. Verified against globals.py / agents.py /
  simulation.py / clearing.py. **(1) Physical IM escrow (`IM_ESCROW=True`).** Initial margin
  now moves as real cash to the CCP (`ccp.im_account`) each cycle: members post on their OWN
  book, clients post their OWN procyclical IM routed through their member (`_post_im`). A
  participant that cannot fund the call **defaults on LIQUIDITY**. On default the defaulter's
  posted IM is **seized first** (`_seize_im`, before the waterfall); excess collateral over the
  realised loss returns to the defaulter's estate (cash-conserving, not mutualised — EMIR Art.
  48). **(2) Net-position capital ratio.** `κ = cash/exposure ≥ 8%` (cash already net of posted
  IM), the Basel leverage ratio (`haynes_mcphail_zhu_2019`, `acosta_smith_2018`). Under escrow the procyclicality channel runs through
  the cash NUMERATOR (a funding squeeze), and leverage self-bounds at `cash/(floor+im_frac)`
  (~7× calm, ~2.6× stress). `capital_ratio()` returns `cash/exposure` under `IM_ESCROW=True`.
  **(3) Static position limit.** `|own notional| ≤ 2·cash`
  (`POSITION_LIMIT_X=2`, `POSITION_LIMIT_CLIENTS_ONLY=False` → all BCMs), a flat gross-leverage
  limit with NO volatility feedback. The sweep showed an uncapped own-account BCM balloons to ~7× and blows the
  waterfall to L4 (a mutualised artifact), hence `CLIENTS_ONLY=False`. **(4) Deleverage buffer
  retired under escrow.** A BCM breaching the floor now **freezes** like an NBCM — no
  Almgren-Chriss capital-ratio deleverage. A **deliberate deviation from the ODD's
  BCM-fire-sale / NBCM-stop asymmetry** (leverage is already self-bounded by escrow);
  `CAP_DELEVERAGE_TARGET` survives only for the legacy non-escrow path. **(5) Close-out split.**
  A defaulted tiered CLIENT is liquidated OPEN-MARKET by its MEMBER via Almgren-Chriss
  (`CLIENT_CLOSEOUT="firesale"` — the client-level price-impact channel); a defaulted MEMBER
  (and a direct client) is resolved by the CCP at 0.80 recovery (`CLOSEOUT_MODE="transfer"`).
  **Bug fixes verified this pass:** (i) `start_firesale` now `extend`s the liquidation queue
  (`_liq_slices.extend(ac_slices(...))`) so clustered/repeated close-outs accumulate rather than
  overwrite; (ii) `_seize_im` returns the EXCESS posted IM over the loss to the defaulter's
  estate (`agent.cash += im - used`) so system cash stays conserved; (iii) `_port_clients`
  receiver-capacity headroom is computed under escrow as `cash/floor − exposure` (the
  net-position form), not the legacy `cash/(floor·im_frac) − exposure`. **Doc overclaim fixed:**
  the member transfer does NOT select a surviving counterparty — the code sets the defaulter's
  book flat and mutualises the (1−recovery) haircut (a simplification of the ODD PositionAuction).
  **Citation fixes:** every "Vuillemey 2023" → **Vuillemey 2020** ("The Value of Central
  Clearing," JF 75(4):2021-2053; bib key should be renamed `vuillemey_2023` → `vuillemey_2020`);
  the FCM-capital scale ($5-10B bank / $50m-1B non-bank) cites `cftc_fcm_data` (CFTC "Financial
  Data for FCMs"), NOT `fia_tracker` (which is CCP-level). Docs: model.md §2.2/§3.1/§3.3/§4/§8.1
  updated. (Results numbers in §6 left as-is — a separate escrow-model experiment run is in
  progress; NB the §6.2 "lowers members' cash/IM ratio" prose pre-dates the net-position ratio.)

- **D73 — CCP own capital lowered $7.5B -> $1.5B (realism; tail-reachable).** `CCP_CASH` is the CCP's
  OWN cash (the SITG source + the Level-5 backstop), not the total prefunded resources (~$9-10B = SITG +
  member fund). $7.5B was too large for a major CCP's own equity (~$1-2B). Lowered to $1.5B: realistic,
  and the L5 backstop is now reachable under extreme stress (so CCP insolvency can appear in the deferred
  reverse-stress / fire-sale hypotheses), while INERT at actual severity — verified behaviour-neutral
  (calm SITG unchanged at $29.4M = 3% of the fund; the SITG cap min(3%*fund, cash) does not bind above
  ~$50M; L5 not reached). Docs: globals comment + model.md loss-absorbing-cash line. (No effect on the
  in-flight thesis-final run, which loaded the old value and is inert to it.)

- **D72 — thesis-final overnight pipeline (`run_thesis_final.sh`).** One detached command:
  pre-flight (deps + live model state) -> RELOCK (`./run_relock.sh`: grid + fresh-seed + surrogate,
  both regimes -> output/relock/) -> VERIFY the relock grid optima still match globals.CALIBRATED
  (warns + tells you to re-copy + re-run steps 4-5 if they drift; θ is stable since the market layer
  is unchanged post-D67, so they should match) -> MCR (`calibrate.py mcr`, Franke/HFABM coverage on
  the locked θ -> output/mcr_{regime}.json) -> EXPERIMENTS (`run_thesis_experiments.py`). All under
  output/thesis_final/. ~13-18h. New driver `run_thesis_experiments.py`: H1 tiered-vs-direct, H2 margin
  regimes (reactive/flat05/flat12/static + a DF_DECOUPLE_IM arm), and gross-vs-net client margining,
  on calm + the COVID stressed window at ACTUAL severity (reverse-stress amplifier and AC fire-sale
  stay OFF — deferred hypotheses). Verified in-sandbox (scipy-free): H1 shows tiered L0/contained vs
  direct L2/mutualised at actual COVID; H2 shows the calm collateral cost (flat-12 ~$29B vs reactive
  ~$14B, ~2x). Launch: `nohup ./run_thesis_final.sh > output/thesis_final_console.log 2>&1 &`
  (env N_SEEDS default 40, MCR_M default 80). SV-MJD robustness relock is a separate night
  (`FV_GBM=1 OUT_TAG=relock_gbm ./run_relock.sh`).

- **D71 — clearing realism: deleverage buffer, client-IM first-loss, client-margin netting toggle.**
  Three changes (verified: calm + COVID c=1 run; closeout_loss now $0 since client IM covers the haircut).
  **(1) Deleverage to a buffer, not the floor.** A solvent BCM breaching the 8% capital floor now
  Almgren-Chriss-deleverages back to `CAP_DELEVERAGE_TARGET=0.12` (1.5x the floor) instead of to the
  floor itself, so it doesn't immediately re-breach next tick (a management buffer; raise toward 0.15 for
  a wider cushion). `simulation.py` deleverage target. **(2) Client IM is first-loss for the close-out
  (SLOIM at the client level).** On a client default the close-out haircut is netted against the client's
  posted house IM (`im_percent`=0.20 of notional): CM bears only `max(0, (1-recovery)*notional - IM)`.
  With recovery 0.80 (20% haircut) = im_percent (20%) the IM exactly covers it, so a client default within
  IM costs the CM nothing beyond the VM shortfall (the realistic role of client margin; the CM/DF only bear
  losses BEYOND client IM). `im_posted` added to client_history. This makes "clients post their own IM"
  explicit; note the model already had client cash as first-loss (position-capped at IM<=cash), so this is
  the close-out-cushion refinement, not a leverage re-tuning. **(3) Client-margin netting toggle**
  (`CLIENT_MARGIN_NETTING`, default "gross"): the CM's IM/capital ratio uses gross client notional
  (Sum|client_pos|, US/CME gross customer margining) by default; set "net" for EU net-omnibus
  (|Sum client_pos|) — offsetting client positions cancel, lowering the CM's IM and mitigating
  capital-ratio stop-out (the Duffie-Zhu 2011 netting benefit / H1 tiering value; the AGENT.md-flagged
  net-omnibus lever). VM and the cover-2 DF already use the NET book. Clean gross-vs-net comparison axis.
  NB: clients are NOT default-fund contributors (deliberate — clients' insulation from mutualisation is
  the tiering value H1 studies; the CM funds the DF, sized on its own+client net book, and recoups via
  fees). Docs: model.md §3.1/§3.3 + §4 table updated.

- **D70 — calibration validation: Moment Coverage Ratio reporting + P0 consistency fixes.** Added the
  Franke-Westerhoff (2012) / HFABM §4.3 **Moment Coverage Ratio** to `calibrate.py` (the p-value/J-test
  was dropped to keep the chapter short; MCR is the more positive, more concrete statistic — the prototype
  showed stressed ~87% mean per-moment coverage). `empirical_moment_sd(..., return_samples=True)` now
  optionally returns the B x K bootstrap matrix; new `moment_coverage_ratio(regime, theta, ...)` builds the
  per-moment 95% CI = m_emp +/- 1.96*s_i over the K=10 point moments (drops ks_stat + kurtosis), runs
  M_long model runs, and reports per-moment MCR, joint MCR, the bootstrap self-coverage ceiling, and the
  0.95^K benchmark -> `output/mcr_{regime}.json`. New CLI verb `python calibrate.py mcr [regime] [M_long]
  [n_days]` (post-hoc, re-runnable on the locked theta; run after a calibration). Validation-only — does
  NOT touch the loss. **P0 fixes:** (i) `validate_grid_fresh.py` now prefers `output/relock/grid_surface_*.csv`
  (the top-level CSV is stashed/restored by run_relock, so standalone it was a stale older-param-set grid
  -> KeyError on zi_mu); (ii) synced the stale locked-theta in model.md §5.1 and this Status block to the
  live D67 lock (calm 0.65/0.38, D 43.94; stressed 0.25/0.32, D 5.49); (iii) documented that grid D
  (3 seeds) and surrogate D (6 seeds) are NOT comparable (sim-sized s_KS) and that only zi_alpha is sharply
  identified. Full validation write-up: CALIBRATION_VALIDATION.md. The final seed-count-matched run +
  re-lock (and the MCR table for the thesis) is the deferred night run.

- **D69 — default close-out switched to transfer-at-recovery (AC fire-sale + reverse-stress deferred).**
  Per Sid: the open-market Almgren-Chriss disposal produced unrealistic price impact, so the DEFAULT
  close-out is now the ODD-style **transfer** (`globals.CLOSEOUT_MODE="transfer"`, in simulation.py
  `_margin_cycle` both default sites): a defaulted client's position is assigned to its CM's
  largest-offsetting client, and a defaulted CM's residual book (after EMIR-48 porting) to the surviving
  member with the largest opposing position, at **`CLOSEOUT_RECOVERY=0.80`** — the realised loss is the
  (1−recovery) haircut, borne by the CM (client default) or added to the mutualised deficit (member
  default); no LOB market orders, no CCP disposal account, no endogenous price impact. **Recovery 0.80**
  is a conservative stressed value: real auctions recover ~par on a hedged book (LCH used ~35% of Lehman's
  IM, no DF, in 2008 — BIS Quarterly Review Dec-2018; IOSCO PD657), worse on concentrated/unhedged books
  (Nasdaq 2018); 0.80 is below Lehman, above the ODD's credit-style 0.60; sensitivity-test it. The
  **open-market AC fire-sale** (the price-impact contagion channel) and the **reverse-stress c-amplifier**
  are KEPT as off-by-default toggles for a later **additional hypothesis** (`CLOSEOUT_MODE="firesale"`;
  run covid_contagion at c=1 for calm/stressed-actual). A solvent BCM's capital-ratio deleverage still
  uses AC (only the default close-out is toggled). Verified: transfer default runs calm + COVID c=1
  (7 client defaults, 0 CM, L0, CCP solvent — the realistic mild picture); firesale toggle still runs.
  Docs: model.md §3.3 rewritten, §4 table + §8.1 clearing element→source map added (model.md is now the
  single methodology reference); topology rebuilt as a cash-scaled network map (clearing_topology.*).

- **D68 — clearing-layer realism pass (IM cap, DF stress cap, SITG, levers; from CLEARING_LAYER_REVIEW.md).**
  Acted on the clearing-layer review (six-investigation synthesis in CLEARING_LAYER_REVIEW.md).
  Changes, all in globals.py unless noted: **(1) IM cap** `IM_CAP=0.30` — `im_fraction` now
  `min(IM_CAP, max(IM_FLOOR, var))`, removing the unphysical IM>notional artifact at high reverse-stress
  c (the reactive EWMA could drive IM>100% as the open book shrank). **(1b) companion DF stress cap**
  `DF_STRESS_CAP=0.35` on `df_stress_move` — necessary because capping IM but not the stress move would
  balloon the SLOIM coefficient `(df_stress_move - im_frac)` and the fund (DF -> ~$16B at 4x COVID,
  vs real CME ~$9B). With both caps the DF stays realistic (~$0.6-1B over the COVID window) and DF/IM
  in stress lands ~3-5% (the real CME/LCH band); **as a clean side effect the right-sized fund is now
  breached to L3 (pooled-DF mutualisation) at c~4 — deeper mutualisation is reached via the realistic
  fund + reverse-stress, the intended lever, not a hack.** **(2) SITG** `ex_df_ratio` 0.10 -> 0.03 and
  relabelled: real SITG is ~1-4% of the fund (CME ES $100M/$9.4B ~2.7%; Clarus disclosures), and EMIR
  Art 45 is 25% of the CCP's CAPITAL not 10% of the fund — the prior 0.10 conflated the two. CCP_CASH
  comment relabelled as TOTAL prefunded resources (L5 backstop), not SITG. (verified: own_df/total_df
  = 0.030.) **(3) DF_DECOUPLE_IM** (default False) — H2 robustness lever in `clearing.recompute_default_fund`:
  sizes the SLOIM on a fixed reference IM (`IM_FLOOR`) so the fund does not shrink as the live IM rises,
  isolating procyclicality from the SLOIM-IM coupling (model.md §6.2). Run both arms for H2. **L4
  dormancy is partly STRUCTURAL to cover-2 sizing** (the fund is sized to cover the 2 largest, so a single member's
  default is normally absorbed by L1-L3 — realistic; L4 assessments are rare in practice). The reverse-stress
  c is the lever for breaching deeper. CALIBRATED untouched (behavioural θ unaffected). Added
  clearing_topology.ipynb + .png (the tiered-network map). Docs: model.md §3/§4/§7 updated; sources table
  in CLEARING_LAYER_REVIEW.md §8. NB calm DF/IM stays ~high (the DF_STRESS_FLOOR 15% vs IM_FLOOR 6%
  mismatch) — a disclosed structural item, not changed (15% is a defensible extreme-but-plausible floor).

- **D67 — V_t-form relock landed + Kalman smoother dropped (V_t = empirical mid).** Two things:
  (1) **Relock copied into CALIBRATED.** The D66 V_t-form grid relock (output/relock/, seed 42,
  new form confirmed: same seed but calm argmin moved and stressed D changed) was reviewed and the
  GRID optima copied into globals.CALIBRATED: calm `ft_sigma_c 0.65, zi_alpha 0.38, zi_delta 0.02,
  zi_mu 0.08125` (D_grid 43.94); stressed UNCHANGED `0.25/0.32/0.02/p_zi 0.18/0.05833` (D_grid 5.49,
  was 5.76 — fits slightly better under the new form). Caveats recorded in the globals comment:
  calm ft_sigma_c is unidentified (D flat ~43.9-44.5 over 0.5-0.95; grid 0.65 / fresh-seed 0.50 /
  surrogate 1.09; zi_alpha ~0.38 is the identified calm lever) and the fresh-seed re-rank puts the
  grid argmin at #3 in BOTH regimes (calm prefers ft_sigma_c 0.50, stressed prefers zi_mu 0.104 —
  all within ~1 D). Stressed ft_sigma_c sits on the 0.25 box floor again (the coarse 4-pt grid can't
  resolve the interior ~0.29). (2) **Kalman smoother dropped.** `data/v_kalman.py generate()` now
  uses the raw per-day log mid as V_t directly instead of the RTS-smoothed series — the MLE finds
  sigma_eps ~ 1e-5 on the 1-min mid so the smoother was near-identity (max |smoothed - mid| < 0.04
  pts). The Kalman MLE / `calibrate` is retained as the diagnostic that JUSTIFIES using the mid.
  fv CSVs regenerated: `V_smooth` column now equals the mid exactly; **sigma_t is byte-identical**
  (it was always EWMA of the raw mid returns, never the smoothed series), so the change is
  behaviour-neutral and CALIBRATED is left as-is. sigma_t half-life stays 30 min (deferred; a longer
  half-life raises calm long-horizon clustering but needs recalibration — see D66/the half-life
  sweep). Docs updated (model.md §2.3 heading "Empirical efficient mid"; v_kalman.py module +
  generate docstrings). **Full recalibration deferred** to the next relock (do `validate_grid_fresh`
  + a finer stressed grid then).

- **D66 — FT reservation made scale-invariant (v0 -> V_t, multiplicative form).** The FT
  reservation is now `R = V_t * (1 + z * ft_sigma_c * sigma_t)` (was `V_t + z * ft_sigma_c *
  sigma_t * v0`). The belief offset `z * ft_sigma_c * sigma_t` is now a *fractional* dispersion
  around the current fundamental, so the price-formation law no longer carries the regime-initial
  `v0` level constant (`v0` is still used for clearing marks). Algebraically this is the "vt"
  variant from the belief-width experiment, which was moment-equivalent to the old form: the
  level factor `V_t/v0` is 1 in calm (V_t ~ v0) and only deviates in the stressed drawdown, where
  the test found Hill identical (2.95) and all stylised-fact moments within seed noise, so the
  locked theta carries over. Doc clarifications landed with it: `sigma_t` is the EWMA realised vol
  of `V_t` (EWMA of r^2, 30-min half-life, reset per session) — an empirical input, not a fitted
  or mean-reverting model; and the stale `sigma_fund_t = sqrt(390)*sigma_t*v0` docstrings
  (globals.py, v_kalman.py) and the "channel through which clustering reaches the mid" comment were
  corrected (the belief-width experiment showed the adaptive sigma_t width is a tail lever that
  *damps* clustering, which enters via the V_t path + MT, not a clustering-transmission channel).
  `params.sigma_fundamental` (globals.__post_init__) is now vestigial — computed for inspection,
  read nowhere in the live path. Re-validate the lock with a fresh-seed grid re-rank
  (`validate_grid_fresh.py`) under the new form before the final lock; the finer stressed grid
  should use the new form.

- **D65 — loop recomposition + RE-LOCK setup (run via `./run_relock.sh`; supersedes the D60
  lock once landed — the D60 record stays in `output/baseline_grid/`).** Common-seed probes
  (sandbox, 2-run config) showed the D60 stressed floors were BINDING, contradicting the
  "benign boundary-sitting" disclosure: ft_sigma_c has an interior optimum ≈0.40 (D 7.64 vs
  9.97 at the 0.5 floor), zi_delta improves to 0.02 (7.82) with the book still bounded and
  driftless (778 vs 517 lots — the D58 0.05 guard was conservative), p_zi improves to 0.10
  (8.29). And `zi_mu` is NO LONGER droppable post-D58 (the C6 evidence is stale): calm has an
  interior optimum ≈0.10 (−3.3 D; Hill 2.71→2.94), stressed −1.0 at 0.05. **New loop:** calm
  4-d {ft_sigma_c, zi_alpha, zi_delta, zi_mu}, stressed 5-d {+ p_zi}; evidence-tightened boxes
  (see calibrate.py); calm p_zi STAYS at the measured L2 pin (measured-beats-fitted; its ~2 D
  cost — 47.4 at 0.65 vs 49.8 — is disclosed). **Also decided with evidence:** (i) `mt_eps` is
  a zero-guard, not a threshold — the minimum nonzero |M| (3.8e-6) exceeds it; never binds
  post-warm-up; keep. (ii) NO 60-min TTL: it would cull only 4.6% of resting ZI orders while
  re-censoring zi_delta's identification (the exact D58 bug-class); the one real stale-order
  case is fixed surgically instead — **D65b cancel-on-freeze** (a newly frozen client's
  standing quotes are cancelled; c=1 trace unchanged). (iii) An MT-specific placement
  parameter is NOT identified (MT-only p 0.35/0.75 moves D by ±0.4 — seed noise); the shared
  geometric placement stands. (iv) `mt_lambda` STAYS pinned (Majewski external horizon): E2's
  ~1 D stressed gain does not justify a 6th stressed grid dimension (4^6 = 4096 nodes).
  (v) FT placement stays AT the reservation (the value-trader semantics — a depth offset
  would break the Chiarella reservation-price design). (vi) The SV-MJD process parameters
  stay MEASURED (D63 realized measures), never SMM-calibrated — fitting the fundamental to
  fix the agents would inject the circularity the design avoids; the gbm agent-θ re-lock with
  the SAME D65 loop runs via `FV_GBM=1 OUT_TAG=relock_gbm ./run_relock.sh`.

- **D64 — completion-run results + dead-code removal (behaviour-neutral).**
  **Run results** (`output/baseline_grid/grid_freshseed_*.json`, `output/baseline_gbm/`):
  (i) **D60 lock VALIDATED** — fresh-common-seed re-ranking of the top-5 grid nodes keeps the
  locked θ at #1 in both regimes: calm 0.80/0.34/0.05 (D_fresh 46.66 vs 48.11 runner-up),
  stressed 0.50/0.26/0.05/0.15 (D_fresh 7.15, BELOW its in-sample 7.92 — no winner's-curse
  inflation). (ii) **SV-MJD stressed optimum = the Kalman stressed optimum exactly**
  (0.50/0.26/0.05/p_zi 0.15; D_fresh 11.44 vs Kalman 7.15 — fits worse, as C1 predicted, but
  the θ is fundamental-invariant: a robustness headline for the thesis). Surrogate agrees
  (stage-2 0.596/0.258/0.073/0.152, R²=0.95). (iii) **SV-MJD calm is near-flat**: grid argmin
  1.25/0.26/0.125 (D 35.91), fresh-seed winner 0.50/0.50/0.05 (39.22 vs 39.57 — inside noise),
  surrogate 1.56/0.23/0.05 (R²=0.72, weak) — three corners of the box within ~0.4 D. Use the
  fresh-seed winner for calm gbm ensembles and DISCLOSE the flatness (consistent with the known
  weak calm identification of ft_sigma_c, amplified on the lighter-tailed synthetic path).
  **Cleanup executed** (all RNG-stream-neutral; verified byte-identical seed-42 covid traces
  at c=1.0/2.5 tiered + c=2.0 direct, before vs after): deleted MarketMaker / VolatilityTrader /
  ContTrader classes + builder branches; ModelParams slimmed (n_mm/n_vt/n_ct, mm_qty/mm_p_edge/
  mm_skew/mm_pov/mm_inventory_*, vt_*/ct_*, mt_lambda_long/n_momentum_long, depth_mean/
  depth_sigma all gone); MM_QTY + AC_LAMBDA + agents.ac_schedule + calibrate._report_r2 + the
  unused scipy-minimize import + the deprecated df_percent/recovery_rate CCP keys deleted;
  MT_LAMBDA_LONG / N_MOMENTUM_LONG env paths retired (E5's FTMT_GATES kept — headline negative
  result). zi_mu's pin re-cited to the ODD §Calibration baseline (CST-structured). agents.py
  666→~410 lines. Callers updated: run_simulation, covid_contagion, analysis_long_run,
  calibrate (POP/CLEARING_IN_LOOP), build_nb (+ notebook regenerated).

- **D63 — SV-MJD fundamental re-grounded on ES realized measures + empirical overnight gaps
  (`data/v_gbm.py` rewritten).** Every FV parameter is now MEASURED from the ES data (drops the
  ODD's FTSE-based `λ=3/day` pin — deviation flagged): diffusion `σ_d` from daily BIPOWER
  variation (jump-robust; Barndorff-Nielsen & Shephard 2004); jump intensity from MANCINI (2009)
  threshold counts (`|r| > 4·σ_loc`: λ=1.93/day calm, 0.97/day stressed); jump size from the
  RV−BV jump variance (share 6.8%/3.9% of RV — on the Huang-Tauchen 2005 ~7% S&P benchmark; the
  old kurtosis matching assigned 23%/30%, the documented Hill-too-heavy bias); OU vol params from
  a jump-robust bipower σ̂ series with EIV-ROBUST moments (e^{−αΔ}=γ₂/γ₁, Var=γ₁²/γ₂ — kills the
  ~13% window-sampling attenuation; vol half-life ≈2h calm / 4.3h stressed); drift = the mean
  intraday log-return used DIRECTLY (the old `−½σ_t²` Itô subtraction double-counted, −1.7% over
  120 stressed days). **Overnight gaps added**: a nonparametric bootstrap draw from the regime's
  EMPIRICAL overnight-return pool at every session boundary (256 calm / 28 COVID gaps; ~40% of
  close-to-close variance; the COVID crash was overnight while intraday netted positive —
  Lou-Polk-Skouras 2019). Stressed episodes default to the empirical 29-day window (120 days of
  compounded COVID gaps gave a meaningless −79% path). Validation: intraday and overnight
  std/mean match empirical in both regimes; 30-episode ensemble maxDD mean −28% (p10 −45%, p90
  −9%) brackets the real COVID −33%. Path kurtosis is BELOW empirical by design — the path
  carries only the measured jump/SV tail share; the agents earn the rest. Generator default
  output renamed `fv_gbm_{regime}.csv` (the old default OVERWROTE the Kalman `fv_*.csv`).
  Calibration support: `FV_GBM=1` env flag in `calibrate.py` (own fv paths/day-boundaries,
  separate `_gbm` LHS cache, stressed subsampled like calm, calm `ft_sigma_c` cap relaxed to the
  stressed (0.5, 2.0) — the (0.5, 1.1) cap was Kalman-path evidence); `validate_grid_fresh.py`
  (fresh-common-seed re-ranking of the top-5 grid nodes — the winner's-curse check, also run on
  the LOCKED Kalman grid); `run_recalib_gbm.sh` (full pipeline, stashes/restores the D60 record).
  **Also recorded (Kalman review):** at 1-min on the MID the local-level MLE yields σ_ε≈1e-5 →
  steady-state gain 0.999–1.000 — the Kalman "smoothing" is near-identity and V_t is effectively
  the observed efficient mid (the mid has no MA(1) bounce to remove; visible smoothing would
  need an indefensible σ_ε ≈ 1.3% of price, or the references' STRUCTURAL state extraction).
  Frame honestly in the thesis; do not claim material noise removal.

- **D62 — H1/H2 experiment levers.** (a) **Margin regimes** (`globals.IM_MODE`): "reactive"
  (NEW DEFAULT — VaR on an EWMA of realised intraday sim returns, half-life 2 RTH days, gaps
  excluded from the estimator; IM ratchets up through a crash as CME's ES margin did in Mar-2020),
  "static" (pre-D62 regime-constant σ_v), "flat" (`IM_FLAT_FRAC`, no vol response — H2
  comparators 0.05 flat-low vs 0.12 through-the-cycle; EMIR Art. 28 APC alternatives /
  BCBS-CPMI-IOSCO 2022). The same σ drives the DF stress sizing. (b) **Direct-clearing mode**
  (`build_clearing_tier(..., direct=True)`, CLI `covid_contagion.py reverse direct`): no client
  tier — no NBCMs, BCMs own-account only, all 90 clients registered as CCP participants
  (cash-funded DF contributions, in the waterfall) with IDENTICAL trading mechanics to the tiered
  run (same house cap, 8%/notional freeze, default at 0), so tiered-vs-direct isolates the
  loss-absorption STRUCTURE (Duffie-Zhu 2011; Galbiati-Soramäki 2013). First paired result
  (c=2.5, seed 42, same path, 25 client defaults in both): **direct reaches L3 mutualisation,
  tiered stays L0** — the CM buffer absorbs what direct clearing socialises; and at c=1 direct
  has 0 client defaults vs tiered 3-4 — the tiered casualties die FROZEN by NBCM stop-outs
  (intermediation transmits distress through operational freezes). (c) `FV_GBM` plumbing in
  `calibrate.py` (see D63).

- **D61 — clearing-tier default management rebuilt to the ODD (+ EMIR Art. 48 porting).**
  Five changes, ODD-grounded (the in-repo `Simudyne_CCPRiskModel_ODDDoc.md` is the reference;
  its L4 is plain pro-rata — the public web ODD's per-survivor cap was NOT adopted):
  (i) **cash-prefunded default fund** — contributions move member cash into a CCP-held `df_cash`
  pool at every daily recompute (top-up = replenishment cash call, the ESRB margin-liquidity
  channel); L1/L3 draws disburse it (pre-D61 the DF was a free bookkeeping buffer — L3
  "mutualisation" cost survivors nothing). (ii) **fill-price VM settlement** (ODD §Margin Call:
  P/L = N·(P_market − P_filled)) — `_fill_qty/_fill_cost` accumulate per agent and the margin
  cycle settles `pos·Δmid + Σq·(mark−px)`; fire-sale slippage is REAL now (pre-D61 a book
  AC-sold within the hour realised ≈0 close-out loss, gutting the D54 claim). (iii) **defaulted
  CM removed** (ODD §13): stops trading, deleverage slices cleared, own+unported book to the CCP;
  surviving clients **PORTED** to client-carrying CMs with spare cash/IM headroom above the 8%
  floor (EMIR Art. 48(5)-(6); greedy by capacity; unported clients closed out + frozen — porting
  in stress is not guaranteed, OFR 2026). (iv) **CCP disposal account**: the CCP is a
  fill-receiving agent (`traders_by_id`); disposal P&L (mark + slippage) RESUMES the originating
  waterfall on losses (symmetric with the client close-out), gains accrue to the CCP; synthetic
  "CCP" clearing_history rows record the levels. (v) **state-based NBCM stop-out** — held while
  cash/IM ≤ 8%, clients frozen via the Phase-1 `cm._stopped` check, lifted on recovery (pre-D61
  the freeze lasted ≤1 cycle: Phase 1 overwrote it). Verified: inventory conservation exactly 0
  through forced multi-CM cascades; calm day unchanged. **Effects at seeds 42/43:** COVID
  containment holds (3 client defaults, L0, DF $0.50B); reverse-stress onset moves ≈2.5× → ≈4×
  COVID (−56%, 3 NBCM defaults, L3) — main driver is (v): persistently-frozen clients stop
  bleeding, so NBCMs survive longer; marginal defaulters now MT (fill-settlement realises
  trend-followers' buy-high slippage) instead of ZI.

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
