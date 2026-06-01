# AGENT.md — AI-agent collaboration briefing

> Briefing for any AI coding assistant joining work on this thesis. This file
> covers *how to work in this repo*. The design spec is `claudereadme.md` —
> read its `Current state` header first.

## 0 · Project at a glance

MSc thesis ABM of a CCP-cleared single-asset (ES front-month futures) market,
pure Python (no Simudyne SDK). Two tiers:

1. **Market microstructure layer** (LOB + traders + exogenous V_t).
2. **Central-clearing tier** (CCP + BCM / NBCM + balance sheets + margin).

**Roster (54 LOB agents + 5 NBCM + 1 CCP off-LOB):**
- 10 FundamentalTrader + 10 BankingClearingMember (FT-cast, own-account)
- 10 MomentumTrader (single-cohort EWMA)
- 20 ZeroIntelligenceTrader (3 Bernoulli rates)
- 4 MarketMaker (HFABM mid-anchored, calibrated spread width)
- 5 NonBankingClearingMember (pure clearing intermediary, no LOB orders)
- 1 CentralCounterparty (registry + cash + DF; 5-level waterfall scaffolded)
- 5 of the 10 BCMs carry client books (plain FT + MT round-robin); rest are
  own-account only. ZI stay direct exchange participants.

**V_t (D33):** Stein-Stein-style stochastic-vol Merton jump-diffusion. σ_t
Vasicek-OU; V_t evolves with stochastic σ_t · Z + Merton jumps. `data/v_gbm.py`
calibrates `(σ_d, λ, δ)` from the variance-kurtosis decomposition (D25) and
`(α, θ, σ_vol)` from 30-min realised-vol time series. Re-anchored θ so
`E[σ_t²] = σ_d²`. σ_t piped into the FT belief width via `SimContext` (D34).

**Calibration (`calibrate.py`):** 6-d θ per regime — `depth_mean`,
`depth_sigma`, `zi_alpha`, `zi_mu`, `zi_delta`, `mm_p_edge`. HFABM-style
surrogate-assisted SMM: LHS → XGBoost surrogate → active learning → multi-start
L-BFGS-B → stage-2 true-simulator grid. **4-component loss** (Hill + V + ACF1
+ ACF2) with Franke-Westerhoff inverse-bootstrap-variance weights.
- ACF1 (return ACF) lags `{1, 5, 10}` — centred 3-lag average each.
- ACF2 (|return| ACF) lags `{1, 2, 5, 10, 15, 30}` (D45 short-lag refocus).
- `ret_kurtosis` is diagnostic-only — Hill is the robust tail measure.

**Pinned (NOT in the loop):** `ft_alpha=mt_alpha=1.0` (D36 — every step),
`mt_lambda=0.05` (D44), `ft_sigma_c=√390` (D34), `qty_max=10` (D47),
`mm_qty=2`, `mm_skew=0.05`, `mt_mu=0` (D40 — MT limit-only), `ft_sigma_c=√390`.

**Reported volume & capital ratios (D50):** Each model qty unit represents
a **50-contract institutional ES block**. `VOLUME_LOT=50`, `CONTRACT_USD=50`
(CME E-mini ES spec). Applied at the reporting + capital-ratio layer only —
LOB dynamics, log-returns, ACFs, Hill all invariant under this multiplicative
relabeling.

## 1 · Cardinal rule

**Read the current repo files before suggesting anything.** This project
iterates aggressively: agent types added, removed, re-added in different
forms (MM gone × removed × re-added × pinned × re-calibrated; VT trialled
× redesigned × dropped; ContTrader added × removed; MT split × folded;
ZI rates pinned × calibrated). Without grounding suggestions in the current
state, you will reinvent something tried and rejected.

`claudereadme.md` is the single source of truth — its `Current state` header
at top is the consolidated snapshot; the historical D-log explains *why*.

## 2 · Reference priority chain

Every design decision must cite one of these. Higher-ranked refs win on conflict.

1. **Simudyne CCP Risk Model ODD** (project's primary spec)
2. Deloitte CCP Risk Model webinar + CCP resilience white paper (clearing-tier
   mechanics; the SV-V_t prompt came from the WEBINAR — white paper has no
   equations; ODD incomplete on the fundamental. Formal OU-on-σ = Stein-Stein 1991)
3. Majewski, Ciliberti, Bouchaud (2018) — Extended Chiarella estimation
4. Gao et al. (2023) — Chiarella-Heston / deep hedging
5. Vytelingum et al. (2025) — ABM Liquidity Risk (was mis-cited "Krishnen")
6. Cont, Stoikov & Talreja (2008) — Stochastic LOB
7. Farmer, Patelli & Zovko (2004) / Daniels — ZI baseline

Supporting (cited mechanism justifications):
- Stein & Stein (1991); Heston (1993) — stochastic-vol V_t (OU-on-σ, D33)
- Cont (2001) — stylised facts (calibration targets)
- Cont (2005) — multi-scale memory / heavy-tailed regime durations
- Gao et al. (2022) HFABM — MM spec + surrogate-SMM methodology
  (arXiv:2208.13654; was mis-cited "Cao et al. 2024". MM is their §3.6, not §3.2;
  loss is eqs 3-10; the "4-8 MMs" claim was wrong — they use 5/20)
- Franke & Westerhoff (2012) — block-bootstrap moment weights (full inverse-cov;
  we use the diagonal inverse-variance form, their §4.4 heuristic)
- Hill (1975); Resnick (2007) — tail-index estimator
- Andersen-Bollerslev-Diebold-Labys (2001, 2003) — RV / long-memory vol
- Corsi (2009) — HAR-RV
- Lamperti, Roventini, Sani (2018) — surrogate calibration *inspiration*
  (per-moment regression; they use one surrogate on an aggregate measure —
  both regression and classification — so "we regress, they classify" was loose)
- Almgren & Chriss (2000) — optimal execution (POV; scaffolded only)
- Gabaix-Plerou-Stanley (2003) — power-law trade sizes / sqrt-impact
  (mechanism tested D49 and rejected for our discrete LOB matching)

Sid keeps PDFs locally. When in doubt, ask Sid to share a specific section.

## 3 · Hard rules (Sid's preferences)

1. **No invented calibration.** Numbers come from literature, data, or
   surrogate-calibrated θ. No made-up defaults.
2. **No synthetic data.** Empirical inputs from WRDS TAQ / DataBento ES
   under `data/` and `data/processed/`.
3. **No print statements** in production code (calibration / data CLIs OK).
4. **No saved images unless asked.**
5. **No excessive comments.** One docstring per class/function with the
   intent + citation; one-line comments for non-obvious mechanics only.
6. **Snippets in chat — don't push to GitHub unless told.**
7. **`README.md` and `claudereadme.md` updated only when Sid asks.**
8. **No emojis in files.**
9. **Citations in code.** Every meaningful design choice has a citation
   comment (`# Cont-Stoikov 2008`, `# ODD §Agents`, etc.).
10. **Flag deviations explicitly.** Departure from a higher-ranked ref →
    a D-row in `claudereadme.md` with reason.
11. **Prefer simplicity.** Complexity costs defensibility. Removing a
    mechanism with weak justification is valid.

## 4 · Repo layout

```
MScThesis_ABM/
├── claudereadme.md         # internal design spec — read Current state header first
├── README.md               # thesis-facing — Sid maintains
├── AGENT.md                # this file
├── data/
│   ├── data.py             # DataBento ingest
│   ├── roll.py             # ES front-month roll + 1-min resample
│   ├── v_gbm.py            # V_t SV-MJD calibration + path generator (D33)
│   ├── p_zi.py             # MBP-10 placement-depth calibration
│   ├── impact.py           # BBO-1m Hasbrouck/AC impact (scaffolded)
│   ├── processed/          # rolled 1-min ES series (calm + stressed)
│   └── fv_{calm,stressed}.csv  # V_t paths (V_smooth + sigma_t columns)
├── model/
│   ├── globals.py          # ModelParams + SimContext + regime dicts + CALIBRATED
│   ├── lob.py              # Order, Fill, LOB
│   ├── agents.py           # BaseTrader + ZI, FT, MT, MM, BCM, NBCM
│   ├── clearing.py         # BalanceSheet + CentralCounterparty
│   └── simulation.py       # Simulation driver + _margin_cycle (D29)
├── run_simulation.py       # entry point + build_traders + build_clearing_tier
├── calibrate.py            # surrogate-assisted SMM (6-d θ per regime)
├── analysis_long_run.py    # N-day sim + 1-min/daily moment comparison
├── analysis_vt_persistence.py  # α-multiplier sweep diagnostic (V_t persistence)
└── output/
    ├── v_gbm_params.json           # V_t offline-calibration outputs
    ├── p_zi_params.json
    ├── impact_params.json
    ├── calibration_lhs_{regime}.csv     # surrogate training set
    ├── calibration_stage2_{regime}.csv  # stage-2 grid
    ├── calibrated_params.json           # validated θ + group contribs
    └── clearing_{regime}.csv            # per-cycle clearing-tier observables
```

## 5 · Calibrated state (latest)

`globals.CALIBRATED` is the source the simulator entry points
(`run_simulation.py`, `analysis_long_run.py`) read. After every calibration
copy `theta_stage2` into the dict.

**Latest — `globals.CALIBRATED` is STALE for BOTH regimes:**
- **Stressed (D48):** validated D = 21.67 — ΔHill 9.53, ΔV 0.55, ΔACF1 2.60,
  ΔACF2 8.98. Hill structurally short (1.13 vs target 3.25). See §8. **The
  validated D48 θ is in `output/calibrated_params.json` but was NEVER copied
  into `globals.CALIBRATED`** — the dict's stressed entry is the pre-D48
  placeholder. Copy-back (§5 step) still pending; note the rails (`depth_sigma`
  → 0.95, `mm_p_edge` → 1.43).
- **Calm (overnight, pre-D48 5-d):** validated D = 27.5 — ΔHill 0.66, ΔV 0.57,
  ΔACF1 7.16, ΔACF2 19.12. Never run at 6-d; absent from `calibrated_params.json`
  (the stressed-only run overwrote it — gotcha #3).

Both stale relative to the current 6-d θ; calm needs a 6-d run, stressed needs
the copy-back. Empirical-volume scale matches via the D50 reporting layer.

## 6 · Calibration pipeline

**Offline (one-shot):**
- `python3 data/v_gbm.py calibrate` — V_t SV-MJD per regime
- `python3 data/v_gbm.py generate-all 42` — V_t paths to `data/fv_*.csv`
- `python3 data/p_zi.py calibrate` — placement-depth from MBP-10

**Agent (`calibrate.py`):**
- CLI: `python3 calibrate.py run [calm|stressed] N_LHS N_DAYS N_RUNS N_REFINE N_PER_REFINE N_STAGE2`
- Lighter recipe (~1 h calm-only): `100 20 8 2 20 40`
- Full overnight: `200 30 16 3 30 80` (both regimes ~6-10 h)
- **Cache invalidation:** delete `output/calibration_lhs_*.csv` and
  `output/calibration_stage2_*.csv` whenever `PARAM_KEYS` or POP changes
  (column-mismatch otherwise).

## 7 · Stage roadmap

| Stage | Status |
|---|---|
| 1 — ZI baseline | done |
| 2 — + FT/MT (calibrated) | done |
| 3 — + Clearing tier scaffold (BCM/NBCM/CCP) | done (D28) |
| 4 — Margin cycle (VM + IM, 60-tick cadence) | wired (D29); **VM units need D51 USD fix to bite** |
| 5 — Default fund (Cover-2 EMIR) | pending |
| 6 — 5-level waterfall + position auction | pending |
| 7 — Fire-sale / Almgren-Chriss distressed liquidation | scaffolded (`mm_pov`, helpers); not wired |

## 8 · Known structural limits (document, don't try to fix without strong cite)

1. **1-min Hill tail-index gap (stressed ~9.5 SDs short).** Target 3.25,
   model 1.13. Root cause: ODD-pinned `JUMP_LAMBDA_DAY=3` plus D25 jump
   kurtosis cap `f_max=0.95` push too much variance into Merton-jump tail.
   Adding 4 MMs (D48) didn't help — bigger fills per V_t-jump sweep
   compound. **Fixes flagged but not tried:** lower `JUMP_LAMBDA_DAY` to 1,
   or cap `f ≤ 0.5` in `data/v_gbm.py` (more variance to smooth-diffusion
   + σ_t). Cite: Andersen-Bollerslev-Diebold (2007) continuous vs jump
   decomposition.

2. **Long-lag |r| ACF unreachable (lags ≥ 30 at 1-min; daily lag-1 ~0.04
   vs target 0.20).** Single OU σ_t has one timescale (1/α ≈ 80 min);
   empirical equity vol has multi-scale memory (HAR-RV; ABDL 2001/3).
   Cont (2005): long-range clustering needs heavy-tailed regime durations,
   not Markov SV. **Tried (D46):** α_mult=0.1 override — lifted daily ACF
   but optimizer found "thin-trading clustering" escape that broke V.
   Reverted; documented as structural limit citing Cont 2005.

3. **VM cycle wiring gap (D51 pending).** `_margin_cycle` line 159
   computes VM in raw model units; client_notional / capital_ratio are USD
   post-D50. One-line fix: `vm *= VOLUME_LOT * CONTRACT_USD`. Without it,
   Stage 3 "cash depletion" is silent and Stage 4 can't be wired.

4. **CM-layer roster too thin for contagion.** 1-2 clients per BCM, all
   homogeneous BCM cash `U[5B, 10B]`. Mech #2 fire-sale needs heterogeneous
   tail of small BCMs and 10-50 clients per CM (ODD §Initialization spec).
   Stage 4b: synthetic-client cohort + log-normal BCM cash.

## 9 · Common gotchas

1. **Theta-dim or POP changes break the cache.** Always wipe
   `output/calibration_lhs_*.csv` + `calibration_stage2_*.csv`.
2. **Re-copy `globals.CALIBRATED` after every calibration.** The dict is
   a manual copy of `theta_stage2` from `output/calibrated_params.json`.
3. **`calibrated_params.json` is rewritten per-regime.** Running calm-only
   wipes stressed entry. To preserve both, run sequentially or merge by hand.
4. **Per-step Bernoulli is ODD-native — no `dt` rescaling.** A `dt`
   multiplier on an arrival rate is a bug.
5. **FT trigger is `|reservation − mid|`, not `|V_t − mid|`** (D9c).
6. **LOB mid falls back to `last_price` when a side empties** (D9d).
7. **MT is limit-only.** D40 removed the market branch; `mt_mu` stays at 0.
8. **FT/MT trade every step (D36).** `ft_alpha = mt_alpha = 1.0` pinned;
   replace-on-new order management. Don't re-add a Bernoulli activation gate.
9. **MM has no inventory skew (D31).** `mm_skew` stays at 0.05 but isn't
   applied in the live `submit_orders` (HFABM Gao et al. 2022 §3.6 convention).
10. **σ_t reaches the LOB only through FT belief width.** MT sees mid
    returns; ZI sees nothing; MM sees mid. If σ_t needs to reach a new
    agent, add it explicitly via `SimContext`.
11. **VOLUME_LOT applies at reporting / capital-ratio layer only.** Don't
    multiply inside the LOB / matching engine.
12. **No xgboost / scipy in the sandbox.** `calibrate.py` is local-only.
    Sandbox smoke uses `run_simulation.py` / `analysis_long_run.py`.

## 10 · What's removed (do NOT re-add without strong cite)

- **VolatilityTrader (D38b/D44).** Coordinated σ_t-burst noise trader.
  Random market orders wash out on aggregation; vol-scaled limits damp.
- **ContTrader (D39/D44).** Cont 2005 §4.1 threshold-with-inertia trader.
  Did not improve long-lag clustering at this LOB; counter-productive in
  the simplified roster.
- **Long-horizon MomentumTrader (D43).** Two-cohort split; folded back.
- **MT market-order branch (D40).** Reverted D27; trend-direction market
  flow corrupted return ACF.
- **Pareto `_draw_qty` (D49).** Gabaix-Plerou (2003) power-law sizes;
  needs sqrt-impact (Gabaix theorem). Our discrete LOB gives linear
  impact, so power-law sizes degraded Hill. Volume scaling is now handled
  by the D50 reporting-layer multiplier.

## 11 · When Sid asks for a change

1. **Read** `claudereadme.md` Current state header, then the relevant code
   block, then `calibrate.py` if a calibrated parameter is involved.
2. **Find the citation.** Which ref justifies it? If none, flag explicitly.
3. **Propose in chat first.** Don't edit files until Sid confirms direction.
4. **Implement** with citation comments + a new D-row in `claudereadme.md`
   if it's a deviation.
5. **Tell Sid the side-effects** — cache invalidation, `CALIBRATED` re-copy,
   regenerate `fv_*.csv`, etc.

## 12 · Tone

- Sid is technical, has read the papers, pushes back on vague suggestions.
  Cite parameter names and equations.
- Lead with diagnosis. "X is happening because Y; fix it with Z."
- Numbers > adjectives. Ratios and absolute values, not "high."
- Honest about structural limits. If a moment is unreachable, say so.
- Short responses for simple questions; tables/bullets only when the
  enumeration is genuine.

---

If anything here conflicts with `claudereadme.md`, `claudereadme.md` wins
for design facts; this file wins for collaboration style.
