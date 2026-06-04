# Overnight calibration campaign — Claude Code brief

**For:** Claude Code, running autonomously overnight on this repo.
**Goal:** run a set of calibration + scenario experiments, log each result, and leave a
comparison table. Interim/screening resolution — these are *exploration* runs, not the
thesis-final calibration.

Read `AGENT.md` first (model state + change history + the ablation log). Reference priority
for any design choice: Simudyne CCP ODD → Deloitte → Majewski (extended Chiarella) →
Gao HFABM / XGB-Chiarella → Gao 2023 → ABM-Liquidity → Cont-Stoikov-Talreja → Farmer ZI.
**Cite a reference for every change; flag deviations explicitly** (several experiments below
deliberately revive designs the change-log marked as rejected — that is intended, re-test
them, but say so in the results).

---

## Ground rules (apply to every experiment)

1. **XGBoost surrogate ONLY** — always `python3 calibrate.py run …`. Never the grid.
2. **Both regimes** every time: `calm` and `stressed`.
3. **Resolution (fast — interim screens).** Use these arg sets to `run <regime> <n_lhs> <n_days> <n_runs> <n_refine> <n_per_refine> <n_stage2>`:
   - Standard (≤5 calibrated params): `48 20 2 1 12 8`
   - High-dim (≥6 calibrated params, i.e. E5): `96 20 2 1 16 12` (more LHS for more dims)
   - These land ~15–25 min/regime on the bare market layer; clearing-in-loop (E1) is ~3× that.
4. **Regenerate fundamentals once at the start:** `python3 -m data.v_kalman generate-all` (D56/D57 — real-ts, gapped). `data/fv_gbm_{regime}.csv` (synthetic) already exist.
5. **Do NOT edit `model/globals.py :: CALIBRATED`** — those are the headline values. Each experiment writes its own θ\* to its own JSON (below). Do NOT push to GitHub.
6. **Isolate experiments behind env-var flags** (spec'd per experiment). Every new behaviour defaults OFF, so the baseline is always recoverable and experiments don't contaminate each other. Work on a branch `calibration-campaign`.
7. **Results logging.** Create `output/campaign/`. After each run write
   `output/campaign/<EID>_<regime>.json` containing: `theta_stage2`, `true_loss_validated`,
   `component_deltas`, `validated` moments, `surrogate_d_accuracy`, and the exact config/flags.
   Then append one row to `output/campaign/SUMMARY.md` (table: EID | regime | D | KS | V | ACF1 | ACF2 | Hill | kurtosis | key θ | notes). `calibrate.py run` already writes most of this to `output/calibrated_params.json`; copy+rename it per experiment so runs don't overwrite each other.
8. **Time budget ≈ 8 h.** Run experiments in the numbered order (cheap→expensive). If you're running long, finish the current regime, log it, and stop — leave a `## Remaining` note in SUMMARY.md listing what didn't run. Partial completion in order is the goal.
9. After all runs, write `output/campaign/FINDINGS.md`: which configs improved the loss vs the E0 baseline, per-component, with a one-paragraph recommendation. Do not change the model's defaults — just report.

---

## Baseline reference (current model, for comparison)

Bare market layer, replace-on-new FT/MT, `mt_lambda=0.05` pinned, ZI cancellation `zi_delta`
the sole order-lifetime mechanism (D58 — the LOB TTL was removed). Population
`POP = 40 FT-equiv + 20 MT + 40 ZI` (`calibrate.py :: POP`). Calibrated loop: calm
`{ft_sigma_c, zi_alpha, zi_delta}`, stressed `{+ p_zi}`.

---

## E0 — Baseline full calibration (control)

**Goal:** the control every variant is compared against, at *this campaign's* resolution.
**Changes:** none.
**Run:**
```
python3 calibrate.py run calm     48 20 2 1 12 8
cp output/calibrated_params.json output/campaign/E0_calm.json
python3 calibrate.py run stressed 48 20 2 1 12 8
cp output/calibrated_params.json output/campaign/E0_stressed.json
```
Log both rows to SUMMARY.md.

---

## E1 — Clearing mechanics inside the calibration loop

**Goal:** calibrate with the CCP/clearing tier ACTIVE, so client freezes / deleverage /
defaults feed back into the price the moments are matched on (as they do in the final model).
**Hypothesis:** in calm, no freezes fire → θ ≈ E0; in stressed, freezes during the crash alter
the moments and shift θ.
**Flag:** `CLEARING_IN_LOOP=1`.
**Changes — `calibrate.py`:**
- `simulate_moments()` currently builds `Simulation(params, traders, seed=s)` with **no ccp**. Gate on the env var: when set, build the clearing tier and pass it. Mirror `run_simulation.py`:
```python
import os
from run_simulation import build_traders, build_clearing_tier   # build_clearing_tier already imported? add if not
...
traders = build_traders(params, seed=s)
ccp = build_clearing_tier(traders, params, seed=s) if os.environ.get("CLEARING_IN_LOOP") else None
hist = Simulation(params, traders, seed=s, ccp=ccp).run(n_steps)
```
- The clearing tier needs CM agents in the population. When `CLEARING_IN_LOOP`, switch `_theta_to_params` to the **run_simulation population** (`n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5, n_bcm_with_clients=5`) instead of `POP` (which has no CMs). Keep VOLUME_LOT/qty as-is.
**Run:** `CLEARING_IN_LOOP=1 python3 calibrate.py run calm 48 20 2 1 12 8` (→ `E1_calm.json`), same for stressed. Expect ~3× wall time.
**Compare:** θ and per-component deltas vs E0; report whether freezes changed the stressed fit.

---

## E2 — `mt_lambda` left in the calibration loop

**Goal:** let SMM choose the momentum trend horizon instead of pinning it at 0.05
(Majewski et al. 2018 fix it externally; this tests whether the data prefers a different λ).
**Flag:** `MT_LAMBDA_IN_LOOP=1`.
**Changes — `calibrate.py`:**
- Add to BOTH regimes in `PARAM_BOUNDS_BY_REGIME` (gated on the env var):
  `"mt_lambda": (0.004, 0.20)` — spans ~3.5-min (λ=0.20) to ~3-hour (λ=0.004) half-life.
- `_theta_to_params`: when `mt_lambda` is in `d`, set `kw["mt_lambda"] = float(d["mt_lambda"])`.
  `ModelParams.mt_lambda` flows to `build_traders` (`lambda_decay=params.mt_lambda`), so no agent change needed.
**Run:** `MT_LAMBDA_IN_LOOP=1 python3 calibrate.py run calm 48 20 2 1 12 8` → `E2_calm.json`; stressed likewise.
**Report:** the fitted `mt_lambda` (and its implied half-life = `ln2 / -ln(1-λ)` min) per regime, and whether the loss beats E0.

---

## E3 — `mt_lambda` fixed at a 3-hour half-life

**Goal:** the long-horizon-trend variant the user asked for.
**Value:** `mt_lambda = 0.00385` (half-life ≈ 180 min). **Flag:** `MT_LAMBDA_FIXED=0.00385`.
**Changes — `calibrate.py` `_theta_to_params`:** if `os.environ.get("MT_LAMBDA_FIXED")`, set
`kw["mt_lambda"] = float(os.environ["MT_LAMBDA_FIXED"])` (pinned, not in the loop — keep the
E0 loop dims). 
**Run:** `MT_LAMBDA_FIXED=0.00385 python3 calibrate.py run calm 48 20 2 1 12 8` → `E3_calm.json`; stressed likewise.
**Compare:** vs E0 and E2 — does a slow trend help or hurt the |return| ACF (clustering) and the tail?

---

## E4 — More / two-cohort momentum traders

The MT count and the long/short split. The two-cohort design EXISTS in code
(`n_momentum_long`, `mt_lambda_long`, `build_traders` lines 71–74) but was dropped (D43/D44) —
**flag that you are reviving a rejected design and re-testing it.** Note the FT:MT:ZI mix
changes when MT count changes; record it (and if the book depth looks off, note that
`VOLUME_LOT` may need rescaling per D69 — but do NOT rescale for these screens).

**E4a — +10 short-λ MT (30 MT total), with the long λ from E3.**
Flags: `N_MOMENTUM=30 MT_LAMBDA_FIXED=0.00385`.
`_theta_to_params` (and the E1 population, but bare here) reads `int(os.environ["N_MOMENTUM"])` for `n_momentum` when set.
Run both regimes → `E4a_calm.json`, `E4a_stressed.json`.

**E4b — 15 long-λ + 15 short-λ (two cohorts, 30 MT total).**
Flags: `N_MOMENTUM=15 N_MOMENTUM_LONG=15 MT_LAMBDA_LONG=0.00385`.
`_theta_to_params`: read `n_momentum`, `n_momentum_long`, and set `kw["mt_lambda_long"]` from `MT_LAMBDA_LONG`. Short cohort keeps `mt_lambda` (0.05 or the E2 loop value); long cohort uses `mt_lambda_long`. `build_traders` already wires both.
Run both regimes → `E4b_*.json`.
**Compare:** E0 vs E4a vs E4b — does a second slow cohort revive long-horizon volatility clustering (the known limit, AGENT.md: a single timescale can't)?

---

## E5 — Re-add FT/MT activation probability + cancellation (all calibrated)

**Goal:** give FT and MT a Bernoulli placement probability (`ft_alpha`/`mt_alpha < 1`) and a
per-resting cancellation rate (`ft_delta`/`mt_delta`), instead of pure replace-on-new every
step. **Flag the rejected design:** AGENT.md "Don't re-add a Bernoulli activation gate" (D36) —
this experiment deliberately re-tests it, now that the TTL is gone (D58) so cancellation is the
only thing removing un-refreshed orders.
**Flag:** `FTMT_GATES=1`.
**Changes — `model/globals.py`:** add fields `ft_delta: float = 0.0` and `mt_delta: float = 0.0`
to `ModelParams` (`ft_alpha`/`mt_alpha` already exist, default 1.0).
**Changes — `model/agents.py`:** in `FundamentalTrader.submit_orders`, after the
`_stopped/has_defaulted` guard and the stale-`_open_oid` drop, add the gate + stochastic cancel:
```python
# Stochastic cancellation of the standing order (D-campaign E5; CST-2008 / Farmer ZI cancel rate).
if self._open_oid is not None and params.ft_delta > 0.0 and rng.random() < params.ft_delta:
    lob.cancel(self._open_oid); self._open_oid = None
# Bernoulli activation gate (re-testing the D36-rejected gate): skip this step w.p. 1-ft_alpha.
if params.ft_alpha < 1.0 and rng.random() >= params.ft_alpha:
    return
```
In `MomentumTrader.submit_orders` add the SAME, but **after** the EWMA `_M` update (the trend
signal must keep updating even on skipped steps) and using `mt_delta`/`mt_alpha`.
**Changes — `calibrate.py`:** when `FTMT_GATES`, add to both regimes' `PARAM_BOUNDS_BY_REGIME`:
`"ft_alpha": (0.2, 1.0)`, `"mt_alpha": (0.2, 1.0)`, `"ft_delta": (0.0, 0.5)`, `"mt_delta": (0.0, 0.5)`;
`_theta_to_params` passes each through to `kw`. This is the high-dim config → use `96 20 2 1 16 12`.
**Run:** `FTMT_GATES=1 python3 calibrate.py run calm 96 20 2 1 16 12` → `E5_calm.json`; stressed likewise.
**Compare:** vs E0 — does intermittent placement + cancellation improve the return ACF / clustering, or (as D36 found) corrupt it? Report honestly.

---

## E6 — Gaps vs single big shock (clearing/contagion scenario, NOT a calibration)

**Goal:** does the cascade differ under many small overnight gaps vs one large intraday shock?
Run the **calibrated** model (use E0 stressed θ, or the best stressed config from E1–E5) through
the **clearing tier** on two fundamentals and compare the cascade. Cite Euronext A9 §5
(reverse-stress / scenario design).
**Changes — extend `covid_contagion.py`** (don't break the existing `run()`):
- Scenario A (gapped, current): the existing `data/fv_stressed.csv` path (overnight gaps retained, D56), amplified by `c` as `run()` already does.
- Scenario B (single shock): build a fundamental that is **flat at V0 then drops by one large jump** of the same total magnitude as scenario A's drawdown (e.g. one step of −25% to −33%), no overnight gaps, then flat. Run the clearing tier on it.
- Add a `scenario_run(kind, c, seed)` and a small sweep over `c` for both; record `drawdown`, `client_defaults`, `cm_defaults`, `mutualised`, `waterfall_level_reached`, `total_df_used`.
**Run:** `python3 covid_contagion.py` (extend its `__main__` to run both scenarios and print/CSV the comparison) → `output/campaign/E6_gaps_vs_shock.csv`.
**Compare:** is contagion driven by the *gap structure* (repeated discrete jumps marked across via VM) or just the *total move*? A single shock of equal size may default fewer/more clients than the gapped path — that's the finding.

---

## Suggested order (cheap → expensive, ~8 h)

E0 → E2 → E3 → E4a → E4b → E6 → E5 → E1.
(E0–E4 + E6 are bare-market or a few contagion runs; E5 is high-dim; E1 is clearing-in-loop ~3×.
This front-loads the cheap, high-value screens so a short night still yields most of the answers.)

## Acceptance / sanity checks per run
- Surrogate held-out R² should be > ~0.5 (else the screen is near-blind — note it, don't trust θ).
- Book depth must stay bounded (no runaway) — spot-check `bid_depth+ask_depth` mean over the run.
- Hill is the tail loss-target; kurtosis is a diagnostic only (do not chase it).
- If a run errors, log the traceback to SUMMARY.md and continue to the next — don't halt the campaign.
