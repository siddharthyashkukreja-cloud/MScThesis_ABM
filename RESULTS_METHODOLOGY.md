# Results & methodology — central-clearing ABM (combined reference)

_Repo-root results + methodology reference (merges the former `RESULTS.md` and `RESULTS_METHODOLOGY.md`)._

**Scope (current framing).** The chapter rests on a **single tested hypothesis — H1, margin procyclicality**
(§3). **The contribution is the instrument, not the count of hypotheses:** the calibrated, validated ES
clearing ABM and its **synthetic-path ensemble** (§4) are the headline contributions; H1 is the illustrative
experiment. Two earlier experiments were **dropped** (full methodology + reasons in Appendix A): **gross-vs-net
netting** (single-asset escrow can't show the fellow-customer contamination that is its real risk; no
measurable resilience effect) and **tiered-vs-direct** (the direct counterfactual is confounded).

**Model state (locked).** IM = **D77** (`im_fraction = max(4%, z·σ·√MPOR)`, `IM_CONF_Z=3.0` FHS, **MPOR=1**,
**no cap**; daily close-to-close EWMA λ=0.94 → ~4% calm to ~11.4% peak). Member solvency = **D78** (Basel III
**capital-adequacy ratio** `cash/exposure ≥ 8%`, both tiers; BCM deleverages to restore 10%, NBCM stops out;
clients freeze at 4%). House-only BCMs follow the same 8% floor under a finite **5× leverage cap**
(`POSITION_LIMIT_X_HOUSE=5.0`, `POSITION_LIMIT_CLIENTS_ONLY=True`); client-clearing BCMs keep the 2× cap.
θ = `globals.CALIBRATED` (7 params/regime). Prose pad: `writing.tex`; figures: `results_figures.ipynb`.

> **⚠ ALL numbers below are a STALE BASELINE.** They predate this session's changes (the BCMo finite house
> cap and the removal of the netting arm), and the 40-seed tables additionally predate D77 (MPOR=1 / no cap).
> **Refresh every magnitude from the overnight parallel run** (`output/results/descriptive/` +
> `output/thesis_final/experiments/`, 100/150 seeds) before quoting. The *directions* are expected to hold;
> the magnitudes need regenerating. Reproduction: `agent_context/analyze_results.py`.

---

## 1. Experimental design (shared methodology)

### 1.1 The instrument and what the seeds vary

The market layer is a discrete call-auction limit-order book of ~100 FT/MT/ZI agents trading around an
**exogenous fundamental** `V_t` = the empirical 1-minute ES front-month mid. For each regime `V_t` is a
**single fixed historical series** (calm = a quiet 2019 stretch, stressed = the COVID Feb–May 2020 episode).
The seeds vary the **agent RNG only** — the fundamental path, and therefore the drawdown, is held fixed:

| Regime | Mean drawdown | Range across seeds | SD |
|---|---|---|---|
| Calm | −5.36 % | −5.41 … −5.30 % | 0.018 |
| Stressed | −25.56 % | −25.68 … −25.48 % | 0.030 |

The residual spread is microstructure noise in the simulated mid, not scenario variation. **This is a
response-to-a-fixed-shock design**: the model replays one historical crash and measures the clearing layer's
response; genuine path/scenario uncertainty needs the synthetic ensemble (§4). Report dispersion across
seeds (mean + [min–max] / quantiles), not just means.

### 1.2 Cleared population and balance sheets

On the LOB: 30 FT + 10 BCM (cast as own-account FT) + 20 MT + 40 ZI = 100 agents. Off the book: 5 NBCM +
1 CCP. Five of the ten BCMs carry client books; 90 clients clear through the ten client-carrying CMs (5
BCMc + 5 NBCM) under a leverage-balanced / capacity-proportional assignment (largest clients → highest-
capital CMs). Loss-absorbing cash: BCM `U[$5,10]B`, NBCM `U[$0.5,3]B`, CCP $1.5B; clients FT `U[$0.5,3]B`,
MT `U[$0.2,1]B`, ZI `U[$0.2,0.5]B`. Calibration is bare-market (clearing off), so no clearing parameter
touches θ. (House-only BCMs: own book capped at 5× cash, so κ ≈ 0.10–0.20 — they run the leverage cycle and
can fail in deep stress, without the over-leverage that an uncapped 8%-floor allows.)

### 1.3 Windows

| Regime | Experiment window | Rationale |
|---|---|---|
| Calm | full 75 sessions (or 0–20) | quiet baseline |
| Stressed | full 73 sessions (or 10–30) | spans the COVID crash **including the trough** (~session 25, deepest point) |

The **dollar IM and default metrics** are measured over the experiment window; the **IM-fraction "clean
metric"** (§3.2) is a deterministic function of the daily σ path, reported over the **full regime daily
series** (75 calm / 73 stressed sessions).

### 1.4 Arms — how each is constructed

The single live experiment flips named globals on the standard cleared population (no engineered fragile
agent):

| Experiment | Toggle | Arms |
|---|---|---|
| **H1 — margin** | `IM_MODE` / `IM_FLAT_FRAC` | reactive (4→11.4 % VaR), flat-4 %, flat-8 %, flat-12 % |

All arms are tiered + gross. The `static` arm is dropped (a no-op under `IM_DAILY` — it silently equals
reactive; see Appendix A.3). The dropped experiments (tiered-vs-direct `direct` toggle; gross-vs-net
`CLIENT_MARGIN_NETTING`) are in Appendix A.

### 1.5 Observables — the `rows.csv` columns (+ new per-run CSVs)

| Column | Definition |
|---|---|
| `drawdown` | `100·(min simulated mid / first mid − 1)` |
| `client_defaults` | number of client default events (`len(client_history)`) |
| `cm_defaults` / `nbcm_defaults` | distinct CM / NBCM `agent_id`s that defaulted |
| `deepest_waterfall` | max waterfall level reached (0–5); `mutualised_l3`/`survivor_l4` = `≥3` / `≥4` |
| `total_df` / `ccp_cash_used` | final default-fund size / `CCP_CASH − ccp.cash` (SITG + L5 drawn) |
| `client_loss_absorbed` | `Σ closeout_loss + Σ shortfall` over clients (mutualised/realised loss) |
| `im_mean` / `im_peak` | mean / max of **total system IM** (house **and** client) |

**New per-run detail CSVs (this session — both core and H1 emit these; the base ODD has no client
observables, so this is all new ground):**

- `client_defaults.csv` — one row per client default: client **type** (FT/MT/ZI), carrying CM + type,
  position-at-default (`assumed_pos`), IM posted, loss, timing → defaults by type/CM + the tail-concentration
  mechanism (median ↓ / p99 ↑ as margin rises).
- `waterfall_events.csv` — per default: level + **L1–L5 decomposition** (own DF / SITG / pooled fund /
  survivor cash / CCP cash) + mutualised → the waterfall-level distribution across all seeds.
- `client_freezes.csv` — every freeze onset: reason (`own_distress` vs `cm_contagion`), κ, type, CM, timing
  → the client→CM contagion channel quantified.
- `porting_events.csv` — per member default: `n_ported` / `n_unported` / unported position (EMIR Art. 48).
- `member_balance_band.csv` (core) — cash / maintenance-margin / DF-contribution bands by member type.

### 1.6 The margin mechanism

`im_fraction(σ) = max( IM_FLOOR , IM_CONF_Z·√MPOR · σ_daily )` with the cap dropped (D77: `MPOR=1`,
`IM_CAP=None`) → `max(0.04, 3.0·σ_daily)`, which self-bounds at ~11.4 % on the COVID path. Under
`IM_DAILY=True` the driving `σ_daily` is the **close-to-close daily RiskMetrics EWMA** (λ=0.94), warmed on
real pre-window daily ES history and looked up by each step's session date (no cold start, native overnight
gaps). IM is recomputed hourly, VM hourly, the cover-2 default fund daily. Flat arms bypass σ entirely
(`im_fraction = IM_FLAT_FRAC`). _NB: the 8 % flat-IM arm is unrelated to the 8 % capital-adequacy ratio._

### 1.7 Default trigger and the waterfall

Under physical escrow, IM moves as real cash to the CCP each cycle; a participant that **cannot fund a VM
or IM call from its own cash defaults on liquidity**. On default the defaulter's posted IM is seized first
(L0, pre-waterfall); a tiered **client** is liquidated by its member (Almgren–Chriss firesale — the client
price-impact channel); a **member** is resolved by the CCP via transfer at `CLOSEOUT_RECOVERY=0.80`, the
residual deficit mutualised through the five-level waterfall:

| Level | Resource | Note |
|---|---|---|
| L0 | defaulter's posted IM | seized before the waterfall (escrow) |
| L1 | defaulter's own DF contribution | |
| L2 | exchange skin-in-the-game | |
| **L3** | **pooled surviving-member DF** | **mutualisation begins** (`mutualised_l3`) |
| L4 | surviving-member cash, pro-rata | the contagion channel |
| L5 | CCP cash | $1.5B backstop |

Solvency floors (D78): every member is held to the ODD's Basel III **capital-adequacy ratio**
`cash/exposure ≥ 8 %`; a **BCM** deleverages its own book toward 10 % on breach, an **NBCM** stops out and
defaults on cash (the bank/non-bank asymmetry). Clients freeze at `cash/exposure ≤ 4 %`. The mutualised loss
is the **realised cash deficit**, not a flat (1−recovery)·notional haircut.

---

## 2. The calibrated model under stress (descriptive lead-in)

Establishes the instrument behaves sensibly before the margin experiment (full-window run, COVID −25.6 %,
current code; `output/results/descriptive/`):

- **The bank leverage cycle fires:** ~3 of 5 client-BCMs breach the 8 % floor and deleverage toward 10 % per
  seed (min κ median ≈ 7.6 %), concentrated in the crash sessions; NBCMs sit between (median κ ≈ 15–25 %).
  House-only BCMs now also ride the floor (κ ≈ 0.10–0.20 under the 5× cap) rather than sitting inert.
- **Members are resilient:** ~0 BCM defaults; NBCM defaults rare. Members deleverage / stop but survive —
  consistent with March 2020 (a margin/funding event, not member failure).
- **Clients default ≈8/seed** (2–18), absorbed by member capital; the mutualised fund is reached only via the
  rarer NBCM failure (~1 stressed seed in 5–6), never by client defaults alone.
- **Levels:** IM ≈ $25–26 B, DF ≈ $4 B → DF/IM ≈ 17 % (realistic band); client IM share ≈ 50 % (a calibration
  caveat vs the ~67–70 % real CME share — see §5).

Descriptive figures (`results_figures.ipynb`, full calibrated window): fundamental vs simulated mid, order
flow, κ-drawdown (the 8 % CAR + client-BCM-deleverage asymmetry, log scale), the IM dollar path + procyclical
fraction (calm flat 4 %, stressed 4 %→11.4 %), posted IM by participant type, margin-call density.

---

## 3. H1 — Margin procyclicality (the hypothesis)

**Claim — two clean angles, one experiment.** (1) **Procyclicality / timing:** reactive (FHS-VaR) IM rises
sharply into the crash, draining member funding exactly when collateral is scarcest; flat / through-the-cycle
margin does not. (2) **Margin level / coverage:** a higher flat margin lowers client defaults but raises the
standing (calm) funding cost — a monotone coverage-vs-cost trade-off.

**Design.** `IM_MODE ∈ {reactive, flat-4 %, flat-8 %, flat-12 %}` × {calm, stressed} × N seeds.
- **flat-8 % ≈ the reactive stressed time-average.** The charged (daily-σ) reactive fraction averages
  **7.66 %** over the stressed window, so flat-8 carries essentially the same average margin as reactive
  without the ramp — **reactive-vs-flat-8 (stressed) isolates pure *timing* (procyclicality)**.
- **flat-4 % (floor) and flat-12 % (peak)** bracket the path and trace the *level / coverage-vs-cost*
  trade-off.

### 3.1 Framing — collateral demand, not amplification

The price path is exogenous (FTs anchor the mid to `V_t`; drawdown is identical across margin arms), so this
is a **collateral-demand / leverage-cycle** result, *not* crash amplification — and the exogenous path is the
binding ceiling: H1 shows first-round funding drain and coverage, never the margin→fire-sale→margin feedback
(that needs the synthetic ensemble). Anchor to **Murphy, Vasios & Vause** (BoE FSP 29; "A CBA of APC") and
the BoE collateral-cycle work — same FHS-VaR method, same flat/through-the-cycle vs reactive comparison.

### 3.2 The clean metric — IM-fraction path (deterministic in σ)

The charged IM fraction is computed on the daily close-to-close EWMA σ — the vol the model margins on, so it
is a property of the rule on the historical path, independent of the seeds. **D77 (live config):** calm holds
flat at the **4 % floor** (vol never triggers the VaR term, ratio 1.00×); **stressed runs 4 % → 11.37 %
(≈2.84×), mean 7.66 %**, elevated through the crash with no flat-top cap artifact — that ramp *is* the
procyclicality signature. (The older 3.0× / 4→12 % band was the MPOR=2 + 12 %-cap run.)

### 3.3 Margin-mode comparison — stressed (STALE baseline, refresh on re-run)

Most-recent committed run (12-seed, D77+D78, **pre-BCMo-cap**; client defaults + IM robust):

| arm | client defaults | calm IM | stressed peak IM |
|---|---|---|---|
| flat-4 % | 14.7 (5.3) | $16.7 B | $16.9 B |
| reactive (4→11.4 %) | 8.9 (5.2) | $16.7 B | $37.9 B |
| flat-8 % | 11.6 (6.5) | $29.9 B | $31.3 B |
| flat-12 % | 3.1 (2.6) | $39.7 B | $43.8 B |

Reading: defaults fall as margin rises (flat-4 → flat-12); **reactive sheds more defaults than flat-8 at the
same ~8 % average** (front-loads into the trough) and matches flat-4's calm cost — its only cost is the
**procyclical spike** ($16.7 B calm → $37.9 B peak). _(On the re-run with the 5× house cap the H1 picture is
the clean one: member defaults NBCM-only, reactive attractive — lowest member-def + mutualisation at flat-4
cost; refresh this table + add member-def + reach-fund columns from the new CSVs.)_

### 3.4 Do members breach their capital ratios? (deleverage vs default)

`rows.csv` records member *defaults* but not capital ratios — "no CM default" does not by itself establish
"no member hit a floor." A member can breach and deleverage *without* defaulting. Instrumented
(`agent_context/probe_capital_ratios.py`): the leverage cycle **does fire** — a handful of bank-CMs cross the
floor and deleverage their own book in a minority of stressed seeds — but it is **contained**: the
deleveraging banks shed risk and survive, so member defaults stay rare and breaches are shallow. **The H1
buffer result is "deleverage-but-survive," not "members are never stressed."**

### 3.5 Caveats specific to H1

- **SLOIM–default-fund coupling.** The baseline cover-2 fund is sized `(stress_move − im_fraction)·notional`,
  so a higher IM mechanically **shrinks** the pooled fund (and goes to 0 once flat IM exceeds the stress
  move). Any statement about how the margin regime changes mutualisation frequency is therefore partly about
  this sizing convention — which is why the H1 driver forces the **ODD fixed-percentage fund** (IM-independent)
  so the margin arms share one consistent fund. A `DF_DECOUPLE_IM` arm isolates the coupling.
- **`static` no-op** — dropped (see Appendix A.3).

---

## 4. Methodological contribution — a synthetic-path ensemble

Beyond the single historical replay, the model ships a **calibrated synthetic ES fundamental** (`data/v_gbm.py`:
stochastic-volatility process fit to ES realised measures + the overnight-gap pool). The current overnight
job (`scripts/overnight_joint.py --log-vol`) calibrates a **two-factor log-volatility OU** (D80: jumps
dropped, the log-vol carries the tail) and runs a **stressed ensemble**, so the cleared market is
re-instantiated on **an ensemble of plausible ES stress paths** — turning the experiment from a single
fixed-shock replay into a **Monte Carlo over calibrated stress scenarios**. Present it as a contribution:
*the calibrated cleared market can be regenerated on many stress paths and the clearing-outcome distribution
reported.* It also **repairs the one weakly-matched stylised fact** — single-factor vol under-matches the
|r|-ACF clustering; the second (slow) factor pulls those moments back into the empirical 95 % band (outputs:
`output/overnight_joint/validation.csv` 2f-vs-1f, `clearing.csv` ensemble).

---

## 5. Cross-cutting QA and caveats

**QA (on the committed runs).** Summary group means match a fresh recomputation from `rows.csv`; conservation
holds (tiered draws $0 CCP cash / $0 mutualised when no member fails); the IM-fraction band reproduces
exactly; drawdown is near-constant within regime (SD 0.02–0.03), confirming the fixed-shock design.

**Caveats to carry into the write-up.**
1. **Exogenous, single-path drawdown** — `V_t` fixed per regime; seeds test agent noise only. The model
   *replays* the crash; the synthetic ensemble (§4) supplies path/scenario uncertainty as a robustness layer.
2. **Collateral demand, not amplification** — price is exogenous; no margin→price feedback.
3. **Rare-event statistics** — member defaults / mutualisation are rare over 5 NBCMs; report frequencies with
   bands; individual arm-to-arm gaps below the full flat ladder span can be within noise.
4. **House/client IM split ≈ 50 %** — the model's large levered member banks vs the 90-client population push
   posted margin toward 50/50 vs the ~67–70 % client share observed empirically; a calibration caveat that
   does not affect the loss-allocation result (cite the ~67.7 % CME figure, not 82 %).
5. **Bare-market θ is not clearing-invariant** — the bare-market calibration understates clearing-active
   stressed dynamics (a disclosed limitation).

---

## 6. Headline numbers (precise restatement — STALE, refresh on re-run)

- **H1 procyclicality:** reactive IM fraction **4 %→11.37 %, 2.84×** over the stressed regime, mean 7.66 %.
- **H1 level-vs-cost (12-seed):** flat-4 **14.7** / reactive **8.9** / flat-8 **11.6** / flat-12 **3.1** client
  defaults; reactive holds flat-4's calm cost ($16.7 B) but spikes to $37.9 B at the trough.
- **Descriptive:** ~0 BCM defaults, NBCM rare; clients ≈8/seed absorbed at member capital; mutualisation only
  via the rare NBCM failure; IM ≈ $25–26 B, DF/IM ≈ 17 %.

---

## 7. Run plan + code state

- **Re-run (overnight, in progress):** `scripts/run_parallel.sh` — core (`run_model_descriptive.py`, 100
  seeds), H1 (`run_thesis_experiments.py`, 150 seeds), synthetic (`overnight_joint.py`), all parallel into
  separate dirs. Refresh §2/§3/§6 with mean + [min–max] bands and the new member-def / reach-fund / waterfall
  columns.
- **Figures:** `results_figures.ipynb` → IM-fraction path (reactive vs flat), defaults-vs-margin-level,
  descriptive (κ-drawdown, IM-by-type, margin-call clustering); **new** from the detail CSVs — waterfall L1–L5
  distribution, client-defaults-by-type, contagion-vs-distress freezes, porting rate, balance-sheet bands.
- **Code:** `model/globals.py` — D77 IM + D78 8 % CAR (`LR_FLOOR_BCM = REG117_FLOOR_NBCM = 0.08`,
  `DELEVERAGE_TARGET_BCM = 0.10`, `POSITION_LIMIT_X_HOUSE = 5.0`).

---

## Appendix A — Dropped experiments (for the record)

### A.1 Tiered vs direct clearing (dropped: confounded counterfactual)

*Original hypothesis:* the client-clearing tier localises losses — client defaults are absorbed at the
carrying member's capital, so the mutualised fund is not reached; removing the tier (clients clear direct)
routes the same defaults straight into SITG / pooled DF / survivors. *Result (40-seed, stale MPOR=2 run):*
tiered drew the mutualised fund in **0/40 seeds** (the ~1.3 client defaults/seed absorbed at member capital);
direct reached **L≥3 in 80 %** (mean depth 2.6), burning $93.8 M SITG + $1.63 B mutualised. *Why dropped:* the
counterfactual is **confounded** — (i) in the direct arm `cm_defaults == client_defaults` because the
directly-clearing clients *are* the members, and (ii) tiered holds **≈3.2× the IM** ($32.6 B vs $10.3 B,
because members post house IM on top of gross client IM), so direct reaches the waterfall partly because it is
*less margined*, not only because it lacks the member buffer. A clean test would need an IM-matched control.
The surviving insight — loss is **relocated** from the shared fund onto member balance sheets, not erased —
is retained as the descriptive finding (§2).

### A.2 Gross vs net client margining (dropped: single-asset can't show contamination)

*Original observation:* net (omnibus) margining nets offsetting clients → **≈0.58–0.60× the posted client IM
of gross/LSOC** (≈40 % saving, the Duffie–Zhu netting benefit) with almost no change in default counts
(1.30 → 1.48 stressed, stale). *Why dropped:* the model **escrows each client's IM individually and is
single-asset**, so the fellow-customer contamination channel — the real risk of net omnibus margining — is
not captured; the result is a clean efficiency number with no measurable resilience side, so it adds a
caveat-heavy section without a robust result. Removed from the driver, notebook, and `writing.tex`. (Netting
as a general CCP *rationale* — Duffie–Zhu multilateral netting — stays in the literature review; it is the
gross-vs-net *experiment* that was cut.)

### A.3 The `static` margin arm (no-op)

With `IM_DAILY=True`, `simulation.py` takes the daily-σ branch *before* checking `IM_MODE`, so `static`
silently uses the reactive daily-σ path — identical to reactive in every cell. Dropped (reactive-vs-flat
already carries the timing comparison).

---

## Appendix B — Reproduction

`agent_context/analyze_results.py` (run from repo root, `PYTHONPATH=.`) regenerates the tables + QA checks
from `output/thesis_final/experiments/rows.csv` and the live `model.globals`. The member capital-ratio /
deleverage measurement (§3.4) is not in `rows.csv`; reproduce it with `N=40 PYTHONPATH=. python3
agent_context/probe_capital_ratios.py`. The IM-fraction band is reproduced by replaying `globals.im_fraction`
over `globals.daily_sigma_series()` reindexed onto each regime's session dates. To regenerate the run:
`N_SEEDS=… PYTHONPATH=. python3 scripts/run_thesis_experiments.py` (or `scripts/run_parallel.sh` for all three
jobs).
