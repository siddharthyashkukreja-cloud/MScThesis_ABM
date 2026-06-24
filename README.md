# An agent-based model of central clearing

An agent-based model (ABM) of a centrally-cleared single-asset futures market — the E-mini
S&P 500 (ES front-month) — built to study CCP systemic risk: clearing-member default,
margin procyclicality, and client-clearing contagion under calm and stressed regimes.

The model has two tiers. A **market** layer (a limit order book with fundamental, momentum,
and zero-intelligence traders around a real-ES fundamental) is calibrated to reproduce the
empirical stylised facts of returns. A **clearing** layer (a CCP, 10 banking + 5 non-banking
clearing members, and 90 cleared clients, with physical IM escrow, procyclical margin, a
cover-2 default fund, and a five-level loss waterfall) is layered on top. Clearing members
carry their own proprietary book (gross-leverage capped at `|own| <= 2x cash`) and post a 15%
house margin on client positions (~6.67x leverage); a defaulted member's book is closed out by
the CCP at 0.80 recovery (transfer), while a defaulted client's book is liquidated open-market
by its member via Almgren-Chriss. The research question: does clearing clients *through
members*, rather than directly, change how a stress propagates to the CCP?

## Documentation

- **`model.md`** — the in-depth reference: agents, equations, the fundamental process, the
  clearing tier, every parameter, the calibration method and its locked result, and the
  contagion results. Start here for anything technical.
- **`AGENT.md`** — working briefing for a developer or AI assistant: current status, project
  conventions, repo map, and the full change history.
- **`sources.md`** (in the thesis repo) — the literature and sources notes.

## Repository layout

```
README.md  model.md  AGENT.md          the three docs (orientation / reference / dev briefing)
calibration_figures.ipynb              calibration-section figures (stylised facts, ACFs, moment table)
results_figures.ipynb                  results-section figures (H1/H2/NET, IM share)
clearing_topology.ipynb                tiered-clearing network map
model/                  the model (importable package)
  globals.py            parameters, regime configs, CALIBRATED theta, clearing constants
  lob.py                order, fill, limit order book
  agents.py             fundamental / momentum / zero-intelligence traders + clearing members
  clearing.py           balance sheets + CCP (escrowed IM, cover-2 default fund, waterfall, close-out)
  simulation.py         the driver: step loop, margin cycle, novation, default management
  run_simulation.py     entry point: builds the population + clearing tier, runs one session
scripts/                drivers (each self-bootstraps the repo root onto sys.path)
  calibrate.py          agent calibration: two-stage surrogate-SMM (`run`; grid = stage-2 local refine)
  validate_grid_fresh.py   fresh-seed re-ranking of grid optima
  run_thesis_experiments.py  the canonical H1/H2/NET ensemble driver
  gbm_lever.py          close-out & leverage stress levers + GBM ensemble (arms baseline/direct/flat/rec06/...)
  verify_rebalance.py   calm/stress rebalance verification + house-margin sensitivity
  analyze_volume_margin.py   volume + margin-by-type figures
  analysis_long_run.py  long-horizon (1-min/daily) moment validation
  run_relock.sh         reproduce the locked calibration
  run_thesis_final.sh   full thesis pipeline (relock -> verify -> MCR/J-test -> experiments)
data/                   data ingest + the offline-calibrated fundamental
  data.py roll.py       DataBento ingest + ES front-month roll / 1-min resample
  v_kalman.py           empirical-mid fundamental (Kalman MLE = efficient-price diagnostic, not applied)
  v_gbm.py              SV-MJD fundamental (robustness alternative)
  p_zi.py impact.py     geometric placement depth (legacy; p_zi now calibrated in the loop) + market-impact regression
  processed/  *.csv     rolled 1-min ES series + generated V_t paths
agent_context/          analysis/review docs, prompts, .tex drafts, scratch scripts, superseded notebooks
output/                 calibration + experiment outputs (relock/, relock_gbm/, thesis_final/)
extras/                 archived drafts, finished campaign scripts, reference PDFs, old output
```
Run everything from the repo root. The model imports as `from model.run_simulation import ...`; the
`scripts/*.py` add the repo root to `sys.path` so they run directly. See `AGENT.md` (FILE INVENTORY)
for the canonical-vs-scratch breakdown of every script.

## How to run

```bash
# 1. Offline data calibration (one-shot)
python3 data/v_kalman.py generate-all      # empirical-mid fundamental -> data/fv_*.csv (Kalman MLE = diagnostic)
python3 data/v_gbm.py calibrate            # SV-MJD parameters from ES realized measures
python3 data/v_gbm.py generate-all 42      # SV-MJD paths -> data/fv_gbm_*.csv (robustness)
python3 data/p_zi.py calibrate             # (legacy) MBP-10 depth — p_zi is now calibrated in the loop, not offline

# 2. One RTH day of the market + clearing simulation
python3 -m model.run_simulation

# 3. Calibrate the agent parameters (two methods, reported together)
python3 scripts/calibrate.py grid calm 5           # grid search, calm
python3 scripts/calibrate.py grid stressed 4       # grid search, stressed
python3 scripts/calibrate.py run stressed          # surrogate-assisted SMM
# the locked optimum is wired into model.globals.CALIBRATED; ./scripts/run_relock.sh reproduces it

# 4. Contagion experiments on the real COVID window
python3 scripts/verify_rebalance.py                # calm-clean / stress-contagion + house-margin sweep
python3 scripts/run_thesis_experiments.py          # the canonical H1 / H2 / NET ensemble
python3 scripts/gbm_lever.py                       # close-out & leverage stress levers + GBM ensemble
```

## Headline results

The market layer reproduces fat tails (Hill ≈ 3), near-zero return autocorrelation, and
short-horizon volatility clustering. In the clearing layer, clients post ≈75% of system IM
(toward the ~67–70% client share reported at major CCPs); calm and actual-COVID are clean/benign under the accurate
floors (client-clearing solvency CFTC Reg 1.17 cash/IM ≥8% for both tiers, plus a bank-only Basel/eSLR
leverage-ratio deleverage trigger; substantial $0.5–3B non-bank clearers) —
a handful of localised client defaults, no member defaults, no mutualisation, matching the historical
record (March 2020 was a margin/funding event). **H1 (tiered vs direct):** the member tier raises the
stress threshold for mutualisation: at the COVID-stressed severity (40 seeds) the tiered structure
localises losses (banks rarely default; mutualised fund drawn in 0/40) while **direct reaches L≥3 in
80% of seeds**; report as waterfall-depth vs drawdown, disclosing the IM confound (tiered $32.6B vs
direct $10.3B). **H2 (margin regime):** reactive daily close-to-close IM runs the 4%→12% band into the
crash (~3.0×, cap binds ~38% of stressed sessions; in line with CME ES IM ~3.9%→~10% of notional) and
drains member funding. **NET:** net (vs gross) client margining cuts posted client IM to ~0.58×
(≈42% saving). (Headline
numbers; the full ensemble is regenerated by the experiment harnesses below — see
`THESIS_SYNTHESIS.md` and `AGENT.md`.)
