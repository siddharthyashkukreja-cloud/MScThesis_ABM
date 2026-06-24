# Onboarding prompt — study this repo and gain full context

_Paste the block below to a fresh agent. It is written to be self-contained; the agent should read
the named files in the repo before doing anything else. Last refreshed 2026-06-21._

---

You are joining an in-progress **MSc thesis project** (Sid, VU Amsterdam; supervisor Dr. Rex Wang
Renjie; second assessor Prof. Albert Menkveld). The repo `mscthesis_abm` implements an **agent-based
model (ABM) of a CCP-cleared E-mini S&P 500 (ES) futures market** and uses it to test three
hypotheses about central clearing. Your job this session is to **study the repository and build
accurate context** before making any change. Do not trust this prompt's numbers over the code/docs —
verify as you read.

## 0. How to work here (read these rules first — they are firm)
- **Code is ground truth.** Verify every claim against the actual code/data, not memory or docs. Where
  a doc and the code disagree, the code wins (and flag it).
- **Sid writes the thesis prose.** Your job is the model, code, experiments, analysis, figures, and
  LaTeX *snippets* — not chapter prose unless explicitly asked. For doc/markdown corrections, just make
  them; for thesis writing, leave it to Sid.
- **The separate `MSc_Thesis` LaTeX folder is READ-ONLY.** Never edit it. Writing fixes go into
  `writing.tex` (the editable "communication pad" at the repo root) and `agent_context/WRITING_FIXES.md`.
- **No invented data or calibration.** No synthetic results; flag any deviation from the referenced
  methods/papers.
- **Be concise and direct.** Minimal preamble; do the work; report briefly.

## 1. Read these, in order (then summarise what you learned)
1. `AGENT.md` — working briefing: current status, committed model state, conventions, repo map, and the
   full dated change history (most-recent first; start at **D76**).
2. `model.md` — the in-depth technical reference: agents, equations, the clearing tier, parameters,
   calibration, validation, and the results chapter (§6).
3. `agent_context/EXPERIMENT_RESULTS.md` — the committed H1/H2/NET results (40 seeds/arm) with full
   tables, dispersion, and caveats.
4. `agent_context/THESIS_SYNTHESIS.md` — the narrative/claims and how each hypothesis should be framed.
5. `agent_context/RESULTS_PLAN.md` — results-chapter plan, run checklist, and the examiner-grade
   methodology checks (some RESOLVED, some open).
6. `clearing_writeups.tex` + `calibration_writeups.tex` — the model/calibration write-ups.
7. Skim `agent_context/CALIBRATION_VALIDATION.md`, `CLEARING_LAYER_REVIEW.md`, and
   `SECOND_READER_REVIEW.md` for methodology depth (these are dated snapshots — treat as historical).

## 2. The model in one screen
- **Two layers.** (a) **Market**: ~100 agents (Fundamental, Momentum, Zero-Intelligence) trading on a
  discrete **call-auction LOB** with a uniform clearing price each minute, around an **exogenous
  fundamental `V_t` = the contemporaneous empirical 1-min ES BBO mid** (not Kalman-smoothed).
  (b) **Clearing tier**: 1 CCP, 10 banking CMs (5 carry clients) + 5 non-banking CMs, 90 clients;
  physical IM escrow, procyclical VaR margin, cover-2 default fund, a 5-level EMIR waterfall, and
  Almgren-Chriss close-out.
- **Calibration.** Surrogate-assisted SMM, two-stage (XGBoost global search → local grid refine;
  HFABM/Gao 2022 §4.2). **7 parameters per regime** (`ft_sigma_c, zi_alpha, zi_delta, zi_mu, p_zi,
  mt_lambda, mt_gamma`), both **calm (2019)** and **stressed (COVID-2020)** regimes. Loss = standardised,
  inverse-SD-weighted moment distance with day-block bootstrap; validated by Moment Coverage Ratio +
  Franke J-test. θ is wired into `model/globals.py::CALIBRATED`.
- **Hypotheses.** H1 tiered vs direct clearing; H2 margin procyclicality (reactive / flat05 / flat12 /
  static, + DF-decouple arm); NET gross vs net client margining.

## 3. Current committed state (verify in `model/globals.py`)
- **IM:** `im_fraction = max(IM_FLOOR, z·σ_daily·√MPOR)` (uncapped — D77), with **IM_FLOOR=0.04,
  IM_CAP=None, IM_CONF_Z=3.0 (FHS), MPOR=1** (CME 1-day futures horizon); daily close-to-close RiskMetrics
  EWMA (λ=0.94) warmed on pre-window history; physical escrow on. `DF_STRESS_FLOOR=0.08`. NB: the committed
  `output/thesis_final/experiments` predates D77 (ran at MPOR=2 + 12% cap) — STALE pending re-run.
- **Experiments DONE** (40 seeds/arm, both regimes) → `output/thesis_final/experiments/`
  (`rows.csv` + `summary_{h1,h2,net}.csv`); driver `scripts/run_thesis_experiments.py`.
- **Headline results** (see EXPERIMENT_RESULTS.md): **H1** tiered draws the mutualised fund in 0/40 vs
  **direct 80% reach L≥3** (disclose IM confound: $32.6B tiered vs $10.3B direct). **H2** procyclicality
  **3.0×** (IM fraction 4%→12%, cap binds ~38% of stressed sessions); defaults flat05 4.1 / reactive
  1.3 / flat12 0.65. **NET** posted client IM **0.58×** gross (~42% saving).
- **Figures:** `calibration_figures.ipynb` (data summary + stylised facts: return CDF, **Hill tail-index
  plot**, ACF of r and |r|, 2×2 price path, and the clearing-network inline) and `results_figures.ipynb`
  (descriptive A–E + H1/H2/NET). `clearing_topology.ipynb`/`.png` is the standalone network figure.

## 4. Repo map (the parts you'll touch)
- `model/` — `globals.py` (all params + `im_fraction`/`df_stress_move` + `CALIBRATED`), `agents.py`,
  `simulation.py`, `clearing.py`, `lob.py`, `run_simulation.py` (`build_traders`, `build_clearing_tier`).
- `scripts/` — `run_thesis_experiments.py` (experiment spine), `wire_lock.py` (wires the surrogate
  optimum into `CALIBRATED`, all 7 params), calibration drivers.
- Notebooks — `calibration_figures.ipynb`, `results_figures.ipynb`, `clearing_topology.ipynb`.
- `data/` — `processed/ES_front_{calm,stressed}_1m.csv` (empirical mid), `processed/ES_front_daily_1d.csv`
  (daily vol warming), `fv_{calm,stressed}.csv` (fundamental paths).
- `agent_context/` — working docs (above). `writing.tex` (repo root) — the editable writing pad.

## 5. Open items (don't assume these are done)
- **`static` H2 arm is a no-op** (≡ reactive): with `IM_DAILY=True`, `simulation.py` (~L238) takes the
  daily-σ branch before the `IM_MODE` check. Fix (freeze σ at session-0 for static) or drop the arm.
- **H2 is collateral-demand / leverage-cycle, NOT crash amplification** — the price path is exogenous
  and identical across arms. Frame accordingly unless a price-feedback arm is added.
- **GBM scenario ensemble is non-runnable** (`gbm_lever.py` expects per-seed files that aren't
  generated) — so all current results are response-to-a-fixed-shock, not path uncertainty.
- **Chapters 5–7 + appendix unwritten;** ch.4 has `---` placeholders (stressed MCR col, R², θ table) to
  fill from `writing.tex`. Bibliography renders numeric not APA — stale BibTeX build (see WRITING_FIXES.md §B).

## 6. First actions
1. Read the §1 files; reconcile any doc-vs-code mismatch (code wins) and note them.
2. Sanity-run the model: `PYTHONPATH=. python3 -c "from model.globals import CALIBRATED; print(CALIBRATED)"`
   and confirm both regimes have 7 params; optionally smoke-run a notebook setup cell.
3. Re-derive one headline number from `output/thesis_final/experiments/rows.csv` to confirm
   EXPERIMENT_RESULTS.md (e.g., H1 direct L≥3 frequency, or NET net/gross ratio).
4. Report back: a short summary of the model, the committed state, the results, and the open items —
   plus any discrepancies you found. Then wait for the specific task.
