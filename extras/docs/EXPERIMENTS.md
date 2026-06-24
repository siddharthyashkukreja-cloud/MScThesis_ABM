# Overnight experiment campaign — procedure & interpretation

Self-contained runbook for the contagion-experiment batch that turns the single-seed
traces into **ensembles** and produces the numbers for results chapters 5–6 (H1, H2),
the E6 structural finding, and the open-disorder sensitivity. The driver
(`run_experiments.py`) and the launcher (`run_overnight_experiments.sh`) are already
written and **smoke-tested**; the open-disorder lever (`globals.REANCHOR_ON_GAP`) is
already wired and behaviour-neutral at its default. This file documents the matrix,
how to run, the outputs, and how to read them tomorrow.

## What it tests

| Family | Question | Lever varied | Recovered by slicing |
|---|---|---|---|
| **H1** | Does the client tier *absorb* or *amplify* stress vs direct clearing? | `tiered` vs `direct` (`build_clearing_tier(direct=)`) | mode, per fundamental |
| **H2** | Does procyclical IM amplify vs through-the-cycle, and at what calm cost? | `IM_MODE` reactive / static / flat-12 / flat-05 | im_regime (tiered) |
| **E6** | Gaps vs a single shock at matched drawdown (overnight-gap question) | `scenario` gapped vs shock | scenario (Kalman, tiered) |
| **Open-disorder** | Does a disorderly open (your "natural shock" idea) change contagion? | `REANCHOR_ON_GAP` True vs False | reanchor (Kalman) |
| **SV-MJD overlay** | Do H1/H2 hold on the synthetic fundamental? | `fund` kalman vs gbm | fund |
| **H2 calm-cost** | Collateral posted in calm per margin scheme | IM_MODE on the calm regime | regime=calm |

Each cell is a **40-seed ensemble** over a reverse-stress amplifier grid
`c ∈ {1.0, 1.25, …, 3.0, 3.5, 4.0}` (Euronext A9 §5). The Kalman arm uses the real
COVID window + `globals.CALIBRATED["stressed"]`; the gbm arm uses
`data/fv_gbm_stressed.csv` + the SV-MJD re-lock θ (`output/relock_gbm/grid_stressed.json`).
One deduplicated master spec set (~5,500 runs) is run once and sliced per family, so
reactive+tiered is computed a single time and shared by H1 and H2.

## How to run

```bash
# from the repo root, in tmux (so it survives a closed terminal):
tmux new -s exp
./run_overnight_experiments.sh          # pre-flight smoke MUST pass, then the full matrix
#   detach: Ctrl-b d   |   reattach: tmux attach -t exp   |   live log: tail -f output/experiments/run.log
```

The script takes a single-instance lock, runs the smoke test, **aborts if any smoke row
errors**, then runs the full matrix and writes `output/experiments/SUMMARY.md`.
It is **resumable**: if interrupted, just re-run it — completed `(config|c|seed)` rows in
`rows.csv` are skipped. Tuning knobs (env): `NPROC` (default = cores−1), `SEEDS`
(default `42:82` = 40 seeds), `CGRID` (comma list).

**Runtime:** ~1.5 s/run; full matrix ≈ 30–60 min on 4–8 cores (≈2.5 h worst case single-core).
Comfortably overnight.

## Outputs (`output/experiments/`)

- `rows.csv` — one row per run (every config column + cascade metrics + IM metrics + `error`). The raw material; everything else is derived.
- `SUMMARY.md` — headline tables, rendered.
- `summary_h1.csv` / `summary_h1_breach.csv` — H1 ensemble means per `c`, and the **mutualisation breach multiplier** (smallest `c` reaching waterfall L3/L4) distribution over seeds, per fundamental × mode.
- `summary_h2.csv` / `summary_h2_breach.csv` / `summary_h2_margin_metrics.csv` — H2 cascade outcomes, breach multipliers, and the BCBS-CPMI-IOSCO 2025 margin metrics (peak-to-trough IM, max single-cycle jump, responsiveness = ΔIM% ÷ Δσ%).
- `summary_h2_calmcost.csv` — mean posted IM in **calm** per margin scheme (the procyclicality-vs-cost trade-off).
- `summary_e6.csv` — gapped vs shock, per `c`.
- `summary_opendisorder.csv` — clean open (reanchor=1) vs disorderly open (reanchor=0), per `c`.

## How to read it tomorrow (expected directions)

These are the claims the ensembles should let you state with error bars. Confirm the
**direction**; the exact multipliers are the result.

1. **H1 (headline).** In `summary_h1_breach.csv`, the L3/L4 breach `c` for **tiered** should be **higher** than for **direct** (the client tier buys mutualisation headroom). Single-seed first pass: direct L3 at c≈2.5, L4 at c≈3; tiered pooled-DF untouched through c=4. The ensemble gives the median + p10/p90 across seeds. If tiered ≥ direct holds across seeds → H1 buffer clause confirmed.
2. **H2.** `summary_h2_breach.csv`: **reactive** (procyclical) should breach at a **lower** c than **flat-12** (through-the-cycle) if procyclicality amplifies — or the difference is small, which is itself a finding. Pair with `summary_h2_calmcost.csv`: flat-12 posts far more IM in calm (the cost of stability). That tension *is* the H2 result (Glasserman-Wu).
3. **E6.** `summary_e6.csv`: at equal `c` (matched drawdown), **shock** should show ~3× the client defaults and reach a deeper waterfall at lower `c` than **gapped** → contagion tracks concentration/speed, not magnitude; the overnight-gap structure is a mitigant.
4. **Open-disorder.** `summary_opendisorder.csv`: reanchor=0 (disorderly open) should show **somewhat more** contagion than reanchor=1 at the same `c` (smoke: 4 vs 1.5 client defaults at c=1). If the gap is modest → the clean-reprice default is a defensible simplification; if large → worth featuring. Either way it answers your overnight-gap question empirically.
5. **SV-MJD overlay.** Compare `fund=gbm` vs `fund=kalman` rows in the H1/H2 summaries: the qualitative ordering (tiered > direct; reactive ≤ flat-12) should survive on the synthetic path → robustness. Do **not** compare absolute D/defaults across fundamentals (different path).

## Turning this into the completed model (next-day steps)

1. **Drop the ensemble numbers into Ch. 6** (`chapters/05_results.tex` / `06_implications.tex`, both currently empty): H1 breach-multiplier table + figure, H2 regime comparison + calm-cost, E6 gaps-vs-shock, open-disorder sensitivity. The `[NUMBERS]` markers in `calibration.tex` point at `output/relock/` for the market-layer tables.
2. **Figures:** add an ensemble plot (breach `c` distribution; defaults vs `c` by mode/regime) — a small `plot_experiments.py` over `rows.csv` (not yet written; trivial follow-up).
3. **Finalise the H1/H2 framing** in the intro/lit-review (the `\citet` + Paddrik/Galbiati edits flagged earlier).
4. **Optional** (only if a result motivates it): H3 collateral-demand decomposition (the IM series is already logged in `rows.csv`), or a multi-path SV-MJD ensemble (regenerate `fv_gbm` per seed) for a richer robustness story.

## Notes / caveats baked in

- Behavioural θ: Kalman arm = `globals.CALIBRATED["stressed"]` (the D65 lock); gbm arm = the SV-MJD re-lock θ. No calibration runs here — purely contagion.
- `D` / loss is **not** used here; these are clearing-tier outcome metrics.
- The open-disorder flag only gates trader re-anchoring at the gap; the reprice and the reactive-IM estimator are unchanged, and the default (`True`) is byte-identical to the prior model (verified: seed-42 c=1 trace = −18.8% / 3 client defaults / L0 / DF $0.63B).
- `output/experiments/rows_smoke.csv` is a pre-flight artifact and can be deleted.
