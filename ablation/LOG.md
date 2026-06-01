# Calibration Ablation Campaign — run log

Autonomous run per `ablation/PLAN.md`. Sid reviews after completion.

## Setup / standing facts (survives context compaction)
- **Branch:** `ablation` (NEVER push, NEVER touch `main`).
- **BASE commit:** `5cbf89082281cce7d319228363a0e8c54ad7573d` ("ablation baseline").
  Per-cell reset: `git restore --source=5cbf8908 -- calibrate.py run_simulation.py model/ data/fv_calm.csv data/fv_stressed.csv`
- **Interpreter:** this machine has no `python` on PATH — using `python3`
  (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, Python 3.13.3).
  `python3 calibrate.py run` is the per-cell command (identical budget every cell:
  the defaults N_LHS=128, N_DAYS=20, N_RUNS=4, N_REFINE=2, N_PER_REFINE=32, N_STAGE2=32, SEED=42).
- **Recorder:** `ablation/record.py` reads `output/calibrated_params.json` and appends one
  results.csv row per regime. Validated against a real training run before C0. Lives in
  `ablation/` so it survives the per-cell code reset.
- **Per-cell run log:** raw calibrator stdout saved to `ablation/run_<cell>.log` (also survives reset).
- Baseline structure: Kalman fundamental; 4-d loop (ft_sigma_c, zi_alpha, zi_mu, zi_delta);
  p_zi fixed at globals.P_ZI (L2/MBP-10); KS+Hill loss (5 components); n_mm=0, n_vt=0.

---

## C0 — Baseline (no edit). Kalman fundamental, 4-d loop, KS+Hill (5-comp) loss, n_mm=0 n_vt=0.
- **Reference D**: calm 48.36, stressed 29.11. These are the comparison anchors for all later cells.
- theta_stage2: calm {ft_sigma_c 0.606, zi_alpha 0.340, zi_mu 0.087, zi_delta 0.221};
  stressed {ft_sigma_c 0.561, zi_alpha 0.282, zi_mu 0.096, zi_delta 0.134}.
- Component split — calm: KS 23.4 dominates, then ACF2 13.3 (clustering miss), V 6.5, ACF1 4.4, Hill 0.83 (good tail).
  stressed: KS 18.4 dominates, Hill 3.9, ACF2 4.0, ACF1 2.6, V 0.06 (vol level matched).
- Diagnostics — calm: Hill 3.02, kurtosis 9.0, ret_std 3.5e-4, surrogate R² 0.90.
  stressed: Hill 2.44, kurtosis 58.4, ret_std 1.8e-3, surrogate R² 0.95.
- Interpretation: KS (whole-distribution fit) is the largest loss term in both regimes; the stressed
  fat tail (kurt 58, Hill 2.4) is what KS+Hill is chasing. Calm's residual is clustering (ACF2). Clean run, exit 0.

## C1 — SV-MJD fundamental (Stein-Stein vol + Merton jumps) instead of Kalman.
- Edit: `python3 data/v_gbm.py generate-all 42 30` regenerated data/fv_{calm,stressed}.csv as
  smooth synthetic SV-MJD series. No calibrate.py code change. Caches deleted; run exit 0.
- **fv length used**: SV-MJD = 11700 bars/regime (30 days × 390). NOT like-for-like vs Kalman BASE
  (calm 102758, stressed 11445 real bars) — the stressed window in particular is synthetic-30d vs
  real-29d. Per PLAN this is a mechanism-level comparison, not a fit comparison. Both regimes' V_t
  had 94 Merton jumps; calm σ_t mean 1.98e-4, stressed σ_t mean 1.22e-3.
- D vs baseline: calm 47.17 (vs 48.36, ≈flat -1.2); stressed 36.92 (vs 29.11, WORSE +7.8).
- Components moved — calm: KS up (24.4 vs 23.4), ACF2 up (14.9 vs 13.3), Hill worse (4.48 vs 0.83),
  V improved (0.91 vs 6.5). stressed: KS up (23.0 vs 18.4), Hill worse (5.73 vs 3.93).
- Diagnostics: calm hill 2.61 / kurt 16.9 (was 3.02 / 9.0 — fatter tails); stressed hill 2.11 / kurt 47.9.
- Interpretation: the smooth SV-MJD fundamental does NOT improve fit — it WORSENS the tail match
  (Hill) in both regimes and the whole-distribution KS, especially in stressed. The Kalman real
  efficient price (BASE) transmits microstructure the synthetic jump-diffusion smooths over;
  the model leans on faithful V_t for its tail. Confirms fundamental choice matters and Kalman is better.
