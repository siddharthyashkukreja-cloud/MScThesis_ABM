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
