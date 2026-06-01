# Calibration Ablation Campaign — autonomous run

Goal: measure how each calibrated parameter and each structural choice affects the
market-layer fit, by re-running `python calibrate.py run` with ONE thing changed at a
time, recording the loss D, every component distance, and the key moments per cell.
Sid reviews tomorrow. Work through the cells in order; if time runs out the
higher-priority cells are already done.

## Standing rules (never violate)
- Work ONLY on git branch `ablation`. NEVER `git push`, never open a PR, never touch `main`.
- Change exactly ONE thing per cell. Reset code to the baseline before the next cell.
- Never alter model logic beyond the explicit edit a cell specifies.
- Records live in `ablation/` and must survive every code reset. Append, never overwrite.
- If a run errors: log the error in `ablation/LOG.md`, reset, and move to the next cell.
  Do not get stuck on one cell.
- Re-read THIS FILE at the start of every cell (it survives context compaction; your
  chat memory may not).

## Baseline
Current repo state = the baseline: Kalman fundamental; 4-d loop
(ft_sigma_c, zi_alpha, zi_mu, zi_delta); p_zi fixed at the L2/MBP-10 values
(globals.P_ZI); KS+Hill loss (5 components); n_mm=0, n_vt=0.

First-time setup:
```
git checkout -b ablation
git add -A && git commit -m "ablation baseline"      # note this hash; call it BASE
mkdir -p ablation
```

## Per-cell protocol
0. Re-read this file. Confirm `git branch` shows `ablation` and `git status` is clean
   except for `ablation/`.
1. Reset code+data to baseline:
   `git restore --source=BASE -- calibrate.py run_simulation.py model/ data/fv_calm.csv data/fv_stressed.csv`
2. Apply the cell's edit (below).
3. Delete LHS caches so nothing stale is reused (the cache guard does NOT detect
   structural changes — always delete): `rm -f output/calibration_lhs_*.csv`
4. Run (identical budget every cell for comparability): `python calibrate.py run`
5. Read `output/calibrated_params.json`; for EACH regime append one row to
   `ablation/results.csv` (columns below) from theta_stage2, true_loss_validated,
   component_deltas, and validated{...}.
6. Append 3-5 lines to `ablation/LOG.md`: cell id, what changed + its literature
   justification, what happened (D vs baseline, which components moved, kurtosis / Hill /
   clustering), one-line interpretation.
7. `git add ablation/ && git commit -m "cell <id>: <desc>"`  (records only; the code
   edit stays uncommitted and is wiped by step 1 next cell).

## results.csv columns
cell, regime, fundamental, mm, vt, free_params, fixed_params, D, dKS, dV, dACF1, dACF2,
dHill, ret_std, ret_kurtosis, hill, acf_r_1, acf_absr_1, acf_absr_5, acf_absr_10,
acf_absr_20, surrogate_r2, n_samples, timestamp

## Cells (priority order)

C0  Baseline — no edit. Run + record. (Reference D under KS+Hill + the stressed cap.)

Structural variants (Sid's priority):
C1  SV-MJD fundamental instead of Kalman — regenerate the fundamental as
    Stein-Stein + Merton jumps: `python data/v_gbm.py generate-all 42 30`. Record the
    fv length used in LOG.md (it differs from Kalman's real 29-day stressed window — the
    comparison is mechanism-level, not like-for-like). Step 1 next cell restores the
    Kalman fv_*.csv (they are tracked); VERIFY `git status` shows fv_*.csv clean before C2.
    Justification: fundamental sensitivity (smooth synthetic vs Kalman real efficient price).
C2  Market Maker on — in calibrate.py POP set n_mm=4 (HFABM mid-anchored, D48);
    leave mm_p_edge=4.0. Justification: HFABM/Vytelingum liquidity provider; tests the
    tail-vs-clustering trade-off (the MM was removed because it damped clustering).
C3  Volatility Trader on — in calibrate.py POP set n_vt=10 (Gao §3.1.3 vol-scaled
    demand, D38); vt_qty_base stays 2.0. Justification: tests whether the VT supplies the
    long-lag clustering the single-cohort MT cannot reach.

Parameter-fix (fix one calibrated param at its literature/logic value; others stay
free. Method: REMOVE the key from PARAM_BOUNDS AND pass the fixed value explicitly as a
kwarg in `_theta_to_params`'s ModelParams(...) call — do not rely on the dataclass default):
C4  Fix ft_sigma_c = sqrt(390) (Chiarella one-daily-V_t-std; globals.FT_SIGMA_C_DEFAULT).
C5  Fix zi_alpha = 0.15  (Cont-Stoikov-Talreja 2008 limit-arrival baseline).
C6  Fix zi_mu    = 0.025 (CST market-arrival baseline).
C7  Fix zi_delta = 0.15  (CST cancellation baseline).
C8  Fix ALL zi rates at CST (alpha 0.15, mu 0.025, delta 0.15); only ft_sigma_c free.
    Tests whether calibrating the ZI rates matters vs pure-literature ZI.

Add-parameter:
C9  p_zi in the loop — add `"p_zi": (0.2, 0.8)` to PARAM_BOUNDS (5-d; the bracket
    straddles both regimes' L2 values 0.543 / 0.343). Tests free depth vs L2-fixed.

Loss ablation (justify the KS+Hill choice):
C10 KS-only  — COMPONENT_NAMES = ("KS","V","ACF1","ACF2").
C11 Hill-only — COMPONENT_NAMES = ("V","ACF1","ACF2","Hill").

## Notes
- ~12 cells x ~45-55 min ≈ 10-11 h. Stop whenever Sid returns; partial is fine.
- Keep the calibration budget identical across cells (the `calibrate.py run` defaults).
- After C1, double-check fv_calm.csv / fv_stressed.csv are back to the Kalman BASE
  versions before continuing.
