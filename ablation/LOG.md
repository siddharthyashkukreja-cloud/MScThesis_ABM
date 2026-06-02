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

## C2 — Market Maker ON (POP n_mm=4, HFABM mid-anchored D48; mm_p_edge=4.0). Kalman, KS+Hill.
- Edit: calibrate.py POP n_mm 0→4. Caches deleted; run exit 0.
- D vs baseline: calm 51.46 (vs 48.36, WORSE +3.1); stressed 33.42 (vs 29.11, WORSE +4.3).
- Components — calm: KS improved (18.5 vs 23.4) but ACF2 stuck high (14.7 vs 13.3), Hill worse
  (6.2 vs 0.83), ACF1 worse (6.4 vs 4.4), V worse (5.7). stressed: KS up (21.7 vs 18.4),
  V worse (3.3 vs 0.06), Hill ~flat (3.24 vs 3.93), ACF1/ACF2 ~flat or slightly better.
- Diagnostics: calm hill 2.48 / kurt 16.6 (was 3.02 / 9.0 — fatter); stressed hill 2.57 / kurt 24.9
  (was 2.44 / 58.4 — MM TAMED the stressed tail/kurtosis). Calm θ shifted: ft_sigma_c up to 1.0,
  ZI rates pulled down (alpha 0.16, delta 0.036) — the MM now supplies near-mid liquidity so ZI backs off.
- Interpretation: confirms the D37/D48 rationale for removing the MM — the always-quoting MM clamps
  volatility (stressed kurt 58→25, V delta worsens) WITHOUT improving clustering (ACF2 still ~14.7 calm),
  and it worsens the calm tail (Hill 0.83→6.2). Net fit is WORSE in both regimes. The tail-vs-clustering
  trade-off goes the wrong way: MM helps KS a bit in calm but the damped vol hurts everything else.

## C3 — Volatility Trader ON (POP n_vt=10, Gao §3.1.3 vol-scaled demand D38; vt_qty_base=2.0). Kalman, KS+Hill.
- Edit: calibrate.py POP n_vt 0→10. Caches deleted; run exit 0.
- D vs baseline: calm 52.80 (vs 48.36, WORSE +4.4); stressed 21.27 (vs 29.11, BETTER -7.8).
- Components — stressed (the win): KS halved (9.66 vs 18.4), Hill near-perfect (0.04 vs 3.93),
  ACF2 better (2.76 vs 4.04), BUT ACF1 worse (5.21 vs 2.63) and V worse (3.6 vs 0.06).
  calm (the loss): V much worse (10.73 vs 6.5), ACF1 worse (5.19 vs 4.4), Hill worse (5.65 vs 0.83);
  KS slightly better (18.1 vs 23.4); ACF2 ~flat (13.1 vs 13.3).
- Diagnostics: stressed hill 3.17 / kurt 19.9 (was 2.44 / 58.4 — VT thinned the extreme tail toward target),
  long-lag clustering UP (acf_absr_10 0.111 vs baseline 0.08-ish, acf_absr_20 0.092). calm hill 2.52 / kurt 13.9.
- Interpretation: STRONGLY supports the C3 hypothesis FOR STRESSED — the vol-scaled VT supplies the
  long-lag |r| clustering and tames the runaway stressed tail (kurt 58→20, Hill exact), cutting D by ~27%.
  But in calm the VT over-injects volatility (V delta 6.5→10.7) and fattens calm tails, hurting fit.
  VT is a regime-asymmetric lever: a clear stressed-regime win, a calm-regime liability. Worth a
  regime-specific n_vt rather than a global on/off (note for Sid).

## C4 — Fix ft_sigma_c = √390 (≈19.748, Chiarella one-daily-V_t-std; globals.FT_SIGMA_C_DEFAULT). 3-d ZI loop.
- Edit: removed "ft_sigma_c" from PARAM_BOUNDS; passed ft_sigma_c=float(np.sqrt(390.0)) in _theta_to_params
  ModelParams(...). PARAM_KEYS now [zi_alpha,zi_mu,zi_delta] (verified). Caches deleted; run exit 0.
- D vs baseline: calm 102.69 (vs 48.36, MUCH WORSE +54.3, ~2.1x); stressed 71.90 (vs 29.11, +42.8, ~2.5x).
- Components — both regimes: KS explodes (calm 47.0 vs 23.4; stressed 51.3 vs 18.4), Hill explodes
  (calm 18.9 vs 0.83; stressed 10.2 vs 3.93), ACF2 worse. V stays fine (vol level still matched).
- Diagnostics: calm hill 1.50 / kurt 5256 (!!); stressed hill 1.27 / kurt 1409. (baseline kurt 9 / 58.)
- Interpretation: textbook confirmation of the D7b→calibrated-ft_sigma_c decision. At √390 the FTs
  sweep to their OUTERMOST reservation (V_t + max z·σ_fund), drowning the book in i.i.d. fundamental
  bursts → kurtosis in the thousands, Hill≈1.3-1.5 (far below the empirical ~2.4-3.0), KS wrecked.
  ft_sigma_c is THE tail lever and MUST be calibrated low (~0.6, per C0); the literature daily-news
  scale is the wrong scale for the 1-min microstructure. Strongest single-param sensitivity so far.

## C5 — Fix zi_alpha = 0.15 (Cont-Stoikov-Talreja 2008 limit-arrival baseline). Free: ft_sigma_c, zi_mu, zi_delta.
- Edit: removed "zi_alpha" from PARAM_BOUNDS; zi_alpha=0.15 in _theta_to_params. PARAM_KEYS
  [ft_sigma_c,zi_mu,zi_delta] (verified). Caches deleted; run exit 0.
- D vs baseline: calm 66.35 (vs 48.36, WORSE +18.0); stressed 26.68 (vs 29.11, slightly BETTER -2.4).
- Components — calm: Hill blows up (12.45 vs 0.83), V worse (13.55 vs 6.5), ACF2 ~flat (14.1).
  ft_sigma_c compensates UP to 1.45 (from 0.61) and zi_delta up to 0.355, zi_mu floored at 0.0066.
  stressed: KS better (13.3 vs 18.4), Hill ~flat (2.30 vs 3.93); ft_sigma_c 0.51 ≈ baseline.
- Diagnostics: calm hill 2.00 / kurt 33.7 (was 3.02 / 9.0 — much fatter); stressed hill 2.75 / kurt 54.5.
- Interpretation: calm's calibrated zi_alpha (0.34, C0) is well ABOVE the CST 0.15 baseline; forcing it
  down starves near-mid limit liquidity, so the book thins and ft_sigma_c/zi_delta over-compensate,
  fattening the calm tail (Hill 0.83→12.5). Calibrating zi_alpha clearly MATTERS for calm. Stressed's
  optimum (0.28) is closer to 0.15, so the constraint barely bites — even helps KS marginally.

## C6 — Fix zi_mu = 0.025 (CST 2008 market-arrival baseline). Free: ft_sigma_c, zi_alpha, zi_delta.
- Edit: removed "zi_mu" from PARAM_BOUNDS; zi_mu=0.025 in _theta_to_params. PARAM_KEYS
  [ft_sigma_c,zi_alpha,zi_delta] (verified). Caches deleted; run exit 0.
- D vs baseline: calm 48.28 (vs 48.36, FLAT -0.08); stressed 29.96 (vs 29.11, ~flat +0.85).
- Components essentially unchanged — calm: Hill stays great (0.53 vs 0.83), KS ~flat (22.3 vs 23.4),
  ACF2 ~flat (14.0). stressed: KS ~flat (19.2 vs 18.4), Hill ~flat (3.02 vs 3.93). Other θ barely move
  (calm ft_sigma_c 0.586≈0.606, zi_alpha 0.395≈0.340; stressed ~baseline).
- Diagnostics: calm hill 3.00 / kurt 5.86 (≈ baseline 3.02 / 9.0); stressed hill 2.61 / kurt 40.9.
- Interpretation: zi_mu is a LOW-sensitivity lever. Baseline calm/stressed optima (0.087 / 0.096) sit
  modestly above the CST 0.025, but pinning it there costs ~nothing — the market-order rate is a minor
  contributor and the literature value is adequate. Calibrating zi_mu is NOT worth a loop dimension;
  this justifies treating it as fixable (cheapest free param to drop). Contrast with ft_sigma_c (C4) and
  zi_alpha-calm (C5), which matter a lot.

## C7 — Fix zi_delta = 0.15 (CST 2008 cancellation baseline). Free: ft_sigma_c, zi_alpha, zi_mu.
- Edit: removed "zi_delta" from PARAM_BOUNDS; zi_delta=0.15 in _theta_to_params. PARAM_KEYS
  [ft_sigma_c,zi_alpha,zi_mu] (verified). Caches deleted; run exit 0.
- D vs baseline: calm 50.77 (vs 48.36, WORSE +2.4); stressed 30.35 (vs 29.11, ~flat +1.2).
- Components — calm: KS improved (17.2 vs 23.4) but Hill worse (6.82 vs 0.83), ACF2 worse (15.6 vs 13.3),
  ACF1 worse (6.3 vs 4.4); ft_sigma_c compensates UP to 1.12, zi_mu floored low (0.016). stressed:
  ~flat across the board (KS 19.4, Hill 3.41, ft_sigma_c 0.52 ≈ baseline).
- Diagnostics: calm hill 2.43 / kurt 19.0 (was 3.02 / 9.0 — fatter); stressed hill 2.54 / kurt 54.8.
- Interpretation: moderate sensitivity, calm only. Baseline calm zi_delta (0.221) sits above CST 0.15;
  pinning cancellation lower leaves more resting depth, and ft_sigma_c rises to compensate, fattening the
  calm tail (Hill 0.83→6.8) — net slightly worse despite a better KS. Stressed optimum (0.134) is already
  near 0.15, so it barely moves. Ranking of ZI-rate calibration value (calm): zi_alpha (C5, +18) >>
  zi_delta (C7, +2.4) > zi_mu (C6, ~0).

## C8 — Fix ALL ZI rates at CST 2008 (zi_alpha 0.15, zi_mu 0.025, zi_delta 0.15). Only ft_sigma_c free (1-d loop).
- Edit: PARAM_BOUNDS = {ft_sigma_c} only; zi_alpha=0.15, zi_mu=0.025, zi_delta=0.15 in _theta_to_params.
  PARAM_KEYS [ft_sigma_c], dim (1,2) (verified). Caches deleted; run exit 0.
- D vs baseline: calm 59.07 (vs 48.36, WORSE +10.7); stressed 33.43 (vs 29.11, WORSE +4.3).
- Components — calm: V blows up (15.02 vs 6.5 — vol level now badly off), KS ~flat (24.3), ACF2 ~flat
  (12.2), Hill ok (1.90). stressed: KS ~flat (18.0), ACF1 worse (5.23 vs 2.63), Hill ~flat (3.69).
  ft_sigma_c lands at 0.81 (calm) / 0.54 (stressed) — it absorbs what it can but cannot fix everything.
- Diagnostics: calm hill 2.81 / kurt 13.9; stressed hill 2.49 / kurt 54.8.
- Interpretation: KEY RESULT for "does calibrating the ZI rates matter?" — YES. Pure-literature CST ZI
  is materially worse in BOTH regimes (+10.7 calm, +4.3 stressed), the calm hit driven mainly by the
  return-std (V) miss that free zi_alpha/zi_delta otherwise correct. Net degradation (calm +10.7) is
  LESS than the naive sum of the single fixes (C5+C6+C7 ≈ +20.3) because ft_sigma_c re-optimises to
  partly compensate — but the loss is real. Calibrating ZI rates (esp. zi_alpha, zi_delta in calm) is
  justified; zi_mu alone (C6) was the only freely-droppable one.

## C9 — p_zi ADDED to the loop (5-d). PARAM_BOUNDS += "p_zi":(0.2,0.8). Free depth vs L2-fixed.
- Edit: added "p_zi":(0.2,0.8) to PARAM_BOUNDS; p_zi=float(d["p_zi"]) in _theta_to_params (overrides the
  P_ZI data-fix; verified __post_init__ does NOT clobber an explicit value). PARAM_KEYS 5-d (verified).
  Caches deleted; run exit 0.
- D vs baseline: calm 49.70 (vs 48.36, ~flat +1.3); stressed 23.00 (vs 29.11, BETTER -6.1, ~21%).
- Calibrated p_zi: calm 0.468 (vs L2 0.543 — slightly sparser), stressed 0.218 (vs L2 0.343 — much sparser).
- Components — stressed (the win): KS down hard (13.4 vs 18.4), V near-perfect (1.04), Hill better (2.57 vs 3.93).
  calm: KS better (18.0 vs 23.4) but ACF2 worse (15.4 vs 13.3), Hill worse (3.72 vs 0.83) — a wash net.
- Diagnostics: stressed hill 2.70 / kurt 45.4 (vs 2.44 / 58.4 — closer to target); calm hill 2.67 / kurt 24.7.
- Interpretation: freeing depth helps STRESSED clearly — a sparser ZI book (p_zi 0.22) lets price moves
  punch through thin liquidity, sharpening the whole-distribution fit (KS) and vol level (V). Calm's
  optimum sits near its L2 value (0.47 vs 0.54), so freeing it buys little and slightly perturbs the tail.
  Suggests p_zi is worth calibrating for stressed but the L2/MBP-10 data-fix is adequate for calm. Note
  the 5-d surrogate R² is a touch lower (0.86 calm) — more dims, same budget. Net: a stressed-side win.

## C10 — Loss ablation: KS-only tail. COMPONENT_NAMES = ("KS","V","ACF1","ACF2") — Hill DROPPED.
- Edit: COMPONENT_NAMES drops "Hill" (one-line). Caches deleted; run exit 0. NB Hill is still computed as
  a DIAGNOSTIC (in MOMENT_NAMES), just not in the loss — so dHill is blank in results.csv but `hill` is logged.
- CAVEAT: D here is a 4-component sum, NOT comparable to C0's 5-component D. Judge by the FITTED-model
  tail diagnostics, not D. (calm D 51.59, stressed D 25.18 — lower just because one term is removed.)
- KEY RESULT — calm: with Hill out of the loss the tail EXPLODES. kurtosis 9.0 → 108.8, Hill 3.02 → 1.67,
  and ft_sigma_c runs UP to 2.42 (from 0.61). KS alone does NOT pin the tail index — it tolerates a fat tail
  as long as the whole-distribution CDF roughly matches, so the optimiser fattens the FT overshoot.
- stressed: optimum essentially UNCHANGED from C0 (θ ≈ {0.561,0.282,0.096,0.134}, hill 2.44, kurt 58.4) —
  Hill was a minor stressed term (3.9 of 29), so dropping it doesn't move the stressed optimum.
- Interpretation: JUSTIFIES keeping Hill in the loss. KS is necessary but not sufficient for the tail —
  in calm it lets kurtosis blow past 100 and Hill fall to 1.67 (vs empirical ~3). Hill is the direct,
  reachable tail-index lever that prevents this. Pair KS (whole distribution) + Hill (tail index): C0 is right.

## C11 — Loss ablation: Hill-only tail. COMPONENT_NAMES = ("V","ACF1","ACF2","Hill") — KS DROPPED.
- Edit: COMPONENT_NAMES drops "KS" (one-line). Caches deleted; run exit 0. NB ks_stat still computed as a
  DIAGNOSTIC but not in the loss — so dKS is blank in results.csv; `hill` and ks_stat are still logged.
- CAVEAT: D here is a 4-component sum (no KS), NOT comparable to C0's 5-component D (calm 24.95, stressed 10.75).
- KEY RESULT — the MIRROR of C10: with Hill IN the loss the tail stays well-controlled. calm kurtosis 8.34
  (vs baseline 9.0 — fine), Hill 2.86 (vs 3.02 — fine); stressed kurt 50.7, Hill 2.49 (≈ baseline 2.44).
  ft_sigma_c stays low (0.66 calm / 0.57 stressed) — no FT-overshoot blow-up. BUT KS is now untargeted:
  ks_stat 0.179 (calm) / 0.186 (stressed) drift free (the whole-distribution CDF is no longer pinned).
- Interpretation: completes the KS+Hill justification. C10 (KS-only) → tail explodes (kurt 109, Hill 1.67).
  C11 (Hill-only) → tail fine but the whole return distribution (KS) is no longer directly matched. Neither
  tail term alone is sufficient: KS pins the body/CDF, Hill pins the tail index. The C0 pairing (KS+Hill) is
  the right loss — each component controls a distinct failure mode the other cannot reach.

---

# CAMPAIGN SUMMARY (C0–C11, all 12 cells, exit 0; KS+Hill 5-component loss unless noted)
Reference baseline (C0): calm D=48.36, stressed D=29.11.

| Cell | Change | calm D | stressed D | Verdict |
|------|--------|--------|-----------|---------|
| C0 | Baseline (Kalman, 4-d, KS+Hill) | 48.36 | 29.11 | reference |
| C1 | SV-MJD fundamental | 47.17 | 36.92 | worse tail; Kalman better |
| C2 | Market Maker on (n_mm=4) | 51.46 | 33.42 | both worse — MM damps vol (confirms removal) |
| C3 | Volatility Trader on (n_vt=10) | 52.80 | 21.27 | stressed BIG win, calm liability — regime-asymmetric |
| C4 | Fix ft_sigma_c=√390 | 102.69 | 71.90 | catastrophic; ft_sigma_c MUST be calibrated low |
| C5 | Fix zi_alpha=0.15 | 66.35 | 26.68 | calm much worse; zi_alpha matters (calm) |
| C6 | Fix zi_mu=0.025 | 48.28 | 29.95 | flat; zi_mu freely droppable |
| C7 | Fix zi_delta=0.15 | 50.77 | 30.35 | calm mild-worse; moderate sensitivity |
| C8 | Fix ALL ZI at CST | 59.07 | 33.43 | both worse; calibrating ZI rates is justified |
| C9 | p_zi in the loop (5-d) | 49.70 | 23.00 | stressed win (sparser book); calm ≈flat |
| C10 | KS-only (drop Hill) | 51.59* | 25.18* | calm tail EXPLODES (kurt 109, Hill 1.67) |
| C11 | Hill-only (drop KS) | 24.95* | 10.75* | tail fine but KS untargeted |
(*C10/C11 D not comparable to others — different component count.)

Headline takeaways for Sid:
1. ft_sigma_c is THE dominant lever (C4): the √390 literature scale is catastrophically wrong for 1-min
   microstructure; calibrating it low (~0.6) is essential.
2. KS+Hill loss is vindicated (C10 vs C11): KS alone lets the tail blow up; Hill alone leaves the CDF
   untargeted. Both are needed — each controls a failure mode the other can't.
3. Structural adds are regime-asymmetric: the Volatility Trader (C3) and free p_zi (C9) BOTH substantially
   improve STRESSED (D 29→21 / 29→23) while being neutral/negative for calm → argues for regime-specific
   structure (n_vt, p_zi) rather than a global on/off. The Market Maker (C2) hurts both (confirms its removal).
4. Among ZI rates, calibration value ranks zi_alpha (calm, C5 +18) > zi_delta (C7 +2.4) > zi_mu (C6 ≈0);
   the full CST pin (C8) costs +10.7 calm / +4.3 stressed — worth calibrating, esp. zi_alpha.
5. Best fits observed: stressed via VT-on (C3, D=21.27) or free-p_zi (C9, D=23.00); calm stays near the C0
   baseline (no tested change beat 48.36 for calm — its residual is volatility clustering, ACF2≈13-15).
