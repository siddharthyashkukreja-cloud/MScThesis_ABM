# Calibration layer — validation and thesis-ready plan

Synthesis of four investigations: Franke & Westerhoff (2012) read start-to-finish + the MCR/
p-value method; the Gao/Lamperti surrogate-SMM lineage read start-to-finish + compared to the
code; a deep `calibrate.py` audit; and a numbers/values validation that **prototyped the MCR in
the sandbox**. Date: 2026-06-15. Code claims are `file:line`; numbers are measured.

> **Current-state update (supersedes the parameter-set specifics below).** The calibrated loop is
> now **7 parameters per regime, the SAME set in BOTH regimes**: `{ft_sigma_c, zi_alpha, zi_delta,
> zi_mu, p_zi, mt_lambda, mt_gamma}` (was calm 4-d `{ft_sigma_c, zi_alpha, zi_delta, zi_mu}` /
> stressed 5-d `{+ p_zi}`). **`p_zi` is now calibrated in BOTH regimes** (calm is no longer pinned at
> the MBP-10 L2-MLE value 0.543) — so **MBP-10 / L2 order-book data is no longer a calibration input
> and `data/p_zi.py` is legacy/optional**; data inputs reduce to 1-min BBO + 1-min OHLCV + daily OHLCV.
> **`mt_lambda` (MT EWMA decay) is now calibrated** (was pinned 0.05) and **`mt_gamma` (MT tanh-activation
> scale) is new and calibrated**. The fundamental `V_t` is the contemporaneous empirical 1-min ES mid
> (the Kalman MLE in `data/v_kalman.py` is retained only as the diagnostic that justifies the raw mid;
> `fv_*.csv` `V_smooth` = the mid exactly). **All calibrated point VALUES below are stale placeholders**
> pending a widened-bounds 7-parameter surrogate re-lock — `globals.CALIBRATED` still holds the old
> 4-d/5-d optima, and calm has no `p_zi`/`mt_lambda`/`mt_gamma` value yet — so treat the specific θ,
> the identifiability spreads, and the demonstrated MCR numbers below as the OLD-structure record. The
> loss composition (`KS + V + ACF1 + ACF2 + Hill`) and the validation methodology (MCR + Franke J-test)
> are unchanged.

---

## 0. Verdict

**The optimisation engine is correct and paper-faithful — in places it exceeds the papers. The
gap to thesis-ready is the reporting/validation layer and three numerical/consistency issues, not
code bugs.** The loss math, the inverse-SD weighting, the XGBoost surrogate mechanics, the grid,
and the offline calibrations (`p_zi`, `v_kalman`) are all sound. What's missing for a defensible
calibration chapter: (1) the Franke/HFABM validation triad (moment **p-values + Moment Coverage
Ratio**) — the headline validation in HFABM §4.3, which the repo omits; (2) the lock silently
overrides its own fresh-seed winner's-curse re-rank; (3) grid and surrogate D are reported as
"agreeing" when they are on different scales and disagree on `ft_sigma_c`; (4) identifiability is
not reported (only `zi_alpha` is sharply identified).

| # | Item | Type | Effort |
|---|------|------|--------|
| P0-1 | Lock ignores its own fresh-seed re-rank (grid argmin is #3 in **both** regimes) | consistency | decide + 1 edit |
| P0-2 | Grid D (3 seeds) vs surrogate D (6 seeds) not comparable (s_KS∝√n_sim) and disagree on ft_sigma_c — stop calling it "agreement" | reporting | re-run at equal n_runs / reword |
| P0-3 | Stale locked-θ everywhere but globals: model.md §5.1 + AGENT.md "Status" show calm 0.95/0.26; live lock is 0.65/0.38 | doc drift | sync |
| P0-4 | `validate_grid_fresh.py` reads the stale top-level CSV (3-d/4-d, no `zi_mu`) → KeyError on re-run | bug | repoint to `output/relock/grid_surface_*.csv` |
| P1-1 | **Add moment p-values + MCR** (Franke 2012 / HFABM §4.3) — prototyped, see §4 | new validation | ~30 lines on existing bootstrap |
| P1-2 | Report identifiability (only `zi_alpha` pinned; ft_sigma_c flat/floor) + surrogate R² + 1-D slices | reporting | from existing surfaces |
| P1-3 | Resolve stressed `ft_sigma_c` sitting on the 0.25 box floor (interior ≈0.5) | numerical | finer/wider grid |
| P2 | Bump bootstrap B→500-1000; fresh-seed seeds→≥4 with CIs; fix stale comments | hygiene | small |

---

## 1. Component-by-component validation

### 1.1 The five loss components — all formula-correct (numerically verified)

`D = KS + V + ACF1 + ACF2 + Hill` (`calibrate.py` `COMPONENT_*`). Each verified against hand-rolled
references to float precision:

- **KS** (`_ks_2samp`, :340) — 2-sample sup|F−F|, correct; target 0 (vs self).
- **V** (`ret_std`, :417) — return std, correct.
- **ACF1** (returns ACF, forward 3-lag smoothed, centres {1,5,10,20}, :355-370) — matches the
  XGB-Chiarella forward-smoothing convention exactly.
- **ACF2** (|r| ACF, :423) — |r| (not r²) is deliberate and cited (r² is outlier-dominated; its
  large bootstrap SD would wash the clustering signal out under standardisation). Sound.
- **Hill** (banded tail index on pooled |r|, :373-402) — correct. One documented subtlety: the
  `k = max(int(frac·n), 20)` floor collapses the band to a point for n ≲ 700; **never binds at sim
  sizes** (n_sim ~ 2.3e4-6.2e4) or on full-length bootstrap resamples, so s_Hill is unaffected —
  but document the floor.
- **Kurtosis** is diagnostic-only (in `MOMENT_NAMES`, in **no** component — verified). Correct, and
  matches the Franke/Gao convention of using Hill rather than kurtosis for the tail.

**Verdict: thesis-ready.** The one *original* design choice — using **both** KS and Hill (XGB-Chiarella
uses KS, HFABM uses Hill; the repo uses both) — is vindicated by the C10/C11 ablation (KS-only → calm
kurtosis explodes, Hill 1.67; Hill-only → body untargeted). Document it as a deliberate improvement.

### 1.2 The weights — inverse-SD block bootstrap (sound, with disclosures)

`empirical_moment_sd` (:674-734): Künsch (1989) moving-block bootstrap (block 390, B=200) → per-moment
SD `s_i`; `compute_grouped_deltas` (:645-671) divides each |sim−hist| by `s_i` (diagonal-SD, L1). This is
the diagonal member of the Franke-Westerhoff family — correctly distinguished from the full `W=Σ⁻¹`,
from HFABM's `1/s_i²` (scale-pathological here: ret_std's ~1e-8 variance would inflate ΔV ~1e8×), and
from XGB-Chiarella's equal weights. **The family choice is well-defended.** Three disclosures:

- **KS dominates the loss** (measured from the relock surfaces): calm ΔKS = 18.82 of D=43.94 (**43%**);
  stressed ΔKS = 3.33 of 5.49 (**61%**). Because `s_KS` is sized to the *sim* sample (KS ∝ 1/√n) while
  the other moments are standardised against full-length resamples, KS is structurally heavier. Keep it
  (KS+Hill is vindicated) but **report the 43%/61% share** rather than implying balanced components.
- **`s_KS` makes D comparable only at a fixed seed count** (verified: s_KS tracks 1/√n_sim almost
  exactly; calm ΔKS 18.82 at 3 seeds → 23.12 at 6). This is the root of P0-2.
- **Block-length comment is stale**: comments cite "longest lag 91", but the longest lag now in the loss
  is **22** (centre 20 + forward 2). Block 390 is still right (= one RTH day ≫ 22); fix the justification.
- **Bump B to 500-1000** for the final lock: on the OLD ~29-day window stressed had only **29 blocks**
  (11,416/390), so its s_i were coarse; calm s_i seed-CV is 4-8% at B=200 (fine). *(The rebuilt full
  stressed window — 28,866 obs / 73 sessions — gives ~74 non-overlapping blocks, so this coarseness is
  much reduced; re-measure at the lock.)*

### 1.3 The surrogate — mechanically faithful; the weakness is the flat surface it fits

`train_surrogate` (XGBoost regressor θ→D, shallow trees + shrinkage; label clip at 90th pct), `_sobol`,
`_refine_round` (2:1 exploit/explore — matches XGB-Chiarella's 200/100), `optimise_surrogate` (pool-argmin,
correct for a piecewise-constant tree), `stage2_grid_search` (Sobol-in-box on the true simulator). All
faithful to Lamperti/Gao. **Measured R²: calm 0.94, stressed 0.76** (stressed surrogate is weaker — why
the grid is the headline there). The real issue is not the code: a high R² over the whole box does **not**
guarantee the argmin is right when the surface is flat — the surrogate puts calm `ft_sigma_c` at **1.09**
(Hill 2.39, tail too heavy) vs the grid's 0.65. This is the D59 lesson ("high R² ≠ reproducible optimum")
and it means the "surrogate agrees on every lever" claim is **false for ft_sigma_c** (they agree on
`zi_alpha` ~0.34-0.38).

### 1.4 The grid — correct headline; report the flatness

`grid_search` (:1056): calm 5⁴=625 nodes, stressed 4⁵=1024, n_runs=3, n_days=20. Exhaustive and
defensible at this dimension (Gao 2023). Two issues: stressed `ft_sigma_c` argmin sits on the **0.25 box
floor** (interior ≈0.5 per the surrogate + D65 probes — re-grid finer/wider, P1-3); and D is on a
different KS scale than the surrogate (P0-2).

### 1.5 Fresh-seed re-rank — sound idea, two real problems

`validate_grid_fresh.py`: re-rank the top-5 grid nodes on fresh common seeds {777, 8888} — the correct
paired winner's-curse design. But: **(a)** it reads the **stale** top-level `output/calibration_grid_*.csv`
(D60-era, 3-d/4-d, no `zi_mu`) → KeyError on re-run; repoint to `output/relock/grid_surface_*.csv` (P0-4).
**(b) The lock does not honour the re-rank**: the grid argmin is **#3 on fresh seeds in both regimes**
(calm fresh-winner `ft_sigma_c=0.50`, D_fresh 46.36 vs argmin's 47.39; stressed fresh-winner `zi_mu=0.104`,
5.99 vs 6.31). `globals.CALIBRATED` keeps the grid argmin. All gaps are ≤1 D (within seed noise), so it's
defensible to keep the argmin — **but only if stated**; silently overriding the re-rank is not (P0-1).
**(c)** Only 2 fresh seed-sets — bump to ≥4 with per-node CIs (P2).

### 1.6 Offline calibrations — clean

`data/p_zi.py` (geometric depth MLE from MBP-10, with a log-normal Δloglik cross-check; Poisson→Bernoulli
cancellation rate) and `data/v_kalman.py` (local-level Kalman MLE, Roll cross-check; `generate` now uses
the raw mid per D67) are both correct and honestly framed. **`data/p_zi.py` is now legacy/optional**:
`p_zi` is calibrated in the loop in both regimes, so the MBP-10 L2 depth is no longer a calibration
input. `v_kalman.py` is now the (diagnostic-only) justification for taking `V_t` = the raw 1-min mid.
No action.

---

## 2. Identifiability and values — the key honest finding

Measured marginal min-D spread per axis (max−min of best-achievable D across that axis's levels):

| param | calm spread | stressed spread | verdict |
|---|---|---|---|
| **zi_alpha** | **28.4** | **9.7** | sharply identified (THE pinned lever, both regimes; argmin 0.38 / 0.32) |
| p_zi | — | 2.36 | moderately identified (stressed; argmin 0.18) |
| ft_sigma_c | 2.59 | 1.04 | calm: **flat** (0.5-1.1); stressed: real but **floor-bound** at 0.25 |
| zi_delta | 1.93 | 1.57 | flat (both on the 0.02 floor) |
| zi_mu | 1.59 | 0.71 | flat |

**Only `zi_alpha` is sharply identified in both regimes.** This is why the three methods disagree on
`ft_sigma_c` (calm 0.65/1.09/0.50; stressed 0.25/0.51/0.25) — the surface is genuinely flat there. The
thesis must **report this** (1-D loss slices around the optimum, à la XGB-Chiarella Fig. 4), not present a
crisp optimum. It is honest, expected (consistent with C4-C8), and pre-empts the obvious examiner question.

Component balance + window artifact (measured): stressed is a genuinely good fit (D=5.49, every non-KS
component < 1.5 SD). Calm D≈44 is mostly KS (18.8) + |r|-ACF2 (10.5); the apparent ret_std/|r|-ACF1
misses are the **window-selection artifact** — the calm sim runs the elevated-vol first-20-sessions of
2019 (empirical ret_std 0.00042, |r|-ACF1 0.225) but is scored against full-year targets (0.00030, 0.296).
Scored on its own window, calm fits well (model matches its own-window moments to 2-3 sig figs).

---

## 3. How to report it — the Franke (2012) MCR + p-value design

The method, from the full read (Franke & Westerhoff 2012, *JEDC* 36; HFABM §4.3 applies it to ES futures):

- **Weighting:** Franke uses the **full** `W = Σ̂⁻¹` (eq 4; Σ̂ = bootstrap covariance, eq A.16). The repo's
  loss is diagonal-L1. **Implication:** for the **MCR this does not matter** (it's a post-hoc, marginal
  per-moment statistic computed on the located θ*); for the **J p-value** compute the quadratic
  `J = (m−m_emp)′ Σ̂⁻¹ (m−m_emp)` at the (diagonally-located) θ* and report it — the estimation criterion
  and the specification-test criterion need not be identical (Franke §5 discusses loss-dependence). Use
  `np.linalg.pinv` + a tiny ridge (Σ̂ is near-singular on correlated ACF lags).
- **Bootstrap:** B = 5000; block 390 (single — justified: max lag 22 ≪ 390; Franke's dual 250/750 blocks
  were for lag-100 moments the repo doesn't have); overlapping Künsch (note vs Franke's non-overlapping).
- **Moments:** drop `ks_stat` (no point-CI) and `ret_kurtosis` (diagnostic) → **K = 10** point moments
  (ret_std, hill, acf_r_{1,5,10,20}, acf_absr_{1,5,10,20}).
- **Two simulation sets** (the part the prototype must separate): **short, T′-length runs** (R≈1000) for
  the J p-value — T′ = the empirical intraday-return count (**calm ≈ 102,500 full-year reference /
  stressed ≈ 28,866** on the rebuilt full window; was ≈ 11,400 on the old ~29-day window); and
  **long runs** (M≈60-100) for the MCR (so model noise is negligible and the MCR isolates *structural* fit).
- **J p-value (eq 9):** critical value = 95th pct of the bootstrap-J null; p = fraction of the MC-J
  distribution above it. Not χ² (deliberately).
- **MCR (Table 4):** per-moment CI = `m_emp ± 1.96·SE_i` (SE_i = the existing bootstrap `s_i`); per-moment
  MCR = % of long runs inside; **joint MCR** = % inside **all** simultaneously; benchmark against the
  **bootstrap self-coverage** (Franke's joint ceiling is **32.6%**, far below 0.95 because the moments are
  correlated; naive-independence = 0.95^K).
- **Reuse `empirical_moment_sd`**: refactor it to return the full B×K bootstrap matrix (one-line opt-in);
  it already builds it. Add a `franke_diagnostics(regime, theta, ...)` + a `calibrate.py franke` CLI verb
  so it re-runs cheaply post-lock without recalibrating.

### 3.1 Demonstrated numbers (sandbox MCR prototype, locked θ, B=2000, M=31)

The prototype (`output/_mcr2.py`, numpy-only — moments reproduce the relock targets to float precision)
already produces sensible MCR tables:

**Stressed — joint MCR 3.2%, mean per-moment coverage 86.8%.** ret_std, Hill, all four return-ACFs, and
|r|-ACF at lags 5/10 cover at **100%**; the single miss is **|r|-ACF1** (model over-clusters, 0.310 vs CI
≤0.293, 9.7%). The joint is low only because that one moment zeroes it. **This is a strong fit.**

**Calm — joint MCR 0%, mean coverage 18.4%** against full-year CIs — but decomposed: ret_std and |r|-ACF1
fall outside **purely from the window artifact** (model matches its own first-20-session window almost
exactly — ret_std 0.00037 vs window 0.00042; |r|-ACF1 0.222 vs window 0.225); the **genuine** gap is
|r|-ACF at lags ≥5 (model ~0.10 vs window ~0.16-0.20 — the documented single-timescale long-horizon
clustering limit, Cont 2005).

**Reporting recommendation:** report MCR **twice for calm** — against full-year CIs (shows the artifact)
and against matched-window CIs (isolates the true gap, lifts coverage substantially); lead with
**per-moment** coverage (joint is brutal by construction — one weak moment zeroes it), and report a joint
over a "stylised-fact core" excluding the known structural-limit moment. MCR is a far harsher, more honest
validation than D (stressed D=5.49 → 87% coverage; calm D=44 → the artifact vs the real gap become visible).

---

## 4. What's missing vs the papers (HFABM is the closest cousin — same asset, ES futures)

The lineage's validation triad is **{surrogate accuracy, Franke p-value, Franke MCR}** (HFABM §4.3). The
repo implements **only the first** (R²) and substitutes a (good, more targeted) fresh-seed re-rank the
papers lack. So the repo is **more rigorous than the papers on optimisation + winner's-curse** (dual
grid+surrogate, fresh-seed re-rank) but **less complete on formal goodness-of-fit reporting**. Add the
p-value + MCR (P1-1) and the calibration goes from "low D" to "statistically not rejected, N% moment
coverage" — the difference between a defensible and an undefensible chapter. Also document the two
deliberate divergences from HFABM (inverse-SD vs 1/variance weighting; KS+Hill dual tail) as evidenced
choices, in the methodology prose (not just code comments).

---

## 5. Thesis-ready checklist

**P0 — fix before locking (consistency/correctness):**
1. Decide the lock vs fresh-seed re-rank: lock the fresh winners (calm `ft_sigma_c=0.50`, stressed
   `zi_mu=0.104`) **or** keep the grid argmin with a one-sentence "within fresh-seed noise (≤1 D)"
   justification (the numbers support it). Make globals / model.md / AGENT.md consistent.
2. Run grid + surrogate at the **same n_runs** for the cross-method table (or normalise ΔKS by √n_sim);
   stop presenting D=43.94 (grid, 3s) as agreeing with the surrogate D (6s). Report that the methods agree
   on `zi_alpha` and disagree on the flat `ft_sigma_c`.
3. Sync the stale locked-θ: model.md §5.1 table and AGENT.md "Status & handoff" still show the D60 lock
   (calm 0.95/0.26); the live lock is the D67 relock (calm 0.65/0.38, D 43.94; stressed D 5.49).
4. Repoint `validate_grid_fresh.py` to `output/relock/grid_surface_*.csv` (the top-level CSVs are stale and
   will KeyError on `zi_mu`).

**P1 — add for thesis-grade validation:**
5. Implement the **moment p-values + MCR** (§3) and report the two per-regime tables (Franke Table 3/4
   style) + a coverage figure, for both Kalman and SV-MJD. The prototype (`output/_mcr2.py`) is the
   starting point — fold in the J p-value, the bootstrap self-coverage ceiling, and the short-T′/long-run
   split it currently omits.
6. Report **identifiability**: 1-D loss slices around the optimum (§2) + the marginal-spread table +
   XGBoost split-frequency importances; disclose that only `zi_alpha` is sharply identified.
7. Resolve the stressed `ft_sigma_c` box-floor (finer/wider grid in [0.25, 0.6]); report the surrogate R²
   (calm 0.94 / stressed 0.76) and flag the low stressed value.

**P2 — hygiene:** bump bootstrap B→500-1000 (report s_i seed-CV); fresh-seed ≥4 seeds with per-node CI;
fix stale comments ("6-d theta", "lag 91"); document the Hill k≥20 floor; per-regime config stamping in
`_merge_save`; remove/regenerate the stale top-level grid CSVs.

**Net:** no loss-math or surrogate bugs. Thesis-readiness = (i) the p-value/MCR reporting layer, (ii) the
four P0 consistency fixes, (iii) honest identifiability reporting + closing the stressed box-floor. P1-1
(p-value + MCR) is the priority — cheap (machinery exists), directly cited (HFABM §4.3 / Franke 2012), and
prototyped.

---

## 6. Sources

Papers (read in full): Franke & Westerhoff 2012 (`extras/references/frank_weights_stylisedfacts.pdf`);
Gao et al. 2022 XGB-Chiarella & HFABM, Lamperti-Roventini-Sani 2018 (`extras/references/{XGB-Chiarella,
HFABM,XGBoostCalibration}.pdf`). Code: `calibrate.py`, `validate_grid_fresh.py`, `data/p_zi.py`,
`data/v_kalman.py`, `model/globals.py` (`CALIBRATED`). Outputs: `output/relock/{grid,sur,grid_freshseed}_*.json`,
`grid_surface_*.csv`. MCR prototype: `output/_mcr2.py` (+ `_ci_*.json`, `_runs_*.csv`). Bib keys for the
methodology: `franke_westerhoff_2012`, `gao_2022_hfabm`, `gao_2022_xgb`, `lamperti_2018`, `kunsch_1989`,
`cont_2001`, `hill_1975` (verify each is in references.bib).
