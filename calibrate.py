"""
calibrate.py — agent-parameter calibration via two-stage surrogate-assisted
SMM (HFABM Gao et al. (2022) §4.2 two-stage workflow with HFABM/Franke & Westerhoff
2012 inverse-bootstrap-variance weights).

**PER-REGIME CALIBRATION (D18b).** Calm and stressed are calibrated as two
INDEPENDENT optimisation problems. Every behavioural parameter is regime-
specific — no shared parameters across regimes. Rationale: the calm/stressed
regimes are economically distinct (different volatility, different book
depth, different microstructure), so forcing a shared parameter to fit both
introduces compromise instead of fit. Independent runs let each regime find
its own optimum.

Methodology (D18c/D18g — HFABM grouping + Franke & Westerhoff per-moment
weighting):
  1. Loss is XGB-Chiarella (Gao et al. 2022 §3.2, eq 6) — the paper's EXACT
     4 grouped components:

         D(θ) = ΔKS(θ) + ΔV(θ) + ΔACF1(θ) + ΔACF2(θ)

     Each component is the mean Franke-STANDARDISED L1 distance over its
     moments — every moment difference |m_sim − m_hist| is divided by that
     moment's empirical block-bootstrap sampling SD s_i before averaging.
     Standardisation makes each component dimensionless (a count of sampling
     SDs), so all four carry equal weight and D is comparable across model
     versions. (NB this is a deliberate improvement on the paper, which uses
     raw equal weights because its four quantities happen to share a scale.)
     The grouped moments (matched to the paper):
         ΔKS    : Kolmogorov-Smirnov 2-sample stat vs the empirical return CDF
                  (eq 7); target 0, /s_KS. Robust whole-distribution fat-tail
                  target. NB s_KS is from full-length resamples while the sim
                  sample is shorter, so ΔKS is somewhat OVER-weighted — watch
                  the per-component line; size-match s_KS if it dominates.
         ΔV     : ret_std
         ΔACF1  : ACF of RETURNS, forward 3-lag avg at centres {1, 10, 20} (§3.2.2)
         ΔACF2  : ACF of SQUARED returns at lags 1..20 (§3.2.3) — vol clustering
         ΔHill  : banded Hill tail index (HFABM §4.1.1) — direct tail lever, paired
                  with KS so both the whole distribution and the tail are matched.
     KURTOSIS stays a diagnostic only (outlier-dominated → matching it is noise-chasing).

  2. Weights: each moment's empirical sampling SD s_i, computed ONCE per
     regime by Künsch (1989) moving-block bootstrap on the historical
     1-min MID log-returns (Franke & Westerhoff 2012; HFABM eq 8). Block
     size 390 (one RTH day) >> the longest ACF lag (91) so re-ordering
     preserves the autocorrelation structure. s_i is fixed across the run
     — the loss is stationary and D(θ) stays comparable when agents change.

  3. Sobol-sample the 6-d behavioural-parameter space (low-discrepancy; beats
     LHS at small N — XGB-Chiarella §3.3.2); simulate each θ on this regime
     (n_runs seeds × n_days days, pooled).
  4. Train a SINGLE XGBoost regressor θ → D(θ) — the scalar loss, not one model
     per moment (XGB-Chiarella §3.3); labels clipped at the LOSS_CLIP_PCTL
     percentile (Remark 2). Held-out R² on D is the proxy-accuracy trust check.
  5. Active-learning refinement (exploration-exploitation, §3.3 Step 4): score
     a Sobol candidate pool with the surrogate; simulate a ~2:1 EXPLOIT (lowest
     predicted D) / EXPLORE (random) mix; append; retrain. Repeat n_refine.
  6. **Stage 1**: POOL ARGMIN — evaluate the D-surrogate over a large Sobol
     pool and take the minimum. A tree ensemble is piecewise-constant, so
     gradient optimisers (L-BFGS-B) are ill-suited → θ*_s.
  7. **Stage 2** (HFABM §4.2): tight Sobol box within ±STAGE2_BOX_FRAC of bound
     width around θ*_s, scored on the TRUE simulator. Pick the simulator-
     evaluated θ with the smallest actual loss as θ*. Corrects for
     surrogate fitting noise.
  8. Validate by re-running the simulator at θ* on FRESH seeds (out-of-sample
     vs the stage-2 selection) and reporting per-moment comparison + grouped
     Δ contributions + total D(θ).

Data-side parameters (V_t GBM σ/μ and v0 from data/v_gbm.py) live in
model/globals.py and auto-populate per regime via ModelParams.__post_init__.
Pinned-structural: ft_sigma_c at √390; qty_max=10;
ft_alpha=mt_alpha=1.0 (D36); mt_lambda=0.05 (D44); mt_mu=0 (D40); mm_qty=2
(D48). The FT has no dead-band (D23); 4 HFABM MMs are live (D48). This script
calibrates **8 behavioural parameters PER REGIME**: ft_alpha, mt_alpha
(FT limit / MT limit activation), mt_mu (MT market-order rate — D27),
depth_mean (shared ZI+MT log-normal placement-depth mean, D20), mt_lambda
(MT EWMA decay — single-type, D13f), and the three ZI rates
zi_alpha / zi_mu / zi_delta (limit / market / cancel — D21). FT/MT limit
orders use replace-on-new order management (D5d). Volatility clustering
and fat tails come from the Merton jumps in the V_t process (D25), not an
agent.

Hard requirements: xgboost (with libomp) and scipy. No fallbacks — if a
required library is missing the run fails loudly. Per-regime LHS training
data is cached to output/calibration_lhs_{regime}.csv; stage-2 grid output
to output/calibration_stage2_{regime}.csv. Delete to regenerate.

Moments: Cont 2001 stylised-fact battery (return std; ACF of returns at
lags {1,5,10}; ACF of |returns| at lags {1,10,30,60,90}) + Hill (1975)
tail index. Computed on the 1-min MID log-returns in both sim and
empirical for an apples-to-apples comparison (excess kurtosis is computed
as a diagnostic but is not a loss moment).

CLI:
  python calibrate.py targets                          # print empirical moment targets
  python calibrate.py run [N D R Rf K G]               # both regimes; defaults below
  python calibrate.py run calm [N D R Rf K G]          # calm only
  python calibrate.py run stressed [N D R Rf K G]      # stressed only
    N=N_LHS, D=N_DAYS, R=N_RUNS, Rf=N_REFINE, K=N_PER_REFINE, G=N_STAGE2
"""

from __future__ import annotations
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

from model.globals import ModelParams, V0, FV_CSV, day_start_steps
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier

# ── config ───────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).parent
OUT_DIR = REPO_DIR / "output"
PROC_DIR = REPO_DIR / "data" / "processed"
REGIME_DATA = {
    "calm":     PROC_DIR / "ES_front_calm_1m.csv",
    "stressed": PROC_DIR / "ES_front_stressed_1m.csv",
}
REGIMES = ("calm", "stressed")
BARS_PER_DAY = 390   # nominal RTH-day length, used only to size the calm
                     # subsample (_sim_steps); the real overnight boundaries are
                     # data-driven via day_start_steps (D57)


@lru_cache(maxsize=None)
def _regime_fv_bars(regime: str) -> int:
    """Bars in the regime's spliced fundamental series (data/fv_{regime}.csv)."""
    return len(pd.read_csv(REPO_DIR / FV_CSV[regime]))


def _sim_steps(regime: str, n_days: int) -> int:
    """Steps to simulate for `regime`. Stressed is a bounded historical episode
    (the 2020 COVID window, ~29 RTH days): simulate the whole series so the sim
    and the empirical targets span the same period, and never run past it — past
    the data Simulation._v_at clamps the fundamental to the last bar (frozen V_t).
    Calm is a long stationary sample, so subsample to n_days."""
    full = _regime_fv_bars(regime)
    return full if regime == "stressed" else min(BARS_PER_DAY * n_days, full)

# Fixed population — the flat ODD star topology (see run_simulation.py).
# 50 agents: 20 FT + 10 MT + 20 ZI, NO MM. The always-quoting MM was removed:
# it clamped spread variability and suppressed volatility clustering (D37 +
# the in-sandbox smoke: stressed lag-1 |r| ACF 0.11 -> 0.44 without it).
# Geometric data-fit placement (globals.P_ZI) supplies near-mid liquidity instead.
POP = dict(n_fundamental=40, n_momentum=20, n_momentum_long=0,
           n_mm=0, n_zi=40, n_vt=0, n_ct=0)   # D53 — clients scaled 2× (30 FT + 20 MT + 40 ZI);
                                              # n_fundamental folds 30 FT + 10 BCM = 40, preserving the
                                              # 40:20:40 FT-equiv:MT:ZI mix (was 20/10/20 = same ratio)

# Per-regime calibration loop (D18b). Every parameter is regime-specific.
# Pinned-structural (NOT calibrated):
#   ft_sigma_c              → √390  (Chiarella one-daily-V_t-std)
#   qty_max                 → QTY_MAX[regime]  (10 calm / 5 stressed)
#   depth_sigma             → 0.3   (log-normal placement-depth shape; D20)
ZI_MU_FIXED = 0.025      # CST-2008 market-order baseline; out of the loop (ablation C6)
PARAM_BOUNDS_BY_REGIME = {
    "calm": {
        "ft_sigma_c": (0.5, 1.1),     # FT belief-width scale (the tail lever; ablation C4).
                                      # Upper bound 10.0 -> 2.0 (D56) -> 1.1 (baseline lock): ft_sigma_c
                                      # is weakly identified in calm (KS+ACF2 dominate the loss), so a
                                      # wide range lets the surrogate drift to an interior fat-tail
                                      # basin (~1.35: Hill delta ~7.4, D~57) while the good optimum
                                      # sits at 0.5-0.9 (D~44). Capping at 1.1 keeps it in that region.
        "zi_alpha":   (0.02, 0.50),   # ZI limit-order arrival per step
        "zi_delta":   (0.05, 0.50),   # ZI per-resting cancellation — floor raised
                                      # 0.005->0.05 (D58): the SOLE order-lifetime lever
                                      # now the LOB TTL is gone; below ~0.05 resting orders
                                      # would accumulate without the backstop.
    },
    "stressed": {
        "ft_sigma_c": (0.5, 2.0),     # tightened 10.0 -> 2.0 (D56), same reasoning as calm —
                                      # the optimum is near the floor (full run found 0.55); a
                                      # concentrated range keeps the short-budget surrogate on it.
        "zi_alpha":   (0.02, 0.50),
        "zi_delta":   (0.05, 0.50),   # floor raised 0.005->0.05 (D58 — see calm)
        "p_zi":       (0.15, 0.80),   # geometric depth — calibrated for stressed only
    },                                # (ablation C9: sparser book sharpens stressed fit)
}
# Loop is 4-d. ft_sigma_c is UNPINNED (was √390≈19.7, D7b "one daily V_t std"):
# the in-sandbox sweep showed it is THE lever on the mid tails. √390 gives
# Hill≈1.5 because the FTs overshoot V_t (they sweep the book to the OUTERMOST
# reservation V_t + max z·σ_fund), fattening the tails and drowning clustering
# in i.i.d. bursts; ft_sigma_c≈1 (σ_fund≈3 ticks) gives Hill≈3.0 AND revives
# long-lag clustering (lag-10 |r| ACF 0.02→0.11) by transmitting V_t faithfully.
# So FT belief dispersion is now a CALIBRATED microstructure-scale quantity, not
# a pinned daily-news scale (supersedes D7b). Trade-off: smaller ft_sigma_c →
# less FT inventory concentration (D6b CCP-layer input) — raise the lower bound
# if that matters more than the market-layer fit. Placement geometric (data-fit
# p_zi); MM removed (n_mm=0). FT/MT still trade every step (ft_alpha=mt_alpha=1).
# Calibrated loop is regime-specific (PARAM_BOUNDS_BY_REGIME): calm 3-d
# {ft_sigma_c, zi_alpha, zi_delta}, stressed 4-d {+ p_zi}. Pinned / out of the loop:
#   zi_mu = ZI_MU_FIXED (0.025, CST-2008; ablation C6 — calibrating it is free).
#   ft_alpha = mt_alpha = 1.0 (D36 — FT/MT submit a limit every step, ODD-
#     faithful §Step Sequence step 3).
#   mt_mu = 0.0 (D40 — MT limit-only; the D27 market branch was reverted —
#     trend-direction market flow corrupted the return ACF).
#   mt_lambda = 0.05 (D44 — pinned; smoke beat the calibrator at this arch).
#   mm_qty = 2 (D48 — pinned structural; `mm_p_edge` is the calibrated MM dial);
#     n_mm = 4 HFABM mid-anchored MMs re-introduced (D48) for tail control.
#   ft_delta/mt_delta (D5d — replace-on-new); k_base (D20 — shared log-normal
#     depth); vt_*/ct_* (D40/D44 — VolatilityTrader & ContTrader removed).
# D30 — ZI rate bounds tightened to literature-grounded ranges. The prior
# (→1.0) bounds let the optimiser run zi_mu to 0.52 / zi_alpha to 0.83 —
# ~20x / ~5x the Cont-Stoikov-Talreja 2008 / ODD §Calibration baselines (0.025 /
# 0.15). zi_mu ≈ 0.5 means half of all ZI activity is book-walking market
# orders → kurtosis ~1200 and a bid-ask bounce. The new caps keep market
# orders a clear minority of ZI flow: zi_mu ≤ 0.10 (4x baseline), zi_alpha
# ≤ 0.50 (3.3x baseline). zi_delta (cancellation) ∈ [0.05, 0.50] — lower bound
# raised from 0.005 (D58: sole order-lifetime control now the LOB TTL is removed).
# ── Campaign experiment flags (overnight calibration campaign) ────────────────
# Each is OFF by default, so absent any env var PARAM_BOUNDS_BY_REGIME is the
# baseline (E0) loop. Gated at import time — every experiment is a fresh process
# with its own environment, so import-time injection is clean and isolated.
#   E2  MT_LAMBDA_IN_LOOP  — add mt_lambda (0.004, 0.20) to both regimes' loops.
#   E5  FTMT_GATES         — add ft_alpha/mt_alpha/ft_delta/mt_delta (high-dim).
if os.environ.get("MT_LAMBDA_IN_LOOP"):
    for _rb in PARAM_BOUNDS_BY_REGIME.values():
        _rb["mt_lambda"] = (0.004, 0.20)   # ~3.5-min (0.20) to ~3-hour (0.004) half-life
if os.environ.get("FTMT_GATES"):
    for _rb in PARAM_BOUNDS_BY_REGIME.values():
        _rb["ft_alpha"] = (0.2, 1.0)
        _rb["mt_alpha"] = (0.2, 1.0)
        _rb["ft_delta"] = (0.0, 0.5)
        _rb["mt_delta"] = (0.0, 0.5)

# Active regime's parameter set. _activate_regime() swaps these at the top of
# each regime's run — calibration is sequential (one regime at a time), so
# module-level state is safe. Default to calm for imports / other tools.
PARAM_KEYS: list = list(PARAM_BOUNDS_BY_REGIME["calm"])
PARAM_BOUNDS_ARR = np.array([PARAM_BOUNDS_BY_REGIME["calm"][k] for k in PARAM_KEYS])


def _activate_regime(regime: str) -> None:
    """Point PARAM_KEYS / PARAM_BOUNDS_ARR at `regime`'s parameter set (calm 3-d,
    stressed 4-d with p_zi)."""
    global PARAM_KEYS, PARAM_BOUNDS_ARR
    PARAM_KEYS = list(PARAM_BOUNDS_BY_REGIME[regime])
    PARAM_BOUNDS_ARR = np.array([PARAM_BOUNDS_BY_REGIME[regime][k] for k in PARAM_KEYS])

# Individual moments — surrogate targets and loss inputs (D18e, revised). ACF
# lags sit where the empirical ES 1-min signal actually lives: the return ACF is
# concentrated at the SHORT end (a bid-ask/microstructure term at lag 1 and a
# transient-impact mean-reversion peaking near lag ~5-10), and flat (~0) by lag
# 30+, so high lags carry no information.
# Loss D(θ) — FIVE standardised-moment components, matched to XGB-Chiarella
# (Gao et al. 2022 §3.2) + HFABM (§4.1.1):
#   KS   = Kolmogorov-Smirnov 2-sample statistic between simulated and empirical
#          return CDFs (paper §3.2.4, eq 7) — robust whole-distribution fat-tail target.
#   V    = return-standard-deviation distance (paper eq 8).
#   ACF1 = returns ACF at centres {1, 5, 10, 20}, FORWARD 3-lag smoothed
#          (paper §3.2.2: lag-c = mean of {c, c+1, c+2}).
#   ACF2 = ABSOLUTE-returns ACF, forward 3-lag smoothed at {1, 5, 10, 20} — the
#          volatility-clustering target. |r| (Cont 2001) is used rather than the
#          paper's r²: r² is outlier-dominated, so its ACF has a large sampling SD
#          and the clustering miss vanishes under the standardisation; |r| is far
#          less noisy, so clustering actually counts.
#   Hill = banded Hill tail index (HFABM §4.1.1). KS + Hill together carry the tail:
#          KS the whole-distribution match, Hill a direct, reachable tail-index lever.
# ret_kurtosis stays a DIAGNOSTIC only — outlier-dominated, so matching it exactly
# would be noise-chasing.
ACF1_CENTERS = (1, 5, 10, 20)           # returns ACF, forward 3-lag smoothed (5-min lag added)
ACF2_CENTERS = (1, 5, 10, 20)           # |returns| ACF centres, forward 3-lag smoothed
HILL_FRACS = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)   # banded-Hill k/n grid
HILL_FRAC = 0.05                        # single-frac default (band primitive)

ACF1_NAMES = tuple(f"acf_r_{c}" for c in ACF1_CENTERS)
ACF2_NAMES = tuple(f"acf_absr_{c}" for c in ACF2_CENTERS)
MOMENT_NAMES = (["ret_std", "ret_kurtosis"]
                + list(ACF1_NAMES) + list(ACF2_NAMES)
                + ["hill_tail_index", "ks_stat"])

COMPONENT_NAMES = ("KS", "V", "ACF1", "ACF2", "Hill")   # KS + banded Hill carry the tail
COMPONENT_MOMENTS = {
    "KS":   ("ks_stat",),            # ΔKS   (XGB-Chiarella §3.2.4 eq 7; target 0, /s_KS)
    "V":    ("ret_std",),            # ΔV     (paper eq 8)
    "ACF1": ACF1_NAMES,              # ΔACF1  (paper eq 9: returns, lags 1/10/20)
    "ACF2": ACF2_NAMES,              # ΔACF2  (|returns| ACF, short lags — clustering)
    "Hill": ("hill_tail_index",),    # ΔHill  (HFABM §4.1.1 tail index; direct tail lever)
}

# Block bootstrap parameters for the per-moment sampling SDs. Block size
# 390 (one RTH day) >> the longest ACF lag (91) — Künsch (1989) moving-
# block bootstrap requires the block to exceed the dependence horizon, or
# block re-ordering destroys the long-lag autocorrelation it is meant to
# preserve (HFABM uses block 1800 >> lag 90 for the same reason; the prior
# block 60 was SHORTER than the lag-90 moment — a bug). 200 resamples.
N_BOOTSTRAP = 200
BOOTSTRAP_BLOCK = 390

# Defaults — MEDIUM budget, sizes are powers of 2 so the Sobol design is
# balanced. The 40-sample runs gave weak/negative held-out surrogate R² (the
# optimiser was near-blind), so this raises N for a learnable surrogate;
# ~1-1.5 h/regime. Quicker iteration: `run 64 10 3 1 16 16`. Thesis-final:
# `run 256 30 8 3 64 64`. The LHS cache auto-regenerates on a param/moment change.
N_LHS = 128
N_DAYS = 20
N_RUNS = 4
N_REFINE = 2
N_PER_REFINE = 32
N_CANDIDATE_POOL = 2048
N_STAGE2 = 32          # HFABM stage-2 refinement size (Sobol in tight box)
STAGE2_BOX_FRAC = 0.10 # box half-width = STAGE2_BOX_FRAC · (hi - lo) per param
SEED = 42
TEST_FRAC = 0.25
LOSS_CLIP_PCTL = 90    # label-clip percentile for the single-D surrogate — XGB-
                       # Chiarella Remark 2 (focus the tree on the low-D region;
                       # the paper clips D to (0,1], we clip at a data percentile
                       # since our D is in sampling-SD units, not their 0-1 scale)

XGB_KWARGS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.9, random_state=0)


# ── moments ──────────────────────────────────────────────────────────────────

# Moments are plain dicts keyed by MOMENT_NAMES — the moment set is now
# programmatic (20 squared-return ACF lags), so the fixed dataclass is gone.
# The xgboost surrogate reads the LHS DataFrame columns, not a Moments object,
# so this change is confined to the moment/loss layer.

def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup_x |F_a(x) - F_b(x)| (XGB-
    Chiarella eq 7) — the distance between the simulated and empirical return
    CDFs; a robust whole-distribution fat-tail measure that complements Hill.
    Pure-numpy (no scipy) so it is unit-testable in any environment."""
    a = np.sort(np.asarray(a, float)); a = a[np.isfinite(a)]
    b = np.sort(np.asarray(b, float)); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _acf(x: np.ndarray, k: int) -> float:
    if len(x) <= k:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).sum())
    return 0.0 if var == 0.0 else float((x[:-k] * x[k:]).sum() / var)


def _acf_smoothed_fwd(x: np.ndarray, center_lag: int) -> float:
    """FORWARD 3-lag smoothing (XGB-Chiarella §3.2.2 / Majewski): the mean of
    the autocorrelations at lags {center, center+1, center+2} — e.g. lag-1 is
    the mean of {1,2,3}. Matches the paper (replaces the prior centred form)."""
    lags = (center_lag, center_lag + 1, center_lag + 2)
    vals = [_acf(x, l) for l in lags]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _hill_estimator(returns: np.ndarray, frac: float = HILL_FRAC) -> float:
    """Hill (1975) tail-index estimator on |returns|. Returns the index α
    such that P(|R| > x) ~ x^(−α) in the upper tail — larger α = lighter
    tail, smaller α = heavier. Pools both tails via the absolute value
    (Resnick 2007 §4)."""
    a = np.abs(np.asarray(returns, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 100:
        return float("nan")
    k = max(int(frac * n), 20)
    if k >= n:
        return float("nan")
    a_sorted = np.sort(a)[::-1]
    top_k = a_sorted[:k]
    threshold = a_sorted[k]
    if threshold <= 0:
        return float("nan")
    xi = float(np.mean(np.log(top_k) - np.log(threshold)))
    return float(1.0 / xi) if xi > 0 else float("nan")


def _hill_banded(returns: np.ndarray, fracs=HILL_FRACS) -> float:
    """Banded Hill index — mean of the single-frac estimator over k/n in
    `fracs`. The 1-min ES Hill curve slopes (no plateau), so a single 5% point
    is fragile and k-sensitive; averaging a band gives a robust, reproducible
    tail index (still comparable across versions — the fracs are fixed)."""
    vals = [_hill_estimator(returns, f) for f in fracs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_moments(log_returns: np.ndarray) -> dict:
    """Moment dict keyed by MOMENT_NAMES from a 1-min log-return series.
    ACF1 = returns ACF (forward 3-lag smoothed at centres ACF1_CENTERS);
    ACF2 = |returns| ACF (forward 3-lag smoothed at centres ACF2_CENTERS); Hill banded.
    `ks_stat` is left NaN here — it is a 2-sample statistic the caller fills
    against the empirical returns (simulate_moments / empirical_moment_sd); the
    empirical target sets it to 0. Needs >= 100 obs (covers lag 22)."""
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return {m: float("nan") for m in MOMENT_NAMES}
    sd = float(r.std())
    kurt = float((((r - r.mean()) / sd) ** 4).mean() - 3.0) if sd > 0 else 0.0
    a = np.abs(r)
    out = {"ret_std": sd, "ret_kurtosis": kurt,
           "hill_tail_index": _hill_banded(r), "ks_stat": float("nan")}
    for c in ACF1_CENTERS:
        out[f"acf_r_{c}"] = _acf_smoothed_fwd(r, c)        # returns ACF
    for c in ACF2_CENTERS:
        out[f"acf_absr_{c}"] = _acf_smoothed_fwd(a, c)     # |returns| ACF (clustering)
    return out


_EMP_RETURNS_CACHE: dict = {}


def _empirical_returns(regime: str) -> np.ndarray:
    """Empirical ES 1-min MID log-returns for a regime, overnight (cross-day)
    returns dropped. Cached. Serves as both the KS reference sample and the
    moment-target source — the mid matches the simulator's own observable."""
    if regime not in _EMP_RETURNS_CACHE:
        df = pd.read_csv(REGIME_DATA[regime], index_col=0, parse_dates=True)
        mid = df["mid"].to_numpy(dtype=float)
        logret = np.diff(np.log(mid))
        dates = pd.DatetimeIndex(df.index).date
        _EMP_RETURNS_CACHE[regime] = logret[dates[1:] == dates[:-1]]
    return _EMP_RETURNS_CACHE[regime]


def empirical_targets() -> dict:
    """Moment-dict targets of the empirical ES 1-min MID log-returns, per
    regime (overnight returns dropped). `ks_stat` target is 0 — the empirical
    distribution's KS distance from itself — so the loss measures the sim's KS
    distance from empirical in s_KS units."""
    targets = {}
    for regime in REGIME_DATA:
        m = compute_moments(_empirical_returns(regime))
        m["ks_stat"] = 0.0
        targets[regime] = m
    return targets


# ── simulator wrapper ────────────────────────────────────────────────────────

def _theta_to_params(theta: np.ndarray, regime: str) -> ModelParams:
    """Build a ModelParams for one regime from the active theta (calm 3-d:
    ft_sigma_c + zi_alpha/zi_delta; stressed 4-d: + p_zi). zi_mu is pinned
    (ZI_MU_FIXED); p_zi passes through only when calibrated (stressed), else it
    falls back to the L2/MBP-10 P_ZI[regime] data-fix. Other structure auto-
    populates from POP + globals."""
    d = dict(zip(PARAM_KEYS, theta))
    kw = dict(
        ft_sigma_c=float(d["ft_sigma_c"]),
        zi_alpha=float(d["zi_alpha"]),
        zi_mu=ZI_MU_FIXED,
        zi_delta=float(d["zi_delta"]),
    )
    if "p_zi" in d:                       # stressed only; calm keeps the L2 P_ZI
        kw["p_zi"] = float(d["p_zi"])
    # E2 — mt_lambda calibrated in the loop (MT_LAMBDA_IN_LOOP).
    if "mt_lambda" in d:
        kw["mt_lambda"] = float(d["mt_lambda"])
    # E3/E4 — mt_lambda pinned externally (MT_LAMBDA_FIXED, e.g. 0.00385 = 3h half-life).
    if os.environ.get("MT_LAMBDA_FIXED"):
        kw["mt_lambda"] = float(os.environ["MT_LAMBDA_FIXED"])
    # E4b — long-cohort EWMA decay (MT_LAMBDA_LONG); build_traders wires n_momentum_long.
    if os.environ.get("MT_LAMBDA_LONG"):
        kw["mt_lambda_long"] = float(os.environ["MT_LAMBDA_LONG"])
    # E5 — FT/MT Bernoulli gate + stochastic cancellation, all calibrated (FTMT_GATES).
    for g in ("ft_alpha", "mt_alpha", "ft_delta", "mt_delta"):
        if g in d:
            kw[g] = float(d[g])
    # Population: POP (bare market) by default. E1 swaps to the run_simulation
    # population WITH clearing members (CLEARING_IN_LOOP); E4 overrides MT counts.
    pop = dict(POP)
    if os.environ.get("CLEARING_IN_LOOP"):
        pop = dict(n_fundamental=30, n_momentum=20, n_momentum_long=0,
                   n_mm=0, n_zi=40, n_vt=0, n_ct=0,
                   n_bcm=10, n_nbcm=5, n_bcm_with_clients=5)
    if os.environ.get("N_MOMENTUM"):
        pop["n_momentum"] = int(os.environ["N_MOMENTUM"])
    if os.environ.get("N_MOMENTUM_LONG"):
        pop["n_momentum_long"] = int(os.environ["N_MOMENTUM_LONG"])
    return ModelParams(
        **pop,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        **kw,
        stressed=(regime == "stressed"),
    )


def _intraday_logret(mid: np.ndarray, regime: str) -> np.ndarray:
    """1-min log-returns with the cross-day (overnight) returns dropped. The sim
    opens each RTH day at the gapped V_t via a session reset (D56), so the
    day-boundary return is an overnight gap; excluding it matches the empirical
    convention (_empirical_returns drops cross-day returns), keeping the
    calibration moments intraday on both sides. Boundaries are the real session
    opens (D57: `day_start_steps`), since a session is ~405 bars and varies, not
    a fixed 390 — the return into open row d is r[d-1]."""
    r = np.diff(np.log(mid))
    drop = [d - 1 for d in day_start_steps(regime) if 0 < d <= len(r)]
    return np.delete(r, drop) if drop else r


def simulate_moments(theta: np.ndarray, regime: str,
                     n_days: int, n_runs: int, seed: int) -> dict:
    """Run the simulator n_runs times on one regime; return the moment dict of
    the pooled 1-min mid log-returns (cross-day returns dropped — D56). `ks_stat`
    is the KS distance of the pooled simulated returns from the empirical returns
    (filled here — it needs both samples)."""
    params = _theta_to_params(theta, regime)
    n_steps = _sim_steps(regime, n_days)
    rets = []
    clearing = bool(os.environ.get("CLEARING_IN_LOOP"))   # E1 — calibrate with the CCP tier active
    for s in range(seed, seed + n_runs):
        traders = build_traders(params, seed=s)
        ccp = build_clearing_tier(traders, params, seed=s) if clearing else None
        hist = Simulation(params, traders, seed=s, ccp=ccp).run(n_steps)
        mid = pd.Series(hist["mid_price"]).ffill().bfill().to_numpy()
        mid = mid[mid > 0]
        if len(mid) > 1:
            rets.append(_intraday_logret(mid, regime))
    if not rets:
        return {m: float("nan") for m in MOMENT_NAMES}
    pooled = np.concatenate(rets)
    m = compute_moments(pooled)
    m["ks_stat"] = _ks_2samp(pooled, _empirical_returns(regime))
    return m


# ── Latin-hypercube sampling ─────────────────────────────────────────────────

def _lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube unit-cube sample (fallback when scipy.stats.qmc absent)."""
    u = np.zeros((n_samples, n_dims))
    strata = np.arange(n_samples) / n_samples
    for d in range(n_dims):
        col = strata + rng.uniform(0, 1.0 / n_samples, size=n_samples)
        rng.shuffle(col)
        u[:, d] = col
    return u


def _sobol(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Low-discrepancy unit-cube sample — Sobol (XGB-Chiarella §3.3.2 Step 1).
    Sobol beats LHS on uniformity at SMALL N (Kucherenko et al. 2015), which is
    the thesis regime: the LOB ABM is far costlier per eval than the paper's
    1-agent price-impact model, so we cannot match its 16384-point pool and a
    well-spread small design matters more. Scrambled for randomisation; falls
    back to LHS if scipy.stats.qmc is unavailable."""
    try:
        import warnings
        from scipy.stats import qmc
        seed = int(rng.integers(0, 2 ** 31 - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # n need not be 2^k — balance is adequate here
            return qmc.Sobol(d=n_dims, scramble=True, seed=seed).random(n_samples)
    except Exception:
        return _lhs(n_samples, n_dims, rng)


def run_lhs(regime: str, n_lhs: int, n_days: int, n_runs: int,
            seed: int = SEED) -> pd.DataFrame:
    """LHS over the 6-d theta space; simulate this regime for each sample
    and record the moment vector."""
    rng = np.random.default_rng(seed)
    u = _sobol(n_lhs, len(PARAM_KEYS), rng)
    thetas = PARAM_BOUNDS_ARR[:, 0] + u * (PARAM_BOUNDS_ARR[:, 1] - PARAM_BOUNDS_ARR[:, 0])
    rows = []
    for i, theta in enumerate(thetas):
        row = {k: float(v) for k, v in zip(PARAM_KEYS, theta)}
        m = simulate_moments(theta, regime, n_days, n_runs, seed=seed + i)
        for name in MOMENT_NAMES:
            row[name] = float(m[name])
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  LHS {i + 1}/{n_lhs}", flush=True)
    return pd.DataFrame(rows)


# ── XGBoost surrogate ────────────────────────────────────────────────────────

def _xgb() -> "xgb.XGBRegressor":
    return xgb.XGBRegressor(**XGB_KWARGS)


def _loss_column(lhs_df: pd.DataFrame, target: dict, sds: dict) -> np.ndarray:
    """Scalar loss D(θ) for every LHS row, assembled from the stored moment
    columns. This is the SINGLE surrogate target (XGB-Chiarella §3.3 regresses
    θ -> D, not per-moment). Recomputed each run from the current target/sds, so
    the cached moment table need not store D."""
    out = np.full(len(lhs_df), np.nan)
    for i in range(len(lhs_df)):
        m = {k: float(lhs_df.iloc[i][k]) for k in MOMENT_NAMES if k in lhs_df.columns}
        d = _true_loss(m, target, sds)
        out[i] = d if np.isfinite(d) else np.nan
    return out


def train_surrogate(lhs_df: pd.DataFrame, target: dict, sds: dict,
                    test_frac: float = TEST_FRAC, seed: int = 0,
                    clip_pctl: float = LOSS_CLIP_PCTL):
    """SINGLE XGBoost regressor θ -> D(θ) — the scalar stylised-facts distance,
    not one model per moment (XGB-Chiarella Gao et al. 2022 §3.3). A single
    smooth surface is more learnable than 20+ noisy per-lag moments, and the
    held-out R² on D IS the proxy-accuracy check the paper relies on. Labels are
    clipped at the `clip_pctl` percentile (Remark 2) so a few huge-D outliers
    don't bias the tree toward the bad region. Returns (model, info)."""
    y = _loss_column(lhs_df, target, sds)
    X = lhs_df[PARAM_KEYS].to_numpy()
    ok = np.isfinite(y)
    info = {"n": int(ok.sum()), "cap": float("nan"),
            "r2": float("nan"), "corr": float("nan")}
    if ok.sum() < 20:
        return None, info
    Xv, yv = X[ok], y[ok]
    cap = float(np.percentile(yv, clip_pctl))
    yc = np.minimum(yv, cap)
    info["cap"] = cap
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(yc))
    n_test = max(3, int(test_frac * len(yc)))
    te, tr = perm[:n_test], perm[n_test:]
    if len(tr) >= 20 and len(te) >= 3:
        probe = _xgb(); probe.fit(Xv[tr], yc[tr])
        pred = probe.predict(Xv[te])
        ss_res = float(((yc[te] - pred) ** 2).sum())
        ss_tot = float(((yc[te] - yc[te].mean()) ** 2).sum())
        info["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        info["corr"] = (float(np.corrcoef(yc[te], pred)[0, 1])
                        if len(te) > 1 else float("nan"))
    model = _xgb(); model.fit(Xv, yc)
    return model, info


def compute_grouped_deltas(sim: dict, hist: dict, sds: dict) -> dict:
    """Grouped, STANDARDISED-MOMENT distances. Each moment's |sim - hist| is
    divided by that moment's empirical block-bootstrap sampling SD s_i, making
    it a dimensionless count of sampling SDs (a z-statistic); the component
    distance Δ_c is the mean over component c's moments, and D = Σ_c Δ_c (L1).

    NB on the weighting (for the write-up): this is INVERSE-SD standardisation —
    the diagonal-SD / L1 member of the Franke & Westerhoff (2012) inverse-
    sampling-variability family. It is NOT the full quadratic inverse-covariance
    form W=Σ⁻¹ (ill-conditioned on one 24-moment sample), NOT HFABM's inverse-
    VARIANCE 1/s_i² (scale-pathological here — ret_std's ~1e-8 sampling variance
    inflates ΔV ~1e8× and collapsed the tail fit; observed, D18g), and NOT the
    XGB-Chiarella equal weights (raw moments span ~4 orders of magnitude, so
    equal weights bury ΔV). Inverse-SD is the principled middle that makes
    heterogeneous moments commensurate and keeps D comparable across model
    versions (s_i and the targets are fixed empirical quantities)."""
    out = {}
    for comp, moments in COMPONENT_MOMENTS.items():
        diffs = []
        for m in moments:
            s = sim.get(m, float("nan"))
            h = hist.get(m, float("nan"))
            sd = sds.get(m, float("nan"))
            if np.isfinite(s) and np.isfinite(h) and np.isfinite(sd) and sd > 0:
                diffs.append(abs(s - h) / sd)
        out[comp] = float(np.mean(diffs)) if diffs else float("nan")
    return out


def empirical_moment_sd(regime: str,
                        n_days: int = N_DAYS, n_runs: int = N_RUNS,
                        n_bootstrap: int = N_BOOTSTRAP,
                        block_size: int = BOOTSTRAP_BLOCK,
                        seed: int = 0) -> dict:
    """Per-moment empirical sampling SD s_i (Franke & Westerhoff 2012
    weighting) via Künsch (1989) moving-block bootstrap on the historical
    1-min MID log-returns. For each of `n_bootstrap` block resamples the
    moment vector is recomputed; s_i is the SD of moment i across the
    resamples — how much moment i wanders by historical sampling alone.

    These s_i are the fixed weights of the calibration loss: each moment
    distance is divided by s_i (compute_grouped_deltas). The block size
    (390 = one RTH day) exceeds the longest ACF lag (91) so block
    re-ordering preserves the autocorrelation structure (Künsch 1989).
    Computed ONCE per regime and held fixed across all LHS / AL / stage-2 /
    optimisation steps so the loss is stationary AND comparable across runs.

    Returns dict keyed by MOMENT_NAMES; values are s_i. Moments with
    degenerate (zero/NaN) bootstrap SD are returned NaN and skipped by the
    loss."""
    rng = np.random.default_rng(seed)
    path = REGIME_DATA[regime]
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    logret = np.diff(np.log(mid))
    dates = pd.DatetimeIndex(df.index).date
    keep = dates[1:] == dates[:-1]                 # drop open-bar returns
    logret = logret[keep]
    n = len(logret)
    if n < 2 * block_size:
        return {m: float("nan") for m in MOMENT_NAMES}

    n_blocks = max(1, n // block_size)
    # KS is a TWO-sample distance whose null sampling SD scales with the SIM
    # sample size (KS ~ 1/sqrt(n_eff)), NOT the full history — so estimate s_KS
    # from a SIM-SIZED block subsample vs the full history. Full-length resamples
    # (as used for the single-sample moments) understated s_KS and let ΔKS
    # dominate the loss (~28-33 SDs). n_sim = pooled simulated return count.
    n_sim = _sim_steps(regime, n_days) * n_runs
    n_blocks_ks = max(1, min(n_blocks, n_sim // block_size))
    samples = {m: [] for m in MOMENT_NAMES}
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        resampled = np.concatenate([logret[s:s + block_size] for s in starts])
        b = compute_moments(resampled)
        starts_ks = rng.integers(0, n - block_size + 1, size=n_blocks_ks)
        resampled_ks = np.concatenate([logret[s:s + block_size] for s in starts_ks])
        b["ks_stat"] = _ks_2samp(resampled_ks, logret)   # sim-sized vs full hist
        for m in MOMENT_NAMES:
            samples[m].append(float(b[m]))
    sds = {}
    for m in MOMENT_NAMES:
        vals = np.asarray(samples[m])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            sds[m] = float("nan")
            continue
        sd = float(vals.std(ddof=1))
        sds[m] = sd if sd > 1e-300 else float("nan")
    return sds


def optimise_surrogate(model, seed: int = 7, n_pool: int = N_CANDIDATE_POOL) -> tuple:
    """Stage-1 optimum by POOL ARGMIN (XGB-Chiarella §3.3 Steps 3-4): evaluate
    the single D-surrogate over a large Sobol candidate pool and take the
    minimum. A gradient-boosted tree is piecewise-constant, so gradient
    optimisers (L-BFGS-B) are ill-suited — zero/garbage gradients between
    splits; dense pool evaluation is faithful and well-matched to a tree
    surrogate. Returns (theta_star, predicted_D_star)."""
    if model is None:
        raise RuntimeError(
            "surrogate untrained (too few finite-loss LHS samples) — "
            "increase N_LHS / N_PER_REFINE.")
    rng = np.random.default_rng(seed)
    u = _sobol(n_pool, len(PARAM_KEYS), rng)
    pool = PARAM_BOUNDS_ARR[:, 0] + u * (PARAM_BOUNDS_ARR[:, 1] - PARAM_BOUNDS_ARR[:, 0])
    preds = np.asarray(model.predict(pool), dtype=float)
    j = int(np.argmin(preds))
    return pool[j], float(preds[j])


def _true_loss(moments: dict, target: dict, sds: dict) -> float:
    """Franke-standardised grouped-L1 loss D(θ) on actual simulated moments.
    This is the scalar the single surrogate regresses (via _loss_column) and the
    score used at stage-2 and validation (no surrogate prediction here)."""
    sim = {m: float(moments[m]) for m in MOMENT_NAMES}
    hist = {m: float(target[m]) for m in MOMENT_NAMES}
    deltas = compute_grouped_deltas(sim, hist, sds)
    total = 0.0
    for c in COMPONENT_NAMES:
        d = deltas.get(c, float("nan"))
        if np.isfinite(d):
            total += d
    return total


def stage2_grid_search(theta_surrogate: np.ndarray, regime: str,
                       target: dict, sds: dict, n_grid: int,
                       n_days: int, n_runs: int, seed: int,
                       box_frac: float = STAGE2_BOX_FRAC) -> tuple:
    """HFABM Gao et al. (2022) §4.2 stage-2 refinement. The surrogate-derived θ* is
    only approximate (surrogate has finite R²); HFABM follows the surrogate
    optimum with a numerical grid search over a feasible bounded set centered
    on θ*, scoring on the TRUE simulator. We use an LHS within a tight box
    `[θ* − box_frac·width, θ* + box_frac·width]` (clipped to the global
    parameter bounds), plus θ* itself as a candidate.

    Returns (theta_best, true_loss_best, dataframe_of_all_candidates) where
    `dataframe_of_all_candidates` has one row per evaluated θ with its
    moments and the `true_loss` it achieved."""
    rng = np.random.default_rng(seed)
    lo, hi = PARAM_BOUNDS_ARR[:, 0], PARAM_BOUNDS_ARR[:, 1]
    width = hi - lo
    box_lo = np.maximum(lo, theta_surrogate - box_frac * width)
    box_hi = np.minimum(hi, theta_surrogate + box_frac * width)
    u = _sobol(n_grid, len(PARAM_KEYS), rng)
    thetas = box_lo + u * (box_hi - box_lo)
    # Include the surrogate optimum itself as a candidate (rank-0 entry).
    thetas = np.vstack([theta_surrogate[None, :], thetas])

    rows = []
    best_loss, best_theta = float("inf"), None
    for i, theta in enumerate(thetas):
        m = simulate_moments(theta, regime, n_days, n_runs,
                             seed=seed + i)
        loss = _true_loss(m, target, sds)
        row = {k: float(v) for k, v in zip(PARAM_KEYS, theta)}
        for name in MOMENT_NAMES:
            row[name] = float(m[name])
        row["true_loss"] = float(loss)
        rows.append(row)
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_theta = float(loss), theta.copy()
        if (i + 1) % 10 == 0:
            print(f"    stage-2 sim {i + 1}/{len(thetas)}  "
                  f"(best so far {best_loss:.4f})", flush=True)
    return best_theta, best_loss, pd.DataFrame(rows)


# ── active learning ─────────────────────────────────────────────────────────

def _refine_round(regime: str, model, target: dict, sds: dict,
                  n_per: int, n_days: int, n_runs: int, seed: int,
                  candidate_pool: int = N_CANDIDATE_POOL) -> pd.DataFrame:
    """One active-learning round (XGB-Chiarella §3.3 Step 4 exploration-
    exploitation): score a Sobol candidate pool with the D-surrogate, then run
    the simulator on a ~2:1 mix of EXPLOIT (lowest predicted D) and EXPLORE
    (random) points — the paper's 200/100 split, scaled to n_per. Returns rows
    to append to the regime's training set. (target/sds are kept in the
    signature for interface symmetry; the model already encodes the loss.)"""
    rng = np.random.default_rng(seed)
    lo, hi = PARAM_BOUNDS_ARR[:, 0], PARAM_BOUNDS_ARR[:, 1]
    cand = lo + _sobol(candidate_pool, len(PARAM_KEYS), rng) * (hi - lo)
    n_exploit = max(1, int(round(2.0 / 3.0 * n_per)))
    n_explore = max(0, n_per - n_exploit)
    if model is not None:
        exploit_idx = np.argsort(np.asarray(model.predict(cand), dtype=float))[:n_exploit]
    else:
        exploit_idx = rng.choice(len(cand), n_exploit, replace=False)
    remaining = np.setdiff1d(np.arange(len(cand)), exploit_idx)
    explore_idx = (rng.choice(remaining, min(n_explore, len(remaining)), replace=False)
                   if len(remaining) else np.array([], dtype=int))
    pick = np.concatenate([np.asarray(exploit_idx), np.asarray(explore_idx)]).astype(int)
    rows = []
    for i, j in enumerate(pick):
        theta = cand[j]
        row = {k: float(v) for k, v in zip(PARAM_KEYS, theta)}
        m = simulate_moments(theta, regime, n_days, n_runs, seed=seed + i)
        for name in MOMENT_NAMES:
            row[name] = float(m[name])
        rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"    refine sim {i + 1}/{len(pick)} "
                  f"(exploit {n_exploit} / explore {n_explore})", flush=True)
    return pd.DataFrame(rows)


def _report_r2(test_r2: dict):
    r2 = [v for v in test_r2.values() if np.isfinite(v)]
    if not r2:
        print("      no held-out R² (training set too small)")
        return
    print(f"      held-out R² (n={len(r2)}): median {np.median(r2):+.2f}, "
          f"range [{min(r2):+.2f}, {max(r2):+.2f}]")
    weak = [m for m, v in test_r2.items()
            if not np.isfinite(v) or v < 0.3]
    if weak:
        print(f"      weak fits (R² < 0.3): {', '.join(weak)}")


def _report_surrogate_d(info):
    """Held-out accuracy of the single D(θ) surrogate (XGB-Chiarella §3.3 — the
    surrogate must be an accurate proxy of the true loss for the stage-1
    pool-argmin optimum to be trustworthy). This subsumes the old per-moment R²
    and the item-4 aggregate-D check: the regressor IS the aggregate D now."""
    if not info or info.get("n", 0) < 20:
        print("      D(θ) surrogate: n/a (too few finite-loss samples)")
        return
    print(f"      D(θ) surrogate (single regressor, n={info['n']}, "
          f"label-clip cap={info['cap']:.2f}): held-out R²={info['r2']:+.2f}, "
          f"Pearson r={info['corr']:+.2f}")


def _report_moment_sds(sds: dict):
    """Print the per-moment empirical sampling SDs s_i (the fixed Franke
    weights) grouped by loss component, and flag any degenerate ones the
    loss will skip."""
    print(f"      per-moment empirical sampling SD s_i (Franke weights, "
          f"{BOOTSTRAP_BLOCK}-bar block bootstrap):")
    for c in COMPONENT_NAMES:
        for m in COMPONENT_MOMENTS[c]:
            s = sds.get(m, float("nan"))
            tag = f"{s:.3e}" if np.isfinite(s) else "NaN  (skipped by loss)"
            print(f"        {c:<5s} {m:<18s} s_i={tag}")
    dropped = [m for m in MOMENT_NAMES if not np.isfinite(sds.get(m, float("nan")))]
    if dropped:
        print(f"      moments skipped by loss (degenerate bootstrap SD): "
              f"{', '.join(dropped)}")


def _report_reachability(lhs_df: pd.DataFrame, target: dict):
    """Diagnostic only: flag targets outside the LHS-observed achievable range.
    Doesn't drop moments from the loss; surfaces structural model-vs-data gaps."""
    unreachable = []
    for m in MOMENT_NAMES:
        vals = lhs_df[m].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        t = float(target[m])
        lo, hi = float(vals.min()), float(vals.max())
        if not (lo <= t <= hi):
            gap = t - hi if t > hi else lo - t
            unreachable.append((m, t, lo, hi, gap))
    if unreachable:
        print(f"      UNREACHABLE targets (model-side gap — flag for design review):")
        for name, t, lo, hi, gap in unreachable:
            print(f"        {name:<22s} target {t:+.4f}  "
                  f"LHS [{lo:+.4f}, {hi:+.4f}]  miss {gap:+.4f}")


# ── end-to-end ───────────────────────────────────────────────────────────────

def run_regime_calibration(regime: str, target: dict,
                           n_lhs: int = N_LHS, n_days: int = N_DAYS,
                           n_runs: int = N_RUNS, n_refine: int = N_REFINE,
                           n_per_refine: int = N_PER_REFINE,
                           n_stage2: int = N_STAGE2,
                           seed: int = SEED) -> dict:
    """Run the surrogate-assisted SMM pipeline for ONE regime independently
    (HFABM Gao et al. (2022) §4.2 two-stage: surrogate → tight grid search around
    optimum on the true simulator; weights from historical block bootstrap)."""
    OUT_DIR.mkdir(exist_ok=True)
    _activate_regime(regime)     # calm 3-d / stressed 4-d (p_zi) parameter set
    lhs_path = OUT_DIR / f"calibration_lhs_{regime}.csv"
    n_steps = 6 + n_refine   # bootstrap + LHS + surrogate + AL × n_refine + stage-1 + stage-2 + validate

    print(f"\n{'='*70}")
    print(f"  CALIBRATING REGIME: {regime}  (HFABM 2-stage, bootstrap weights)")
    print(f"{'='*70}")

    # Step 1 — per-moment empirical sampling SDs (Franke & Westerhoff 2012
    # weights). Computed ONCE per regime; fixed across all LHS / AL /
    # stage-2 / optimisation steps so the loss is stationary and the D(θ)
    # values stay comparable across model versions.
    step = 1
    print(f"[{regime}][{step}/{n_steps}] Franke weights from historical 1-min mid returns "
          f"({N_BOOTSTRAP} block resamples × {BOOTSTRAP_BLOCK}-bar blocks)")
    moment_sds = empirical_moment_sd(regime, n_days=n_days, n_runs=n_runs, seed=seed)
    _report_moment_sds(moment_sds)

    # Step 1 — initial LHS
    step += 1
    _cols = set(pd.read_csv(lhs_path, nrows=0).columns) if lhs_path.exists() else set()
    # Regenerate if columns don't EXACTLY match the current param + moment set —
    # catches a structural change (e.g. dropped depth/MM params, new moments)
    # whose stale moment VALUES would silently corrupt the surrogate.
    cache_ok = (lhs_path.exists()
                and set(MOMENT_NAMES).issubset(_cols)
                and set(PARAM_KEYS).issubset(_cols)
                and not (_cols - set(PARAM_KEYS) - set(MOMENT_NAMES)))
    if cache_ok:
        lhs_df = pd.read_csv(lhs_path)
        print(f"[{regime}][{step}/{n_steps}] Reusing cached: {lhs_path} ({len(lhs_df)} samples)")
    else:
        if lhs_path.exists():
            print(f"[{regime}][{step}/{n_steps}] Cached LHS lacks current moment columns "
                  f"(loss set changed) — regenerating")
        else:
            print(f"[{regime}][{step}/{n_steps}] Initial LHS: {n_lhs} samples × {n_days} days × {n_runs} seeds")
        t0 = time.perf_counter()
        lhs_df = run_lhs(regime, n_lhs, n_days, n_runs, seed=seed)
        lhs_df.to_csv(lhs_path, index=False)
        print(f"      {time.perf_counter() - t0:.0f}s → {lhs_path}")

    # Step 2 — initial surrogate
    step += 1
    print(f"[{regime}][{step}/{n_steps}] Initial XGBoost surrogate training")
    model, sinfo = train_surrogate(lhs_df, target, moment_sds)
    _report_surrogate_d(sinfo)
    _report_reachability(lhs_df, target)

    # Active-learning rounds
    for r in range(n_refine):
        step += 1
        print(f"[{regime}][{step}/{n_steps}] Active-learning round {r+1}/{n_refine}: "
              f"{n_per_refine} new sims")
        t0 = time.perf_counter()
        new_rows = _refine_round(regime, model, target, moment_sds, n_per_refine,
                                 n_days, n_runs, seed=seed + 1000 * (r + 1))
        lhs_df = pd.concat([lhs_df, new_rows], ignore_index=True)
        lhs_df.to_csv(lhs_path, index=False)
        model, sinfo = train_surrogate(lhs_df, target, moment_sds, seed=r + 1)
        print(f"      total samples now {len(lhs_df)}, "
              f"{time.perf_counter() - t0:.0f}s")
        _report_surrogate_d(sinfo)
        _report_reachability(lhs_df, target)

    # Stage-1 optimisation on the surrogate
    step += 1
    print(f"[{regime}][{step}/{n_steps}] Stage 1: surrogate optimum (pool argmin over Sobol candidates)")
    theta_surrogate, loss_surrogate = optimise_surrogate(model)
    print(f"      surrogate-predicted D at stage-1 optimum: {loss_surrogate:.4f}")

    # Stage-2 grid search around surrogate optimum on the TRUE simulator
    # (HFABM Gao et al. (2022) §4.2). Corrects for surrogate fitting noise.
    step += 1
    print(f"[{regime}][{step}/{n_steps}] Stage 2: grid search around θ* on true simulator "
          f"({n_stage2} sims in ±{STAGE2_BOX_FRAC*100:.0f}% box)")
    t0 = time.perf_counter()
    theta_star, true_loss_star, stage2_df = stage2_grid_search(
        theta_surrogate, regime, target, moment_sds,
        n_grid=n_stage2, n_days=n_days, n_runs=n_runs,
        seed=seed + 50_000,
    )
    stage2_path = OUT_DIR / f"calibration_stage2_{regime}.csv"
    stage2_df.to_csv(stage2_path, index=False)
    print(f"      stage-2 best TRUE loss: {true_loss_star:.4f}  "
          f"(surrogate-predicted {loss_surrogate:.4f})  "
          f"[{time.perf_counter() - t0:.0f}s → {stage2_path}]")

    # Validation at stage-2 best
    step += 1
    print(f"[{regime}][{step}/{n_steps}] Validating at stage-2 optimum")
    validated = simulate_moments(theta_star, regime, n_days, n_runs,
                                 seed=seed + 9991)
    validated_loss = _true_loss(validated, target, moment_sds)

    print(f"\n{'':4}{'moment':<20}{'target':>12}{'simulator':>14}{'|s-t|':>12}{'|s-t|/s_i':>12}")
    # All moments (ret_kurtosis is now in MOMENT_NAMES but in no component, so
    # it prints as a diagnostic — HFABM convention; Hill/KS are the loss tails).
    all_moment_names = list(MOMENT_NAMES)
    for m in all_moment_names:
        t = float(target[m])
        v = float(validated[m])
        sd = moment_sds.get(m, float("nan"))
        if m == "ret_kurtosis":
            std_col = "  (diag)"
        elif np.isfinite(sd) and sd > 0:
            std_col = f"{abs(v - t) / sd:>12.2f}"
        else:
            std_col = f"{'—':>12}"
        print(f"    {m:<20}{t:>12.4f}{v:>14.4f}{abs(v - t):>12.4f}{std_col}")

    sim_dict = {m: float(validated[m]) for m in MOMENT_NAMES}
    hist_dict = {m: float(target[m]) for m in MOMENT_NAMES}
    deltas_validated = compute_grouped_deltas(sim_dict, hist_dict, moment_sds)
    print(f"\n    grouped Δ contributions (Franke-standardised, {len(COMPONENT_NAMES)} in loss):")
    for c in COMPONENT_NAMES:
        d = deltas_validated.get(c, float("nan"))
        print(f"      Δ{c:<6s} = {d:.4f}   (mean sampling-SDs off)")
    print(f"\n    validated true loss D(θ) at stage-2 θ: {validated_loss:.4f}")

    return {
        "theta_stage1": {k: float(v) for k, v in zip(PARAM_KEYS, theta_surrogate)},
        "theta_stage2": {k: float(v) for k, v in zip(PARAM_KEYS, theta_star)},
        "surrogate_loss_stage1": float(loss_surrogate),
        "true_loss_stage2": float(true_loss_star),
        "true_loss_validated": float(validated_loss),
        "component_deltas": {c: (float(deltas_validated[c])
                                 if np.isfinite(deltas_validated.get(c, float("nan")))
                                 else None)
                             for c in COMPONENT_NAMES},
        "surrogate_d_accuracy": {"r2": sinfo["r2"], "corr": sinfo["corr"],
                                 "n": sinfo["n"], "clip_cap": sinfo["cap"]},
        "moment_sds": {m: (float(v) if np.isfinite(v) else None)
                       for m, v in moment_sds.items()},
        "n_total_samples": len(lhs_df),
        "n_stage2": n_stage2,
        "target": dict(target),
        "validated": dict(validated),
    }


def grid_search(regime: str, target: dict, sds: dict,
                n_per_dim: int, n_days: int = N_DAYS, n_runs: int = N_RUNS,
                seed: int = SEED) -> tuple:
    """Exhaustive grid search over the regime's parameter box, minimising the
    SAME loss D(theta) the surrogate uses. This is the calibration method of Gao
    et al. (2023) 'Deeper Hedging' / Chiarella-Heston (§3.3: kappa,beta,omega,
    theta,phi calibrated by grid search minimising D(theta)) — transparent and
    exhaustive, which is defensible at this low dimension (calm 3-d / stressed
    4-d). Evaluates the TRUE simulator at every node (no surrogate). Returns
    (theta_best, D_best, grid_df); grid_df has one row per node with theta, D,
    each component delta and the moments — the full loss surface for the report."""
    _activate_regime(regime)
    keys = list(PARAM_KEYS)
    lo, hi = PARAM_BOUNDS_ARR[:, 0], PARAM_BOUNDS_ARR[:, 1]
    axes = [np.linspace(lo[i], hi[i], n_per_dim) for i in range(len(keys))]
    nodes = list(itertools.product(*axes))
    print(f"  grid: {len(keys)}-d x {n_per_dim}/dim = {len(nodes)} nodes "
          f"({n_days}d x {n_runs} seeds each)")
    rows, best_theta, best_D = [], None, float("inf")
    for i, node in enumerate(nodes):
        theta = np.array(node, dtype=float)
        m = simulate_moments(theta, regime, n_days, n_runs, seed=seed + i)
        D = _true_loss(m, target, sds)
        deltas = compute_grouped_deltas(
            {k: float(m[k]) for k in MOMENT_NAMES},
            {k: float(target[k]) for k in MOMENT_NAMES}, sds)
        row = {k: float(v) for k, v in zip(keys, theta)}
        row["D"] = float(D) if np.isfinite(D) else float("nan")
        for c in COMPONENT_NAMES:
            row[f"d{c}"] = float(deltas.get(c, float("nan")))
        for name in MOMENT_NAMES:
            row[name] = float(m[name])
        rows.append(row)
        if np.isfinite(D) and D < best_D:
            best_theta, best_D = theta, float(D)
        if (i + 1) % max(1, len(nodes) // 10) == 0:
            print(f"    node {i+1}/{len(nodes)}  best D so far {best_D:.4f}")
    return best_theta, best_D, pd.DataFrame(rows)


def _merge_save(path, config: dict, results: dict) -> dict:
    """Merge per-regime results into an existing JSON so calm-only and
    stressed-only runs ACCUMULATE instead of overwriting (resolves the
    'rewritten per regime' gotcha — both regimes persist in one file)."""
    prev = {}
    if path.exists():
        try:
            prev = json.load(open(path))
        except Exception:
            prev = {}
    merged = dict(prev.get("results", {}))
    merged.update(results)
    payload = {"regimes": sorted(merged), "config": config, "results": merged}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_grid(regimes: tuple = REGIMES, n_per_dim: int = 7,
             n_days: int = N_DAYS, n_runs: int = N_RUNS, seed: int = SEED) -> dict:
    """Grid-search calibration per regime, reported ALONGSIDE the surrogate `run`
    (same loss, same targets — the two should agree on theta*). Writes the full
    loss surface to output/calibration_grid_{regime}.csv and the optima to
    output/calibrated_params_grid.json."""
    OUT_DIR.mkdir(exist_ok=True)
    targets = empirical_targets()
    results = {}
    for regime in regimes:
        print(f"\n{'='*70}\n  GRID SEARCH: {regime}\n{'='*70}")
        sds = empirical_moment_sd(regime, n_days=n_days, n_runs=n_runs, seed=seed)
        t0 = time.perf_counter()
        theta_best, D_best, grid_df = grid_search(
            regime, targets[regime], sds, n_per_dim, n_days, n_runs, seed=seed)
        grid_path = OUT_DIR / f"calibration_grid_{regime}.csv"
        grid_df.to_csv(grid_path, index=False)
        keys = list(PARAM_KEYS)
        best_row = grid_df.loc[grid_df["D"].idxmin()]
        results[regime] = {
            "theta_grid": {k: float(theta_best[i]) for i, k in enumerate(keys)},
            "D_grid": float(D_best),
            "component_deltas": {c: float(best_row[f"d{c}"]) for c in COMPONENT_NAMES},
            "target": {name: float(targets[regime][name]) for name in MOMENT_NAMES},
            "moments": {name: float(best_row[name]) for name in MOMENT_NAMES},
            "n_per_dim": int(n_per_dim), "n_nodes": int(len(grid_df)),
        }
        print(f"  [{regime}] best D {D_best:.4f} at {results[regime]['theta_grid']}  "
              f"[{time.perf_counter()-t0:.0f}s -> {grid_path}]")
    _merge_save(OUT_DIR / "calibrated_params_grid.json",
                {"n_per_dim": n_per_dim, "n_days": n_days, "n_runs": n_runs, "seed": seed},
                results)
    print(f"\nSaved output/calibrated_params_grid.json (regimes now: "
          f"{', '.join(sorted(results))} merged with any prior)")
    return results


def run_calibration(regimes: tuple = REGIMES,
                    n_lhs: int = N_LHS, n_days: int = N_DAYS,
                    n_runs: int = N_RUNS, n_refine: int = N_REFINE,
                    n_per_refine: int = N_PER_REFINE,
                    n_stage2: int = N_STAGE2,
                    seed: int = SEED) -> dict:
    """Calibrate each regime independently (D18b). Writes one combined
    output/calibrated_params.json with results nested by regime."""
    OUT_DIR.mkdir(exist_ok=True)
    targets = empirical_targets()
    results = {}
    for regime in regimes:
        results[regime] = run_regime_calibration(
            regime, targets[regime],
            n_lhs=n_lhs, n_days=n_days, n_runs=n_runs,
            n_refine=n_refine, n_per_refine=n_per_refine,
            n_stage2=n_stage2, seed=seed,
        )
    payload = _merge_save(
        OUT_DIR / "calibrated_params.json",
        {"n_lhs": n_lhs, "n_refine": n_refine, "n_per_refine": n_per_refine,
         "n_days": n_days, "n_runs": n_runs, "n_stage2": n_stage2, "seed": seed},
        results)
    print(f"\nSaved output/calibrated_params.json (regimes now: "
          f"{', '.join(sorted(payload['results']))} merged with any prior)")
    return payload


def _print_targets():
    for regime, m in empirical_targets().items():
        print(f"[{regime}]")
        for name in MOMENT_NAMES:
            print(f"  {name:<16}{m[name]:+.5f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "targets":
        _print_targets()
    elif cmd == "run":
        # Optional regime arg: `run calm [...]` or `run stressed [...]`
        if len(sys.argv) > 2 and sys.argv[2] in REGIMES:
            regimes = (sys.argv[2],)
            arg_offset = 3
        else:
            regimes = REGIMES
            arg_offset = 2
        n_lhs        = int(sys.argv[arg_offset])     if len(sys.argv) > arg_offset     else N_LHS
        n_days       = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else N_DAYS
        n_runs       = int(sys.argv[arg_offset + 2]) if len(sys.argv) > arg_offset + 2 else N_RUNS
        n_refine     = int(sys.argv[arg_offset + 3]) if len(sys.argv) > arg_offset + 3 else N_REFINE
        n_per_refine = int(sys.argv[arg_offset + 4]) if len(sys.argv) > arg_offset + 4 else N_PER_REFINE
        n_stage2     = int(sys.argv[arg_offset + 5]) if len(sys.argv) > arg_offset + 5 else N_STAGE2
        run_calibration(regimes=regimes, n_lhs=n_lhs, n_days=n_days,
                        n_runs=n_runs, n_refine=n_refine,
                        n_per_refine=n_per_refine, n_stage2=n_stage2)
    elif cmd == "grid":
        # Grid-search calibration (Gao 2023 Deeper-Hedging method), reported
        # alongside `run`. Usage: grid [regime] [n_per_dim] [n_days] [n_runs]
        if len(sys.argv) > 2 and sys.argv[2] in REGIMES:
            regimes = (sys.argv[2],)
            arg_offset = 3
        else:
            regimes = REGIMES
            arg_offset = 2
        n_per_dim = int(sys.argv[arg_offset])     if len(sys.argv) > arg_offset     else 7
        n_days    = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else N_DAYS
        n_runs    = int(sys.argv[arg_offset + 2]) if len(sys.argv) > arg_offset + 2 else N_RUNS
        run_grid(regimes=regimes, n_per_dim=n_per_dim, n_days=n_days, n_runs=n_runs)
    else:
        print("usage: python calibrate.py [ targets | "
              "run [calm|stressed] [N_LHS N_DAYS N_RUNS N_REFINE N_PER_REFINE N_STAGE2] | "
              "grid [calm|stressed] [N_PER_DIM N_DAYS N_RUNS] ]")
