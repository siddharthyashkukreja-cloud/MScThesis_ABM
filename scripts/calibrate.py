"""
Agent-parameter calibration via two-stage surrogate-assisted SMM (HFABM Gao
et al. (2022) §4.2 two-stage workflow with Franke & Westerhoff 2012 inverse-
bootstrap-variance weights).

Calm and stressed are calibrated as two independent optimisation problems.
Every behavioural parameter is regime-specific — no shared parameters across
regimes. The regimes are economically distinct (different volatility, book
depth, microstructure), so a shared parameter would compromise both fits;
independent runs let each regime find its own optimum.

Methodology:
  1. Loss is XGB-Chiarella-style (Gao et al. 2022 §3.2) — five grouped
     components (KS and Hill carry the tail jointly):

         D(θ) = ΔKS(θ) + ΔV(θ) + ΔACF1(θ) + ΔACF2(θ) + ΔHill(θ)

     Each component is the mean Franke-standardised L1 distance over its
     moments — every moment difference |m_sim − m_hist| is divided by that
     moment's empirical block-bootstrap sampling SD s_i before averaging.
     Standardisation makes each component dimensionless (a count of sampling
     SDs), so all carry equal weight and D is comparable across model
     versions.
     The grouped moments:
         ΔKS : Kolmogorov-Smirnov 2-sample stat vs the empirical return CDF
                  (eq 7); target 0, /s_KS. Robust whole-distribution fat-tail
                  target. s_KS is from full-length resamples while the sim
                  sample is shorter, so ΔKS is somewhat over-weighted — watch
                  the per-component line; size-match s_KS if it dominates.
         ΔV : ret_std
         ΔACF1 : ACF of returns, forward 3-lag avg at centres {1, 5, 10, 20}
         ΔACF2 : ACF of |returns|, forward 3-lag avg at {1, 5, 10, 20} — clustering
         ΔHill : banded Hill tail index (HFABM §4.1.1) — direct tail lever, paired
                  with KS so both the whole distribution and the tail are matched.
     Kurtosis stays a diagnostic only (outlier-dominated, so matching it is noise-chasing).

  2. Weights: each moment's empirical sampling SD s_i, computed once per
     regime by Künsch (1989) moving-block bootstrap on the historical
     1-min mid log-returns (Franke & Westerhoff 2012; HFABM eq 8). Block
     size 390 (one RTH day) exceeds the longest ACF lag (91) so re-ordering
     preserves the autocorrelation structure. s_i is fixed across the run,
     keeping the loss stationary and D(θ) comparable.

  3. Sobol-sample the behavioural-parameter space (low-discrepancy; beats
     LHS at small N — XGB-Chiarella §3.3.2); simulate each θ on this regime
     (n_runs seeds × n_days days, pooled).
  4. Train a single XGBoost regressor θ → D(θ) — the scalar loss, not one
     model per moment (XGB-Chiarella §3.3); labels clipped at the
     LOSS_CLIP_PCTL percentile (Remark 2). Held-out R² on D is the
     proxy-accuracy trust check.
  5. Active-learning refinement (exploration-exploitation, §3.3 Step 4): score
     a Sobol candidate pool with the surrogate; simulate a ~2:1 exploit (lowest
     predicted D) / explore (random) mix; append; retrain. Repeat n_refine.
  6. Stage 1: pool argmin — evaluate the D-surrogate over a large Sobol
     pool and take the minimum. A tree ensemble is piecewise-constant, so
     gradient optimisers (L-BFGS-B) are ill-suited.
  7. Stage 2 (HFABM §4.2): tight Sobol box within ±STAGE2_BOX_FRAC of bound
     width around the stage-1 optimum, scored on the true simulator. Pick the
     simulator-evaluated θ with the smallest actual loss as θ*. Corrects for
     surrogate fitting noise.
  8. Validate by re-running the simulator at θ* on fresh seeds (out-of-sample
     vs the stage-2 selection) and reporting per-moment comparison + grouped
     Δ contributions + total D(θ).

Data-side parameters (the Kalman fundamental, data/v_kalman.py; the SV-MJD
scenario generator, data/v_gbm.py, under FV_GBM=1; the v0/σ_v dicts) live in
model/globals.py and auto-populate per regime via ModelParams. Pinned-
structural: zi_mu=0.025 (CST 2008); ft_alpha=mt_alpha=1.0; mt_lambda=0.05;
mt_mu=0; qty_max=10; no market maker; geometric placement with the data-fit
p_zi (calm). The calibrated loop is regime-specific: calm 3-d {ft_sigma_c,
zi_alpha, zi_delta}, stressed 4-d {+ p_zi}. FT/MT use replace-on-new order
management; the FT has no dead-band.

Requires xgboost (with libomp) and scipy; a missing library fails the run
loudly. Per-regime LHS training data is cached to
output/calibration_lhs_{regime}.csv; stage-2 grid output to
output/calibration_stage2_{regime}.csv. Delete to regenerate.

Moments: Cont 2001 stylised-fact battery (return std; returns ACF at
{1,5,10,20}; |returns| ACF at {1,5,10,20}; the KS distribution distance) +
banded Hill (1975) tail index. Computed on the 1-min mid log-returns in
both sim and empirical, overnight (cross-session) returns excluded on both
sides (excess kurtosis is a diagnostic, not a loss moment).

CLI:
  python calibrate.py targets # print empirical moment targets
  python calibrate.py run [N D R Rf K G] # both regimes; defaults below
  python calibrate.py run calm [N D R Rf K G] # calm only
  python calibrate.py run stressed [N D R Rf K G] # stressed only
    N=N_LHS, D=N_DAYS, R=N_RUNS, Rf=N_REFINE, K=N_PER_REFINE, G=N_STAGE2
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from model.globals import ModelParams, V0, FV_CSV, day_start_steps
from model.simulation import Simulation
from model.run_simulation import build_traders, build_clearing_tier

# ── config ───────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
OUT_DIR = REPO_DIR / "output"
PROC_DIR = REPO_DIR / "data" / "processed"
REGIME_DATA = {
    "calm":     PROC_DIR / "ES_front_calm_1m.csv",
    "stressed": PROC_DIR / "ES_front_stressed_1m.csv",
}
REGIMES = ("calm", "stressed")
BARS_PER_DAY = 390   # nominal RTH-day length, used only to size the calm
                     # subsample (_sim_steps); the real overnight boundaries are
                     # data-driven via day_start_steps

# SV-MJD recalibration flag. FV_GBM=1 drives the agent loop on the synthetic
# Stein-Stein/Heston + Merton fundamental (data/fv_gbm_{regime}.csv,
# data/v_gbm.py) instead of the Kalman path. Empirical targets and bootstrap
# weights are unchanged — the loss still matches the real ES moments, testing
# whether the agent layer reproduces the stylised facts on an independent
# fundamental. Separate LHS cache per flag.
FV_GBM = bool(os.environ.get("FV_GBM"))


def _fv_rel(regime: str) -> str:
    return f"data/fv_gbm_{regime}.csv" if FV_GBM else FV_CSV[regime]


@lru_cache(maxsize=None)
def _regime_fv_bars(regime: str) -> int:
    """Bars in the regime's ACTIVE fundamental series (Kalman or SV-MJD)."""
    return len(pd.read_csv(REPO_DIR / _fv_rel(regime)))


@lru_cache(maxsize=None)
def _day_starts(regime: str) -> tuple:
    """Session-open rows of the active fundamental. Kalman: the real data-driven
    boundaries (day_start_steps); SV-MJD: the synthetic path's own `ts`
    boundaries (390-bar generator days)."""
    if not FV_GBM:
        return day_start_steps(regime)
    ts = pd.read_csv(REPO_DIR / _fv_rel(regime), usecols=["ts"], parse_dates=["ts"])["ts"]
    d = ts.dt.normalize().to_numpy()
    return tuple(int(i) for i in np.flatnonzero(np.r_[True, d[1:] != d[:-1]]))


def _sim_steps(regime: str, n_days: int) -> int:
    """Steps to simulate for `regime`. Stressed (Kalman) is a bounded historical
    episode (the 2020 COVID window, ~29 RTH days): simulate the whole series so
    the sim and the empirical targets span the same period, and never run past
    it — past the data Simulation._v_at clamps the fundamental to the last bar
    (frozen V_t). Calm subsamples to n_days. Under FV_GBM the synthetic stressed
    path has no 'same period' to match, so it subsamples like calm (keeps the
    sim sample size — and hence s_KS — consistent across fundamentals)."""
    full = _regime_fv_bars(regime)
    if regime == "stressed" and not FV_GBM:
        return full
    return min(BARS_PER_DAY * n_days, full)

# Fixed population — the flat ODD star topology (see run_simulation.py).
# No market maker: an always-quoting MM clamps spread variability and
# suppresses volatility clustering. Geometric data-fit placement (globals.P_ZI)
# supplies near-mid liquidity instead.
POP = dict(n_fundamental=40, n_momentum=20, n_zi=40)
                                              # n_fundamental folds 30 FT + 10 BCM, giving a
                                              # 40:20:40 FT-equiv:MT:ZI mix

# Per-regime calibration loop. Structural parameters are not calibrated:
# qty_max from QTY_MAX[regime], and calm placement depth from the measured L2 p_zi.
ZI_MU_FIXED = 0.025      # ODD-baseline ZI market-order rate (Cont-Stoikov-Talreja);
                         # the fallback when zi_mu is not in the active loop.
# Search boxes are literature-anchored and trimmed to the economically sensible
# basin, so the low-dimensional grids keep useful node spacing (the surrogate
# searches the same boxes). Regime-specific choices:
# * zi_mu is calibrated in both regimes (calm has an interior optimum near 0.10,
#   pulling Hill 2.71->2.94 toward the 2.96 target); the 0.15 cap keeps ZI market
#   orders a minority of ZI flow.
# * stressed ft_sigma_c has an interior optimum near 0.4, which the box brackets.
# * zi_delta floor 0.02 (mean order life ~50 min, under one session) keeps resting
#   depth bounded and driftless.
# * stressed p_zi floor 0.08 admits a sparser book than the measured L2 0.343 (still
#   inside the box); calm p_zi stays at the measured L2 value, the more defensible
#   choice where a direct measurement is available.
PARAM_BOUNDS_BY_REGIME = {
    # D68: bounds widened on the four edges the n_days=75 grid argmin sat on, to
    # let the surrogate find an interior optimum (calm zi_alpha was at its 0.50
    # ceiling, calm/stressed ft_sigma_c at the floor, stressed p_zi at the 0.38
    # ceiling). zi_mu cap (0.15) and zi_delta floor (0.02) are kept: they are
    # economic priors (market orders a minority of ZI flow; order life < 1 session),
    # not failed optima, so their binding is reported, not relaxed.
    # D69: mt_lambda + mt_gamma added to the loop. With the tanh-activation MT (share of MTs acting
    # scales with trend strength) the momentum channel is the main driver of the |r|-ACF clustering
    # profile and the residual bounce, so its horizon (mt_lambda) and strength (mt_gamma) must be
    # calibrated rather than pinned. Sweeps confirm both are sensitive to acf_r1 / acf|r|1 / acf|r|5.
    "calm": {
        "ft_sigma_c": (0.25, 1.1),    # widened down (argmin was on the 0.5 floor)
        "zi_alpha":   (0.02, 0.80),   # widened up (argmin was on the 0.50 ceiling)
        "zi_delta":   (0.02, 0.34),
        "p_zi":       (0.08, 0.80),   # D70: calm p_zi now CALIBRATED (was L2-fit 0.543) -> removes the MBP-10 book-data dependency
        "zi_mu":      (0.0125, 0.15),
        "mt_lambda":  (0.004, 0.20),  # EWMA decay: ~3-hour to ~3.5-min half-life
        "mt_gamma":   (0.1, 2.0),     # tanh activation strength (share of MTs acting vs |M|/sigma_v)
    },
    "stressed": {
        "ft_sigma_c": (0.10, 0.85),   # widened down (argmin was on the 0.25 floor)
        "zi_alpha":   (0.08, 0.44),   # interior optimum ~0.26
        "zi_delta":   (0.02, 0.32),
        "p_zi":       (0.08, 0.55),   # widened up (argmin was on the 0.38 ceiling; measured L2 0.343)
        "zi_mu":      (0.0125, 0.15),
        "mt_lambda":  (0.004, 0.20),  # EWMA decay: ~3-hour to ~3.5-min half-life
        "mt_gamma":   (0.1, 2.0),     # tanh activation strength (share of MTs acting vs |M|/sigma_v)
    },
}
# ft_sigma_c is calibrated, not pinned: it is the dominant lever on the mid-return
# tails. A wide belief width makes FTs overshoot V_t (sweeping the book to the
# outermost reservation V_t + max z·σ_fund), fattening the tails and drowning
# clustering in i.i.d. bursts; ft_sigma_c≈1 (σ_fund≈3 ticks) reproduces Hill≈3 and
# revives the |r|-ACF by transmitting V_t faithfully. The trade-off is less FT
# inventory concentration at smaller values.
#
# Calibrated loop (PARAM_BOUNDS_BY_REGIME): calm {ft_sigma_c, zi_alpha, zi_delta,
# zi_mu}, stressed {+ p_zi}. Pinned / out of the loop: ft_alpha = mt_alpha = 1.0
# (FT/MT submit a limit every step, ODD §Step Sequence step 3); mt_mu = 0.0 (MT is
# limit-only — trend-direction market flow corrupts the return ACF); mt_lambda =
# 0.05; ft_delta = mt_delta = 0 (replace-on-new). The ZI-rate boxes stay near the
# Cont-Stoikov-Talreja 2008 / ODD baselines (0.025 / 0.15) so market orders remain a
# minority of ZI flow — an early unconstrained run drove zi_mu to ~0.5 (half of ZI
# activity book-walking), giving kurtosis ~1200 and a persistent bid-ask bounce.
#
# Under FV_GBM the calm ft_sigma_c cap is relaxed to the stressed range: the SV-MJD
# path is intentionally lighter-tailed (its jumps carry only the measured RV share,
# BNS 2004 / Mancini 2009), so the FT tail lever may legitimately sit higher.
if FV_GBM:
    PARAM_BOUNDS_BY_REGIME["calm"]["ft_sigma_c"] = (0.5, 2.0)

# ── Campaign experiment flags (overnight calibration campaign) ────────────────
# Each is OFF by default, so absent any env var PARAM_BOUNDS_BY_REGIME is the
# baseline (E0) loop. Gated at import time — every experiment is a fresh process
# with its own environment, so import-time injection is clean and isolated.
# E2 MT_LAMBDA_IN_LOOP — add mt_lambda (0.004, 0.20) to both regimes' loops.
# E5 FTMT_GATES — add ft_alpha/mt_alpha/ft_delta/mt_delta (high-dim).
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

# Individual moments — surrogate targets and loss inputs. ACF
# lags sit where the empirical ES 1-min signal actually lives: the return ACF is
# concentrated at the SHORT end (a bid-ask/microstructure term at lag 1 and a
# transient-impact mean-reversion peaking near lag ~5-10), and flat (~0) by lag
# 30+, so high lags carry no information.
# Loss D(θ) — FIVE standardised-moment components, matched to XGB-Chiarella
# (Gao et al. 2022 §3.2) + HFABM (§4.1.1):
# KS = Kolmogorov-Smirnov 2-sample statistic between simulated and empirical
# return CDFs (paper §3.2.4, eq 7) — robust whole-distribution fat-tail target.
# V = return-standard-deviation distance (paper eq 8).
# ACF1 = returns ACF at centres {1, 5, 10, 20}, FORWARD 3-lag smoothed
# (paper §3.2.2: lag-c = mean of {c, c+1, c+2}).
# ACF2 = ABSOLUTE-returns ACF, forward 3-lag smoothed at {1, 5, 10, 20} — the
# volatility-clustering target. |r| (Cont 2001) is used rather than the
# paper's r²: r² is outlier-dominated, so its ACF has a large sampling SD
# and the clustering miss vanishes under the standardisation; |r| is far
# less noisy, so clustering actually counts.
# Hill = banded Hill tail index (HFABM §4.1.1). KS + Hill together carry the tail:
# KS the whole-distribution match, Hill a direct, reachable tail-index lever.
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
    "KS":   ("ks_stat",),            # ΔKS (XGB-Chiarella §3.2.4 eq 7; target 0, /s_KS)
    "V":    ("ret_std",),            # ΔV (paper eq 8)
    "ACF1": ACF1_NAMES,              # ΔACF1 (paper eq 9: returns, lags 1/10/20)
    "ACF2": ACF2_NAMES,              # ΔACF2 (|returns| ACF, short lags — clustering)
    "Hill": ("hill_tail_index",),    # ΔHill (HFABM §4.1.1 tail index; direct tail lever)
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
N_DAYS = 75       # matched calm/stressed window length (~75 sessions each) so the KS
                  # standardiser — and hence the total loss D — is comparable across regimes.
                  # Calm uses the quiet 2019-09-13..12-27 slice; stressed runs its full ~75-
                  # session COVID-episode window. (Was 20 with the old short windows.)
N_RUNS = 4        # grid AND surrogate both use N_RUNS -> identical seed count by construction;
                  # raise for the definitive final run (e.g. 8), it applies to both.
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
        zi_mu=float(d["zi_mu"]) if "zi_mu" in d else ZI_MU_FIXED,  # in-loop since D65
        zi_delta=float(d["zi_delta"]),
    )
    if "p_zi" in d:                       # stressed only; calm keeps the L2 P_ZI
        kw["p_zi"] = float(d["p_zi"])
    # mt_lambda + mt_gamma calibrated in the loop (D69 — momentum horizon + tanh activation strength).
    if "mt_lambda" in d:
        kw["mt_lambda"] = float(d["mt_lambda"])
    if "mt_gamma" in d:
        kw["mt_gamma"] = float(d["mt_gamma"])
    # mt_lambda can be pinned externally (MT_LAMBDA_FIXED, e.g. 0.00385 = 3h half-life).
    if os.environ.get("MT_LAMBDA_FIXED"):
        kw["mt_lambda"] = float(os.environ["MT_LAMBDA_FIXED"])
    # Optional campaign flag: FT/MT Bernoulli gate + stochastic cancellation (FTMT_GATES).
    for g in ("ft_alpha", "mt_alpha", "ft_delta", "mt_delta"):
        if g in d:
            kw[g] = float(d[g])
    # Population: POP (bare market) by default. E1 swaps to the run_simulation
    # population WITH clearing members (CLEARING_IN_LOOP); E4 overrides MT counts.
    pop = dict(POP)
    if os.environ.get("CLEARING_IN_LOOP"):
        pop = dict(n_fundamental=30, n_momentum=20, n_zi=40,
                   n_bcm=10, n_nbcm=5, n_bcm_with_clients=5)
    if os.environ.get("N_MOMENTUM"):
        pop["n_momentum"] = int(os.environ["N_MOMENTUM"])
    return ModelParams(
        **pop,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        fv_csv=_fv_rel(regime),            # Kalman default; SV-MJD under FV_GBM
        **kw,
        stressed=(regime == "stressed"),
    )


def _intraday_logret(mid: np.ndarray, regime: str) -> np.ndarray:
    """1-min log-returns with the cross-day (overnight) returns dropped. The sim
    opens each RTH day at the gapped V_t via a session reset, so the
    day-boundary return is an overnight gap; excluding it matches the empirical
    convention (_empirical_returns drops cross-day returns), keeping the
    calibration moments intraday on both sides. Boundaries are the real session
    opens, since a session is ~405 bars and varies, not
    a fixed 390 — the return into open row d is r[d-1]."""
    r = np.diff(np.log(mid))
    drop = [d - 1 for d in _day_starts(regime) if 0 < d <= len(r)]
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
            print(f" LHS {i + 1}/{n_lhs}", flush=True)
    return pd.DataFrame(rows)


# ── XGBoost surrogate ────────────────────────────────────────────────────────

def _xgb():
    import xgboost as xgb          # lazy: only the surrogate needs xgboost, so the
    return xgb.XGBRegressor(**XGB_KWARGS)   # moment / MCR tools can import calibrate without it


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
                        seed: int = 0, return_samples: bool = False):
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
    # The full B x K bootstrap matrix is reused by moment_coverage_ratio for the
    # per-moment CIs and the bootstrap self-coverage ceiling (one bootstrap pass).
    return (sds, samples) if return_samples else sds


def moment_coverage_ratio(regime: str, theta: np.ndarray,
                          n_days: int = N_DAYS, n_runs: int = 1, m_long: int = 50,
                          n_bootstrap: int = 1000, block_size: int = BOOTSTRAP_BLOCK,
                          seed: int = 0, out_dir: str = "output") -> dict:
    """Moment Coverage Ratio (Franke & Westerhoff 2012; HFABM 2022 §4.3.2) — a
    POST-HOC validation of a located theta; it does NOT touch the calibration loss.
    For each POINT moment the empirical 95% CI is m_emp +/- 1.96*s_i, with s_i the
    Kunsch block-bootstrap sampling SD (the same weights as the loss). The per-moment
    MCR is the fraction of `m_long` independent model runs whose simulated moment lands
    inside the CI; the JOINT MCR is the fraction inside ALL CIs simultaneously. Because
    the moments are correlated the joint ceiling is far below 0.95, so it is reported
    against the bootstrap's own self-coverage (the achievable ceiling) and the naive
    0.95^K independence benchmark. KS (no point-CI) and kurtosis (diagnostic) are
    excluded -> K=10 point moments. Writes output/mcr_{regime}.json."""
    import json
    mcr_moments = [m for m in MOMENT_NAMES if m not in ("ks_stat", "ret_kurtosis")]
    emp = empirical_targets()[regime]
    sds, boot = empirical_moment_sd(regime, n_days=n_days, n_runs=n_runs,
                                    n_bootstrap=n_bootstrap, block_size=block_size,
                                    seed=seed, return_samples=True)
    ci = {m: (emp[m] - 1.96 * sds[m], emp[m] + 1.96 * sds[m])
          for m in mcr_moments if np.isfinite(sds.get(m, np.nan))}
    runs = [simulate_moments(theta, regime, n_days, n_runs, seed=10_000 + 13 * r)
            for r in range(m_long)]
    cov, med = {}, {}
    for m in ci:
        v = np.array([rr[m] for rr in runs if np.isfinite(rr[m])])
        lo, hi = ci[m]
        cov[m] = float(np.mean((v >= lo) & (v <= hi))) if v.size else float("nan")
        med[m] = float(np.median(v)) if v.size else float("nan")
    joint = float(np.mean([all(ci[m][0] <= rr[m] <= ci[m][1] for m in ci) for rr in runs]))
    nb = len(boot[next(iter(ci))])
    self_cov = {m: float(np.mean([ci[m][0] <= boot[m][b] <= ci[m][1] for b in range(nb)]))
                for m in ci}
    self_joint = float(np.mean([all(ci[m][0] <= boot[m][b] <= ci[m][1] for m in ci)
                                for b in range(nb)]))
    # NOTE: the J specification test lives in scripts/franke_diagnostics.py, which uses the
    # correct per-run J distribution (Franke 2012 eq. 9: bootstrap-J 95th-percentile critical
    # value, p = fraction of model runs below it). The previous in-function J computed the
    # statistic on the M-averaged moment vector, which shrinks the model spread ~M-fold and
    # makes the p-value uninformative; it has been removed. This function reports the MCR only.
    result = {
        "regime": regime, "theta": {k: float(v) for k, v in zip(PARAM_KEYS, theta)},
        "m_long": m_long, "n_bootstrap": n_bootstrap, "block_size": block_size,
        "moments": {m: {"empirical": float(emp[m]),
                        "ci_lo": float(ci[m][0]), "ci_hi": float(ci[m][1]),
                        "model_median": med[m], "mcr_pct": 100.0 * cov[m],
                        "bootstrap_self_pct": 100.0 * self_cov[m]} for m in ci},
        "joint_mcr_pct": 100.0 * joint,
        "bootstrap_joint_ceiling_pct": 100.0 * self_joint,
        "joint_mcr_pct_of_ceiling": (100.0 * joint / self_joint if self_joint > 0
                                     else float("nan")),
        "independence_benchmark_pct": 100.0 * (0.95 ** len(ci)),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"mcr_{regime}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[MCR {regime}]  joint {100*joint:.1f}%  (bootstrap ceiling {100*self_joint:.1f}%, "
          f"{100*joint/max(self_joint,1e-9):.0f}% of ceiling)  mean per-moment "
          f"{100*np.nanmean(list(cov.values())):.1f}%   (J-test: see franke_diagnostics.py)")
    print(f"{'moment':>16}{'emp':>11}{'CI_lo':>11}{'CI_hi':>11}{'model_med':>11}{'MCR%':>7}{'self%':>7}")
    for m in ci:
        print(f"{m:>16}{emp[m]:>11.4g}{ci[m][0]:>11.4g}{ci[m][1]:>11.4g}"
              f"{med[m]:>11.4g}{100*cov[m]:>7.1f}{100*self_cov[m]:>7.1f}")
    return result


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
            print(f" stage-2 sim {i + 1}/{len(thetas)} "
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
            print(f" refine sim {i + 1}/{len(pick)} "
                  f"(exploit {n_exploit} / explore {n_explore})", flush=True)
    return pd.DataFrame(rows)


def _report_surrogate_d(info):
    """Held-out accuracy of the single D(θ) surrogate (XGB-Chiarella §3.3 — the
    surrogate must be an accurate proxy of the true loss for the stage-1
    pool-argmin optimum to be trustworthy). This subsumes the old per-moment R²
    and the item-4 aggregate-D check: the regressor IS the aggregate D now."""
    if not info or info.get("n", 0) < 20:
        print(" D(θ) surrogate: n/a (too few finite-loss samples)")
        return
    print(f" D(θ) surrogate (single regressor, n={info['n']}, "
          f"label-clip cap={info['cap']:.2f}): held-out R²={info['r2']:+.2f}, "
          f"Pearson r={info['corr']:+.2f}")


def _report_moment_sds(sds: dict):
    """Print the per-moment empirical sampling SDs s_i (the fixed Franke
    weights) grouped by loss component, and flag any degenerate ones the
    loss will skip."""
    print(f" per-moment empirical sampling SD s_i (Franke weights, "
          f"{BOOTSTRAP_BLOCK}-bar block bootstrap):")
    for c in COMPONENT_NAMES:
        for m in COMPONENT_MOMENTS[c]:
            s = sds.get(m, float("nan"))
            tag = f"{s:.3e}" if np.isfinite(s) else "NaN (skipped by loss)"
            print(f" {c:<5s} {m:<18s} s_i={tag}")
    dropped = [m for m in MOMENT_NAMES if not np.isfinite(sds.get(m, float("nan")))]
    if dropped:
        print(f" moments skipped by loss (degenerate bootstrap SD): "
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
        print(f" UNREACHABLE targets (model-side gap — flag for design review):")
        for name, t, lo, hi, gap in unreachable:
            print(f" {name:<22s} target {t:+.4f} "
                  f"LHS [{lo:+.4f}, {hi:+.4f}] miss {gap:+.4f}")


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
    lhs_path = OUT_DIR / f"calibration_lhs_{regime}{'_gbm' if FV_GBM else ''}.csv"
    n_steps = 6 + n_refine   # bootstrap + LHS + surrogate + AL × n_refine + stage-1 + stage-2 + validate

    print(f"\n{'='*70}")
    print(f" CALIBRATING REGIME: {regime} (HFABM 2-stage, bootstrap weights)")
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
        print(f" {time.perf_counter() - t0:.0f}s → {lhs_path}")

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
        print(f" total samples now {len(lhs_df)}, "
              f"{time.perf_counter() - t0:.0f}s")
        _report_surrogate_d(sinfo)
        _report_reachability(lhs_df, target)

    # Stage-1 optimisation on the surrogate
    step += 1
    print(f"[{regime}][{step}/{n_steps}] Stage 1: surrogate optimum (pool argmin over Sobol candidates)")
    theta_surrogate, loss_surrogate = optimise_surrogate(model)
    print(f" surrogate-predicted D at stage-1 optimum: {loss_surrogate:.4f}")

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
    print(f" stage-2 best TRUE loss: {true_loss_star:.4f} "
          f"(surrogate-predicted {loss_surrogate:.4f}) "
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
            std_col = " (diag)"
        elif np.isfinite(sd) and sd > 0:
            std_col = f"{abs(v - t) / sd:>12.2f}"
        else:
            std_col = f"{'—':>12}"
        print(f" {m:<20}{t:>12.4f}{v:>14.4f}{abs(v - t):>12.4f}{std_col}")

    sim_dict = {m: float(validated[m]) for m in MOMENT_NAMES}
    hist_dict = {m: float(target[m]) for m in MOMENT_NAMES}
    deltas_validated = compute_grouped_deltas(sim_dict, hist_dict, moment_sds)
    print(f"\n grouped Δ contributions (Franke-standardised, {len(COMPONENT_NAMES)} in loss):")
    for c in COMPONENT_NAMES:
        d = deltas_validated.get(c, float("nan"))
        print(f" Δ{c:<6s} = {d:.4f} (mean sampling-SDs off)")
    print(f"\n validated true loss D(θ) at stage-2 θ: {validated_loss:.4f}")

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
    print(f" grid: {len(keys)}-d x {n_per_dim}/dim = {len(nodes)} nodes "
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
            print(f" node {i+1}/{len(nodes)} best D so far {best_D:.4f}")
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
        print(f"\n{'='*70}\n GRID SEARCH: {regime}\n{'='*70}")
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
        print(f" [{regime}] best D {D_best:.4f} at {results[regime]['theta_grid']} "
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
    """Calibrate each regime independently. Writes one combined
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
            print(f" {name:<16}{m[name]:+.5f}")


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
    elif cmd == "mcr":
        # Moment Coverage Ratio validation on the LOCKED theta (globals.CALIBRATED).
        # Post-hoc, cheap, re-runnable without recalibrating. Run after a calibration.
        # Usage: mcr [calm|stressed] [M_LONG] [N_DAYS]
        from model.globals import CALIBRATED
        if len(sys.argv) > 2 and sys.argv[2] in REGIMES:
            regimes = (sys.argv[2],); arg_offset = 3
        else:
            regimes = REGIMES; arg_offset = 2
        m_long = int(sys.argv[arg_offset])     if len(sys.argv) > arg_offset     else 50
        n_days = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else N_DAYS
        for r in regimes:
            _activate_regime(r)
            theta = np.array([CALIBRATED[r][k] for k in PARAM_KEYS], dtype=float)
            moment_coverage_ratio(r, theta, n_days=n_days, m_long=m_long)
    else:
        print("usage: python calibrate.py [ targets | "
              "run [calm|stressed] [N_LHS N_DAYS N_RUNS N_REFINE N_PER_REFINE N_STAGE2] | "
              "grid [calm|stressed] [N_PER_DIM N_DAYS N_RUNS] | "
              "mcr [calm|stressed] [M_LONG N_DAYS] ]")
