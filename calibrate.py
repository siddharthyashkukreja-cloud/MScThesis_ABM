"""
Calibration module for Stage 8 (run before Stage 4 margin layer per
project plan).

Two pieces:
- Direct calibration: parameters with empirical analogs in WRDS Intraday
  Indicators (v0 from OPrice, sigma_v from IVol_t_m).
- Indirect calibration: stylized-fact minimization (Krishnen ABM Liquidity
  Risk + Gao 2023 surrogate matching) over (ft_sigma, mt_sigma,
  mt_lambda_ewma, mt_threshold).

Citations follow the project priority chain:
- ODD §Calibration for default parameter values
- Cont (2001) "Empirical properties of asset returns: stylized facts and
  statistical issues" for the moment-matching targets
- Gao et al. (2023) "Deeper Hedging" §3.2 for surrogate-modelling approach
- Krishnen (ABM Liquidity Risk) for stylized-fact minimization framework
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from model.globals import ModelParams
from model.simulation import Simulation
from run_simulation import build_traders


# ─── WRDS Intraday Indicators units verification ────────────────────────

def verify_ivol_units(csv_path: str = "data/thesis_data_calm.csv") -> dict:
    """
    Determine the units of IVol_t_m in the WRDS Intraday Indicators.

    Three plausible conventions:
      H1: per-second realised variance (sum of squared bar log-returns
          divided by trading-day seconds, ~23,400)
      H2: daily integrated variance directly (sum of squared bar
          log-returns over the full trading day)
      H3: per-bar variance (already divided by NObsUsed1)

    For each hypothesis we compute the implied per-bar return std and
    compare to the empirical realised volatility from open-to-close
    log-returns. The one whose 5-min-bar std (when aggregated to a full
    day via sqrt(78)) matches empirical daily std wins.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("date").reset_index(drop=True)

    # Empirical daily log-returns
    log_oc = np.log(df["DPrice"] / df["OPrice"])
    log_cc = np.log(df["DPrice"]).diff()
    daily_std_oc = float(log_oc.std())
    daily_std_cc = float(log_cc.std())

    # Implied per-bar std under each hypothesis (using IVol_t_m row by row)
    iv = df["IVol_t_m"].values
    n = df["NObsUsed1"].values
    sec_per_bar = 5 * 60   # 5-min bars

    sigma_h1 = np.sqrt(iv * sec_per_bar)             # per-second RV * bar seconds
    sigma_h2 = np.sqrt(iv / n)                        # daily IV / bars
    sigma_h3 = np.sqrt(iv)                            # already per-bar

    # Daily std implied = per-bar std * sqrt(N_bars)
    daily_h1 = sigma_h1 * np.sqrt(n)
    daily_h2 = sigma_h2 * np.sqrt(n)
    daily_h3 = sigma_h3 * np.sqrt(n)

    out = {
        "n_rows": len(df),
        "empirical_daily_std_open_close": daily_std_oc,
        "empirical_daily_std_close_close": daily_std_cc,
        "H1_per_second_RV": {
            "implied_per_5min_std_mean": float(sigma_h1.mean()),
            "implied_daily_std_mean":    float(daily_h1.mean()),
            "ratio_to_empirical_oc":     float(daily_h1.mean() / daily_std_oc),
        },
        "H2_daily_IV": {
            "implied_per_5min_std_mean": float(sigma_h2.mean()),
            "implied_daily_std_mean":    float(daily_h2.mean()),
            "ratio_to_empirical_oc":     float(daily_h2.mean() / daily_std_oc),
        },
        "H3_per_bar_var": {
            "implied_per_5min_std_mean": float(sigma_h3.mean()),
            "implied_daily_std_mean":    float(daily_h3.mean()),
            "ratio_to_empirical_oc":     float(daily_h3.mean() / daily_std_oc),
        },
    }
    # Pick whichever hypothesis gives ratio closest to 1.0 IN LOG SPACE
    # (so 0.5x and 2x are equidistant from 1x, and a 0.13 ratio is FAR
    # from 1, not closer than 2.24).
    best = min(("H1_per_second_RV", "H2_daily_IV", "H3_per_bar_var"),
               key=lambda h: abs(np.log(out[h]["ratio_to_empirical_oc"])))
    out["best_hypothesis"] = best
    # Also report a model-free direct estimate for sigma_v: empirical
    # daily OC std divided by sqrt(N_bars). No IVol assumption needed.
    out["sigma_v_from_empirical_std"] = float(daily_std_oc / np.sqrt(n.mean()))
    return out


# ─── Direct calibration ────────────────────────────────────────────────

@dataclass
class DirectParams:
    """Parameters that come directly from data without simulation."""
    v0: float                      # OPrice for the chosen day
    sigma_v_per_5min: float        # from IVol_t_m
    mu_v_per_5min: float           # from log(DPrice/OPrice) / N_bars

    # Optional jump-component estimates (from Lee-Mykland 2008 style filter,
    # very simple version using the daily-return tail beyond 3 sigma)
    jump_lambda_per_5min: float = 0.0
    jump_mean_per_5min: float = 0.0
    jump_std_per_5min: float = 0.0


def extract_direct_params(csv_path: str,
                          source: str = "empirical",
                          ivol_hypothesis: str = "H1_per_second_RV",
                          day_index: Optional[int] = None,
                          aggregate: bool = True) -> DirectParams:
    """
    Pull v0, sigma_v, mu_v from WRDS Intraday Indicators.

    Two sources for sigma_v:
      'empirical' : use empirical daily-OC log-return std divided by
                    sqrt(N_bars). Model-free, no IVol assumption needed.
                    Primary calibration source.
      'ivol'      : use IVol_t_m under the chosen unit hypothesis.
                    Secondary; helpful for cross-validation. Note that
                    IVol estimators tend to overshoot the empirical OC
                    std for SPY (microstructure-noise bias).

    aggregate=True takes dataset-wide median per-bar sigma; False uses
    a specific day_index.
    """
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    n = df["NObsUsed1"].values
    sec_per_bar = 5 * 60
    drift_per_bar = np.log(df["DPrice"] / df["OPrice"]) / n
    log_oc = np.log(df["DPrice"] / df["OPrice"])

    if source == "ivol":
        iv = df["IVol_t_m"].values
        if ivol_hypothesis == "H1_per_second_RV":
            sigma_per_bar = np.sqrt(iv * sec_per_bar)
        elif ivol_hypothesis == "H2_daily_IV":
            sigma_per_bar = np.sqrt(iv / n)
        elif ivol_hypothesis == "H3_per_bar_var":
            sigma_per_bar = np.sqrt(iv)
        else:
            raise ValueError(f"unknown hypothesis {ivol_hypothesis!r}")
        sigma_v = float(np.median(sigma_per_bar)) if aggregate else float(sigma_per_bar[day_index])
    elif source == "empirical":
        # Constant-vol assumption: per-bar variance = daily variance / N_bars
        sigma_v = float(log_oc.std() / np.sqrt(n.mean()))
    else:
        raise ValueError(f"unknown source {source!r} (use 'empirical' or 'ivol')")

    if aggregate:
        return DirectParams(
            v0=float(df["OPrice"].mean()),
            sigma_v_per_5min=sigma_v,
            mu_v_per_5min=float(drift_per_bar.mean()),
        )
    assert day_index is not None
    return DirectParams(
        v0=float(df["OPrice"].iloc[day_index]),
        sigma_v_per_5min=sigma_v,
        mu_v_per_5min=float(drift_per_bar.iloc[day_index]),
    )


# ─── Stylized-fact targets ─────────────────────────────────────────────

@dataclass
class Moments:
    """Stylized-fact moment vector (Cont 2001 + microstructure)."""
    ret_std: float
    ret_kurtosis_excess: float
    acf_r_lag1: float
    acf_r_lag5: float
    acf_abs_r_lag1: float
    acf_abs_r_lag5: float
    acf_abs_r_lag20: float

    def to_array(self) -> np.ndarray:
        return np.array([self.ret_std, self.ret_kurtosis_excess,
                         self.acf_r_lag1, self.acf_r_lag5,
                         self.acf_abs_r_lag1, self.acf_abs_r_lag5,
                         self.acf_abs_r_lag20])


def _acf(x: np.ndarray, k: int) -> float:
    if len(x) <= k:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).sum())
    if var == 0.0:
        return 0.0
    return float((x[:-k] * x[k:]).sum() / var)


def compute_moments(log_returns: np.ndarray) -> Moments:
    r = np.asarray(log_returns)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return Moments(*([float("nan")] * 7))
    abs_r = np.abs(r)
    sd = float(r.std())
    if sd > 0:
        kurt = float((((r - r.mean()) / sd) ** 4).mean() - 3.0)
    else:
        kurt = 0.0
    return Moments(
        ret_std=sd,
        ret_kurtosis_excess=kurt,
        acf_r_lag1=_acf(r, 1),
        acf_r_lag5=_acf(r, 5),
        acf_abs_r_lag1=_acf(abs_r, 1),
        acf_abs_r_lag5=_acf(abs_r, 5),
        acf_abs_r_lag20=_acf(abs_r, 20),
    )


def empirical_moments_daily(csv_path: str) -> Moments:
    """Compute Cont 2001 stylized-fact moments from daily WRDS data."""
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    log_ret = np.log(df["DPrice"]).diff().dropna().values
    return compute_moments(log_ret)


# ─── Indirect calibration: stylized-fact minimization ──────────────────

def _simulate_moments(params: ModelParams, seed: int, n_steps: int,
                      n_runs: int, bars_per_day: int = 78) -> Moments:
    """Simulate ABM, extract end-of-day mid-prices, compute moments on
    DAILY log-returns to match the WRDS daily target. Ratios across
    n_runs are concatenated for a larger sample.
    """
    daily_rs = []
    for s in range(seed, seed + n_runs):
        traders = build_traders(params, seed=s)
        sim = Simulation(params, traders, seed=s)
        h = sim.run(n_steps)
        mid = pd.Series(h["mid_price"]).ffill().bfill().values
        # Take mid at end of each trading day (last bar of day i)
        n_days = n_steps // bars_per_day
        if n_days < 2:
            continue
        eod = mid[bars_per_day - 1::bars_per_day][:n_days]
        daily_log_ret = np.diff(np.log(eod))
        daily_rs.append(daily_log_ret)
    if not daily_rs:
        return Moments(*([float("nan")] * 7))
    all_r = np.concatenate(daily_rs)
    return compute_moments(all_r)


# ─── Multi-regime calibration ──────────────────────────────────────────

def make_params(direct: DirectParams,
                ft_sigma_rel: float, mt_sigma_rel: float, mt_lambda: float,
                kappa_v: float, theta_v: float, xi_v: float,
                mm_half_spread_bps: float = 30.0,
                mm_inventory_skew_bps: float = 0.5) -> ModelParams:
    return ModelParams(
        n_zi=10, n_fundamental=0, n_momentum=0,
        n_bcm=10, n_bcm_mm=4, n_bcm_with_clients=3, n_nbcm=10,
        v0=direct.v0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10, zi_offset_p=0.5, zi_offset_max=20,
        ft_sigma_rel=ft_sigma_rel, mt_sigma_rel=mt_sigma_rel,
        mt_lambda_ewma=mt_lambda, mt_threshold=1e-4,
        mu_v=direct.mu_v_per_5min, sigma_v=direct.sigma_v_per_5min,
        kappa_v=kappa_v, theta_v=theta_v, xi_v=xi_v,
        jump_lambda=0.0, jump_mean=0.0, jump_std=0.01,
        mm_half_spread_bps=mm_half_spread_bps, mm_qty=50,
        mm_inventory_skew_bps=mm_inventory_skew_bps,
    )


def _moments_diff(target: Moments, sim: Moments,
                  eps: float = 0.05) -> np.ndarray:
    """Per-moment normalised differences for use in the joint loss.

    Each diff is (sim - target) / max(|target|, eps), so each moment
    contributes equally regardless of its natural magnitude. eps avoids
    blow-up when target is near zero (e.g., ACF values can be ~0).
    """
    t = target.to_array()
    s = sim.to_array()
    diff = np.zeros_like(t)
    for i, (ti, si) in enumerate(zip(t, s)):
        if not (np.isfinite(ti) and np.isfinite(si)):
            diff[i] = 1e2
            continue
        denom = max(abs(ti), eps)
        diff[i] = (si - ti) / denom
    return diff


def regime_loss(theta_vec: np.ndarray,
                calm_target: Moments, stress_target: Moments,
                calm_direct: DirectParams, stress_direct: DirectParams,
                n_steps: int = 78 * 30, n_runs: int = 2,
                seed: int = 42,
                regime_weights: tuple = (1.0, 1.0),
                moment_weights: Optional[np.ndarray] = None) -> float:
    """Joint loss across calm and stressed regimes.

    theta_vec = [ft_sigma, mt_sigma, mt_lambda, kappa_v, log10(xi_v)].
    theta_v is set per-regime from each regime's direct sigma_v
    (theta_v = direct.sigma_v_per_5min ** 2). xi_v passed log10-scaled
    for sampling efficiency; mapped back here.
    """
    ft_sigma_rel, mt_sigma_rel, mt_lambda, kappa_v, log10_xi_v = theta_vec
    xi_v = float(10 ** log10_xi_v)

    if moment_weights is None:
        # std, kurt, acf_r1, acf_r5, acf_|r|1, acf_|r|5, acf_|r|20
        # Per-moment normalisation already balances scales; weights here
        # just emphasise vol clustering moments per Cont 2001 fact #3.
        moment_weights = np.array([1.0, 1.0, 0.5, 0.3, 2.0, 1.5, 1.0])

    losses = []
    for w, target, direct in (
        (regime_weights[0], calm_target,   calm_direct),
        (regime_weights[1], stress_target, stress_direct),
    ):
        theta_v_regime = direct.sigma_v_per_5min ** 2
        params = make_params(direct, ft_sigma_rel, mt_sigma_rel, mt_lambda,
                              kappa_v, theta_v_regime, xi_v)
        sim_m = _simulate_moments(params, seed=seed, n_steps=n_steps, n_runs=n_runs)
        diff = _moments_diff(target, sim_m)
        losses.append(w * float(np.sum(moment_weights * diff * diff)))
    return sum(losses)


# ─── Latin Hypercube Sampling + local refinement ───────────────────────

def lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified Latin Hypercube Sampling on [0,1]^n_dims."""
    out = np.zeros((n_samples, n_dims))
    strata = np.arange(n_samples) / n_samples
    for d in range(n_dims):
        offsets = rng.uniform(0, 1.0 / n_samples, size=n_samples)
        col = strata + offsets
        rng.shuffle(col)
        out[:, d] = col
    return out


# 5-D search space: [ft_sigma_rel, mt_sigma_rel, mt_lambda, kappa_v, log10(xi_v)]
# ft_sigma_rel and mt_sigma_rel are V-fractions; sweep 0.05% to 5%.
SEARCH_BOUNDS = np.array([
    [0.0005, 0.05],  # ft_sigma_rel  (linear; 5 bps to 500 bps)
    [0.0005, 0.05],  # mt_sigma_rel  (linear)
    [0.50, 0.99],    # mt_lambda_ewma (linear)
    [0.001, 1.0],    # kappa_v   (linear)
    [-7.0, -3.0],    # log10(xi_v) -> xi in [1e-7, 1e-3]
])


def _u_to_theta(u_row: np.ndarray) -> np.ndarray:
    """Map LHS [0,1]^5 sample to actual param values."""
    lo = SEARCH_BOUNDS[:, 0]
    hi = SEARCH_BOUNDS[:, 1]
    return lo + u_row * (hi - lo)


def calibrate_lhs(csv_calm: str = "data/thesis_data_calm.csv",
                  csv_stress: str = "data/thesis_data_stressed.csv",
                  n_lhs: int = 60,
                  n_steps: int = 78 * 30,
                  n_runs: int = 2,
                  seed: int = 42) -> dict:
    """Latin Hypercube Sampling over 5 free behavioural+CIR params,
    fit jointly to calm + stressed daily-return moments.
    """
    calm_direct = extract_direct_params(csv_calm)
    stress_direct = extract_direct_params(csv_stress)
    calm_target = empirical_moments_daily(csv_calm)
    stress_target = empirical_moments_daily(csv_stress)

    rng = np.random.default_rng(seed)
    samples_u = lhs(n_lhs, 5, rng)

    rows = []
    for i in range(n_lhs):
        theta = _u_to_theta(samples_u[i])
        loss = regime_loss(theta, calm_target, stress_target,
                            calm_direct, stress_direct,
                            n_steps=n_steps, n_runs=n_runs, seed=seed + i)
        rows.append({
            "ft_sigma":  float(theta[0]),
            "mt_sigma":  float(theta[1]),
            "mt_lambda": float(theta[2]),
            "kappa_v":   float(theta[3]),
            "log10_xi":  float(theta[4]),
            "xi_v":      float(10 ** theta[4]),
            "loss":      float(loss),
        })

    df = pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return {
        "calm_target":    asdict(calm_target),
        "stress_target":  asdict(stress_target),
        "calm_direct":    asdict(calm_direct),
        "stress_direct":  asdict(stress_direct),
        "best": best,
        "all_results": df,
    }


def _objective(target: Moments, sim: Moments,
               weights: Optional[np.ndarray] = None) -> float:
    """Weighted L2 distance between target and simulated moment vectors,
    using log-ratio scaling so vol terms aren't dominated by units."""
    t = target.to_array()
    s = sim.to_array()
    valid = np.isfinite(t) & np.isfinite(s)
    if not valid.any():
        return float("inf")
    if weights is None:
        weights = np.array([2.0, 1.0, 1.0, 0.5, 1.5, 1.0, 0.5])
    # std uses log-ratio (scale-free), the rest are already ~ O(1)
    diff = np.zeros_like(t)
    diff[0] = np.log(s[0] / t[0]) if t[0] > 0 and s[0] > 0 else 1e3
    diff[1:] = s[1:] - t[1:]
    diff[~valid] = 1e3
    return float(np.sum(weights * diff * diff))


def calibrate_grid(csv_path: str = "data/thesis_data_calm.csv",
                   sigma_source: str = "empirical",
                   ivol_hypothesis: str = "H1_per_second_RV",
                   n_steps: int = 78 * 60,   # 60 trading days per run
                   n_runs: int = 3,
                   seed: int = 42) -> dict:
    """
    Direct calibration of (v0, sigma_v, mu_v) from data; grid search
    over (ft_sigma, mt_sigma, mt_lambda_ewma) by stylized-fact min.

    Grid is small by design (Stage 8 first pass); refine with Bayesian
    optimisation later if needed.
    """
    direct = extract_direct_params(csv_path, source=sigma_source,
                                   ivol_hypothesis=ivol_hypothesis)
    target = empirical_moments_daily(csv_path)

    # Coarse grid
    ft_sigma_grid = [0.1, 0.25, 0.5, 1.0]
    mt_sigma_grid = [0.1, 0.25, 0.5, 1.0]
    mt_lambda_grid = [0.7, 0.9, 0.95, 0.99]

    rows = []
    best = (None, float("inf"))
    for fts, mts, mtl in product(ft_sigma_grid, mt_sigma_grid, mt_lambda_grid):
        params = ModelParams(
            n_zi=10, n_fundamental=0, n_momentum=0,
            n_bcm=10, n_nbcm=10, n_bcm_with_clients=3,
            v0=direct.v0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
            zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
            zi_qty_min=1, zi_qty_max=10,
            zi_offset_p=0.5, zi_offset_max=20,
            ft_sigma=fts, mt_sigma=mts, mt_lambda_ewma=mtl,
            mt_threshold=1e-4,
            mu_v=direct.mu_v_per_5min, sigma_v=direct.sigma_v_per_5min,
            jump_lambda=0.0, jump_mean=0.0, jump_std=0.01,
        )
        sim_m = _simulate_moments(params, seed=seed,
                                  n_steps=n_steps, n_runs=n_runs)
        loss = _objective(target, sim_m)
        rows.append({"ft_sigma": fts, "mt_sigma": mts, "mt_lambda": mtl,
                     **asdict(sim_m), "loss": loss})
        if loss < best[1]:
            best = ({"ft_sigma": fts, "mt_sigma": mts, "mt_lambda": mtl,
                     "sim_moments": sim_m, "loss": loss}, loss)

    return {
        "direct": asdict(direct),
        "target_moments": asdict(target),
        "best": best[0],
        "grid_results": pd.DataFrame(rows).sort_values("loss"),
    }


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        out = verify_ivol_units()
        print("WRDS IVol_t_m unit-hypothesis comparison")
        print("=" * 60)
        print(f"empirical daily std (open->close): {out['empirical_daily_std_open_close']:.4e}")
        print(f"empirical daily std (close->close): {out['empirical_daily_std_close_close']:.4e}")
        print()
        for h in ["H1_per_second_RV", "H2_daily_IV", "H3_per_bar_var"]:
            d = out[h]
            print(f"{h}:")
            print(f"  implied 5-min std mean   : {d['implied_per_5min_std_mean']:.4e}")
            print(f"  implied daily std mean   : {d['implied_daily_std_mean']:.4e}")
            print(f"  ratio vs empirical (OC)  : {d['ratio_to_empirical_oc']:.3f}")
        print()
        print(f"Best fit (log-ratio): {out['best_hypothesis']}")
        print(f"Direct empirical sigma_v (5-min): {out['sigma_v_from_empirical_std']:.4e}")
        print()
        print("Note: IVol estimators tend to overshoot empirical OC std")
        print("for SPY (microstructure-noise bias). For calibration, use")
        print("source='empirical' as primary; treat IVol-derived as cross-check.")
    elif cmd == "direct":
        params = extract_direct_params("data/thesis_data_calm.csv")
        print("Direct calibration (calm regime, dataset-wide):")
        for k, v in asdict(params).items():
            print(f"  {k:30}  {v:.6e}")
    elif cmd == "calibrate":
        res = calibrate_grid()
        print("=" * 60)
        print("DIRECT (calm):")
        for k, v in res["direct"].items():
            print(f"  {k:30}  {v:.6e}")
        print()
        print("EMPIRICAL TARGET MOMENTS:")
        for k, v in res["target_moments"].items():
            print(f"  {k:30}  {v:+.4f}")
        print()
        print(f"BEST FIT (loss={res['best']['loss']:.4f}):")
        for k in ("ft_sigma", "mt_sigma", "mt_lambda"):
            print(f"  {k:30}  {res['best'][k]}")
        print()
        print("Top 10 grid results:")
        print(res["grid_results"].head(10).to_string(index=False))
    elif cmd == "lhs":
        # Usage: python calibrate.py lhs [n_lhs] [n_days] [n_runs]
        n_lhs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        print(f"LHS calibration: {n_lhs} points, {n_days} days/run, {n_runs} runs/point")
        import time
        t0 = time.perf_counter()
        res = calibrate_lhs(n_lhs=n_lhs, n_steps=78 * n_days, n_runs=n_runs)
        elapsed = time.perf_counter() - t0
        os.makedirs("output", exist_ok=True)
        res["all_results"].to_csv("output/calibration_lhs.csv", index=False)
        print(f"\nwall: {elapsed:.1f}s ({elapsed/n_lhs:.2f}s/point)")
        print(f"\nSaved to output/calibration_lhs.csv")
        print(f"\nTOP 10:")
        print(res["all_results"].head(10).to_string(index=False))
        print(f"\nBEST PARAMETERS:")
        for k, v in res["best"].items():
            print(f"  {k:12} {v}")
    else:
        print(f"unknown cmd {cmd!r}; try: verify | direct | calibrate | lhs")
