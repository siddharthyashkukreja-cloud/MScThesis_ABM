"""
Calibration — Stage 8 (run BEFORE Stage 4 margin per project plan).

Pipeline:
  1. Direct calibration: v0, sigma_v, mu_v from WRDS daily aggregates.
  2. LHS sampling: generate (theta, moments) training data by simulating
     the ABM at N parameter vectors over D days x R seeds.
  3. XGBoost surrogate: train one XGBRegressor per moment dimension.
  4. Multi-start optimisation on the surrogate -> theta*.
  5. Validation: simulate at theta*, compare predicted vs actual moments.
  6. Feature importance: per-moment ranking of which params drive each
     stylized fact.

Citations (project priority chain):
  - ODD §Calibration for default parameter values
  - Cont (2001) "Empirical properties of asset returns" for moment targets
  - Krishnen ABM Liquidity Risk; Gao 2023 "Deeper Hedging" for surrogate
    matching of stylized facts
  - SMAC (Hutter et al. 2011); HFABM Cao 2024 §4.2 for ML-surrogate ABM
    calibration
  - Chen & Guestrin (2016) for XGBoost
  - Stein (1987), Helton-Davis (2003) for LHS

Requires: numpy, pandas. Optional but recommended on user's laptop:
  - xgboost  (for the surrogate; pip install xgboost)
  - scipy    (for L-BFGS-B; pip install scipy)
A pure-numpy fallback is provided when these aren't available, but XGBoost
is strongly recommended for thesis-quality calibration.
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model.globals import ModelParams
from model.simulation import Simulation
from run_simulation import build_traders


# ─── WRDS Intraday Indicators units verification ────────────────────────

def verify_ivol_units(csv_path: str = "data/thesis_data_calm.csv") -> dict:
    """Determine units of IVol_t_m by comparing implied vs empirical std."""
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    log_oc = np.log(df["DPrice"] / df["OPrice"])
    log_cc = np.log(df["DPrice"]).diff()
    daily_std_oc = float(log_oc.std())
    daily_std_cc = float(log_cc.std())

    iv = df["IVol_t_m"].values
    n = df["NObsUsed1"].values
    sec_per_bar = 5 * 60

    sigma_h1 = np.sqrt(iv * sec_per_bar)
    sigma_h2 = np.sqrt(iv / n)
    sigma_h3 = np.sqrt(iv)

    daily_h1 = sigma_h1 * np.sqrt(n)
    daily_h2 = sigma_h2 * np.sqrt(n)
    daily_h3 = sigma_h3 * np.sqrt(n)

    out = {
        "n_rows": len(df),
        "empirical_daily_std_open_close": daily_std_oc,
        "empirical_daily_std_close_close": daily_std_cc,
        "H1_per_second_RV": {"implied_per_5min_std_mean": float(sigma_h1.mean()),
                              "implied_daily_std_mean": float(daily_h1.mean()),
                              "ratio_to_empirical_oc": float(daily_h1.mean() / daily_std_oc)},
        "H2_daily_IV":      {"implied_per_5min_std_mean": float(sigma_h2.mean()),
                              "implied_daily_std_mean": float(daily_h2.mean()),
                              "ratio_to_empirical_oc": float(daily_h2.mean() / daily_std_oc)},
        "H3_per_bar_var":   {"implied_per_5min_std_mean": float(sigma_h3.mean()),
                              "implied_daily_std_mean": float(daily_h3.mean()),
                              "ratio_to_empirical_oc": float(daily_h3.mean() / daily_std_oc)},
    }
    out["best_hypothesis"] = min(
        ("H1_per_second_RV", "H2_daily_IV", "H3_per_bar_var"),
        key=lambda h: abs(np.log(out[h]["ratio_to_empirical_oc"])),
    )
    out["sigma_v_from_empirical_std"] = float(daily_std_oc / np.sqrt(n.mean()))
    return out


# ─── Direct calibration ────────────────────────────────────────────────

@dataclass
class DirectParams:
    v0: float
    sigma_v_per_5min: float
    mu_v_per_5min: float
    jump_lambda_per_5min: float = 0.0385
    jump_mean_per_5min: float = 0.0
    jump_std_per_5min: float = 0.01


def extract_direct_params(csv_path: str,
                          source: str = "empirical",
                          aggregate: bool = True) -> DirectParams:
    """Pull v0, sigma_v, mu_v from WRDS Intraday Indicators.

    sigma_v source:
      'empirical' (recommended): empirical daily-OC std / sqrt(N_bars).
                                 No IVol unit assumption needed.
      'ivol_h1':                 sqrt(IVol_t_m * 300) -- per-second RV.
                                 Tends to overshoot empirical by ~2x.
    """
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    n = df["NObsUsed1"].values
    drift_per_bar = np.log(df["DPrice"] / df["OPrice"]) / n
    log_oc = np.log(df["DPrice"] / df["OPrice"])

    if source == "empirical":
        sigma_v = float(log_oc.std() / np.sqrt(n.mean()))
    elif source == "ivol_h1":
        sigma_v = float(np.median(np.sqrt(df["IVol_t_m"].values * 300)))
    else:
        raise ValueError(f"unknown source {source!r}")

    return DirectParams(
        v0=float(df["OPrice"].mean()) if aggregate else float(df["OPrice"].iloc[0]),
        sigma_v_per_5min=sigma_v,
        mu_v_per_5min=float(drift_per_bar.mean()),
    )


# ─── Stylized-fact moments ──────────────────────────────────────────────

@dataclass
class Moments:
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


MOMENT_KEYS = ["ret_std", "ret_kurtosis_excess",
               "acf_r_lag1", "acf_r_lag5",
               "acf_abs_r_lag1", "acf_abs_r_lag5", "acf_abs_r_lag20"]


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
    kurt = float((((r - r.mean()) / sd) ** 4).mean() - 3.0) if sd > 0 else 0.0
    return Moments(
        ret_std=sd, ret_kurtosis_excess=kurt,
        acf_r_lag1=_acf(r, 1), acf_r_lag5=_acf(r, 5),
        acf_abs_r_lag1=_acf(abs_r, 1), acf_abs_r_lag5=_acf(abs_r, 5),
        acf_abs_r_lag20=_acf(abs_r, 20),
    )


def empirical_moments_daily(csv_path: str) -> Moments:
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    log_ret = np.log(df["DPrice"]).diff().dropna().values
    return compute_moments(log_ret)


# ─── ABM simulation -> moments ─────────────────────────────────────────

def make_params(direct: DirectParams,
                ft_sigma_rel: float, mt_sigma_rel: float, mt_lambda: float,
                kappa_v: float, theta_v: float, xi_v: float) -> ModelParams:
    """ModelParams from direct + behavioural params; uses current 98-agent
    real-bank-tier population (15 BCM + 5 NBCM + 78 clients)."""
    return ModelParams(
        n_zi=0, n_fundamental=0, n_momentum=0,
        n_bcm=15, n_bcm_mm=8, n_bcm_with_clients=8, n_nbcm=5,
        clients_per_book=6, client_book_ft=2, client_book_mt=2, client_book_zi=2,
        v0=direct.v0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10, zi_offset_p=0.5, zi_offset_max=20,
        ft_sigma_rel=ft_sigma_rel, mt_sigma_rel=mt_sigma_rel,
        mt_lambda_ewma=mt_lambda, mt_threshold=1e-4,
        mu_v=direct.mu_v_per_5min, sigma_v=direct.sigma_v_per_5min,
        kappa_v=kappa_v, theta_v=theta_v, xi_v=xi_v,
        jump_lambda=direct.jump_lambda_per_5min,
        jump_mean=direct.jump_mean_per_5min,
        jump_std=direct.jump_std_per_5min,
        mm_half_spread_bps=30.0, mm_qty=50, mm_inventory_skew_bps=0.5,
    )


def simulate_moments(params: ModelParams, seed: int, n_steps: int,
                     n_runs: int, bars_per_day: int = 78) -> Moments:
    """Run sim and extract moments on DAILY end-of-day log-returns."""
    daily_rs = []
    for s in range(seed, seed + n_runs):
        traders = build_traders(params, seed=s)
        sim = Simulation(params, traders, seed=s)
        h = sim.run(n_steps)
        mid = pd.Series(h["mid_price"]).ffill().bfill().values
        n_days = n_steps // bars_per_day
        if n_days < 2:
            continue
        eod = mid[bars_per_day - 1::bars_per_day][:n_days]
        daily_rs.append(np.diff(np.log(eod)))
    if not daily_rs:
        return Moments(*([float("nan")] * 7))
    return compute_moments(np.concatenate(daily_rs))


# ─── Latin Hypercube Sampling ───────────────────────────────────────────

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
SEARCH_BOUNDS = np.array([
    [0.0005, 0.05],  # ft_sigma_rel  (5 to 500 bps of V)
    [0.0005, 0.05],  # mt_sigma_rel
    [0.50, 0.99],    # mt_lambda_ewma
    [0.001, 1.0],    # kappa_v
    [-7.0, -3.0],    # log10(xi_v) -> xi in [1e-7, 1e-3]
])
PARAM_NAMES = ["ft_sigma_rel", "mt_sigma_rel", "mt_lambda_ewma",
               "kappa_v", "log10_xi_v"]


def _u_to_theta(u: np.ndarray) -> np.ndarray:
    return SEARCH_BOUNDS[:, 0] + u * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])


def generate_training_data(calm_direct: DirectParams,
                           stress_direct: DirectParams,
                           n_lhs: int = 200, n_steps: int = 78 * 90,
                           n_runs: int = 3, seed: int = 42) -> pd.DataFrame:
    """LHS-sample theta, simulate calm + stressed, return training DataFrame.

    Each row: [theta_5d, calm moments, stressed moments].
    Used as training data for the XGBoost surrogate.
    """
    rng = np.random.default_rng(seed)
    samples_u = lhs(n_lhs, 5, rng)
    rows = []
    for i in range(n_lhs):
        theta = _u_to_theta(samples_u[i])
        ft_rel, mt_rel, lam, kappa, log10_xi = theta
        xi = float(10 ** log10_xi)

        params_calm = make_params(calm_direct, ft_rel, mt_rel, lam,
                                  kappa, calm_direct.sigma_v_per_5min ** 2, xi)
        params_str  = make_params(stress_direct, ft_rel, mt_rel, lam,
                                  kappa, stress_direct.sigma_v_per_5min ** 2, xi)
        m_calm = simulate_moments(params_calm, seed=seed + i,
                                  n_steps=n_steps, n_runs=n_runs)
        m_str  = simulate_moments(params_str, seed=seed + i,
                                  n_steps=n_steps, n_runs=n_runs)
        row = {"ft_sigma_rel": ft_rel, "mt_sigma_rel": mt_rel,
               "mt_lambda_ewma": lam, "kappa_v": kappa,
               "log10_xi_v": log10_xi}
        for k, v in asdict(m_calm).items():
            row[f"calm_{k}"] = v
        for k, v in asdict(m_str).items():
            row[f"stress_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ─── XGBoost surrogate (Form 1: direct surrogate) ──────────────────────

def train_xgb_surrogate(df: pd.DataFrame,
                        target_prefix: str = "calm_",
                        moment_keys: Optional[List[str]] = None) -> dict:
    """Train one XGBRegressor per moment, on (theta -> moment[regime]).

    Returns dict {moment_key: trained_model} + 'feature_names' entry.
    Requires xgboost. Print clear error if not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit(
            "xgboost is required for the surrogate calibration.\n"
            "Install: pip install xgboost\n"
            "(also recommended: pip install scipy)"
        )

    if moment_keys is None:
        moment_keys = MOMENT_KEYS
    X = df[PARAM_NAMES].values
    models = {"_feature_names": PARAM_NAMES, "_target_prefix": target_prefix}
    for m in moment_keys:
        col = f"{target_prefix}{m}"
        if col not in df.columns:
            continue
        y = df[col].values
        valid = np.isfinite(y)
        if valid.sum() < 10:
            continue
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, reg_alpha=0.01, reg_lambda=0.5,
            random_state=0, n_jobs=1,
        )
        model.fit(X[valid], y[valid])
        models[m] = model
    return models


def predict_moments(models: dict, theta: np.ndarray) -> Dict[str, float]:
    """Surrogate forward pass: theta -> predicted moments dict."""
    X = np.atleast_2d(theta)
    out = {}
    for m, model in models.items():
        if m.startswith("_"):
            continue
        out[m] = float(model.predict(X)[0])
    return out


def surrogate_loss(theta: np.ndarray, models: dict, target: Moments,
                   weights: Optional[np.ndarray] = None,
                   eps: float = 0.05) -> float:
    """Weighted L2 distance from surrogate-predicted moments to target."""
    if weights is None:
        weights = np.array([1.0, 1.0, 0.5, 0.3, 2.0, 1.5, 1.0])
    pred = predict_moments(models, theta)
    target_arr = target.to_array()
    diff = []
    for i, k in enumerate(MOMENT_KEYS):
        if k not in pred:
            diff.append(0.0)
            continue
        denom = max(abs(target_arr[i]), eps)
        diff.append((pred[k] - target_arr[i]) / denom)
    diff = np.array(diff)
    return float(np.sum(weights * diff * diff))


def joint_surrogate_loss(theta: np.ndarray, calm_models: dict, stress_models: dict,
                         calm_target: Moments, stress_target: Moments,
                         weights: Optional[np.ndarray] = None,
                         regime_weights: Tuple[float, float] = (1.0, 1.0)) -> float:
    """Sum of calm and stressed surrogate losses."""
    return (regime_weights[0] * surrogate_loss(theta, calm_models, calm_target, weights)
            + regime_weights[1] * surrogate_loss(theta, stress_models, stress_target, weights))


def optimise_surrogate(calm_models: dict, stress_models: dict,
                       calm_target: Moments, stress_target: Moments,
                       n_starts: int = 40, seed: int = 0) -> dict:
    """Multi-start optimisation on the joint surrogate loss.

    Uses scipy.optimize.minimize (L-BFGS-B) when available; falls back to
    a pure-numpy multi-start random + greedy-coord-descent local search.
    """
    rng = np.random.default_rng(seed)

    def objective(theta):
        return joint_surrogate_loss(theta, calm_models, stress_models,
                                    calm_target, stress_target)

    try:
        from scipy.optimize import minimize
        bounds = [tuple(b) for b in SEARCH_BOUNDS]
        best_x, best_f = None, float("inf")
        for _ in range(n_starts):
            x0 = SEARCH_BOUNDS[:, 0] + rng.uniform(size=5) * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_f:
                best_x, best_f = res.x, res.fun
        method = "scipy.L-BFGS-B"
    except ImportError:
        # Pure-numpy fallback: random + coord-descent
        best_x, best_f = None, float("inf")
        for _ in range(n_starts):
            x = SEARCH_BOUNDS[:, 0] + rng.uniform(size=5) * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
            f = objective(x)
            for _refine in range(50):
                d = rng.integers(0, 5)
                step = rng.normal() * 0.05 * (SEARCH_BOUNDS[d, 1] - SEARCH_BOUNDS[d, 0])
                xn = x.copy()
                xn[d] = float(np.clip(xn[d] + step, SEARCH_BOUNDS[d, 0], SEARCH_BOUNDS[d, 1]))
                fn = objective(xn)
                if fn < f:
                    x, f = xn, fn
            if f < best_f:
                best_x, best_f = x, f
        method = "pure-numpy random + coord descent"

    return {"theta_star": best_x, "loss_star": best_f, "method": method}


def feature_importance_report(models: dict) -> pd.DataFrame:
    """Per-moment ranking of which params drive that moment."""
    rows = []
    for moment, model in models.items():
        if moment.startswith("_"):
            continue
        for feat, imp in zip(PARAM_NAMES, model.feature_importances_):
            rows.append({"moment": moment, "param": feat, "importance": float(imp)})
    return (pd.DataFrame(rows)
            .pivot(index="moment", columns="param", values="importance")
            .reindex(MOMENT_KEYS))


def validate(theta_star: np.ndarray, calm_direct: DirectParams,
             stress_direct: DirectParams, n_steps: int = 78 * 60,
             n_runs: int = 3, seed: int = 999) -> dict:
    """Run actual simulator at theta_star; return predicted-vs-actual moments."""
    ft_rel, mt_rel, lam, kappa, log10_xi = theta_star
    xi = float(10 ** log10_xi)
    p_calm = make_params(calm_direct, ft_rel, mt_rel, lam, kappa,
                         calm_direct.sigma_v_per_5min ** 2, xi)
    p_str  = make_params(stress_direct, ft_rel, mt_rel, lam, kappa,
                         stress_direct.sigma_v_per_5min ** 2, xi)
    m_calm = simulate_moments(p_calm, seed=seed, n_steps=n_steps, n_runs=n_runs)
    m_str  = simulate_moments(p_str, seed=seed, n_steps=n_steps, n_runs=n_runs)
    return {"calm_actual_moments": asdict(m_calm),
            "stress_actual_moments": asdict(m_str)}


# ─── End-to-end XGBoost calibration ────────────────────────────────────

def calibrate_xgb(csv_calm: str = "data/thesis_data_calm.csv",
                  csv_stress: str = "data/thesis_data_stressed.csv",
                  n_lhs: int = 200, n_days: int = 90, n_runs: int = 3,
                  seed: int = 42, save_dir: str = "output") -> dict:
    """End-to-end XGBoost surrogate calibration.

    1. Direct calibration of v0, sigma_v, mu_v per regime.
    2. LHS-generate (theta, moments) training data.
    3. Train one XGBRegressor per moment for each regime.
    4. Multi-start optimise on joint surrogate loss -> theta*.
    5. Validate by running actual simulator at theta*.
    6. Feature importance report.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=== STEP 1: direct calibration ===")
    calm_d   = extract_direct_params(csv_calm)
    stress_d = extract_direct_params(csv_stress)
    print(f"  CALM  : v0={calm_d.v0:.2f}  sigma_v_5min={calm_d.sigma_v_per_5min:.4e}")
    print(f"  STRESS: v0={stress_d.v0:.2f}  sigma_v_5min={stress_d.sigma_v_per_5min:.4e}")

    calm_t   = empirical_moments_daily(csv_calm)
    stress_t = empirical_moments_daily(csv_stress)

    print(f"\n=== STEP 2: LHS sampling ({n_lhs} points x {n_days}d x {n_runs} seeds) ===")
    t0 = time.perf_counter()
    df = generate_training_data(calm_d, stress_d,
                                n_lhs=n_lhs, n_steps=78 * n_days,
                                n_runs=n_runs, seed=seed)
    csv_lhs = os.path.join(save_dir, "calibration_training.csv")
    df.to_csv(csv_lhs, index=False)
    print(f"  wall: {time.perf_counter() - t0:.1f}s; saved to {csv_lhs}")

    print(f"\n=== STEP 3: train XGBoost surrogate ===")
    t0 = time.perf_counter()
    calm_models   = train_xgb_surrogate(df, target_prefix="calm_")
    stress_models = train_xgb_surrogate(df, target_prefix="stress_")
    print(f"  trained {len([m for m in calm_models if not m.startswith('_')])} models per regime "
          f"in {time.perf_counter() - t0:.1f}s")

    print(f"\n=== STEP 4: multi-start optimisation on surrogate ===")
    t0 = time.perf_counter()
    opt = optimise_surrogate(calm_models, stress_models, calm_t, stress_t)
    print(f"  method: {opt['method']}")
    print(f"  loss*  : {opt['loss_star']:.4f}")
    print(f"  theta* :")
    for name, val in zip(PARAM_NAMES, opt["theta_star"]):
        if name == "log10_xi_v":
            print(f"    {name:18} {val:+.3f}  -> xi_v = {10**val:.3e}")
        else:
            print(f"    {name:18} {val:.6f}")
    print(f"  wall: {time.perf_counter() - t0:.1f}s")

    print(f"\n=== STEP 5: validate by simulating at theta* ===")
    t0 = time.perf_counter()
    valid = validate(opt["theta_star"], calm_d, stress_d)
    pred_calm = predict_moments(calm_models, opt["theta_star"])
    pred_str  = predict_moments(stress_models, opt["theta_star"])
    print(f"  wall: {time.perf_counter() - t0:.1f}s")

    def fmt_compare(label, target, predicted, actual):
        print(f"\n  {label}:")
        print(f"    {'moment':<24} {'target':>10} {'pred':>10} {'actual':>10}")
        target_arr = target.to_array()
        for i, k in enumerate(MOMENT_KEYS):
            t_ = target_arr[i]
            p_ = predicted.get(k, float("nan"))
            a_ = actual.get(k, float("nan"))
            print(f"    {k:<24} {t_:>+10.4f} {p_:>+10.4f} {a_:>+10.4f}")

    fmt_compare("CALM", calm_t, pred_calm, valid["calm_actual_moments"])
    fmt_compare("STRESS", stress_t, pred_str, valid["stress_actual_moments"])

    print(f"\n=== STEP 6: feature importance ===")
    fi_calm   = feature_importance_report(calm_models)
    fi_stress = feature_importance_report(stress_models)
    print("\n  CALM regime:")
    print(fi_calm.round(3).to_string())
    print("\n  STRESS regime:")
    print(fi_stress.round(3).to_string())

    out = {
        "direct": {"calm": asdict(calm_d), "stress": asdict(stress_d)},
        "targets": {"calm": asdict(calm_t), "stress": asdict(stress_t)},
        "theta_star": dict(zip(PARAM_NAMES, opt["theta_star"].tolist())),
        "loss_star": opt["loss_star"],
        "predicted_at_theta": {"calm": pred_calm, "stress": pred_str},
        "actual_at_theta": valid,
        "feature_importance_calm": fi_calm.to_dict(),
        "feature_importance_stress": fi_stress.to_dict(),
        "training_csv": csv_lhs,
        "n_lhs": n_lhs, "n_days": n_days, "n_runs": n_runs,
    }
    out_json = os.path.join(save_dir, "calibration_xgb.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nSaved to {out_json}")
    return out


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if cmd == "verify":
        out = verify_ivol_units()
        print("WRDS IVol_t_m unit-hypothesis comparison")
        print("=" * 60)
        print(f"empirical daily std (open->close):  {out['empirical_daily_std_open_close']:.4e}")
        print(f"empirical daily std (close->close): {out['empirical_daily_std_close_close']:.4e}")
        for h in ["H1_per_second_RV", "H2_daily_IV", "H3_per_bar_var"]:
            d = out[h]
            print(f"\n{h}:")
            print(f"  implied 5-min std mean : {d['implied_per_5min_std_mean']:.4e}")
            print(f"  implied daily std mean : {d['implied_daily_std_mean']:.4e}")
            print(f"  ratio vs empirical (OC): {d['ratio_to_empirical_oc']:.3f}")
        print(f"\nBest fit (log-ratio): {out['best_hypothesis']}")
        print(f"Direct empirical sigma_v (5-min): {out['sigma_v_from_empirical_std']:.4e}")

    elif cmd == "direct":
        for label, path in [("CALM", "data/thesis_data_calm.csv"),
                            ("STRESS", "data/thesis_data_stressed.csv")]:
            d = extract_direct_params(path)
            print(f"\n{label} direct calibration:")
            for k, v in asdict(d).items():
                print(f"  {k:30} {v:.6e}")

    elif cmd == "xgb":
        n_lhs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        calibrate_xgb(n_lhs=n_lhs, n_days=n_days, n_runs=n_runs)

    else:
        print(f"Unknown command: {cmd!r}")
        print(f"Usage:")
        print(f"  python calibrate.py verify              # WRDS IVol units check")
        print(f"  python calibrate.py direct              # v0, sigma_v, mu_v from data")
        print(f"  python calibrate.py xgb [N D R]         # full XGBoost calibration")
        print(f"                                          # (defaults: N=200 LHS, D=90 days, R=3 seeds)")
        print(f"")
        print(f"Recommended on laptop:")
        print(f"  pip install xgboost scipy")
        print(f"  python calibrate.py xgb 200 90 3")
        sys.exit(1)
