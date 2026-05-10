"""
Calibration via XGBoost surrogate (Form 1 — direct surrogate).

Workflow:
  1. Direct calibration: read v0, mu_v, sigma_v from data.
  2. LHS over 5-D behavioural+CIR space -> generate (theta, moments) pairs.
  3. Train XGBoost regressor per moment.
  4. Multi-start L-BFGS-B on surrogate loss to find theta*.
  5. Validate by running simulator at theta*; save calibrated params.

CLI:
  python calibrate.py verify              # WRDS IVol_t_m unit check
  python calibrate.py direct              # data-derived sigma_v, v0, mu_v
  python calibrate.py run [N D R]         # end-to-end (default 200 90 3)

Outputs:
  output/calibration_lhs.csv              # LHS training data
  output/calibrated_params.json           # final calibrated parameter set
"""
from __future__ import annotations
import os, json, time, sys
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from model.globals import ModelParams
from model.simulation import Simulation
from run_simulation import build_traders


# ─── 1. Empirical / direct calibration ────────────────────────────────

@dataclass
class DirectParams:
    v0: float
    sigma_v_per_5min: float
    mu_v_per_5min: float


def extract_direct_params(csv_path: str) -> DirectParams:
    """v0, sigma_v, mu_v from WRDS daily aggregates (model-free)."""
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    n = df["NObsUsed1"].values
    log_oc = np.log(df["DPrice"] / df["OPrice"])
    drift_per_bar = log_oc / n
    return DirectParams(
        v0=float(df["OPrice"].mean()),
        sigma_v_per_5min=float(log_oc.std() / np.sqrt(n.mean())),
        mu_v_per_5min=float(drift_per_bar.mean()),
    )


def verify_ivol_units(csv_path: str = "data/thesis_data_calm.csv") -> dict:
    """Cross-check IVol_t_m units against three hypotheses."""
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    log_oc = np.log(df["DPrice"] / df["OPrice"])
    daily_std_oc = float(log_oc.std())
    iv = df["IVol_t_m"].values
    n = df["NObsUsed1"].values
    sec_per_bar = 5 * 60

    sigmas = {
        "H1_per_second_RV":   np.sqrt(iv * sec_per_bar),
        "H2_daily_IV":        np.sqrt(iv / n),
        "H3_per_bar_var":     np.sqrt(iv),
    }
    out = {"empirical_daily_std_oc": daily_std_oc}
    for name, s in sigmas.items():
        implied_daily = (s * np.sqrt(n)).mean()
        out[name] = {
            "implied_5min_std": float(s.mean()),
            "implied_daily_std": float(implied_daily),
            "ratio_to_empirical": float(implied_daily / daily_std_oc),
        }
    out["best_hypothesis"] = min(
        sigmas.keys(),
        key=lambda h: abs(np.log(out[h]["ratio_to_empirical"]))
    )
    return out


# ─── 2. Moment computation ────────────────────────────────────────────

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
        return np.array([
            self.ret_std, self.ret_kurtosis_excess,
            self.acf_r_lag1, self.acf_r_lag5,
            self.acf_abs_r_lag1, self.acf_abs_r_lag5, self.acf_abs_r_lag20,
        ])


MOMENT_NAMES = list(Moments.__dataclass_fields__.keys())


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
        ret_std=sd,
        ret_kurtosis_excess=kurt,
        acf_r_lag1=_acf(r, 1),
        acf_r_lag5=_acf(r, 5),
        acf_abs_r_lag1=_acf(abs_r, 1),
        acf_abs_r_lag5=_acf(abs_r, 5),
        acf_abs_r_lag20=_acf(abs_r, 20),
    )


def empirical_moments_daily(csv_path: str) -> Moments:
    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    log_ret = np.log(df["DPrice"]).diff().dropna().values
    return compute_moments(log_ret)


# ─── 3. Simulation wrappers ───────────────────────────────────────────

def make_params(direct: DirectParams,
                ft_sigma_rel: float, mt_sigma_rel: float, mt_lambda: float,
                kappa_v: float, theta_v: float, xi_v: float) -> ModelParams:
    return ModelParams(
        n_zi=0, n_fundamental=0, n_momentum=0,
        n_bcm=15, n_bcm_mm=8, n_bcm_with_clients=8, n_nbcm=5,
        clients_per_book=6, client_book_ft=2, client_book_mt=2, client_book_zi=2,
        v0=direct.v0, tick_size=0.01, dt_minutes=5.0, order_ttl=2,
        zi_alpha=0.15, zi_mu=0.025, zi_delta=0.025,
        zi_qty_min=1, zi_qty_max=10,
        dir_qty_min=5, dir_qty_max=50,
        zi_offset_p=0.5, zi_offset_max=20,
        ft_sigma_rel=ft_sigma_rel, mt_sigma_rel=mt_sigma_rel,
        mt_lambda_ewma=mt_lambda, mt_threshold=1e-4,
        mu_v=direct.mu_v_per_5min, sigma_v=direct.sigma_v_per_5min,
        kappa_v=kappa_v, theta_v=theta_v, xi_v=xi_v,
        jump_lambda=0.0385, jump_mean=0.0, jump_std=0.01,
        mm_half_spread_bps=30.0, mm_qty=50, mm_inventory_skew_bps=0.5,
    )


def _simulate_moments(params: ModelParams, seed: int, n_steps: int,
                      n_runs: int, bars_per_day: int = 78) -> Moments:
    """Run simulator n_runs times, extract end-of-day mid prices, compute
    moments on daily log-returns concatenated across runs."""
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


def evaluate_theta(theta_vec: np.ndarray,
                   calm_direct: DirectParams,
                   stress_direct: DirectParams,
                   n_steps: int, n_runs: int, seed: int) -> dict:
    """Simulate calm + stressed regimes at theta; return both moments dicts."""
    ft, mt, lam, kappa, log10_xi = theta_vec
    xi = float(10 ** log10_xi)
    out = {}
    for label, direct in (("calm", calm_direct), ("stress", stress_direct)):
        params = make_params(direct, ft, mt, lam, kappa,
                             direct.sigma_v_per_5min ** 2, xi)
        m = _simulate_moments(params, seed=seed, n_steps=n_steps, n_runs=n_runs)
        out[label] = m
    return out


# ─── 4. LHS training-data generation ─────────────────────────────────

# 5-D search space: ft_sigma_rel, mt_sigma_rel, mt_lambda, kappa_v, log10(xi_v)
SEARCH_BOUNDS = np.array([
    [0.0005, 0.05],   # ft_sigma_rel  (5 bps to 500 bps)
    [0.0005, 0.05],   # mt_sigma_rel
    [0.50, 0.99],     # mt_lambda_ewma
    [0.001, 1.0],     # kappa_v
    [-7.0, -3.0],     # log10(xi_v) -> xi in [1e-7, 1e-3]
])
PARAM_NAMES = ["ft_sigma_rel", "mt_sigma_rel", "mt_lambda", "kappa_v", "log10_xi"]


def _lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n_samples, n_dims))
    strata = np.arange(n_samples) / n_samples
    for d in range(n_dims):
        col = strata + rng.uniform(0, 1.0 / n_samples, size=n_samples)
        rng.shuffle(col)
        out[:, d] = col
    return out


def _u_to_theta(u_row: np.ndarray) -> np.ndarray:
    return SEARCH_BOUNDS[:, 0] + u_row * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])


def run_lhs(n_lhs: int, n_steps: int, n_runs: int, seed: int = 42,
            calm_csv: str = "data/thesis_data_calm.csv",
            stress_csv: str = "data/thesis_data_stressed.csv") -> pd.DataFrame:
    """LHS over 5-D theta; for each, simulate calm + stressed and record moments."""
    calm_d = extract_direct_params(calm_csv)
    stress_d = extract_direct_params(stress_csv)

    rng = np.random.default_rng(seed)
    samples_u = _lhs(n_lhs, 5, rng)

    rows = []
    for i in range(n_lhs):
        theta = _u_to_theta(samples_u[i])
        out = evaluate_theta(theta, calm_d, stress_d, n_steps, n_runs, seed=seed + i)
        row = {p: float(v) for p, v in zip(PARAM_NAMES, theta)}
        row["xi_v"] = float(10 ** theta[4])
        for regime in ("calm", "stress"):
            for m_name in MOMENT_NAMES:
                row[f"{regime}_{m_name}"] = float(getattr(out[regime], m_name))
        rows.append(row)
    return pd.DataFrame(rows)


# ─── 5. XGBoost surrogate ─────────────────────────────────────────────

def train_xgb_surrogate(lhs_df: pd.DataFrame) -> dict:
    """Train one XGBRegressor per (regime, moment) target.

    Returns: dict[(regime, moment_name)] -> trained model
    Requires xgboost; raises ImportError if unavailable.
    """
    import xgboost as xgb
    X = lhs_df[PARAM_NAMES].values
    models = {}
    for regime in ("calm", "stress"):
        for m in MOMENT_NAMES:
            col = f"{regime}_{m}"
            if col not in lhs_df.columns:
                continue
            y = lhs_df[col].values
            mask = np.isfinite(y)
            if mask.sum() < 10:
                continue
            mdl = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.9,
                reg_alpha=0.01, reg_lambda=0.1, random_state=0,
            )
            mdl.fit(X[mask], y[mask])
            models[(regime, m)] = mdl
    return models


def surrogate_predict(models: dict, theta: np.ndarray) -> dict:
    """Predict moments at theta. Returns dict[regime][moment_name] -> value."""
    X = np.atleast_2d(theta)
    out = {"calm": {}, "stress": {}}
    for (regime, m), mdl in models.items():
        out[regime][m] = float(mdl.predict(X)[0])
    return out


def surrogate_loss(theta: np.ndarray,
                   models: dict,
                   target_calm: Moments,
                   target_stress: Moments,
                   moment_weights: Optional[np.ndarray] = None,
                   regime_weights: tuple = (1.0, 1.0),
                   eps: float = 0.05) -> float:
    """Weighted, per-moment-normalised L2 loss on surrogate predictions.

    Higher weights for vol-clustering moments (Cont 2001 fact #3).
    """
    if moment_weights is None:
        moment_weights = np.array([1.0, 1.0, 0.5, 0.3, 2.0, 1.5, 1.0])
    pred = surrogate_predict(models, theta)
    total = 0.0
    for w, regime, target in (
        (regime_weights[0], "calm",   target_calm),
        (regime_weights[1], "stress", target_stress),
    ):
        for j, m in enumerate(MOMENT_NAMES):
            t = float(getattr(target, m))
            s = pred[regime].get(m, float("nan"))
            if not np.isfinite(t) or not np.isfinite(s):
                continue
            denom = max(abs(t), eps)
            d = (s - t) / denom
            total += w * moment_weights[j] * d * d
    return total


def optimise_surrogate(models: dict,
                       target_calm: Moments,
                       target_stress: Moments,
                       n_starts: int = 30,
                       seed: int = 7) -> tuple:
    """Multi-start L-BFGS-B on surrogate. Returns (theta*, loss*).

    Falls back to random search if scipy is unavailable.
    """
    rng = np.random.default_rng(seed)

    def loss(t):
        return surrogate_loss(t, models, target_calm, target_stress)

    try:
        from scipy.optimize import minimize
        bounds = [tuple(b) for b in SEARCH_BOUNDS]
        best_x, best_f = None, float("inf")
        for _ in range(n_starts):
            x0 = SEARCH_BOUNDS[:, 0] + rng.random(5) * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_f:
                best_x, best_f = res.x, float(res.fun)
        return best_x, best_f
    except ImportError:
        # Fallback: dense random search if scipy unavailable
        n_random = max(2000, 100 * n_starts)
        u = rng.random((n_random, 5))
        thetas = SEARCH_BOUNDS[:, 0] + u * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
        losses = np.array([loss(t) for t in thetas])
        best = int(np.argmin(losses))
        return thetas[best], float(losses[best])


def feature_importance(models: dict) -> pd.DataFrame:
    """Per-(regime, moment) feature-importance ranking."""
    rows = []
    for (regime, m), mdl in models.items():
        for feat, imp in zip(PARAM_NAMES, mdl.feature_importances_):
            rows.append({"regime": regime, "moment": m, "param": feat,
                         "importance": float(imp)})
    return pd.DataFrame(rows)


# ─── 6. End-to-end pipeline ───────────────────────────────────────────

def run_calibration(n_lhs: int = 200, n_days: int = 90, n_runs: int = 3,
                    out_dir: str = "output", seed: int = 42) -> dict:
    """Full calibration: LHS -> XGBoost -> optimise -> validate -> save."""
    os.makedirs(out_dir, exist_ok=True)

    calm_csv = "data/thesis_data_calm.csv"
    stress_csv = "data/thesis_data_stressed.csv"
    calm_d = extract_direct_params(calm_csv)
    stress_d = extract_direct_params(stress_csv)
    target_calm = empirical_moments_daily(calm_csv)
    target_stress = empirical_moments_daily(stress_csv)

    # Step 1: LHS training data
    print(f"\n[1/4] Generating LHS training data: {n_lhs} pts × {n_days} days × {n_runs} seeds")
    t0 = time.perf_counter()
    lhs_df = run_lhs(n_lhs, n_steps=78 * n_days, n_runs=n_runs, seed=seed)
    lhs_csv = os.path.join(out_dir, "calibration_lhs.csv")
    lhs_df.to_csv(lhs_csv, index=False)
    print(f"      wall: {time.perf_counter()-t0:.1f}s; saved -> {lhs_csv}")

    # Step 2: train XGBoost
    print(f"\n[2/4] Training XGBoost surrogate")
    t0 = time.perf_counter()
    models = train_xgb_surrogate(lhs_df)
    print(f"      wall: {time.perf_counter()-t0:.1f}s; trained {len(models)} regressors")

    # Step 3: optimise
    print(f"\n[3/4] Optimising surrogate (multi-start L-BFGS-B)")
    t0 = time.perf_counter()
    theta_star, loss_star = optimise_surrogate(models, target_calm, target_stress)
    print(f"      wall: {time.perf_counter()-t0:.1f}s; surrogate loss at theta*: {loss_star:.4f}")
    print("      theta*:")
    for p, v in zip(PARAM_NAMES, theta_star):
        print(f"        {p:14}  {v:+.6e}")

    # Step 4: validate by running actual simulator
    print(f"\n[4/4] Validating: running simulator at theta*")
    t0 = time.perf_counter()
    actual = evaluate_theta(theta_star, calm_d, stress_d,
                            n_steps=78 * n_days, n_runs=n_runs, seed=seed + 9999)
    print(f"      wall: {time.perf_counter()-t0:.1f}s")
    pred = surrogate_predict(models, theta_star)

    print("\n              moment           target      surrogate     simulator")
    for regime, target in (("calm", target_calm), ("stress", target_stress)):
        for m in MOMENT_NAMES:
            t = float(getattr(target, m))
            s = pred[regime].get(m, float("nan"))
            a = float(getattr(actual[regime], m))
            print(f"      {regime:6}  {m:18}  {t:+9.4f}    {s:+9.4f}    {a:+9.4f}")

    # Save
    final = {
        "ft_sigma_rel": float(theta_star[0]),
        "mt_sigma_rel": float(theta_star[1]),
        "mt_lambda_ewma": float(theta_star[2]),
        "kappa_v": float(theta_star[3]),
        "xi_v": float(10 ** theta_star[4]),
        "calm_v0": float(calm_d.v0),
        "calm_sigma_v_per_5min": float(calm_d.sigma_v_per_5min),
        "calm_mu_v_per_5min": float(calm_d.mu_v_per_5min),
        "stress_v0": float(stress_d.v0),
        "stress_sigma_v_per_5min": float(stress_d.sigma_v_per_5min),
        "stress_mu_v_per_5min": float(stress_d.mu_v_per_5min),
        "n_lhs": n_lhs, "n_days": n_days, "n_runs": n_runs, "seed": seed,
        "surrogate_loss": float(loss_star),
        "validated_calm_moments": asdict(actual["calm"]),
        "validated_stress_moments": asdict(actual["stress"]),
        "target_calm_moments": asdict(target_calm),
        "target_stress_moments": asdict(target_stress),
    }
    out_path = os.path.join(out_dir, "calibrated_params.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved calibrated params -> {out_path}")

    # Feature importance
    fi = feature_importance(models)
    fi_path = os.path.join(out_dir, "calibration_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"Saved feature importance -> {fi_path}")
    return final


# ─── CLI ──────────────────────────────────────────────────────────────

def _print_verify(out: dict):
    print("WRDS IVol_t_m unit-hypothesis comparison")
    print("=" * 60)
    print(f"empirical daily std (open->close): {out['empirical_daily_std_oc']:.4e}")
    for h in ("H1_per_second_RV", "H2_daily_IV", "H3_per_bar_var"):
        d = out[h]
        print(f"\n{h}:")
        print(f"  implied 5-min std       : {d['implied_5min_std']:.4e}")
        print(f"  implied daily std       : {d['implied_daily_std']:.4e}")
        print(f"  ratio vs empirical OC   : {d['ratio_to_empirical']:.3f}")
    print(f"\nBest fit (log-ratio): {out['best_hypothesis']}")


def _print_direct(direct: DirectParams, label: str):
    print(f"Direct calibration ({label}):")
    for k, v in asdict(direct).items():
        print(f"  {k:25}  {v:.6e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "verify":
        _print_verify(verify_ivol_units())
    elif cmd == "direct":
        _print_direct(extract_direct_params("data/thesis_data_calm.csv"),  "calm")
        print()
        _print_direct(extract_direct_params("data/thesis_data_stressed.csv"), "stressed")
    elif cmd == "run":
        n_lhs  = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        run_calibration(n_lhs=n_lhs, n_days=n_days, n_runs=n_runs)
    else:
        print(f"unknown cmd {cmd!r}; try: verify | direct | run [N D R]")
