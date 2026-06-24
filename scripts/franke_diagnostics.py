"""Franke (2012) / HFABM (Gao 2022) goodness-of-fit on the locked theta: the
Moment Coverage Ratio (MCR) and the J specification p-value. Reuses calibrate.py's
moment battery, so it needs NO xgboost. The theta is read from globals.CALIBRATED
(or pass one inline), so it runs on whatever optimum is currently locked.

K = 10 point moments: ret_std, Hill, returns-ACF {1,5,10,20}, |r|-ACF {1,5,10,20}.
  * MCR  : per-moment CI = m_emp +/- 1.96 s_m (s_m = block-bootstrap SD); per-moment
           MCR = % of M long sim runs inside; joint = % inside all K at once.
  * J-test: J = (m_sim - m_emp)' Sigma^-1 (m_sim - m_emp); null J from the bootstrap
           resamples; p = fraction of null J above J_obs (not chi^2, per Franke).

Run from the repo root:
  python3 scripts/franke_diagnostics.py [calm|stressed] [M] [B]
Env knobs for a quick sandbox check: NDAYS (truncate the path), SEED0.
"""
from __future__ import annotations
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path[:0] = [_ROOT, _HERE]
import json
import numpy as np, pandas as pd
from model.globals import ModelParams, CALIBRATED, FV_CSV, day_start_steps
from model.run_simulation import build_traders
from model.simulation import Simulation
import calibrate as C

POINT = ["ret_std", "hill_tail_index",
         "acf_r_1", "acf_r_5", "acf_r_10", "acf_r_20",
         "acf_absr_1", "acf_absr_5", "acf_absr_10", "acf_absr_20"]
LABEL = {"ret_std": "ret SD", "hill_tail_index": "Hill",
         "acf_r_1": "ACF r(1)", "acf_r_5": "ACF r(5)", "acf_r_10": "ACF r(10)", "acf_r_20": "ACF r(20)",
         "acf_absr_1": "ACF|r|(1)", "acf_absr_5": "ACF|r|(5)", "acf_absr_10": "ACF|r|(10)", "acf_absr_20": "ACF|r|(20)"}


def _vec(m):
    return np.array([m[k] for k in POINT], float)


def _intraday(r, opens):
    mask = np.ones(len(r), bool)
    for o in opens:
        if 0 < o <= len(r):
            mask[o - 1] = False
    return r[mask]


def empirical_vec(regime):
    return _vec(C.compute_moments(C._empirical_returns(regime)))


def bootstrap_matrix(regime, B, block=390, seed=0):
    """B x K block-bootstrap moment matrix of the empirical returns (Kunsch)."""
    r = C._empirical_returns(regime); n = len(r); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); out = np.empty((B, len(POINT)))
    for b in range(B):
        starts = rng.integers(0, max(1, n - block), nb)
        x = np.concatenate([r[s:s + block] for s in starts])[:n]
        out[b] = _vec(C.compute_moments(x))
    return out


def sim_matrix(regime, theta, M, seed0=1000, ndays=None):
    """M x K matrix of moments from M independent sim runs at theta (market-only)."""
    df = pd.read_csv(FV_CSV[regime]); V = df["V_smooth"].to_numpy(float)
    SIG = df["sigma_t"].to_numpy(float) if "sigma_t" in df.columns else None
    n = len(V)
    if ndays:
        n = min(n, int(ndays) * 390)
    opens = [o for o in day_start_steps(regime) if o < n]
    out = np.empty((M, len(POINT)))
    for i in range(M):
        p = ModelParams(n_fundamental=40, n_momentum=20, n_zi=40, n_bcm=0, n_nbcm=0,
                        n_bcm_with_clients=0, v0=float(V[0]), tick_size=0.25, dt_minutes=1.0,
                        **theta, stressed=(regime == "stressed"))
        tr = build_traders(p, seed=seed0 + i)
        sim = Simulation(p, tr, seed=seed0 + i, ccp=None); sim.v_array = V[:n]
        if SIG is not None:
            sim.sigma_t_array = SIG[:n]
        sim.run(n)
        mid = np.array([x for x in sim.history["mid_price"]], float)
        mid = mid[np.isfinite(mid)]
        out[i] = _vec(C.compute_moments(_intraday(np.diff(np.log(mid)), opens)))
    return out


def _load_theta(regime, theta_path=None):
    """theta from a calibrate-output JSON (results[regime].theta_stage2) if given,
    else from globals.CALIBRATED."""
    if theta_path and os.path.exists(theta_path):
        j = json.load(open(theta_path))
        return dict(j["results"][regime]["theta_stage2"]), theta_path
    return dict(CALIBRATED[regime]), "globals.CALIBRATED"


def run(regime, M=60, B=2000, seed0=1000, ndays=None, theta_path=None):
    theta, src = _load_theta(regime, theta_path)
    print(f"[franke] {regime}: theta from {src}")
    m_emp = empirical_vec(regime)
    BM = bootstrap_matrix(regime, B, seed=0)
    s = BM.std(axis=0, ddof=1)                          # per-moment bootstrap SD
    SM = sim_matrix(regime, theta, M, seed0=seed0, ndays=ndays)

    lo, hi = m_emp - 1.96 * s, m_emp + 1.96 * s          # MCR
    inside = (SM >= lo) & (SM <= hi)
    per_moment = inside.mean(axis=0)
    joint = inside.all(axis=1).mean()
    boot_inside = (BM >= lo) & (BM <= hi)                # bootstrap self-coverage ceiling
    joint_ceiling = boot_inside.all(axis=1).mean()

    # J specification test (Franke & Westerhoff 2012, eq. 9): W = Sigma^-1; the
    # bootstrap gives the null J-distribution, J_0.95 is the critical value, and the
    # p-value is the fraction of model runs whose J falls below it (i.e. not rejected).
    Sigma = np.cov(BM, rowvar=False)
    Sinv = np.linalg.pinv(Sigma + 1e-12 * np.eye(len(POINT)))
    J_null = np.einsum("bi,ij,bj->b", BM - m_emp, Sinv, BM - m_emp)
    J95 = float(np.percentile(J_null, 95))
    J_model = np.einsum("ci,ij,cj->c", SM - m_emp, Sinv, SM - m_emp)
    p_val = float((J_model < J95).mean())
    J_obs = float(J_model.mean())

    print(f"\n=== {regime}  (theta={ {k: round(v,4) for k,v in theta.items()} })  M={M} B={B} ===")
    print(f"{'moment':12s} {'emp':>10s} {'sim mean':>10s} {'95% CI':>22s} {'cover%':>7s}")
    for i, k in enumerate(POINT):
        print(f"{LABEL[k]:12s} {m_emp[i]:10.5f} {SM[:,i].mean():10.5f}"
              f"   [{lo[i]:8.5f},{hi[i]:8.5f}] {100*per_moment[i]:6.1f}")
    print(f"\nmean per-moment coverage : {100*per_moment.mean():.1f}%")
    print(f"joint MCR                : {100*joint:.1f}%   (bootstrap self-coverage ceiling {100*joint_ceiling:.1f}%)")
    print(f"J p-value                : {p_val:.3f}   (J_obs={J_obs:.1f})")

    res = dict(regime=regime, theta=theta, M=M, B=B,
               moments=POINT, emp=m_emp.tolist(), sim_mean=SM.mean(axis=0).tolist(),
               s=s.tolist(), per_moment_mcr=per_moment.tolist(),
               joint_mcr=float(joint), joint_ceiling=float(joint_ceiling),
               J_obs=J_obs, p_value=p_val)
    os.makedirs("output/relock", exist_ok=True)
    with open(f"output/relock/franke_{regime}.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    reg = sys.argv[1] if len(sys.argv) > 1 else "stressed"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    theta_path = sys.argv[4] if len(sys.argv) > 4 else None
    run(reg, M=M, B=B, ndays=os.environ.get("NDAYS"), theta_path=theta_path)
