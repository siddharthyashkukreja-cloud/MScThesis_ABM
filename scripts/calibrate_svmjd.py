#!/usr/bin/env python3
"""
scripts/calibrate_svmjd.py - Reverse (indirect-inference) calibration of the
SV-MJD synthetic fundamental.

Instead of MEASURING the OU stochastic-volatility parameters from realized
measures (data/v_gbm.calibrate), this CHOOSES them to minimise the SAME
stylised-facts loss D(theta) the trading agents are calibrated against - a
simulated method of moments / indirect inference, run with the same
surrogate-assisted (XGBoost) two-stage search as scripts/calibrate.py.

Design
------
  * FREE   : the two OU stochastic-vol parameters that drive volatility
             clustering - alpha (mean-reversion speed) and sigma_vol (vol-of-vol).
             sigma_vol is reparameterised as a VARIANCE FRACTION
                 f = sv_var / sigma_d^2  in (0,1),   sigma_vol = sigma_d*sqrt(2*alpha*f),
             so the generator's anchoring constraint sv_var < sigma_d^2
             (E[sigma_t^2] = sigma_d^2) holds for EVERY candidate - no infeasible
             corners, and f is the readable "share of variance carried by the
             stochastic-vol layer".
  * FIXED  : sigma_d (diffusion level - pins return SD), the Merton jumps
             (lambda_J, delta - Mancini count; tails already match), the drift mu
             (period-return), and the empirical overnight-gap pool.
  * TARGET : the ABM-driven simulated-MID stylised facts. For each candidate the
             SV-MJD path is generated, the FIXED calibrated agents
             (globals.CALIBRATED) are run on it (bare-market population, no
             clearing - the calibration convention), and D is the
             Franke-standardised grouped-moment loss vs the empirical ES targets
             - identical machinery to the agent calibration.

Usage
-----
    python scripts/calibrate_svmjd.py --regime stressed \
        --n-sobol 48 --n-supplement 24 --n-grid 5 --n-runs 4 --n-days 40

Smoke run (seconds, just checks the loop wires up and responds):
    python scripts/calibrate_svmjd.py --regime stressed --smoke
"""
import os
os.environ.setdefault("FV_GBM", "1")   # MUST precede the calibrate import (read at import time)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.globals import ModelParams, CALIBRATED            # noqa: E402
from model.run_simulation import build_traders               # noqa: E402
from model.simulation import Simulation                      # noqa: E402
from data import v_gbm                                       # noqa: E402
import scripts.calibrate as C                                # noqa: E402
from scripts.calibrate import (                              # noqa: E402
    compute_moments, empirical_targets, empirical_moment_sd,
    _true_loss, MOMENT_NAMES,
)

PARAMS_JSON = REPO / "output" / "v_gbm_params.json"


# ---------------------------------------------------------------------------
# candidate -> SV-MJD path -> fixed-agent ABM -> stylised-facts loss
# ---------------------------------------------------------------------------
def _v0(regime: str, base: dict) -> float:
    V0 = getattr(C, "V0", None)
    if isinstance(V0, dict) and regime in V0:
        return float(V0[regime])
    return float(base[regime]["v0"])


def _build_fixed_params(regime: str, base: dict) -> ModelParams:
    """ModelParams with the FIXED calibrated agents on the SV-MJD fundamental,
    using the bare-market population (POP) and no clearing tier - exactly the
    convention scripts.calibrate.simulate_moments uses for the agent fit."""
    return ModelParams(
        **C.POP,
        v0=_v0(regime, base),
        tick_size=0.25,
        dt_minutes=1.0,
        fv_csv=C._fv_rel(regime),               # 'data/fv_gbm_{regime}.csv' under FV_GBM
        **CALIBRATED[regime],
        stressed=(regime == "stressed"),
    )


def sigma_vol_of(alpha: float, frac: float, sigma_d: float) -> float:
    """sigma_vol implied by (alpha, variance-fraction): sv_var = frac*sigma_d^2."""
    return float(sigma_d * np.sqrt(2.0 * alpha * frac))


def eval_candidate(alpha, frac, regime, base, target, sds,
                   n_days, n_runs, agent_seed, path_seed):
    """Generate the SV-MJD path for (alpha, frac), run the fixed agents on it,
    return (loss, moments). sigma_d / jumps / mu / overnight stay at their
    measured values; only the OU vol dynamics move."""
    sigma_d = float(base[regime]["sigma_d"])
    sigma_vol = sigma_vol_of(alpha, frac, sigma_d)
    out = REPO / C._fv_rel(regime)
    v_gbm.generate(regime, seed=path_seed, n_days=n_days, out_path=out,
                   sv_override={"alpha": float(alpha), "sigma_vol": sigma_vol})

    params = _build_fixed_params(regime, base)
    n_steps = C._sim_steps(regime, n_days)
    rets = []
    for s in range(agent_seed, agent_seed + n_runs):
        traders = build_traders(params, seed=s)
        hist = Simulation(params, traders, seed=s, ccp=None).run(n_steps)
        mid = pd.Series(hist["mid_price"]).ffill().bfill().to_numpy()
        mid = mid[mid > 0]
        if len(mid) > 1:
            rets.append(C._intraday_logret(mid, regime))
    if not rets:
        return float("inf"), {m: float("nan") for m in MOMENT_NAMES}
    pooled = np.concatenate(rets)
    m = compute_moments(pooled)
    m["ks_stat"] = C._ks_2samp(pooled, C._empirical_returns(regime))
    return _true_loss(m, target, sds), m


# ---------------------------------------------------------------------------
# two-stage surrogate-assisted search  (Sobol -> XGBoost -> exploit/explore -> grid)
# ---------------------------------------------------------------------------
def run(regime, bounds, n_sobol, n_pool, n_supplement, n_grid, stage2_frac,
        n_runs, n_days, agent_seed, path_seed, seed=0, verbose=True):
    base = json.loads(PARAMS_JSON.read_text())
    target = empirical_targets()[regime]
    sds = empirical_moment_sd(regime)
    lo, hi = bounds[:, 0], bounds[:, 1]

    # caches are keyed on the path length / ts boundaries, which are invariant
    # across candidates (n_days, path_seed fixed) - clear once so a stale CSV
    # from an earlier run can't poison them.
    C._regime_fv_bars.cache_clear()
    C._day_starts.cache_clear()

    X, y, moms = [], [], []

    def _ev(a, f, tag):
        t0 = time.time()
        D, m = eval_candidate(a, f, regime, base, target, sds,
                              n_days, n_runs, agent_seed, path_seed)
        X.append([a, f]); y.append(D); moms.append(m)
        if verbose:
            print(f"  [{tag}] alpha={a:.3e} f={f:.3f} "
                  f"sigma_vol={sigma_vol_of(a, f, base[regime]['sigma_d']):.3e} "
                  f"-> D={D:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        return D

    # --- Stage 1a: Sobol design, true ABM ---
    if verbose:
        print(f"[stage 1] {n_sobol} Sobol evals over alpha{tuple(bounds[0])} "
              f"x f{tuple(bounds[1])}", flush=True)
    u = C._sobol(n_sobol, 2, np.random.default_rng(seed))
    for a, f in lo + u * (hi - lo):
        _ev(float(a), float(f), "sobol")

    # --- Stage 1b: XGBoost surrogate -> supplement with exploit + explore ---
    info = {}
    if n_supplement > 0 and np.isfinite(y).sum() >= 8:
        yv = np.asarray(y); ok = np.isfinite(yv)
        cap = float(np.percentile(yv[ok], 90))
        model = C._xgb(); model.fit(np.asarray(X)[ok], np.minimum(yv[ok], cap))
        pool = lo + C._sobol(n_pool, 2, np.random.default_rng(seed + 1)) * (hi - lo)
        pred = model.predict(pool)
        n_exploit = (2 * n_supplement) // 3
        order = np.argsort(pred)
        exploit = pool[order[:n_exploit]]
        rng = np.random.default_rng(seed + 2)
        explore = pool[rng.choice(len(pool), n_supplement - n_exploit, replace=False)]
        # held-out surrogate quality (informational)
        info["surrogate_cap"] = cap
        if verbose:
            print(f"[stage 1b] surrogate trained on {int(ok.sum())} pts; "
                  f"{n_exploit} exploit + {n_supplement - n_exploit} explore", flush=True)
        for a, f in np.vstack([exploit, explore]):
            _ev(float(a), float(f), "supp")

    # --- Stage 2: tight grid around the running argmin ---
    yv = np.asarray(y)
    b = int(np.nanargmin(yv))
    a0, f0 = X[b]
    if n_grid and n_grid > 1:
        da = stage2_frac * (hi[0] - lo[0]); df = stage2_frac * (hi[1] - lo[1])
        ag = np.linspace(max(lo[0], a0 - da), min(hi[0], a0 + da), n_grid)
        fg = np.linspace(max(lo[1], f0 - df), min(hi[1], f0 + df), n_grid)
        if verbose:
            print(f"[stage 2] {n_grid}x{n_grid} grid around alpha={a0:.3e} f={f0:.3f}",
                  flush=True)
        for a in ag:
            for f in fg:
                _ev(float(a), float(f), "grid")

    # --- result + baseline (measured) reference ---
    yv = np.asarray(y); b = int(np.nanargmin(yv))
    a_star, f_star = X[b]
    sigma_d = float(base[regime]["sigma_d"])
    sv = base[regime]["sv"]
    f_meas = (sv["sigma_vol"] ** 2) / (2.0 * sv["alpha"]) / (sigma_d ** 2)
    if verbose:
        print(f"\n[measured baseline] evaluating alpha={sv['alpha']:.3e} "
              f"f={f_meas:.3f} for reference", flush=True)
    D_meas, m_meas = eval_candidate(sv["alpha"], f_meas, regime, base, target, sds,
                                    n_days, n_runs, agent_seed, path_seed)

    result = {
        "regime": regime,
        "free": ["alpha", "sigma_vol(via variance-fraction f)"],
        "fixed": {"sigma_d": sigma_d, "jump_lambda_day": base[regime]["jump_lambda_day"],
                  "jump_delta": base[regime]["jump_delta"], "mean_return": base[regime]["mean_return"]},
        "bounds": {"alpha": list(bounds[0]), "f": list(bounds[1])},
        "budget": {"n_sobol": n_sobol, "n_supplement": n_supplement, "n_grid": n_grid,
                   "n_runs": n_runs, "n_days": n_days, "path_seed": path_seed,
                   "agent_seed": agent_seed, "total_evals": len(y)},
        "reverse_optimum": {
            "alpha": a_star, "f": f_star,
            "sigma_vol": sigma_vol_of(a_star, f_star, sigma_d),
            "theta_eff": float(np.sqrt(max(sigma_d ** 2 * (1.0 - f_star), 0.0))),
            "loss_D": float(yv[b]), "moments": moms[b],
        },
        "measured_baseline": {
            "alpha": sv["alpha"], "f": f_meas, "sigma_vol": sv["sigma_vol"],
            "loss_D": float(D_meas), "moments": m_meas,
        },
        "all_evals": [{"alpha": x[0], "f": x[1], "D": float(d)} for x, d in zip(X, y)],
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="stressed", choices=["calm", "stressed"])
    ap.add_argument("--alpha-lo", type=float, default=3e-4)
    ap.add_argument("--alpha-hi", type=float, default=6e-3)
    ap.add_argument("--f-lo", type=float, default=0.05)
    ap.add_argument("--f-hi", type=float, default=0.90)
    ap.add_argument("--n-sobol", type=int, default=48)
    ap.add_argument("--n-pool", type=int, default=4096)
    ap.add_argument("--n-supplement", type=int, default=24)
    ap.add_argument("--n-grid", type=int, default=5)
    ap.add_argument("--stage2-frac", type=float, default=0.15)
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--path-seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-back", action="store_true",
                    help="patch output/v_gbm_params.json[regime]['sv'] with the reverse optimum "
                         "(alpha, sigma_vol); a .bak is saved first")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget to verify the loop wires up")
    a = ap.parse_args()

    if a.smoke:
        a.n_sobol, a.n_supplement, a.n_grid, a.n_runs, a.n_days = 4, 0, 0, 1, 6

    bounds = np.array([[a.alpha_lo, a.alpha_hi], [a.f_lo, a.f_hi]], dtype=float)
    t0 = time.time()
    res = run(a.regime, bounds, a.n_sobol, a.n_pool, a.n_supplement, a.n_grid,
              a.stage2_frac, a.n_runs, a.n_days, a.agent_seed, a.path_seed)
    res["wall_seconds"] = round(time.time() - t0, 1)

    ro, mb = res["reverse_optimum"], res["measured_baseline"]
    print("\n" + "=" * 70)
    print(f"REVERSE SV-MJD CALIBRATION - {a.regime}")
    print(f"  reverse optimum : alpha={ro['alpha']:.3e}  f={ro['f']:.3f}  "
          f"sigma_vol={ro['sigma_vol']:.3e}  ->  D={ro['loss_D']:.4f}")
    print(f"  measured base   : alpha={mb['alpha']:.3e}  f={mb['f']:.3f}  "
          f"sigma_vol={mb['sigma_vol']:.3e}  ->  D={mb['loss_D']:.4f}")
    impr = 100.0 * (mb["loss_D"] - ro["loss_D"]) / mb["loss_D"] if mb["loss_D"] else float("nan")
    print(f"  loss reduction  : {impr:+.1f}%   ({res['budget']['total_evals']} evals, "
          f"{res['wall_seconds']}s)")
    print("=" * 70)

    out = Path(a.out) if a.out else REPO / "output" / f"svmjd_reverse_{a.regime}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")

    if a.write_back and not a.smoke:
        doc = json.loads(PARAMS_JSON.read_text())
        (PARAMS_JSON.with_suffix(".json.bak")).write_text(json.dumps(doc, indent=2))
        doc[a.regime]["sv"]["alpha"] = ro["alpha"]
        doc[a.regime]["sv"]["sigma_vol"] = ro["sigma_vol"]
        doc[a.regime]["sv"]["calibration"] = "reverse_indirect_inference"
        PARAMS_JSON.write_text(json.dumps(doc, indent=2))
        print(f"patched {PARAMS_JSON} (sv.alpha, sv.sigma_vol); backup at {PARAMS_JSON}.bak")


if __name__ == "__main__":
    main()
