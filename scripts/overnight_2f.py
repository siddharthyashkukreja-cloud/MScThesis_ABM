#!/usr/bin/env python3
"""
scripts/overnight_2f.py - unattended overnight run for the TWO-FACTOR (superposition-OU)
synthetic fundamental.

Three stages, saved incrementally to output/overnight_2f/ so an interruption never loses
work (calibration.json is rewritten after EVERY eval with the best-so-far):

  STAGE A  Reverse-calibrate the two-factor OU volatility AND the Merton jumps to the
           empirical stylised facts (indirect inference / SMM with an XGBoost surrogate and
           active learning). Free 6-vector:
               f          total stochastic-variance fraction  (sv_var / sigma_d^2)   -> clustering
               w          slow factor's share of that variance                        -> clustering
               alpha_f    fast OU mean-reversion rate (per 1-min step)                -> clustering
               ratio      alpha_slow = alpha_f * ratio   (so alpha_slow < alpha_f always)
               lam_mult   jump-intensity multiplier on the measured Mancini lambda    -> tail (Hill)
               delta_mult jump-size multiplier on the measured delta                  -> tail (Hill)
           The two OU factors target the |r|-ACF clustering; the jumps target the (too-thin)
           tail that the single-factor run left ~6 SD off. sigma_d, the drift and the overnight
           pool stay measured; total OU variance is anchored (E[sigma_t^2] = sigma_d^2). Loss =
           ABM-driven simulated-mid moments vs empirical ES, fixed agents = globals.CALIBRATED.
           WALL-CLOCK BUDGETED. (Pass --lam-mult-lo/hi 1 1 --delta-mult-lo/hi 1 1 to fix jumps.)

  STAGE B  Generate an N_PATHS ensemble with the calibrated two-factor parameters AND with the
           measured single-factor parameters; validate both against the empirical stylised
           facts (per-moment standardised gap / 95%-CI coverage) so the morning shows whether
           the second factor repaired the |r|-ACF clustering.

  STAGE C  Run the full clearing ABM on each calibrated two-factor path; record the
           clearing-outcome distribution (client/member defaults, IM, default fund, waterfall).

SELF-CONTAINED: own fundamental CSV (data/fv_gbm_2f_stressed.csv) and own output dir, so it
does NOT collide with a single-factor calibrate_svmjd.py run still in progress.

Run overnight (8h budget by default):
    cd <repo> && nohup python scripts/overnight_2f.py --regime stressed > output/overnight_2f.out 2>&1 &

Smoke (seconds, no xgboost needed):
    python scripts/overnight_2f.py --regime stressed --smoke
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                      # fv_csv paths resolve relative to the repo root
sys.path.insert(0, str(REPO))

from model.globals import ModelParams, CALIBRATED                    # noqa: E402
from model.run_simulation import build_traders, build_clearing_tier  # noqa: E402
from model.simulation import Simulation                              # noqa: E402
from data import v_gbm                                               # noqa: E402
import model.globals as G                                            # noqa: E402
import scripts.calibrate as C                                        # noqa: E402
from scripts.calibrate import (                                      # noqa: E402
    compute_moments, empirical_targets, empirical_moment_sd,
    _true_loss, _empirical_returns, MOMENT_NAMES,
)

OUT = REPO / "output" / "overnight_2f"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "paths").mkdir(exist_ok=True)
LOG = OUT / "run.log"
FV2F = "data/fv_gbm_2f_stressed.csv"        # calibration scratch path (relative to repo)
BARS = 390


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# self-contained helpers (independent of calibrate's FV_GBM caches)
# ---------------------------------------------------------------------------
def _v0(regime, base):
    V0 = getattr(C, "V0", None)
    return float(V0[regime]) if isinstance(V0, dict) and regime in V0 else float(base[regime]["v0"])


def _day_starts(csv):
    ts = pd.read_csv(csv, usecols=["ts"], parse_dates=["ts"])["ts"]
    d = ts.dt.normalize().to_numpy()
    return [int(i) for i in np.flatnonzero(np.r_[True, d[1:] != d[:-1]])]


def _intraday_logret(mid, day_starts):
    r = np.diff(np.log(np.asarray(mid, float)))
    drop = [d - 1 for d in day_starts if 0 < d <= len(r)]
    return np.delete(r, drop) if drop else r


def sv_override_2f(theta, sigma_d):
    """Map the free vector (f, w, alpha_fast, ratio, ...) -> two-factor sv_override dict."""
    f, w, a_f, ratio = [float(x) for x in theta[:4]]
    a_s = a_f * ratio
    vt = f * sigma_d ** 2
    return {"two_factor": True, "alpha_fast": a_f, "alpha_slow": a_s,
            "v_fast": (1.0 - w) * vt, "v_slow": w * vt}


def jump_override_2f(theta, base, regime):
    """Map the free vector's jump multipliers (theta[4]=lam_mult, theta[5]=delta_mult) ->
    Merton jump override, scaling the measured Mancini intensity/size. None if 4-D (jumps fixed)."""
    if len(theta) < 6:
        return None
    lam_mult, delta_mult = float(theta[4]), float(theta[5])
    return {"jump_lambda_day": lam_mult * float(base[regime]["jump_lambda_day"]),
            "jump_delta": delta_mult * float(base[regime]["jump_delta"])}


def _market_params(regime, base, fv_csv, clearing):
    v0 = _v0(regime, base)
    if clearing:
        return ModelParams(n_fundamental=30, n_momentum=20, n_zi=40,
                           n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
                           v0=v0, tick_size=0.25, dt_minutes=1.0, fv_csv=fv_csv,
                           **CALIBRATED[regime], stressed=(regime == "stressed"))
    return ModelParams(**C.POP, v0=v0, tick_size=0.25, dt_minutes=1.0, fv_csv=fv_csv,
                       **CALIBRATED[regime], stressed=(regime == "stressed"))


def run_market(regime, base, fv_csv, n_steps, n_runs, agent_seed, clearing=False):
    """Run the ABM on the fundamental CSV; return (pooled intraday returns, last sim, last ccp)."""
    ds = _day_starts(REPO / fv_csv)
    params = _market_params(regime, base, fv_csv, clearing)
    rets, last = [], (None, None)
    for s in range(agent_seed, agent_seed + n_runs):
        traders = build_traders(params, seed=s)
        ccp = build_clearing_tier(traders, params, seed=s, direct=False) if clearing else None
        sim = Simulation(params, traders, seed=s, ccp=ccp)
        sim.run(n_steps)
        mid = pd.Series(sim.history["mid_price"]).ffill().bfill().to_numpy()
        mid = mid[mid > 0]
        if len(mid) > 1:
            rets.append(_intraday_logret(mid, ds))
        last = (sim, ccp)
    pooled = np.concatenate(rets) if rets else np.array([])
    return pooled, last[0], last[1]


def moments_of(pooled, regime):
    if len(pooled) < 2:
        return {m: float("nan") for m in MOMENT_NAMES}
    m = compute_moments(pooled)
    m["ks_stat"] = C._ks_2samp(pooled, _empirical_returns(regime))
    return m


# ---------------------------------------------------------------------------
# STAGE A : two-factor reverse calibration (wall-clock budgeted + checkpointed)
# ---------------------------------------------------------------------------
def calibrate(regime, base, bounds, budget):
    sigma_d = float(base[regime]["sigma_d"])
    target = empirical_targets()[regime]
    sds = empirical_moment_sd(regime)
    lo, hi = bounds[:, 0], bounds[:, 1]
    D = len(lo)
    sv = base[regime]["sv"]
    f_meas = (sv["sigma_vol"] ** 2) / (2.0 * sv["alpha"]) / sigma_d ** 2
    X, Y = [], []
    t0 = time.time()
    calib_s = budget["calib_hours"] * 3600.0
    names = ["f", "w", "alpha_fast", "ratio", "lam_mult", "delta_mult"][:D]

    def cur_result():
        b = int(np.nanargmin(Y)) if len(Y) and np.isfinite(Y).any() else -1
        ro = None
        if b >= 0:
            xb = X[b]
            ro = {n: xb[i] for i, n in enumerate(names)}
            ro["sv_override"] = sv_override_2f(xb, sigma_d)
            ro["jump_override"] = jump_override_2f(xb, base, regime)
            ro["loss_D"] = float(Y[b])
        return {"regime": regime, "free": names,
                "bounds": {n: list(bounds[i]) for i, n in enumerate(names)},
                "budget": budget, "total_evals": len(Y), "elapsed_min": round((time.time() - t0) / 60, 1),
                "reverse_optimum": ro,
                "measured_single_factor": {"alpha": sv["alpha"], "sigma_vol": sv["sigma_vol"],
                                           "f": f_meas, "loss_D": budget.get("_D_meas")},
                "all_evals": [{**{n: x[i] for i, n in enumerate(names)}, "D": float(d)}
                              for x, d in zip(X, Y)]}

    def ev(vec, tag):
        ts = time.time()
        v_gbm.generate(regime, seed=budget["path_seed"], n_days=budget["n_days"], out_path=REPO / FV2F,
                       sv_override=sv_override_2f(vec, sigma_d),
                       jump_override=jump_override_2f(vec, base, regime))
        pooled, _, _ = run_market(regime, base, FV2F, budget["n_days"] * BARS,
                                  budget["n_runs"], budget["agent_seed"], clearing=False)
        Dv = _true_loss(moments_of(pooled, regime), target, sds)
        X.append([float(x) for x in vec]); Y.append(float(Dv))
        jmsg = f" lamx={vec[4]:.2f} dx={vec[5]:.2f}" if len(vec) >= 6 else ""
        log(f"  [{tag} {len(Y)}] f={vec[0]:.3f} w={vec[1]:.3f} a_f={vec[2]:.2e} ratio={vec[3]:.3f}{jmsg} "
            f"-> D={Dv:.3f} ({time.time()-ts:.0f}s, {(time.time()-t0)/60:.0f}m)")
        (OUT / "calibration.json").write_text(json.dumps(cur_result(), indent=2))   # checkpoint
        return Dv

    log(f"STAGE A: 2-factor reverse calibration ({D}-D), regime={regime}, budget {budget['calib_hours']:.1f}h")
    # measured single-factor baseline (reference, evaluated identically)
    v_gbm.generate(regime, seed=budget["path_seed"], n_days=budget["n_days"], out_path=REPO / FV2F)
    pooled, _, _ = run_market(regime, base, FV2F, budget["n_days"] * BARS,
                              budget["n_runs"], budget["agent_seed"], clearing=False)
    budget["_D_meas"] = round(_true_loss(moments_of(pooled, regime), target, sds), 3)
    log(f"  measured-1f baseline D={budget['_D_meas']}")

    # Stage 1: Sobol seed (stop early if it eats >45% of the budget)
    u = C._sobol(budget["sobol_init"], D, np.random.default_rng(0))
    for vec in lo + u * (hi - lo):
        ev(vec, "sobol")
        if time.time() - t0 > 0.45 * calib_s:
            break

    # Stage 2: XGBoost active-learning batches until the budget is spent
    rng = np.random.default_rng(99)
    batch = budget["batch"]
    while batch > 0 and (time.time() - t0) < calib_s:
        try:
            yv = np.asarray(Y); ok = np.isfinite(yv)
            if ok.sum() < 8:
                break
            cap = float(np.percentile(yv[ok], 90))
            model = C._xgb(); model.fit(np.asarray(X)[ok], np.minimum(yv[ok], cap))
            pool = lo + C._sobol(budget["n_pool"], D, rng) * (hi - lo)
            pred = model.predict(pool)
            ne = (2 * batch) // 3
            cand = np.vstack([pool[np.argsort(pred)[:ne]],
                              pool[rng.choice(len(pool), batch - ne, replace=False)]])
        except ImportError:
            log("  (xgboost unavailable - extending Sobol instead)")
            cand = lo + C._sobol(batch, D, rng) * (hi - lo)
        for vec in cand:
            ev(vec, "active")
            if (time.time() - t0) > calib_s:
                break

    res = cur_result()
    (OUT / "calibration.json").write_text(json.dumps(res, indent=2))
    ro = res["reverse_optimum"]
    impr = 100 * (budget["_D_meas"] - ro["loss_D"]) / budget["_D_meas"] if budget["_D_meas"] else float("nan")
    log(f"STAGE A done: 2f optimum D={ro['loss_D']:.3f} vs measured-1f D={budget['_D_meas']} "
        f"({impr:+.1f}%); {len(Y)} evals in {(time.time()-t0)/60:.0f}m; wrote calibration.json")
    return res


# ---------------------------------------------------------------------------
# STAGE B : ensemble + stylised-facts validation (2f vs measured 1f)
# ---------------------------------------------------------------------------
def validate(regime, base, opt, budget, deadline):
    target = empirical_targets()[regime]
    sds = empirical_moment_sd(regime)
    mcr_moments = [m for m in MOMENT_NAMES if m not in ("ks_stat", "ret_kurtosis")]
    ov2 = opt["reverse_optimum"]["sv_override"]
    jo2 = opt["reverse_optimum"].get("jump_override")
    np_paths, ens_days = budget["n_paths"], budget["ens_days"]

    def ensemble(label, sv_override, jump_override):
        mlist = []
        for i in range(1, np_paths + 1):
            csv = f"output/overnight_2f/paths/fv_{label}_s{i}.csv"
            v_gbm.generate(regime, seed=100 + i, n_days=ens_days, out_path=REPO / csv,
                           sv_override=sv_override, jump_override=jump_override)
            pooled, _, _ = run_market(regime, base, csv, ens_days * BARS, 1,
                                      budget["agent_seed"], clearing=False)
            mlist.append(moments_of(pooled, regime))
            if i % 10 == 0:
                log(f"  [{label}] {i}/{np_paths} paths")
            if time.time() > deadline and i >= 10:
                log(f"  [{label}] stopping at {i} paths (time budget)")
                break
        return pd.DataFrame(mlist)

    log(f"STAGE B: validating {np_paths}-path ensembles ({ens_days}d) - 2f (calibrated) vs 1f (measured)")
    m2 = ensemble("2f", ov2, jo2)
    m1 = ensemble("1f", None, None)            # None -> measured single-factor + measured jumps

    rows = []
    for m in mcr_moments:
        t, s = target[m], sds.get(m, np.nan)
        if not np.isfinite(s):
            continue
        z2 = abs(m2[m].mean() - t) / s
        z1 = abs(m1[m].mean() - t) / s
        rows.append(dict(moment=m, empirical=round(t, 4),
                         oneF_mean=round(m1[m].mean(), 4), oneF_gapSD=round(z1, 2),
                         twoF_mean=round(m2[m].mean(), 4), twoF_gapSD=round(z2, 2),
                         twoF_in95=("yes" if z2 < 1.96 else "no")))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "validation.csv", index=False)
    cov2 = int((tab["twoF_in95"] == "yes").sum()); K = len(tab)
    D2 = float(np.nanmean([_true_loss(dict(r), target, sds) for r in m2.to_dict("records")]))
    D1 = float(np.nanmean([_true_loss(dict(r), target, sds) for r in m1.to_dict("records")]))
    summary = dict(K=K, n_paths_2f=len(m2), n_paths_1f=len(m1), twoF_coverage=cov2,
                   twoF_meanGapSD=round(tab["twoF_gapSD"].mean(), 2),
                   oneF_meanGapSD=round(tab["oneF_gapSD"].mean(), 2),
                   twoF_ensembleD=round(D2, 2), oneF_ensembleD=round(D1, 2))
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"STAGE B done: 2f coverage {cov2}/{K}; mean |gap| 2f={summary['twoF_meanGapSD']}SD "
        f"vs 1f={summary['oneF_meanGapSD']}SD; ensemble D 2f={D2:.1f} vs 1f={D1:.1f}")
    clust = tab[tab["moment"].str.startswith("acf_absr")]
    log("  |r|-ACF clustering moments (the target):\n" + clust.to_string(index=False))
    return tab, summary


# ---------------------------------------------------------------------------
# STAGE C : clearing ensemble on the calibrated two-factor paths
# ---------------------------------------------------------------------------
def clearing(regime, base, opt, budget, deadline):
    G.IM_DAILY = False; G.IM_MODE = "reactive"
    G.CLIENT_MARGIN_NETTING = "gross"; G.DF_DECOUPLE_IM = False
    np_paths, ens_days = budget["n_paths"], budget["ens_days"]
    ov2 = opt["reverse_optimum"]["sv_override"]
    jo2 = opt["reverse_optimum"].get("jump_override")
    log(f"STAGE C: clearing ABM on the {np_paths} calibrated 2f paths")
    rows = []
    for i in range(1, np_paths + 1):
        csv = f"output/overnight_2f/paths/fv_2f_s{i}.csv"
        if not (REPO / csv).exists():
            v_gbm.generate(regime, seed=100 + i, n_days=ens_days, out_path=REPO / csv,
                           sv_override=ov2, jump_override=jo2)
        try:
            _, sim, ccp = run_market(regime, base, csv, ens_days * BARS, 1,
                                     budget["agent_seed"], clearing=True)
            ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
            mem = ch[ch["kind"].isin(["BCM", "NBCM"])] if len(ch) else ch
            im_t = ch.groupby("t")["initial_margin"].sum() if len(ch) else pd.Series([0.0])
            rows.append(dict(path=i, client_def=len(cl),
                cm_def=int(mem[mem["has_defaulted"]]["agent_id"].nunique()) if len(mem) else 0,
                min_kappa=round(float(mem["capital_ratio"].replace([np.inf, -np.inf], np.nan).min()), 3) if len(mem) else np.nan,
                IM_mean_B=round(im_t.mean() / 1e9, 2), IM_peak_B=round(im_t.max() / 1e9, 2),
                DF_B=round(float(ccp.total_df) / 1e9, 2),
                deepest_wf=int(ch["waterfall_level"].max()) if len(ch) else 0))
            pd.DataFrame(rows).to_csv(OUT / "clearing.csv", index=False)
        except Exception as e:
            log(f"  path {i} clearing FAILED: {e}")
        if i % 10 == 0:
            log(f"  clearing {i}/{np_paths}")
        if time.time() > deadline and i >= 10:
            log(f"  clearing stopping at {i} paths (time budget)")
            break
    clr = pd.DataFrame(rows)
    if len(clr):
        summ = dict(n_paths=len(clr), member_defaults_total=int(clr.cm_def.sum()),
                    deepest_waterfall=int(clr.deepest_wf.max()),
                    client_def_min=int(clr.client_def.min()), client_def_max=int(clr.client_def.max()),
                    client_def_mean=round(float(clr.client_def.mean()), 1),
                    IM_peak_B_mean=round(float(clr.IM_peak_B.mean()), 2),
                    DF_B_mean=round(float(clr.DF_B.mean()), 2))
        (OUT / "clearing_summary.json").write_text(json.dumps(summ, indent=2))
        log("STAGE C done: " + json.dumps(summ))
        return summ
    return None


def write_summary(opt, valsum, clrsum):
    L = ["# Overnight two-factor (superposition-OU) run\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n", "## Stage A - calibration"]
    ro = opt["reverse_optimum"]
    L += [f"- 2f optimum: f={ro['f']:.3f}, w={ro['w']:.3f}, alpha_fast={ro['alpha_fast']:.3e}, "
          f"alpha_slow={ro['alpha_fast']*ro['ratio']:.3e} (ratio={ro['ratio']:.3f})"]
    if ro.get("jump_override"):
        L += [f"- jumps (freed): lambda x{ro.get('lam_mult', 1):.2f} = {ro['jump_override']['jump_lambda_day']:.2f}/day, "
              f"delta x{ro.get('delta_mult', 1):.2f} = {ro['jump_override']['jump_delta']:.3e}"]
    L += [f"- loss D: 2f={ro['loss_D']:.2f} vs measured-1f={opt['measured_single_factor']['loss_D']}",
          f"- evals: {opt['total_evals']} in {opt['elapsed_min']} min\n", "## Stage B - stylised-facts validation"]
    if valsum:
        L += [f"- 2f coverage: {valsum['twoF_coverage']}/{valsum['K']} moments in empirical 95% CI "
              f"({valsum['n_paths_2f']} paths)",
              f"- mean standardised gap: 2f={valsum['twoF_meanGapSD']}SD vs 1f={valsum['oneF_meanGapSD']}SD",
              f"- ensemble loss D: 2f={valsum['twoF_ensembleD']} vs 1f={valsum['oneF_ensembleD']}\n"]
    else:
        L += ["- (not completed)\n"]
    L += ["## Stage C - clearing ensemble"]
    if clrsum:
        L += [f"- paths: {clrsum['n_paths']}; member defaults total: {clrsum['member_defaults_total']}; "
              f"deepest waterfall: {clrsum['deepest_waterfall']}",
              f"- client defaults: {clrsum['client_def_min']}-{clrsum['client_def_max']} (mean {clrsum['client_def_mean']})",
              f"- IM peak (mean) ${clrsum['IM_peak_B_mean']}B; default fund (mean) ${clrsum['DF_B_mean']}B"]
    else:
        L += ["- (not completed)"]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    log("wrote SUMMARY.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="stressed", choices=["calm", "stressed"])
    ap.add_argument("--max-hours", type=float, default=8.0, help="total wall-clock budget")
    ap.add_argument("--calib-frac", type=float, default=0.7, help="fraction of budget for calibration")
    ap.add_argument("--n-days", type=int, default=30, help="path length for calibration evals")
    ap.add_argument("--n-runs", type=int, default=3, help="agent seeds pooled per calibration eval")
    ap.add_argument("--sobol-init", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="active-learning batch size")
    ap.add_argument("--n-pool", type=int, default=4096)
    ap.add_argument("--n-paths", type=int, default=30, help="ensemble size")
    ap.add_argument("--ens-days", type=int, default=40, help="ensemble path length")
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--path-seed", type=int, default=7)
    ap.add_argument("--alpha-lo", type=float, default=8e-4)
    ap.add_argument("--alpha-hi", type=float, default=6e-3)
    ap.add_argument("--f-lo", type=float, default=0.15)
    ap.add_argument("--f-hi", type=float, default=0.85)
    ap.add_argument("--w-lo", type=float, default=0.10)
    ap.add_argument("--w-hi", type=float, default=0.90)
    ap.add_argument("--ratio-lo", type=float, default=0.03)
    ap.add_argument("--ratio-hi", type=float, default=0.40)
    ap.add_argument("--lam-mult-lo", type=float, default=0.5)
    ap.add_argument("--lam-mult-hi", type=float, default=4.0)
    ap.add_argument("--delta-mult-lo", type=float, default=0.7)
    ap.add_argument("--delta-mult-hi", type=float, default=3.0)
    ap.add_argument("--skip-clearing", action="store_true")
    ap.add_argument("--calib-only", action="store_true",
                    help="run only STAGE A (calibration); skip path simulation + clearing")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        (a.max_hours, a.n_days, a.n_runs, a.sobol_init, a.batch,
         a.n_paths, a.ens_days) = 0.2, 6, 1, 4, 0, 3, 6

    bounds = np.array([[a.f_lo, a.f_hi], [a.w_lo, a.w_hi],
                       [a.alpha_lo, a.alpha_hi], [a.ratio_lo, a.ratio_hi],
                       [a.lam_mult_lo, a.lam_mult_hi], [a.delta_mult_lo, a.delta_mult_hi]], dtype=float)
    budget = dict(n_days=a.n_days, n_runs=a.n_runs, sobol_init=a.sobol_init, batch=a.batch,
                  n_pool=a.n_pool, calib_hours=(a.max_hours if a.calib_only else a.max_hours * a.calib_frac),
                  n_paths=a.n_paths, ens_days=a.ens_days, agent_seed=a.agent_seed, path_seed=a.path_seed)
    base = json.loads((REPO / "output" / "v_gbm_params.json").read_text())

    t0 = time.time()
    deadline = t0 + a.max_hours * 3600.0
    try:
        LOG.write_text("")            # fresh log each run
    except OSError:
        pass
    log("=" * 70)
    log(f"OVERNIGHT 2-FACTOR RUN - regime={a.regime} max_hours={a.max_hours} budget={budget}")
    opt = valsum = clrsum = None
    try:
        opt = calibrate(a.regime, base, bounds, budget)
    except Exception:
        log("STAGE A CRASHED:\n" + traceback.format_exc()); return
    if opt is None or opt["reverse_optimum"] is None:
        log("no calibration optimum - aborting"); return
    if not a.calib_only:
        try:
            _, valsum = validate(a.regime, base, opt, budget, deadline)
        except Exception:
            log("STAGE B CRASHED (continuing):\n" + traceback.format_exc())
        if not a.skip_clearing:
            try:
                clrsum = clearing(a.regime, base, opt, budget, deadline)
            except Exception:
                log("STAGE C CRASHED (continuing):\n" + traceback.format_exc())
    else:
        log("calib-only mode: skipping path simulation (STAGE B) + clearing (STAGE C)")
    write_summary(opt, valsum, clrsum)
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Outputs in {OUT}")


if __name__ == "__main__":
    main()
