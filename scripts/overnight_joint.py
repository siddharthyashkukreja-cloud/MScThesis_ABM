#!/usr/bin/env python3
"""
scripts/overnight_joint.py — JOINT agent + fundamental-value (SV-MJD) calibration.

Frees 6 parameters and fits them TOGETHER to the empirical stylised facts, so the
volatility clustering is earned by the momentum agents (the canonical ABM mechanism —
Lux-Marchesi, Cont 2007) rather than forced into the fundamental, where it hits the OU
positivity clamp and the i.i.d.-jump dilution that the FV-only run ran into.

Free 9-vector (order matters; D80 — pure log-vol, jumps dropped):
    mt_gamma     momentum tanh-activation strength            -> |r|-ACF clustering
    ft_sigma_c   FT belief-width (V_t transmission fidelity)  -> tail + clustering
    f            two-factor (log-)vol variance s_x^2          -> clustering + tail amplitude
    w            slow factor's share of that variance         -> long-lag clustering
    ratio        alpha_slow = alpha_fast * ratio              -> long-lag timescale
    sigma_d      volatility LEVEL (anchors E[sigma^2])        -> mid ret_std (auction attenuation)
    zi_alpha     ZI limit-order rate                          -> background liquidity / vol
    zi_mu        ZI market-order rate                         -> background liquidity / vol
    zi_delta     ZI cancellation rate                         -> resting-depth / vol

Fixed: mt_lambda + p_zi (held at globals.CALIBRATED), alpha_fast (measured OU rate),
the Merton jumps (DROPPED -- delta=0, the log-vol carries the tail), and the DRIFT.

Drift is a FIXED input (not calibrated): mu set so each path declines ~DRAWDOWN net over a
~WINDOW-day (~2-month) window INCLUDING the overnight pool — a COVID-class stress path.
The drift only shifts the mean, which is not a loss moment, so it is orthogonal to the
stylised-facts calibration.

Loss = ABM-driven simulated-mid moments vs empirical ES (same machinery as the agent
calibration). WALL-CLOCK BUDGETED, calibration.json checkpointed every eval.

Run a short calibration-only sanity check first:
    python3 scripts/overnight_joint.py --regime stressed --calib-only --max-hours 0.4 --sobol-init 24
Full overnight:
    nohup python3 scripts/overnight_joint.py --regime stressed > output/overnight_joint.out 2>&1 &
Smoke:
    python3 scripts/overnight_joint.py --regime stressed --smoke
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
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from model.globals import ModelParams, CALIBRATED                    # noqa: E402
from model.run_simulation import build_traders, build_clearing_tier  # noqa: E402
from model.simulation import Simulation                              # noqa: E402
from data import v_gbm                                               # noqa: E402
import model.globals as G                                            # noqa: E402
import scripts.calibrate as C                                        # noqa: E402
from scripts.calibrate import (                                      # noqa: E402
    compute_moments, empirical_targets, empirical_moment_sd, _true_loss,
    _empirical_returns, MOMENT_NAMES,
)
from scripts.overnight_2f import _day_starts, _intraday_logret, _v0  # noqa: E402

OUT = REPO / "output" / "overnight_joint"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "paths").mkdir(exist_ok=True)
LOG = OUT / "run.log"
FVJ = "data/fv_gbm_joint_stressed.csv"           # own scratch path (no collision)
BARS = 390

# Joint free set (D80): jumps DROPPED (log-vol carries the tail; jumps only diluted clustering);
# mt_lambda FIXED at its standalone-calibrated value (it did not move in the joint fit); ADDED
# ratio (slow-factor timescale, a long-lag-clustering lever), sigma_d (volatility LEVEL, to clear
# the call-auction ret_std undershoot) and the three ZI rates -> a fully joint agent+FV calibration.
FREE = ["mt_gamma", "ft_sigma_c", "f", "w", "ratio", "sigma_d", "zi_alpha", "zi_mu", "zi_delta"]
BOUNDS = {  # stressed; agent boxes match scripts/calibrate PARAM_BOUNDS_BY_REGIME["stressed"]
    "mt_gamma": (0.1, 2.0), "ft_sigma_c": (0.10, 0.85),
    "f": (0.10, 1.40),            # log-vol s_x^2 (total log-vol variance); linear-mode bound set in main
    "w": (0.10, 0.90), "ratio": (0.02, 0.40), "sigma_d": (1.0e-3, 2.2e-3),
    "zi_alpha": (0.08, 0.44), "zi_mu": (0.0125, 0.15), "zi_delta": (0.02, 0.32)}
RATIO_FIXED = 0.1            # default alpha_slow = alpha_fast * ratio when ratio not free
LAM_MULT_FIXED = 1.0         # (jumps dropped; retained only for the measured-baseline reference)
LOG_VOL = False             # --log-vol: OU on log-volatility (fat tails via vol, not i.i.d. jumps)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# drift: ~DRAWDOWN net decline over ~WINDOW sessions, incl. the overnight pool
# ---------------------------------------------------------------------------
def drift_for_drawdown(base, regime, drawdown, window_days):
    R = float(np.log(1.0 - drawdown))                       # target total log-return
    pool = np.asarray(base[regime]["overnight_pool"], float)
    overnight = float(pool.mean()) * (window_days - 1)      # overnight contribution
    n_intraday = window_days * (BARS - 1)                   # ~one overnight gap per day
    return (R - overnight) / n_intraday


# ---------------------------------------------------------------------------
# free 6-vector -> (agent overrides, sv_override, jump_override)
# ---------------------------------------------------------------------------
def free_to_overrides(theta, base, regime):
    d = dict(zip(FREE, [float(x) for x in theta]))
    sigma_d = float(base[regime]["sigma_d"])
    a_f = float(base[regime]["sv"]["alpha"])               # measured fast OU rate (fixed)
    sigma_d = d.get("sigma_d", sigma_d)                    # vol LEVEL: free if in the set, else measured
    ratio = d.get("ratio", RATIO_FIXED)                    # slow-factor timescale: free if in the set
    w = d["w"]
    if LOG_VOL:
        vt = d["f"]                                        # f = total LOG-vol variance s_x^2
        sv = {"two_factor": True, "log_vol": True, "sigma_d": sigma_d,
              "alpha_fast": a_f, "alpha_slow": a_f * ratio,
              "v_fast": (1.0 - w) * vt, "v_slow": w * vt}
    else:
        vt = d["f"] * sigma_d ** 2
        sv = {"two_factor": True, "sigma_d": sigma_d, "alpha_fast": a_f, "alpha_slow": a_f * ratio,
              "v_fast": (1.0 - w) * vt, "v_slow": w * vt}
    dm = d.get("delta_mult", 0.0)                          # jumps DROPPED -> delta 0 (no i.i.d. tail)
    jmp = {"jump_lambda_day": LAM_MULT_FIXED * float(base[regime]["jump_lambda_day"]),
           "jump_delta": dm * float(base[regime]["jump_delta"])}
    # only the FREE agent params override CALIBRATED; the rest (mt_lambda, p_zi, ...) stay calibrated
    agent = {k: d[k] for k in ("mt_lambda", "mt_gamma", "ft_sigma_c",
                               "zi_alpha", "zi_mu", "zi_delta", "p_zi") if k in d}
    return agent, sv, jmp


def _params(regime, base, fv_csv, agent, clearing):
    kw = {**CALIBRATED[regime], **(agent or {})}            # override the freed agent params
    if clearing:
        return ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                           n_bcm_with_clients=5, v0=_v0(regime, base), tick_size=0.25,
                           dt_minutes=1.0, fv_csv=fv_csv, **kw, stressed=(regime == "stressed"))
    return ModelParams(**C.POP, v0=_v0(regime, base), tick_size=0.25, dt_minutes=1.0,
                       fv_csv=fv_csv, **kw, stressed=(regime == "stressed"))


def run_market(regime, base, fv_csv, n_steps, n_runs, agent_seed, agent, clearing=False):
    ds = _day_starts(REPO / fv_csv)
    p = _params(regime, base, fv_csv, agent, clearing)
    rets, last = [], (None, None)
    for s in range(agent_seed, agent_seed + n_runs):
        tr = build_traders(p, seed=s)
        ccp = build_clearing_tier(tr, p, seed=s, direct=False) if clearing else None
        sim = Simulation(p, tr, seed=s, ccp=ccp); sim.run(n_steps)
        mid = pd.Series(sim.history["mid_price"]).ffill().bfill().to_numpy(); mid = mid[mid > 0]
        if len(mid) > 1:
            rets.append(_intraday_logret(mid, ds))
        last = (sim, ccp)
    return (np.concatenate(rets) if rets else np.array([])), last[0], last[1]


def moments_of(pooled, regime):
    if len(pooled) < 2:
        return {m: float("nan") for m in MOMENT_NAMES}
    m = compute_moments(pooled)
    m["ks_stat"] = C._ks_2samp(pooled, _empirical_returns(regime))
    return m


# ---------------------------------------------------------------------------
# STAGE A : joint calibration (wall-clock budgeted, checkpointed, active learning)
# ---------------------------------------------------------------------------
def calibrate(regime, base, bounds, budget):
    target = empirical_targets()[regime]; sds = empirical_moment_sd(regime)
    lo, hi = bounds[:, 0], bounds[:, 1]; D = len(lo)
    mu = budget["drift"]
    X, Y = [], []
    if budget.get("warm_start"):                           # seed the surrogate from a prior run's evals (same loss)
        wp = budget["warm_start"]
        wp = wp if os.path.isabs(wp) else str(REPO / wp)
        try:
            wd = json.loads(Path(wp).read_text()); nw = 0
            for e in wd.get("all_evals", []):
                try:
                    vec = [float(e[k]) for k in FREE]; dv = float(e["D"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.all(np.isfinite(vec)) and np.isfinite(dv):
                    X.append(vec); Y.append(dv); nw += 1
            log(f"  warm-start: seeded {nw} prior evals from {budget['warm_start']}")
        except (OSError, ValueError) as ex:
            log(f"  warm-start FAILED ({ex}); starting cold")
    t0 = time.time(); calib_s = budget["calib_hours"] * 3600.0

    def cur():
        b = int(np.nanargmin(Y)) if len(Y) and np.isfinite(Y).any() else -1
        ro = None
        if b >= 0:
            agent, sv, jmp = free_to_overrides(X[b], base, regime)
            ro = {**dict(zip(FREE, X[b])), "agent_overrides": agent, "sv_override": sv,
                  "jump_override": jmp, "mu_override": mu, "loss_D": float(Y[b])}
        return {"regime": regime, "free": FREE, "fixed": {"ratio": RATIO_FIXED, "lam_mult": LAM_MULT_FIXED,
                "drift": mu, "drawdown": budget["drawdown"], "window_days": budget["window_days"]},
                "bounds": {k: list(bounds[i]) for i, k in enumerate(FREE)}, "budget": budget,
                "total_evals": len(Y), "elapsed_min": round((time.time() - t0) / 60, 1),
                "reverse_optimum": ro, "measured_baseline_D": budget.get("_D_base"),
                "all_evals": [{**dict(zip(FREE, x)), "D": float(d)} for x, d in zip(X, Y)]}

    def ev(vec, tag):
        ts = time.time()
        agent, sv, jmp = free_to_overrides(vec, base, regime)
        allret = []   # pool over n_path_seeds FUNDAMENTAL paths (fixed CRN set) -> path-robust loss
        for ps in range(budget["path_seed"], budget["path_seed"] + budget["n_path_seeds"]):
            v_gbm.generate(regime, seed=ps, n_days=budget["n_days"], out_path=REPO / FVJ,
                           sv_override=sv, jump_override=jmp, mu_override=mu)
            r, _, _ = run_market(regime, base, FVJ, budget["n_days"] * BARS,
                                 budget["n_runs"], budget["agent_seed"], agent, clearing=False)
            if len(r):
                allret.append(r)
        pooled = np.concatenate(allret) if allret else np.array([])
        Dv = _true_loss(moments_of(pooled, regime), target, sds)
        X.append([float(x) for x in vec]); Y.append(float(Dv))
        log(f"  [{tag} {len(Y)}] " + " ".join(f"{n}={v:.3g}" for n, v in zip(FREE, vec))
            + f" -> D={Dv:.3f} ({time.time()-ts:.0f}s, {(time.time()-t0)/60:.0f}m)")
        (OUT / "calibration.json").write_text(json.dumps(cur(), indent=2))
        return Dv

    log(f"STAGE A: JOINT agent+FV calibration ({D}-D), regime={regime}, budget {budget['calib_hours']:.1f}h, "
        f"drift={mu:.3e}/min (~{100*budget['drawdown']:.0f}% over {budget['window_days']}d)")
    # measured baseline: CALIBRATED agents + measured single-factor FV + measured jumps (same drift, same path set)
    allret = []
    for ps in range(budget["path_seed"], budget["path_seed"] + budget["n_path_seeds"]):
        v_gbm.generate(regime, seed=ps, n_days=budget["n_days"], out_path=REPO / FVJ, mu_override=mu)
        r, _, _ = run_market(regime, base, FVJ, budget["n_days"] * BARS,
                             budget["n_runs"], budget["agent_seed"], None, clearing=False)
        if len(r):
            allret.append(r)
    pooled = np.concatenate(allret) if allret else np.array([])
    budget["_D_base"] = round(_true_loss(moments_of(pooled, regime), target, sds), 3)
    log(f"  measured baseline D={budget['_D_base']}  (CALIBRATED agents + 1-factor FV)")

    u = C._sobol(budget["sobol_init"], D, np.random.default_rng(0))
    for vec in lo + u * (hi - lo):
        ev(vec, "sobol")
        if time.time() - t0 > 0.45 * calib_s:
            break
    rng = np.random.default_rng(99); batch = budget["batch"]
    while batch > 0 and (time.time() - t0) < calib_s:
        try:
            yv = np.asarray(Y); ok = np.isfinite(yv)
            if ok.sum() < 8:
                break
            cap = float(np.percentile(yv[ok], 90))
            model = C._xgb(); model.fit(np.asarray(X)[ok], np.minimum(yv[ok], cap))
            pool = lo + C._sobol(budget["n_pool"], D, rng) * (hi - lo)
            pred = model.predict(pool); ne = (2 * batch) // 3
            cand = np.vstack([pool[np.argsort(pred)[:ne]], pool[rng.choice(len(pool), batch - ne, replace=False)]])
        except ImportError:
            log("  (xgboost unavailable - extending Sobol)"); cand = lo + C._sobol(batch, D, rng) * (hi - lo)
        for vec in cand:
            ev(vec, "active")
            if (time.time() - t0) > calib_s:
                break

    res = cur(); (OUT / "calibration.json").write_text(json.dumps(res, indent=2))
    ro = res["reverse_optimum"]
    impr = 100 * (budget["_D_base"] - ro["loss_D"]) / budget["_D_base"] if budget["_D_base"] else float("nan")
    log(f"STAGE A done: joint optimum D={ro['loss_D']:.3f} vs baseline D={budget['_D_base']} "
        f"({impr:+.1f}%); {len(Y)} evals in {(time.time()-t0)/60:.0f}m")
    return res


# ---------------------------------------------------------------------------
# STAGE B : ensemble + stylised-facts validation (joint optimum vs baseline)
# ---------------------------------------------------------------------------
def validate(regime, base, opt, budget, deadline):
    target = empirical_targets()[regime]; sds = empirical_moment_sd(regime)
    mcr = [m for m in MOMENT_NAMES if m not in ("ks_stat", "ret_kurtosis")]
    ro = opt["reverse_optimum"]; mu = budget["drift"]
    npaths, days = budget["n_paths"], budget["ens_days"]

    def ensemble(label, sv, jmp, agent):
        ms = []
        for i in range(1, npaths + 1):
            csv = f"output/overnight_joint/paths/fv_{label}_s{i}.csv"
            v_gbm.generate(regime, seed=100 + i, n_days=days, out_path=REPO / csv,
                           sv_override=sv, jump_override=jmp, mu_override=mu)
            pooled, _, _ = run_market(regime, base, csv, days * BARS, 1, budget["agent_seed"], agent)
            ms.append(moments_of(pooled, regime))
            if i % 10 == 0:
                log(f"  [{label}] {i}/{npaths}")
            if time.time() > deadline and i >= 10:
                break
        return pd.DataFrame(ms)

    log(f"STAGE B: validating {npaths}-path ensembles ({days}d) - joint vs baseline")
    mj = ensemble("joint", ro["sv_override"], ro["jump_override"], ro["agent_overrides"])
    mb = ensemble("base", None, None, None)
    rows = []
    for m in mcr:
        s = sds.get(m, np.nan)
        if not np.isfinite(s):
            continue
        rows.append(dict(moment=m, empirical=round(target[m], 4),
                         base_gapSD=round(abs(mb[m].mean() - target[m]) / s, 2),
                         joint_mean=round(mj[m].mean(), 4),
                         joint_gapSD=round(abs(mj[m].mean() - target[m]) / s, 2),
                         joint_in95=("yes" if abs(mj[m].mean() - target[m]) / s < 1.96 else "no")))
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "validation.csv", index=False)
    cov = int((tab["joint_in95"] == "yes").sum()); K = len(tab)
    Dj = float(np.nanmean([_true_loss(dict(r), target, sds) for r in mj.to_dict("records")]))
    Db = float(np.nanmean([_true_loss(dict(r), target, sds) for r in mb.to_dict("records")]))
    summ = dict(K=K, joint_coverage=cov, joint_meanGapSD=round(tab["joint_gapSD"].mean(), 2),
                base_meanGapSD=round(tab["base_gapSD"].mean(), 2),
                joint_ensembleD=round(Dj, 2), base_ensembleD=round(Db, 2))
    (OUT / "validation_summary.json").write_text(json.dumps(summ, indent=2))
    log(f"STAGE B done: joint coverage {cov}/{K}; mean |gap| joint={summ['joint_meanGapSD']} "
        f"vs base={summ['base_meanGapSD']}SD; ensemble D joint={Dj:.1f} vs base={Db:.1f}")
    log("  clustering + tail:\n" + tab[tab.moment.str.startswith(("acf_absr", "hill"))].to_string(index=False))
    return tab, summ


def clearing(regime, base, opt, budget, deadline):
    G.IM_DAILY = False; G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"; G.DF_DECOUPLE_IM = False
    ro = opt["reverse_optimum"]; mu = budget["drift"]; npaths, days = budget["n_paths"], budget["ens_days"]
    log(f"STAGE C: clearing ABM on the {npaths} joint paths")
    rows = []
    for i in range(1, npaths + 1):
        csv = f"output/overnight_joint/paths/fv_joint_s{i}.csv"
        if not (REPO / csv).exists():
            v_gbm.generate(regime, seed=100 + i, n_days=days, out_path=REPO / csv,
                           sv_override=ro["sv_override"], jump_override=ro["jump_override"], mu_override=mu)
        try:
            _, sim, ccp = run_market(regime, base, csv, days * BARS, 1, budget["agent_seed"],
                                     ro["agent_overrides"], clearing=True)
            ch = pd.DataFrame(sim.clearing_history); cl = pd.DataFrame(sim.client_history)
            mem = ch[ch["kind"].isin(["BCM", "NBCM"])] if len(ch) else ch
            im_t = ch.groupby("t")["initial_margin"].sum() if len(ch) else pd.Series([0.0])
            rows.append(dict(path=i, client_def=len(cl),
                cm_def=int(mem[mem["has_defaulted"]]["agent_id"].nunique()) if len(mem) else 0,
                IM_peak_B=round(im_t.max() / 1e9, 2), DF_B=round(float(ccp.total_df) / 1e9, 2),
                deepest_wf=int(ch["waterfall_level"].max()) if len(ch) else 0))
            pd.DataFrame(rows).to_csv(OUT / "clearing.csv", index=False)
        except Exception as e:
            log(f"  path {i} clearing FAILED: {e}")
        if time.time() > deadline and i >= 10:
            break
    clr = pd.DataFrame(rows)
    if len(clr):
        summ = dict(n_paths=len(clr), member_defaults_total=int(clr.cm_def.sum()),
                    deepest_waterfall=int(clr.deepest_wf.max()),
                    client_def_mean=round(float(clr.client_def.mean()), 1),
                    IM_peak_B_mean=round(float(clr.IM_peak_B.mean()), 2))
        (OUT / "clearing_summary.json").write_text(json.dumps(summ, indent=2))
        log("STAGE C done: " + json.dumps(summ)); return summ
    return None


def write_summary(opt, vs, cs):
    ro = opt["reverse_optimum"]
    L = ["# Overnight JOINT agent+FV calibration\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n",
         "## Stage A - calibration",
         f"- joint optimum: " + ", ".join(f"{k}={ro[k]:.3f}" for k in FREE),
         f"- loss D: joint={ro['loss_D']:.2f} vs baseline={opt['measured_baseline_D']}  "
         f"({opt['total_evals']} evals, {opt['elapsed_min']} min)",
         f"- drift={opt['fixed']['drift']:.3e}/min (~{100*opt['fixed']['drawdown']:.0f}% over "
         f"{opt['fixed']['window_days']}d); jumps dropped (log-vol carries the tail)\n", "## Stage B - validation"]
    L.append((f"- joint coverage {vs['joint_coverage']}/{vs['K']}; mean |gap| joint={vs['joint_meanGapSD']} "
              f"vs base={vs['base_meanGapSD']}SD; ensemble D joint={vs['joint_ensembleD']} vs base={vs['base_ensembleD']}")
             if vs else "- (not run)")
    L += ["\n## Stage C - clearing"]
    L.append((f"- {cs['n_paths']} paths; member defaults {cs['member_defaults_total']}; deepest waterfall "
              f"{cs['deepest_waterfall']}; client defaults mean {cs['client_def_mean']}; IM peak ${cs['IM_peak_B_mean']}B")
             if cs else "- (not run)")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n"); log("wrote SUMMARY.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="stressed", choices=["stressed"])
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--calib-frac", type=float, default=0.7)
    ap.add_argument("--n-days", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=1)            # agent seeds per path
    ap.add_argument("--n-path-seeds", type=int, default=3)      # fundamental paths pooled per eval (path-robust loss)
    ap.add_argument("--sobol-init", type=int, default=110)   # 9-D needs a denser seed
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-pool", type=int, default=4096)
    ap.add_argument("--n-paths", type=int, default=80)         # ensemble size ~ clearing-rate precision (>=60)
    ap.add_argument("--ens-days", type=int, default=42)          # ~2-month stress path
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--path-seed", type=int, default=7)
    ap.add_argument("--drawdown", type=float, default=0.30)
    ap.add_argument("--window-days", type=int, default=42)
    ap.add_argument("--skip-clearing", action="store_true")
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--log-vol", action="store_true",
                    help="OU on log-volatility: fat tails via volatility clustering, not i.i.d. jumps")
    ap.add_argument("--smoke", action="store_true")
    # --- opt-in re-run options (defaults preserve the standard run exactly) ---
    ap.add_argument("--out-tag", default="",
                    help="suffix on the (log-vol) output dir + FVJ csv, to re-run WITHOUT overwriting a prior run")
    ap.add_argument("--zi-delta-hi", type=float, default=None,
                    help="override the zi_delta upper bound (e.g. 0.50) to unpin it for a wider-bound re-run")
    ap.add_argument("--warm-start", default=None,
                    help="path to a prior calibration.json; seeds the surrogate with its evals "
                         "(use the SAME loss config: regime, n_days, path_seed, n_path_seeds)")
    a = ap.parse_args()

    global LOG_VOL, OUT, LOG, FVJ
    if a.log_vol:
        LOG_VOL = True
        tag = ("_" + a.out_tag) if a.out_tag else ""       # separate dir/FVJ so a re-run never overwrites a prior one
        OUT = REPO / "output" / ("overnight_joint_logvol" + tag)
        (OUT / "paths").mkdir(parents=True, exist_ok=True)
        LOG = OUT / "run.log"; FVJ = "data/fv_gbm_joint_logvol" + tag + ".csv"

    if a.smoke:
        (a.max_hours, a.n_days, a.n_runs, a.n_path_seeds, a.sobol_init, a.batch, a.n_paths, a.ens_days) = 0.2, 6, 1, 1, 4, 0, 3, 6

    base = json.loads((REPO / "output" / "v_gbm_params.json").read_text())
    drift = drift_for_drawdown(base, a.regime, a.drawdown, a.window_days)
    bounds = np.array([BOUNDS[k] for k in FREE], dtype=float)
    if not a.log_vol:
        bounds[FREE.index("f")] = [0.15, 0.85]             # linear-OU variance fraction (vs log-vol s_x^2)
    if a.zi_delta_hi is not None:                          # widen the zi_delta upper bound for a re-run
        bounds[FREE.index("zi_delta"), 1] = float(a.zi_delta_hi)
    budget = dict(n_days=a.n_days, n_runs=a.n_runs, n_path_seeds=a.n_path_seeds, sobol_init=a.sobol_init,
                  batch=a.batch, n_pool=a.n_pool,
                  calib_hours=(a.max_hours if a.calib_only else a.max_hours * a.calib_frac),
                  n_paths=a.n_paths, ens_days=a.ens_days, agent_seed=a.agent_seed, path_seed=a.path_seed,
                  drift=drift, drawdown=a.drawdown, window_days=a.window_days,
                  warm_start=a.warm_start)

    t0 = time.time(); deadline = t0 + a.max_hours * 3600.0
    try:
        LOG.write_text("")
    except OSError:
        pass
    log("=" * 70)
    log(f"OVERNIGHT JOINT RUN - regime={a.regime} max_hours={a.max_hours} budget={budget}")
    opt = vs = cs = None
    try:
        opt = calibrate(a.regime, base, bounds, budget)
    except Exception:
        log("STAGE A CRASHED:\n" + traceback.format_exc()); return
    if opt is None or opt["reverse_optimum"] is None:
        log("no optimum - aborting"); return
    if not a.calib_only:
        try:
            _, vs = validate(a.regime, base, opt, budget, deadline)
        except Exception:
            log("STAGE B CRASHED:\n" + traceback.format_exc())
        if not a.skip_clearing:
            try:
                cs = clearing(a.regime, base, opt, budget, deadline)
            except Exception:
                log("STAGE C CRASHED:\n" + traceback.format_exc())
    else:
        log("calib-only mode: skipping path simulation (STAGE B) + clearing (STAGE C)")
    write_summary(opt, vs, cs)
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Outputs in {OUT}")


if __name__ == "__main__":
    main()
