import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Thesis-final contagion experiments at ACTUAL severity (calm + stressed regimes; the
reverse-stress amplifier and the open-market Almgren-Chriss fire-sale are deferred
hypotheses and stay OFF). The clearing layer is the D69-D71 model: transfer-at-recovery
close-out, client IM first-loss, deleverage buffer.

Arm (single-hypothesis design):
  H1   margin regime reactive / flat-4% / flat-8% / flat-12%  -> procyclicality (timing) + coverage/cost
(Dropped vs the old spec: gross-vs-net netting (single-asset, no measurable resilience effect),
tiered-vs-direct, the static no-op arm, the DF-decouple robustness arm.)

Runs N_SEEDS (env, default 40) seeds per arm over the FULL calibrated window of each regime (calm 75 /
stressed 73 sessions) so the reactive arm matches the §5.1 descriptive numbers; WINDOW_DAYS="10:30"
focuses the crash trough (faster). Writes output/thesis_final/experiments/{rows.csv, summary_*.csv}.
Pure numpy/pandas (no scipy/xgboost) — runnable standalone after the relock.

Usage:  N_SEEDS=12 python3 run_thesis_experiments.py     (~50 min full window; ~14 min with WINDOW_DAYS=10:30)
"""
import os
import itertools
import numpy as np
import pandas as pd

from model import globals as G
from model.globals import (ModelParams, CALIBRATED, day_start_steps, CCP_CASH)
from model.run_simulation import build_traders, build_clearing_tier
from model.simulation import Simulation

OUTDIR = os.environ.get("OUTDIR", "output/thesis_final/experiments")
N_SEEDS = int(os.environ.get("N_SEEDS", "40"))
SEEDS = list(range(42, 42 + N_SEEDS))
COLLECT_DETAIL = False            # main() flips True to dump per-client-default + waterfall-event CSVs
_CLIENT_ROWS, _WF_ROWS, _FREEZE_ROWS, _PORT_ROWS = [], [], [], []

_FV = {r: pd.read_csv(f"data/fv_{r}.csv") for r in ("calm", "stressed")}
_FLAT = {"reactive": ("reactive", None), "static": ("static", None),
         "flat04": ("flat", 0.04), "flat08": ("flat", 0.08), "flat12": ("flat", 0.12)}


def _window(regime):
    """(V_smooth, sigma_t, start, n_steps). FULL calibrated window by default (matches the descriptive
    run, so the reactive arm equals the §5.1 numbers). WINDOW_DAYS="10:30" focuses the crash trough."""
    V = _FV[regime]["V_smooth"].to_numpy(float)
    SIG = _FV[regime]["sigma_t"].to_numpy(float)
    wd = os.environ.get("WINDOW_DAYS", "").strip()
    if wd:
        starts = list(day_start_steps(regime)); a, b = (int(x) for x in wd.split(":"))
        s = starts[a] if a < len(starts) else 0
        e = starts[b] if b < len(starts) else len(V)
    else:
        s, e = 0, len(V)
    return V, SIG, s, e - s


def run_scenario(regime, seed, direct=False, margin="reactive", decouple=False):
    """One cleared-population run; returns the clearing observables. Flips the relevant
    globals (IM_MODE / IM_FLAT_FRAC / DF_DECOUPLE_IM) for the arm. Client margining is always
    gross (the net arm was dropped)."""
    im_mode, flat = _FLAT[margin]
    G.IM_MODE = im_mode
    if flat is not None:
        G.IM_FLAT_FRAC = flat
    G.CLIENT_MARGIN_NETTING = "gross"
    G.DF_DECOUPLE_IM = decouple
    G.DF_ODD_FIXED = True   # H1 margin experiment: IM-independent ODD fund so flat/reactive arms share
                            # one consistent DF (SLOIM zeroes the fund at high flat IM). Baseline stays SLOIM.
    V, SIG, s, n = _window(regime)
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                    n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                    **CALIBRATED[regime], stressed=(regime == "stressed"))
    tr = build_traders(p, seed=seed)
    ccp = build_clearing_tier(tr, p, seed=seed, direct=direct)
    sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=s)
    sim.v_array = V[s:s + n]
    sim.sigma_t_array = SIG[s:s + n]
    sim.run(n)
    ch = pd.DataFrame(sim.clearing_history)
    cl = pd.DataFrame(sim.client_history)
    if COLLECT_DETAIL:
        if len(cl):
            _CLIENT_ROWS.append(cl.assign(regime=regime, seed=seed, margin=margin))
        for w in getattr(ccp, "_waterfall_log", []) or []:
            _WF_ROWS.append({**w, "regime": regime, "seed": seed, "margin": margin})
        for f in getattr(sim, "freeze_log", []) or []:
            _FREEZE_ROWS.append({**f, "regime": regime, "seed": seed, "margin": margin})
        for pp in getattr(sim, "porting_log", []) or []:
            _PORT_ROWS.append({**pp, "regime": regime, "seed": seed, "margin": margin})
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    if len(ch):
        im_by_t = ch.groupby("t")["initial_margin"].sum()
        im_mean, im_peak = float(im_by_t.mean()), float(im_by_t.max())
        deepest = int(ch["waterfall_level"].max())
        cm_def = int(ch[ch["has_defaulted"]]["agent_id"].nunique())
        nbcm_def = int(ch[ch["has_defaulted"] & (ch["kind"] == "NBCM")]["agent_id"].nunique())
    else:
        im_mean = im_peak = 0.0
        deepest = cm_def = nbcm_def = 0
    return dict(
        exp="", regime=regime, seed=seed, direct=direct, margin=margin,
        decouple=decouple,
        drawdown=100.0 * (mid.min() / mid[0] - 1.0) if len(mid) else float("nan"),
        client_defaults=len(cl), cm_defaults=cm_def, nbcm_defaults=nbcm_def,
        deepest_waterfall=deepest,
        mutualised_l3=bool(deepest >= 3), survivor_l4=bool(deepest >= 4),
        total_df=float(ccp.total_df), ccp_cash_used=float(CCP_CASH - ccp.cash),
        client_loss_absorbed=float(cl["closeout_loss"].sum() + cl["shortfall"].sum()) if len(cl) else 0.0,
        im_mean=im_mean, im_peak=im_peak)


def main():
    global COLLECT_DETAIL
    COLLECT_DETAIL = True                 # also dump per-client-default + waterfall-event detail
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []

    # H1 — margin procyclicality (tiered, gross): reactive vs flat-4 / flat-8 / flat-12
    for regime, margin, seed in itertools.product(
            ("calm", "stressed"), ("reactive", "flat04", "flat08", "flat12"), SEEDS):
        r = run_scenario(regime, seed, margin=margin); r["exp"] = "H1"; rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/rows.csv", index=False)

    obs = ["drawdown", "client_defaults", "cm_defaults", "nbcm_defaults", "deepest_waterfall",
           "mutualised_l3", "total_df", "ccp_cash_used", "client_loss_absorbed", "im_mean", "im_peak"]
    margin = df[df.exp == "H1"].groupby(["regime", "margin"])[obs].mean().reset_index()
    margin.to_csv(f"{OUTDIR}/summary_h1.csv", index=False)

    # per-client-default detail (type/CM/position/loss) + per-default waterfall L1-L5
    (pd.concat(_CLIENT_ROWS, ignore_index=True) if _CLIENT_ROWS else pd.DataFrame()
     ).to_csv(f"{OUTDIR}/client_defaults.csv", index=False)
    pd.DataFrame(_WF_ROWS).to_csv(f"{OUTDIR}/waterfall_events.csv", index=False)
    pd.DataFrame(_FREEZE_ROWS).to_csv(f"{OUTDIR}/client_freezes.csv", index=False)
    pd.DataFrame(_PORT_ROWS).to_csv(f"{OUTDIR}/porting_events.csv", index=False)

    print(f"thesis experiments: {len(df)} runs -> {OUTDIR}/rows.csv  (N_SEEDS={N_SEEDS})")
    print(f"  detail: {len(_CLIENT_ROWS)} client-default frames, {len(_WF_ROWS)} waterfall events")
    print("\n[H1 margin procyclicality]"); print(margin.to_string(index=False))


if __name__ == "__main__":
    main()
