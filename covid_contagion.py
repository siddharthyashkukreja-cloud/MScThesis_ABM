"""
Client-clearing contagion experiment on the real COVID crash window (D55).

Points the simulation at the ES 2020-03 rows of the stressed series (data/fv_stressed.csv)
via Simulation(v_start=...) and runs the standard cleared population — no engineered
fragile client; clients are the normal, uniform-house-margin end-users. Two outputs:

  trace    one cascade run at actual COVID severity (client/CM defaults, deepest waterfall
           level, default-fund and CCP-cash trajectory);
  reverse  the Euronext-A9 §5 reverse-stress test — amplify the real path by a multiplier c
           and report the smallest c that breaches cover-2 (mutualisation onset).

Usage:  python3 covid_contagion.py [trace|reverse]
Behavioural theta is the stressed-regime CALIBRATED optimum; margins/DF/waterfall are the
D55 methodology (procyclical VaR IM, cash/IM capital ratio, cover-2 SLOIM, deficit
waterfall, BCM VaR house limit). Real data only — c scales the observed returns, no
synthetic path.
"""
import sys
import numpy as np
import pandas as pd

from model.globals import ModelParams, CALIBRATED, day_start_steps
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier

_V = pd.read_csv("data/fv_stressed.csv")["V_smooth"].to_numpy()
_SIG = pd.read_csv("data/fv_stressed.csv")["sigma_t"].to_numpy()

# COVID window: ~10 RTH sessions from Mar 9 2020 (the crash onset) of
# data/fv_stressed.csv (which starts 2020-02-24, so Mar 9 ≈ session index 10).
# Anchored to REAL session opens (D57) so the window begins on a session boundary
# and spans whole sessions — real sessions are ~405 bars and vary, not a fixed 390.
_STARTS = day_start_steps("stressed")
_W0, _WN = 10, 10
WINDOW_START = _STARTS[_W0]
WINDOW_STEPS = (_STARTS[_W0 + _WN] if _W0 + _WN < len(_STARTS)
                else len(_V)) - WINDOW_START


def run(c: float, seed: int = 42):
    """Run the COVID window with the observed path amplified by multiplier c (c=1 is
    actual COVID). Returns the cascade summary."""
    p = ModelParams(
        n_fundamental=30, n_momentum=20, n_mm=0, n_zi=40, n_vt=0, n_ct=0,
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5, v0=float(_V[WINDOW_START]),
        tick_size=0.25, dt_minutes=1.0, **CALIBRATED["stressed"], stressed=True)
    traders = build_traders(p, seed=seed)
    ccp = build_clearing_tier(traders, p, seed=seed)
    sim = Simulation(p, traders, seed=seed, ccp=ccp, v_start=WINDOW_START)
    w = _V[WINDOW_START:WINDOW_START + WINDOW_STEPS]
    sim.v_array = w[0] * np.exp(c * (np.log(w) - np.log(w[0])))
    sim.sigma_t_array = _SIG[WINDOW_START:WINDOW_START + WINDOW_STEPS]
    sim.run(WINDOW_STEPS)
    ch = pd.DataFrame(sim.clearing_history)
    cl = pd.DataFrame(sim.client_history)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    return dict(
        c=c, drawdown=100.0 * (mid.min() / mid[0] - 1.0),
        client_defaults=len(cl),
        cm_defaults=int(ch[ch["has_defaulted"]]["agent_id"].nunique()) if len(ch) else 0,
        nbcm_defaults=int(ch[ch["has_defaulted"] & (ch["kind"] == "NBCM")]["agent_id"].nunique()) if len(ch) else 0,
        deepest_waterfall=int(ch["waterfall_level"].max()) if len(ch) else 0,
        total_df=ccp.total_df, ccp_cash=ccp.cash, cl=cl)


from model.globals import CCP_CASH


def _gapped_v(c: float) -> np.ndarray:
    """Scenario A — the real gapped COVID window, amplified by c (overnight limit-down
    gaps RETAINED, D56). Same construction as run()."""
    w = _V[WINDOW_START:WINDOW_START + WINDOW_STEPS]
    return w[0] * np.exp(c * (np.log(w) - np.log(w[0])))


def _shock_v(c: float) -> np.ndarray:
    """Scenario B — flat at V0, ONE large intraday jump to the same trough level as the
    gapped path (matched total drawdown), then flat. No overnight gaps. Isolates the
    price-delivery STRUCTURE (many discrete gaps vs one shock) at matched total move."""
    g = _gapped_v(c)
    out = np.full(len(g), g[0], dtype=float)
    out[len(g) // 2:] = g.min()             # single −X% jump at the window midpoint
    return out


def scenario_run(kind: str, c: float, seed: int = 42) -> dict:
    """Run the cleared population + CCP tier through one fundamental path and return the
    cascade summary. kind='gapped' (Scenario A — real overnight gaps) or 'shock'
    (Scenario B — one equal-magnitude jump). sigma_t (the procyclical-IM driver) is held
    at the real window's local vol for BOTH so only the price-delivery structure differs.
    Euronext A9 §5 reverse-stress / scenario design."""
    p = ModelParams(
        n_fundamental=30, n_momentum=20, n_mm=0, n_zi=40, n_vt=0, n_ct=0,
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5, v0=float(_V[WINDOW_START]),
        tick_size=0.25, dt_minutes=1.0, **CALIBRATED["stressed"], stressed=True)
    traders = build_traders(p, seed=seed)
    ccp = build_clearing_tier(traders, p, seed=seed)
    sim = Simulation(p, traders, seed=seed, ccp=ccp, v_start=WINDOW_START)
    sim.v_array = _gapped_v(c) if kind == "gapped" else _shock_v(c)
    sim.sigma_t_array = _SIG[WINDOW_START:WINDOW_START + WINDOW_STEPS]
    sim.run(WINDOW_STEPS)
    ch = pd.DataFrame(sim.clearing_history)
    cl = pd.DataFrame(sim.client_history)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    deepest = int(ch["waterfall_level"].max()) if len(ch) else 0
    return dict(
        kind=kind, c=float(c), seed=int(seed),
        drawdown=100.0 * (mid.min() / mid[0] - 1.0) if len(mid) else float("nan"),
        client_defaults=len(cl),
        cm_defaults=int(ch[ch["has_defaulted"]]["agent_id"].nunique()) if len(ch) else 0,
        nbcm_defaults=int(ch[ch["has_defaulted"] & (ch["kind"] == "NBCM")]["agent_id"].nunique()) if len(ch) else 0,
        waterfall_level_reached=deepest,
        mutualised=bool(deepest >= 4),                   # L4 = surviving-member cash pro-rata
        total_df=float(ccp.total_df),                    # sized cover-2 DF
        ccp_cash_used=float(CCP_CASH - ccp.cash),        # SITG (L2) + exchange (L5) drawn
    )


def scenario_sweep(out_csv: str = "output/campaign/E6_gaps_vs_shock.csv",
                   cs=(1.0, 1.5, 2.0, 2.5, 3.0), seeds=(42, 43)):
    """E6 — gaps vs single shock. Sweep the amplifier c for both scenarios over a few
    seeds; write one row per (kind, c, seed) and print a per-c mean comparison."""
    import os as _os
    rows = [scenario_run(kind, c, s)
            for kind in ("gapped", "shock") for c in cs for s in seeds]
    df = pd.DataFrame(rows)
    _os.makedirs(_os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"E6 gaps-vs-shock — {len(df)} runs -> {out_csv}")
    agg = (df.groupby(["kind", "c"])[["drawdown", "client_defaults", "cm_defaults",
                                      "nbcm_defaults", "waterfall_level_reached",
                                      "ccp_cash_used"]].mean().reset_index())
    print(agg.to_string(index=False))
    return df


def trace(seed: int = 42):
    r = run(1.0, seed)
    print(f"COVID window cascade trace (seed {seed}): drawdown {r['drawdown']:+.1f}%")
    print(f"  client defaults={r['client_defaults']}  CM defaults={r['cm_defaults']} "
          f"(NBCM {r['nbcm_defaults']})  deepest waterfall=L{r['deepest_waterfall']}")
    print(f"  default fund=${r['total_df']/1e9:.2f}B  CCP cash=${r['ccp_cash']/1e9:.2f}B")
    if len(r["cl"]):
        print(r["cl"][["t", "kind", "cm_kind", "shortfall", "assumed_pos"]].head(8).to_string(index=False))


def reverse(seed: int = 42):
    print(f"Reverse stress (Euronext A9 §5) — amplify COVID path by c, seed {seed}:")
    print(f"{'c':>5}{'drawdown%':>11}{'client_def':>12}{'CM_def':>8}{'NBCM_def':>10}{'waterfall':>11}{'DF $B':>8}")
    breach = None
    for c in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        r = run(c, seed)
        print(f"{c:>5.1f}{r['drawdown']:>11.1f}{r['client_defaults']:>12}{r['cm_defaults']:>8}"
              f"{r['nbcm_defaults']:>10}{('L'+str(r['deepest_waterfall'])):>11}{r['total_df']/1e9:>8.2f}")
        if breach is None and r["deepest_waterfall"] >= 3:
            breach = c
    if breach:
        print(f"  cover-2 mutualisation (>= L3 pooled DF) onsets at c >= {breach:g} (~{breach:g}x COVID)")
    else:
        print("  cover-2 not breached to L3 over the sweep")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "trace"
    if mode == "scenario":
        scenario_sweep()
    elif mode == "reverse":
        reverse()
    else:
        trace()
