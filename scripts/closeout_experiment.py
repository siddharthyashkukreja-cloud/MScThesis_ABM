#!/usr/bin/env python3
"""Counterfactual: does the member-default tail survive an INSTANT auction-transfer
close-out of defaulted CLIENT books, instead of the member assuming the position and
Almgren-Chriss liquidating it over AC_HORIZON?

  baseline   CLIENT_CLOSEOUT="firesale"  -> member assumes the defaulted client position and
                                            self-liquidates via AC over AC_HORIZON=30min;
                                            interim VM marks against the member (the NBCM-warehousing
                                            channel the question is about).
  cf         CLIENT_CLOSEOUT="transfer"  -> member takes a fixed (1-CLOSEOUT_RECOVERY)=20% haircut
                                            up front (0.8 recovery), no open-market AC spiral.

Runs the locked-optimum clearing model on chosen synthetic stress paths under both modes and
records member/client defaults + mutualisation. RESUMABLE + time-bounded (fits the shell limit):
appends one row per (path, mode) to output/closeout_exp.csv; rerun until all rows present.
  Env: CE_PATHS="38,36,71,51,60,23"  CE_MODES="firesale,transfer"  CE_DAYS=42
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.overnight_joint as oj          # noqa: E402
import model.globals as G                      # noqa: E402

REGIME = "stressed"
PATHS = [int(x) for x in os.environ.get("CE_PATHS", "38,36,71,51,60,23").split(",")]
MODES = os.environ.get("CE_MODES", "firesale,transfer").split(",")
DAYS  = int(os.environ.get("CE_DAYS", 42))
OUT   = oj.REPO / "output" / "closeout_exp.csv"
BUDGET = 36   # seconds per call

def run_one(path_i, mode, base, agent):
    # clearing config identical to synth_results.py (STAGE C), then flip the client close-out
    G.IM_DAILY = False; G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"; G.DF_DECOUPLE_IM = False
    G.CLIENT_CLOSEOUT = mode          # "firesale" (assume+AC) vs "transfer" (instant 0.8-recovery)
    csv = f"output/overnight_joint/paths/fv_joint_s{path_i}.csv"
    nsteps = len(pd.read_csv(oj.REPO / csv))
    _, sim, ccp = oj.run_market(REGIME, base, csv, nsteps, 1, 0, agent, clearing=True)
    ch = pd.DataFrame(sim.clearing_history)
    mem = ch[ch["kind"].isin(["BCM", "NBCM"])] if len(ch) else ch
    cm_def = int(mem[mem["has_defaulted"]]["agent_id"].nunique()) if len(mem) else 0
    deepest = int(ch["waterfall_level"].max()) if len(ch) else 0
    wlog = pd.DataFrame(getattr(ccp, "_waterfall_log", []) or [])
    mutualised = float(wlog["mutualised"].sum()) if len(wlog) else 0.0
    return dict(path=path_i, mode=mode, recovery=(G.CLOSEOUT_RECOVERY if mode == "transfer" else np.nan),
                client_def=len(sim.client_history), cm_def=cm_def, deepest_wf=deepest,
                client_loss_absorbed=round(float(ch["client_loss_absorbed"].sum()) / 1e9, 3) if "client_loss_absorbed" in ch else 0.0,
                mutualised_B=round(mutualised / 1e9, 3))

def main():
    t0 = time.time()
    base = json.loads((oj.REPO / "output" / "v_gbm_params.json").read_text())
    agent = json.load(open(oj.REPO / "output" / "overnight_joint_logvol" / "calibration.json"))["reverse_optimum"]["agent_overrides"]
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT); done = {(int(r.path), r.mode) for r in d.itertuples()}
    todo = [(p, m) for p in PATHS for m in MODES if (p, m) not in done]
    print(f"{len(done)} done; {len(todo)} to run: {todo[:6]}{'...' if len(todo) > 6 else ''}", flush=True)
    for p, m in todo:
        row = run_one(p, m, base, agent)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  path {p:>2} {m:<9} client_def={row['client_def']:>3} cm_def={row['cm_def']:>2} "
              f"deepest_wf={row['deepest_wf']} mutualised=${row['mutualised_B']}B ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > BUDGET:
            print("  (time budget hit; rerun to continue)", flush=True); break
    if OUT.exists() and not todo:
        print("\n=== closeout_exp.csv ===\n" + pd.read_csv(OUT).to_string(index=False))

if __name__ == "__main__":
    main()
