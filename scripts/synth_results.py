#!/usr/bin/env python3
"""Synthetic-ensemble RESULTS at the LOCKED joint optimum — generate once, SAVE EVERYTHING as CSV.

The overnight run kept only aggregate moments + a clearing summary, so figures had to be regenerated
ad hoc. This runs the full market+clearing ABM on each synthetic stress path at the locked optimum
(output/overnight_joint_logvol/calibration.json) and persists, per path, into output/synth_results/:
    mids/mid_s{i}.csv      the simulated mid series           (-> stylised-fact figures, e.g. |r|-ACF)
    client_defaults.csv    one row per client default          (type / carrying CM / IM / loss / t)
    waterfall_events.csv   one row per member default          (level + L1..L5 + mutualised)
    client_freezes.csv     one row per freeze onset            (own-distress vs CM-contagion, kappa)
    porting_events.csv     one row per member default          (n_ported / n_unported)
    im_path.csv            total initial margin per (path, t)  (-> the procyclical IM figure)
    summary.csv            per-path scalars                    (client/member defaults, IM peak, DF, wf)
It REUSES the saved FV price paths (output/overnight_joint/paths/fv_joint_s*.csv); regenerates any
missing one from the locked optimum. RESUMABLE: skips paths already in summary.csv and appends, so it
can be run in chunks / restarted.  Env: SR_PATHS (default 80), SR_DAYS (default 42, matches validation).

  nohup env SR_PATHS=80 SR_DAYS=42 python3 scripts/synth_results.py > output/synth_results.log 2>&1 &
"""
import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import v_gbm                       # noqa: E402
import scripts.overnight_joint as oj          # noqa: E402
import model.globals as G                     # noqa: E402

REGIME = "stressed"
N      = int(os.environ.get("SR_PATHS", 80))
DAYS   = int(os.environ.get("SR_DAYS", 42))
OUT    = oj.REPO / "output" / "synth_results"
(OUT / "mids").mkdir(parents=True, exist_ok=True)

def _append(df, name):
    f = OUT / name
    df.to_csv(f, mode="a", header=not f.exists(), index=False)

def main():
    t0 = time.time()
    # clearing config identical to the validation/clearing run (STAGE C)
    G.IM_DAILY = False; G.IM_MODE = "reactive"; G.CLIENT_MARGIN_NETTING = "gross"; G.DF_DECOUPLE_IM = False
    base = json.loads((oj.REPO / "output" / "v_gbm_params.json").read_text())
    ro = json.load(open(oj.REPO / "output" / "overnight_joint_logvol" / "calibration.json"))["reverse_optimum"]
    sv, jmp, agent, mu = ro["sv_override"], ro["jump_override"], ro["agent_overrides"], ro["mu_override"]

    done = set(pd.read_csv(OUT / "summary.csv")["path"]) if (OUT / "summary.csv").exists() else set()
    print(f"synth_results: {N} paths x {DAYS}d at locked optimum; {len(done)} already done", flush=True)

    for i in range(1, N + 1):
        if i in done:
            continue
        csv = f"output/overnight_joint/paths/fv_joint_s{i}.csv"
        if not (oj.REPO / csv).exists():                       # regenerate a missing path from the optimum
            v_gbm.generate(REGIME, seed=100 + i, n_days=DAYS, out_path=oj.REPO / csv,
                           sv_override=sv, jump_override=jmp, mu_override=mu)
        r, sim, ccp = oj.run_market(REGIME, base, csv, DAYS * oj.BARS, 1, 0, agent, clearing=True)

        pd.DataFrame({"mid": pd.Series(sim.history["mid_price"]).ffill().bfill()}).to_csv(
            OUT / "mids" / f"mid_s{i}.csv", index=False)
        pd.DataFrame({"ret": r}).to_csv(OUT / "mids" / f"ret_s{i}.csv", index=False)   # intraday log-returns (overnight excluded) -> stylised-fact figures

        ch = pd.DataFrame(sim.clearing_history)
        cl = pd.DataFrame(sim.client_history)
        if len(cl):
            cl.insert(0, "path", i); _append(cl, "client_defaults.csv")
        if len(ch):
            wf = ch[ch.get("waterfall_level", 0) > 0]
            if len(wf):
                wf = wf.copy(); wf.insert(0, "path", i); _append(wf, "waterfall_events.csv")
            imt = ch.groupby("t")["initial_margin"].sum().reset_index(name="im_total")
            imt.insert(0, "path", i); _append(imt, "im_path.csv")
        for attr, name in [("freeze_log", "client_freezes.csv"), ("porting_log", "porting_events.csv")]:
            recs = getattr(sim, attr, []) or []
            if recs:
                df = pd.DataFrame(recs); df.insert(0, "path", i); _append(df, name)

        mem = ch[ch["kind"].isin(["BCM", "NBCM"])] if len(ch) else ch
        imp = ch.groupby("t")["initial_margin"].sum().max() if len(ch) else 0.0
        _append(pd.DataFrame([dict(path=i, client_def=len(cl),
            cm_def=int(mem[mem["has_defaulted"]]["agent_id"].nunique()) if len(mem) else 0,
            IM_peak_B=round(float(imp) / 1e9, 2), DF_B=round(float(ccp.total_df) / 1e9, 2),
            deepest_wf=int(ch["waterfall_level"].max()) if len(ch) else 0)]), "summary.csv")
        print(f"  path {i}/{N}  client_def={len(cl)}  cm_def={int(mem[mem['has_defaulted']]['agent_id'].nunique()) if len(mem) else 0}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE ({time.time()-t0:.0f}s) -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
