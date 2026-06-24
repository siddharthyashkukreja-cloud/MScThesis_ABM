#!/usr/bin/env python3
"""No-clearing intraday returns on the saved FV paths, for the STYLISED-FACTS figure.

The calibration/validation is on the bare market layer (no clearing), and the moment table
(tab:cal-synth) reports those bare-market moments; but synth_results.py saved WITH-clearing returns
(correct for the clearing figures, where margin/liquidation dynamics damp the clustering). This
re-runs the market WITHOUT clearing on the same saved FV paths and saves the returns for the facts
figure (CCDF + |r|-ACF). RESUMABLE + time-bounded so it fits the shell limit; run a few times.
    -> output/synth_results/mids_noclr/ret_s{i}.csv     Env: NOCLR_N (default 16)
"""
import os, sys, json, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.overnight_joint as oj   # noqa: E402

N  = int(os.environ.get("NOCLR_N", 16))
LO = int(os.environ.get("NOCLR_LO", 1))            # path range [LO, HI] — for 4-way parallel workers
HI = int(os.environ.get("NOCLR_HI", N))
OUTD = oj.REPO / "output" / "synth_results" / "mids_noclr"
OUTD.mkdir(parents=True, exist_ok=True)

def main():
    t0 = time.time()
    base = json.loads((oj.REPO / "output" / "v_gbm_params.json").read_text())
    agent = json.load(open(oj.REPO / "output" / "overnight_joint_logvol" / "calibration.json"))["reverse_optimum"]["agent_overrides"]
    done = 0
    for i in range(LO, HI + 1):
        f = OUTD / f"ret_s{i}.csv"
        if f.exists():
            done += 1; continue
        rel = f"output/overnight_joint/paths/fv_joint_s{i}.csv"
        if not (oj.REPO / rel).exists():
            continue
        nsteps = len(pd.read_csv(oj.REPO / rel))
        r, _, _ = oj.run_market("stressed", base, rel, nsteps, 1, 0, agent, clearing=False)
        pd.DataFrame({"ret": r}).to_csv(f, index=False)
        done += 1
        print(f"  noclr path {i}  ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > 37:          # stay within one shell call; resume next call
            print(f"  (time budget hit; {done}/{N} done, resume to continue)", flush=True)
            break
    print(f"have {done}/{N} no-clearing return paths -> {OUTD}", flush=True)

if __name__ == "__main__":
    main()
