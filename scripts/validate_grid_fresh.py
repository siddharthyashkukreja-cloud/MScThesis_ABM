import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""validate_grid_fresh.py — fresh-seed re-validation of a grid optimum.

The grid argmin is selected over noisy 3-seed losses with per-node seeds, so
the reported D_grid is biased low (winner's curse) and the argmin can be the
luckiest node rather than the best one. This CLI re-evaluates the top-K grid
nodes on fresh common seed sets and reports the re-ranked losses — run it
after any grid calibration (set FV_GBM=1 to validate an SV-MJD grid; the loss
weights and targets are recomputed identically to the grid run).

Usage:  python3 validate_grid_fresh.py <regime> [top_k=5] [n_days=20] [n_runs=3]
Reads:  output/calibration_grid_{regime}.csv
Writes: output/grid_freshseed_{regime}[_gbm].json
"""
import json
import sys

import numpy as np
import pandas as pd

import calibrate as C

FRESH_SEEDS = (777, 8888)      # two fresh common seed sets, shared across nodes


def main():
    regime = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    C._activate_regime(regime)
    keys = list(C.PARAM_KEYS)
    sds = C.empirical_moment_sd(regime, n_days=n_days, n_runs=n_runs, seed=42)
    target = C.empirical_targets()[regime]
    # Prefer the current relock surface (the top-level CSV is stashed/restored by
    # run_relock.sh, so standalone it can be a stale older-param-set grid that lacks
    # zi_mu -> KeyError); fall back to the top-level for an in-pipeline call.
    _surf = C.OUT_DIR / "relock" / f"grid_surface_{regime}.csv"
    df = pd.read_csv(_surf if _surf.exists()
                     else C.OUT_DIR / f"calibration_grid_{regime}.csv")
    top = df.nsmallest(top_k, "D").reset_index(drop=True)
    print(f"[{regime}{' FV_GBM' if C.FV_GBM else ''}] fresh-seed validation of "
          f"top-{top_k} grid nodes ({n_days}d x {n_runs} runs x seeds {FRESH_SEEDS})")
    rows = []
    for i, row in top.iterrows():
        theta = np.array([row[k] for k in keys], dtype=float)
        fresh = []
        for s in FRESH_SEEDS:
            m = C.simulate_moments(theta, regime, n_days, n_runs, seed=s)
            fresh.append(float(C._true_loss(m, target, sds)))
        rows.append({**{k: float(row[k]) for k in keys},
                     "D_grid": float(row["D"]),
                     "D_fresh_mean": float(np.mean(fresh)),
                     "D_fresh": fresh})
        print(f"  node {i}: " + "  ".join(f"{k}={row[k]:g}" for k in keys)
              + f"  D_grid={row['D']:.2f}  D_fresh={np.mean(fresh):.2f}")
    rows.sort(key=lambda r: r["D_fresh_mean"])
    out = C.OUT_DIR / f"grid_freshseed_{regime}{'_gbm' if C.FV_GBM else ''}.json"
    out.write_text(json.dumps({"regime": regime, "fv_gbm": C.FV_GBM,
                               "n_days": n_days, "n_runs": n_runs,
                               "seeds": list(FRESH_SEEDS), "ranked": rows},
                              indent=2))
    b = rows[0]
    print(f"  fresh-seed best: " + "  ".join(f"{k}={b[k]:g}" for k in keys)
          + f"  (D_fresh {b['D_fresh_mean']:.2f})  -> {out}")


if __name__ == "__main__":
    main()
