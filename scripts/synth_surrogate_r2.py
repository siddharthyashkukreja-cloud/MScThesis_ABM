#!/usr/bin/env python3
"""Held-out R^2 of the joint synthetic-calibration surrogate (theta -> D).

The joint run (output/overnight_joint_logvol/calibration.json) logged its 1698 true
(theta, D) evaluations but never logged a surrogate R^2. This recomputes it with the
EXACT recipe used for the empirical surrogate -- it imports _xgb / TEST_FRAC /
LOSS_CLIP_PCTL straight from scripts.calibrate, so it stays in sync with
train_surrogate (90th-pctl label clip, 25% hold-out, seed-0 permutation, XGBoost
300 trees / depth 4 / lr 0.05 / subsample 0.8 / colsample 0.9).

Run in an env where xgboost is installed:
    python scripts/synth_surrogate_r2.py
Then drop the printed R^2 into tab:cal-synth (replaces the 0.94 placeholder).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.calibrate import _xgb, TEST_FRAC, LOSS_CLIP_PCTL  # exact surrogate + constants

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(REPO, "output", "overnight_joint_logvol", "calibration.json")
# the 9 joint free parameters, in the order stored in all_evals
PK = ["mt_gamma", "ft_sigma_c", "f", "w", "ratio", "sigma_d", "zi_alpha", "zi_mu", "zi_delta"]


def main():
    d = json.load(open(CALIB))
    ev = d["all_evals"]
    X = np.array([[e[k] for k in PK] for e in ev], dtype=float)
    y = np.array([e["D"] for e in ev], dtype=float)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]

    # --- identical to scripts.calibrate.train_surrogate ---
    cap = float(np.percentile(y, LOSS_CLIP_PCTL))
    yc = np.minimum(y, cap)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(yc))
    n_test = max(3, int(TEST_FRAC * len(yc)))
    te, tr = perm[:n_test], perm[n_test:]

    model = _xgb()
    model.fit(X[tr], yc[tr])
    pred = model.predict(X[te])
    ss_res = float(((yc[te] - pred) ** 2).sum())
    ss_tot = float(((yc[te] - yc[te].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    corr = float(np.corrcoef(yc[te], pred)[0, 1])

    print(f"n={len(yc)}  clip(p{LOSS_CLIP_PCTL})={cap:.2f}  n_test={n_test}")
    print(f"held-out surrogate R^2 = {r2:.4f}   corr = {corr:.4f}")
    print(f"-> put {r2:.2f} in tab:cal-synth (surrogate R^2 row)")


if __name__ == "__main__":
    main()
