"""
Long-horizon simulator run + multi-frequency moment comparison. Uses the
calibrated θ for the chosen regime (read from
`output/calibrated_params.json` if present, else from
`globals.CALIBRATED`). Reports return moments and ACFs at 1-min and daily
aggregation, alongside the empirical ES targets.

Daily mid is the LAST 1-min mid of each simulated trading day (end-of-RTH
close). Empirical daily close is taken the same way from the rolled
front-month ES tape (`data/processed/ES_front_{regime}_1m.csv`).

CLI:
    python analysis_long_run.py [regime] [n_days]
        regime  ∈ {calm, stressed}   default: calm
        n_days                        default: 100  (V_t paths cover 120)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.globals import ModelParams, V0, CALIBRATED
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier, BARS_PER_DAY

ROOT = Path(__file__).parent
EMP_PATH = {
    "calm":     ROOT / "data" / "processed" / "ES_front_calm_1m.csv",
    "stressed": ROOT / "data" / "processed" / "ES_front_stressed_1m.csv",
}


def _acf(x: np.ndarray, k: int) -> float:
    if len(x) <= k:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).sum())
    return 0.0 if var == 0.0 else float((x[:-k] * x[k:]).sum() / var)


def _acf_smooth(x: np.ndarray, c: int) -> float:
    """Centred 3-lag avg (Franke-Westerhoff smoothing; same as
    `calibrate._acf_smoothed`)."""
    lags = (1, 2) if c <= 1 else (c - 1, c, c + 1)
    vals = [v for v in (_acf(x, l) for l in lags) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _hill(r: np.ndarray, frac: float = 0.05) -> float:
    a = np.abs(np.asarray(r, float))
    a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 100:
        return float("nan")
    k = max(int(frac * n), 20)
    if k >= n:
        return float("nan")
    a_sorted = np.sort(a)[::-1]
    thr = a_sorted[k]
    if thr <= 0:
        return float("nan")
    xi = float(np.mean(np.log(a_sorted[:k]) - np.log(thr)))
    return 1.0 / xi if xi > 0 else float("nan")


def _moments(r: np.ndarray, lags_r=(1, 5, 10), lags_a=(1, 5, 10, 20, 30, 60)) -> dict:
    r = r[np.isfinite(r)]
    sd = float(r.std()) if len(r) else float("nan")
    kurt = (float((((r - r.mean()) / sd) ** 4).mean() - 3.0)
            if sd > 0 else float("nan"))
    a = np.abs(r)
    out = {"n": len(r), "ret_std": sd, "kurt": kurt, "hill": _hill(r)}
    for L in lags_r:
        out[f"acf_r_{L}"] = _acf_smooth(r, L)
    for L in lags_a:
        out[f"acf_a_{L}"] = _acf_smooth(a, L)
    return out


def _print_row(label: str, m: dict, lags_r=(1, 5, 10), lags_a=(1, 5, 10, 20, 30, 60)):
    print(f"\n  {label}  (n={m['n']:,d}):")
    print(f"    ret_std = {m['ret_std']:.3e}    kurt = {m['kurt']:>7.1f}    "
          f"Hill = {m['hill']:.3f}")
    print(f"    acf(r):    " + "  ".join(
        f"lag{L}={m[f'acf_r_{L}']:+.4f}" for L in lags_r))
    print(f"    acf(|r|):  " + "  ".join(
        f"lag{L}={m[f'acf_a_{L}']:+.4f}" for L in lags_a))


def _theta(regime: str) -> dict:
    """Calibrated θ from `output/calibrated_params.json` if present
    (preferred — most recent calibration), else `globals.CALIBRATED`."""
    cfg = ROOT / "output" / "calibrated_params.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        if regime in d.get("results", {}):
            theta = d["results"][regime].get("theta_stage2")
            if theta:
                print(f"  using theta_stage2 from {cfg.name}")
                return theta
    print(f"  using globals.CALIBRATED[{regime}] (calibrated_params.json missing)")
    return dict(CALIBRATED[regime])


def empirical_returns(regime: str):
    df = pd.read_csv(EMP_PATH[regime], index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    log_p = np.log(mid)
    # 1-min log-returns, drop cross-day boundary
    r_1m = np.diff(log_p)
    dates = pd.DatetimeIndex(df.index).date
    keep = dates[1:] == dates[:-1]
    r_1m = r_1m[keep]
    # Daily close = last mid of each calendar day
    df_idx = pd.DatetimeIndex(df.index)
    last_per_day = df["mid"].groupby(df_idx.date).last().to_numpy()
    r_d = np.diff(np.log(last_per_day))
    return r_1m, r_d


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "calm"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    if regime not in V0:
        print(f"unknown regime '{regime}'; valid: calm, stressed")
        sys.exit(1)
    print(f"long-run simulation: regime={regime}, n_days={n_days}")
    theta = _theta(regime)

    params = ModelParams(
        n_fundamental=10, n_momentum=10, n_momentum_long=0,
        n_mm=4, n_zi=20, n_vt=0, n_ct=0,                # D48 — 4 HFABM MMs
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        stressed=(regime == "stressed"),
        **theta,
    )
    traders = build_traders(params, seed=42)
    ccp = build_clearing_tier(traders, params, seed=42)
    hist = Simulation(params, traders, seed=42, ccp=ccp).run(n_days * BARS_PER_DAY)
    mid = pd.Series(hist["mid_price"]).ffill().bfill().to_numpy()
    mid = mid[mid > 0]

    r_1m_mdl = np.diff(np.log(mid))
    # daily close = last mid of each simulated 390-bar day
    daily_mid = mid[BARS_PER_DAY - 1::BARS_PER_DAY]
    r_d_mdl = np.diff(np.log(daily_mid))

    r_1m_emp, r_d_emp = empirical_returns(regime)

    print(f"\n{'='*68}")
    print(f"  MODEL  vs  EMPIRICAL  —  {regime}")
    print(f"{'='*68}")

    LA_1m = (1, 5, 10, 20, 30, 60)
    LA_d  = (1, 2, 5, 10, 20)
    _print_row(f"MODEL 1-min",    _moments(r_1m_mdl, lags_a=LA_1m), lags_a=LA_1m)
    _print_row(f"EMP   1-min",    _moments(r_1m_emp, lags_a=LA_1m), lags_a=LA_1m)
    _print_row(f"MODEL daily",    _moments(r_d_mdl,  lags_a=LA_d),  lags_a=LA_d)
    _print_row(f"EMP   daily",    _moments(r_d_emp,  lags_a=LA_d),  lags_a=LA_d)

    out_csv = ROOT / "output" / f"long_run_{regime}.csv"
    pd.DataFrame({"mid_1m": mid}).to_csv(out_csv, index=False)
    print(f"\n  saved {out_csv}")


if __name__ == "__main__":
    main()
